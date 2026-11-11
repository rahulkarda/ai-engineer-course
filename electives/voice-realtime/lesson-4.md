# Elective: Voice & Realtime — Lesson 4
# LLM Streaming + TTS Pipeline

> Objective: Stream LLM tokens directly into a TTS queue, implement sentence-boundary buffering.  
> Time: ~60 min  
> Prereqs: [[lesson-3.md]]

---

## Concept

The **LLM→TTS handoff** is where streaming matters most. If you wait for the full
LLM response before synthesizing speech, you add the entire LLM generation time
to TTFA. With streaming, TTS can start on the first complete sentence while the
LLM is still generating the rest of the response.

### The sentence-boundary trick

TTS models produce much more natural output on complete sentences. Streaming
individual tokens to TTS creates choppy, artifact-ridden audio. The optimal
strategy:

1. LLM streams tokens.
2. A **sentence buffer** accumulates tokens and detects sentence boundaries
   (`.`, `!`, `?`, `\n\n`).
3. Each complete sentence is dispatched to TTS immediately.
4. TTS synthesizes and enqueues audio chunks in parallel with LLM continuing.

This reduces TTFA to approximately:
```
TTFA ≈ STT_latency + LLM_first_sentence_latency + TTS_first_chunk_latency
```

### Anthropic streaming API

The Anthropic SDK provides a streaming context manager:

```python
async with client.messages.stream(
    model="claude-haiku-5-5",
    max_tokens=300,
    messages=[{"role": "user", "content": transcript}],
) as stream:
    async for text_chunk in stream.text_stream:
        # text_chunk is a string delta (1–10 tokens typically)
        ...
```

`stream.text_stream` is an async iterator of string deltas. Each delta may
contain 0–N characters; do not assume a delta equals a word or sentence.

### TTS options

| Provider | Mode | Quality | Notes |
|----------|------|---------|-------|
| ElevenLabs streaming | WebSocket / HTTP chunked | excellent | best voice quality |
| OpenAI TTS | HTTP batch | very good | no streaming in basic API |
| Azure Neural TTS | WebSocket SSML | excellent | enterprise |
| Coqui TTS (local) | batch | good | open source, offline |
| Mock TTS | in-process | deterministic | use for all tests |

---

## How it works internally

### Token streaming under the hood

Anthropic's streaming protocol uses Server-Sent Events (SSE). The SDK wraps
these into `content_block_delta` events with `delta.text` field. The `text_stream`
async iterator extracts these deltas and yields them one by one.

### Double-buffered audio playback

To eliminate gaps between TTS chunks, use a double buffer:

```
TTS queue:  [chunk_1][chunk_2][chunk_3] → playing chunk_1, synthesizing chunk_3
```

The audio player always has one chunk ready while synthesizing the next.
In Python: two `asyncio.Queue` entries are pre-fetched before the player starts.

### Sentence detection

A robust sentence splitter for TTS contexts:

- Split on `. `, `! `, `? `, `.\n`, `!\n`, `?\n`.
- Do NOT split on abbreviations (Mr., Dr., vs.).
- Do NOT split mid-number (3.14).
- Minimum sentence length guard (≥ 10 chars) to avoid dispatching fragments.

---

## Worked example

```python
"""
voice_pipeline/llm_tts.py
LLM streaming → sentence buffer → TTS pipeline stage.
Uses Anthropic claude-haiku-5-5 for reasoning; mock TTS for synthesis.
"""
import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

import anthropic

log = logging.getLogger(__name__)

# Sentence boundary pattern — splits on terminal punctuation followed by space or newline
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"])|(?<=[.!?])\n')
# Minimum chars before we consider a sentence "complete enough" for TTS
_MIN_SENTENCE_CHARS = 10


@dataclass
class AudioChunk:
    data: bytes
    text_source: str
    timestamp: float = field(default_factory=time.monotonic)


@runtime_checkable
class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> AudioChunk: ...


class MockTTSProvider:
    """
    Mock TTS: encodes text as UTF-8 bytes, simulates synthesis latency.
    Never calls any external API.
    """
    def __init__(self, latency_ms: float = 50.0) -> None:
        self._latency_ms = latency_ms

    async def synthesize(self, text: str) -> AudioChunk:
        await asyncio.sleep(self._latency_ms / 1000)
        return AudioChunk(data=text.encode("utf-8"), text_source=text)


class ElevenLabsTTSProvider:
    """
    ElevenLabs streaming TTS.
    Requires: uv add elevenlabs  and ELEVENLABS_API_KEY env var.
    Documents the pattern; use MockTTSProvider in tests.
    """
    def __init__(self, voice_id: str = "Rachel") -> None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        try:
            from elevenlabs.client import ElevenLabs  # noqa: PLC0415
            self._client = ElevenLabs(api_key=api_key)
            self._voice_id = voice_id
        except ImportError as e:
            raise RuntimeError(
                "elevenlabs not installed. Run: uv add elevenlabs"
            ) from e

    async def synthesize(self, text: str) -> AudioChunk:
        loop = asyncio.get_event_loop()
        # ElevenLabs SDK is synchronous; run in executor
        result = await loop.run_in_executor(
            None,
            lambda: b"".join(
                self._client.generate(text=text, voice=self._voice_id, stream=False)
            ),
        )
        return AudioChunk(data=result, text_source=text)


class SentenceBuffer:
    """
    Accumulates LLM token deltas; yields complete sentences for TTS dispatch.
    """
    def __init__(self, min_chars: int = _MIN_SENTENCE_CHARS) -> None:
        self._buf = ""
        self._min_chars = min_chars

    def feed(self, delta: str) -> list[str]:
        """Return list of complete sentences ready for TTS (may be empty)."""
        self._buf += delta
        sentences: list[str] = []
        while True:
            match = _SENTENCE_END.search(self._buf)
            if not match:
                break
            candidate = self._buf[: match.start() + 1].strip()
            if len(candidate) >= self._min_chars:
                sentences.append(candidate)
                self._buf = self._buf[match.end():]
            else:
                break
        return sentences

    def flush(self) -> str | None:
        """Return any remaining buffered text (final token in response)."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining if remaining else None


async def llm_stage(
    transcript_in: asyncio.Queue,
    sentence_out: asyncio.Queue[str],
    *,
    model: str = "claude-haiku-5-5",
    system_prompt: str = "You are a helpful voice assistant. Keep answers concise.",
    max_tokens: int = 300,
) -> None:
    """
    Reads Transcript objects, streams LLM response, emits sentences.

    Uses Anthropic streaming API. Requires ANTHROPIC_API_KEY.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    while True:
        transcript = await transcript_in.get()
        log.info("LLM: processing '%s'", transcript.text)
        buf = SentenceBuffer()
        start = time.monotonic()
        first_token = True

        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": transcript.text}],
        ) as stream:
            async for delta in stream.text_stream:
                if first_token:
                    log.info(
                        "LLM: first token in %.0f ms",
                        (time.monotonic() - start) * 1000,
                    )
                    first_token = False
                for sentence in buf.feed(delta):
                    log.debug("LLM→TTS sentence: '%s'", sentence[:40])
                    await sentence_out.put(sentence)

        # Flush any trailing text
        remainder = buf.flush()
        if remainder:
            await sentence_out.put(remainder)

        log.info("LLM: response complete in %.0f ms", (time.monotonic() - start) * 1000)
        transcript_in.task_done()


async def tts_stage(
    sentence_in: asyncio.Queue[str],
    audio_out: asyncio.Queue[AudioChunk],
    provider: TTSProvider | None = None,
) -> None:
    """
    Reads sentences, synthesizes audio, enqueues AudioChunks.

    Args:
        sentence_in: sentences from LLM stage.
        audio_out: downstream queue for AudioChunk objects.
        provider: TTSProvider; defaults to MockTTSProvider.
    """
    _provider = provider or MockTTSProvider()
    while True:
        sentence = await sentence_in.get()
        start = time.monotonic()
        chunk = await _provider.synthesize(sentence)
        latency = (time.monotonic() - start) * 1000
        log.info(
            "TTS: synthesized %d chars in %.0f ms",
            len(sentence),
            latency,
        )
        await audio_out.put(chunk)
        sentence_in.task_done()


class MockLLMProvider:
    """
    Mock LLM: streams a fixed response as individual words.
    Used when ANTHROPIC_API_KEY is not available (e.g., CI tests).
    """
    def __init__(self, response: str = "This is a mock LLM response for testing.") -> None:
        self._response = response

    async def stream_tokens(self) -> AsyncIterator[str]:
        words = self._response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.01)
            yield word + (" " if i < len(words) - 1 else "")


async def mock_llm_stage(
    transcript_in: asyncio.Queue,
    sentence_out: asyncio.Queue[str],
    provider: MockLLMProvider | None = None,
) -> None:
    """Drop-in replacement for llm_stage that never calls Anthropic."""
    _provider = provider or MockLLMProvider()
    while True:
        transcript = await transcript_in.get()
        log.info("MockLLM: processing '%s'", transcript.text)
        buf = SentenceBuffer()
        async for delta in _provider.stream_tokens():
            for sentence in buf.feed(delta):
                await sentence_out.put(sentence)
        remainder = buf.flush()
        if remainder:
            await sentence_out.put(remainder)
        transcript_in.task_done()
```

---

## Exercises

1. Write pytest tests for `SentenceBuffer`:
   - Feed tokens one character at a time for `"Hello world. How are you?"` —
     assert two sentences are emitted.
   - Verify `"3.14 is pi"` is NOT split on the decimal point.
   - Verify `flush()` returns leftover text without trailing punctuation.

2. The `llm_stage` uses `claude-haiku-5-5`. Add a `model` parameter to let
   callers pass `claude-sonnet-5-5` for higher quality turns. Add a test using
   `MockLLMProvider` that verifies the stage parameter is forwarded correctly
   (use a `MockLLMProvider` subclass that records its init args).

3. `MockTTSProvider` simulates fixed latency. Add a `jitter_ms` parameter that
   adds random noise (±jitter) to simulate real-world variance. Seed the RNG
   for deterministic tests.

4. Implement a `TTSCache` wrapper around any `TTSProvider`: cache the last N
   synthesized sentences by text hash so repeated phrases (e.g., "I'm sorry,
   could you repeat that?") are served from cache without latency.

*(Solutions: [solutions/lesson-4.md](solutions/lesson-4.md))*

---

## Self-check

- [ ] I can stream Anthropic tokens and detect sentence boundaries in the delta stream
- [ ] I understand why sentence-boundary buffering improves TTS quality
- [ ] I can implement a `TTSProvider` protocol and wire it into a pipeline stage
- [ ] I can replace the real LLM with a mock for testing
