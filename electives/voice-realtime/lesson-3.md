# Elective: Voice & Realtime — Lesson 3
# STT Options & Streaming Transcription

> Objective: Understand STT architectures, stream audio to a transcription service, handle interim/final results.  
> Time: ~60 min  
> Prereqs: [[lesson-2.md]]

---

## Concept

**Speech-to-Text (STT)** converts audio bytes into a text transcript. The choice
of STT provider affects latency, accuracy, cost, and how much streaming you get.

### The two STT paradigms

**Batch (offline) STT:**  
Send a complete audio file → receive transcript. Simple API; high latency because
you must wait for the whole utterance. Example: Whisper local model.

**Streaming STT:**  
Open a WebSocket or gRPC stream → push audio frames as they arrive → receive
*interim* (partial) and *final* transcripts in real time. Example: Deepgram Live,
Google Cloud STT streaming. Enables sub-400 ms first-word latency.

### STT provider comparison

| Provider | Mode | Latency | Quality | Cost | Notes |
|----------|------|---------|---------|------|-------|
| Whisper tiny | batch local | 200–500 ms | fair | free | no streaming |
| Whisper medium | batch local | 1–3 s | good | free | slower on CPU |
| Deepgram Nova-2 | streaming | 100–300 ms | excellent | ~$0.0043/min | best for prod |
| AssemblyAI | streaming | 200–400 ms | excellent | ~$0.005/min | |
| Google Cloud STT | streaming | 150–350 ms | excellent | ~$0.006/min | |
| Azure Speech | streaming | 150–350 ms | excellent | ~$0.005/min | |

**Recommendation:**  
- Development/testing: mock STT (this lesson).
- Low-latency production: Deepgram streaming.
- Air-gapped / offline: Whisper local.

### Interim vs. final transcripts

Streaming STT continuously emits:

- **Interim**: partial hypothesis, may change. Useful for display; risky for LLM input.
- **Final**: committed result, will not change. Use this for LLM input.

Some providers emit a `speech_final` flag (end of utterance as detected server-
side) and a separate `is_final` flag (transcript will not change). Treat `speech_final`
as the trigger to call the LLM.

---

## How it works internally

### Whisper architecture

Whisper is an encoder-decoder Transformer: a CNN-based log-mel spectrogram encoder
(analyzes frequency patterns over time) → a decoder that autoregressively generates
tokens. Because it is encoder-decoder and processes fixed-length audio windows
(30 s), it cannot natively stream — you must either:

1. Process chunks with overlapping windows (VAD-chunked Whisper).
2. Use `faster-whisper` with streaming segmentation.

### Deepgram streaming protocol

Deepgram accepts raw PCM over WebSocket:

1. `GET wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000`
   with `Authorization: Token <KEY>` header.
2. Client sends binary audio frames as WebSocket binary messages.
3. Server sends JSON `TranscriptResponse` with `.channel.alternatives[0].transcript`
   and `.is_final`, `.speech_final` booleans.
4. Client sends `{"type": "CloseStream"}` to end gracefully.

### STT abstraction

A clean architecture wraps any provider behind a common interface:

```python
class STTProvider(Protocol):
    async def transcribe_segment(self, audio: bytes) -> str: ...
    async def stream_transcribe(
        self, audio_queue: asyncio.Queue[bytes],
        transcript_queue: asyncio.Queue[Transcript],
    ) -> None: ...
```

---

## Worked example

```python
"""
voice_pipeline/stt.py
STT stage with mock and real-provider abstractions.
No real API calls in default usage.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    is_final: bool = True
    confidence: float = 1.0
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)


@runtime_checkable
class STTProvider(Protocol):
    """Minimal STT interface — batch or streaming."""
    async def transcribe(self, audio_bytes: bytes) -> Transcript: ...


class MockSTTProvider:
    """
    Deterministic mock STT.
    Returns a predictable transcript based on audio length.
    Never calls any external API.
    """
    def __init__(self, response_template: str = "mock transcript {n}") -> None:
        self._call_count = 0
        self._template = response_template

    async def transcribe(self, audio_bytes: bytes) -> Transcript:
        self._call_count += 1
        # Simulate processing delay proportional to audio length
        sim_duration_ms = len(audio_bytes) / 32  # 16kHz 16-bit → 32 bytes/ms
        await asyncio.sleep(min(sim_duration_ms / 1000 * 0.1, 0.05))  # 10% of real time
        text = self._template.format(n=self._call_count)
        return Transcript(
            text=text,
            is_final=True,
            confidence=0.95,
            duration_ms=sim_duration_ms,
        )


class WhisperSTTProvider:
    """
    Local Whisper STT (batch).
    Requires: uv add openai-whisper  (also needs ffmpeg)
    """
    def __init__(self, model_name: str = "base") -> None:
        try:
            import whisper  # noqa: PLC0415
            self._model = whisper.load_model(model_name)
        except ImportError as e:
            raise RuntimeError(
                "openai-whisper not installed. Run: uv add openai-whisper"
            ) from e

    async def transcribe(self, audio_bytes: bytes) -> Transcript:
        import tempfile  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        # Whisper expects float32 numpy array or a file path
        loop = asyncio.get_event_loop()
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        result = await loop.run_in_executor(
            None, lambda: self._model.transcribe(audio_np)
        )
        return Transcript(
            text=result["text"].strip(),
            is_final=True,
            confidence=1.0,
        )


class DeepgramStreamingSTT:
    """
    Deepgram streaming STT over WebSocket.
    Requires: uv add deepgram-sdk  and DEEPGRAM_API_KEY env var.
    This class documents the pattern; use MockSTTProvider in tests.
    """
    def __init__(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set")
        try:
            from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions  # noqa: PLC0415
            self._client = DeepgramClient(api_key)
            self._LiveTranscriptionEvents = LiveTranscriptionEvents
            self._LiveOptions = LiveOptions
        except ImportError as e:
            raise RuntimeError(
                "deepgram-sdk not installed. Run: uv add deepgram-sdk"
            ) from e

    async def stream(
        self,
        audio_queue: asyncio.Queue[bytes],
        transcript_queue: asyncio.Queue[Transcript],
    ) -> None:
        """
        Opens a Deepgram live session, feeds audio, emits Transcripts.
        Pattern only — not tested without real credentials.
        """
        options = self._LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="linear16",
            sample_rate=16_000,
            interim_results=True,
        )
        connection = self._client.listen.websocket.v("1")

        def on_transcript(_, result, **__) -> None:  # type: ignore[no-untyped-def]
            alt = result.channel.alternatives[0]
            t = Transcript(
                text=alt.transcript,
                is_final=result.is_final,
                confidence=alt.confidence,
            )
            asyncio.get_event_loop().call_soon_threadsafe(
                transcript_queue.put_nowait, t
            )

        connection.on(self._LiveTranscriptionEvents.Transcript, on_transcript)
        connection.start(options)
        try:
            while True:
                chunk = await audio_queue.get()
                if chunk == b"":  # sentinel
                    break
                connection.send(chunk)
                audio_queue.task_done()
        finally:
            connection.finish()


async def stt_stage(
    segment_in: asyncio.Queue[bytes],
    transcript_out: asyncio.Queue[Transcript],
    provider: STTProvider | None = None,
) -> None:
    """
    Async pipeline stage: reads audio segments, emits final Transcripts.

    Args:
        segment_in: complete speech segments (bytes) from VAD stage.
        transcript_out: downstream queue for Transcript objects.
        provider: STTProvider; defaults to MockSTTProvider.
    """
    _provider = provider or MockSTTProvider()
    while True:
        audio = await segment_in.get()
        start = time.monotonic()
        transcript = await _provider.transcribe(audio)
        latency = (time.monotonic() - start) * 1000
        log.info(
            "STT: '%s' (final=%s, %.0f ms)",
            transcript.text,
            transcript.is_final,
            latency,
        )
        if transcript.is_final and transcript.text.strip():
            await transcript_out.put(transcript)
        segment_in.task_done()
```

---

## Exercises

1. Write a pytest test for `MockSTTProvider.transcribe()` that:
   - Calls it 3 times with different audio lengths.
   - Asserts `is_final=True` and `text` contains the call count.
   - Asserts shorter audio produces shorter simulated delay.

2. Add a `ScriptedSTTProvider` that takes a list of strings at construction
   time and returns them in order, one per `transcribe()` call. This is useful
   for deterministic integration tests. What should it return if called more
   times than there are scripts?

3. The `stt_stage` drops empty transcripts. What happens if the STT provider
   returns whitespace-only text (e.g., `"  "`)? Fix the filter and add a test.

4. Implement a `split_on_sentence` utility that takes a streaming sequence of
   `Transcript` objects (some `is_final=False`) and yields full sentences as
   they become detectable. Use simple `.` / `?` / `!` detection.

*(Solutions: [solutions/lesson-3.md](solutions/lesson-3.md))*

---

## Self-check

- [ ] I can explain why Whisper cannot natively stream but Deepgram can
- [ ] I can implement a `STTProvider` protocol and a mock that satisfies it
- [ ] I understand the interim vs. final transcript distinction
- [ ] I can wire an `stt_stage` coroutine into an async pipeline
