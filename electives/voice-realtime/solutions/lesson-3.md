# Solutions: Lesson 3 — STT Options & Streaming Transcription

## Exercise 1 — pytest for MockSTTProvider

```python
import asyncio
import pytest
from voice_pipeline.stt import MockSTTProvider


@pytest.mark.asyncio
async def test_mock_stt_returns_incrementing_count():
    provider = MockSTTProvider(response_template="utterance {n}")
    for i in range(1, 4):
        # Different audio lengths
        audio = b"\x00" * (320 * i)
        transcript = await provider.transcribe(audio)
        assert transcript.is_final is True
        assert str(i) in transcript.text


@pytest.mark.asyncio
async def test_mock_stt_shorter_audio_faster():
    provider = MockSTTProvider()
    import time
    short_audio = b"\x00" * 320    # ~10 ms of audio
    long_audio = b"\x00" * 32000   # ~1 s of audio

    t0 = time.monotonic()
    await provider.transcribe(short_audio)
    short_time = time.monotonic() - t0

    t0 = time.monotonic()
    await provider.transcribe(long_audio)
    long_time = time.monotonic() - t0

    assert short_time < long_time
```

## Exercise 2 — ScriptedSTTProvider

```python
from voice_pipeline.stt import Transcript


class ScriptedSTTProvider:
    """
    Returns pre-scripted transcripts in order.
    Raises StopIteration (wrapped) if called more times than scripts.
    """
    def __init__(self, scripts: list[str]) -> None:
        self._scripts = iter(scripts)
        self._exhausted = False

    async def transcribe(self, audio_bytes: bytes) -> Transcript:
        if self._exhausted:
            return Transcript(text="", is_final=True, confidence=0.0)
        try:
            text = next(self._scripts)
            return Transcript(text=text, is_final=True, confidence=0.99)
        except StopIteration:
            self._exhausted = True
            return Transcript(text="", is_final=True, confidence=0.0)
```

Design note: returning an empty transcript (rather than raising) is safer for
pipeline integration — the `stt_stage` will filter it out.

## Exercise 3 — Whitespace-only filter fix

Problem: `"  ".strip()` is falsy, but `"  "` is truthy, so the original check
`if transcript.is_final and transcript.text.strip()` is actually correct. The bug
would appear if we used `if transcript.text:` without `.strip()`.

Fixed stage (explicit):

```python
async def stt_stage(segment_in, transcript_out, provider=None):
    _provider = provider or MockSTTProvider()
    while True:
        audio = await segment_in.get()
        transcript = await _provider.transcribe(audio)
        # Correct: use .strip() to catch whitespace-only text
        if transcript.is_final and transcript.text.strip():
            await transcript_out.put(transcript)
        segment_in.task_done()
```

Test:

```python
@pytest.mark.asyncio
async def test_stt_stage_drops_whitespace_transcript():
    class WhitespaceSTT:
        async def transcribe(self, audio: bytes) -> Transcript:
            return Transcript(text="   ", is_final=True)

    segment_q: asyncio.Queue[bytes] = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    await segment_q.put(b"\x00" * 640)

    task = asyncio.create_task(stt_stage(segment_q, out_q, WhitespaceSTT()))
    await asyncio.sleep(0.05)
    task.cancel()
    assert out_q.empty(), "Whitespace transcript should have been dropped"
```

## Exercise 4 — Sentence splitter for interim transcripts

```python
import re
from collections.abc import AsyncIterator

_SENT_END = re.compile(r'(?<=[.!?])\s')


async def split_on_sentence(
    source: AsyncIterator[Transcript],
) -> AsyncIterator[str]:
    """
    Buffers interim transcripts; yields full sentences when detected.
    Yields the final buffer on stream exhaustion.
    """
    buf = ""
    async for transcript in source:
        buf += transcript.text
        while True:
            m = _SENT_END.search(buf)
            if not m:
                break
            sentence = buf[: m.start() + 1].strip()
            buf = buf[m.end():]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()
```
