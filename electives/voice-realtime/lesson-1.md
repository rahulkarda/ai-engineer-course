# Elective: Voice & Realtime — Lesson 1
# Pipeline Anatomy & Latency Budgets

> Objective: Map the full voice-agent pipeline and understand where latency comes from.  
> Time: ~45 min  
> Prereqs: async Python basics, LLM API familiarity

---

## Concept

A voice agent is a **real-time perception-reasoning-action loop**:

```
Microphone ──► VAD ──► STT ──► LLM ──► TTS ──► Speaker
               │                │         │
           silence/speech    tokens    audio chunks
           detection        stream     stream
```

Each arrow is an **async stage** that produces output before the previous stage
finishes. That is the key insight: naive sequential processing feels broken;
streaming at every boundary is what creates a conversational feel.

### The five stages

| Stage | Component | Job |
|-------|-----------|-----|
| 1 | **VAD** (Voice Activity Detection) | Detect speech vs. silence; gate the pipeline |
| 2 | **STT** (Speech-to-Text) | Convert audio frames → text transcript |
| 3 | **LLM** | Reason over transcript → generate response text |
| 4 | **TTS** (Text-to-Speech) | Convert response text → audio bytes |
| 5 | **Playback** | Stream audio bytes to speaker |

### Turn-based vs. streaming

**Turn-based (simplest):**  
Record until silence → transcribe whole utterance → call LLM → synthesize whole
response → play. Easiest to implement; latency = sum of all stage times.

**Streaming:**  
Each stage starts as soon as the first chunk arrives from the previous stage.
LLM streams tokens; TTS synthesizes the first sentence before the LLM finishes.
Latency = dominated by the slowest *first-chunk* time, not total time.

---

## Latency Budgets

Conversational feel requires **Time-To-First-Audio (TTFA) ≤ 1 second** after the
user stops speaking. Here is a realistic budget breakdown:

| Stage | Turn-based | Streaming |
|-------|-----------|-----------|
| VAD end-of-utterance detection | 300–500 ms | 300–500 ms |
| STT (cloud, e.g. Deepgram) | 200–600 ms | 150–400 ms (first chunk) |
| STT (local Whisper tiny/base) | 400–800 ms | N/A (batch) |
| LLM first token (claude-haiku-5-5) | 300–600 ms | 300–600 ms |
| TTS first audio chunk | 100–400 ms | 80–300 ms |
| **Total TTFA** | **~1.3–2.1 s** | **~0.8–1.3 s** |

> Rule of thumb: **500 ms feels instant, 1 s acceptable, >2 s feels laggy.**

**Dominant bottleneck** is almost always the LLM first-token latency for short
turns. For long turns, STT can dominate.

### Optimization levers

1. **Use the smallest model that meets quality bar** — `claude-haiku-5-5` for
   short turns, escalate to `claude-sonnet-5-5` only for complex reasoning.
2. **Stream LLM tokens into TTS immediately** — sentence-boundary detection lets
   TTS start on the first complete sentence.
3. **Overlap TTS synthesis with LLM generation** — double-buffered audio queue.
4. **Local VAD** (webrtcvad) is near-zero latency vs. server VAD.
5. **Predictive prefill** — start LLM call before STT fully finishes if
   confidence is high (advanced, risky).

---

## How it works internally

### asyncio event loop model

Each pipeline stage is a coroutine connected by `asyncio.Queue` objects:

```
AudioIn  →[queue_raw]→  VAD  →[queue_speech]→  STT  →[queue_text]→
    LLM  →[queue_tokens]→  TTS  →[queue_audio]→  AudioOut
```

`asyncio.Queue` is a FIFO channel that lets producers and consumers run at
different rates. A consumer `await queue.get()` suspends until data arrives,
yielding the event loop to other coroutines.

### Back-pressure

If TTS is slower than LLM token generation, `queue_tokens` fills up.
`asyncio.Queue(maxsize=N)` applies back-pressure: the producer blocks on
`await queue.put(item)` when the queue is full, naturally regulating flow.

---

## Worked example

A minimal pipeline skeleton demonstrating the queue-based design (fully mocked —
no audio hardware or real APIs needed):

```python
"""
voice_pipeline/skeleton.py
Minimal async pipeline skeleton with mock stages.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    data: bytes
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class Transcript:
    text: str
    is_final: bool = True
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class LLMToken:
    text: str
    is_final: bool = False
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class AudioOutput:
    data: bytes
    text_source: str
    timestamp: float = field(default_factory=time.monotonic)


async def mock_vad_stage(
    raw_in: asyncio.Queue[AudioChunk],
    speech_out: asyncio.Queue[AudioChunk],
    *,
    silence_threshold_frames: int = 3,
) -> None:
    """Pass audio through; drop 'silence' frames (every 5th chunk, for demo)."""
    frame_count = 0
    while True:
        chunk = await raw_in.get()
        frame_count += 1
        if frame_count % 5 == 0:
            log.debug("VAD: silence frame dropped")
        else:
            await speech_out.put(chunk)
        raw_in.task_done()


async def mock_stt_stage(
    speech_in: asyncio.Queue[AudioChunk],
    text_out: asyncio.Queue[Transcript],
) -> None:
    """Accumulate frames; emit a transcript every 4 frames."""
    buffer: list[AudioChunk] = []
    utterance = 0
    while True:
        chunk = await speech_in.get()
        buffer.append(chunk)
        if len(buffer) >= 4:
            utterance += 1
            # Simulate STT latency
            await asyncio.sleep(0.05)
            transcript = Transcript(text=f"mock utterance {utterance}", is_final=True)
            await text_out.put(transcript)
            buffer.clear()
            log.info("STT: emitted %s", transcript.text)
        speech_in.task_done()


async def mock_llm_stage(
    text_in: asyncio.Queue[Transcript],
    token_out: asyncio.Queue[LLMToken],
) -> None:
    """Emit 5 tokens per transcript with simulated streaming delay."""
    while True:
        transcript = await text_in.get()
        log.info("LLM: processing '%s'", transcript.text)
        words = ["This", " is", " a", " mock", " response."]
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)  # simulate token latency
            is_final = i == len(words) - 1
            await token_out.put(LLMToken(text=word, is_final=is_final))
        text_in.task_done()


async def mock_tts_stage(
    token_in: asyncio.Queue[LLMToken],
    audio_out: asyncio.Queue[AudioOutput],
) -> None:
    """Buffer tokens into sentences; synthesize and emit audio."""
    buffer = ""
    while True:
        token = await token_in.get()
        buffer += token.text
        if token.is_final or "." in token.text:
            await asyncio.sleep(0.03)  # simulate TTS latency
            audio = AudioOutput(data=buffer.encode(), text_source=buffer)
            await audio_out.put(audio)
            log.info("TTS: synthesized '%s'", buffer.strip())
            buffer = ""
        token_in.task_done()


async def audio_player(audio_in: asyncio.Queue[AudioOutput]) -> None:
    while True:
        audio = await audio_in.get()
        log.info("PLAY: %s", audio.text_source.strip())
        audio_in.task_done()


async def audio_source(raw_out: asyncio.Queue[AudioChunk], *, n_chunks: int = 20) -> None:
    """Produce n_chunks of mock audio frames."""
    for i in range(n_chunks):
        await asyncio.sleep(0.01)
        await raw_out.put(AudioChunk(data=b"\x00" * 320))
    log.info("AudioSource: finished producing %d chunks", n_chunks)


async def run_pipeline(n_chunks: int = 20) -> None:
    q_raw: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)
    q_speech: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=16)
    q_text: asyncio.Queue[Transcript] = asyncio.Queue(maxsize=8)
    q_tokens: asyncio.Queue[LLMToken] = asyncio.Queue(maxsize=32)
    q_audio: asyncio.Queue[AudioOutput] = asyncio.Queue(maxsize=8)

    start = time.monotonic()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(audio_source(q_raw, n_chunks=n_chunks))
        tg.create_task(mock_vad_stage(q_raw, q_speech))
        tg.create_task(mock_stt_stage(q_speech, q_text))
        tg.create_task(mock_llm_stage(q_text, q_tokens))
        tg.create_task(mock_tts_stage(q_tokens, q_audio))
        tg.create_task(audio_player(q_audio))

    elapsed = time.monotonic() - start
    log.info("Pipeline finished in %.2f s", elapsed)


if __name__ == "__main__":
    asyncio.run(run_pipeline())
```

Run with:

```bash
python -m voice_pipeline.skeleton
```

You will see the pipeline stages process in parallel — STT starts before VAD
finishes, LLM starts before STT finishes, and TTS overlaps with LLM streaming.

---

## Exercises

1. Modify `mock_stt_stage` to emit *interim* transcripts (set `is_final=False`)
   every 2 frames and a final transcript every 6 frames. Update `mock_llm_stage`
   to only process final transcripts.

2. Add a `latency_tracker` coroutine that receives `AudioOutput` items and logs
   the time from `AudioChunk.timestamp` (first frame) to `AudioOutput.timestamp`.
   What is the end-to-end latency for the mock pipeline?

3. Change `q_tokens` to `maxsize=2` and reduce TTS sleep to `0.001 s`. Observe
   how back-pressure slows the LLM stage. Explain why this is desirable in
   production.

4. Add a `PipelineMetrics` Pydantic model that tracks: total utterances
   processed, average TTFA, max TTFA. Populate it in `run_pipeline`.

*(Solutions: [solutions/lesson-1.md](solutions/lesson-1.md))*

---

## Self-check

- [ ] I can draw the five-stage voice pipeline from memory
- [ ] I can explain why streaming reduces TTFA vs. turn-based
- [ ] I can connect pipeline stages with `asyncio.Queue` and explain back-pressure
- [ ] I can identify which stage dominates latency in a real system
