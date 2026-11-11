# Elective: Voice & Realtime — Lesson 2
# VAD & Turn Detection

> Objective: Implement Voice Activity Detection and reliable turn-end detection.  
> Time: ~60 min  
> Prereqs: [[lesson-1.md]]

---

## Concept

**Voice Activity Detection (VAD)** is a binary classifier running on short audio
frames (~10–30 ms) that answers: *is a human speaking right now?*

Without VAD, STT runs on silence, background noise, and half-words — burning
compute and producing garbage transcripts. VAD gates the pipeline, letting only
speech frames through.

**Turn detection** is the harder problem: *has the user finished their turn?*
A pause mid-sentence is not a turn end. A 600 ms silence after a question usually
is. Getting this wrong causes two failure modes:

- **Early cut-off**: agent interrupts the user mid-sentence → frustrating.
- **Late cut-off**: agent waits too long → sluggish, feels unresponsive.

### VAD approaches

| Approach | Library | Latency | Accuracy | Notes |
|----------|---------|---------|---------|-------|
| Energy threshold | custom | near-zero | poor | fails in noise |
| webrtcvad | `webrtcvad` | near-zero | good | WebRTC's production VAD |
| Silero VAD | `torch` | ~5 ms/frame | excellent | best accuracy, heavier |
| Cloud VAD | Deepgram/others | +50 ms RTT | excellent | server-side, adds latency |

`webrtcvad` is the standard choice for local, low-latency VAD. It classifies
16-bit 16 kHz PCM frames of exactly 10, 20, or 30 ms.

### Turn-end heuristics

1. **Silence duration gate**: end-of-turn when VAD reports N consecutive silent
   frames (e.g., 600 ms = 30 frames × 20 ms).
2. **Trailing silence + minimum utterance length**: require at least 300 ms of
   speech before accepting any silence as a turn end.
3. **Semantic completion** (advanced): send partial STT to a lightweight model
   and ask if the sentence is syntactically complete.

---

## How it works internally

### webrtcvad internals

webrtcvad uses a Gaussian Mixture Model trained on hundreds of thousands of
speech/noise samples. It outputs a binary decision per frame using features
derived from sub-band energy levels. Mode 3 (most aggressive filtering) rejects
more noise but may clip soft speech endings.

### Frame sizing

Audio arrives as a continuous byte stream. VAD requires fixed-size frames:

```
Frame size = sample_rate × frame_duration_ms / 1000 × bytes_per_sample
           = 16000 × 20 / 1000 × 2  = 640 bytes per frame (20 ms)
```

A `FrameBuffer` class must accumulate raw bytes and yield fixed-size frames.

### State machine

```
IDLE ──[speech frame]──► SPEAKING
SPEAKING ──[speech frame]──► SPEAKING (reset silence counter)
SPEAKING ──[silence frame]──► TRAILING_SILENCE (increment counter)
TRAILING_SILENCE ──[speech frame]──► SPEAKING (reset)
TRAILING_SILENCE ──[silence >= threshold]──► END_OF_TURN → emit
END_OF_TURN ──► IDLE
```

---

## Worked example

```python
"""
voice_pipeline/vad.py
Mock-compatible VAD stage with turn-detection state machine.
Install webrtcvad only for real audio: uv add webrtcvad
"""
import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000        # Hz
FRAME_DURATION_MS = 20      # 10, 20, or 30 ms for webrtcvad
BYTES_PER_SAMPLE = 2        # 16-bit PCM
FRAME_BYTES = SAMPLE_RATE * FRAME_DURATION_MS * BYTES_PER_SAMPLE // 1000  # 640


class VADState(Enum):
    IDLE = auto()
    SPEAKING = auto()
    TRAILING_SILENCE = auto()


@dataclass
class SpeechSegment:
    """Collected audio frames forming one complete utterance."""
    frames: list[bytes] = field(default_factory=list)
    speech_frame_count: int = 0

    def total_duration_ms(self) -> float:
        return len(self.frames) * FRAME_DURATION_MS

    def as_bytes(self) -> bytes:
        return b"".join(self.frames)


@runtime_checkable
class VADBackend(Protocol):
    """Anything that can classify an audio frame."""
    def is_speech(self, frame: bytes, sample_rate: int) -> bool: ...


class MockVADBackend:
    """Deterministic mock: first 3 frames = silence, rest = speech."""
    def __init__(self, speech_start_frame: int = 3) -> None:
        self._count = 0
        self._speech_start = speech_start_frame

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        self._count += 1
        return self._count > self._speech_start


class WebRTCVADBackend:
    """Real webrtcvad backend — only import when available."""
    def __init__(self, aggressiveness: int = 2) -> None:
        try:
            import webrtcvad  # noqa: PLC0415
            self._vad = webrtcvad.Vad(aggressiveness)
        except ImportError as e:
            raise RuntimeError(
                "webrtcvad not installed. Run: uv add webrtcvad"
            ) from e

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        return self._vad.is_speech(frame, sample_rate)


class TurnDetector:
    """
    State machine: collects audio frames, emits SpeechSegment on turn end.

    Args:
        backend: VADBackend instance (real or mock).
        silence_end_ms: silence duration (ms) that signals end of turn.
        min_speech_ms: minimum speech before silence can end a turn.
    """
    def __init__(
        self,
        backend: VADBackend,
        silence_end_ms: int = 600,
        min_speech_ms: int = 250,
    ) -> None:
        self._backend = backend
        self._silence_frames = silence_end_ms // FRAME_DURATION_MS
        self._min_speech_frames = min_speech_ms // FRAME_DURATION_MS
        self._state = VADState.IDLE
        self._segment = SpeechSegment()
        self._silence_count = 0
        self._frame_buffer: deque[bytes] = deque()

    def feed(self, raw_bytes: bytes) -> list[SpeechSegment]:
        """
        Feed raw audio bytes; returns list of completed SpeechSegments.
        May return 0 or more segments per call.
        """
        # Buffer bytes, yield fixed frames
        self._frame_buffer.extend(
            raw_bytes[i : i + FRAME_BYTES]
            for i in range(0, len(raw_bytes), FRAME_BYTES)
            if len(raw_bytes) - i >= FRAME_BYTES
        )
        completed: list[SpeechSegment] = []
        while self._frame_buffer:
            frame = self._frame_buffer.popleft()
            result = self._process_frame(frame)
            if result:
                completed.append(result)
        return completed

    def _process_frame(self, frame: bytes) -> SpeechSegment | None:
        is_speech = self._backend.is_speech(frame, SAMPLE_RATE)

        if self._state == VADState.IDLE:
            if is_speech:
                self._state = VADState.SPEAKING
                self._segment = SpeechSegment()
                self._segment.frames.append(frame)
                self._segment.speech_frame_count = 1
                log.debug("VAD: speech started")

        elif self._state == VADState.SPEAKING:
            self._segment.frames.append(frame)
            if is_speech:
                self._segment.speech_frame_count += 1
                self._silence_count = 0
            else:
                self._state = VADState.TRAILING_SILENCE
                self._silence_count = 1

        elif self._state == VADState.TRAILING_SILENCE:
            self._segment.frames.append(frame)
            if is_speech:
                self._state = VADState.SPEAKING
                self._silence_count = 0
                self._segment.speech_frame_count += 1
            else:
                self._silence_count += 1
                enough_speech = (
                    self._segment.speech_frame_count >= self._min_speech_frames
                )
                if self._silence_count >= self._silence_frames and enough_speech:
                    log.info(
                        "VAD: end of turn, %.0f ms speech",
                        self._segment.speech_frame_count * FRAME_DURATION_MS,
                    )
                    finished = self._segment
                    self._state = VADState.IDLE
                    self._segment = SpeechSegment()
                    self._silence_count = 0
                    return finished
        return None


async def vad_stage(
    raw_in: asyncio.Queue[bytes],
    segment_out: asyncio.Queue[SpeechSegment],
    backend: VADBackend | None = None,
    silence_end_ms: int = 600,
) -> None:
    """Async stage: reads raw audio bytes, emits complete SpeechSegments."""
    _backend = backend or MockVADBackend()
    detector = TurnDetector(_backend, silence_end_ms=silence_end_ms)
    while True:
        raw = await raw_in.get()
        for segment in detector.feed(raw):
            await segment_out.put(segment)
        raw_in.task_done()
```

---

## Exercises

1. The `MockVADBackend` marks everything after frame 3 as speech forever.
   Extend it to simulate a realistic pattern: speech for 20 frames, silence
   for 30 frames, speech for 15 frames. Verify `TurnDetector` emits two
   `SpeechSegment` objects.

2. Write a pytest test for `TurnDetector` that feeds exactly enough speech +
   silence frames to trigger one end-of-turn event, and asserts the returned
   `SpeechSegment.total_duration_ms()` is correct.

3. Add a `padding_ms` parameter to `TurnDetector`: include that many ms of
   pre-speech audio (from a ring buffer) in the segment, so the first syllable
   is not clipped. Hint: keep a `collections.deque` of recent frames.

4. The state machine ignores frames shorter than `FRAME_BYTES`. What happens
   at the very end of an audio stream (tail bytes)? Add a `flush() -> SpeechSegment | None`
   method that forces emission of any buffered speech.

*(Solutions: [solutions/lesson-2.md](solutions/lesson-2.md))*

---

## Self-check

- [ ] I can explain the difference between VAD and turn detection
- [ ] I can implement the IDLE→SPEAKING→TRAILING_SILENCE state machine
- [ ] I can size audio frames correctly for webrtcvad
- [ ] I understand the min-speech guard and why it prevents early cut-off
