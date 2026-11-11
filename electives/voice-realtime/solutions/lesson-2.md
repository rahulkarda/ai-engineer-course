# Solutions: Lesson 2 — VAD & Turn Detection

## Exercise 1 — Realistic MockVADBackend pattern

```python
class RealisticMockVADBackend:
    """
    Simulates: 20 frames speech, 30 frames silence, 15 frames speech.
    Frame sequence:
      0-19:  speech
      20-49: silence
      50-64: speech
    """
    _PATTERN: list[bool] = (
        [True] * 20 + [False] * 30 + [True] * 15
    )

    def __init__(self) -> None:
        self._count = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        result = (
            self._PATTERN[self._count]
            if self._count < len(self._PATTERN)
            else False
        )
        self._count += 1
        return result


def test_realistic_mock_emits_two_segments():
    backend = RealisticMockVADBackend()
    detector = TurnDetector(backend, silence_end_ms=600, min_speech_ms=200)
    # Feed enough frames: 20 speech + 30 silence + 15 speech + 30 silence (to end turn 2)
    total_frames = 20 + 30 + 15 + 30
    frames = [b"\x00" * FRAME_BYTES] * total_frames
    segments: list[SpeechSegment] = []
    for frame in frames:
        segments.extend(detector.feed(frame))
    assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
```

## Exercise 2 — pytest for TurnDetector

```python
import pytest
from voice_pipeline.vad import TurnDetector, MockVADBackend, FRAME_BYTES, FRAME_DURATION_MS


class ScriptedVADBackend:
    """Returns True/False per the provided sequence, then False."""
    def __init__(self, sequence: list[bool]) -> None:
        self._seq = sequence
        self._idx = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        if self._idx < len(self._seq):
            result = self._seq[self._idx]
            self._idx += 1
            return result
        return False


def test_turn_detector_emits_correct_duration():
    # 15 speech frames + 30 silence frames = end of turn
    # min_speech=200ms → 10 frames; silence_end=600ms → 30 frames
    speech_frames = 15
    silence_frames = 30
    seq = [True] * speech_frames + [False] * silence_frames
    backend = ScriptedVADBackend(seq)
    detector = TurnDetector(backend, silence_end_ms=600, min_speech_ms=200)
    frames = [b"\x00" * FRAME_BYTES] * (speech_frames + silence_frames)
    segments: list = []
    for frame in frames:
        segments.extend(detector.feed(frame))
    assert len(segments) == 1
    # Duration = (speech + silence) frames × frame_duration_ms
    expected_ms = (speech_frames + silence_frames) * FRAME_DURATION_MS
    assert segments[0].total_duration_ms() == pytest.approx(expected_ms)
```

## Exercise 3 — Pre-speech padding ring buffer

```python
from collections import deque

class TurnDetectorWithPadding(TurnDetector):
    def __init__(self, *args, padding_ms: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        padding_frames = padding_ms // FRAME_DURATION_MS
        self._ring: deque[bytes] = deque(maxlen=max(1, padding_frames))

    def _process_frame(self, frame: bytes) -> SpeechSegment | None:
        # Keep a rolling buffer of pre-speech frames
        if self._state == VADState.IDLE:
            self._ring.append(frame)
        result = super()._process_frame(frame)
        if result and len(result.frames) > 0:
            # Prepend ring buffer frames to segment
            prepend = list(self._ring)[:-1]  # exclude current frame (already in segment)
            result.frames = prepend + result.frames
        return result
```

## Exercise 4 — flush() method

```python
def flush(self) -> SpeechSegment | None:
    """Force-emit any buffered speech regardless of silence duration."""
    if self._state in (VADState.SPEAKING, VADState.TRAILING_SILENCE):
        if self._segment.speech_frame_count >= self._min_speech_frames:
            finished = self._segment
            self._state = VADState.IDLE
            self._segment = SpeechSegment()
            self._silence_count = 0
            log.info("VAD: flushed segment (%.0f ms speech)", finished.speech_frame_count * FRAME_DURATION_MS)
            return finished
    return None
```

Test:

```python
def test_flush_emits_incomplete_segment():
    seq = [True] * 20  # 20 speech frames, no silence
    backend = ScriptedVADBackend(seq)
    detector = TurnDetectorWithPadding(backend, silence_end_ms=600, min_speech_ms=200)
    frames = [b"\x00" * FRAME_BYTES] * 20
    segments = []
    for frame in frames:
        segments.extend(detector.feed(frame))
    assert len(segments) == 0  # not finished naturally
    flushed = detector.flush()
    assert flushed is not None
    assert flushed.speech_frame_count == 20
```
