# Elective: Voice & Realtime — Lesson 5
# Barge-in, Interruption, and Duplex Audio

> Objective: Implement barge-in (user interrupts agent speech) and true full-duplex audio handling.  
> Time: ~60 min  
> Prereqs: [[lesson-4.md]]

---

## Concept

**Barge-in** is the ability of the user to interrupt the agent mid-response —
just like interrupting a person in conversation. Without it, users must wait for
the agent to finish speaking before they can respond, which feels robotic.

**Full-duplex audio** means the microphone and speaker operate simultaneously.
This is physically possible on modern hardware but creates a software challenge:
the VAD must distinguish between the agent's playback audio (leaked into the mic)
and the user's actual voice.

### The barge-in problem

When the agent is speaking:
1. The agent's TTS audio plays through the speaker.
2. The microphone picks up the agent's voice (acoustic echo).
3. VAD may mis-classify echo as user speech.
4. The agent interrupts itself.

Solutions ranked by complexity:

| Approach | Complexity | Quality |
|----------|-----------|---------|
| Half-duplex lockout: mute mic while agent speaks | trivial | poor UX |
| Echo suppression flag: tell VAD to ignore audio during playback | easy | good |
| Acoustic Echo Cancellation (AEC): webrtcvad AEC or OS-level AEC | medium | excellent |
| Wake-word gating: user must say wake word to barge-in | easy | acceptable for some UX |

The **echo suppression flag** approach is the most practical starting point: keep
a `is_agent_speaking: asyncio.Event` flag; VAD ignores speech frames while the
event is set. The downside is it also ignores real user speech that overlaps with
agent speech — acceptable for most use cases.

### Cancellation architecture

When barge-in is detected:
1. **Cancel the TTS stage**: discard pending `sentence_in` queue items.
2. **Stop audio playback**: signal the player to stop mid-chunk.
3. **Cancel the LLM stage**: abandon in-flight streaming generation.
4. **Resume from VAD**: start processing the new user utterance.

In asyncio, this maps to `asyncio.Task.cancel()` + `asyncio.CancelledError`
handling in each stage.

---

## How it works internally

### asyncio cancellation

```python
task = asyncio.create_task(llm_stage(...))
# ... barge-in detected ...
task.cancel()
try:
    await task
except asyncio.CancelledError:
    pass  # clean shutdown
```

Each stage must handle `CancelledError` gracefully: flush queues, release
resources, reset state. Use `try/finally` blocks.

### The barge-in event

```
AudioIn → VAD → [barge_in_event fires] → cancel TTS/LLM tasks → re-enter SPEAKING state
```

`barge_in_event` is an `asyncio.Event` shared across the pipeline coordinator.
VAD sets it; the coordinator awaits it in a `select`-style `asyncio.wait`.

### Interruptible pipeline pattern

```python
async def run_turn(transcript: Transcript, barge_in: asyncio.Event) -> None:
    llm_task = asyncio.create_task(llm_and_tts(transcript))
    barge_in_task = asyncio.create_task(barge_in.wait())
    done, pending = await asyncio.wait(
        [llm_task, barge_in_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if barge_in_task in done:
        log.info("Barge-in detected — turn cancelled")
        barge_in.clear()
```

---

## Worked example

```python
"""
voice_pipeline/barge_in.py
Barge-in handling and duplex audio coordination.
All audio is mocked — no hardware required.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

log = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()  # STT+LLM in flight
    SPEAKING = auto()    # TTS playing


@dataclass
class AudioChunk:
    data: bytes
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class PipelineConfig:
    echo_suppression: bool = True      # mute VAD during agent speech
    min_barge_in_ms: float = 200.0     # ignore speech < this during playback
    tts_queue_maxsize: int = 4


class BargInCoordinator:
    """
    Coordinates barge-in between the VAD/audio-in path and the TTS/audio-out path.

    Responsibilities:
    - Track whether agent is speaking (controls echo suppression).
    - Expose `barge_in_event` for the pipeline to await.
    - Expose `speaking_event` for VAD to check before classifying speech.
    """
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self.barge_in_event: asyncio.Event = asyncio.Event()
        self.speaking_event: asyncio.Event = asyncio.Event()
        self._speech_start: float | None = None
        self._state = AgentState.IDLE

    @property
    def state(self) -> AgentState:
        return self._state

    def agent_started_speaking(self) -> None:
        self._state = AgentState.SPEAKING
        self.speaking_event.set()
        log.debug("BargIn: agent started speaking")

    def agent_stopped_speaking(self) -> None:
        self._state = AgentState.LISTENING
        self.speaking_event.clear()
        self._speech_start = None
        log.debug("BargIn: agent stopped speaking")

    def user_speech_detected(self) -> None:
        """Called by VAD when a speech frame arrives."""
        if not self._config.echo_suppression or not self.speaking_event.is_set():
            # Not suppressing, or agent not speaking — check barge-in
            if self._speech_start is None:
                self._speech_start = time.monotonic()
            elapsed = (time.monotonic() - self._speech_start) * 1000
            if elapsed >= self._config.min_barge_in_ms:
                log.info("BargIn: barge-in detected after %.0f ms speech", elapsed)
                self.barge_in_event.set()
        else:
            log.debug("BargIn: echo suppressed")

    def reset_for_new_turn(self) -> None:
        self.barge_in_event.clear()
        self._speech_start = None
        self._state = AgentState.LISTENING


async def interruptible_tts_player(
    sentence_queue: asyncio.Queue[str],
    coordinator: BargInCoordinator,
    tts_provider: object,  # TTSProvider protocol
    stop_event: asyncio.Event,
) -> None:
    """
    Plays TTS sentences one by one; stops immediately on stop_event.
    Notifies coordinator of speaking state changes.
    """
    coordinator.agent_started_speaking()
    try:
        while not stop_event.is_set():
            try:
                sentence = sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Queue drained — agent finished speaking naturally
                break
            log.info("TTS player: '%s'", sentence[:40])
            chunk = await tts_provider.synthesize(sentence)  # type: ignore[attr-defined]
            if stop_event.is_set():
                log.info("TTS player: interrupted mid-sentence")
                break
            # Simulate playback duration
            play_duration = len(chunk.data) / 16000  # mock: 1 byte = 1 sample
            await asyncio.sleep(min(play_duration, 0.05))
    finally:
        coordinator.agent_stopped_speaking()
        log.info("TTS player: stopped")


async def run_agent_turn(
    transcript_text: str,
    coordinator: BargInCoordinator,
    mock_llm_response: str = "This is the agent response. It may be interrupted.",
) -> str:
    """
    Simulates one complete agent turn with barge-in support.
    Returns "completed" or "interrupted".
    """
    from voice_pipeline.llm_tts import MockTTSProvider, SentenceBuffer  # noqa: PLC0415

    coordinator.reset_for_new_turn()
    stop_event = asyncio.Event()
    sentence_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
    tts = MockTTSProvider(latency_ms=20)

    # Produce sentences into queue
    buf = SentenceBuffer()
    words = mock_llm_response.split()
    for i, word in enumerate(words):
        delta = word + (" " if i < len(words) - 1 else "")
        for s in buf.feed(delta):
            await sentence_queue.put(s)
    remainder = buf.flush()
    if remainder:
        await sentence_queue.put(remainder)

    # Run TTS player and barge-in watcher concurrently
    player_task = asyncio.create_task(
        interruptible_tts_player(sentence_queue, coordinator, tts, stop_event)
    )
    barge_in_task = asyncio.create_task(coordinator.barge_in_event.wait())

    done, pending = await asyncio.wait(
        [player_task, barge_in_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    # Cancel whatever didn't finish
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    if barge_in_task in done:
        stop_event.set()
        log.info("Turn result: interrupted by barge-in")
        return "interrupted"

    log.info("Turn result: completed")
    return "completed"


async def demo_barge_in() -> None:
    """Demonstrate barge-in: fire barge-in after 100 ms."""
    config = PipelineConfig(echo_suppression=False, min_barge_in_ms=50)
    coord = BargInCoordinator(config)

    async def delayed_barge_in() -> None:
        await asyncio.sleep(0.1)
        # Simulate user speech frames
        for _ in range(5):
            coord.user_speech_detected()
            await asyncio.sleep(0.01)

    barge_task = asyncio.create_task(delayed_barge_in())
    result = await run_agent_turn(
        "Hello, how are you?",
        coord,
        mock_llm_response=(
            "I am doing very well. Thank you for asking. "
            "The weather today is quite pleasant."
        ),
    )
    await barge_task
    log.info("Demo complete — turn result: %s", result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_barge_in())
```

---

## Exercises

1. Write a pytest test for `BargInCoordinator`:
   - With `echo_suppression=True`, verify that `user_speech_detected()` does
     NOT set `barge_in_event` while `speaking_event` is set.
   - With `echo_suppression=False`, verify it DOES set `barge_in_event` after
     enough calls.

2. `interruptible_tts_player` stops cleanly but does not drain `sentence_queue`.
   Add a `discard_queue` helper that reads all remaining items from a queue
   without blocking, and call it after stopping. Write a test.

3. The `run_agent_turn` function currently mocks the LLM response with a fixed
   string. Refactor it to accept a `MockLLMProvider` instance instead, and
   demonstrate streaming directly into the sentence queue using the provider's
   `stream_tokens()` method.

4. Implement a `ConversationHistory` Pydantic model that stores `role` and
   `content` pairs. Modify `run_agent_turn` to accept and update history, so
   subsequent turns have context.

*(Solutions: [solutions/lesson-5.md](solutions/lesson-5.md))*

---

## Self-check

- [ ] I can explain echo suppression vs. AEC and when each is appropriate
- [ ] I can implement barge-in using `asyncio.wait` + `asyncio.Event`
- [ ] I understand how `asyncio.Task.cancel()` propagates through a pipeline
- [ ] I can design an `AgentState` machine for a duplex voice agent
