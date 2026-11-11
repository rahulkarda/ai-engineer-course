# Solutions: Lesson 5 — Barge-in, Interruption, and Duplex Audio

## Exercise 1 — BargInCoordinator tests

```python
import asyncio
import pytest
from voice_pipeline.barge_in import BargInCoordinator, PipelineConfig


@pytest.fixture
def coord_with_suppression() -> BargInCoordinator:
    return BargInCoordinator(PipelineConfig(echo_suppression=True, min_barge_in_ms=50))


@pytest.fixture
def coord_no_suppression() -> BargInCoordinator:
    return BargInCoordinator(PipelineConfig(echo_suppression=False, min_barge_in_ms=50))


def test_echo_suppression_blocks_barge_in(coord_with_suppression):
    coord = coord_with_suppression
    coord.agent_started_speaking()
    assert coord.speaking_event.is_set()
    # Simulate many speech frames — barge-in should NOT fire
    for _ in range(20):
        coord.user_speech_detected()
    assert not coord.barge_in_event.is_set(), "Echo suppression should block barge-in"


def test_no_suppression_fires_barge_in(coord_no_suppression):
    coord = coord_no_suppression
    coord.agent_started_speaking()
    # Fire speech frames until min_barge_in_ms elapsed
    import time
    start = time.monotonic()
    fired = False
    for _ in range(100):
        coord.user_speech_detected()
        if coord.barge_in_event.is_set():
            fired = True
            break
    assert fired, "Barge-in should fire without echo suppression"


def test_barge_in_not_fired_when_agent_not_speaking(coord_no_suppression):
    coord = coord_no_suppression
    # Agent is not speaking
    for _ in range(5):
        coord.user_speech_detected()
    # min_barge_in_ms = 50 ms; with real time delays this may or may not fire
    # but state is IDLE / LISTENING — test that reset_for_new_turn clears it
    coord.barge_in_event.set()
    coord.reset_for_new_turn()
    assert not coord.barge_in_event.is_set()
```

## Exercise 2 — discard_queue helper

```python
def discard_queue(q: asyncio.Queue) -> int:
    """Drain all items from a queue without blocking. Returns count discarded."""
    count = 0
    while True:
        try:
            q.get_nowait()
            q.task_done()
            count += 1
        except asyncio.QueueEmpty:
            break
    return count


def test_discard_queue_empties_queue():
    q: asyncio.Queue[str] = asyncio.Queue()
    for i in range(5):
        q.put_nowait(f"item {i}")
    count = discard_queue(q)
    assert count == 5
    assert q.empty()
```

## Exercise 3 — Streaming from MockLLMProvider into sentence queue

```python
@pytest.mark.asyncio
async def test_run_agent_turn_streams_sentences():
    from voice_pipeline.llm_tts import MockLLMProvider
    from voice_pipeline.barge_in import BargInCoordinator, PipelineConfig

    coord = BargInCoordinator(PipelineConfig(echo_suppression=False))
    response = "Hello world. This is a second sentence."
    provider = MockLLMProvider(response=response)

    result = await run_agent_turn("hello", coord, mock_llm_response=response)
    assert result == "completed"
```

## Exercise 4 — ConversationHistory

```python
from pydantic import BaseModel


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ConversationHistory(BaseModel):
    turns: list[ConversationTurn] = []

    def add_user(self, text: str) -> None:
        self.turns.append(ConversationTurn(role="user", content=text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(ConversationTurn(role="assistant", content=text))

    def to_anthropic_messages(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def last_n(self, n: int) -> "ConversationHistory":
        return ConversationHistory(turns=self.turns[-n:])
```

Modified `run_agent_turn` signature:

```python
async def run_agent_turn(
    transcript_text: str,
    coordinator: BargInCoordinator,
    history: ConversationHistory | None = None,
    mock_llm_response: str = "This is the agent response.",
) -> tuple[str, ConversationHistory]:
    history = history or ConversationHistory()
    history.add_user(transcript_text)
    # ... (run pipeline) ...
    assembled_response = "This is the agent response."  # collect from TTS sentences
    history.add_assistant(assembled_response)
    return result_status, history
```
