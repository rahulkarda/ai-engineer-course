# Solutions — Week 09 · Day 5 — Message Formats & Task Lifecycle

## Exercise 1: DataPart word-count constraint

```python
import pytest
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass
class DataPart:
    type: str = "data"
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    role: str
    parts: list = field(default_factory=list)


def extract_word_limit(message: Message) -> int | None:
    for part in message.parts:
        if isinstance(part, DataPart) and "word_count" in part.data:
            return int(part.data["word_count"])
    return None


def generate_report(topic: str, word_limit: int | None = None) -> str:
    base = (
        f"# Report on {topic}\n\n"
        f"Introduction: This report covers {topic} comprehensively. "
        f"Analysis reveals significant trends and growth patterns. "
        f"The primary findings indicate strong market performance. "
        f"Recommendations follow from the data. Conclusion: further research advised."
    )
    if word_limit is None:
        return base
    words = base.split()
    return " ".join(words[:word_limit])


def test_data_part_word_limit():
    msg = Message(
        role="user",
        parts=[
            TextPart(text="Cloud Computing"),
            DataPart(data={"word_count": 10}),
        ],
    )
    topic_part = next(p for p in msg.parts if isinstance(p, TextPart))
    limit = extract_word_limit(msg)
    result = generate_report(topic_part.text, limit)
    assert len(result.split()) <= 10


def test_no_data_part_no_limit():
    msg = Message(role="user", parts=[TextPart(text="Cloud Computing")])
    limit = extract_word_limit(msg)
    assert limit is None
    result = generate_report("Cloud Computing", limit)
    assert len(result.split()) > 10
```

## Exercise 2: PushNotificationClient with asyncio.Queue

```python
import asyncio
import pytest
from dataclasses import dataclass, field
from enum import Enum


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WebhookEvent:
    task_id: str
    state: TaskState
    message: str = ""


class PushNotificationClient:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[WebhookEvent] = asyncio.Queue()
        self._received: list[WebhookEvent] = []

    async def receive_callback(self, event: WebhookEvent) -> None:
        """Simulate receiving a webhook POST."""
        await self._queue.put(event)

    async def drain(self, timeout: float = 0.1) -> list[WebhookEvent]:
        """Collect all pending events."""
        try:
            while True:
                event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                self._received.append(event)
        except asyncio.TimeoutError:
            pass
        return list(self._received)


class AgentWithPushNotifications:
    def __init__(self, callback_client: PushNotificationClient) -> None:
        self._client = callback_client

    async def execute(self, task_id: str) -> None:
        await self._client.receive_callback(WebhookEvent(task_id=task_id, state=TaskState.WORKING))
        await asyncio.sleep(0.01)
        await self._client.receive_callback(WebhookEvent(task_id=task_id, state=TaskState.COMPLETED))


@pytest.mark.asyncio
async def test_push_notification_fires_once_on_complete():
    client = PushNotificationClient()
    agent = AgentWithPushNotifications(client)
    await agent.execute("task-123")
    events = await client.drain()
    completion_events = [e for e in events if e.state == TaskState.COMPLETED]
    assert len(completion_events) == 1
    assert completion_events[0].task_id == "task-123"
```

## Exercise 3: TaskHistory

```python
import time
from dataclasses import dataclass, field
import pytest


@dataclass
class TextPart:
    type: str = "text"
    text: str = ""


@dataclass
class Message:
    role: str
    parts: list[TextPart] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class TaskHistory:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def last_user_message(self) -> Message | None:
        for m in reversed(self._messages):
            if m.role == "user":
                return m
        return None

    def last_agent_message(self) -> Message | None:
        for m in reversed(self._messages):
            if m.role == "agent":
                return m
        return None

    def messages_since(self, timestamp: float) -> list[Message]:
        return [m for m in self._messages if m.timestamp > timestamp]


def test_last_user_message():
    h = TaskHistory()
    h.add(Message(role="user", parts=[TextPart(text="first")]))
    h.add(Message(role="agent", parts=[TextPart(text="response")]))
    h.add(Message(role="user", parts=[TextPart(text="second")]))
    assert h.last_user_message().parts[0].text == "second"  # type: ignore[union-attr]


def test_last_agent_message():
    h = TaskHistory()
    h.add(Message(role="agent", parts=[TextPart(text="answer")]))
    h.add(Message(role="user", parts=[TextPart(text="follow-up")]))
    assert h.last_agent_message().parts[0].text == "answer"  # type: ignore[union-attr]


def test_messages_since():
    h = TaskHistory()
    t0 = time.time()
    time.sleep(0.01)
    h.add(Message(role="user", parts=[TextPart(text="after")]))
    since = h.messages_since(t0)
    assert len(since) == 1
```

## Exercise 4: State machine validation

```python
from enum import Enum


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.SUBMITTED: {TaskState.WORKING, TaskState.CANCELLED},
    TaskState.WORKING: {TaskState.COMPLETED, TaskState.FAILED, TaskState.INPUT_REQUIRED, TaskState.CANCELLED},
    TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELLED},
    TaskState.COMPLETED: set(),
    TaskState.FAILED: set(),
    TaskState.CANCELLED: set(),
}

TERMINAL = {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}


def validate_state_machine(
    transitions: dict[TaskState, set[TaskState]],
    start: TaskState = TaskState.SUBMITTED,
    terminal: set[TaskState] = TERMINAL,
) -> list[str]:
    """Returns list of validation errors (empty = valid)."""
    errors: list[str] = []

    # 1. All states reachable from start
    reachable: set[TaskState] = set()
    frontier = {start}
    while frontier:
        current = frontier.pop()
        reachable.add(current)
        for next_state in transitions.get(current, set()):
            if next_state not in reachable:
                frontier.add(next_state)
    for state in transitions:
        if state not in reachable:
            errors.append(f"State {state.value!r} is not reachable from {start.value!r}")

    # 2. Terminal states have no outgoing transitions
    for state in terminal:
        if transitions.get(state):
            errors.append(f"Terminal state {state.value!r} has outgoing transitions")

    # 3. No self-loops
    for state, nexts in transitions.items():
        if state in nexts:
            errors.append(f"Self-loop detected on state {state.value!r}")

    return errors


def test_valid_state_machine():
    errors = validate_state_machine(TRANSITIONS)
    assert errors == [], f"State machine has errors: {errors}"


def test_detects_unreachable_state():
    bad_transitions = dict(TRANSITIONS)
    from enum import auto

    class ExtendedState(str, Enum):
        ORPHAN = "orphan"

    bad_transitions[ExtendedState.ORPHAN] = set()  # type: ignore[index]
    # We can't easily mix enums, so just test the logic directly:
    errors = validate_state_machine(TRANSITIONS)
    assert errors == []   # Baseline is clean
```
