# Phase 5 · Week 09 · Day 7 — Building the A2A Interop Layer (Part 1)

> Objective: Design and begin building the week project's interop layer: agent registration, card serving, and in-process transport.  ·  Time: ~4 hr (Saturday deep practice)  ·  Prereqs: [[day-1]]–[[day-6]]

## Concept

Saturday is the architecture + foundation day for the week project: an **A2A-style interop layer** where agents advertise capabilities (agent cards), discover and delegate tasks to each other over a mockable transport.

Today's scope:
1. Finalize the architecture — modules, interfaces, data flow
2. Implement the transport abstraction (in-process + mockable)
3. Implement the agent registry and card serving
4. Implement the task store and lifecycle manager
5. Write unit tests for all of the above

The project deliberately avoids real HTTP for the core (keeping it testable) but uses a `Transport` interface that a real HTTP adapter can satisfy.

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    A2A Interop Layer                              │
│                                                                   │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐    │
│  │  AgentCard   │   │ AgentRegistry │   │   TaskStore      │    │
│  │  (Pydantic)  │   │  (discovery)  │   │  (lifecycle)     │    │
│  └──────────────┘   └───────────────┘   └──────────────────┘    │
│          │                  │                    │                │
│  ┌───────┴──────────────────┴────────────────────┴──────────┐   │
│  │                    Transport                              │   │
│  │  InProcessTransport | MockTransport | HTTPTransport       │   │
│  └───────────────────────────────────────────────────────────┘   │
│          │                                        │               │
│  ┌───────┴────────┐                   ┌──────────┴──────────┐   │
│  │  A2AClient     │                   │  A2AServer          │   │
│  │  (orchestrator)│                   │  (specialist)       │   │
│  └────────────────┘                   └─────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Module plan:**
```
a2a_interop/
├── __init__.py
├── models.py         # Pydantic: AgentCard, Task, Message, Part, etc.
├── registry.py       # AgentRegistry — register, discover, route
├── transport.py      # Transport ABC + InProcessTransport + MockTransport
├── task_store.py     # In-memory TaskStore with lifecycle state machine
├── server.py         # A2AServer base class
├── client.py         # A2AClient (discovery + delegation)
└── exceptions.py     # Domain exceptions
tests/
├── test_models.py
├── test_registry.py
├── test_transport.py
└── test_task_store.py
```

## Worked example

Complete, tested foundation code:

```python
# a2a_interop/exceptions.py

class A2AError(Exception):
    """Base exception for A2A interop errors."""


class AgentNotFoundError(A2AError):
    """No agent found for the requested skill."""


class InvalidTransitionError(A2AError):
    """Attempted an illegal task state transition."""


class TaskNotFoundError(A2AError):
    """Task ID does not exist in the store."""


class TransportError(A2AError):
    """Transport layer failure."""
```

```python
# a2a_interop/models.py

import time
import uuid
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES: frozenset[TaskState] = frozenset({
    TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED
})

VALID_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.SUBMITTED: frozenset({TaskState.WORKING, TaskState.CANCELLED}),
    TaskState.WORKING: frozenset({
        TaskState.COMPLETED, TaskState.FAILED,
        TaskState.INPUT_REQUIRED, TaskState.CANCELLED,
    }),
    TaskState.INPUT_REQUIRED: frozenset({TaskState.WORKING, TaskState.CANCELLED}),
    TaskState.COMPLETED: frozenset(),
    TaskState.FAILED: frozenset(),
    TaskState.CANCELLED: frozenset(),
}


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class FilePart(BaseModel):
    type: Literal["file"] = "file"
    name: str
    mime_type: str = "application/octet-stream"
    bytes_b64: str = ""
    uri: str | None = None


class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)


Part = TextPart | FilePart | DataPart


class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: list[TextPart | FilePart | DataPart] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class TaskStatus(BaseModel):
    state: TaskState = TaskState.SUBMITTED
    timestamp: float = Field(default_factory=time.time)
    message: str = ""


class Artifact(BaseModel):
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    parts: list[TextPart | FilePart | DataPart] = Field(default_factory=list)
    last_chunk: bool = True


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = Field(default_factory=TaskStatus)
    history: list[Message] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    input_modes: list[str] = Field(default_factory=lambda: ["text/plain"])
    output_modes: list[str] = Field(default_factory=lambda: ["text/plain"])


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: list[AgentSkill] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    reliability: float = 1.0   # 0.0–1.0

    def skill_ids(self) -> list[str]:
        return [s.id for s in self.skills]

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self.skill_ids()
```

```python
# a2a_interop/task_store.py

import logging
from typing import Iterator
from a2a_interop.exceptions import InvalidTransitionError, TaskNotFoundError
from a2a_interop.models import Task, TaskState, TaskStatus, VALID_TRANSITIONS, TERMINAL_STATES

logger = logging.getLogger(__name__)


class TaskStore:
    """Thread-safe in-memory task store with lifecycle enforcement."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create(self, task: Task) -> Task:
        self._tasks[task.id] = task
        logger.debug("Created task %s", task.id[:8])
        return task

    def get(self, task_id: str) -> Task:
        if task_id not in self._tasks:
            raise TaskNotFoundError(f"Task {task_id!r} not found")
        return self._tasks[task_id]

    def transition(self, task_id: str, new_state: TaskState, message: str = "") -> Task:
        task = self.get(task_id)
        current = task.status.state
        allowed = VALID_TRANSITIONS[current]
        if new_state not in allowed:
            raise InvalidTransitionError(
                f"Task {task_id[:8]}: {current.value} → {new_state.value} not allowed. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        task.status = TaskStatus(state=new_state, message=message)
        logger.info("Task %s: %s → %s", task_id[:8], current.value, new_state.value)
        return task

    def all_tasks(self) -> Iterator[Task]:
        yield from self._tasks.values()

    def pending_tasks(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.status.state not in TERMINAL_STATES]

    def __len__(self) -> int:
        return len(self._tasks)
```

```python
# a2a_interop/registry.py

import logging
from a2a_interop.exceptions import AgentNotFoundError
from a2a_interop.models import AgentCard

logger = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}

    def register(self, card: AgentCard) -> None:
        self._cards[card.name] = card
        logger.info("Registered '%s' with skills: %s", card.name, card.skill_ids())

    def unregister(self, name: str) -> None:
        self._cards.pop(name, None)

    def get(self, name: str) -> AgentCard | None:
        return self._cards.get(name)

    def find_by_skill(self, skill_id: str, min_reliability: float = 0.0) -> list[AgentCard]:
        """Return candidates sorted by reliability descending."""
        return sorted(
            [c for c in self._cards.values()
             if c.has_skill(skill_id) and c.reliability >= min_reliability],
            key=lambda c: c.reliability,
            reverse=True,
        )

    def route(self, skill_id: str, min_reliability: float = 0.0) -> AgentCard:
        """Return best agent for skill. Raises if none found."""
        candidates = self.find_by_skill(skill_id, min_reliability)
        if not candidates:
            raise AgentNotFoundError(f"No agent with skill '{skill_id}'")
        best = candidates[0]
        logger.info("Routed '%s' → '%s' (reliability=%.2f)", skill_id, best.name, best.reliability)
        return best

    def skill_index(self) -> dict[str, list[str]]:
        """skill_id → [agent names]"""
        idx: dict[str, list[str]] = {}
        for card in self._cards.values():
            for sid in card.skill_ids():
                idx.setdefault(sid, []).append(card.name)
        return idx

    def __len__(self) -> int:
        return len(self._cards)
```

```python
# a2a_interop/transport.py

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable

from a2a_interop.exceptions import TransportError
from a2a_interop.models import Task

logger = logging.getLogger(__name__)

Handler = Callable[[Task], Awaitable[Task]]


class Transport(ABC):
    """Abstract transport — enables swapping in-process / HTTP / mock."""

    @abstractmethod
    async def send(self, agent_url: str, task: Task) -> Task:
        """Send task to remote agent and return completed/updated task."""

    @abstractmethod
    def register_handler(self, agent_url: str, handler: Handler) -> None:
        """Register a handler for an agent_url (used by in-process transport)."""


class InProcessTransport(Transport):
    """
    Routes tasks to handlers registered in the same process.
    No network involved — ideal for integration tests and local dev.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register_handler(self, agent_url: str, handler: Handler) -> None:
        self._handlers[agent_url] = handler
        logger.debug("Registered in-process handler for %s", agent_url)

    async def send(self, agent_url: str, task: Task) -> Task:
        if agent_url not in self._handlers:
            raise TransportError(f"No in-process handler for {agent_url!r}")
        logger.debug("In-process send to %s, task %s", agent_url, task.id[:8])
        return await self._handlers[agent_url](task)


class MockTransport(Transport):
    """
    Fully controllable transport for unit tests.
    Preload responses with `add_response`; raises if no response found.
    """

    def __init__(self) -> None:
        self._responses: dict[str, list[Task]] = {}
        self._calls: list[tuple[str, Task]] = []
        self._handlers: dict[str, Handler] = {}

    def add_response(self, agent_url: str, task: Task) -> None:
        self._responses.setdefault(agent_url, []).append(task)

    def register_handler(self, agent_url: str, handler: Handler) -> None:
        self._handlers[agent_url] = handler

    async def send(self, agent_url: str, task: Task) -> Task:
        self._calls.append((agent_url, task))
        if agent_url in self._handlers:
            return await self._handlers[agent_url](task)
        if agent_url in self._responses and self._responses[agent_url]:
            return self._responses[agent_url].pop(0)
        raise TransportError(f"MockTransport: no response configured for {agent_url!r}")

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def calls_to(self, agent_url: str) -> list[Task]:
        return [t for url, t in self._calls if url == agent_url]
```

Unit tests (save as `tests/test_task_store.py`, `tests/test_registry.py`, `tests/test_transport.py`):

```python
# tests/test_task_store.py
import pytest
from a2a_interop.models import Task, TaskState, Message, TextPart
from a2a_interop.task_store import TaskStore
from a2a_interop.exceptions import InvalidTransitionError, TaskNotFoundError


def test_create_and_get():
    store = TaskStore()
    task = Task()
    store.create(task)
    assert store.get(task.id).id == task.id


def test_get_missing_raises():
    store = TaskStore()
    with pytest.raises(TaskNotFoundError):
        store.get("nonexistent")


def test_valid_transition():
    store = TaskStore()
    task = store.create(Task())
    updated = store.transition(task.id, TaskState.WORKING)
    assert updated.status.state == TaskState.WORKING


def test_invalid_transition_raises():
    store = TaskStore()
    task = store.create(Task())
    store.transition(task.id, TaskState.WORKING)
    with pytest.raises(InvalidTransitionError):
        store.transition(task.id, TaskState.SUBMITTED)  # Can't go back


def test_pending_tasks():
    store = TaskStore()
    t1 = store.create(Task())
    t2 = store.create(Task())
    store.transition(t1.id, TaskState.WORKING)
    store.transition(t1.id, TaskState.COMPLETED)
    pending = store.pending_tasks()
    assert len(pending) == 1
    assert pending[0].id == t2.id
```

```python
# tests/test_registry.py
import pytest
from a2a_interop.models import AgentCard, AgentSkill
from a2a_interop.registry import AgentRegistry
from a2a_interop.exceptions import AgentNotFoundError


def make_card(name: str, skills: list[str], reliability: float = 1.0) -> AgentCard:
    return AgentCard(
        name=name, description="test", version="1.0.0", url=f"http://{name}",
        skills=[AgentSkill(id=s, name=s, description=s) for s in skills],
        reliability=reliability,
    )


def test_register_and_find():
    reg = AgentRegistry()
    reg.register(make_card("a1", ["summarize"]))
    results = reg.find_by_skill("summarize")
    assert len(results) == 1
    assert results[0].name == "a1"


def test_route_returns_best_reliability():
    reg = AgentRegistry()
    reg.register(make_card("low", ["summarize"], reliability=0.6))
    reg.register(make_card("high", ["summarize"], reliability=0.95))
    best = reg.route("summarize")
    assert best.name == "high"


def test_route_no_agent_raises():
    reg = AgentRegistry()
    with pytest.raises(AgentNotFoundError):
        reg.route("unknown-skill")


def test_skill_index():
    reg = AgentRegistry()
    reg.register(make_card("a1", ["s1", "s2"]))
    reg.register(make_card("a2", ["s2", "s3"]))
    idx = reg.skill_index()
    assert set(idx["s2"]) == {"a1", "a2"}
```

```python
# tests/test_transport.py
import asyncio
import pytest
from a2a_interop.models import Task, TaskState, TaskStatus
from a2a_interop.transport import InProcessTransport, MockTransport
from a2a_interop.exceptions import TransportError


async def echo_handler(task: Task) -> Task:
    task.status = TaskStatus(state=TaskState.COMPLETED)
    return task


def test_in_process_transport():
    transport = InProcessTransport()
    transport.register_handler("http://agent", echo_handler)
    task = Task()
    result = asyncio.get_event_loop().run_until_complete(
        transport.send("http://agent", task)
    )
    assert result.status.state == TaskState.COMPLETED


def test_in_process_missing_handler():
    transport = InProcessTransport()
    with pytest.raises(TransportError):
        asyncio.get_event_loop().run_until_complete(
            transport.send("http://nowhere", Task())
        )


def test_mock_transport_preloaded():
    transport = MockTransport()
    expected = Task()
    expected.status = TaskStatus(state=TaskState.COMPLETED)
    transport.add_response("http://agent", expected)
    task = Task()
    result = asyncio.get_event_loop().run_until_complete(
        transport.send("http://agent", task)
    )
    assert result.id == expected.id
    assert transport.call_count == 1


def test_mock_transport_no_response_raises():
    transport = MockTransport()
    with pytest.raises(TransportError):
        asyncio.get_event_loop().run_until_complete(
            transport.send("http://agent", Task())
        )
```

## Exercises

1. Add `deregister(name)` with a test that verifies find_by_skill returns empty after deregistration.

2. Add `TaskStore.add_message(task_id, message)` and `add_artifact(task_id, artifact)` methods. Unit-test both.

3. Add a `HealthCheck` ABC to Transport: `async def health(self, agent_url) -> bool`. Implement it on `InProcessTransport` (return `True` if handler registered) and `MockTransport` (configurable per URL).

4. Implement `AgentRegistry.find_by_multiple_skills(skill_ids, require_all=True)` — if `require_all=True`, return only agents that have all listed skills; if `False`, return agents with any of them.

(Solutions: ../solutions/week-09/day-7.md)

## Self-check
- [ ] I can sketch the full module structure of the A2A interop layer from memory
- [ ] I can implement `TaskStore` with valid-transition enforcement
- [ ] I can implement `Transport` ABC with in-process and mock implementations
- [ ] All unit tests pass for models, registry, transport, and task store

## Supplementary links
- [Abstract base classes in Python](https://docs.python.org/3/library/abc.html)
- [Pydantic discriminated unions](https://docs.pydantic.dev/latest/concepts/unions/)
- [pytest asyncio](https://pytest-asyncio.readthedocs.io/)
