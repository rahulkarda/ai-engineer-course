# Solutions — Week 09 · Day 1 — The Agent Interoperability Problem

## Exercise 1: Extended registry with reliability scoring

```python
from dataclasses import dataclass
import pytest

AGENT_REGISTRY: dict[str, dict] = {
    "nlp-agent-v1": {"capabilities": ["summarization", "translation"], "reliability": 0.95},
    "nlp-agent-v2": {"capabilities": ["summarization", "key-extraction", "sentiment"], "reliability": 0.88},
    "sql-agent": {"capabilities": ["sql-generation", "query-explanation"], "reliability": 0.92},
}


def find_agents_for_task(task: str) -> list[tuple[str, float]]:
    """Return list of (agent_id, reliability) sorted by reliability descending."""
    matches = [
        (agent_id, info["reliability"])
        for agent_id, info in AGENT_REGISTRY.items()
        if task in info["capabilities"]
    ]
    return sorted(matches, key=lambda x: x[1], reverse=True)


def test_find_returns_all_matches():
    results = find_agents_for_task("summarization")
    assert len(results) == 2


def test_find_sorted_by_reliability():
    results = find_agents_for_task("summarization")
    reliabilities = [r for _, r in results]
    assert reliabilities == sorted(reliabilities, reverse=True)


def test_find_no_matches():
    results = find_agents_for_task("image-generation")
    assert results == []
```

## Exercise 2: TaskHandle design

```python
from dataclasses import dataclass, field
from enum import Enum
import uuid


class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskHandle:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskState = TaskState.SUBMITTED
    result: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}

    def poll(self) -> "TaskHandle":
        """In a real system, this would call the remote agent's GET /tasks/{id}."""
        return self


def test_task_handle_defaults():
    handle = TaskHandle()
    assert handle.status == TaskState.SUBMITTED
    assert handle.result is None
    assert not handle.is_terminal


def test_task_handle_completed():
    handle = TaskHandle(status=TaskState.COMPLETED, result="done")
    assert handle.is_terminal
    assert handle.result == "done"
```

## Exercise 3: Message translation

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentAMessage:
    action: str
    text: str


@dataclass
class AgentBExpects:
    task: str
    content: str


FIELD_MAPPINGS: dict[str, dict[str, str]] = {
    "AtoB": {"action": "task", "text": "content"},
    "BtoA": {"task": "action", "content": "text"},
}


def translate_message(msg: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Translate a message dict using a field mapping."""
    return {mapping[k]: v for k, v in msg.items() if k in mapping}


def test_a_to_b_mapping():
    msg_a = {"action": "analyze", "text": "quarterly report"}
    translated = translate_message(msg_a, FIELD_MAPPINGS["AtoB"])
    assert translated == {"task": "analyze", "content": "quarterly report"}


def test_b_to_a_mapping():
    msg_b = {"task": "summarize", "content": "meeting transcript"}
    translated = translate_message(msg_b, FIELD_MAPPINGS["BtoA"])
    assert translated == {"action": "summarize", "text": "meeting transcript"}
```

## Exercise 4: Research notes (model answer)

The Google A2A protocol addresses the three failure modes as follows:

- **Semantic mismatch:** A2A defines a standard message envelope with typed `Parts` (`TextPart`, `FilePart`, `DataPart`). Agents express their input/output expectations in the `AgentCard` skills list. All callers use the same envelope regardless of the agent's internal implementation.

- **No capability discovery:** Every A2A server publishes an `AgentCard` at `/.well-known/agent.json`. Orchestrators fetch this URL to discover skills automatically. A central registry can index multiple cards for routing without bespoke integration.

- **Lifecycle opacity:** A2A defines six explicit task states (`submitted`, `working`, `input-required`, `completed`, `failed`, `cancelled`). Clients can poll `GET /tasks/{id}` or receive Server-Sent Events for real-time updates. The `input-required` state handles multi-turn interactions without breaking the protocol.
