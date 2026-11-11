# Solutions — Phase 5 · Week 04 · Day 1 — Typed Agent State

---

## Exercise 1

Extend `ConversationState` to include a `tool_calls: list[ToolCall]` field where `ToolCall` is a
Pydantic model with `name: str`, `args: dict[str, Any]`, `result: str | None = None`. Write a
unit test.

```python
"""
typed_state.py — Typed agent state with ToolCall support.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """A single tool invocation recorded in agent state."""

    name: str
    args: dict[str, Any]
    result: str | None = None


class Message(BaseModel):
    """A single conversation message."""

    role: str  # "user" | "assistant" | "system"
    content: str


class ConversationState(BaseModel):
    """Typed agent state for a multi-turn conversation."""

    messages: list[Message] = Field(default_factory=list)
    turn_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
```

```python
"""
tests/test_typed_state_ex1.py — Tests for ToolCall field.
"""
from __future__ import annotations

import pytest

from typed_state import ConversationState, Message, ToolCall


def test_tool_call_defaults() -> None:
    tc = ToolCall(name="search", args={"query": "python"})
    assert tc.name == "search"
    assert tc.args == {"query": "python"}
    assert tc.result is None


def test_tool_call_with_result() -> None:
    tc = ToolCall(name="calculate", args={"expression": "2+2"}, result="4")
    assert tc.result == "4"


def test_conversation_state_stores_tool_calls() -> None:
    state = ConversationState(
        messages=[Message(role="user", content="Hello")],
        turn_count=1,
        tool_calls=[
            ToolCall(name="search", args={"query": "weather"}, result="Sunny"),
        ],
    )
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].name == "search"


def test_tool_calls_default_empty() -> None:
    state = ConversationState()
    assert state.tool_calls == []


def test_tool_call_args_arbitrary_types() -> None:
    tc = ToolCall(name="fetch", args={"ids": [1, 2, 3], "flag": True})
    assert tc.args["ids"] == [1, 2, 3]
```

---

## Exercise 2

Implement `merge_states(base: ConversationState, update: ConversationState) -> ConversationState`
that appends messages and sums `turn_count`. Test it.

```python
"""
state_merge.py — Merge two ConversationState instances.
"""
from __future__ import annotations

import logging

from typed_state import ConversationState

logger = logging.getLogger(__name__)


def merge_states(
    base: ConversationState, update: ConversationState
) -> ConversationState:
    """Return a new state combining *base* and *update*.

    - messages: base + update (appended)
    - turn_count: summed
    - tool_calls: base + update (appended)
    """
    return ConversationState(
        messages=base.messages + update.messages,
        turn_count=base.turn_count + update.turn_count,
        tool_calls=base.tool_calls + update.tool_calls,
    )
```

```python
"""
tests/test_state_merge.py — Tests for merge_states.
"""
from __future__ import annotations

import pytest

from typed_state import ConversationState, Message, ToolCall
from state_merge import merge_states


def test_merge_messages_appended() -> None:
    base = ConversationState(
        messages=[Message(role="user", content="Hi")],
        turn_count=1,
    )
    update = ConversationState(
        messages=[Message(role="assistant", content="Hello")],
        turn_count=1,
    )
    merged = merge_states(base, update)
    assert len(merged.messages) == 2
    assert merged.messages[0].content == "Hi"
    assert merged.messages[1].content == "Hello"


def test_merge_turn_count_summed() -> None:
    base = ConversationState(turn_count=3)
    update = ConversationState(turn_count=7)
    merged = merge_states(base, update)
    assert merged.turn_count == 10


def test_merge_tool_calls_appended() -> None:
    base = ConversationState(
        tool_calls=[ToolCall(name="search", args={"q": "a"})]
    )
    update = ConversationState(
        tool_calls=[ToolCall(name="calculate", args={"expr": "1+1"})]
    )
    merged = merge_states(base, update)
    assert len(merged.tool_calls) == 2


def test_merge_empty_states() -> None:
    merged = merge_states(ConversationState(), ConversationState())
    assert merged.turn_count == 0
    assert merged.messages == []


def test_merge_does_not_mutate_base() -> None:
    base = ConversationState(
        messages=[Message(role="user", content="Hi")], turn_count=1
    )
    update = ConversationState(
        messages=[Message(role="assistant", content="Hey")], turn_count=1
    )
    _ = merge_states(base, update)
    assert len(base.messages) == 1  # unchanged
```

---

## Exercise 3

Add a validator that raises `ValueError` if `turn_count` exceeds 100.

```python
"""
state_validated.py — ConversationState with turn_count validator.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

MAX_TURNS = 100


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: str | None = None


class Message(BaseModel):
    role: str
    content: str


class ConversationState(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    turn_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @field_validator("turn_count")
    @classmethod
    def turn_count_within_limit(cls, v: int) -> int:
        if v > MAX_TURNS:
            raise ValueError(
                f"turn_count {v} exceeds maximum allowed value of {MAX_TURNS}"
            )
        return v
```

```python
"""
tests/test_state_validated.py — Tests for turn_count validator.
"""
from __future__ import annotations

import pytest

from state_validated import ConversationState, Message


def test_valid_turn_count() -> None:
    state = ConversationState(turn_count=50)
    assert state.turn_count == 50


def test_turn_count_at_limit() -> None:
    state = ConversationState(turn_count=100)
    assert state.turn_count == 100


def test_turn_count_exceeds_limit() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        ConversationState(turn_count=101)
    assert "turn_count" in str(exc_info.value)


def test_turn_count_zero() -> None:
    state = ConversationState()
    assert state.turn_count == 0


def test_negative_turn_count_allowed() -> None:
    # Validator only blocks > 100; negatives pass (add min=0 if desired)
    state = ConversationState(turn_count=0)
    assert state.turn_count == 0
```

---

## Exercise 4

Implement a snapshot history (up to 5 snapshots) with rollback.

```python
"""
state_history.py — State snapshot history with rollback.
"""
from __future__ import annotations

import logging
from collections import deque
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_SNAPSHOTS = 5


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]
    result: str | None = None


class Message(BaseModel):
    role: str
    content: str


class ConversationState(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    turn_count: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)


class StateHistory:
    """Stores up to MAX_SNAPSHOTS immutable snapshots of ConversationState."""

    def __init__(self) -> None:
        self._snapshots: deque[ConversationState] = deque(maxlen=MAX_SNAPSHOTS)

    def snapshot(self, state: ConversationState) -> None:
        """Push a deep copy of *state* onto the history."""
        self._snapshots.append(state.model_copy(deep=True))
        logger.debug("Snapshot saved. Total snapshots: %d", len(self._snapshots))

    def rollback(self, n: int) -> ConversationState:
        """Return snapshot at index *n* (0 = oldest stored, -1 = latest).

        Raises IndexError when *n* is out of range.
        """
        snapshots = list(self._snapshots)
        return snapshots[n].model_copy(deep=True)

    def __len__(self) -> int:
        return len(self._snapshots)
```

```python
"""
tests/test_state_history.py — Tests for StateHistory snapshot and rollback.
"""
from __future__ import annotations

import pytest

from state_history import ConversationState, Message, StateHistory


def _make_state(turn: int, content: str) -> ConversationState:
    return ConversationState(
        messages=[Message(role="user", content=content)],
        turn_count=turn,
    )


def test_snapshot_stores_state() -> None:
    history = StateHistory()
    state = _make_state(1, "hello")
    history.snapshot(state)
    assert len(history) == 1


def test_rollback_returns_correct_state() -> None:
    history = StateHistory()
    history.snapshot(_make_state(1, "first"))
    history.snapshot(_make_state(2, "second"))
    rolled = history.rollback(0)
    assert rolled.turn_count == 1
    assert rolled.messages[0].content == "first"


def test_rollback_latest_snapshot() -> None:
    history = StateHistory()
    history.snapshot(_make_state(1, "a"))
    history.snapshot(_make_state(2, "b"))
    rolled = history.rollback(-1)
    assert rolled.turn_count == 2


def test_max_five_snapshots_enforced() -> None:
    history = StateHistory()
    for i in range(7):
        history.snapshot(_make_state(i, f"msg-{i}"))
    assert len(history) == 5
    # Oldest should be index 2 (0,1 evicted)
    assert history.rollback(0).turn_count == 2


def test_rollback_returns_deep_copy() -> None:
    history = StateHistory()
    state = _make_state(1, "original")
    history.snapshot(state)
    rolled = history.rollback(-1)
    rolled.turn_count = 999
    assert history.rollback(-1).turn_count == 1  # snapshot unaffected


def test_rollback_out_of_range() -> None:
    history = StateHistory()
    with pytest.raises(IndexError):
        history.rollback(0)
```

---

## Exercise 5

Implement `state_diff(s1, s2) -> dict` returning changed fields.

```python
"""
state_diff.py — Compute field-level diff between two ConversationState instances.
"""
from __future__ import annotations

import logging
from typing import Any

from state_history import ConversationState

logger = logging.getLogger(__name__)


def state_diff(
    s1: ConversationState, s2: ConversationState
) -> dict[str, dict[str, Any]]:
    """Return a dict of fields that differ between *s1* and *s2*.

    Each entry has the form:
        field_name -> {"before": <s1_value>, "after": <s2_value>}
    """
    d1 = s1.model_dump()
    d2 = s2.model_dump()
    diff: dict[str, dict[str, Any]] = {}
    for key in d1:
        if d1[key] != d2.get(key):
            diff[key] = {"before": d1[key], "after": d2.get(key)}
    for key in d2:
        if key not in d1:
            diff[key] = {"before": None, "after": d2[key]}
    return diff
```

```python
"""
tests/test_state_diff.py — Tests for state_diff.
"""
from __future__ import annotations

import pytest

from state_history import ConversationState, Message, ToolCall
from state_diff import state_diff


def test_no_diff_for_equal_states() -> None:
    s = ConversationState(turn_count=5)
    assert state_diff(s, s) == {}


def test_diff_turn_count() -> None:
    s1 = ConversationState(turn_count=1)
    s2 = ConversationState(turn_count=3)
    diff = state_diff(s1, s2)
    assert "turn_count" in diff
    assert diff["turn_count"]["before"] == 1
    assert diff["turn_count"]["after"] == 3


def test_diff_messages() -> None:
    s1 = ConversationState(messages=[Message(role="user", content="hi")])
    s2 = ConversationState(
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
    )
    diff = state_diff(s1, s2)
    assert "messages" in diff
    assert len(diff["messages"]["after"]) == 2


def test_diff_multiple_fields() -> None:
    s1 = ConversationState(turn_count=0)
    s2 = ConversationState(
        turn_count=2,
        tool_calls=[ToolCall(name="search", args={"q": "test"})],
    )
    diff = state_diff(s1, s2)
    assert "turn_count" in diff
    assert "tool_calls" in diff


def test_diff_does_not_include_unchanged_fields() -> None:
    s1 = ConversationState(turn_count=5, messages=[Message(role="user", content="x")])
    s2 = ConversationState(turn_count=5, messages=[Message(role="user", content="x")])
    assert state_diff(s1, s2) == {}
```
