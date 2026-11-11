# Phase 5 · Week 04 · Day 1 — Typed Agent State

> Objective: Design robust, validated state schemas for multi-node LangGraph agents  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-03]]

## Concept

Agent state is the single source of truth flowing through every node in a graph. Each node reads state, performs work, and returns a partial update — LangGraph merges those updates into the running state object.

Two approaches exist for defining state:

| Approach | Pros | Cons |
|---|---|---|
| `TypedDict` | Lightweight, no runtime overhead | No validation, no defaults, no methods |
| `Pydantic BaseModel` | Full validation, defaults, computed fields, serialisation | Slightly more verbose; must configure `model_config` |

For production agents, use **Pydantic BaseModel**. It catches malformed state updates at the boundary rather than propagating corrupt data through downstream nodes.

Key design rules:
- State fields should be **append-only or monotonically evolving** (e.g., message lists grow; turn count increases).
- Avoid mutable defaults — use `Field(default_factory=...)`.
- Separate **conversation data** (messages) from **operational metadata** (turn count, timestamps, error flags).
- Snapshots (frozen copies) enable rollback, debugging, and evaluation harnesses.

## How it works internally

LangGraph's `StateGraph` stores state as a plain dict internally. When you annotate the graph with a state schema, LangGraph:

1. Validates the initial state dict against your schema at graph entry.
2. After each node returns a dict of updates, merges them via **reducer functions** (default: last-write-wins; annotated fields can use custom reducers like `operator.add` for lists).
3. Re-validates if you use Pydantic — this catches a node that returns a wrong type for a field.

When you use `Annotated[list[Message], operator.add]`, LangGraph knows to **append** the returned list rather than replace the existing one. This is the correct pattern for message accumulation.

State evolution:
- Node returns `{"messages": [new_msg]}` → LangGraph appends to existing list.
- Node returns `{"turn_count": state.turn_count + 1}` → last-write-wins replaces the field.
- Snapshots are just `model.model_dump()` calls captured at node boundaries.

## Worked example

```python
"""
day_1_typed_state.py

Demonstrates typed agent state with Pydantic v2 and a
simulated LangGraph-style node execution loop.
No LLM required — all state transitions are deterministic.
"""

from __future__ import annotations

import logging
import operator
from datetime import datetime, timezone
from typing import Annotated, Any

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str                          # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMetadata(BaseModel):
    session_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = "claude-opus-5-5"
    error: str | None = None


class ConversationState(BaseModel):
    """
    Top-level agent state.

    messages uses Annotated so downstream reducers know to append.
    In a real StateGraph this annotation drives the merge strategy;
    here we replicate that logic manually.
    """
    messages: list[Message] = Field(default_factory=list)
    metadata: ConversationMetadata
    turn_count: int = 0
    snapshots: list[dict[str, Any]] = Field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable copy of the current state."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Node functions (pure: state in, partial update out)
# ---------------------------------------------------------------------------

def user_input_node(state: ConversationState, user_text: str) -> dict[str, Any]:
    """Simulate receiving user input."""
    msg = Message(role="user", content=user_text)
    logger.info("user_input_node: received '%s'", user_text[:60])
    return {"messages": [msg], "turn_count": state.turn_count + 1}


def echo_assistant_node(state: ConversationState) -> dict[str, Any]:
    """
    Simulate an assistant node — echoes the last user message.
    In production this would call an LLM.
    """
    last_user_msg = next(
        (m for m in reversed(state.messages) if m.role == "user"), None
    )
    if last_user_msg is None:
        logger.warning("echo_assistant_node: no user message found")
        return {}

    reply_text = f"Echo: {last_user_msg.content}"
    msg = Message(role="assistant", content=reply_text)
    logger.info("echo_assistant_node: replying with '%s'", reply_text[:60])
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# Graph runner (manual, no LangGraph dependency needed for this lesson)
# ---------------------------------------------------------------------------

def apply_update(state: ConversationState, update: dict[str, Any]) -> ConversationState:
    """
    Merge a node's partial update back into state.

    Replicates LangGraph's default reducer behaviour:
    - list fields: append (operator.add)
    - scalar fields: last-write-wins
    """
    current = state.model_dump()

    for key, value in update.items():
        if key not in current:
            logger.error("apply_update: unknown field '%s' — skipping", key)
            continue
        if isinstance(current[key], list):
            # append strategy (mirrors Annotated[list, operator.add])
            current[key] = current[key] + (value if isinstance(value, list) else [value])
        else:
            current[key] = value

    return ConversationState(**current)


def run_graph(initial_state: ConversationState, turns: list[str]) -> ConversationState:
    """Drive the graph for a sequence of user turns."""
    state = initial_state
    logger.info("Graph started — session %s", state.metadata.session_id)

    for user_text in turns:
        # capture snapshot before each turn
        snap = state.snapshot()
        state = ConversationState(
            **{**state.model_dump(), "snapshots": state.snapshots + [snap]}
        )

        # node 1: ingest user message
        update = user_input_node(state, user_text)
        state = apply_update(state, update)

        # node 2: generate assistant reply
        update = echo_assistant_node(state)
        state = apply_update(state, update)

        logger.info(
            "Turn %d complete — %d messages in state",
            state.turn_count,
            len(state.messages),
        )

    logger.info("Graph finished — %d turns, %d snapshots", state.turn_count, len(state.snapshots))
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initial = ConversationState(
        metadata=ConversationMetadata(session_id="session-001"),
    )

    final_state = run_graph(
        initial_state=initial,
        turns=["Hello, agent!", "What can you do?", "Tell me more."],
    )

    print(f"\nFinal turn count: {final_state.turn_count}")
    print(f"Total messages:   {len(final_state.messages)}")
    print(f"Snapshots stored: {len(final_state.snapshots)}")
    print("\nMessage log:")
    for m in final_state.messages:
        print(f"  [{m.role:9s}] {m.content}")
```

**Run:**
```bash
python day_1_typed_state.py
```

Expected output includes three user messages, three assistant echoes, and three snapshots (one per turn).

## Exercises

1. Add a `ToolCallRecord` model (tool name, input dict, output str, timestamp) and a `tool_calls: list[ToolCallRecord]` field to `ConversationState`. Add a node that appends a fake tool call each turn and verify state merging works correctly.

2. Introduce a `status` field of type `Literal["active", "paused", "completed"]` with a validator that prevents transitioning from `"completed"` back to `"active"`. Write a pytest test that asserts the validator raises `ValidationError`.

3. Modify `apply_update` to raise a custom `StateUpdateError` (inherit from `ValueError`) when an unknown field key is encountered, instead of logging and skipping. Update the tests accordingly.

4. Implement a `diff_snapshots(snap_a, snap_b)` function that returns only the keys that changed between two snapshots. Use it to print a per-turn diff log in `run_graph`.

(Solutions: ../solutions/week-04/day-1.md)

## Self-check

- [ ] I can explain why Pydantic BaseModel is preferred over TypedDict for production agent state.
- [ ] I can explain what a reducer function does in LangGraph and name two built-in strategies.
- [ ] I can implement a `ConversationState` model from scratch with correct field defaults and an append-only messages list.
- [ ] I can implement a pure node function that returns a partial state update dict.
- [ ] I can explain what a state snapshot is and why it is useful for debugging and evaluation.
