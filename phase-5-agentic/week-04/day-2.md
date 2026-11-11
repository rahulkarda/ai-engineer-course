# Phase 5 · Week 04 · Day 2 — Short-Term Memory: Buffer & Scratchpad

> Objective: Build conversation buffer and scratchpad memory components that keep in-context memory bounded  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Short-term memory is everything the agent currently holds **in the context window**. It is fast and zero-latency — no retrieval step — but strictly finite.

Two complementary patterns:

**Conversation Buffer** — the rolling history of user/assistant exchanges. Without management, it grows unbounded and eventually exceeds the model's context limit. A sliding window keeps only the most recent N messages.

**Scratchpad Memory** — a separate working space for intermediate reasoning: tool outputs, partial computations, draft plans. Unlike the conversation buffer (which preserves dialogue), the scratchpad is volatile — it is built up during a single agent run and discarded or summarised at the end.

Design decisions:
- **max_messages** caps the buffer (FIFO eviction: oldest messages drop first).
- **max_tokens** is a stricter budget; requires a token counter.
- Scratchpad entries are keyed by step name so nodes can update their own slot without affecting others.
- Separate the two stores — never mix dialogue history with reasoning scratch notes.

## How it works internally

**FIFO eviction:** `collections.deque(maxlen=N)` gives O(1) append + automatic eviction of the oldest item when capacity is exceeded. For token-based eviction, maintain a running total and pop from the left until the total is within budget.

**Token counting without an API call:** A cheap heuristic: `len(text.split()) * 1.3` overestimates slightly. For production, use `tiktoken` (OpenAI) or the Anthropic `client.count_tokens()` method. In this lesson, a heuristic counter is sufficient.

**Scratchpad as dict:** Each reasoning step stores its output under a string key. When the scratchpad is serialised into the prompt, entries are rendered in insertion order (Python 3.7+ dict guarantee). Steps can overwrite their own key (idempotent retry) or append to a list value.

**Why not use the conversation buffer for scratchpad?** Mixing them forces the model to search through dialogue noise when it needs reasoning context — and makes eviction logic conflate unrelated content types.

## Worked example

```python
"""
day_2_short_term_memory.py

ConversationBuffer — sliding-window message store with FIFO eviction.
ScratchpadMemory  — keyed working memory for intermediate reasoning steps.

No LLM required. All logic is deterministic.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str                          # "user" | "assistant" | "system"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def token_estimate(self) -> int:
        """Rough heuristic: word count * 1.3 rounded up."""
        return int(len(self.content.split()) * 1.3) + 4  # +4 for role overhead


class ScratchpadEntry(BaseModel):
    step: str
    content: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# ConversationBuffer
# ---------------------------------------------------------------------------

class ConversationBuffer:
    """
    Sliding-window conversation history.

    Eviction strategy: FIFO — oldest messages drop first when max_messages
    is exceeded. Token budget is a secondary, softer limit checked on
    get_context(); it does not evict proactively.
    """

    def __init__(
        self,
        max_messages: int = 20,
        token_budget: int = 4096,
        system_prompt: str = "",
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self._buffer: deque[Message] = deque(maxlen=max_messages)
        self.max_messages = max_messages
        self.token_budget = token_budget
        self.system_prompt = system_prompt
        logger.info(
            "ConversationBuffer init — max_messages=%d, token_budget=%d",
            max_messages,
            token_budget,
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> Message:
        """Append a message. Oldest is auto-evicted when buffer is full."""
        msg = Message(role=role, content=content)
        evicted = len(self._buffer) == self.max_messages
        self._buffer.append(msg)
        if evicted:
            logger.debug("add_message: max_messages reached — oldest evicted")
        logger.info(
            "add_message: role=%s, tokens~%d, buffer_size=%d/%d",
            role,
            msg.token_estimate(),
            len(self._buffer),
            self.max_messages,
        )
        return msg

    def trim_oldest(self, n: int = 1) -> list[Message]:
        """Manually remove the n oldest messages. Returns removed messages."""
        removed: list[Message] = []
        for _ in range(min(n, len(self._buffer))):
            removed.append(self._buffer.popleft())
        logger.info("trim_oldest: removed %d message(s)", len(removed))
        return removed

    def clear(self) -> None:
        self._buffer.clear()
        logger.info("ConversationBuffer cleared")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_context(self, enforce_token_budget: bool = True) -> list[Message]:
        """
        Return the current window, optionally trimmed to token_budget.

        Trimming is applied right-to-left (keep most recent messages) so
        recent context is preserved.
        """
        messages = list(self._buffer)

        if not enforce_token_budget:
            return messages

        # count tokens from newest to oldest; keep a suffix that fits
        system_tokens = int(len(self.system_prompt.split()) * 1.3)
        budget_remaining = self.token_budget - system_tokens
        kept: list[Message] = []

        for msg in reversed(messages):
            cost = msg.token_estimate()
            if budget_remaining - cost < 0:
                logger.info(
                    "get_context: token budget exhausted — dropped %d older message(s)",
                    len(messages) - len(kept),
                )
                break
            budget_remaining -= cost
            kept.append(msg)

        return list(reversed(kept))

    @property
    def size(self) -> int:
        return len(self._buffer)

    def total_tokens(self) -> int:
        return sum(m.token_estimate() for m in self._buffer)


# ---------------------------------------------------------------------------
# ScratchpadMemory
# ---------------------------------------------------------------------------

class ScratchpadMemory:
    """
    Keyed working memory for intermediate reasoning steps.

    Each agent node writes its output under a named key.
    The scratchpad is meant to be short-lived (one agent run).
    """

    def __init__(self) -> None:
        self._store: dict[str, ScratchpadEntry] = {}
        logger.info("ScratchpadMemory init")

    def write(self, step: str, content: str) -> ScratchpadEntry:
        """Create or overwrite a step's scratch entry."""
        entry = ScratchpadEntry(step=step, content=content)
        self._store[step] = entry
        logger.info("ScratchpadMemory.write: step='%s', tokens~%d", step, len(content.split()))
        return entry

    def read(self, step: str) -> str | None:
        """Return content for a step, or None if not present."""
        entry = self._store.get(step)
        return entry.content if entry else None

    def delete(self, step: str) -> bool:
        """Remove a step. Returns True if it existed."""
        existed = step in self._store
        self._store.pop(step, None)
        if existed:
            logger.info("ScratchpadMemory.delete: step='%s' removed", step)
        return existed

    def render(self) -> str:
        """Serialise scratchpad into a prompt-ready string block."""
        if not self._store:
            return ""
        lines = ["<scratchpad>"]
        for key, entry in self._store.items():
            lines.append(f"[{key}]")
            lines.append(entry.content)
        lines.append("</scratchpad>")
        return "\n".join(lines)

    def clear(self) -> None:
        self._store.clear()
        logger.info("ScratchpadMemory cleared")

    @property
    def steps(self) -> list[str]:
        return list(self._store.keys())


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def simulate_agent_run() -> None:
    """Simulate a multi-turn agent run using both memory types."""
    system = "You are a helpful assistant."
    buffer = ConversationBuffer(max_messages=6, token_budget=500, system_prompt=system)
    scratch = ScratchpadMemory()

    # turn 1
    buffer.add_message("user", "What is the capital of France?")
    scratch.write("intent_detection", "factual_lookup: geography")
    scratch.write("tool_plan", "no tool needed — answer from knowledge")
    buffer.add_message("assistant", "The capital of France is Paris.")
    scratch.delete("tool_plan")

    # turn 2
    buffer.add_message("user", "What is its population?")
    scratch.write("intent_detection", "factual_lookup: population, referent=Paris")
    buffer.add_message("assistant", "Paris has roughly 2.1 million people in the city proper.")

    # turn 3 — force eviction by adding many messages
    for i in range(5):
        buffer.add_message("user", f"Follow-up question {i + 1}: tell me more.")
        buffer.add_message("assistant", f"Answer {i + 1}: here is some relevant information.")

    context = buffer.get_context()
    logger.info(
        "After 3 turns + 5 follow-ups: context window holds %d/%d messages",
        len(context),
        buffer.size,
    )

    print("\n--- Context window sent to model ---")
    for m in context:
        print(f"  [{m.role:9s}] {m.content[:70]}")

    print("\n--- Scratchpad ---")
    print(scratch.render())

    print(f"\nTotal tokens in buffer: {buffer.total_tokens()}")


if __name__ == "__main__":
    simulate_agent_run()
```

**Run:**
```bash
python day_2_short_term_memory.py
```

Observe FIFO eviction in the logs: the earliest messages are dropped as the buffer fills beyond `max_messages=6`.

## Exercises

1. Add a `max_tokens` hard eviction mode to `ConversationBuffer.add_message`: after appending, proactively trim from the left until `total_tokens() <= max_tokens`. Write a pytest test that asserts the buffer never exceeds the token budget after 20 sequential additions.

2. Implement `ScratchpadMemory.append(step, line)` that adds a line to an existing step's content (newline-separated) rather than replacing it. Write a test verifying that three `append` calls to the same key accumulate correctly.

3. Implement `ConversationBuffer.get_context_as_dicts() -> list[dict]` returning `[{"role": ..., "content": ...}, ...]` — the exact format Anthropic's Messages API expects. Write a pytest fixture that builds a buffer, calls `get_context_as_dicts()`, and validates against the expected schema using Pydantic.

4. Add a `pin_message(msg: Message)` method to `ConversationBuffer` that marks a message as pinned. Pinned messages must never be evicted by FIFO or token-budget trimming. Write tests for both eviction paths.

(Solutions: ../solutions/week-04/day-2.md)

## Self-check

- [ ] I can explain the difference between conversation buffer memory and scratchpad memory, and when to use each.
- [ ] I can explain why FIFO eviction preserves recent context and when that assumption breaks down.
- [ ] I can implement `ConversationBuffer` with `max_messages`, `add_message()`, `get_context()`, and `trim_oldest()` from scratch.
- [ ] I can implement `ScratchpadMemory` with `write()`, `read()`, and `render()` from scratch.
- [ ] I can explain the token estimation heuristic and its limitations compared to a tokeniser.
