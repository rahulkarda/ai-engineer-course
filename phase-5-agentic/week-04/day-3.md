# Phase 5 · Week 04 · Day 3 — Context Window Management & Summarization

> Objective: Build a context manager that summarises old messages to stay within token budgets indefinitely  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Naïve FIFO buffering has a critical flaw: it permanently loses information. Once a message is evicted, the agent has no memory of it. For long-running conversations this means the agent "forgets" its own commitments, earlier user preferences, and established facts.

**Summarization strategy** solves this by compressing the evicted history into a compact summary block that is prepended to the context on every turn. The agent always has access to the gist of the full history, even when raw messages are gone.

The pipeline:
1. Maintain a rolling buffer of recent messages (short-term, verbatim).
2. Track a `summary` string representing everything older than the buffer.
3. When buffer exceeds a threshold, extract the oldest N messages, call an LLM to summarise them, and merge the result into the existing summary.
4. On each LLM call, prepend `[Summary of earlier conversation]: <summary>` before the raw buffer.

**Progressive summarization** builds the summary incrementally — each round merges new content into the existing summary rather than re-summarising the whole history from scratch. This keeps summarisation cost O(new messages) not O(total history).

Trade-offs:
- Summarisation introduces latency (one extra LLM call) and cost.
- Information loss is real — summaries compress and omit detail.
- Raw history can be stored separately (long-term store, day 4) for auditing even when excluded from context.

## How it works internally

**Token budget accounting:**

```
total_budget = model_context_limit
reserved_for_output = 1024  (typical)
available = total_budget - reserved_for_output
system_tokens = count(system_prompt)
summary_tokens = count(summary)
raw_budget = available - system_tokens - summary_tokens
```

The raw buffer fills `raw_budget`. When a new message would overflow it, the summarisation trigger fires.

**Summarisation trigger:** fire when `raw_token_count > trigger_threshold` (e.g., 80% of `raw_budget`). Summarise the oldest `batch_size` messages, remove them from the buffer, append their summary to the running summary string.

**Merge prompt pattern:**
```
Existing summary:
<existing_summary>

New messages to incorporate:
<message_batch>

Write an updated, concise summary that preserves all key facts, decisions, and commitments.
```

This is idempotent: if the LLM produces identical output for identical input, the state converges.

## Worked example

```python
"""
day_3_context_summarization.py

ContextManager — token-budgeted sliding window with summarisation trigger.
SummarizationMemory — wraps ContextManager and calls a (mocked) LLM to
                      progressively summarise evicted messages.

LLM is mocked: set ANTHROPIC_API_KEY env var to any value for the live path,
but the code calls mock_summarize() by default so no real API call is made.
Model string is claude-opus-5-5.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def token_estimate(self) -> int:
        return int(len(self.content.split()) * 1.3) + 4


# ---------------------------------------------------------------------------
# Mock LLM summariser (used when ANTHROPIC_API_KEY is absent or in tests)
# ---------------------------------------------------------------------------

def mock_summarize(existing_summary: str, messages: list[Message]) -> str:
    """
    Deterministic mock: concatenates existing summary with a compressed
    representation of the messages. Does not call any external API.
    """
    turns = "; ".join(
        f"{m.role}: {m.content[:40]}" for m in messages
    )
    if existing_summary:
        return f"{existing_summary} | {turns}"
    return f"[Summary] {turns}"


def live_summarize(existing_summary: str, messages: list[Message]) -> str:
    """
    Real summarisation via Anthropic API.
    Only called when ANTHROPIC_API_KEY is set.
    Requires: pip install anthropic
    """
    import anthropic  # type: ignore[import]

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    batch_text = "\n".join(f"{m.role}: {m.content}" for m in messages)
    prompt_parts = []
    if existing_summary:
        prompt_parts.append(f"Existing summary:\n{existing_summary}\n")
    prompt_parts.append(
        f"New messages to incorporate:\n{batch_text}\n\n"
        "Write an updated, concise summary that preserves all key facts, "
        "decisions, and commitments. Be brief."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
    )
    return response.content[0].text.strip()


def summarize(existing_summary: str, messages: list[Message]) -> str:
    """Dispatch to mock or live summariser based on env."""
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("summarize: using live Anthropic API (model=%s)", MODEL)
        return live_summarize(existing_summary, messages)
    logger.info("summarize: ANTHROPIC_API_KEY not set — using mock")
    return mock_summarize(existing_summary, messages)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Manages the raw message buffer within a token budget.

    Does NOT perform summarisation itself — it signals when the budget
    is exceeded and provides the eviction batch to the caller.
    """

    def __init__(self, token_budget: int = 2048) -> None:
        if token_budget < 256:
            raise ValueError("token_budget must be >= 256")
        self.token_budget = token_budget
        self._messages: list[Message] = []
        self._used_tokens: int = 0
        logger.info("ContextManager init — token_budget=%d", token_budget)

    def add_message(self, role: str, content: str) -> tuple[Message, list[Message]]:
        """
        Append a message. Returns (new_message, evicted_batch).

        evicted_batch is non-empty when the buffer exceeded the budget;
        those messages must be summarised by the caller.
        """
        msg = Message(role=role, content=content)
        self._messages.append(msg)
        self._used_tokens += msg.token_estimate()
        logger.info(
            "add_message: role=%s, used_tokens=%d/%d",
            role,
            self._used_tokens,
            self.token_budget,
        )

        evicted: list[Message] = []
        # evict oldest messages until we are within budget
        while self._used_tokens > self.token_budget and len(self._messages) > 1:
            oldest = self._messages.pop(0)
            self._used_tokens -= oldest.token_estimate()
            evicted.append(oldest)
            logger.debug("add_message: evicted role=%s tokens=%d", oldest.role, oldest.token_estimate())

        if evicted:
            logger.info(
                "add_message: evicted %d message(s) — summarisation needed",
                len(evicted),
            )

        return msg, evicted

    def get_windowed_context(self) -> list[Message]:
        """Return current in-budget messages."""
        return list(self._messages)

    @property
    def used_tokens(self) -> int:
        return self._used_tokens


# ---------------------------------------------------------------------------
# SummarizationMemory
# ---------------------------------------------------------------------------

class SummarizationMemory:
    """
    Combines ContextManager with progressive summarisation.

    Maintains a running summary of evicted messages and prepends it to
    every context retrieval so the model always has the full history gist.
    """

    def __init__(self, token_budget: int = 2048) -> None:
        self._ctx = ContextManager(token_budget=token_budget)
        self._summary: str = ""
        self._total_messages_seen: int = 0
        self._summarization_count: int = 0
        logger.info("SummarizationMemory init")

    def add_message(self, role: str, content: str) -> None:
        """Add a message; triggers summarisation if eviction occurs."""
        _msg, evicted = self._ctx.add_message(role, content)
        self._total_messages_seen += 1

        if evicted:
            self._summary = summarize(self._summary, evicted)
            self._summarization_count += 1
            logger.info(
                "SummarizationMemory: summarisation #%d — summary length %d chars",
                self._summarization_count,
                len(self._summary),
            )

    def get_context(self) -> dict[str, str | list[Message]]:
        """
        Return the full context payload:
          {
            "summary": str,              # empty string if no eviction yet
            "messages": list[Message],   # current windowed messages
          }
        """
        return {
            "summary": self._summary,
            "messages": self._ctx.get_windowed_context(),
        }

    def render_for_prompt(self) -> str:
        """Format context as a prompt-ready string."""
        ctx = self.get_context()
        parts: list[str] = []
        if ctx["summary"]:
            parts.append(f"[Earlier conversation summary]\n{ctx['summary']}\n")
        parts.append("[Recent messages]")
        for m in ctx["messages"]:  # type: ignore[union-attr]
            parts.append(f"{m.role}: {m.content}")
        return "\n".join(parts)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_messages_seen": self._total_messages_seen,
            "messages_in_window": len(self._ctx.get_windowed_context()),
            "summarization_count": self._summarization_count,
            "used_tokens": self._ctx.used_tokens,
        }


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def simulate_long_conversation(num_turns: int = 30) -> None:
    """Simulate a conversation that exceeds the token budget."""
    mem = SummarizationMemory(token_budget=300)

    topics = [
        "Tell me about Paris.",
        "What about its history?",
        "Who built the Eiffel Tower?",
        "When was it built?",
        "What is French cuisine known for?",
        "Name some famous French artists.",
        "What is the French political system?",
        "How large is France by area?",
        "What languages are spoken there?",
        "What is the GDP of France?",
    ]

    for i in range(num_turns):
        user_text = topics[i % len(topics)]
        assistant_text = f"That is a great question about topic {i + 1}. Here is a detailed answer with several sentences to consume tokens."
        mem.add_message("user", user_text)
        mem.add_message("assistant", assistant_text)

    print("\n--- Final rendered context ---")
    print(mem.render_for_prompt()[:800])
    print("\n--- Stats ---")
    for k, v in mem.stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    simulate_long_conversation(num_turns=30)
```

**Run:**
```bash
python day_3_context_summarization.py
```

With `token_budget=300`, summarisation triggers several times across 30 turns. Observe the summary growing progressively in the logs.

**Run with real API (optional):**
```bash
ANTHROPIC_API_KEY=your_key python day_3_context_summarization.py
```

## Exercises

1. Write a pytest test for `ContextManager.add_message` that:
   - Adds 10 messages, each with content `"word " * 20` (20 words).
   - Asserts that `used_tokens` never exceeds `token_budget` after any addition.
   - Asserts that every returned `evicted` batch is non-empty when overflow occurs.

2. Add a `summarization_threshold` parameter to `SummarizationMemory` (e.g., `0.8`). Summarisation should trigger when `used_tokens / token_budget > threshold`, not only on overflow. Rewrite the trigger logic and add a test.

3. Implement `SummarizationMemory.rehydrate(summary: str, messages: list[Message])` that restores memory state from a previously serialised snapshot. Write a pytest test that serialises, clears, and rehydrates, then verifies `render_for_prompt()` produces identical output.

4. The current mock `mock_summarize` produces a pipe-delimited string. Write a pytest parametrize test over three different `existing_summary` inputs (empty, short, long) and verify the output format is consistent and never shorter than the input summary.

(Solutions: ../solutions/week-04/day-3.md)

## Self-check

- [ ] I can explain why FIFO eviction alone is insufficient for long-running agents.
- [ ] I can explain the progressive summarisation pattern and why it is cheaper than full re-summarisation.
- [ ] I can implement `ContextManager` with token-budget eviction from scratch.
- [ ] I can implement `SummarizationMemory` that wraps `ContextManager` and triggers summarisation on eviction.
- [ ] I can explain the token budget formula and calculate `raw_budget` given system prompt and summary sizes.
