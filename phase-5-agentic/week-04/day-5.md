# Phase 5 · Week 04 · Day 5 — Episodic vs Semantic Memory

> Objective: Distinguish episodic and semantic memory, implement both types, and combine them in an agent memory system.  ·  Time: ~1 hr  ·  Prereqs: [[day-4.md]]

## Concept

Human memory has two fundamentally different modes:

**Episodic memory** — *what happened, when*. Specific events anchored in time. "At 2pm yesterday, the user asked me about refunds." These are **instances**: concrete, time-stamped, personal.

**Semantic memory** — *general knowledge*. Facts not tied to a specific moment. "Python lists are mutable." These are **abstractions**: context-independent, timeless, impersonal.

From an ML perspective:
- Episodic entries are **sequential records** — stored in insertion order, retrieved by recency or time range.
- Semantic facts are **embedded representations** — stored as vectors, retrieved by similarity to a query.

Both types are necessary for a capable agent:
- Episodic: tracks conversation history, user actions, recent context.
- Semantic: stores grounded facts, learned domain knowledge, tool documentation.

Neither replaces the other. An agent that only has episodic memory cannot generalize. One that only has semantic memory cannot follow a multi-turn conversation.

## How it works internally

**Episodic storage**

Each episode is a `(timestamp, event_text, metadata)` triple. Retrieval patterns:
- *Recency*: return the N most recent events (sort by timestamp descending, slice).
- *Time range*: filter by `start_time <= timestamp <= end_time`.
- *Tag filter*: filter metadata for a specific key/value.

Storage is append-only. Episodes accumulate. Without periodic summarization (see Week 04 Day 3), the list grows unboundedly. Practical agents cap episodic storage or compress old episodes into semantic facts.

**Semantic storage**

Semantic facts live in a vector store. On write, the text is embedded; on read, a query is embedded and cosine similarity finds the nearest facts. The key property: retrieval is **content-addressed**, not position-addressed. You do not need to know *when* a fact was stored — you just describe what you need.

**Combining them**

When an agent receives a new message, it typically:
1. Appends the turn to episodic memory (always).
2. Queries semantic memory for relevant background facts (similarity search).
3. Constructs a prompt from: system instructions + semantic facts + recent episodes + current message.
4. After responding, optionally extracts new semantic facts from the exchange and stores them.

## Worked example

```python
"""
day5_memory_types.py

Episodic vs Semantic memory — runnable demo.
Uses deterministic hash-based fake embeddings (no LLM/API key required).
"""
from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake embedder — deterministic, no API key required
# ---------------------------------------------------------------------------

EMBED_DIM = 16


def _fake_embed(text: str) -> list[float]:
    """Produce a deterministic pseudo-random unit vector from text."""
    seed = int(hashlib.md5(text.lower().encode()).hexdigest(), 16)
    raw = []
    for i in range(EMBED_DIM):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        raw.append((seed / 0xFFFFFFFF) * 2 - 1)
    magnitude = math.sqrt(sum(v * v for v in raw))
    return [v / magnitude for v in raw]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Episode(BaseModel):
    """A single episodic memory entry."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticFact(BaseModel):
    """A semantic fact with its embedding."""
    fact: str
    embedding: list[float]
    source: str = "unknown"


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """
    Stores time-stamped events.
    Retrieval: by recency or time range.
    """

    def __init__(self, max_episodes: int = 100) -> None:
        self._episodes: list[Episode] = []
        self._max = max_episodes
        self._log = logging.getLogger(self.__class__.__name__)

    def add(self, event: str, **metadata: Any) -> Episode:
        ep = Episode(event=event, metadata=metadata)
        self._episodes.append(ep)
        if len(self._episodes) > self._max:
            dropped = self._episodes.pop(0)
            self._log.debug("Evicted oldest episode: %s", dropped.event[:40])
        self._log.info("Stored episode: %s", event[:60])
        return ep

    def recent(self, n: int = 5) -> list[Episode]:
        """Return the n most recent episodes, newest last."""
        return self._episodes[-n:]

    def by_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> list[Episode]:
        return [ep for ep in self._episodes if start <= ep.timestamp <= end]

    def by_tag(self, key: str, value: Any) -> list[Episode]:
        return [ep for ep in self._episodes if ep.metadata.get(key) == value]

    def __len__(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
# SemanticMemory
# ---------------------------------------------------------------------------

class SemanticMemory:
    """
    Stores general facts as vector embeddings.
    Retrieval: cosine similarity search.
    """

    def __init__(self, embed_fn=_fake_embed) -> None:
        self._facts: list[SemanticFact] = []
        self._embed = embed_fn
        self._log = logging.getLogger(self.__class__.__name__)

    def add(self, fact: str, source: str = "unknown") -> None:
        embedding = self._embed(fact)
        sf = SemanticFact(fact=fact, embedding=embedding, source=source)
        self._facts.append(sf)
        self._log.info("Stored semantic fact: %s", fact[:60])

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        """Return (score, fact) pairs sorted by descending similarity."""
        if not self._facts:
            return []
        q_vec = self._embed(query)
        scored = [
            (_cosine_similarity(q_vec, sf.embedding), sf.fact)
            for sf in self._facts
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._facts)


# ---------------------------------------------------------------------------
# AgentMemorySystem — combines both
# ---------------------------------------------------------------------------

class AgentMemorySystem:
    """
    Combines episodic and semantic memory.
    Typical agent loop usage:
      1. on_turn_start: retrieve_context(query) → inject into prompt
      2. on_turn_end: record_turn(user_msg, agent_msg)
      3. periodically: extract_and_store_fact(fact)
    """

    def __init__(self) -> None:
        self.episodic = EpisodicMemory(max_episodes=50)
        self.semantic = SemanticMemory()
        self._log = logging.getLogger(self.__class__.__name__)

    def record_turn(self, user_msg: str, agent_msg: str, turn: int = 0) -> None:
        self.episodic.add(f"User: {user_msg}", role="user", turn=turn)
        self.episodic.add(f"Agent: {agent_msg}", role="agent", turn=turn)

    def store_fact(self, fact: str, source: str = "conversation") -> None:
        self.semantic.add(fact, source=source)

    def retrieve_context(
        self,
        query: str,
        recent_n: int = 4,
        top_k_facts: int = 2,
    ) -> dict[str, Any]:
        recent_episodes = [ep.event for ep in self.episodic.recent(recent_n)]
        relevant_facts = [fact for _, fact in self.semantic.retrieve(query, top_k_facts)]
        self._log.info(
            "Context retrieved — %d recent episodes, %d semantic facts",
            len(recent_episodes),
            len(relevant_facts),
        )
        return {
            "recent_episodes": recent_episodes,
            "relevant_facts": relevant_facts,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    memory = AgentMemorySystem()

    # Store some domain facts (semantic)
    memory.store_fact("Python lists are mutable sequences.", source="docs")
    memory.store_fact("Pydantic v2 uses model_validate instead of parse_obj.", source="docs")
    memory.store_fact("The user's preferred language is Python.", source="preference")

    # Simulate a multi-turn conversation (episodic)
    turns = [
        ("How do I parse a dict in Pydantic?", "Use model_validate(dict)."),
        ("What about lists?", "Python lists are mutable; Pydantic can validate list fields."),
        ("Can I use dataclasses instead?", "Yes, but Pydantic offers richer validation."),
    ]
    for i, (user, agent) in enumerate(turns):
        memory.record_turn(user, agent, turn=i)

    # Retrieve context for a new query
    query = "Pydantic parsing methods"
    ctx = memory.retrieve_context(query, recent_n=4, top_k_facts=2)

    print("\n=== Recent Episodes ===")
    for ep in ctx["recent_episodes"]:
        print(" ", ep)

    print("\n=== Relevant Semantic Facts ===")
    for fact in ctx["relevant_facts"]:
        print(" ", fact)

    print(f"\nEpisodic store size: {len(memory.episodic)}")
    print(f"Semantic store size: {len(memory.semantic)}")


if __name__ == "__main__":
    main()
```

Expected output (abbreviated):
```
INFO EpisodicMemory — Stored episode: User: How do I parse a dict in Pydantic?
...
=== Recent Episodes ===
  User: How do I parse a dict in Pydantic?
  Agent: Use model_validate(dict).
  ...
=== Relevant Semantic Facts ===
  Pydantic v2 uses model_validate instead of parse_obj.
  Python lists are mutable sequences.
```

## Exercises

1. Add a `by_role(role: str)` method to `EpisodicMemory` that filters episodes by `metadata["role"]`. Write a pytest test that stores 4 episodes (2 user, 2 agent) and verifies the filter returns exactly 2 each.

2. Extend `SemanticMemory` with a `delete(fact_text: str) -> bool` method that removes all facts matching the given text exactly. Return `True` if at least one was removed. Test the case where the fact does not exist.

3. Add a `summarize_old_episodes(keep_recent: int = 10) -> str` method to `AgentMemorySystem`. It should join all episodes older than the most-recent `keep_recent` into a single summary string and remove them from episodic storage, reducing the total episode count. (Use simple text joining — no LLM needed here.)

4. Implement a `SemanticMemory.update(old_fact: str, new_fact: str)` method that replaces the text and re-embeds. Test that the new fact is retrievable and the old is not.

5. Swap the fake embedder for a configurable one: `SemanticMemory` should accept `embed_fn: Callable[[str], list[float]]` in its constructor (already done above). Write a test that passes a lambda returning a fixed zero vector and verifies that all similarity scores are 0.

(Solutions: ../solutions/week-04/day-5.md)

## Self-check

- [ ] I can explain the difference between episodic and semantic memory without looking at notes
- [ ] I can articulate why time-range retrieval is natural for episodic but not semantic memory
- [ ] I can implement `EpisodicMemory.recent()` and `by_time_range()` without reference
- [ ] I can implement cosine-similarity-based retrieval in `SemanticMemory` without reference
- [ ] I can describe the combined retrieval pipeline in `AgentMemorySystem.retrieve_context()`
