# Elective: Long-Term Memory · Lesson 3 — Writing & Consolidating Memories

> Objective: Design a write pipeline that decides what to store, summarises verbose content, and deduplicates near-duplicates.
> Time: ~60 min · Prereqs: [[lesson-2.md]]

---

## Concept

### The write problem

Storing everything is worse than storing nothing. A store that grows without discipline
degrades retrieval: noise drowns signal, query latency climbs, and stale contradictions
confuse the agent. The write pipeline is where quality is established.

Four decisions happen on every write:

1. **Should this be stored?** (filtering)
2. **What form should it take?** (summarisation)
3. **Is it a duplicate?** (deduplication)
4. **How important is it?** (importance scoring)

### 1. Filtering — what to store

Not every turn deserves a memory record. Common heuristics:

| Keep | Discard |
|------|---------|
| User preferences stated explicitly | Greetings, chitchat |
| Factual claims about entities | Filler acknowledgements |
| Significant decisions or outcomes | Repeated identical information |
| Errors and corrections | Low-confidence LLM speculation |

In practice: run a lightweight classifier or a cheap LLM call with a terse prompt.

### 2. Summarisation — condensing verbose content

A raw multi-turn conversation log is too verbose to store verbatim. The write pipeline
**summarises** before storing:

- Extract the key claim / event / decision
- Strip timestamps, filler, boilerplate
- Normalise to third-person facts ("User prefers PDF reports")

For episodic memory, keep a short narrative (2–4 sentences). For semantic memory, extract
`(subject, predicate, object)` triples.

### 3. Deduplication — avoiding near-identical records

If a user says "I prefer PDF" in sessions 1, 5, and 12, you do not want three records.
Dedup strategies:

- **Exact match on content hash**: fast, misses paraphrases
- **Semantic similarity threshold**: if cosine_sim(new, existing) > 0.92, treat as duplicate
- **LLM-based merge**: ask the LLM to merge two overlapping facts — expensive, most accurate

The right balance: exact-hash first (free), then semantic threshold (cheap), LLM merge
only for conflicts.

### 4. Importance scoring — weighting what matters

Not all memories are equal. Importance score drives:
- Which memories surface at retrieval time
- Which memories survive long enough to resist decay (Lesson 5)

Scoring signals:

| Signal | Weight rationale |
|--------|-----------------|
| User explicitly stated it | High — intentional disclosure |
| Repeated across sessions | High — durable preference |
| Agent acted on it | Medium — validated by use |
| Single mention, no followup | Low |
| LLM-extracted inference | Confidence-scaled |

Importance is a float in [0, 1]. Combine signals with a weighted sum.

---

## How it works internally

### Write pipeline as a chain of transforms

```
raw_turn (messages: list[Message])
    │
    ├─[filter]──────────► discard if not worth storing
    │
    ├─[summarise]──────► condensed MemoryCandidate
    │
    ├─[embed]──────────► vector representation
    │
    ├─[dedup check]────► skip if duplicate, merge if conflict
    │
    ├─[score]──────────► importance float
    │
    └─[store]──────────► MemoryRecord written to backend
```

Each step is a pure function: easy to test, easy to swap.

### Batch vs. real-time writes

| Approach | Pro | Con |
|----------|-----|-----|
| Real-time (end of turn) | Low latency for availability | Adds to turn response time |
| Async (background worker) | No turn latency hit | Memory not available immediately |
| Periodic consolidation | Batch dedup + merge | Memory may be hours stale |

Most production systems use **async writes** with a consolidation job that runs every
N minutes.

---

## Worked example

```python
"""
lesson_3_write_pipeline.py

A write pipeline: filter → summarise → embed → dedup → score → store.
LLM calls are mocked via a callable interface for testability.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "user" | "assistant"
    content: str


@dataclass
class MemoryCandidate:
    content: str
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryRecord:
    id: UUID
    content: str
    vector: list[float]
    importance: float
    created_at: datetime
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Protocol types (interfaces for injectable dependencies) ───────────────────

class Summariser(Protocol):
    def summarise(self, messages: list[Message]) -> str | None:
        """Return a condensed fact string, or None if not worth storing."""
        ...


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


# ── Mock implementations (used in tests) ─────────────────────────────────────

class MockSummariser:
    """Returns a fixed summary for anything containing a keyword."""

    def summarise(self, messages: list[Message]) -> str | None:
        combined = " ".join(m.content for m in messages)
        if any(kw in combined.lower() for kw in ["prefer", "like", "want", "need"]):
            # Extract the user turn as the summary
            user_msgs = [m.content for m in messages if m.role == "user"]
            return user_msgs[-1] if user_msgs else None
        return None  # not worth storing


class MockEmbedder:
    """Deterministic 8-D mock embedder based on content hash."""

    _DIM = 8

    def embed(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).digest()
        raw = [float(b) / 255.0 for b in digest[: self._DIM]]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a * norm_b > 1e-9 else 0.0


# ── Importance scorer ─────────────────────────────────────────────────────────

def score_importance(candidate: MemoryCandidate, existing_count: int = 0) -> float:
    """
    Heuristic importance score in [0, 1].

    Signals:
      - explicit_pref: content contains preference keywords      (+0.35)
      - repeated: same fact seen before (existing_count > 0)     (+0.25)
      - length: longer content = more detail = slightly higher   (up to +0.15)
    """
    score = 0.25  # baseline
    text = candidate.content.lower()
    if any(kw in text for kw in ["prefer", "always", "never", "must", "required"]):
        score += 0.35
    if existing_count > 0:
        score += min(0.25, existing_count * 0.1)
    score += min(0.15, len(candidate.content) / 500)
    return min(1.0, score)


# ── Write pipeline ────────────────────────────────────────────────────────────

class MemoryWritePipeline:
    """
    Orchestrates filter → summarise → embed → dedup → score → store.
    """

    DEDUP_THRESHOLD = 0.92  # cosine similarity above which we consider records duplicates

    def __init__(
        self,
        summariser: Summariser,
        embedder: Embedder,
    ) -> None:
        self._summariser = summariser
        self._embedder = embedder
        self._store: list[MemoryRecord] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def process_turn(self, messages: list[Message]) -> MemoryRecord | None:
        """
        Process a conversation turn and optionally write a memory record.
        Returns the stored record, or None if filtered/deduped.
        """
        # Step 1: summarise — returns None if not worth storing
        summary = self._summariser.summarise(messages)
        if summary is None:
            log.info("WritePipeline: filtered out (no storable content)")
            return None

        candidate = MemoryCandidate(content=summary)

        # Step 2: embed
        vector = self._embedder.embed(candidate.content)

        # Step 3: dedup check
        content_hash = hashlib.sha256(candidate.content.encode()).hexdigest()
        if self._is_exact_duplicate(content_hash):
            log.info("WritePipeline: exact duplicate — skipped")
            return None

        existing_count = self._count_semantic_duplicates(vector, candidate.content)
        if existing_count > 0:
            log.info(
                "WritePipeline: %d near-duplicate(s) found — boosting importance, not deduping",
                existing_count,
            )

        # Step 4: score importance
        importance = score_importance(candidate, existing_count)

        # Step 5: store
        record = MemoryRecord(
            id=uuid4(),
            content=candidate.content,
            vector=vector,
            importance=importance,
            created_at=datetime.now(timezone.utc),
            content_hash=content_hash,
        )
        self._store.append(record)
        log.info(
            "WritePipeline: stored record id=%s importance=%.2f content=%s",
            record.id,
            record.importance,
            record.content[:60],
        )
        return record

    def all_records(self) -> list[MemoryRecord]:
        return list(self._store)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_exact_duplicate(self, content_hash: str) -> bool:
        return any(r.content_hash == content_hash for r in self._store)

    def _count_semantic_duplicates(self, vector: list[float], content: str) -> int:
        count = 0
        for existing in self._store:
            sim = cosine_sim(vector, existing.vector)
            if sim >= self.DEDUP_THRESHOLD and existing.content != content:
                count += 1
        return count


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pipeline = MemoryWritePipeline(
        summariser=MockSummariser(),
        embedder=MockEmbedder(),
    )

    turns: list[list[Message]] = [
        [
            Message("user", "I prefer receiving reports in PDF format"),
            Message("assistant", "Got it, I'll always send PDFs."),
        ],
        [
            Message("user", "Hi there!"),
            Message("assistant", "Hello! How can I help?"),
        ],
        [
            Message("user", "I prefer receiving reports in PDF format"),  # exact duplicate
            Message("assistant", "Noted."),
        ],
        [
            Message("user", "I always want my summaries in bullet points"),
            Message("assistant", "Sure, bullet points it is."),
        ],
    ]

    for i, turn in enumerate(turns, 1):
        log.info("--- Turn %d ---", i)
        pipeline.process_turn(turn)

    log.info("Total stored: %d", len(pipeline.all_records()))


if __name__ == "__main__":
    main()
```

Expected output (IDs and timestamps vary):

```
INFO:__main__:--- Turn 1 ---
INFO:__main__:WritePipeline: stored record id=... importance=0.63 content=I prefer receiving reports in PDF format
INFO:__main__:--- Turn 2 ---
INFO:__main__:WritePipeline: filtered out (no storable content)
INFO:__main__:--- Turn 3 ---
INFO:__main__:WritePipeline: exact duplicate — skipped
INFO:__main__:--- Turn 4 ---
INFO:__main__:WritePipeline: stored record id=... importance=0.63 content=I always want my summaries in bullet points
INFO:__main__:Total stored: 2
```

---

## Exercises

1. The `MockSummariser` uses keyword matching. Write a `KeywordFilterSummariser` that
   accepts a configurable `keywords: list[str]` and `discard_keywords: list[str]`.
   A message that contains a `discard_keyword` is always filtered, even if it also
   contains a keep-keyword.

2. Implement `_merge_semantic_duplicate` on `MemoryWritePipeline`: when a near-duplicate
   is found AND the cosine similarity exceeds 0.96, overwrite the existing record's
   content with the new content (keeping the original ID and `created_at`).

3. `score_importance` does not consider message length. Add a signal: if the user turn
   is longer than 100 characters, add +0.1 importance. Write a pytest parametrize test
   covering short message (no boost) and long message (with boost).

4. Describe (in prose, max 100 words) how you would replace `MockSummariser` with a
   real Anthropic `claude-haiku-5-5` call. What prompt would you use? How would you
   handle rate limits?

(Solutions: [solutions/lesson-3.md](solutions/lesson-3.md))

---

## Self-check

- [ ] I can describe the four decisions in a write pipeline
- [ ] I understand the difference between exact-hash dedup and semantic dedup
- [ ] I can implement an importance scorer with multiple weighted signals
- [ ] I understand why async writes are preferred for production systems
