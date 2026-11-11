# Elective: Long-Term Memory · Lesson 4 — Retrieval & Relevance Scoring

> Objective: Build a retrieval pipeline that ranks memories by a composite score combining recency, importance, and semantic relevance.
> Time: ~60 min · Prereqs: [[lesson-3.md]]

---

## Concept

Raw cosine similarity is a poor retrieval signal in isolation. A memory that is highly
similar but two years old may be less useful than a moderately similar memory from
yesterday. Production memory systems use a **composite retrieval score** that blends:

1. **Semantic relevance** — how similar is this memory to the current query?
2. **Recency** — how recently was this memory created or accessed?
3. **Importance** — how significant was this memory when it was stored?

### The Generative Agents formula (Park et al., 2023)

The landmark "Generative Agents" paper (Stanford + Google) proposes:

```
retrieval_score = α·relevance + β·recency + γ·importance
```

With α = β = γ = 1/3 (equal weight) as a starting point. All three components are
normalised to [0, 1] before combining.

### 1. Semantic relevance

Standard cosine similarity between query embedding and memory embedding.

```
relevance = cosine_sim(query_vector, memory_vector)
```

Normalise to [0, 1] by mapping the score through min-max or simply clipping to [0, 1]
(cosine similarity is already in [-1, 1], and negative scores are effectively 0 for
thematic memory).

### 2. Recency

Recency decays exponentially with time:

```
recency = exp(-λ · Δt)
```

Where:
- `Δt` = hours since `last_accessed`
- `λ` = decay rate (lower = slower decay)

Common values: λ = 0.01 (slow decay over days), λ = 0.1 (faster, hours matter).

The exponential function maps to [0, 1]: just-accessed → 1.0, ancient → near 0.

### 3. Importance

The importance score written at store time (Lesson 3). Already in [0, 1].

### Combining the three

```python
def retrieval_score(
    relevance: float,
    recency: float,
    importance: float,
    alpha: float = 1/3,
    beta: float = 1/3,
    gamma: float = 1/3,
) -> float:
    return alpha * relevance + beta * recency + gamma * importance
```

### Tuning weights

The equal-weight default is a reasonable starting point. In practice:

| Scenario | Preferred weight |
|---------|----------------|
| Customer support (need latest context) | Increase β (recency) |
| Knowledge assistant (need facts) | Increase α (relevance) |
| Personalisation (preferences matter most) | Increase γ (importance) |

Tune via offline evaluation (Lesson 6).

### Top-k retrieval with re-ranking

Typical retrieval flow:

1. **Candidate retrieval**: query vector store for top-N by cosine similarity (fast, approximate)
2. **Re-rank**: compute composite score for each candidate (cheap, exact)
3. **Return top-k**: final ranked list (k << N)

N = 50–200, k = 3–10 is a typical range.

---

## How it works internally

### Why recency must use `last_accessed`, not `created_at`

A memory accessed 10 times should feel "fresh" even if it was created months ago.
Updating `last_accessed` on retrieval (the `touch()` call from Lesson 1) implements
**access-based reinforcement**: frequently used memories stay recency-warm.

### Normalisation subtleties

Cosine similarity in [-1, 1] should be clipped to [0, 1] before scoring, because
negative similarity has no meaningful "recency" or "importance" analogue — we never
want a memory to subtract from the score.

```python
relevance_normalised = max(0.0, cosine_sim(q, m))
```

---

## Worked example

```python
"""
lesson_4_retrieval.py

Composite retrieval scoring: relevance + recency + importance.
No external databases. Uses in-memory store from lesson_2 patterns.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Memory record with scoring fields ────────────────────────────────────────

@dataclass
class MemoryRecord:
    id: UUID
    content: str
    vector: list[float]
    importance: float
    created_at: datetime
    last_accessed: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_accessed = datetime.now(timezone.utc)


# ── Scoring functions ─────────────────────────────────────────────────────────

def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a * norm_b > 1e-9 else 0.0


def compute_recency(record: MemoryRecord, decay_rate: float = 0.01) -> float:
    """
    Exponential decay based on hours since last access.

    decay_rate = 0.01  →  half-life ≈ 69 hours (~3 days)
    decay_rate = 0.1   →  half-life ≈ 7 hours
    """
    now = datetime.now(timezone.utc)
    delta_hours = (now - record.last_accessed).total_seconds() / 3600
    return math.exp(-decay_rate * delta_hours)


def composite_score(
    record: MemoryRecord,
    query_vector: list[float],
    decay_rate: float = 0.01,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
) -> tuple[float, dict[str, float]]:
    """
    Returns (composite_score, breakdown_dict) for transparency.
    """
    relevance = max(0.0, cosine_sim(query_vector, record.vector))
    recency = compute_recency(record, decay_rate)
    importance = record.importance

    score = alpha * relevance + beta * recency + gamma * importance
    breakdown = {
        "relevance": round(relevance, 4),
        "recency": round(recency, 4),
        "importance": round(importance, 4),
        "composite": round(score, 4),
    }
    return score, breakdown


# ── Retrieval engine ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    record: MemoryRecord
    score: float
    breakdown: dict[str, float]


class MemoryRetriever:
    """
    Two-stage retrieval: candidate fetch by relevance, then composite re-rank.
    """

    def __init__(
        self,
        records: list[MemoryRecord],
        decay_rate: float = 0.01,
        alpha: float = 1 / 3,
        beta: float = 1 / 3,
        gamma: float = 1 / 3,
    ) -> None:
        self._records = records
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 5,
        candidate_n: int = 20,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        1. Fetch top candidate_n by raw cosine similarity.
        2. Re-rank by composite score.
        3. Return top_k above min_score.
        """
        # Stage 1: candidate retrieval by relevance
        candidates = sorted(
            self._records,
            key=lambda r: cosine_sim(query_vector, r.vector),
            reverse=True,
        )[:candidate_n]

        # Stage 2: composite scoring
        scored: list[RetrievalResult] = []
        for rec in candidates:
            score, breakdown = composite_score(
                rec, query_vector, self.decay_rate,
                self.alpha, self.beta, self.gamma,
            )
            if score >= min_score:
                scored.append(RetrievalResult(record=rec, score=score, breakdown=breakdown))

        # Stage 3: sort by composite score, touch accessed records
        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:top_k]
        for r in results:
            r.record.touch()

        return results


# ── Mock embedder ─────────────────────────────────────────────────────────────

def mock_embed(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.md5(text.encode()).digest()
    raw = [float(digest[i % 16]) / 255.0 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


# ── Demo ──────────────────────────────────────────────────────────────────────

def _make_record(
    content: str,
    importance: float,
    hours_old: float,
) -> MemoryRecord:
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    created = now - timedelta(hours=hours_old)
    return MemoryRecord(
        id=uuid4(),
        content=content,
        vector=mock_embed(content),
        importance=importance,
        created_at=created,
        last_accessed=created,
    )


def main() -> None:
    records = [
        _make_record("User prefers PDF report format", importance=0.9, hours_old=1),
        _make_record("Agent sent quarterly sales summary", importance=0.4, hours_old=48),
        _make_record("User wants dark mode enabled", importance=0.7, hours_old=5),
        _make_record("User asked about PDF exports", importance=0.6, hours_old=24),
        _make_record("Meeting scheduled for next Tuesday", importance=0.5, hours_old=168),
    ]

    retriever = MemoryRetriever(records, decay_rate=0.01)
    query_vector = mock_embed("report format preference")

    results = retriever.retrieve(query_vector, top_k=3)

    log.info("Top-3 retrieved memories for 'report format preference':")
    for i, r in enumerate(results, 1):
        log.info(
            "  %d. [%.3f] %s | breakdown=%s",
            i, r.score, r.record.content[:55], r.breakdown,
        )


if __name__ == "__main__":
    main()
```

---

## Exercises

1. The `composite_score` function currently weights all three components equally.
   Add a `weights: tuple[float, float, float]` parameter (α, β, γ) that must sum to 1.0.
   Raise `ValueError` if they don't. Write a pytest parametrize test covering
   valid weights, invalid weights, and the default equal-weight case.

2. Implement a `TimeAwareRetriever` subclass that automatically **raises β** (recency
   weight) from 0.33 to 0.5 when the query vector includes a time-sensitive keyword
   (you can detect this by checking cosine similarity to a pre-embedded set of
   keywords like "latest", "recent", "today", "now").

3. Write a pytest test that creates 10 records with varying `last_accessed` times
   (0 to 240 hours old) and verifies that, all else being equal, `compute_recency`
   returns strictly decreasing values as `hours_old` increases.

4. The two-stage retrieval uses `candidate_n=20` then re-ranks. Explain (max 80 words)
   what happens to retrieval quality if `candidate_n` is set too low (e.g. 3). Give a
   concrete example where this would fail.

(Solutions: [solutions/lesson-4.md](solutions/lesson-4.md))

---

## Self-check

- [ ] I can explain the three components of composite retrieval scoring
- [ ] I can implement exponential recency decay with a configurable decay rate
- [ ] I understand why two-stage retrieval (candidate fetch → re-rank) is used
- [ ] I can tune retrieval weights for different agent use cases
