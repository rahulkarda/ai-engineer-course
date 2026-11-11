# Elective: Long-Term Memory · Lesson 6 — Evaluating Memory Systems

> Objective: Define and implement metrics for memory quality, and build a test harness that validates a memory service end-to-end.
> Time: ~60 min · Prereqs: [[lesson-5.md]]

---

## Concept

### Why memory evaluation is hard

Memory systems have no single "correct" answer. Evaluation must measure:

1. **Retrieval quality** — do the right memories surface for a given query?
2. **Write quality** — is what gets stored accurate, non-redundant, and well-scored?
3. **Temporal behaviour** — do decay and forgetting work correctly over time?
4. **End-to-end agent quality** — does the agent's responses improve when memory is active?

### Retrieval metrics

Adapted from information retrieval (IR):

#### Precision@k

Of the top-k retrieved memories, what fraction are truly relevant?

```
Precision@k = |{relevant} ∩ {retrieved top-k}| / k
```

High precision → low noise in what the agent sees.

#### Recall@k

Of all truly relevant memories, what fraction appear in the top-k?

```
Recall@k = |{relevant} ∩ {retrieved top-k}| / |{relevant}|
```

High recall → the agent doesn't miss important context.

#### Mean Reciprocal Rank (MRR)

How highly ranked is the first truly relevant result?

```
MRR = (1/|Q|) Σ (1 / rank of first relevant result)
```

MRR is useful when only the first result matters (e.g., top memory injected into context).

#### Mean Average Precision (MAP)

Average of precision scores at each rank position where a relevant memory appears.
More holistic than MRR.

### Write quality metrics

| Metric | How to measure |
|--------|---------------|
| **Fact extraction accuracy** | Human-labelled test set: does stored content match the ground truth fact? |
| **Dedup rate** | Fraction of duplicate inputs that were correctly identified and skipped |
| **Importance calibration** | Correlation between assigned importance and human-rated importance |
| **Staleness rate** | Fraction of active memories that are outdated (require manual check) |

### Latency metrics

Memory adds latency to every agent turn:

| Operation | Target |
|-----------|--------|
| Write (async) | < 500 ms (background) |
| Retrieval (top-k) | < 100 ms |
| Full consolidation pass | < 5 s (scheduled, not per-turn) |

### End-to-end agent quality

The gold standard: A/B test with and without memory enabled, evaluated by:

- Response relevance (LLM judge or human)
- Personalisation accuracy (does the agent use stored preferences?)
- Context continuity (does the agent recall facts from prior sessions?)

---

## How it works internally

### Building a labelled test dataset

For offline evaluation:

1. Collect or create a set of **queries** (what the agent might search for)
2. For each query, annotate **relevant memory IDs** (ground truth)
3. Run the retriever and compare to ground truth

This is identical to IR evaluation (e.g., MS-MARCO, BEIR), just at smaller scale.

### LLM-as-judge for end-to-end quality

When human annotation is expensive, use a stronger LLM (claude-opus-5-5) to evaluate
whether a response correctly used relevant memories:

```python
prompt = f"""
Does the agent response below demonstrate awareness of the relevant memory?

Memory: {memory_content}
Agent response: {agent_response}

Answer YES or NO, then explain in one sentence.
"""
```

This is sometimes called **G-Eval** (GPT-based evaluation). Use it sparingly — it adds
cost and has its own biases.

---

## Worked example

```python
"""
lesson_6_evaluation.py

A memory evaluation harness computing Precision@k, Recall@k, and MRR.
Memory system is mocked — evaluation logic is what matters.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Types ─────────────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    id: UUID
    content: str
    metadata: dict[str, Any]


@dataclass
class QueryCase:
    """A single evaluation query with ground-truth relevant memory IDs."""
    query: str
    relevant_ids: set[UUID]


@dataclass
class EvalResult:
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    retrieved_ids: list[UUID]


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[UUID], relevant: set[UUID], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for uid in top_k if uid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[UUID], relevant: set[UUID], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[UUID], relevant: set[UUID]) -> float:
    for i, uid in enumerate(retrieved, 1):
        if uid in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(results: list[EvalResult]) -> float:
    if not results:
        return 0.0
    return sum(r.reciprocal_rank for r in results) / len(results)


def mean_average_precision(
    retrieved_lists: list[list[UUID]],
    relevant_sets: list[set[UUID]],
    k: int,
) -> float:
    """MAP@k over multiple queries."""
    aps: list[float] = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if not relevant:
            continue
        precision_sum = 0.0
        hits = 0
        for rank, uid in enumerate(retrieved[:k], 1):
            if uid in relevant:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / len(relevant))
    return sum(aps) / len(aps) if aps else 0.0


# ── Retriever interface ───────────────────────────────────────────────────────

class MockRetriever:
    """
    A deterministic mock retriever for evaluation testing.
    Returns records in a fixed order controlled by test fixtures.
    """

    def __init__(self, ordered_results: list[list[MemoryRecord]]) -> None:
        """ordered_results[i] is the ranked list returned for query i."""
        self._results = ordered_results
        self._call_count = 0

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]:  # noqa: ARG002
        result = self._results[self._call_count % len(self._results)]
        self._call_count += 1
        return result[:top_k]


# ── Evaluation harness ────────────────────────────────────────────────────────

class MemoryEvaluator:
    """
    Runs a set of query cases through a retriever and computes IR metrics.
    """

    def __init__(self, retriever: MockRetriever, k: int = 5) -> None:
        self._retriever = retriever
        self.k = k

    def evaluate(self, cases: list[QueryCase]) -> list[EvalResult]:
        results: list[EvalResult] = []
        for case in cases:
            retrieved_records = self._retriever.retrieve(case.query, top_k=self.k)
            retrieved_ids = [r.id for r in retrieved_records]

            p = precision_at_k(retrieved_ids, case.relevant_ids, self.k)
            r = recall_at_k(retrieved_ids, case.relevant_ids, self.k)
            rr = reciprocal_rank(retrieved_ids, case.relevant_ids)

            results.append(EvalResult(
                query=case.query,
                precision_at_k=p,
                recall_at_k=r,
                reciprocal_rank=rr,
                retrieved_ids=retrieved_ids,
            ))
            log.info(
                "Query='%s' P@%d=%.2f R@%d=%.2f RR=%.2f",
                case.query, self.k, p, self.k, r, rr,
            )

        return results

    def summarise(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        n = len(results)
        return {
            f"mean_precision@{self.k}": sum(r.precision_at_k for r in results) / n,
            f"mean_recall@{self.k}": sum(r.recall_at_k for r in results) / n,
            "mrr": mean_reciprocal_rank(results),
            "map@k": mean_average_precision(
                [r.retrieved_ids for r in results],
                [c.relevant_ids for c in []],  # placeholder — see exercise 2
                self.k,
            ),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Build some memory records
    r1 = MemoryRecord(id=uuid4(), content="User prefers PDF reports", metadata={})
    r2 = MemoryRecord(id=uuid4(), content="User uses dark mode", metadata={})
    r3 = MemoryRecord(id=uuid4(), content="User asked about Q3 results", metadata={})
    r4 = MemoryRecord(id=uuid4(), content="Agent sent PDF summary", metadata={})
    r5 = MemoryRecord(id=uuid4(), content="Meeting scheduled for Tuesday", metadata={})

    # Define queries with ground-truth relevant IDs
    cases = [
        QueryCase(
            query="report format preference",
            relevant_ids={r1.id, r4.id},
        ),
        QueryCase(
            query="display settings",
            relevant_ids={r2.id},
        ),
    ]

    # Mock retriever returns a specific ordered list per query
    # Query 1: r1 is rank-1 (good), r4 is rank-3, r3/r2/r5 are irrelevant
    # Query 2: r2 is rank-2 (missed rank-1 opportunity)
    retriever = MockRetriever([
        [r1, r3, r4, r5, r2],   # for query 1
        [r3, r2, r5, r1, r4],   # for query 2
    ])

    evaluator = MemoryEvaluator(retriever, k=3)
    results = evaluator.evaluate(cases)

    log.info("=== Summary ===")
    for k, v in evaluator.summarise(results).items():
        log.info("  %s = %.4f", k, v)


if __name__ == "__main__":
    main()
```

---

## Exercises

1. Fix `MemoryEvaluator.summarise` to correctly pass the relevant sets for `map@k`.
   The `relevant_sets` list should come from the `cases` passed to `evaluate` — store
   them as an instance variable during the `evaluate` call and reuse them in `summarise`.

2. Write a pytest test that creates 3 query cases, a deterministic `MockRetriever`,
   and verifies that `precision_at_k`, `recall_at_k`, and `mrr` all return expected
   float values (test at least one query with a miss and one with a perfect hit).

3. Implement a `LatencyBenchmark` class that runs `retriever.retrieve(...)` N times,
   records wall-clock time per call using `time.perf_counter`, and returns
   `{mean_ms, p50_ms, p95_ms, p99_ms}`. Write a test that verifies the benchmark
   runs without error on a mock retriever.

4. Describe (max 100 words) how you would design an LLM-as-judge evaluation for
   end-to-end memory quality. What prompt would you use? What scoring rubric? What
   biases should you guard against?

(Solutions: [solutions/lesson-6.md](solutions/lesson-6.md))

---

## Self-check

- [ ] I can compute Precision@k, Recall@k, and MRR from scratch
- [ ] I understand what MAP@k measures and when to prefer it over MRR
- [ ] I can build an evaluation harness with a mock retriever
- [ ] I can describe how to use an LLM as a judge for end-to-end memory quality
