# Solutions: Lesson 6 — Evaluating Memory Systems

---

## Exercise 1 — Fix `map@k` in `summarise`

Store the cases used during `evaluate` as an instance attribute, then use them in `summarise`:

```python
class MemoryEvaluator:
    def __init__(self, retriever: MockRetriever, k: int = 5) -> None:
        self._retriever = retriever
        self.k = k
        self._last_cases: list[QueryCase] = []

    def evaluate(self, cases: list[QueryCase]) -> list[EvalResult]:
        self._last_cases = cases   # store for summarise()
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
        return results

    def summarise(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        n = len(results)
        map_k = mean_average_precision(
            [r.retrieved_ids for r in results],
            [c.relevant_ids for c in self._last_cases],
            self.k,
        )
        return {
            f"mean_precision@{self.k}": sum(r.precision_at_k for r in results) / n,
            f"mean_recall@{self.k}": sum(r.recall_at_k for r in results) / n,
            "mrr": mean_reciprocal_rank(results),
            f"map@{self.k}": map_k,
        }
```

---

## Exercise 2 — pytest test for precision, recall, MRR

```python
import pytest
from uuid import uuid4
from lesson_6_evaluation import (
    MemoryRecord, QueryCase, MockRetriever, MemoryEvaluator,
    precision_at_k, recall_at_k, reciprocal_rank,
)


def test_precision_recall_mrr():
    r1 = MemoryRecord(id=uuid4(), content="PDF reports", metadata={})
    r2 = MemoryRecord(id=uuid4(), content="dark mode", metadata={})
    r3 = MemoryRecord(id=uuid4(), content="meetings", metadata={})

    # Query 1: r1 is the only relevant, and it is rank-1 → perfect
    # Query 2: r2 is relevant but ranked 2nd → partial
    # Query 3: nothing relevant retrieved → all zeros
    cases = [
        QueryCase("pdf query", relevant_ids={r1.id}),
        QueryCase("dark mode query", relevant_ids={r2.id}),
        QueryCase("unrelated", relevant_ids={uuid4()}),  # relevant not in results
    ]

    retriever = MockRetriever([
        [r1, r3, r2],    # query 1: r1 at rank 1
        [r3, r2, r1],    # query 2: r2 at rank 2
        [r1, r2, r3],    # query 3: relevant UUID not present
    ])

    evaluator = MemoryEvaluator(retriever, k=3)
    results = evaluator.evaluate(cases)

    # Query 1: P@3 = 1/3, R@3 = 1/1, RR = 1.0
    assert results[0].precision_at_k == pytest.approx(1 / 3, abs=1e-6)
    assert results[0].recall_at_k == pytest.approx(1.0)
    assert results[0].reciprocal_rank == pytest.approx(1.0)

    # Query 2: P@3 = 1/3, R@3 = 1/1, RR = 1/2
    assert results[1].precision_at_k == pytest.approx(1 / 3, abs=1e-6)
    assert results[1].recall_at_k == pytest.approx(1.0)
    assert results[1].reciprocal_rank == pytest.approx(0.5)

    # Query 3: all zeros
    assert results[2].precision_at_k == 0.0
    assert results[2].recall_at_k == 0.0
    assert results[2].reciprocal_rank == 0.0
```

---

## Exercise 3 — `LatencyBenchmark`

```python
import time
import statistics
from dataclasses import dataclass


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class LatencyBenchmark:
    def __init__(self, retriever: MockRetriever) -> None:
        self._retriever = retriever

    def run(self, query: str, n: int = 100, top_k: int = 5) -> LatencyStats:
        timings: list[float] = []
        for _ in range(n):
            start = time.perf_counter()
            self._retriever.retrieve(query, top_k=top_k)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            timings.append(elapsed)
        timings.sort()
        return LatencyStats(
            mean_ms=statistics.mean(timings),
            p50_ms=timings[int(0.50 * n)],
            p95_ms=timings[int(0.95 * n)],
            p99_ms=timings[int(0.99 * n)],
        )
```

Test:

```python
def test_latency_benchmark_runs():
    from uuid import uuid4
    records = [MemoryRecord(id=uuid4(), content=f"record {i}", metadata={}) for i in range(5)]
    retriever = MockRetriever([records])
    bench = LatencyBenchmark(retriever)
    stats = bench.run("test query", n=50)
    assert stats.mean_ms >= 0
    assert stats.p50_ms >= 0
    assert stats.p95_ms >= stats.p50_ms
    assert stats.p99_ms >= stats.p95_ms
```

---

## Exercise 4 — LLM-as-judge design

**Prompt** (judge model: `claude-opus-5-5`):

```
System: You are an evaluator. Rate whether an agent response demonstrates 
knowledge of a specific memory on a scale of 0–2:
  0 = ignores the memory
  1 = partially reflects it
  2 = clearly and correctly uses it

Respond with a single digit (0, 1, or 2) followed by one sentence of reasoning.
```

**Rubric**: 0 = no memory use, 1 = partial, 2 = full. Average score over N samples.

**Biases to guard against**:
- **Verbosity bias**: longer responses may score higher even without using the memory.
  Mitigate: strip response to first 200 tokens before judging.
- **Self-consistency bias**: if judge and evaluated agent use the same model family,
  the judge may be lenient. Mitigate: use a different judge model or human spot-checks.
- **Prompt anchoring**: showing the memory first anchors the judge to expect it.
  Mitigate: randomise whether memory is shown before or after the response, run blind.
