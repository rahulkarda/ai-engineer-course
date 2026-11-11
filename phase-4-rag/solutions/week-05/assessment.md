# Solutions — Week 05 Assessment

## Conceptual answers

**1. Reducing cross-encoder latency**

- **Quantisation**: use an INT8 or FP16 quantised version of the model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` has a ONNX-quantised variant). Reduces inference time by 2–4×.
- **Reduce K**: if the fast retriever is high-quality, reduce from 50 to 10 candidates. Linear reduction in inference time.
- **Smaller model**: use `MiniLM-L-6` instead of `electra-base` (same family, 80% faster, ~2% NDCG drop).

**2. MMR formula**

`score_i = λ · sim(d_i, query) − (1−λ) · max_{d_j ∈ S} sim(d_i, d_j)`

- λ=0: `score = −max_redundancy` → algorithm picks the document furthest from all selected docs → maximum diversity.
- λ=1: `score = relevance` → redundancy term vanishes → equivalent to standard relevance ranking.

**3. HyDE hurts factual questions**

HyDE is effective when query vocabulary differs from document vocabulary ("how does X work?" vs "X operates by…"). For factual lookups, the query is already close in embedding space to the answer, so HyDE adds LLM hallucination noise without closing a semantic gap. Solution: detect question type (keyword match or classifier); apply HyDE only to open-ended questions.

**4. Parent doc > context window**

- Hierarchical parents: store document as medium chunks (1 000–2 000 tokens); use those as parents instead of full text.
- Map-reduce: send multiple chunks in parallel and aggregate answers before final synthesis.

**5. "Is 0.65 NDCG@5 good?"**

Need: baseline NDCG (random, BM25, previous system); how NDCG@5 was measured (eval set size, relevance annotation method); whether eval set represents real query distribution.

**6. Near-duplicate vs exact-duplicate**

Near-duplicate uses embedding similarity (e.g., cosine > 0.97). Exact uses content hash (MD5/SHA). Real-world scenario: overlapping sliding-window chunks share 50% text — different hashes, very high cosine similarity.

**7. P@5 up, R@10 down: is this improvement?**

For LLM RAG (fixed top-5 context): yes, precision improvement is the primary goal. For recall-critical search (compliance, medical): no, the recall drop is a regression. State the trade-off; don't declare universal improvement.

---

## Coding challenge: RetrievalEvaluator

```python
"""
assessment_solution.py
"""

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RetrievalEvaluator:
    cases: list[tuple[str, set[str]]]  # (query, relevant_ids)

    def evaluate(
        self,
        retrieve_fn: Callable[[str], list[str]],
        k: int = 5,
    ) -> dict[str, float]:
        p_scores, r_scores, rr_scores, ndcg_scores = [], [], [], []
        for query, relevant_ids in self.cases:
            retrieved = retrieve_fn(query)
            p_scores.append(self._precision_at_k(retrieved, relevant_ids, k))
            r_scores.append(self._recall_at_k(retrieved, relevant_ids, k))
            rr_scores.append(self._reciprocal_rank(retrieved, relevant_ids))
            ndcg_scores.append(self._ndcg_at_k(retrieved, relevant_ids, k))

        result = {
            "precision": float(np.mean(p_scores)),
            "recall": float(np.mean(r_scores)),
            "mrr": float(np.mean(rr_scores)),
            "ndcg": float(np.mean(ndcg_scores)),
        }
        log.info("Eval@%d: %s", k, result)
        return result

    def compare(
        self,
        baseline_fn: Callable[[str], list[str]],
        improved_fn: Callable[[str], list[str]],
        k: int = 5,
    ) -> dict[str, dict[str, float]]:
        baseline = self.evaluate(baseline_fn, k=k)
        improved = self.evaluate(improved_fn, k=k)
        delta = {metric: improved[metric] - baseline[metric] for metric in baseline}
        return {"baseline": baseline, "improved": improved, "delta": delta}

    # ------------------------------------------------------------------
    # Private metric helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if k == 0:
            return 0.0
        return sum(1 for d in retrieved[:k] if d in relevant) / k

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 1.0
        return sum(1 for d in retrieved[:k] if d in relevant) / len(relevant)

    @staticmethod
    def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        # Binary graded: relevant=1, non-relevant=0
        grades = {doc_id: 1 for doc_id in relevant}
        dcg = sum(
            (2 ** grades.get(doc_id, 0) - 1) / math.log2(i + 2)
            for i, doc_id in enumerate(retrieved[:k])
        )
        # Ideal: all relevant docs first
        ideal = sorted(grades.values(), reverse=True)
        idcg = sum(
            (2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k])
        )
        return dcg / idcg if idcg > 0 else 0.0
```

### Test file

```python
"""tests/test_assessment.py"""

import pytest
from assessment_solution import RetrievalEvaluator


@pytest.fixture()
def evaluator():
    return RetrievalEvaluator(cases=[
        ("q1", {"d1", "d2"}),
        ("q2", {"d3"}),
    ])


def test_perfect_retriever_precision_1(evaluator):
    retrieve = lambda q: {"q1": ["d1", "d2", "d4"], "q2": ["d3", "d1"]}[q]
    result = evaluator.evaluate(retrieve, k=2)
    assert result["precision"] == pytest.approx(1.0)


def test_perfect_retriever_mrr_1(evaluator):
    retrieve = lambda q: {"q1": ["d1", "d2"], "q2": ["d3"]}[q]
    result = evaluator.evaluate(retrieve, k=2)
    assert result["mrr"] == pytest.approx(1.0)


def test_empty_retrieved_zero_metrics(evaluator):
    result = evaluator.evaluate(lambda q: [], k=5)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["mrr"] == 0.0
    assert result["ndcg"] == 0.0


def test_compare_returns_delta(evaluator):
    baseline = lambda q: ["d4", "d5"]   # nothing relevant
    improved = lambda q: {"q1": ["d1"], "q2": ["d3"]}[q]
    out = evaluator.compare(baseline, improved, k=1)
    assert out["delta"]["precision"] > 0
    assert out["delta"]["mrr"] > 0


def test_k_larger_than_retrieved_no_error(evaluator):
    retrieve = lambda q: ["d1"]
    result = evaluator.evaluate(retrieve, k=10)
    assert isinstance(result["precision"], float)


def test_no_relevant_docs_recall_is_1():
    ev = RetrievalEvaluator(cases=[("q", set())])
    result = ev.evaluate(lambda q: ["a"], k=5)
    assert result["recall"] == pytest.approx(1.0)
```

## Rubric mapping

| Dimension | Full-marks criteria |
|---|---|
| Independent implementation (40 pts) | `evaluate` computes all 4 metrics; `compare` returns correct delta; tests verify hand-calculated values |
| Conceptual understanding (25 pts) | All 7 questions answered: latency options, MMR λ extremes, HyDE limitation, parent overflow, metric interpretation, dedup distinction, P vs R trade-off |
| Debugging (20 pts) | Edge cases: empty retrieved list (zero metrics), no relevant docs (recall=1), k > list length (no error) |
| Code quality (15 pts) | Type hints, logging, ruff-clean, ≥6 meaningful tests |
