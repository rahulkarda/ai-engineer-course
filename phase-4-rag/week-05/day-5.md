# Phase 4 · Week 05 · Day 5 — Deduplication & Measuring Retrieval Improvement

> Objective: Implement robust deduplication strategies and quantify retrieval quality with precision@k, recall@k, MRR, and NDCG.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Deduplication beyond content hashes

Day 4 introduced content-hash deduplication (exact matches). Real pipelines also need:

- **Near-duplicate removal**: chunks from the same parent or overlapping windows produce near-identical text. Detect with cosine similarity threshold (e.g., >0.97 → same chunk).
- **Source deduplication**: if you already have the parent, drop its children from the result set.
- **Cross-query deduplication**: after query expansion, different queries may surface the same chunk with different scores — keep the highest score, not the first occurrence.

### Measuring retrieval quality

You cannot improve what you do not measure. Four standard metrics for information retrieval:

| Metric | What it measures | Formula |
|---|---|---|
| Precision@k | Fraction of top-k that are relevant | `|relevant ∩ top-k| / k` |
| Recall@k | Fraction of all relevant docs in top-k | `|relevant ∩ top-k| / |relevant|` |
| MRR | Where the first relevant doc appears | `1 / rank_of_first_relevant` |
| NDCG@k | Quality of ranking with graded relevance | discounted cumulative gain |

For a **before/after comparison** of your reranking pipeline:
1. Curate a small evaluation set: 20–50 (query, relevant_doc_ids) pairs.
2. Run the baseline retriever (BM25 + bi-encoder, no reranker).
3. Run the full pipeline (hybrid + cross-encoder rerank + MMR).
4. Compute metrics for both. Report the delta.

Even 20 eval pairs give meaningful signal on a focused corpus.

## How it works internally

### NDCG@k in detail

Graded relevance scores: 0 = not relevant, 1 = partially relevant, 2 = highly relevant.

```
DCG@k  = sum_i (2^rel_i - 1) / log2(i + 2)   for i in 0..k-1
IDCG@k = DCG of the ideal (sorted) ranking
NDCG@k = DCG@k / IDCG@k
```

For binary relevance (0/1), NDCG reduces to a weighted form of precision. For graded relevance, it rewards placing the most-relevant documents first.

### Cosine near-duplicate detection

Two chunks are near-duplicates if their embedding cosine similarity > threshold τ. When building the result set greedily:

1. Add the first candidate.
2. For each subsequent candidate, compute its max cosine similarity to already-added candidates.
3. If max similarity > τ, skip (it's a near-duplicate).

This is the same structure as MMR with λ=0 — but we use a hard threshold rather than a soft penalty.

## Worked example

```python
"""
day5_metrics.py

Retrieval quality metrics: Precision@k, Recall@k, MRR, NDCG@k.
All pure Python + numpy — no external IR library required.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class RetrievalEvalCase:
    """A single query with its relevant document IDs."""
    query: str
    relevant_ids: set[str]        # binary relevance
    graded_ids: dict[str, int]    # doc_id -> relevance grade (0/1/2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of first relevant document; 0 if none found."""
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved_ids: list[str], graded_ids: dict[str, int], k: int) -> float:
    """Discounted Cumulative Gain at k with graded relevance."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = graded_ids.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: list[str], graded_ids: dict[str, int], k: int) -> float:
    """Normalised DCG at k."""
    ideal_sorted = sorted(graded_ids.values(), reverse=True)
    ideal_ids = [f"ideal_{i}" for i in range(len(ideal_sorted))]
    ideal_grades = {doc_id: grade for doc_id, grade in zip(ideal_ids, ideal_sorted)}
    idcg = dcg_at_k(ideal_ids, ideal_grades, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved_ids, graded_ids, k) / idcg


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    precision: float
    recall: float
    mrr: float
    ndcg: float


def evaluate_retrieval(
    cases: list[RetrievalEvalCase],
    retrieve_fn: "Callable[[str], list[str]]",
    k: int = 5,
) -> EvalResult:
    """
    Run retrieve_fn on each eval case and compute macro-averaged metrics.

    Parameters
    ----------
    cases:
        Eval set with relevant_ids and graded_ids per query.
    retrieve_fn:
        Function that takes a query string and returns a list of doc_ids
        in ranked order.
    k:
        Cutoff for precision, recall, NDCG.
    """
    from typing import Callable
    p_scores, r_scores, rr_scores, ndcg_scores = [], [], [], []

    for case in cases:
        retrieved = retrieve_fn(case.query)
        p_scores.append(precision_at_k(retrieved, case.relevant_ids, k))
        r_scores.append(recall_at_k(retrieved, case.relevant_ids, k))
        rr_scores.append(reciprocal_rank(retrieved, case.relevant_ids))
        ndcg_scores.append(ndcg_at_k(retrieved, case.graded_ids, k))

    result = EvalResult(
        precision=float(np.mean(p_scores)),
        recall=float(np.mean(r_scores)),
        mrr=float(np.mean(rr_scores)),
        ndcg=float(np.mean(ndcg_scores)),
    )
    log.info(
        "Eval@%d — P=%.3f R=%.3f MRR=%.3f NDCG=%.3f",
        k,
        result.precision,
        result.recall,
        result.mrr,
        result.ndcg,
    )
    return result


# ---------------------------------------------------------------------------
# Near-duplicate deduplication
# ---------------------------------------------------------------------------
def deduplicate_near_duplicates(
    doc_ids: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.97,
) -> list[str]:
    """
    Remove near-duplicate documents by cosine similarity.

    Parameters
    ----------
    doc_ids: ordered list of document IDs.
    embeddings: unit-normalised embeddings, shape (N, D).
    threshold: cosine similarity above which two docs are considered duplicates.

    Returns
    -------
    Filtered doc_ids with near-duplicates removed (greedy, keeps first occurrence).
    """
    selected_indices: list[int] = []
    selected_embs: list[np.ndarray] = []

    for i, emb in enumerate(embeddings):
        if selected_embs:
            sims = np.stack(selected_embs) @ emb
            if float(np.max(sims)) > threshold:
                log.debug("Near-duplicate removed: %s (sim=%.4f)", doc_ids[i], np.max(sims))
                continue
        selected_indices.append(i)
        selected_embs.append(emb)

    return [doc_ids[i] for i in selected_indices]
```

### Tests

```python
"""
tests/unit/test_metrics.py
"""

import numpy as np
import pytest

from day5_metrics import (
    RetrievalEvalCase,
    deduplicate_near_duplicates,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], {"a"}, k=2) == 0.5


class TestRecall:
    def test_all_found(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant_returns_1(self):
        assert recall_at_k(["a"], set(), k=1) == 1.0


class TestMRR:
    def test_first_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_relevant(self):
        assert reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_none_relevant(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        # Ideal order: grade 2 first, then 1, then 0
        graded = {"a": 2, "b": 1, "c": 0}
        assert ndcg_at_k(["a", "b", "c"], graded, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        graded = {"a": 2, "b": 1}
        score = ndcg_at_k(["b", "a"], graded, k=2)
        assert score < 1.0

    def test_no_relevant_returns_zero(self):
        assert ndcg_at_k(["x"], {}, k=1) == 0.0


class TestNearDuplicateDedup:
    def test_exact_duplicates_removed(self):
        # Two identical unit vectors: cosine sim = 1.0 > threshold 0.97
        emb = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        result = deduplicate_near_duplicates(["a", "b", "c"], emb, threshold=0.97)
        assert "b" not in result

    def test_orthogonal_vectors_kept(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = deduplicate_near_duplicates(["a", "b"], emb, threshold=0.97)
        assert result == ["a", "b"]
```

## Exercises

1. Implement `f1_at_k(precision, recall) -> float` and extend `EvalResult` with an `f1` field. Update `evaluate_retrieval` to populate it.
2. Write a `compare_pipelines` function that takes two `retrieve_fn` callables (baseline and improved), runs both through `evaluate_retrieval`, and prints a formatted table showing the delta for each metric.
3. Create a 10-case synthetic eval set (hardcoded) for a fictional RAG corpus about Python best practices. Run `evaluate_retrieval` with a mock retriever that returns results in a fixed order. Verify the metric values match hand-calculated results.

(Solutions: ../solutions/week-05/day-5.md)

## Self-check

- [ ] I can implement Precision@k, Recall@k, MRR, and NDCG@k from scratch.
- [ ] I understand the difference between binary and graded relevance.
- [ ] I can explain what MRR measures and when it matters.
- [ ] I can implement cosine near-duplicate deduplication.
- [ ] I can design a minimal eval set for a retrieval pipeline and interpret the metrics.

## Supplementary links

- [TREC evaluation guidelines](https://trec.nist.gov/)
- [BEIR: Heterogeneous Benchmark for Zero-shot Evaluation](https://github.com/beir-cellar/beir)
- [pytrec_eval Python bindings](https://github.com/cvangysel/pytrec_eval)
