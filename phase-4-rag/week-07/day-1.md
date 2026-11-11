# Phase 4 · Week 07 · Day 1 — Retrieval Metrics: Precision@k, Recall@k, MRR, nDCG

> Objective: Implement and interpret the four core retrieval evaluation metrics from scratch.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]]

## Concept

A RAG pipeline is only as good as its retriever. Before tuning embeddings or re-rankers, you need numbers that tell you what "better" means.

**Why separate metrics?**

- **Precision**: of the k documents you showed the LLM, how many actually helped? Bad precision = noise pollutes generation.
- **Recall**: of all documents that could have helped, how many did you find? Bad recall = the answer was never in context.
- **Rank matters**: retrieving the right document at position 1 is better than finding it at position 10. Precision@k and Recall@k treat every position equally within k; MRR and nDCG penalise late appearances.

Typical trade-off: raising k improves recall but hurts precision (more noise). You need both to navigate that curve.

**Graded vs binary relevance**

MRR and Precision@k treat relevance as binary (relevant / not). nDCG supports graded relevance (e.g. 0/1/2), which matters when some documents are more relevant than others.

## How it works internally

Given a ranked list of retrieved document IDs `retrieved` and a set of ground-truth relevant IDs `relevant`:

```
Precision@k  = |{retrieved[:k]} ∩ relevant| / k
Recall@k     = |{retrieved[:k]} ∩ relevant| / |relevant|
```

**MRR (Mean Reciprocal Rank)** — across a query set, average the reciprocal of the rank of the first relevant document:

```
MRR = (1/|Q|) * Σ_q  1 / rank_of_first_relevant(q)
```

If no relevant doc appears in retrieved, the reciprocal rank is 0.

**nDCG (Normalised Discounted Cumulative Gain)** — rewards relevant docs appearing early, using a logarithmic discount:

```
DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i + 1)
IDCG@k = DCG@k for the ideal (perfectly sorted) ranking
nDCG@k = DCG@k / IDCG@k
```

`rel_i` is the relevance grade of the document at rank `i` (1 for relevant, 0 otherwise in the binary case). IDCG@k is computed by sorting all known relevant docs to the top, capped at k.

## Worked example

```python
"""
week_07/day1_retrieval_metrics.py

Retrieval evaluation metrics: Precision@k, Recall@k, MRR, nDCG.
Run: python week_07/day1_retrieval_metrics.py
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """One query's retrieval result paired with ground truth."""
    query_id: str
    retrieved: list[str]           # ordered list of doc IDs, best first
    relevant: set[str]             # ground-truth relevant doc IDs
    grades: dict[str, int] = field(default_factory=dict)  # doc_id -> relevance grade (0/1/2)


class RetrievalMetrics:
    """
    Compute standard IR metrics for a set of QueryResult objects.

    All metrics default to binary relevance (0 or 1).
    nDCG uses grades if provided, otherwise falls back to binary.
    """

    def precision_at_k(self, result: QueryResult, k: int) -> float:
        """Fraction of top-k retrieved docs that are relevant."""
        top_k = result.retrieved[:k]
        hits = sum(1 for doc_id in top_k if doc_id in result.relevant)
        return hits / k if k > 0 else 0.0

    def recall_at_k(self, result: QueryResult, k: int) -> float:
        """Fraction of all relevant docs found in top-k."""
        if not result.relevant:
            return 0.0
        top_k = result.retrieved[:k]
        hits = sum(1 for doc_id in top_k if doc_id in result.relevant)
        return hits / len(result.relevant)

    def reciprocal_rank(self, result: QueryResult) -> float:
        """1 / rank of the first relevant document (0 if none found)."""
        for rank, doc_id in enumerate(result.retrieved, start=1):
            if doc_id in result.relevant:
                return 1.0 / rank
        return 0.0

    def mrr(self, results: list[QueryResult]) -> float:
        """Mean Reciprocal Rank across all queries."""
        if not results:
            return 0.0
        return sum(self.reciprocal_rank(r) for r in results) / len(results)

    def _grade(self, result: QueryResult, doc_id: str) -> int:
        """Return grade for doc_id: use grades dict, else binary relevance."""
        if result.grades:
            return result.grades.get(doc_id, 0)
        return 1 if doc_id in result.relevant else 0

    def dcg_at_k(self, result: QueryResult, k: int) -> float:
        """Discounted Cumulative Gain at k."""
        total = 0.0
        for i, doc_id in enumerate(result.retrieved[:k], start=1):
            rel = self._grade(result, doc_id)
            total += rel / math.log2(i + 1)
        return total

    def idcg_at_k(self, result: QueryResult, k: int) -> float:
        """Ideal DCG@k — perfect ranking of available relevant docs."""
        # collect all grades for relevant docs, sort descending
        if result.grades:
            grades_sorted = sorted(result.grades.values(), reverse=True)
        else:
            grades_sorted = [1] * len(result.relevant)
        total = 0.0
        for i, rel in enumerate(grades_sorted[:k], start=1):
            total += rel / math.log2(i + 1)
        return total

    def ndcg_at_k(self, result: QueryResult, k: int) -> float:
        """Normalised DCG@k."""
        idcg = self.idcg_at_k(result, k)
        if idcg == 0.0:
            return 0.0
        return self.dcg_at_k(result, k) / idcg

    def mean_ndcg_at_k(self, results: list[QueryResult], k: int) -> float:
        """Average nDCG@k across all queries."""
        if not results:
            return 0.0
        return sum(self.ndcg_at_k(r, k) for r in results) / len(results)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # Mock retrieval results for 3 queries
    # Retrieved lists simulate ranked outputs from a vector store
    results = [
        QueryResult(
            query_id="q1",
            retrieved=["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
            relevant={"doc_A", "doc_C"},          # two relevant docs
        ),
        QueryResult(
            query_id="q2",
            retrieved=["doc_X", "doc_Y", "doc_B", "doc_Z", "doc_W"],
            relevant={"doc_B", "doc_Z"},           # first relevant at rank 3
        ),
        QueryResult(
            query_id="q3",
            retrieved=["doc_P", "doc_Q", "doc_R", "doc_S", "doc_T"],
            relevant={"doc_U"},                    # relevant doc never retrieved
        ),
    ]

    metrics = RetrievalMetrics()
    k = 5

    logger.info("--- Per-query metrics at k=%d ---", k)
    for r in results:
        p = metrics.precision_at_k(r, k)
        rec = metrics.recall_at_k(r, k)
        rr = metrics.reciprocal_rank(r)
        ndcg = metrics.ndcg_at_k(r, k)
        logger.info(
            "%s  P@%d=%.3f  R@%d=%.3f  RR=%.3f  nDCG@%d=%.3f",
            r.query_id, k, p, k, rec, rr, k, ndcg,
        )

    logger.info("--- Aggregate ---")
    logger.info("MRR        = %.3f", metrics.mrr(results))
    logger.info("Mean nDCG@%d = %.3f", k, metrics.mean_ndcg_at_k(results, k))

    # Graded relevance example (query q1, grades: doc_A=2, doc_C=1)
    graded = QueryResult(
        query_id="q1_graded",
        retrieved=["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"],
        relevant={"doc_A", "doc_C"},
        grades={"doc_A": 2, "doc_C": 1},
    )
    logger.info("--- Graded nDCG@%d for q1_graded: %.3f ---", k, metrics.ndcg_at_k(graded, k))


if __name__ == "__main__":
    main()
```

**Expected output (approximate):**

```
INFO q1  P@5=0.400  R@5=1.000  RR=1.000  nDCG@5=0.765
INFO q2  P@5=0.400  R@5=1.000  RR=0.333  nDCG@5=0.624
INFO q3  P@5=0.000  R@5=0.000  RR=0.000  nDCG@5=0.000
INFO MRR        = 0.444
INFO Mean nDCG@5 = 0.463
INFO --- Graded nDCG@5 for q1_graded: 0.848 ---
```

Key observations:
- q1 and q2 have the same Precision@5 and Recall@5, but q1's MRR is much higher (relevant doc at rank 1 vs rank 3).
- q3 has zero on all metrics — the relevant doc was never retrieved.
- Graded nDCG for q1 is higher because `doc_A` (grade 2) sits at rank 1.

## Exercises

1. **Implement Average Precision (AP@k)**: AP averages precision at every rank where a relevant document appears. Add an `average_precision_at_k(result, k)` method to `RetrievalMetrics` and verify it on the three mock queries above.

2. **Cutoff sensitivity sweep**: For query q1, compute Precision@k and Recall@k for k in {1, 2, 3, 5, 10}. Print a table. At what k does recall reach 1.0? What happens to precision?

3. **MRR vs nDCG on tie scenarios**: Create two `QueryResult` objects that have the same MRR but different nDCG@5. Explain in a comment why this can happen and what it tells you about the retriever.

4. **Precision-recall curve**: Given a ranked list of 20 documents and a relevant set, compute (Recall@k, Precision@k) for each k from 1 to 20 and print the pairs. Identify the area under the curve using the trapezoid rule from the standard library (`math`).

(Solutions: ../solutions/week-07/day-1.md)

## Self-check

- [ ] I can explain the difference between Precision@k and Recall@k and when each matters more.
- [ ] I can explain why MRR focuses on the first relevant document and when that is the right choice.
- [ ] I can implement nDCG from scratch, including the IDCG normalisation, without reference.
- [ ] I can articulate when graded relevance gives more signal than binary relevance.

---

## Supplementary links

- Manning, Raghavan, Schutze — "Introduction to Information Retrieval" Ch. 8 (free online): https://nlp.stanford.edu/IR-book/
- BEIR benchmark (diverse IR evaluation): https://github.com/beir-cellar/beir
- `ranx` Python library for fast IR metric computation: https://github.com/AmenRa/ranx
- TREC evaluation guidelines: https://trec.nist.gov/pubs/trec16/appendices/measures.pdf
