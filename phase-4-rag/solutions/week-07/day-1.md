# Solutions — Phase 4 · Week 07 · Day 1

## Exercise 1

```python
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Compute retrieval evaluation metrics using binary relevance."""

    def precision_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        """Fraction of top-k retrieved docs that are relevant."""
        if k <= 0:
            raise ValueError("k must be positive")
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for doc in top_k if doc in relevant)
        score = hits / k
        logger.debug("precision@%d: %d/%d = %.4f", k, hits, k, score)
        return score

    def recall_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        """Fraction of relevant docs found in top-k retrieved."""
        if k <= 0:
            raise ValueError("k must be positive")
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        hits = sum(1 for doc in top_k if doc in relevant)
        score = hits / len(relevant)
        logger.debug("recall@%d: %d/%d = %.4f", k, hits, len(relevant), score)
        return score

    def mrr(self, retrieved: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank — reciprocal of rank of first relevant doc."""
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                score = 1.0 / rank
                logger.debug("MRR: first hit at rank %d, score=%.4f", rank, score)
                return score
        logger.debug("MRR: no relevant doc found, score=0.0")
        return 0.0

    def ndcg_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain at k (binary relevance)."""
        if k <= 0:
            raise ValueError("k must be positive")
        top_k = retrieved[:k]

        # Actual DCG
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc in enumerate(top_k, start=1)
            if doc in relevant
        )

        # Ideal DCG: all relevant docs at top positions
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        if idcg == 0.0:
            return 0.0
        score = dcg / idcg
        logger.debug("nDCG@%d: DCG=%.4f, IDCG=%.4f, score=%.4f", k, dcg, idcg, score)
        return score


# --- Tests ---

def test_retrieval_metrics() -> None:
    metrics = RetrievalMetrics()

    # Test set 1: perfect retrieval
    retrieved_1 = ["d1", "d2", "d3", "d4", "d5"]
    relevant_1 = {"d1", "d2", "d3"}

    assert metrics.precision_at_k(retrieved_1, relevant_1, k=3) == 1.0
    assert metrics.recall_at_k(retrieved_1, relevant_1, k=3) == 1.0
    assert metrics.mrr(retrieved_1, relevant_1) == 1.0
    assert metrics.ndcg_at_k(retrieved_1, relevant_1, k=5) == 1.0

    # Test set 2: first relevant doc at position 3
    retrieved_2 = ["d4", "d5", "d1", "d2", "d3"]
    relevant_2 = {"d1", "d2", "d3"}

    assert metrics.precision_at_k(retrieved_2, relevant_2, k=3) == pytest.approx(1 / 3, abs=1e-9)
    assert metrics.recall_at_k(retrieved_2, relevant_2, k=3) == pytest.approx(1 / 3, abs=1e-9)
    assert metrics.mrr(retrieved_2, relevant_2) == pytest.approx(1 / 3, abs=1e-9)

    # Test set 3: no relevant docs retrieved
    retrieved_3 = ["d6", "d7", "d8"]
    relevant_3 = {"d1", "d2"}

    assert metrics.precision_at_k(retrieved_3, relevant_3, k=3) == 0.0
    assert metrics.recall_at_k(retrieved_3, relevant_3, k=3) == 0.0
    assert metrics.mrr(retrieved_3, relevant_3) == 0.0
    assert metrics.ndcg_at_k(retrieved_3, relevant_3, k=3) == 0.0

    print("All Exercise 1 tests passed.")


if __name__ == "__main__":
    import pytest
    logging.basicConfig(level=logging.DEBUG)
    test_retrieval_metrics()
```

## Exercise 2

```python
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    def precision_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        return sum(1 for doc in top_k if doc in relevant) / k

    def recall_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        if not relevant:
            return 0.0
        top_k = retrieved[:k]
        return sum(1 for doc in top_k if doc in relevant) / len(relevant)


def analyze_k_tradeoff() -> None:
    """Show how precision and recall change as k grows."""
    metrics = RetrievalMetrics()

    # Fixed result set: relevant docs scattered throughout 10 positions
    retrieved = ["d1", "d6", "d2", "d7", "d3", "d8", "d9", "d10", "d11", "d12"]
    relevant = {"d1", "d2", "d3", "d4", "d5"}  # 5 relevant docs, 3 are in top 10

    print(f"{'k':>4} | {'precision@k':>12} | {'recall@k':>10}")
    print("-" * 35)
    for k in range(1, 11):
        p = metrics.precision_at_k(retrieved, relevant, k)
        r = metrics.recall_at_k(retrieved, relevant, k)
        bar_p = "#" * int(p * 20)
        bar_r = "#" * int(r * 20)
        print(f"{k:>4} | {p:>8.4f} {bar_p:<12} | {r:>6.4f} {bar_r}")

    print()
    print("Observations:")
    print("  Precision@k: generally DECREASES as k grows. Retrieving more")
    print("  documents dilutes the signal — later positions are less likely")
    print("  to be relevant. It can stay flat or briefly rise when a relevant")
    print("  doc appears at the next position, but the trend is downward.")
    print()
    print("  Recall@k: monotonically NON-DECREASING as k grows. Adding more")
    print("  retrieved docs can only find new relevant ones or not — it can")
    print("  never lose a relevant doc already found. It plateaus once all")
    print("  relevant docs are retrieved or the list is exhausted.")


if __name__ == "__main__":
    analyze_k_tradeoff()
```

**Explanation:**

Precision@k tends to decrease as k grows because later-ranked documents are progressively less likely to be relevant — the ranker has placed its best guesses at the top. Each irrelevant document added to the top-k pool lowers the fraction of relevant ones.

Recall@k is monotonically non-decreasing. Extending the retrieved list to position k+1 can only add a relevant document or not; it cannot remove one already counted. Once k reaches the total number of relevant documents in the corpus, recall plateaus at 1.0.

This precision-recall tradeoff means optimizing for both simultaneously requires a balanced metric such as F1 or nDCG.

## Exercise 3

```python
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    def mrr(self, retrieved: list[str], relevant: set[str]) -> float:
        for rank, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(self, retrieved: list[str], relevant: set[str], k: int) -> float:
        top_k = retrieved[:k]
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc in enumerate(top_k, start=1)
            if doc in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0


def demonstrate_mrr_vs_ndcg() -> None:
    metrics = RetrievalMetrics()

    # Both result sets have MRR = 1.0 (first relevant doc at rank 1).
    # But nDCG@5 differs because the remaining relevant docs appear at
    # different positions.

    relevant = {"d1", "d2", "d3"}  # 3 relevant docs total

    # Set A: first relevant at 1, others at 2 and 3 (near-ideal)
    result_set_a = ["d1", "d2", "d3", "d4", "d5"]
    # Set B: first relevant at 1, others buried at 4 and 5
    result_set_b = ["d1", "d4", "d5", "d2", "d3"]

    mrr_a = metrics.mrr(result_set_a, relevant)
    mrr_b = metrics.mrr(result_set_b, relevant)
    ndcg_a = metrics.ndcg_at_k(result_set_a, relevant, k=5)
    ndcg_b = metrics.ndcg_at_k(result_set_b, relevant, k=5)

    print(f"Result set A (relevant at ranks 1, 2, 3): MRR={mrr_a:.4f}, nDCG@5={ndcg_a:.4f}")
    print(f"Result set B (relevant at ranks 1, 4, 5): MRR={mrr_b:.4f}, nDCG@5={ndcg_b:.4f}")
    assert mrr_a == mrr_b, "MRR should be equal"
    assert ndcg_a > ndcg_b, "nDCG should differ"
    print()
    print("Both have MRR=1.0 because the FIRST relevant doc is at rank 1 in both.")
    print("nDCG@5 is higher for A because docs at ranks 2 and 3 receive higher")
    print("logarithmic discounts than docs at ranks 4 and 5.")


if __name__ == "__main__":
    demonstrate_mrr_vs_ndcg()
```

**Explanation:**

MRR (Mean Reciprocal Rank) only examines where the *first* relevant document appears. Once that rank is fixed, all remaining relevant documents are invisible to MRR — it is a single-point metric. Both result sets have their first relevant hit at rank 1, so both have MRR = 1.0.

nDCG sums a logarithmically discounted gain for *every* relevant document in the list. Documents at higher ranks receive less discount (`log2(rank+1)` grows with rank), so a list that places all three relevant docs at ranks 1–3 accumulates more DCG than one where two of the three are buried at ranks 4–5. This makes nDCG sensitive to the quality of the full ranking, not just the first hit.

## Exercise 4

```python
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def mean_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """
    Compute mean of each metric key across all per-query metric dicts.

    Args:
        results: List of per-query metric dicts. Keys may include any
                 combination of 'precision', 'recall', 'mrr', 'ndcg',
                 'ap' (average precision), etc. Missing keys in a dict
                 are skipped when computing the mean for that key.

    Returns:
        Dict mapping each observed metric key to its mean value.
        Prefixed with 'mean_' (e.g. 'mrr' -> 'mean_mrr', 'ap' -> 'MAP').

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("results list is empty — cannot compute mean metrics")

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for query_result in results:
        for key, value in query_result.items():
            sums[key] += value
            counts[key] += 1

    averaged: dict[str, float] = {}
    for key in sums:
        mean_val = sums[key] / counts[key]
        # Use conventional names: MAP for average precision, mean_* otherwise
        out_key = "MAP" if key == "ap" else f"mean_{key}"
        averaged[out_key] = mean_val
        logger.debug("mean_metrics: %s = %.4f (over %d queries)", out_key, mean_val, counts[key])

    return averaged


# --- Tests ---

def test_mean_metrics_basic() -> None:
    results = [
        {"precision": 0.8, "recall": 0.6, "mrr": 1.0, "ndcg": 0.9, "ap": 0.85},
        {"precision": 0.6, "recall": 0.4, "mrr": 0.5, "ndcg": 0.7, "ap": 0.65},
        {"precision": 0.4, "recall": 0.8, "mrr": 1.0, "ndcg": 0.8, "ap": 0.75},
    ]
    averaged = mean_metrics(results)

    assert abs(averaged["mean_precision"] - (0.8 + 0.6 + 0.4) / 3) < 1e-9
    assert abs(averaged["mean_recall"] - (0.6 + 0.4 + 0.8) / 3) < 1e-9
    assert abs(averaged["mean_mrr"] - (1.0 + 0.5 + 1.0) / 3) < 1e-9
    assert abs(averaged["mean_ndcg"] - (0.9 + 0.7 + 0.8) / 3) < 1e-9
    assert abs(averaged["MAP"] - (0.85 + 0.65 + 0.75) / 3) < 1e-9
    print("test_mean_metrics_basic passed")


def test_mean_metrics_missing_keys() -> None:
    """Keys missing from some dicts are averaged only over queries that have them."""
    results = [
        {"mrr": 1.0, "ndcg": 0.9},
        {"mrr": 0.5},               # ndcg missing
        {"mrr": 0.0, "ndcg": 0.4},
    ]
    averaged = mean_metrics(results)
    assert abs(averaged["mean_mrr"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9
    assert abs(averaged["mean_ndcg"] - (0.9 + 0.4) / 2) < 1e-9
    print("test_mean_metrics_missing_keys passed")


def test_mean_metrics_single_query() -> None:
    results = [{"mrr": 0.75, "ap": 0.5}]
    averaged = mean_metrics(results)
    assert averaged["mean_mrr"] == 0.75
    assert averaged["MAP"] == 0.5
    print("test_mean_metrics_single_query passed")


def test_mean_metrics_empty_raises() -> None:
    try:
        mean_metrics([])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("test_mean_metrics_empty_raises passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_mean_metrics_basic()
    test_mean_metrics_missing_keys()
    test_mean_metrics_single_query()
    test_mean_metrics_empty_raises()
    print("All Exercise 4 tests passed.")
```
