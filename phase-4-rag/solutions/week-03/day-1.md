# Solutions — Week 03 · Day 1

## Exercise 1 — L2 distance with unnormalised vectors

```python
def exact_search_l2(corpus: np.ndarray, query: np.ndarray, k: int) -> SearchResult:
    """Brute-force L2 search (smaller = more similar)."""
    t0 = time.perf_counter()
    diff = corpus - query          # (n, d)
    dists = np.sum(diff ** 2, axis=1)
    top_k_ids = np.argpartition(dists, k)[:k].tolist()
    elapsed = (time.perf_counter() - t0) * 1000
    return SearchResult(sorted(top_k_ids, key=lambda i: dists[i]), elapsed)
```

The recall ranking **can** change because L2 distance is not scale-invariant: vectors with large magnitude dominate cosine similarity but not L2. For normalised vectors, `cosine_distance = 2 - 2·cosine_similarity`, so rankings are equivalent. After unnormalising, they diverge.

## Exercise 2 — Batch exact search

```python
def batch_exact_search(
    corpus: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, float]:
    """
    Args:
        corpus: (n, d)
        queries: (q, d)
    Returns:
        ids: (q, k), elapsed_ms
    """
    t0 = time.perf_counter()
    scores = queries @ corpus.T    # (q, n)
    ids = np.argpartition(scores, -k, axis=1)[:, -k:]
    elapsed = (time.perf_counter() - t0) * 1000
    return ids, elapsed

# Single query per-query latency: total_ms / q
```

Per-query latency drops significantly because matrix multiplication is vectorised over BLAS.

## Exercise 3 — Recall curve

```python
import matplotlib.pyplot as plt   # uv add matplotlib

n_probes = [10, 20, 50, 100, 200, 500, 1000]
recalls = []
for n_probe in n_probes:
    ann = naive_ann(corpus, query, K, n_probe=n_probe)
    recalls.append(recall_at_k(exact.indices, ann.indices))

plt.plot(n_probes, recalls, marker="o")
plt.xscale("log")
plt.xlabel("n_probe")
plt.ylabel(f"recall@{K}")
plt.title("Naive ANN: recall vs probe budget")
plt.savefig("recall_curve.png", dpi=120)
```

The curve is concave and approaches 1.0 only as n\_probe → n. A real ANN algorithm achieves high recall with far lower n\_probe by using structure.

## Exercise 4 — RAM estimate

```
n = 10_000_000, d = 1536, dtype = float32 (4 bytes)
RAM = 10_000_000 × 1536 × 4 = 61_440_000_000 bytes ≈ 57.2 GB
```

This is the raw vector matrix only, before any index structure overhead.

## Exercise 5 — Pytest for exact search

```python
import pytest
import numpy as np
from day1_exact_vs_ann import exact_search, generate_corpus, N_VECTORS, DIM, K

@pytest.fixture
def corpus():
    return generate_corpus(N_VECTORS, DIM, 42)

@pytest.fixture
def query(corpus):
    return generate_corpus(1, DIM, 99)[0]

def test_exact_search_returns_k_results(corpus, query):
    result = exact_search(corpus, query, K)
    assert len(result.indices) == K

def test_exact_search_unique_ids(corpus, query):
    result = exact_search(corpus, query, K)
    assert len(set(result.indices)) == K

def test_exact_search_ids_in_range(corpus, query):
    result = exact_search(corpus, query, K)
    assert all(0 <= i < N_VECTORS for i in result.indices)
```
