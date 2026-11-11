# Solutions — Week 03 · Day 2

## Exercise 1 — nlist sensitivity

```python
for nlist in (10, 50, 100, 200):
    ivf = TinyIVF(nlist=nlist)
    ivf.train(corpus)
    result, _ = bench(f"IVF nlist={nlist} nprobe=10", ivf.search, query, K, 10)
    log.info("  recall@%d = %.3f", K, recall_at_k(ground_truth, result))
```

Expected pattern: with `nprobe=10` fixed, higher `nlist` means each cell is smaller → a fixed probe budget covers a smaller fraction of the index → **recall drops**. The sweet spot is `nlist ≈ sqrt(n)` with `nprobe` tuned to hit target recall.

## Exercise 2 — RAM estimate function

```python
def ivf_ram_estimate(n: int, d: int, nlist: int) -> float:
    """Return approximate RAM in MB for IVFFlat (no compression)."""
    centroid_bytes = nlist * d * 4
    vector_bytes = n * d * 4
    total = centroid_bytes + vector_bytes
    return total / (1024 ** 2)

# Example: n=1_000_000, d=128, nlist=1000
# → 1000*128*4 + 1_000_000*128*4 = 512_000 + 512_000_000 = 512.5 MB
```

## Exercise 3 — HNSW streaming inserts vs IVF

IVF assigns each vector to a Voronoi cell determined by pre-trained centroids. A new vector can be inserted into the appropriate posting list without retraining — BUT if the new vector comes from a distribution shift, it may fall into a poor cell. More critically, if you stream enough new vectors, the centroid placement becomes stale and recall degrades, necessitating a full retrain.

HNSW inserts a new node by running a greedy search to find its M nearest neighbors at each layer, then adds bidirectional edges. The graph adapts locally. No global training step is needed and no distribution assumptions are made, so streaming inserts are safe indefinitely.

## Exercise 4 — HNSW RAM overhead estimate

With `M=16`, a typical HNSW node appears in on average ≈ 1.33 layers (controlled by `m_L = 1/ln(M)`).
Edges per node ≈ M × 1.33 ≈ 21.
Each edge is an int32 (4 bytes): 21 × 4 = 84 bytes overhead per vector.
Plus the raw float32 vector: 128 × 4 = 512 bytes.
**Total ≈ 596 bytes/vector** (compared to 512 for a flat index).

## Exercise 5 — Pytest for flat_search

```python
import pytest
import numpy as np
from day2_index_comparison import flat_search, make_data, N, DIM, K

@pytest.fixture(scope="module")
def data():
    return make_data(N, DIM, 42)

def test_flat_search_returns_k_results(data):
    corpus, query = data
    result = flat_search(corpus, query, K)
    assert len(result) == K

def test_flat_search_all_in_range(data):
    corpus, query = data
    result = flat_search(corpus, query, K)
    assert all(0 <= i < N for i in result)

def test_flat_search_unique(data):
    corpus, query = data
    result = flat_search(corpus, query, K)
    assert len(set(result)) == K
```
