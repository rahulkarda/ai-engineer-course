# Solutions — Week 02 · Day 4

## Exercise answers

### Exercise 1 — Pairwise similarity matrix

```python
# embeddings/similarity.py  (addition)

def pairwise_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return the N×N cosine similarity matrix for *vectors* (shape N×D, unit-normalised).

    Uses matrix multiplication: cos_sim(i,j) = vectors[i] · vectors[j]
    when all vectors are unit-normalised.
    """
    return vectors @ vectors.T  # shape (N, N)
```

No Python loop needed. For N=10,000 and D=384 this takes ~100ms on a modern CPU vs. minutes for a nested loop.

### Exercise 2 — `top_k` with threshold

```python
def top_k(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 5,
    threshold: float | None = None,
) -> list[tuple[int, float]]:
    sims = cosine_similarity_matrix(query, corpus)
    indices = np.argsort(sims)[::-1][:k]
    results = [(int(i), float(sims[i])) for i in indices]
    if threshold is not None:
        results = [(i, s) for i, s in results if s >= threshold]
    return results
```

### Exercise 3 — Why `corpus @ query` is cosine similarity

Comment to add to `cosine_similarity_matrix`:

```python
# When vectors are unit-normalised (||v|| = 1 for all v):
#   cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
#                    = dot(a, b) / (1 * 1)
#                    = dot(a, b)
#
# corpus @ query computes [dot(row_0, query), dot(row_1, query), ...]
# which equals [cosine_sim(row_0, query), ...] after normalisation.
# This is O(N*D) — a single BLAS matrix-vector multiply.
```

### Exercise 4 — Storage calculation

```
1,000,000 chunks × 1536 dims × 4 bytes (float32) = 6,144,000,000 bytes
                                                   = 6,144 MB
                                                   ≈ 6.1 GB

For text-embedding-3-small with matryoshka-truncated 512 dims:
1,000,000 × 512 × 4 = 2,048 MB ≈ 2.0 GB
```

This is just raw vector storage. Add index overhead (HNSW ≈ 30–100% extra depending on M and ef_construction).

### Exercise 5 — Symmetry test

```python
def test_cosine_similarity_symmetric():
    rng = np.random.default_rng(42)
    a = normalise(rng.standard_normal(16).astype(np.float32))
    b = normalise(rng.standard_normal(16).astype(np.float32))
    assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < 1e-6
```

`dot(a,b) = dot(b,a)` by definition of the dot product, so cosine similarity is always symmetric.
