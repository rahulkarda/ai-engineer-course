# Phase 4 · Week 03 · Day 1 — Vector Database Concepts & ANN vs Exact Search

> Objective: Understand what a vector database is, why approximate nearest-neighbor search exists, and when to choose ANN over exact search.  ·  Time: ~1 hr  ·  Prereqs: [[week-02/day-7]]

## Concept

### What is a vector database?

A **vector database** is a storage and retrieval system whose primary primitive is a dense floating-point vector — an embedding — rather than a row or document. Every item stored has an associated vector (e.g., 384 or 1536 floats). Retrieval is by **similarity** rather than equality: "find the 10 vectors closest to this query vector."

Core operations:
- `upsert(id, vector, metadata)` — add or replace a vector
- `query(vector, k, filter)` — return the k most-similar stored vectors
- `delete(id)` — remove a vector
- `persist()` — flush to durable storage

### Why not a regular database?

A SQL `WHERE embedding = ?` makes no sense for approximate similarity. An index like B-tree works on ordinal comparisons; a KD-tree degrades badly above ~20 dimensions. Embeddings typically have 384–3072 dimensions. Specialised data structures and algorithms are required.

### Exact nearest-neighbor (Exact NN)

Exact search computes the distance between the query and **every** stored vector, then returns the true top-k. This is:
- Correct — always returns the true answer
- O(n · d) per query (n = number of vectors, d = dimensionality)
- Fine for n < ~100 k; impractical at tens of millions

### Approximate nearest-neighbor (ANN)

ANN algorithms trade a small accuracy loss for dramatically faster retrieval. They pre-build an index that prunes the search space.

Key concept: **recall\@k** — the fraction of true top-k results that appear in the ANN result set. A recall of 0.99 means 99 % of exact answers are returned.

ANN is almost always the right choice when:
- n > 100 k vectors
- Query latency must be < 50 ms
- A 1–5 % recall loss is acceptable (it almost always is in RAG)

## How it works internally

### Distance metrics

The choice of metric must match how the embedding model was trained.

| Metric | Formula | Typical use |
|--------|---------|-------------|
| L2 (Euclidean) | `sqrt(Σ(aᵢ − bᵢ)²)` | FAISS default, image embeddings |
| Cosine similarity | `dot(a,b) / (‖a‖·‖b‖)` | Text embeddings (OpenAI, sentence-transformers) |
| Inner product (IP) | `Σ aᵢ·bᵢ` | Equivalent to cosine when vectors are L2-normalised |
| Manhattan (L1) | `Σ|aᵢ − bᵢ|` | Sparse or integer-valued vectors |

**Normalise before storing** when your model trains with cosine similarity. Then IP == cosine, and you can use the faster IP path.

### The curse of dimensionality

In high-dimensional spaces, all pairwise distances converge. KD-trees become linear-scan equivalents above ~20 dims. This forces ANN algorithms to abandon exact partitioning and use probabilistic or graph-based structures.

## Worked example

```python
"""
day1_exact_vs_ann.py — compare exact vs ANN recall on a tiny synthetic dataset.

Install: uv add numpy  (no heavy deps needed today)
"""

import logging
import os
import time
from typing import NamedTuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RNG_SEED = int(os.getenv("RNG_SEED", "42"))
N_VECTORS = 5_000
DIM = 128
K = 10


class SearchResult(NamedTuple):
    indices: list[int]
    elapsed_ms: float


def generate_corpus(n: int, d: int, seed: int) -> np.ndarray:
    """Return float32 matrix of shape (n, d) with L2-normalised rows."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def exact_search(corpus: np.ndarray, query: np.ndarray, k: int) -> SearchResult:
    """Brute-force inner-product search (== cosine on normalised vectors)."""
    t0 = time.perf_counter()
    scores = corpus @ query          # shape (n,)
    top_k = int(np.argpartition(scores, -k)[-k:].tolist()[0])  # warm up
    top_k_ids = np.argpartition(scores, -k)[-k:].tolist()
    elapsed = (time.perf_counter() - t0) * 1000
    return SearchResult(sorted(top_k_ids, key=lambda i: -scores[i]), elapsed)


def naive_ann(
    corpus: np.ndarray,
    query: np.ndarray,
    k: int,
    n_probe: int = 50,
    seed: int = 0,
) -> SearchResult:
    """
    Simulated ANN: randomly select n_probe candidates, then exact-search them.
    This is NOT a real ANN algorithm — it demonstrates the recall/speed trade-off.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    candidates = rng.choice(len(corpus), size=n_probe, replace=False)
    sub_corpus = corpus[candidates]
    scores = sub_corpus @ query
    top_local = np.argpartition(scores, -k)[-k:]
    top_k_ids = [int(candidates[i]) for i in top_local]
    elapsed = (time.perf_counter() - t0) * 1000
    return SearchResult(sorted(top_k_ids, key=lambda i: -(corpus[i] @ query)), elapsed)


def recall_at_k(exact: list[int], approx: list[int]) -> float:
    return len(set(exact) & set(approx)) / len(exact)


def main() -> None:
    corpus = generate_corpus(N_VECTORS, DIM, RNG_SEED)
    query = generate_corpus(1, DIM, RNG_SEED + 99)[0]

    exact = exact_search(corpus, query, K)
    log.info("Exact search: %.2f ms, top-3 ids=%s", exact.elapsed_ms, exact.indices[:3])

    for n_probe in (20, 50, 100, 500):
        ann = naive_ann(corpus, query, K, n_probe=n_probe)
        recall = recall_at_k(exact.indices, ann.indices)
        log.info(
            "ANN (n_probe=%4d): %.2f ms, recall@%d=%.2f",
            n_probe, ann.elapsed_ms, K, recall,
        )


if __name__ == "__main__":
    main()
```

Run: `uv run python day1_exact_vs_ann.py`

Expected output (approximate):
```
INFO Exact search: 0.45 ms, top-3 ids=[...]
INFO ANN (n_probe=  20): 0.08 ms, recall@10=0.20
INFO ANN (n_probe= 500): 0.18 ms, recall@10=0.60
```

The takeaway: with a random probe strategy, recall grows with probe budget but never reaches 1.0 without scanning everything. Real ANN algorithms (IVF, HNSW) achieve high recall with far fewer probes by using structure.

## Exercises

1. Modify `generate_corpus` to produce **unnormalised** vectors, then switch the similarity metric to L2 distance. Does the recall ranking change? Why?
2. Add a `batch_exact_search` function that processes 100 queries at once using matrix multiplication. Measure the per-query latency compared to the single-query version.
3. Plot recall\@10 vs n\_probe (10, 20, 50, 100, 200, 500, 1000) for the naive ANN. What shape is the curve?
4. Research: for a 10M-vector index with 1536-dim float32 vectors, how many GB of RAM does the raw vector matrix require?
5. Write a pytest test that asserts `exact_search` always returns exactly `k` results.

(Solutions: ../solutions/week-03/day-1.md)

## Self-check

- [ ] I can explain the difference between exact NN and ANN search in one sentence
- [ ] I can list three distance metrics and state which to use for text embeddings
- [ ] I understand why recall@k < 1.0 is acceptable for RAG workloads
- [ ] I can calculate the RAM footprint of a float32 vector index given n and d

## Supplementary

- [FAISS wiki — Similarity search overview](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Pinecone — What is a vector database?](https://www.pinecone.io/learn/vector-database/)
- [ANN Benchmarks](https://ann-benchmarks.com/)
