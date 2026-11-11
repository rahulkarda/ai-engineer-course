# Phase 4 · Week 03 · Day 2 — Indexing Algorithms: Flat, IVF, and HNSW

> Objective: Understand how flat, IVF, and HNSW indexes work internally, and choose the right index for a given workload.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Three fundamental index families cover the vast majority of production vector search needs.

### 1. Flat index (brute-force)

No structure. Query computes distance to every vector.

- **Recall**: always 1.0 (exact)
- **Build time**: O(1) — just store vectors
- **Query time**: O(n · d)
- **When to use**: n < 50 k; correctness critical; offline batch jobs

### 2. IVF — Inverted File Index

Divides the vector space into `nlist` Voronoi cells using k-means clustering. Each vector is assigned to its nearest centroid. At query time, only `nprobe` cells (those whose centroids are nearest the query) are searched.

**Build phase**:
1. Run k-means on a training set → `nlist` centroids
2. Assign each vector to its nearest centroid
3. Store vectors in per-centroid "posting lists"

**Query phase**:
1. Find the `nprobe` nearest centroids to the query
2. Linearly scan the vectors in those `nprobe` lists
3. Return top-k overall

**Parameters**:
- `nlist` — number of cells. Rule of thumb: `sqrt(n)` to `4·sqrt(n)`. More cells = faster query, worse recall per probe.
- `nprobe` — cells searched at query time. Increasing it raises recall at the cost of latency. `nprobe=1` is fastest; `nprobe=nlist` equals flat search.

**Recall\@k vs nprobe**: recall increases roughly logarithmically with nprobe.

### 3. HNSW — Hierarchical Navigable Small World

HNSW builds a multi-layer proximity graph. It combines two ideas:
- **NSW (Navigable Small World)**: a proximity graph where each node connects to its nearest neighbors. Greedy search traverses the graph and finds near-optimal answers quickly.
- **Hierarchical layers**: like a skip list, upper layers have fewer nodes with long-range connections; lower layers are denser. This lets the search "zoom in" from coarse to fine.

**Build phase** (inserting a new vector `q`):
1. Randomly assign `q` a maximum layer `l` (exponential distribution, controlled by `m_L`)
2. From the top layer down to layer `l+1`: greedily find the single nearest neighbor — this becomes the entry point for the next layer
3. From layer `l` down to layer 0: find the `efConstruction` nearest neighbors, add bidirectional edges, keep only the best `M` connections per node using a heuristic that favors diversity

**Query phase** (searching for `q`):
1. Enter at the top layer, greedily descend to find the single nearest node
2. At layer 0, run a beam search with beam width `ef`
3. Return top-k from the beam

**Parameters**:
- `M` — max edges per node per layer (typical: 16–64). Higher M = better recall, more RAM, slower build.
- `efConstruction` — beam width during build (typical: 100–400). Higher = better recall, slower build.
- `ef` (search) — beam width at query time (typical: 50–200). Higher = better recall, slower query. Always ≥ k.

### Comparison table

| Index | Build cost | Query cost | Recall | RAM overhead | Streaming inserts |
|-------|-----------|-----------|--------|--------------|-------------------|
| Flat | O(1) | O(n·d) | 1.0 | none | yes |
| IVF | O(n·d) (k-means) | O(nprobe·n/nlist·d) | tunable | low | no (needs retrain) |
| HNSW | O(n·M·log n) | O(log n) | ~0.99 | ~(M·4) bytes/vec | yes |

## How it works internally

### HNSW graph navigation — step by step

Consider a 2-D example with 9 points and 3 layers:

```
Layer 2:  A ──── E
Layer 1:  A ─ B ─ E ─ H
Layer 0:  A─B─C─D─E─F─G─H─I  (all nodes, many edges)
```

Query `q` is closest to `F`:
1. Layer 2: start at `A`, check neighbors `{E}`. `E` is closer to `q` → move to `E`.
2. Layer 1: start at `E`, check `{B, H}`. `H` is closest to `q` → move to `H`.
3. Layer 0: beam search from `H`, discover `G, F, I`. Return `F` as nearest.

Total nodes visited: ~7 out of 9. For 1M nodes, HNSW typically visits O(log n) ≈ 20 nodes.

### Why IVF needs a training set

k-means requires seeing a representative sample of vectors to place centroids well. If you build an IVF index and then add vectors from a different distribution, recall degrades. HNSW has no training step — every insert is self-contained.

## Worked example

```python
"""
day2_index_comparison.py — benchmark flat vs simulated-IVF vs HNSW-like recall.

Uses only numpy; no FAISS yet (that's Day 3).
Shows the parameter-recall-latency relationship for each index type.

Install: uv add numpy
"""

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RNG_SEED = int(os.getenv("RNG_SEED", "42"))
N = 10_000
DIM = 64
K = 5


def make_data(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised corpus (n, d) and a single query (d,)."""
    rng = np.random.default_rng(seed)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    query = rng.standard_normal(d).astype(np.float32)
    query /= np.linalg.norm(query)
    return corpus, query


# ── Flat index ─────────────────────────────────────────────────────────────


def flat_search(corpus: np.ndarray, query: np.ndarray, k: int) -> list[int]:
    scores = corpus @ query
    return np.argpartition(scores, -k)[-k:].tolist()


# ── Tiny IVF (educational, not FAISS) ──────────────────────────────────────


@dataclass
class TinyIVF:
    nlist: int
    centroids: np.ndarray = field(init=False)
    posting_lists: dict[int, list[int]] = field(default_factory=dict, init=False)
    corpus: np.ndarray = field(init=False)

    def train(self, corpus: np.ndarray, max_iter: int = 20) -> None:
        """Mini k-means to place centroids."""
        self.corpus = corpus
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.choice(len(corpus), self.nlist, replace=False)
        self.centroids = corpus[idx].copy()
        for _ in range(max_iter):
            assignments = np.argmax(corpus @ self.centroids.T, axis=1)
            new_centroids = np.zeros_like(self.centroids)
            for c in range(self.nlist):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = corpus[mask].mean(axis=0)
                    new_centroids[c] /= np.linalg.norm(new_centroids[c]) + 1e-9
                else:
                    new_centroids[c] = self.centroids[c]
            self.centroids = new_centroids
        # Build posting lists
        assignments = np.argmax(corpus @ self.centroids.T, axis=1)
        self.posting_lists = {c: [] for c in range(self.nlist)}
        for i, c in enumerate(assignments):
            self.posting_lists[int(c)].append(i)

    def search(self, query: np.ndarray, k: int, nprobe: int) -> list[int]:
        centroid_scores = self.centroids @ query
        probe_cells = np.argpartition(centroid_scores, -nprobe)[-nprobe:].tolist()
        candidates = [i for c in probe_cells for i in self.posting_lists[c]]
        if not candidates:
            return []
        sub = self.corpus[candidates]
        scores = sub @ query
        top_local = np.argpartition(scores, -min(k, len(scores)))[-k:]
        return [candidates[i] for i in top_local]


def recall_at_k(exact: list[int], approx: list[int]) -> float:
    return len(set(exact) & set(approx)) / max(len(exact), 1)


def bench(label: str, fn, *args) -> tuple[list[int], float]:
    t0 = time.perf_counter()
    result = fn(*args)
    ms = (time.perf_counter() - t0) * 1000
    log.info("%-40s %.2f ms", label, ms)
    return result, ms


def main() -> None:
    corpus, query = make_data(N, DIM, RNG_SEED)

    ground_truth, _ = bench("Flat (exact)", flat_search, corpus, query, K)

    ivf = TinyIVF(nlist=100)
    ivf.train(corpus)

    for nprobe in (1, 5, 10, 20, 50):
        result, _ = bench(f"IVF nlist=100 nprobe={nprobe}", ivf.search, query, K, nprobe)
        log.info("  recall@%d = %.3f", K, recall_at_k(ground_truth, result))


if __name__ == "__main__":
    main()
```

Run: `uv run python day2_index_comparison.py`

## Exercises

1. Modify `TinyIVF` to accept `nlist` as a constructor parameter and run experiments for `nlist` in `{10, 50, 100, 200}` with fixed `nprobe=10`. How does nlist affect recall?
2. Implement a function `ivf_ram_estimate(n, d, nlist)` that returns the approximate RAM in MB: centroids use `nlist × d × 4` bytes; vectors use `n × d × 4` bytes.
3. Explain in 2–3 sentences why HNSW supports streaming inserts but IVF does not.
4. For HNSW with `M=16`: if each node stores edges at each layer and an average node appears in 1.5 layers, estimate the RAM overhead per vector (ignoring the raw vector storage).
5. Write a pytest test for `flat_search` that verifies the result set is a subset of `range(N)` and has exactly `K` unique elements.

(Solutions: ../solutions/week-03/day-2.md)

## Self-check

- [ ] I can describe the IVF build and query phases without notes
- [ ] I can explain HNSW layer assignment and greedy descent
- [ ] I can choose between Flat/IVF/HNSW given n, latency target, and recall requirement
- [ ] I understand the role of `M`, `efConstruction`, and `ef` in HNSW

## Supplementary

- [HNSW paper (Malkov & Yashunin 2018)](https://arxiv.org/abs/1603.09320)
- [Pinecone — HNSW explained](https://www.pinecone.io/learn/hnsw/)
- [FAISS index factory guide](https://github.com/facebookresearch/faiss/wiki/The-index-factory)
