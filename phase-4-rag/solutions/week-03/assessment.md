# Solutions — Week 03 Assessment

## Conceptual answers

**1. Exact NN vs ANN**

Exact NN computes the true nearest neighbors by comparing the query to every vector; it is always correct but O(n·d) per query. ANN uses an index (IVF, HNSW, etc.) to prune the search space, achieving sub-linear query time at the cost of occasionally missing some true neighbors. The trade-off is quantified as **recall\@k**: the fraction of exact top-k results returned by the ANN search. A value of 0.99 means 1% of true answers are missed.

**2. IVF build process**

1. Run k-means on a training sample to produce `nlist` centroids (Voronoi cells)
2. Assign each stored vector to its nearest centroid and record it in that centroid's posting list
3. At query time, compute distances to all `nlist` centroids and select the closest `nprobe` cells
4. Linearly scan only those `nprobe` posting lists and return the best k

`nlist` determines granularity: more cells = faster query per probe but worse recall for fixed nprobe. `nprobe` directly tunes the recall–latency trade-off: more probes = higher recall, higher latency.

**3. HNSW query phase**

1. Enter at the highest (sparsest) layer at the designated entry-point node
2. Greedily move to the neighbor closest to the query; repeat until no improvement (local minimum)
3. Descend to the next lower layer using the current best node as the new entry point
4. At layer 0, run a beam search with beam width `efSearch`; the beam maintains the top-efSearch candidates found so far
5. Return the top-k from the final beam

`efSearch` controls beam width at layer 0: higher efSearch = more nodes visited = higher recall, higher latency. Must be ≥ k.

**4. KD-trees at 1536 dims**

KD-trees partition space with axis-aligned splits. In high dimensions the number of cells to visit to guarantee finding the nearest neighbor is exponential in d — the tree's benefit over linear scan disappears. For d ≫ 20, all pairwise distances concentrate around a mean value, so the "nearest" neighbor is barely closer than average, and the tree must visit virtually every leaf. At 1536 dims, a KD-tree is strictly slower than a flat scan due to tree traversal overhead.

**5. 16× memory reduction**

Apply **Product Quantization (PQ)**. Split each d=1536 vector into m sub-vectors (e.g., m=48, sub-vector size 32), quantise each sub-vector with a 256-entry codebook (8-bit codes). Storage per vector drops from 1536×4=6144 bytes to 48 bytes — a 128× compression. For 16× specifically, use m=d/16=96 sub-vectors. The metric that degrades is **recall\@k** — typically 2–6% loss at default probe settings, recoverable by increasing nprobe.

**6. pgvector scenarios**

1. An existing web application already runs on Postgres (e.g., a SaaS product tracking users and documents). Adding pgvector avoids a second database tier, lets you JOIN vector search results with user rows, and reuses existing connection pooling, monitoring, and backups.
2. A strict compliance or regulated environment requires ACID transactions across both semantic search and relational data (e.g., deleting a user's embeddings atomically with their profile). pgvector inherits Postgres transactions; a separate vector DB would need a distributed 2PC or saga pattern.

**7. Pre-filter vs post-filter**

**Pre-filter** restricts the ANN search to only the matching subset before traversing the index. Advantage: correct (all returned results match the filter). Disadvantage: if the filtered subset is very small, the ANN graph may have poor connectivity within it, degrading recall.

**Post-filter** runs ANN on the full index, then discards non-matching results. Advantage: simple, uses the full index graph quality. Disadvantage: wastes compute scanning non-matching vectors; may return fewer than k results if many are filtered out.

**8. IVF returns -1 IDs (two causes)**

1. Index was not trained before adding vectors — `is_trained` is False; FAISS returns -1 for all results.
2. `index.ntotal == 0` — search was called before any vectors were added.

---

## Coding challenge — SmartVectorStore

```python
"""Assessment solution: SmartVectorStore with fallback padding."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.vector_store.base import SearchResult, VectorRecord, VectorStore
from src.vector_store.in_memory import InMemoryVectorStore

log = logging.getLogger(__name__)


class SmartVectorStore:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def upsert(self, records: list[VectorRecord]) -> None:
        self._store.upsert(records)

    def smart_query(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
        fallback_without_filter: bool = True,
    ) -> list[SearchResult]:
        """
        Filtered query with optional fallback padding to reach k results.
        """
        if filter:
            results = self._store.query(vector, k=k, filter=filter)
        else:
            return self._store.query(vector, k=k)

        if len(results) < k and fallback_without_filter:
            needed = k - len(results)
            existing_ids = {r.id for r in results}
            # Over-fetch to account for duplicates
            fallback = self._store.query(vector, k=k + len(existing_ids))
            for r in fallback:
                if r.id not in existing_ids:
                    results.append(r)
                    existing_ids.add(r.id)
                    if len(results) == k:
                        break
            log.debug("Padded results from %d to %d", k - needed, len(results))

        return sorted(results, key=lambda r: -r.score)[:k]
```

### Tests

```python
# tests/test_smart_vector_store.py
import numpy as np
import pytest
from src.vector_store.base import VectorRecord
from src.vector_store.in_memory import InMemoryVectorStore
# paste SmartVectorStore above into src/smart_store.py
from src.smart_store import SmartVectorStore

DIM = 32
RNG = np.random.default_rng(7)


def make_store(n: int, n_a: int) -> SmartVectorStore:
    base = InMemoryVectorStore(dim=DIM)
    records = [
        VectorRecord(
            id=f"r{i}",
            vector=RNG.standard_normal(DIM).astype(np.float32),
            metadata={"cat": "A" if i < n_a else "B"},
        )
        for i in range(n)
    ]
    base.upsert(records)
    return SmartVectorStore(base)


def test_no_filter_returns_k():
    store = make_store(100, 50)
    q = RNG.standard_normal(DIM).astype(np.float32)
    assert len(store.smart_query(q, k=10)) == 10


def test_restrictive_filter_pads_to_k():
    # Only 3 'A' records
    store = make_store(100, 3)
    q = RNG.standard_normal(DIM).astype(np.float32)
    results = store.smart_query(q, k=10, filter={"cat": "A"}, fallback_without_filter=True)
    assert len(results) == 10


def test_no_fallback_returns_fewer_than_k():
    store = make_store(100, 2)
    q = RNG.standard_normal(DIM).astype(np.float32)
    results = store.smart_query(q, k=10, filter={"cat": "A"}, fallback_without_filter=False)
    assert len(results) <= 2


def test_corpus_smaller_than_k():
    store = make_store(3, 3)
    q = RNG.standard_normal(DIM).astype(np.float32)
    results = store.smart_query(q, k=10)
    assert len(results) <= 3


def test_sorted_by_score_descending():
    store = make_store(50, 25)
    q = RNG.standard_normal(DIM).astype(np.float32)
    results = store.smart_query(q, k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
```
