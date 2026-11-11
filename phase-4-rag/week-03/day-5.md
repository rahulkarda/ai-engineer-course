# Phase 4 · Week 03 · Day 5 — Metadata Filtering, Persistence, and Scaling

> Objective: Master metadata filter patterns, implement durable persistence, and understand production scaling strategies.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Metadata filtering

Pure vector search returns the most similar vectors globally. In production you almost always need to **pre-filter** or **post-filter** by structured attributes:

- `"Only documents from user 42"`
- `"Only chunks from PDFs uploaded this week"`
- `"Only paragraphs from the 'legal' namespace"`

**Pre-filter (index-level)**: the vector DB narrows the candidate set before running ANN. Correct but can hurt recall if the filtered subset is very small.

**Post-filter**: run ANN on the full index, then discard results that don't match. Simpler but wastes compute and can return fewer than k results.

**FAISS approach**: FAISS has no native metadata store — store metadata separately (dict, SQLite) and post-filter, or use `IDSelector` to pre-filter by ID.

**Chroma approach**: metadata is first-class; the `where` clause is applied inside the HNSW search as a pre-filter.

**pgvector approach**: add a SQL `WHERE` clause before the ORDER BY; the planner may use the vector index plus a bitmap index on the filter column.

### Filter operators (Chroma)

```python
# Equality
{"category": {"$eq": "legal"}}

# Comparison
{"score": {"$gte": 0.8}}

# Set membership
{"tag": {"$in": ["RAG", "LLM"]}}

# Boolean AND / OR
{"$and": [{"category": {"$eq": "legal"}}, {"year": {"$gte": 2023}}]}
```

### Persistence patterns

| Backend | Persist mechanism | Restore |
|---------|-----------------|---------|
| FAISS | `faiss.write_index(path)` | `faiss.read_index(path)` |
| Chroma | `PersistentClient(path)` | re-open same path |
| pgvector | Postgres WAL + PITR | standard Postgres backup |
| Qdrant | local disk or cloud | restart container |

### Scaling considerations

**Vertical scaling** (single node):
- More RAM → bigger flat index or higher M in HNSW
- More CPU cores → parallelise multiple queries (FAISS is not multithreaded per query by default; use `faiss.omp_set_num_threads(N)`)

**Horizontal scaling**:
- **Sharding**: partition vectors across nodes; a scatter-gather coordinator queries all shards and merges top-k. Used by Qdrant, Milvus, Weaviate.
- **Replication**: read replicas for query throughput; one primary for writes.

**Index rebuild strategy**:
- IVF centroids become stale if data distribution shifts. Rebuild the index periodically (nightly batch is common).
- HNSW supports incremental inserts; no rebuild needed unless quality degrades.

**Quantisation for scale**:
- Product Quantization (PQ): split d-dim vectors into `m` sub-vectors, quantise each with a 256-entry codebook. Storage drops from `d·4` bytes to `m` bytes per vector. 16–32× compression with ~2–5% recall loss.
- FAISS `IndexIVFPQ` combines IVF partitioning with PQ compression.

## How it works internally

### IDSelector for FAISS pre-filtering

```python
import faiss, numpy as np

# Keep only vectors whose IDs are in a specific set
selector = faiss.IDSelectorArray(np.array([0, 2, 4, 6], dtype=np.int64))
params = faiss.SearchParametersHNSW()
params.sel = selector
D, I = index.search_with_params(query, k, params)
```

This tells FAISS to skip vectors not in the set during graph traversal, emulating pre-filtering.

### Chroma's HNSW filter integration

Chroma passes the metadata predicate to `hnswlib` as a filter callback. During beam search, candidate nodes are accepted only if they pass the predicate. This means the beam may expand more before finding k valid results — internally it over-fetches.

## Worked example

```python
"""
day5_metadata_persist_scale.py — metadata filtering + persistence patterns.

Install: uv add chromadb faiss-cpu numpy
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

import chromadb
import faiss
import numpy as np
from chromadb.config import Settings

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RNG_SEED = int(os.getenv("RNG_SEED", "42"))
DIM = 64
N = 1_000


def make_corpus(n: int, d: int, seed: int) -> tuple[np.ndarray, list[str], list[dict]]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    faiss.normalize_L2(vecs)
    ids = [str(uuid.uuid4()) for _ in range(n)]
    meta = [
        {
            "source": f"file_{i % 10}.pdf",
            "year": 2020 + (i % 5),
            "category": "legal" if i % 3 == 0 else "technical",
        }
        for i in range(n)
    ]
    return vecs, ids, meta


# ── 1. Chroma metadata filtering ─────────────────────────────────────────


def demo_chroma_filter(vecs: np.ndarray, ids: list[str], meta: list[dict]) -> None:
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    col = client.create_collection("filter_demo", metadata={"hnsw:space": "cosine"})
    col.add(embeddings=vecs.tolist(), ids=ids, metadatas=meta)

    query = vecs[0].tolist()

    # --- simple equality filter
    r1 = col.query(
        query_embeddings=[query],
        n_results=5,
        where={"category": {"$eq": "legal"}},
        include=["metadatas"],
    )
    log.info("Filtered (legal) count: %d", len(r1["ids"][0]))
    for m in r1["metadatas"][0]:
        assert m["category"] == "legal", "Filter violated"

    # --- compound AND filter
    r2 = col.query(
        query_embeddings=[query],
        n_results=5,
        where={"$and": [{"category": {"$eq": "technical"}}, {"year": {"$gte": 2023}}]},
        include=["metadatas"],
    )
    log.info("Filtered (technical ∧ year≥2023) count: %d", len(r2["ids"][0]))


# ── 2. FAISS + external metadata + post-filter ────────────────────────────


def demo_faiss_postfilter(vecs: np.ndarray, meta: list[dict]) -> None:
    index = faiss.IndexIDMap(faiss.IndexFlatIP(DIM))
    ids_arr = np.arange(len(vecs), dtype=np.int64)
    index.add_with_ids(vecs, ids_arr)

    query = vecs[0:1]
    _D, I = index.search(query, 20)   # over-fetch

    # post-filter: keep only 'legal' category, return top 5
    legal_ids = [
        int(i) for i in I[0] if i >= 0 and meta[int(i)]["category"] == "legal"
    ][:5]
    log.info("FAISS post-filter (legal) top-5 ids: %s", legal_ids)


# ── 3. FAISS IDSelector pre-filter ────────────────────────────────────────


def demo_faiss_preselector(vecs: np.ndarray, meta: list[dict]) -> None:
    index = faiss.IndexHNSWFlat(DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)

    legal_set = np.array(
        [i for i, m in enumerate(meta) if m["category"] == "legal"], dtype=np.int64
    )
    selector = faiss.IDSelectorBatch(legal_set)
    params = faiss.SearchParametersHNSW()
    params.sel = selector

    query = vecs[0:1]
    _D, I = index.search_with_params(query, 5, params)
    log.info("FAISS pre-selector (legal) top-5 ids: %s", I[0].tolist())
    for i in I[0]:
        if i >= 0:
            assert meta[i]["category"] == "legal"


# ── 4. Persistence — round-trip FAISS index ────────────────────────────────


def demo_faiss_persist(vecs: np.ndarray) -> None:
    index = faiss.IndexHNSWFlat(DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index.add(vecs)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "index.faiss")
        faiss.write_index(index, path)
        size_kb = Path(path).stat().st_size / 1024
        log.info("Saved FAISS index: %.1f KB", size_kb)

        loaded = faiss.read_index(path)
        assert loaded.ntotal == index.ntotal
        _D, I = loaded.search(vecs[:1], 5)
        log.info("Round-trip OK, top-5 ids: %s", I[0].tolist())


# ── 5. OMP parallelism ────────────────────────────────────────────────────


def demo_omp_threads(vecs: np.ndarray) -> None:
    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    queries = vecs[:100]

    import time
    for n_threads in (1, 2, 4):
        faiss.omp_set_num_threads(n_threads)
        t0 = time.perf_counter()
        for _ in range(10):
            index.search(queries, 5)
        ms = (time.perf_counter() - t0) * 100   # per call
        log.info("threads=%d  avg_ms_per_batch=%.3f", n_threads, ms)


def main() -> None:
    vecs, ids, meta = make_corpus(N, DIM, RNG_SEED)

    log.info("=== Chroma metadata filter ===")
    demo_chroma_filter(vecs, ids, meta)

    log.info("=== FAISS post-filter ===")
    demo_faiss_postfilter(vecs, meta)

    log.info("=== FAISS IDSelector pre-filter ===")
    demo_faiss_preselector(vecs, meta)

    log.info("=== FAISS persist round-trip ===")
    demo_faiss_persist(vecs)

    log.info("=== OMP threads ===")
    demo_omp_threads(vecs)


if __name__ == "__main__":
    main()
```

Run: `uv run python day5_metadata_persist_scale.py`

## Exercises

1. Add a `"score"` float field to the metadata (random 0–1). Write a Chroma query that filters `score >= 0.7` and returns 10 results. Assert all returned metadata have `score >= 0.7`.
2. Implement a simple `MetadataStore` class backed by a Python dict that maps `int → dict`. Integrate it with a FAISS flat index to support `search_with_filter(query, k, predicate)` via post-filtering.
3. Extend `demo_faiss_persist` to also save and load the metadata dict alongside the FAISS index using Python's `json` module. Test round-trip correctness.
4. Research product quantization: with `DIM=128`, `m=16` sub-vectors, and an 8-bit codebook, what is the compression ratio vs float32? What is the expected recall\@10 degradation (consult the ANN benchmarks site)?
5. Write a pytest test verifying that `faiss.omp_set_num_threads(1)` and `faiss.omp_set_num_threads(4)` both return the same search results for the same query.

(Solutions: ../solutions/week-03/day-5.md)

## Self-check

- [ ] I can write Chroma `where` clauses with `$eq`, `$gte`, `$in`, `$and`, `$or`
- [ ] I can choose between pre-filter and post-filter and know the trade-offs
- [ ] I can persist a FAISS index to disk and restore it correctly
- [ ] I can explain horizontal sharding and when to use PQ compression

## Supplementary

- [Chroma filtering docs](https://docs.trychroma.com/guides/usage-guide#filtering-by-metadata)
- [FAISS IDSelector](https://faiss.ai/cpp_api/struct/faiss__IDSelector.html)
- [Product Quantization explained](https://www.pinecone.io/learn/product-quantization/)
- [pgvector filtering with indexes](https://github.com/pgvector/pgvector#filtering)
