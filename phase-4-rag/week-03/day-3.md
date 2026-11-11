# Phase 4 · Week 03 · Day 3 — FAISS Hands-On

> Objective: Build, query, save, and load FAISS indexes using faiss-cpu; understand index types in code.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

FAISS (Facebook AI Similarity Search) is a C++ library with a Python wrapper that implements dozens of index types efficiently on CPU and GPU. It is the standard reference implementation for vector search benchmarks.

Key facts:
- Operates on **float32** arrays exclusively (convert other dtypes before use)
- All vectors must have the same dimension `d`
- IDs are 64-bit integers; by default FAISS uses sequential 0-based IDs — use `IndexIDMap` to attach your own IDs
- FAISS is **not** a full database: no metadata, no persistence by default, no server mode

### FAISS index hierarchy (common types)

| Class | Description |
|-------|-------------|
| `IndexFlatL2` | Exact, L2 distance |
| `IndexFlatIP` | Exact, inner product |
| `IndexIVFFlat` | IVF + flat storage, L2 or IP |
| `IndexHNSWFlat` | HNSW + flat storage |
| `IndexIVFPQ` | IVF + product quantization (compressed) |
| `IndexIDMap` | Wrapper: maps arbitrary int64 IDs to any base index |

### Index factory string syntax

```python
import faiss
index = faiss.index_factory(d, "IVF256,Flat", faiss.METRIC_INNER_PRODUCT)
```

Common factory strings: `"Flat"`, `"IVF256,Flat"`, `"HNSW32"`, `"IVF256,PQ16"`.

### Lifecycle

```
create → (train if needed) → add → search → (serialize → deserialize)
```

- `IndexFlatL2/IP` and `IndexHNSWFlat` never need training
- `IndexIVFFlat` and any IVF variant require `index.train(training_vectors)` before `index.add()`
- Training is needed only once; you can add more vectors after training

## How it works internally

### Memory layout

FAISS stores vectors in a raw contiguous `float32` C array. `add(xb)` copies numpy data into this array. For `IndexHNSWFlat`, the graph adjacency lists are stored alongside. There is no copy-on-write; all mutations are in-place.

### Serialisation

`faiss.write_index(index, path)` serialises to a binary format that includes the index type, dimension, all vectors, and any trained parameters (centroids for IVF). `faiss.read_index(path)` fully reconstructs the index — no re-training needed.

### Thread safety

`search` is thread-safe (read-only). `add` and `train` are not thread-safe; wrap with a lock if using multiple writer threads.

## Worked example

```python
"""
day3_faiss_hands_on.py — FAISS IndexFlatIP, IndexIVFFlat, IndexHNSWFlat demo.

Install: uv add faiss-cpu numpy
"""

import logging
import os
import tempfile
from pathlib import Path

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

RNG_SEED = int(os.getenv("RNG_SEED", "42"))
N = 20_000
DIM = 128
K = 5
N_TRAIN = 10_000   # for IVF training


def make_data(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    faiss.normalize_L2(vecs)   # in-place L2 normalisation
    return vecs


def recall(exact_ids: np.ndarray, approx_ids: np.ndarray) -> float:
    hits = sum(
        len(set(e) & set(a)) for e, a in zip(exact_ids.tolist(), approx_ids.tolist())
    )
    return hits / exact_ids.size


# ── 1. IndexFlatIP — exact, inner product ──────────────────────────────────

def demo_flat(corpus: np.ndarray, queries: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatIP(DIM)
    index.add(corpus)
    log.info("FlatIP ntotal=%d", index.ntotal)
    _D, I = index.search(queries, K)
    return I   # shape (n_queries, K)


# ── 2. IndexIVFFlat ────────────────────────────────────────────────────────

def demo_ivf(
    corpus: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    nlist = 128
    quantizer = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    assert not index.is_trained

    train_data = corpus[:N_TRAIN]
    index.train(train_data)
    assert index.is_trained

    index.add(corpus)
    log.info("IVFFlat ntotal=%d nlist=%d", index.ntotal, nlist)

    for nprobe in (1, 4, 16, 32, 64):
        index.nprobe = nprobe
        _D, I = index.search(queries, K)
        log.info("  nprobe=%3d  recall@%d=%.3f", nprobe, K, recall(ground_truth, I))


# ── 3. IndexHNSWFlat ───────────────────────────────────────────────────────

def demo_hnsw(
    corpus: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    M = 32
    index = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(corpus)   # no training needed
    log.info("HNSW ntotal=%d M=%d", index.ntotal, M)

    for ef in (10, 32, 64, 128, 256):
        index.hnsw.efSearch = ef
        _D, I = index.search(queries, K)
        log.info("  efSearch=%3d  recall@%d=%.3f", ef, K, recall(ground_truth, I))


# ── 4. IndexIDMap — custom string → int64 mapping ─────────────────────────

def demo_idmap(corpus: np.ndarray) -> None:
    base_index = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIDMap(base_index)

    # Assign IDs 1000..1000+N
    ids = np.arange(1000, 1000 + len(corpus), dtype=np.int64)
    index.add_with_ids(corpus, ids)
    log.info("IDMap ntotal=%d", index.ntotal)

    _D, I = index.search(corpus[:3], 3)
    log.info("IDMap result ids (first query): %s", I[0].tolist())
    assert all(i >= 1000 for i in I[0].tolist()), "IDs should be ≥ 1000"


# ── 5. Serialisation ───────────────────────────────────────────────────────

def demo_serialize(corpus: np.ndarray, queries: np.ndarray) -> None:
    index = faiss.IndexHNSWFlat(DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index.add(corpus[:1000])

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "hnsw.index")
        faiss.write_index(index, path)
        log.info("Saved index to %s (%.1f KB)", path, Path(path).stat().st_size / 1024)

        loaded = faiss.read_index(path)
        _D, I = loaded.search(queries[:2], K)
        log.info("Loaded index results (first query): %s", I[0].tolist())


def main() -> None:
    corpus = make_data(N, DIM, RNG_SEED)
    queries = make_data(50, DIM, RNG_SEED + 1)

    log.info("=== Flat (exact) ===")
    ground_truth = demo_flat(corpus, queries)

    log.info("=== IVF ===")
    demo_ivf(corpus, queries, ground_truth)

    log.info("=== HNSW ===")
    demo_hnsw(corpus, queries, ground_truth)

    log.info("=== IDMap ===")
    demo_idmap(corpus)

    log.info("=== Serialise / load ===")
    demo_serialize(corpus, queries)


if __name__ == "__main__":
    main()
```

Run: `uv run python day3_faiss_hands_on.py`

## Exercises

1. Create an `IndexIVFFlat` with `nlist=64` using the `index_factory` string `"IVF64,Flat"`. Train and query it; verify results match the direct-construction approach.
2. Add vectors to a `FlatIP` index in two batches. Verify `ntotal` doubles after the second add. Then remove the first batch using `remove_ids` and confirm `ntotal` halves.
3. Build an `IndexHNSWFlat` with `M=16` and one with `M=64`. Compare RAM usage (`index.ntotal × M × 4 × 2` bytes estimate) and recall\@10 for the same query set.
4. Wrap an `IndexHNSWFlat` in `IndexIDMap`. Store 500 vectors with IDs that are your own hash-like integers (e.g., `hash(f"doc_{i}") % (2**62)`). Query and confirm the returned IDs match your scheme.
5. Write a pytest test that serialises a Flat index, deletes it from disk, reads it back from a bytes buffer using `faiss.deserialize_index(faiss.serialize_index(index))`, and asserts `ntotal` is preserved.

(Solutions: ../solutions/week-03/day-3.md)

## Self-check

- [ ] I can build and query `IndexFlatIP`, `IndexIVFFlat`, and `IndexHNSWFlat` from scratch
- [ ] I understand when `train()` is required and what happens if you skip it
- [ ] I can attach custom int64 IDs using `IndexIDMap`
- [ ] I can serialise and deserialise a FAISS index to/from disk

## Supplementary

- [FAISS Python API reference](https://faiss.ai/python_api/index.html)
- [FAISS — Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS guidelines for choosing an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)
