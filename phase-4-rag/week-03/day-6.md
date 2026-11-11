# Phase 4 · Week 03 · Day 6 (Saturday) — Build: Vector Store Abstraction

> Objective: Build a production-ready vector store abstraction with pluggable backends (in-memory, FAISS, Chroma), supporting upsert/query/filter/persist, with full test coverage.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]]–[[day-5]]

## Goal

By end of today you will have:
- A `VectorStore` abstract base class (ABC) with a clean interface
- Three concrete backends: `InMemoryVectorStore`, `FAISSVectorStore`, `ChromaVectorStore`
- Shared test suite exercising all backends via parametrize
- CLI entry point to demonstrate all backends in one run

The design prioritises **correctness over performance** — this is a learning artifact, not a FAISS wrapper library.

## Architecture

```
src/
  vector_store/
    __init__.py
    base.py          ← ABC + shared types
    in_memory.py     ← numpy brute-force backend
    faiss_backend.py ← FAISS IVFFlat + HNSW backend
    chroma_backend.py← Chroma in-memory backend
    factory.py       ← create_store(backend, **kwargs) factory
tests/
  test_vector_store.py  ← pytest, parametrized over all backends
main.py                 ← CLI demo
```

## Step 1 — Base types and ABC (30 min)

```python
# src/vector_store/base.py
"""Shared types and VectorStore ABC."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    """A single stored record."""
    id: str
    vector: np.ndarray          # float32, shape (d,)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)


@dataclass
class SearchResult:
    id: str
    score: float                # cosine similarity in [−1, 1]
    metadata: dict[str, Any]


class VectorStore(abc.ABC):
    """Abstract vector store. All backends must implement these methods."""

    @abc.abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """Add or replace records by ID."""

    @abc.abstractmethod
    def query(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return top-k most similar records, optionally filtered by metadata."""

    @abc.abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Remove records by ID."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return total number of stored vectors."""

    @abc.abstractmethod
    def persist(self, path: Path) -> None:
        """Save index and metadata to disk."""

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        """Load index and metadata from disk (replaces current state)."""

    # ── Convenience ──────────────────────────────────────────────────────

    def upsert_one(self, record: VectorRecord) -> None:
        self.upsert([record])
```

## Step 2 — In-memory backend (30 min)

```python
# src/vector_store/in_memory.py
"""Brute-force in-memory backend using numpy."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import SearchResult, VectorRecord, VectorStore

log = logging.getLogger(__name__)


class InMemoryVectorStore(VectorStore):
    """Pure-numpy cosine similarity store. Exact search, no training required."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, records: list[VectorRecord]) -> None:
        for r in records:
            if r.vector.shape != (self.dim,):
                raise ValueError(f"Expected dim {self.dim}, got {r.vector.shape}")
            self._records[r.id] = r
        log.debug("upserted %d records, total=%d", len(records), self.count())

    def query(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        candidates = list(self._records.values())
        if filter:
            candidates = [r for r in candidates if _match(r.metadata, filter)]
        if not candidates:
            return []

        mat = np.stack([r.vector for r in candidates])  # (m, d)
        q = vector.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        scores = (mat / norms) @ q_norm

        top_k = min(k, len(candidates))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(-scores[idx])]
        return [
            SearchResult(id=candidates[i].id, score=float(scores[i]), metadata=candidates[i].metadata)
            for i in idx
        ]

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self._records.pop(id_, None)

    def count(self) -> int:
        return len(self._records)

    def persist(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "records": {
                id_: {"vector": r.vector.tolist(), "metadata": r.metadata}
                for id_, r in self._records.items()
            },
        }
        (path / "store.json").write_text(json.dumps(data))
        log.info("Persisted %d records to %s", self.count(), path)

    def load(self, path: Path) -> None:
        path = Path(path)
        data = json.loads((path / "store.json").read_text())
        self.dim = data["dim"]
        self._records = {
            id_: VectorRecord(
                id=id_,
                vector=np.array(v["vector"], dtype=np.float32),
                metadata=v["metadata"],
            )
            for id_, v in data["records"].items()
        }
        log.info("Loaded %d records from %s", self.count(), path)


def _match(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
    """Simple equality filter: all key-value pairs must match."""
    return all(metadata.get(k) == v for k, v in filter.items())
```

## Step 3 — FAISS backend (45 min)

```python
# src/vector_store/faiss_backend.py
"""FAISS-backed vector store with IndexIDMap + HNSWFlat."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .base import SearchResult, VectorRecord, VectorStore

log = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS HNSWFlat index wrapped in IndexIDMap so caller-supplied string IDs
    are preserved via a str→int64 mapping table.
    """

    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200) -> None:
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._next_int: int = 0
        self._build_index()

    def _build_index(self) -> None:
        base = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = self.ef_construction
        self._index = faiss.IndexIDMap(base)

    def upsert(self, records: list[VectorRecord]) -> None:
        new_ids: list[str] = []
        new_vecs: list[np.ndarray] = []
        for r in records:
            vec = r.vector.astype(np.float32).copy()
            faiss.normalize_L2(vec.reshape(1, -1))
            if r.id not in self._id_to_int:
                int_id = self._next_int
                self._next_int += 1
                self._id_to_int[r.id] = int_id
                self._int_to_id[int_id] = r.id
                new_ids.append(r.id)
                new_vecs.append(vec)
            else:
                # Update metadata only (HNSW doesn't support in-place update)
                log.debug("upsert %s: updating metadata only (vector unchanged)", r.id)
            self._metadata[r.id] = r.metadata

        if new_vecs:
            mat = np.stack(new_vecs)
            ints = np.array(
                [self._id_to_int[id_] for id_ in new_ids], dtype=np.int64
            )
            self._index.add_with_ids(mat, ints)
            log.debug("added %d new vectors, total=%d", len(new_vecs), self.count())

    def query(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._index.ntotal == 0:
            return []
        q = vector.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(q)
        fetch_k = min(k * 10 if filter else k, self._index.ntotal)
        scores, int_ids = self._index.search(q, fetch_k)
        results: list[SearchResult] = []
        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id < 0:
                continue
            str_id = self._int_to_id[int(int_id)]
            meta = self._metadata.get(str_id, {})
            if filter and not all(meta.get(fk) == fv for fk, fv in filter.items()):
                continue
            results.append(SearchResult(id=str_id, score=float(score), metadata=meta))
            if len(results) == k:
                break
        return results

    def delete(self, ids: list[str]) -> None:
        int_ids = np.array(
            [self._id_to_int[id_] for id_ in ids if id_ in self._id_to_int],
            dtype=np.int64,
        )
        if len(int_ids):
            self._index.remove_ids(int_ids)
        for id_ in ids:
            int_id = self._id_to_int.pop(id_, None)
            if int_id is not None:
                self._int_to_id.pop(int_id, None)
            self._metadata.pop(id_, None)

    def count(self) -> int:
        return self._index.ntotal

    def persist(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        meta = {
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "next_int": self._next_int,
            "id_to_int": self._id_to_int,
            "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
            "metadata": self._metadata,
        }
        (path / "meta.json").write_text(json.dumps(meta))
        log.info("Persisted FAISS index (%d vectors) to %s", self.count(), path)

    def load(self, path: Path) -> None:
        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))
        meta = json.loads((path / "meta.json").read_text())
        self.dim = meta["dim"]
        self.m = meta["m"]
        self.ef_construction = meta["ef_construction"]
        self._next_int = meta["next_int"]
        self._id_to_int = meta["id_to_int"]
        self._int_to_id = {int(k): v for k, v in meta["int_to_id"].items()}
        self._metadata = meta["metadata"]
        log.info("Loaded FAISS index (%d vectors) from %s", self.count(), path)
```

## Step 4 — Chroma backend (30 min)

```python
# src/vector_store/chroma_backend.py
"""Chroma in-process backend."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

from .base import SearchResult, VectorRecord, VectorStore

log = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Chroma-backed store. Uses in-memory client; call persist() to write to disk."""

    def __init__(self, dim: int, collection_name: str = "default") -> None:
        self.dim = dim
        self.collection_name = collection_name
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._col = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, records: list[VectorRecord]) -> None:
        self._col.upsert(
            embeddings=[r.vector.tolist() for r in records],
            ids=[r.id for r in records],
            metadatas=[r.metadata for r in records],
        )
        log.debug("upserted %d records, total=%d", len(records), self.count())

    def query(
        self,
        vector: np.ndarray,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._col.count() == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector.tolist()],
            "n_results": min(k, self._col.count()),
            "include": ["distances", "metadatas"],
        }
        if filter:
            # Convert simple {k: v} to Chroma's $eq format
            if len(filter) == 1:
                key, val = next(iter(filter.items()))
                kwargs["where"] = {key: {"$eq": val}}
            else:
                kwargs["where"] = {
                    "$and": [{k: {"$eq": v}} for k, v in filter.items()]
                }
        res = self._col.query(**kwargs)
        results: list[SearchResult] = []
        for id_, dist, meta in zip(
            res["ids"][0], res["distances"][0], res["metadatas"][0]
        ):
            results.append(SearchResult(id=id_, score=1.0 - dist, metadata=meta))
        return results

    def delete(self, ids: list[str]) -> None:
        self._col.delete(ids=ids)

    def count(self) -> int:
        return self._col.count()

    def persist(self, path: Path) -> None:
        """Pickle the internal state (for portability in this learning context)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "dim": self.dim,
            "collection_name": self.collection_name,
            "all_data": self._col.get(include=["embeddings", "metadatas"]),
        }
        (path / "chroma_state.pkl").write_bytes(pickle.dumps(state))
        log.info("Persisted Chroma collection (%d) to %s", self.count(), path)

    def load(self, path: Path) -> None:
        path = Path(path)
        state = pickle.loads((path / "chroma_state.pkl").read_bytes())  # noqa: S301
        self.dim = state["dim"]
        self.collection_name = state["collection_name"]
        data = state["all_data"]
        if data["ids"]:
            self._col.upsert(
                embeddings=data["embeddings"],
                ids=data["ids"],
                metadatas=data["metadatas"],
            )
        log.info("Loaded Chroma collection (%d) from %s", self.count(), path)
```

## Step 5 — Factory (15 min)

```python
# src/vector_store/factory.py
"""Factory for creating VectorStore instances by backend name."""

from __future__ import annotations

from typing import Any

from .base import VectorStore
from .chroma_backend import ChromaVectorStore
from .faiss_backend import FAISSVectorStore
from .in_memory import InMemoryVectorStore

_BACKENDS: dict[str, type[VectorStore]] = {
    "memory": InMemoryVectorStore,
    "faiss": FAISSVectorStore,
    "chroma": ChromaVectorStore,
}


def create_store(backend: str, **kwargs: Any) -> VectorStore:
    """
    Create a VectorStore for the named backend.

    Args:
        backend: one of "memory", "faiss", "chroma"
        **kwargs: passed to the backend constructor (must include `dim`)
    """
    klass = _BACKENDS.get(backend)
    if klass is None:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(_BACKENDS)}")
    return klass(**kwargs)
```

## Step 6 — Test suite (45 min)

```python
# tests/test_vector_store.py
"""Parametrized tests for all VectorStore backends."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.vector_store.base import VectorRecord, VectorStore
from src.vector_store.chroma_backend import ChromaVectorStore
from src.vector_store.faiss_backend import FAISSVectorStore
from src.vector_store.factory import create_store
from src.vector_store.in_memory import InMemoryVectorStore

DIM = 32
RNG = np.random.default_rng(42)


def make_record(id_: str, seed: int | None = None, **meta) -> VectorRecord:
    rng = np.random.default_rng(seed or hash(id_) % 2**32)
    vec = rng.standard_normal(DIM).astype(np.float32)
    return VectorRecord(id=id_, vector=vec, metadata=meta)


@pytest.fixture(
    params=["memory", "faiss", "chroma"],
    ids=["memory", "faiss", "chroma"],
)
def store(request: pytest.FixtureRequest) -> VectorStore:
    return create_store(request.param, dim=DIM)


class TestUpsertQuery:
    def test_upsert_increases_count(self, store: VectorStore) -> None:
        store.upsert([make_record("a"), make_record("b")])
        assert store.count() == 2

    def test_query_returns_k_results(self, store: VectorStore) -> None:
        records = [make_record(f"r{i}", seed=i) for i in range(20)]
        store.upsert(records)
        query = RNG.standard_normal(DIM).astype(np.float32)
        results = store.query(query, k=5)
        assert len(results) == 5

    def test_query_result_ids_are_in_store(self, store: VectorStore) -> None:
        records = [make_record(f"id{i}", seed=i) for i in range(10)]
        store.upsert(records)
        valid_ids = {r.id for r in records}
        results = store.query(records[0].vector, k=5)
        assert all(r.id in valid_ids for r in results)

    def test_self_query_is_top_result(self, store: VectorStore) -> None:
        records = [make_record(f"v{i}", seed=i) for i in range(50)]
        store.upsert(records)
        target = records[0]
        results = store.query(target.vector, k=1)
        assert results[0].id == target.id

    def test_upsert_idempotent(self, store: VectorStore) -> None:
        r = make_record("dup")
        store.upsert([r, r])
        assert store.count() == 1

    def test_upsert_updates_metadata(self, store: VectorStore) -> None:
        r = make_record("upd", category="old")
        store.upsert([r])
        r2 = VectorRecord(id="upd", vector=r.vector, metadata={"category": "new"})
        store.upsert([r2])
        results = store.query(r.vector, k=1)
        assert results[0].metadata.get("category") == "new"


class TestDelete:
    def test_delete_reduces_count(self, store: VectorStore) -> None:
        store.upsert([make_record("x"), make_record("y")])
        store.delete(["x"])
        assert store.count() == 1

    def test_delete_nonexistent_is_noop(self, store: VectorStore) -> None:
        store.upsert([make_record("z")])
        store.delete(["no_such_id"])
        assert store.count() == 1


class TestFilter:
    def test_filter_equality(self, store: VectorStore) -> None:
        store.upsert([
            make_record("a1", category="A"),
            make_record("b1", category="B"),
            make_record("a2", category="A"),
        ])
        query = RNG.standard_normal(DIM).astype(np.float32)
        results = store.query(query, k=5, filter={"category": "A"})
        assert all(r.metadata["category"] == "A" for r in results)
        assert len(results) <= 2

    def test_filter_empty_result(self, store: VectorStore) -> None:
        store.upsert([make_record("x", category="X")])
        query = RNG.standard_normal(DIM).astype(np.float32)
        results = store.query(query, k=5, filter={"category": "NONE"})
        assert results == []


class TestPersist:
    def test_persist_load_round_trip(self, store: VectorStore) -> None:
        records = [make_record(f"p{i}", seed=i) for i in range(10)]
        store.upsert(records)
        query = records[0].vector
        original_results = store.query(query, k=5)

        with tempfile.TemporaryDirectory() as tmp:
            store.persist(Path(tmp))
            store.load(Path(tmp))
            loaded_results = store.query(query, k=5)

        orig_ids = [r.id for r in original_results]
        loaded_ids = [r.id for r in loaded_results]
        assert orig_ids == loaded_ids


class TestFactory:
    @pytest.mark.parametrize("backend", ["memory", "faiss", "chroma"])
    def test_factory_creates_correct_type(self, backend: str) -> None:
        store = create_store(backend, dim=DIM)
        assert isinstance(store, VectorStore)

    def test_factory_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_store("badbackend", dim=DIM)
```

## Step 7 — CLI demo (15 min)

```python
# main.py
"""CLI demo: exercise all three backends."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np

from src.vector_store.base import VectorRecord
from src.vector_store.factory import create_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DIM = int(os.getenv("DEMO_DIM", "64"))
N = int(os.getenv("DEMO_N", "200"))


def run_demo(backend: str) -> None:
    log.info("=== %s backend ===", backend.upper())
    store = create_store(backend, dim=DIM)

    rng = np.random.default_rng(42)
    records = [
        VectorRecord(
            id=str(uuid.uuid4()),
            vector=rng.standard_normal(DIM).astype(np.float32),
            metadata={"category": "A" if i % 2 == 0 else "B", "index": i},
        )
        for i in range(N)
    ]
    store.upsert(records)
    log.info("Inserted %d vectors", store.count())

    query = rng.standard_normal(DIM).astype(np.float32)
    results = store.query(query, k=5)
    log.info("Top-5 ids: %s", [r.id[:8] for r in results])
    log.info("Top-5 scores: %s", [f"{r.score:.4f}" for r in results])

    filtered = store.query(query, k=5, filter={"category": "A"})
    log.info("Filtered (A) count: %d", len(filtered))

    store.delete([records[0].id])
    log.info("After delete, count=%d", store.count())

    with tempfile.TemporaryDirectory() as tmp:
        store.persist(Path(tmp))
        store.load(Path(tmp))
        log.info("After persist+load, count=%d", store.count())


def main() -> None:
    for backend in ("memory", "faiss", "chroma"):
        run_demo(backend)


if __name__ == "__main__":
    main()
```

## Running the project

```bash
# Install deps
uv add faiss-cpu chromadb numpy pytest

# Run tests
uv run pytest tests/ -v

# Run CLI demo
uv run python main.py
```

## Exercises

1. Add a `batch_query(vectors, k)` method to the ABC and implement it in all three backends. Ensure the FAISS backend sends all queries in a single `index.search()` call.
2. Add a `count_by_filter(filter)` method to `InMemoryVectorStore` that counts records matching a filter without doing a vector search.
3. Extend `FAISSVectorStore.persist` to also save the HNSW `efSearch` setting, so it is restored on `load`.
4. Write a pytest fixture `large_store` that inserts 5000 records into each backend and measure the `query` latency with `time.perf_counter`. Assert all backends return within 500 ms for k=10.
5. Add a `Qdrant` stub backend class that raises `NotImplementedError` for all methods and register it in the factory. Write a test that confirms calling `upsert` raises `NotImplementedError`.

(Solutions: ../solutions/week-03/day-6.md)

## Self-check

- [ ] I built a working ABC with three concrete backends
- [ ] My test suite is parametrized and covers upsert, query, filter, delete, persist
- [ ] I understand why HNSW doesn't support in-place vector updates
- [ ] I can extend the factory with a new backend without modifying existing code
