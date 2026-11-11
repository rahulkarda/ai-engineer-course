# Solutions — Week 03 · Day 5

## Exercise 1 — Chroma score filter

```python
import chromadb, numpy as np, uuid
from chromadb.config import Settings

DIM = 64
N = 500
rng = np.random.default_rng(42)
vecs = rng.standard_normal((N, DIM)).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

client = chromadb.Client(Settings(anonymized_telemetry=False))
col = client.get_or_create_collection("score_demo", metadata={"hnsw:space": "cosine"})

ids = [str(uuid.uuid4()) for _ in range(N)]
meta = [{"score": float(rng.uniform(0, 1))} for _ in range(N)]
col.add(embeddings=vecs.tolist(), ids=ids, metadatas=meta)

query = vecs[0].tolist()
results = col.query(
    query_embeddings=[query],
    n_results=10,
    where={"score": {"$gte": 0.7}},
    include=["metadatas"],
)
for m in results["metadatas"][0]:
    assert m["score"] >= 0.7, f"Filter violated: {m['score']}"
print(f"Returned {len(results['ids'][0])} results, all score >= 0.7")
```

## Exercise 2 — MetadataStore + FAISS post-filter

```python
from __future__ import annotations
from typing import Any, Callable
import numpy as np
import faiss


class MetadataStore:
    def __init__(self) -> None:
        self._data: dict[int, dict[str, Any]] = {}

    def set(self, int_id: int, metadata: dict[str, Any]) -> None:
        self._data[int_id] = metadata

    def get(self, int_id: int) -> dict[str, Any]:
        return self._data.get(int_id, {})


class FAISSWithMetadata:
    def __init__(self, dim: int) -> None:
        self._index = faiss.IndexFlatIP(dim)
        self._meta = MetadataStore()
        self._count = 0

    def add(self, vecs: np.ndarray, metadata: list[dict]) -> None:
        start = self._count
        self._index.add(vecs)
        for i, m in enumerate(metadata):
            self._meta.set(start + i, m)
        self._count += len(vecs)

    def search_with_filter(
        self,
        query: np.ndarray,
        k: int,
        predicate: Callable[[dict], bool],
        overfetch: int = 20,
    ) -> list[tuple[int, float]]:
        fetch = min(overfetch, self._index.ntotal)
        scores, ids = self._index.search(query.reshape(1, -1), fetch)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0 and predicate(self._meta.get(int(idx))):
                results.append((int(idx), float(score)))
            if len(results) == k:
                break
        return results
```

## Exercise 3 — FAISS persist + metadata JSON

```python
import faiss, json, numpy as np, tempfile
from pathlib import Path

DIM = 64
N = 100
corpus = np.random.default_rng(0).standard_normal((N, DIM)).astype(np.float32)
faiss.normalize_L2(corpus)
metadata = {i: {"label": f"item_{i}"} for i in range(N)}

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    faiss.write_index(faiss.IndexFlatIP(DIM), str(tmp_path / "index.faiss"))
    # Re-add to fresh index
    idx = faiss.IndexFlatIP(DIM)
    idx.add(corpus)
    faiss.write_index(idx, str(tmp_path / "index.faiss"))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    # Load
    loaded_idx = faiss.read_index(str(tmp_path / "index.faiss"))
    loaded_meta = json.loads((tmp_path / "metadata.json").read_text())

    assert loaded_idx.ntotal == N
    assert loaded_meta["0"]["label"] == "item_0"
    print("Round-trip OK")
```

## Exercise 4 — PQ compression analysis

```
DIM=128, m=16, k*=256 (8-bit codebook):
Sub-vector size = 128 / 16 = 8 floats per sub-vector
Original storage per vector: 128 × 4 = 512 bytes
PQ storage per vector: 16 × 1 (8-bit code) = 16 bytes
Compression ratio: 512 / 16 = 32×

Expected recall@10 degradation (from ANN benchmarks, SIFT-128 PQ16):
Typically 3–8% recall loss at default nprobe settings.
With nprobe tuning, often < 2% loss. Trade-off is very favourable for RAM-limited deployments.
```

Reference: https://ann-benchmarks.com — search for `sift-128-euclidean` with `IndexIVFPQ`.

## Exercise 5 — Thread-safety of search results

```python
import faiss, numpy as np, pytest

DIM = 64
N = 500
corpus = np.random.default_rng(1).standard_normal((N, DIM)).astype(np.float32)
faiss.normalize_L2(corpus)
query = corpus[:5].copy()

@pytest.fixture(scope="module")
def index():
    idx = faiss.IndexFlatIP(DIM)
    idx.add(corpus)
    return idx

def test_single_thread_results(index):
    faiss.omp_set_num_threads(1)
    D1, I1 = index.search(query, 5)
    faiss.omp_set_num_threads(4)
    D2, I2 = index.search(query, 5)
    np.testing.assert_array_equal(I1, I2)
    np.testing.assert_allclose(D1, D2, atol=1e-5)
```
