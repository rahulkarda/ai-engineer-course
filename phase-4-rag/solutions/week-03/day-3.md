# Solutions — Week 03 · Day 3

## Exercise 1 — index_factory IVF64

```python
import faiss, numpy as np

DIM = 128
corpus = np.random.default_rng(42).standard_normal((5000, DIM)).astype(np.float32)
faiss.normalize_L2(corpus)
queries = corpus[:10].copy()

# Using index_factory
index = faiss.index_factory(DIM, "IVF64,Flat", faiss.METRIC_INNER_PRODUCT)
index.train(corpus)
index.add(corpus)
index.nprobe = 8
D, I = index.search(queries, 5)
print(I[:3])
```

Results match direct construction (`IndexIVFFlat(quantizer, DIM, 64, METRIC_INNER_PRODUCT)`) up to tie-breaking.

## Exercise 2 — Batch add then remove_ids

```python
index = faiss.IndexFlatIP(DIM)
batch1 = corpus[:100].copy()
batch2 = corpus[100:200].copy()

index.add(batch1)
assert index.ntotal == 100
index.add(batch2)
assert index.ntotal == 200

# remove_ids requires an IDSelectorArray with the int64 IDs to remove
# For IndexFlat without IDMap, IDs are sequential 0..ntotal-1
# Use IDMap to test arbitrary removal:
base = faiss.IndexFlatIP(DIM)
id_map = faiss.IndexIDMap(base)
ids1 = np.arange(0, 100, dtype=np.int64)
ids2 = np.arange(100, 200, dtype=np.int64)
id_map.add_with_ids(batch1, ids1)
id_map.add_with_ids(batch2, ids2)
assert id_map.ntotal == 200
id_map.remove_ids(ids1)
assert id_map.ntotal == 100
```

## Exercise 3 — M=16 vs M=64 RAM and recall

```python
import faiss, numpy as np, sys

DIM = 128
N = 10_000
corpus = np.random.default_rng(42).standard_normal((N, DIM)).astype(np.float32)
faiss.normalize_L2(corpus)
queries = corpus[:50].copy()

for M in (16, 64):
    idx = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT)
    idx.add(corpus)
    # Rough RAM: N * M * 4 * 2 bytes (bidirectional, single layer approx)
    ram_est_mb = N * M * 4 * 2 / 1024**2
    D, I = idx.search(queries, 10)
    print(f"M={M}: est RAM={ram_est_mb:.2f} MB")
```

M=64 uses ~4× more graph RAM but achieves higher recall\@10, especially at low efSearch.

## Exercise 4 — IDMap with custom IDs

```python
import hashlib, numpy as np, faiss

N = 500
DIM = 128
corpus = np.random.default_rng(0).standard_normal((N, DIM)).astype(np.float32)
faiss.normalize_L2(corpus)

# Custom IDs: large integers derived from a hash
custom_ids = np.array(
    [int(hashlib.md5(f"doc_{i}".encode()).hexdigest(), 16) % (2**62) for i in range(N)],
    dtype=np.int64,
)

base = faiss.IndexHNSWFlat(DIM, 16, faiss.METRIC_INNER_PRODUCT)
idx = faiss.IndexIDMap(base)
idx.add_with_ids(corpus, custom_ids)

D, I = idx.search(corpus[:3], 3)
# Verify returned IDs are from our custom set
custom_set = set(custom_ids.tolist())
assert all(int(i) in custom_set for row in I for i in row if i >= 0)
print("Custom IDs verified:", I[0].tolist())
```

## Exercise 5 — Serialize to bytes round-trip

```python
import faiss, numpy as np, pytest

DIM = 64
N = 200

@pytest.fixture
def flat_index():
    corpus = np.random.default_rng(7).standard_normal((N, DIM)).astype(np.float32)
    idx = faiss.IndexFlatIP(DIM)
    idx.add(corpus)
    return idx, corpus

def test_bytes_roundtrip(flat_index):
    idx, corpus = flat_index
    original_total = idx.ntotal

    serialized = faiss.serialize_index(idx)     # returns np.ndarray (uint8)
    loaded = faiss.deserialize_index(serialized)

    assert loaded.ntotal == original_total

    q = corpus[:5]
    D1, I1 = idx.search(q, 3)
    D2, I2 = loaded.search(q, 3)
    np.testing.assert_array_equal(I1, I2)
    np.testing.assert_allclose(D1, D2, atol=1e-5)
```
