# Solutions — Week 03 · Day 4

## Exercise 1 — Chroma upsert (update metadata)

```python
import chromadb, numpy as np, uuid
from chromadb.config import Settings

DIM = 64
rng = np.random.default_rng(42)
vec = rng.standard_normal(DIM).astype(np.float32).tolist()
item_id = str(uuid.uuid4())

client = chromadb.Client(Settings(anonymized_telemetry=False))
col = client.create_collection("upsert_test", metadata={"hnsw:space": "cosine"})

col.add(embeddings=[vec], ids=[item_id], metadatas=[{"category": "original"}])
results = col.get(ids=[item_id], include=["metadatas"])
assert results["metadatas"][0]["category"] == "original"

# Upsert to update metadata
col.upsert(embeddings=[vec], ids=[item_id], metadatas=[{"category": "updated"}])
results2 = col.get(ids=[item_id], include=["metadatas"])
assert results2["metadatas"][0]["category"] == "updated", "Metadata not updated"
print("Upsert update: OK")
```

## Exercise 2 — Delete and count

```python
col.add(
    embeddings=[vec],
    ids=["to_delete"],
    metadatas=[{"x": 1}],
)
before = col.count()
col.delete(ids=["to_delete"])
after = col.count()
assert after == before - 1
print(f"Count before={before}, after={after}")
```

## Exercise 3 — Compound AND filter

```python
# Assuming collection already has metadata with 'category' and 'source' fields
results = col.query(
    query_embeddings=[vec],
    n_results=10,
    where={
        "$and": [
            {"category": {"$eq": "science"}},
            {"source": {"$ne": "doc_0"}},
        ]
    },
    include=["metadatas"],
)
print(f"Results matching (science AND source != doc_0): {len(results['ids'][0])}")
# Result count depends on corpus composition:
# science docs = N/2 = 250 (every even index), minus those with source=doc_0 (just 1)
# → approximately 249 candidates, query returns up to 10
```

## Exercise 4 — pgvector ivfflat index

```sql
-- Create ivfflat index (cosine distance)
CREATE INDEX ON embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

-- After bulk load, VACUUM ANALYZE updates table statistics so the
-- planner knows to use the index rather than seqscan:
VACUUM ANALYZE embeddings;

-- At query time, tune probes:
SET ivfflat.probes = 5;
```

`VACUUM ANALYZE` is needed after bulk INSERT because ivfflat requires accurate page-count statistics to decide whether an index scan is cheaper than a sequential scan. Without it, the planner may do a seqscan even when the index exists.

## Exercise 5 — Chroma vs pgvector comparison

**Chroma** is an embedded Python library ideal for prototyping: zero infrastructure, first-class metadata filtering, and automatic HNSW indexing. Persistent mode stores data as local files. Not designed for multi-server deployments or strict ACID guarantees.

**pgvector** adds vector search to an existing Postgres instance, inheriting ACID transactions, row-level security, connection pooling, and the ability to JOIN vectors with relational data. It supports both ivfflat and HNSW indexes. Production-ready for mid-scale workloads (tens of millions of vectors) but requires Postgres operational expertise and a running server.

**Choose Chroma** for local dev, notebooks, single-process RAG apps. **Choose pgvector** when you already run Postgres and need transactional consistency or relational joins alongside vector retrieval.
