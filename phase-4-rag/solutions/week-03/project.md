# Solutions — Week 03 Project

The project specification and full implementation are in the lesson file (`day-6.md`). This solution file provides:
1. A self-check rubric for marking your own project
2. Sample output to verify correctness
3. Answers to common pitfalls

## Self-check rubric

Work through each item before looking at the comparison notes below.

### Core (required — 60 pts)

| # | Check | Points |
|---|-------|--------|
| 1 | `VectorStore` ABC has all 6 abstract methods with correct signatures | 5 |
| 2 | `InMemoryVectorStore.query` normalises both the corpus and query before computing IP | 10 |
| 3 | `FAISSVectorStore.upsert` correctly handles idempotent re-insertion (metadata update, no duplicate vector) | 10 |
| 4 | `ChromaVectorStore.query` converts simple `{k: v}` filter to Chroma `{k: {"$eq": v}}` format | 10 |
| 5 | `persist` + `load` round-trip: `query` returns identical top-5 IDs before and after | 10 |
| 6 | Factory raises `ValueError` for unknown backends | 5 |
| 7 | `pytest -v` exits 0 with ≥ 15 tests | 10 |

### Code quality (15 pts)

| # | Check | Points |
|---|-------|--------|
| 8 | All public methods have type hints | 5 |
| 9 | `logging` used, no `print` statements | 5 |
| 10 | No hardcoded credentials; any env vars accessed via `os.getenv` | 5 |

### Extension (25 pts — choose ≥ 1)

| Extension | Points |
|-----------|--------|
| A — pgvector backend passes a filtered query test with a live Postgres | 25 |
| B — `Retriever.index` + `Retriever.retrieve` work with a dummy embed function | 25 |
| C — Benchmark script produces a markdown table with latency and recall | 25 |

**Pass gate: ≥ 75 pts (75 %)**

## Common pitfalls and fixes

### Pitfall 1 — FAISS upsert adds duplicates

**Symptom**: `count()` grows on every upsert of the same ID.
**Cause**: FAISS HNSW does not support in-place updates; re-adding the same ID creates a second entry.
**Fix**: Track inserted IDs in `_id_to_int`. In `upsert`, skip `add_with_ids` if the ID already exists; update only metadata.

### Pitfall 2 — Chroma filter with dict values raises TypeError

**Symptom**: `InvalidArgumentError: where clause must use operator form`.
**Cause**: Passing `{"category": "legal"}` instead of `{"category": {"$eq": "legal"}}`.
**Fix**: Wrap scalar filter values in `{"$eq": value}` before passing to Chroma.

### Pitfall 3 — InMemoryVectorStore query returns wrong rankings after unnormalised insert

**Symptom**: Self-query (query a vector against itself) doesn't return score ≈ 1.0.
**Cause**: Vectors stored without normalisation; cosine score computed incorrectly.
**Fix**: In `query`, normalise each stored vector and the query vector before computing the dot product (as shown in the reference implementation).

### Pitfall 4 — FAISS persist loses ID mapping

**Symptom**: After `load`, all IDs appear as `None`.
**Cause**: Only `index.faiss` was saved; `meta.json` containing `id_to_int`/`int_to_id` was missing.
**Fix**: Always save both files together; load verifies both exist.

## Expected test output

```
tests/test_vector_store.py::TestUpsertQuery::test_upsert_increases_count[memory] PASSED
tests/test_vector_store.py::TestUpsertQuery::test_upsert_increases_count[faiss] PASSED
tests/test_vector_store.py::TestUpsertQuery::test_upsert_increases_count[chroma] PASSED
... (15+ tests)
====================== 15 passed in 4.2s ======================
```
