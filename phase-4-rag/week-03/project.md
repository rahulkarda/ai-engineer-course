# Week 03 Project — Vector Store Abstraction

## Overview

Build a production-ready vector store abstraction with pluggable backends, metadata filtering, CRUD operations, and persistence.

## Repository layout

```
week03-vector-store/
├── src/
│   └── vector_store/
│       ├── __init__.py
│       ├── base.py           ← ABC + VectorRecord + SearchResult
│       ├── in_memory.py      ← numpy brute-force backend
│       ├── faiss_backend.py  ← FAISS HNSW + IDMap backend
│       ├── chroma_backend.py ← Chroma in-process backend
│       └── factory.py        ← create_store(backend, **kwargs)
├── tests/
│   └── test_vector_store.py  ← parametrized over all backends
├── main.py                   ← CLI demo
├── pyproject.toml
└── .env.example
```

## Deliverables

### Core (required)

- [ ] `VectorStore` ABC with `upsert`, `query`, `delete`, `count`, `persist`, `load`
- [ ] `InMemoryVectorStore` — exact cosine, JSON persistence
- [ ] `FAISSVectorStore` — HNSW + IDMap, binary + JSON persistence
- [ ] `ChromaVectorStore` — in-process, pickle persistence
- [ ] `create_store(backend, dim)` factory
- [ ] Pytest suite: ≥ 15 tests passing across all 3 backends

### Extensions (choose ≥ 1)

- [ ] **A** — `PgVectorStore` with `DATABASE_URL` from env
- [ ] **B** — `Retriever` wrapper with `index(docs, embed_fn)` and `retrieve(query, embed_fn, k, filter)`
- [ ] **C** — Benchmark script reporting latency and recall\@10 in a markdown table

## Acceptance criteria

| Criterion | Pass |
|-----------|------|
| All core backends: upsert + query return correct results | Required |
| Filter tests: only matching metadata is returned | Required |
| Persist + load round-trip: query results identical | Required |
| `pytest -v` exits 0 | Required |
| No hardcoded secrets or credentials | Required |
| Type hints on all public methods | Required |
| `logging` used instead of `print` | Required |
| At least one extension completed | Recommended |

## Setup

```bash
# Create project
mkdir week03-vector-store && cd week03-vector-store
uv init
uv add faiss-cpu chromadb numpy pytest psycopg

# Run tests
uv run pytest tests/ -v

# Run CLI demo
uv run python main.py
```

## Grading

Use the Week 03 assessment rubric. The project accounts for the **Independent implementation (40%)** dimension.

Project solution: ../solutions/week-03/project.md
