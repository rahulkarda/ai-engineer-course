# Week 04 Project — Hybrid Retriever System

> Build a production-quality hybrid retriever combining BM25 + dense retrieval with RRF fusion, metadata filtering, and retrieval quality evaluation.

## Goal

A importable Python package `hybrid_retriever` that any downstream RAG system can use:

```python
from hybrid_retriever import HybridRetriever, RetrieverConfig
from hybrid_retriever.filters import eq, gte, and_

config = RetrieverConfig(embedding_model="all-MiniLM-L6-v2", top_k=5)
retriever = HybridRetriever(config)
retriever.add_documents(docs)
results = retriever.retrieve("semantic search over embeddings", filter_fn=eq("lang", "en"))
```

## Deliverables

### 1 — Package `src/hybrid_retriever/`

| File | Must implement |
|---|---|
| `models.py` | `Document`, `SearchResult`, `RetrieverConfig` (Pydantic) |
| `bm25.py` | `BM25Retriever` — from scratch, no `rank-bm25` |
| `dense.py` | `SemanticRetriever` — FAISS + sentence-transformers |
| `fusion.py` | `reciprocal_rank_fusion`, `score_fusion` |
| `filters.py` | `eq`, `gte`, `lte`, `contains`, `and_`, `or_`, `not_` |
| `hybrid.py` | `HybridRetriever` — composes all components |
| `evaluation.py` | `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k` |

### 2 — Test suite `tests/`

Five test files, ≥ 5 tests each (≥ 25 total). `SentenceTransformer` must be mocked in all tests — no real encoder calls.

### 3 — Comparison script `scripts/compare_retrievers.py`

- Loads the 20-doc eval corpus from day-6.
- Runs BM25, Dense, and Hybrid.
- Prints a markdown table: Precision@5, Recall@5, MRR for each retriever × each query.
- Prints aggregate row (mean across queries).

### 4 — Serialisation

`HybridRetriever.save(directory)` and `.load(directory)` must work. Verified by integration test.

## Acceptance criteria

- [ ] `pytest tests/` — zero failures.
- [ ] `python scripts/compare_retrievers.py` runs end-to-end and prints a results table.
- [ ] Hybrid MRR ≥ max(BM25 MRR, Dense MRR) for ≥ 3 of 5 eval queries (or explain why it doesn't in a comment).
- [ ] All source files: type hints, `logging`, `os.getenv` for any external service key, no hardcoded secrets.
- [ ] No god files — each module ≤ 300 lines.
- [ ] `RetrieverConfig` uses Pydantic with field validation (gt=0 where appropriate).

## Extension challenges (optional)

1. **Reranker.** Add a `CrossEncoderReranker` that re-scores the hybrid top-20 using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Compare MRR before/after reranking.
2. **Streaming index.** Implement a background thread that batches incoming `add_document` calls and flushes to the index every N documents or T seconds (producer-consumer pattern with `queue.Queue`).
3. **REST API.** Wrap `HybridRetriever` in a FastAPI app with endpoints `POST /index`, `POST /search`, `GET /health`. Return `SearchResult` list as JSON.
4. **Evaluation suite.** Download a small BEIR dataset (NFCorpus, SciFact) and run your retriever. Compare to the published BM25 and dense baselines.

## Project timeline

| Day | Task |
|---|---|
| Day 6 (4 hr block) | Build all 7 source files + 5 test files + compare script |
| Day 7 review | Debug, run full test suite, run comparison, write findings comment |

Answer key: ../solutions/week-04/project.md
