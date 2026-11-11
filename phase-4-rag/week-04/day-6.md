# Phase 4 · Week 04 · Day 6 — Project Build: Hybrid Retriever System (4 hrs)

> Objective: Build a complete, tested hybrid retriever combining dense + BM25 with RRF fusion over a document corpus, with a retrieval quality comparison script.  ·  Time: ~4 hrs  ·  Prereqs: [[day-1]] through [[day-5]]

## Overview

Today you build the Week 4 project in full. The system has four components:

```
project/
├── src/hybrid_retriever/
│   ├── __init__.py
│   ├── models.py          # Document, SearchResult, RetrieverConfig (Pydantic)
│   ├── bm25.py            # BM25Retriever (from scratch)
│   ├── dense.py           # SemanticRetriever (FAISS + sentence-transformers)
│   ├── fusion.py          # RRF + score_fusion helpers
│   ├── hybrid.py          # HybridRetriever (composes bm25 + dense)
│   ├── filters.py         # Metadata filter predicates
│   └── evaluation.py      # Retrieval quality metrics (precision@k, recall@k, MRR)
├── tests/
│   ├── test_bm25.py
│   ├── test_dense.py
│   ├── test_fusion.py
│   ├── test_hybrid.py
│   └── test_evaluation.py
├── scripts/
│   └── compare_retrievers.py   # quality comparison CLI
└── pyproject.toml
```

## Build plan (time estimates)

| Step | Time |
|---|---|
| Scaffold project, pyproject.toml | 15 min |
| models.py | 15 min |
| bm25.py (adapt from day-3) | 20 min |
| dense.py (adapt from day-1) | 20 min |
| fusion.py (RRF + score fusion) | 20 min |
| filters.py | 15 min |
| hybrid.py | 25 min |
| evaluation.py | 20 min |
| tests (all 5 files) | 45 min |
| compare_retrievers.py script | 20 min |
| Debug & polish | 25 min |

## Specification

### models.py

```python
"""Data models for the hybrid retriever system."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    document: Document
    score: float
    retriever: str  # "bm25" | "dense" | "hybrid"


class RetrieverConfig(BaseModel):
    # BM25
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    # Dense
    embedding_model: str = "all-MiniLM-L6-v2"
    # Hybrid
    candidate_k: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    # General
    top_k: int = Field(default=5, gt=0)
```

### bm25.py

Implement `BM25Retriever` with:
- `add_documents(docs: list[Document]) -> None`
- `retrieve(query: str, top_k: int, filter_fn=None) -> list[SearchResult]`
- `save(path) / load(path)` classmethods

### dense.py

Implement `SemanticRetriever` with:
- `add_documents(docs: list[Document]) -> None`
- `retrieve(query: str, top_k: int, filter_fn=None) -> list[SearchResult]`
- `save(path) / load(path)` classmethods (FAISS write_index + json for doc list)

### fusion.py

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Returns [(doc_id, rrf_score)] sorted descending."""
    ...

def score_fusion(
    ranked_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Min-max normalise each list, then weighted-average. Returns [(doc_id, score)]."""
    ...
```

### hybrid.py

`HybridRetriever` composes BM25 + Dense + fusion:

```python
class HybridRetriever:
    def __init__(self, config: RetrieverConfig) -> None: ...
    def add_documents(self, docs: list[Document]) -> None: ...
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_fn: Callable | None = None,
        fusion: Literal["rrf", "score"] = "rrf",
    ) -> list[SearchResult]: ...
    def save(self, directory: Path) -> None: ...
    @classmethod
    def load(cls, directory: Path) -> "HybridRetriever": ...
```

### filters.py

```python
Filter = Callable[[Document], bool]

def eq(field: str, value: Any) -> Filter: ...
def gte(field: str, value: Any) -> Filter: ...
def lte(field: str, value: Any) -> Filter: ...
def contains(field: str, value: Any) -> Filter: ...  # for list-type metadata
def and_(*filters: Filter) -> Filter: ...
def or_(*filters: Filter) -> Filter: ...
def not_(f: Filter) -> Filter: ...
```

### evaluation.py

```python
def precision_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    ...

def recall_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    ...

def mean_reciprocal_rank(
    results_list: list[list[SearchResult]], relevant_ids_list: list[set[str]]
) -> float:
    """MRR over multiple queries."""
    ...

def ndcg_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    ...
```

### compare_retrievers.py (CLI script)

```python
"""
compare_retrievers.py — compare BM25, Dense, and Hybrid retrieval quality.

Usage: python scripts/compare_retrievers.py
"""
# Loads a fixed test corpus + queries with known relevant doc IDs.
# Runs BM25, Dense, Hybrid.
# Prints a markdown table of Precision@5, Recall@5, MRR for each retriever.
```

## Exercises (build tasks)

1. Implement all seven source files per the specifications above. All code must have type hints, use `logging` (not print), and use `os.getenv` for any API key (embedding model is local, so none needed here).
2. Write all five test files. Each test file must have at least 5 tests. Use `pytest.fixture` for shared corpus/retriever setup. Mock the `SentenceTransformer` in `test_dense.py` using `unittest.mock.patch`.
3. Run `compare_retrievers.py` over this evaluation corpus (20 docs, 5 queries, ground-truth relevance provided below) and report which retriever wins on each metric.
4. Try at least two different `rrf_k` values (30 and 60) and two `embedding_model` values (`all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`). Report findings in a brief comment block at the top of `compare_retrievers.py`.
5. Add a `save`/`load` round-trip integration test that: indexes 10 docs, saves to a temp dir, loads, and asserts the top-1 result for a query is identical before and after.

### Evaluation corpus

```python
EVAL_CORPUS = [
    {"id": "d01", "text": "Python is a high-level interpreted language.", "cat": "prog"},
    {"id": "d02", "text": "FastAPI enables async REST APIs in Python.", "cat": "prog"},
    {"id": "d03", "text": "BM25 ranks documents by probabilistic term weighting.", "cat": "retrieval"},
    {"id": "d04", "text": "Dense embeddings encode meaning into float vectors.", "cat": "retrieval"},
    {"id": "d05", "text": "FAISS provides fast ANN search over embeddings.", "cat": "retrieval"},
    {"id": "d06", "text": "Pydantic validates data using Python type hints.", "cat": "prog"},
    {"id": "d07", "text": "LangGraph builds stateful agent workflows.", "cat": "agents"},
    {"id": "d08", "text": "Semantic search finds documents by meaning.", "cat": "retrieval"},
    {"id": "d09", "text": "BM25 beats TF-IDF with TF saturation.", "cat": "retrieval"},
    {"id": "d10", "text": "HNSW is a graph-based ANN index structure.", "cat": "retrieval"},
    {"id": "d11", "text": "RRF fuses ranked lists without score normalisation.", "cat": "retrieval"},
    {"id": "d12", "text": "Python 3.12 introduced type parameter syntax.", "cat": "prog"},
    {"id": "d13", "text": "pytest is the standard Python testing framework.", "cat": "prog"},
    {"id": "d14", "text": "Agent tools are functions an LLM can invoke.", "cat": "agents"},
    {"id": "d15", "text": "RAG reduces LLM hallucinations by grounding in docs.", "cat": "rag"},
    {"id": "d16", "text": "Chunking splits documents into retrieval-sized pieces.", "cat": "rag"},
    {"id": "d17", "text": "Cosine similarity measures angle between two vectors.", "cat": "retrieval"},
    {"id": "d18", "text": "Metadata filters restrict retrieval to relevant subsets.", "cat": "retrieval"},
    {"id": "d19", "text": "LLM context window limits how much text fits in a prompt.", "cat": "rag"},
    {"id": "d20", "text": "MRR measures mean reciprocal rank of first relevant result.", "cat": "eval"},
]

EVAL_QUERIES = [
    {"query": "keyword ranking algorithm", "relevant": {"d03", "d09"}},
    {"query": "Python web API framework", "relevant": {"d01", "d02", "d06"}},
    {"query": "approximate nearest neighbor index", "relevant": {"d05", "d10"}},
    {"query": "RAG retrieval context", "relevant": {"d15", "d16", "d19"}},
    {"query": "how does RRF work", "relevant": {"d11"}},
]
```

(Solutions: ../solutions/week-04/day-6.md)

## Self-check

- [ ] All seven source files are implemented with type hints and logging.
- [ ] All five test files pass with zero failures (`pytest tests/`).
- [ ] `compare_retrievers.py` prints a results table showing which retriever wins on which metric.
- [ ] Hybrid wins or ties on MRR vs. BM25-only and Dense-only for at least 3 of 5 queries.
- [ ] Save/load round-trip integration test passes.
