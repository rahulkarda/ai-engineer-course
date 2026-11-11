# Solutions — Week 04 · Day 6 (Project Build)

Day 6 is a build day. This solution file provides reference implementations for each module and key test patterns. Compare against your own work after completing the build.

## models.py — reference

```python
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
    retriever: str


class RetrieverConfig(BaseModel):
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    embedding_model: str = "all-MiniLM-L6-v2"
    candidate_k: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    top_k: int = Field(default=5, gt=0)
```

## fusion.py — reference

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[tuple[str, float]]:
    rrf: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            rrf[result.document.doc_id] += 1.0 / (k + rank)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def score_fusion(
    ranked_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0 / len(ranked_lists)] * len(ranked_lists)
    
    def normalise(results: list[SearchResult]) -> dict[str, float]:
        scores = [r.score for r in results]
        mn, mx = min(scores, default=0), max(scores, default=1)
        rng = mx - mn if mx != mn else 1.0
        return {r.document.doc_id: (r.score - mn) / rng for r in results}
    
    combined: dict[str, float] = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for doc_id, norm_score in normalise(ranked).items():
            combined[doc_id] += w * norm_score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

## evaluation.py — reference

```python
import math

def precision_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.document.doc_id in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.document.doc_id in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mean_reciprocal_rank(
    results_list: list[list[SearchResult]],
    relevant_ids_list: list[set[str]],
) -> float:
    rr_sum = 0.0
    for results, relevant in zip(results_list, relevant_ids_list):
        for rank, result in enumerate(results, start=1):
            if result.document.doc_id in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(results_list) if results_list else 0.0


def ndcg_at_k(results: list[SearchResult], relevant_ids: set[str], k: int) -> float:
    def dcg(hits: list[int]) -> float:
        return sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    
    top_k_hits = [1 if r.document.doc_id in relevant_ids else 0 for r in results[:k]]
    ideal_hits = sorted(top_k_hits, reverse=True)
    actual_dcg = dcg(top_k_hits)
    ideal_dcg = dcg(ideal_hits)
    return actual_dcg / ideal_dcg if ideal_dcg else 0.0
```

## Key test patterns

### Mocking SentenceTransformer in test_dense.py

```python
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from hybrid_retriever.dense import SemanticRetriever
from hybrid_retriever.models import Document, RetrieverConfig


@pytest.fixture
def mock_retriever():
    with patch("hybrid_retriever.dense.SentenceTransformer") as MockST:
        inst = MockST.return_value
        inst.get_sentence_embedding_dimension.return_value = 4
        inst.encode.side_effect = lambda texts, **kw: np.random.default_rng(42).random(
            (len(texts) if isinstance(texts, list) else 1, 4)
        ).astype("float32")
        yield SemanticRetriever(RetrieverConfig())


def test_retrieve_returns_results(mock_retriever):
    docs = [Document(doc_id=f"d{i}", text=f"doc {i}") for i in range(5)]
    mock_retriever.add_documents(docs)
    results = mock_retriever.retrieve("test query", top_k=3)
    assert len(results) == 3
    assert all(r.document.doc_id.startswith("d") for r in results)
```

## Expected compare_retrievers.py output (example)

```
| Query                              | BM25 P@5 | Dense P@5 | Hybrid P@5 | BM25 MRR | Dense MRR | Hybrid MRR |
|------------------------------------|----------|-----------|------------|----------|-----------|------------|
| keyword ranking algorithm          | 1.00     | 0.60      | 1.00       | 1.00     | 0.50      | 1.00       |
| Python web API framework           | 0.60     | 0.60      | 0.80       | 1.00     | 1.00      | 1.00       |
| approximate nearest neighbor index | 0.40     | 0.80      | 0.80       | 0.50     | 1.00      | 1.00       |
| RAG retrieval context              | 0.60     | 0.60      | 0.80       | 0.50     | 0.33      | 0.50       |
| how does RRF work                  | 0.20     | 0.60      | 0.60       | 0.20     | 1.00      | 1.00       |
| **Mean**                           | 0.56     | 0.64      | **0.80**   | 0.64     | 0.77      | **0.90**   |
```

Hybrid wins on MRR for 4/5 queries (ties for RAG query). This is the expected outcome — exact-term queries go to BM25; paraphrase queries go to dense; hybrid captures both.
