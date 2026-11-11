# Week 04 Assessment — Retrieval: Semantic, Keyword & Hybrid

> Time: ~90 min  ·  Pass gate: 75%

## Conceptual (interview-style) — 25 pts

Answer each question in 3–5 sentences. Brevity and precision matter.

1. **(4 pts)** Explain the TF saturation effect in BM25. What happens to the BM25 term score as term frequency *f* → ∞ for a fixed k₁? Why is this desirable?

2. **(4 pts)** What is Reciprocal Rank Fusion? Why is it preferred over score normalisation fusion when you have no labelled tuning data?

3. **(4 pts)** A user queries "GDPR article 17". Your dense retriever returns legal summaries about the right to erasure (semantically correct) but misses the exact statutory text. Your BM25 retriever returns the exact text. How does hybrid retrieval address this, and at what cost?

4. **(4 pts)** Compare HNSW and a flat (brute-force) FAISS index on: (a) build time, (b) query time, (c) accuracy. When would you choose flat?

5. **(4 pts)** Describe the HyDE technique. When would it help, and what is its risk?

6. **(5 pts)** You have a 10M-document corpus. Estimate the memory footprint of: (a) BM25 inverted index (vocab 500k, avg posting list 200 entries, each entry 8 bytes), (b) dense FAISS index (dim=384, float32). Which fits in RAM on a 32GB machine?

## Coding challenge — 40 pts

Build a `HybridSearchService` class that satisfies the following spec. Do **not** use `rank-bm25` or any other BM25 library — implement from scratch.

```python
from pathlib import Path
from typing import Any, Callable, Literal

class HybridSearchService:
    """
    Acceptance criteria:
    A. Constructor accepts a RetrieverConfig (Pydantic model with fields:
       bm25_k1, bm25_b, embedding_model, candidate_k, rrf_k, top_k).
    B. index_documents(docs: list[dict]) where each dict has keys:
       "id" (str), "text" (str), "metadata" (dict).
    C. search(
           query: str,
           top_k: int | None = None,
           filter_fn: Callable | None = None,
           fusion: Literal["rrf", "score"] = "rrf",
       ) -> list[dict]
       Returns list of {"id": str, "text": str, "score": float, "metadata": dict}
       sorted by score descending.
    D. save(directory: Path) -> None — persists both BM25 index and FAISS index.
    E. classmethod load(directory: Path) -> HybridSearchService — reloads both.
    F. evaluate(
           queries: list[str],
           relevant_ids: list[set[str]],
           k: int = 5,
       ) -> dict[str, float]
       Returns {"precision@k": float, "recall@k": float, "mrr": float}.
    """
    ...
```

**Acceptance tests** (your implementation must pass all of these):

```python
import pytest
from unittest.mock import patch, MagicMock
import numpy as np

def make_service():
    # Stub out the sentence-transformer so tests run without GPU/internet
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        instance = MockST.return_value
        instance.get_sentence_embedding_dimension.return_value = 4
        instance.encode.side_effect = lambda texts, **kw: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 4
        ).astype("float32")
        svc = HybridSearchService(RetrieverConfig())
    return svc

DOCS = [
    {"id": "d1", "text": "Python is a programming language", "metadata": {"cat": "prog"}},
    {"id": "d2", "text": "BM25 is a ranking algorithm", "metadata": {"cat": "retrieval"}},
    {"id": "d3", "text": "FAISS provides vector similarity search", "metadata": {"cat": "retrieval"}},
    {"id": "d4", "text": "FastAPI builds REST APIs in Python", "metadata": {"cat": "prog"}},
    {"id": "d5", "text": "RRF fuses multiple ranked lists", "metadata": {"cat": "retrieval"}},
]

def test_search_returns_top_k():
    svc = make_service()
    svc.index_documents(DOCS)
    results = svc.search("Python programming", top_k=2)
    assert len(results) == 2

def test_search_result_fields():
    svc = make_service()
    svc.index_documents(DOCS)
    results = svc.search("ranking algorithm")
    assert all("id" in r and "text" in r and "score" in r for r in results)

def test_metadata_filter():
    svc = make_service()
    svc.index_documents(DOCS)
    f = lambda doc: doc.metadata.get("cat") == "retrieval"
    results = svc.search("algorithm", filter_fn=f, top_k=10)
    assert all(r["metadata"]["cat"] == "retrieval" for r in results)

def test_save_load_roundtrip(tmp_path):
    svc = make_service()
    svc.index_documents(DOCS)
    svc.save(tmp_path)
    svc2 = HybridSearchService.load(tmp_path)
    r1 = svc.search("Python", top_k=1)
    r2 = svc2.search("Python", top_k=1)
    assert r1[0]["id"] == r2[0]["id"]

def test_evaluate_returns_metrics():
    svc = make_service()
    svc.index_documents(DOCS)
    metrics = svc.evaluate(
        queries=["Python", "BM25"],
        relevant_ids=[{"d1", "d4"}, {"d2"}],
        k=3,
    )
    assert "precision@k" in metrics and "recall@k" in metrics and "mrr" in metrics

def test_score_fusion_mode():
    svc = make_service()
    svc.index_documents(DOCS)
    results = svc.search("ranking", fusion="score")
    assert len(results) > 0
```

## Debugging challenge — 20 pts

The following `bm25_retrieve` function has **3 bugs**. Identify each bug, explain why it is wrong, and provide the fix.

```python
def bm25_retrieve(query: str, corpus: list[str], k1: float = 1.5, b: float = 0.75, top_k: int = 5):
    import math, re
    from collections import Counter
    
    tokenize = lambda t: re.findall(r"[a-z]+", t)
    tokenized = [tokenize(d) for d in corpus]
    N = len(corpus)
    avgdl = sum(len(t) for t in tokenized) / N
    
    df = Counter()
    for tokens in tokenized:
        for term in tokens:           # Bug 1
            df[term] += 1
    
    def idf(t):
        return math.log(N / df.get(t, 1))   # Bug 2
    
    query_terms = tokenize(query)
    scores = []
    for i, tokens in enumerate(tokenized):
        tf = Counter(tokens)
        dl = len(tokens)
        s = 0
        for t in query_terms:
            f = tf.get(t, 0)
            s += idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))  # Bug 3
        scores.append((i, s))
    
    return sorted(scores, key=lambda x: x[1])[:top_k]  # Bug 4 — bonus
```

(Hint: there are 3 required bugs + 1 bonus bug — finding all 4 earns full marks.)

## Code quality & testing — 15 pts

Evaluate **your coding challenge implementation** against:

| Criterion | Marks |
|---|---|
| Type hints on all functions/methods | 3 |
| `logging` used (not `print`) | 2 |
| No secrets / hardcoded credentials | 2 |
| Pydantic models for config and results | 3 |
| Test mocking of external calls (sentence-transformer) | 3 |
| `pathlib.Path` used for file I/O | 2 |

## Rubric (pass ≥ 75 / 100)

| Dimension | Weight |
|---|---|
| Independent implementation (coding challenge) | 40% |
| Conceptual understanding (interview questions) | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-04/assessment.md
