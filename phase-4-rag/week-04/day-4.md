# Phase 4 · Week 04 · Day 4 — Hybrid Retrieval & Reciprocal Rank Fusion

> Objective: Combine dense and sparse retrievers via score fusion and RRF; understand why hybrid beats each alone.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-3]]

## Concept

No single retriever dominates all queries:

| Query type | Semantic wins | Keyword (BM25) wins |
|---|---|---|
| Paraphrase / synonym | "how do LLMs encode context" | — |
| Exact term | — | "GPT-4o model ID" |
| Rare proper noun | — | "CVE-2024-1234" |
| Cross-lingual | yes | — |
| Short factual | varies | often (precise) |

**Hybrid retrieval** runs both retrievers, then fuses their ranked lists. The result gets the recall of semantic search and the precision of keyword search.

### Two fusion strategies

#### 1 — Score normalisation + weighted sum (Score Fusion)

Normalise each retriever's scores to [0, 1], then combine:

```
hybrid_score(d) = α · normalise(bm25_score(d)) + (1-α) · normalise(dense_score(d))
```

**Problem:** BM25 and cosine scores live in incomparable ranges, and normalisation is sensitive to outliers (one very high-scoring document compresses everyone else).

#### 2 — Reciprocal Rank Fusion (RRF) — preferred

RRF (Cormack et al., 2009) ignores raw scores entirely; it only uses **rank positions**:

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$

where **k** is a smoothing constant (default **60** — empirically robust across benchmarks).

**Properties:**
- Score-scale invariant — works even if BM25 returns 0–10 and cosine returns 0–1.
- Robust to outliers — rank 1 and rank 2 differ by 1/(60+1) − 1/(60+2) ≈ 0.00027, not an order of magnitude.
- Documents missing from one list get rank = N+1 (or ∞ contribution = 0).
- No hyperparameter tuning of α needed.

**When score fusion is better:** when you have labelled relevance data and can tune α per query type.

## How it works internally

```
Query
  ├─► BM25Retriever.retrieve(q, top_k=100)  → [(doc_id, score), ...]  ranked list R_bm25
  └─► SemanticRetriever.retrieve(q, top_k=100) → [(doc_id, score), ...] ranked list R_dense

RRF fusion:
  for each doc_id in union(R_bm25, R_dense):
      rrf_score = 1/(k + rank_in_bm25) + 1/(k + rank_in_dense)
      # if doc absent from one list: contribution = 0

Sort by rrf_score descending → top_k
```

Retrieval top-k (candidate pool, e.g. 100) >> final top-k returned to user (e.g. 5). The large pool ensures both retrievers can contribute documents even when their raw rankings diverge.

## Worked example

```python
"""
day4_hybrid_rrf.py — Hybrid retrieval with Reciprocal Rank Fusion.

Run: python day4_hybrid_rrf.py
Deps: uv add sentence-transformers faiss-cpu numpy pydantic
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import faiss  # type: ignore[import-untyped]
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal BM25 (from day 3 — inline for self-containment)
# ---------------------------------------------------------------------------
class BM25Config(BaseModel):
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)


@dataclass
class BM25Retriever:
    config: BM25Config = field(default_factory=BM25Config)
    _docs: list[str] = field(default_factory=list, init=False)
    _doc_lengths: list[int] = field(default_factory=list, init=False)
    _total_len: int = field(default=0, init=False)
    _inverted: dict[str, dict[int, int]] = field(
        default_factory=lambda: defaultdict(dict), init=False
    )

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def add_documents(self, texts: list[str]) -> None:
        for text in texts:
            doc_id = len(self._docs)
            self._docs.append(text)
            tokens = self._tokenize(text)
            self._doc_lengths.append(len(tokens))
            self._total_len += len(tokens)
            for term, freq in Counter(tokens).items():
                self._inverted[term][doc_id] = freq

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        terms = self._tokenize(query)
        n = len(self._docs)
        avgdl = self._total_len / n if n else 1.0
        candidates: set[int] = set()
        for t in terms:
            candidates.update(self._inverted.get(t, {}).keys())
        scores: dict[int, float] = {}
        for doc_id in candidates:
            dl = self._doc_lengths[doc_id]
            s = 0.0
            for t in terms:
                f = self._inverted.get(t, {}).get(doc_id, 0)
                df = len(self._inverted.get(t, {}))
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                k1, b = self.config.k1, self.config.b
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
            scores[doc_id] = s
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked  # [(doc_id, score)]


# ---------------------------------------------------------------------------
# Semantic retriever
# ---------------------------------------------------------------------------
@dataclass
class SemanticRetriever:
    model_name: str = "all-MiniLM-L6-v2"
    _model: SentenceTransformer = field(init=False, repr=False)
    _index: faiss.IndexFlatIP = field(init=False, repr=False)
    _docs: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(self.model_name)
        dim = self._model.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(dim)

    def add_documents(self, docs: list[str]) -> None:
        self._docs = docs
        vecs: np.ndarray = self._model.encode(docs, normalize_embeddings=True)
        self._index.add(vecs)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(q, top_k)
        return [(int(idx), float(scores[0][r])) for r, idx in enumerate(indices[0])]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    ranked_lists: each list is [(doc_id, score), ...] sorted by descending score.
    Returns: [(doc_id, rrf_score), ...] sorted descending.
    """
    rrf_scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------
@dataclass
class HybridRetriever:
    bm25: BM25Retriever
    semantic: SemanticRetriever
    candidate_k: int = 50   # pool size per retriever
    rrf_k: int = 60         # RRF smoothing constant

    def add_documents(self, docs: list[str]) -> None:
        self.bm25.add_documents(docs)
        self.semantic.add_documents(docs)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
        bm25_results = self.bm25.retrieve(query, top_k=self.candidate_k)
        dense_results = self.semantic.retrieve(query, top_k=self.candidate_k)
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k=self.rrf_k)
        docs = self.bm25._docs  # shared corpus reference
        return [(doc_id, score, docs[doc_id]) for doc_id, score in fused[:top_k]]


# ---------------------------------------------------------------------------
# Demo + comparison
# ---------------------------------------------------------------------------
CORPUS = [
    "Python is a high-level interpreted language used in data science and ML.",
    "FastAPI is a modern web framework for building APIs with Python type hints.",
    "Transformers are neural network architectures using self-attention mechanisms.",
    "Retrieval-Augmented Generation enhances LLM responses with retrieved context.",
    "BM25 is a probabilistic keyword ranking function for information retrieval.",
    "Dense embeddings encode semantic meaning as fixed-length float vectors.",
    "FAISS provides efficient approximate nearest-neighbor search over embeddings.",
    "A vector database stores and indexes embeddings for ANN retrieval at scale.",
    "Fine-tuning adapts a pre-trained model on task-specific labelled examples.",
    "Prompt engineering crafts instructions to control LLM output quality.",
    "CVE-2024-1234 is a critical vulnerability in the foo-library package v2.3.",
    "The model ID for GPT-4o is gpt-4o-2024-05-13 in the OpenAI API.",
]

QUERIES = [
    ("Semantic: paraphrase", "how do language models use context from documents"),
    ("Keyword: exact term", "CVE-2024-1234"),
    ("Keyword: model ID", "GPT-4o model ID gpt-4o"),
    ("Hybrid: mixed", "Python API framework embeddings"),
]


def compare(corpus: list[str], queries: list[tuple[str, str]]) -> None:
    bm25 = BM25Retriever()
    sem = SemanticRetriever()
    hybrid = HybridRetriever(bm25=bm25, semantic=sem)
    hybrid.add_documents(corpus)

    for label, q in queries:
        log.info("\n=== %s: '%s' ===", label, q)
        bm25_results = bm25.retrieve(q, top_k=3)
        dense_results = sem.retrieve(q, top_k=3)
        hybrid_results = hybrid.retrieve(q, top_k=3)

        log.info("BM25  top-3:")
        for doc_id, score in bm25_results:
            log.info("  [%.4f] %s", score, corpus[doc_id][:70])
        log.info("Dense top-3:")
        for doc_id, score in dense_results:
            log.info("  [%.4f] %s", score, corpus[doc_id][:70])
        log.info("Hybrid top-3:")
        for doc_id, score, text in hybrid_results:
            log.info("  [%.4f] %s", score, text[:70])


def main() -> None:
    compare(CORPUS, QUERIES)


if __name__ == "__main__":
    main()
```

### What to observe

- "CVE-2024-1234" — BM25 wins (exact ID), semantic misses it.
- "how do language models use context from documents" — dense wins (paraphrase of RAG concept).
- "Python API framework embeddings" — hybrid correctly surfaces the FastAPI + embedding docs together.

## Exercises

1. **RRF k sensitivity.** Replace k=60 with k={1, 10, 30, 100}. For the mixed query, how does the ranking change? Which k gives the most stable hybrid ranking?
2. **Score fusion baseline.** Implement `score_fusion(bm25_results, dense_results, alpha=0.5)` using min-max normalisation per list. Compare its top-3 to RRF for all four queries. Which is more consistent?
3. **Missing-list penalty.** Currently documents absent from one retriever contribute 0 from that retriever. Add a `penalty` mode: absent docs get rank = N + 1 (end of list). Does this change the hybrid output for the "CVE" query?
4. **Retrieval overlap.** For each query, compute: `|bm25_top10 ∩ dense_top10| / 10`. High overlap = redundant hybrid. Report the overlap for all four queries.
5. **Unit tests.** Write `tests/test_hybrid.py` with: (a) RRF returns doc_id in both lists with higher score than one-list-only docs; (b) fused list length ≤ top_k; (c) HybridRetriever.retrieve returns correct doc text; (d) mocked retrievers (no real encoder needed — stub out `SemanticRetriever.retrieve`).

(Solutions: ../solutions/week-04/day-4.md)

## Self-check

- [ ] I can derive the RRF formula and explain why k=60 is robust.
- [ ] I can implement RRF fusion from scratch in under 15 lines.
- [ ] I understand why score-scale invariance matters for multi-retriever fusion.
- [ ] I can describe two failure modes of score normalisation fusion.
- [ ] I can build a `HybridRetriever` wrapping BM25 + dense without reference.

## Supplementary links

- [RRF paper (Cormack et al., 2009)](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Pinecone hybrid search guide](https://docs.pinecone.io/guides/search/understanding-hybrid-search)
- [Elasticsearch hybrid search](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#knn-semantic-search)
