# Solutions — Week 05 Project

This is the project reference. It documents the target architecture, key design decisions, and a reference `before_after_eval.py` implementation.

## Architecture decisions

### Why inject the retriever via Protocol?

Dependency injection via `Protocol` makes `RerankingPipeline` testable without importing `sentence-transformers` or any vector database. Tests mock the `Retriever` protocol — no network, no model download.

### Why Pydantic for config?

`PipelineConfig` as a `BaseModel` gives free validation (e.g., `score_threshold` in [0,1]), serialisation, and environment variable loading via `pydantic-settings`. Catching invalid config at startup is better than a cryptic `numpy` error at inference time.

### Why sigmoid normalisation on cross-encoder scores?

Raw logits from MS-MARCO models are unbounded. Sigmoid maps them to (0,1) so `score_threshold` has an interpretable meaning (0.3 = "weak relevance cut-off") regardless of which model is used.

---

## Reference: `before_after_eval.py`

```python
"""
before_after_eval.py

Demonstrates measurable improvement from the reranking pipeline
over a BM25 baseline on a synthetic eval corpus.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from models import Chunk
from metrics import RetrievalEvalCase, evaluate_retrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic corpus: 25 documents about RAG engineering
# ---------------------------------------------------------------------------
CORPUS: list[dict] = [
    {"id": "d01", "text": "Cross-encoders rerank candidate documents by scoring query-document pairs jointly.", "topic": "reranking"},
    {"id": "d02", "text": "Bi-encoders encode query and document independently; similarity is cosine distance.", "topic": "encoding"},
    {"id": "d03", "text": "BM25 ranks documents using term frequency and inverse document frequency.", "topic": "keyword"},
    {"id": "d04", "text": "Hybrid retrieval merges BM25 and dense vector search with reciprocal rank fusion.", "topic": "hybrid"},
    {"id": "d05", "text": "Maximal marginal relevance selects a diverse relevant subset from top-K candidates.", "topic": "diversity"},
    {"id": "d06", "text": "Parent document retrieval returns full documents after child chunks are matched.", "topic": "expansion"},
    {"id": "d07", "text": "Query expansion generates alternative phrasings to improve recall.", "topic": "expansion"},
    {"id": "d08", "text": "HyDE embeds a hypothetical answer document to bridge query-document vocabulary gap.", "topic": "expansion"},
    {"id": "d09", "text": "NDCG measures ranking quality with graded relevance labels.", "topic": "metrics"},
    {"id": "d10", "text": "MRR measures the rank of the first relevant document in the result list.", "topic": "metrics"},
    {"id": "d11", "text": "Precision at k is the fraction of top-k results that are relevant.", "topic": "metrics"},
    {"id": "d12", "text": "Recall at k measures the fraction of all relevant documents found in top-k.", "topic": "metrics"},
    {"id": "d13", "text": "Chunk size affects retrieval precision: smaller chunks are more precise.", "topic": "chunking"},
    {"id": "d14", "text": "Vector databases store embeddings and support ANN search.", "topic": "infrastructure"},
    {"id": "d15", "text": "Reranking improves precision without retraining the retrieval model.", "topic": "reranking"},
    {"id": "d16", "text": "Cross-encoder models attend to both query and document simultaneously.", "topic": "reranking"},
    {"id": "d17", "text": "Sentence transformers provide pre-trained bi-encoder models for semantic search.", "topic": "encoding"},
    {"id": "d18", "text": "Sigmoid normalisation maps unbounded logits to the (0,1) range.", "topic": "implementation"},
    {"id": "d19", "text": "Deduplication removes redundant chunks before passing context to the LLM.", "topic": "optimization"},
    {"id": "d20", "text": "Window expansion returns neighbouring chunks around a matched chunk.", "topic": "expansion"},
    {"id": "d21", "text": "Chunk overlap reduces boundary truncation artifacts in retrieval.", "topic": "chunking"},
    {"id": "d22", "text": "Reciprocal rank fusion combines rankings from multiple retrieval systems.", "topic": "hybrid"},
    {"id": "d23", "text": "MS-MARCO is a large-scale dataset for training information retrieval models.", "topic": "dataset"},
    {"id": "d24", "text": "Cross-encoder inference cannot be precomputed because it requires the query.", "topic": "reranking"},
    {"id": "d25", "text": "Lambda in MMR controls the trade-off between relevance and diversity.", "topic": "diversity"},
]

# ---------------------------------------------------------------------------
# 20-query eval set
# ---------------------------------------------------------------------------
EVAL_CASES: list[RetrievalEvalCase] = [
    RetrievalEvalCase("how does cross-encoder reranking work",
        {"d01", "d15", "d16", "d24"}, {"d01": 2, "d15": 2, "d16": 2, "d24": 1}),
    RetrievalEvalCase("difference between bi-encoder and cross-encoder",
        {"d02", "d01", "d16"}, {"d02": 2, "d01": 1, "d16": 1}),
    RetrievalEvalCase("how is BM25 calculated",
        {"d03"}, {"d03": 2}),
    RetrievalEvalCase("what is hybrid retrieval",
        {"d04", "d22"}, {"d04": 2, "d22": 1}),
    RetrievalEvalCase("MMR diversity retrieval",
        {"d05", "d25"}, {"d05": 2, "d25": 1}),
    RetrievalEvalCase("parent document retrieval explained",
        {"d06"}, {"d06": 2}),
    RetrievalEvalCase("query expansion techniques for RAG",
        {"d07", "d08"}, {"d07": 2, "d08": 1}),
    RetrievalEvalCase("what is HyDE",
        {"d08"}, {"d08": 2}),
    RetrievalEvalCase("how to measure retrieval quality",
        {"d09", "d10", "d11", "d12"}, {"d09": 2, "d10": 2, "d11": 1, "d12": 1}),
    RetrievalEvalCase("NDCG vs MRR",
        {"d09", "d10"}, {"d09": 2, "d10": 2}),
    RetrievalEvalCase("chunk size effect on retrieval",
        {"d13", "d21"}, {"d13": 2, "d21": 1}),
    RetrievalEvalCase("vector database for embeddings",
        {"d14"}, {"d14": 2}),
    RetrievalEvalCase("sentence transformers bi-encoder",
        {"d17", "d02"}, {"d17": 2, "d02": 1}),
    RetrievalEvalCase("logit normalisation sigmoid",
        {"d18"}, {"d18": 2}),
    RetrievalEvalCase("removing duplicate chunks",
        {"d19"}, {"d19": 2}),
    RetrievalEvalCase("window chunk expansion",
        {"d20"}, {"d20": 2}),
    RetrievalEvalCase("reciprocal rank fusion",
        {"d22", "d04"}, {"d22": 2, "d04": 1}),
    RetrievalEvalCase("cross-encoder scoring cannot be precomputed",
        {"d24", "d01"}, {"d24": 2, "d01": 1}),
    RetrievalEvalCase("lambda parameter MMR",
        {"d25", "d05"}, {"d25": 2, "d05": 1}),
    RetrievalEvalCase("precision recall at k metrics",
        {"d11", "d12"}, {"d11": 2, "d12": 2}),
]


# ---------------------------------------------------------------------------
# Baseline: BM25
# ---------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, corpus: list[dict]) -> None:
        tokenised = [doc["text"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenised)
        self._corpus = corpus

    def retrieve(self, query: str) -> list[str]:
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1]
        return [self._corpus[i]["id"] for i in order]


# ---------------------------------------------------------------------------
# Full pipeline: BM25 + simple reranking simulation
# (In a real system, wire RerankingPipeline; here we simulate score-based rerank)
# ---------------------------------------------------------------------------
def make_pipeline_retriever(corpus: list[dict]) -> "Callable[[str], list[str]]":
    """
    Simulates the full pipeline by boosting cross-encoder-relevant documents.
    In a real deployment: call RerankingPipeline.retrieve() and extract doc_ids.
    """
    from typing import Callable
    bm25_retriever = BM25Retriever(corpus)
    doc_map = {d["id"]: d for d in corpus}

    def retrieve(query: str) -> list[str]:
        bm25_ids = bm25_retriever.retrieve(query)[:20]  # top-20 candidates
        # Simulate cross-encoder: boost topic-matched documents
        query_lower = query.lower()
        def score(doc_id):
            doc = doc_map[doc_id]
            # Simple overlap as a cross-encoder proxy for this synthetic demo
            q_words = set(query_lower.split())
            d_words = set(doc["text"].lower().split())
            return len(q_words & d_words)
        reranked = sorted(bm25_ids, key=score, reverse=True)
        return reranked[:10]

    return retrieve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    corpus = CORPUS
    k = 5

    bm25 = BM25Retriever(corpus)
    baseline_fn = bm25.retrieve
    pipeline_fn = make_pipeline_retriever(corpus)

    baseline = evaluate_retrieval(EVAL_CASES, baseline_fn, k=k)
    improved = evaluate_retrieval(EVAL_CASES, pipeline_fn, k=k)

    header = f"{'Metric':<12} {'Baseline':>10} {'Pipeline':>10} {'Delta':>10}"
    log.info("\n%s\n%s", header, "-" * len(header))
    for metric in ("precision", "recall", "mrr", "ndcg"):
        b = getattr(baseline, metric)
        p = getattr(improved, metric)
        delta = p - b
        sign = "+" if delta >= 0 else ""
        log.info("%-12s %10.3f %10.3f %s%10.3f", metric.upper()[:4]+f"@{k}", b, p, sign, delta)

    improved_ndcg = improved.ndcg > baseline.ndcg
    log.info("Improvement observed (NDCG@%d): %s", k, "YES" if improved_ndcg else "NO")


if __name__ == "__main__":
    main()
```

## Checklist verification

Before tagging `p4w5-complete`:

- [ ] `pytest tests/ -v` — all green
- [ ] `ruff check src/ tests/` — zero violations
- [ ] `before_after_eval.py` prints a 4-row metrics table
- [ ] No hardcoded secrets; `RAG_` env vars override all config defaults
- [ ] `pyproject.toml` lists all dependencies
