# Phase 4 · Week 05 · Day 1 — Why Rerank? Bi-encoders vs Cross-encoders

> Objective: Understand the retrieval quality gap that reranking closes and the architectural difference between bi-encoders and cross-encoders.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-1]] (hybrid retrieval)

## Concept

Retrieval pipelines are a two-stage problem. The first stage (recall) fetches candidates fast. The second stage (rerank) scores those candidates precisely.

**Why the gap exists**

Bi-encoder retrievers — the backbone of semantic search — embed query and document *independently* and compare with dot-product or cosine similarity. This is fast: both embeddings are precomputed, and similarity is a single matrix operation. But the independence is also the weakness: neither embedding sees the other, so fine-grained token-level interactions between query terms and document phrases are invisible.

Cross-encoders take a different path. They receive `(query, document)` as a *single concatenated input* and produce a scalar relevance score. Every attention head attends jointly to both sequences. This captures phrasing, negation, numeric specificity, and co-reference — things a bi-encoder silently skips.

The catch: cross-encoders cannot precompute document representations. Scoring 1 000 documents at query time costs 1 000 forward passes through a transformer. That is too slow for a corpus of millions; it is fine as a second-pass filter over the 50–200 candidates your fast retriever already surfaced.

**The standard two-stage pattern**

```
Query
  │
  ├─► Fast retrieval (BM25 + bi-encoder / hybrid)  ─► top-K candidates (K=50..200)
  │
  └─► Cross-encoder reranker                       ─► top-N final results  (N=3..10)
```

Stage 1 optimises *recall* — get the right documents into the set.  
Stage 2 optimises *precision* — push the best documents to the top.

## How it works internally

### Bi-encoder (recall stage)

```
query ──► encoder_Q ──► q_vec  (dim D)
doc   ──► encoder_D ──► d_vec  (dim D)
score = q_vec · d_vec
```

- Encoder weights may be shared or separate.
- Document vectors are precomputed and indexed (FAISS, pgvector, Qdrant).
- At query time: one encoder call + ANN search. Sub-millisecond at scale.
- Training signal: contrastive loss (InfoNCE) on (query, positive, negatives).

### Cross-encoder (rerank stage)

```
[CLS] query [SEP] document [SEP]
         │
    BERT-style encoder
         │
    linear head ──► scalar score
```

- Full attention across both sequences — every query token attends to every document token.
- No precomputation possible: each `(query, doc)` pair is a fresh forward pass.
- Training signal: pointwise (MSE on relevance labels) or pairwise (margin ranking loss).
- Typical models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `cross-encoder/ms-marco-electra-base` (accurate).

### Why precision improves

A bi-encoder for "Python exceptions best practices" maps "best practices" and "tips and tricks" to nearby vectors. A cross-encoder processing "Python exceptions best practices [SEP] Top 10 tips and tricks" can see that "tips" ≠ "practices" and score accordingly. The joint attention captures nuance the independent embeddings flatten.

## Worked example

```python
"""
day1_biencoder_vs_crossencoder.py

Demonstrates the quality difference between bi-encoder ranking
and cross-encoder reranking on a small toy corpus.
"""

import logging
import os
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Toy corpus — RAG engineering domain
# ---------------------------------------------------------------------------
CORPUS: list[str] = [
    "Chunking strategy affects retrieval recall significantly.",
    "Cross-encoders rerank candidate documents by scoring query-document pairs jointly.",
    "BM25 uses term frequency and inverse document frequency for keyword matching.",
    "Hybrid retrieval combines dense vectors with sparse BM25 scores.",
    "Cross-encoder models attend to both query and document tokens simultaneously.",
    "Vector databases store embeddings and support approximate nearest-neighbour search.",
    "Reranking improves precision by re-scoring the top-K candidates from a fast retriever.",
]

QUERY = "How do cross-encoders improve retrieval precision?"

# ---------------------------------------------------------------------------
# Stage 1: Bi-encoder ranking
# ---------------------------------------------------------------------------
@dataclass
class RankedDoc:
    rank: int
    score: float
    text: str


def biencoder_rank(
    query: str,
    corpus: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[RankedDoc]:
    model = SentenceTransformer(model_name)
    log.info("Encoding corpus with bi-encoder …")
    doc_embs = model.encode(corpus, normalize_embeddings=True)
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores: np.ndarray = doc_embs @ q_emb
    order = np.argsort(scores)[::-1]
    return [
        RankedDoc(rank=i + 1, score=float(scores[idx]), text=corpus[idx])
        for i, idx in enumerate(order)
    ]


# ---------------------------------------------------------------------------
# Stage 2: Cross-encoder rerank
# ---------------------------------------------------------------------------
def crossencoder_rerank(
    query: str,
    candidates: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[RankedDoc]:
    model = CrossEncoder(model_name)
    log.info("Reranking %d candidates with cross-encoder …", len(candidates))
    pairs = [[query, doc] for doc in candidates]
    scores: np.ndarray = model.predict(pairs)
    order = np.argsort(scores)[::-1]
    return [
        RankedDoc(rank=i + 1, score=float(scores[idx]), text=candidates[idx])
        for i, idx in enumerate(order)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Query: %s", QUERY)

    bi_ranked = biencoder_rank(QUERY, CORPUS)
    log.info("\n=== Bi-encoder ranking ===")
    for doc in bi_ranked:
        log.info("#%d  score=%.4f  %s", doc.rank, doc.score, doc.text[:70])

    # Take top-5 as candidates for reranking (simulating real pipeline)
    top5 = [d.text for d in bi_ranked[:5]]
    ce_ranked = crossencoder_rerank(QUERY, top5)
    log.info("\n=== Cross-encoder reranking of top-5 ===")
    for doc in ce_ranked:
        log.info("#%d  score=%.4f  %s", doc.rank, doc.score, doc.text[:70])


if __name__ == "__main__":
    main()
```

**Install deps (do not run yet — just note the command):**

```bash
uv add sentence-transformers
```

**What to observe:**  
The bi-encoder often surfaces "Hybrid retrieval combines dense vectors…" in top-3 because "dense" is close to embedding-space concepts. The cross-encoder demotes it because it doesn't actually answer the question about precision, and promotes the two cross-encoder-specific sentences.

## Exercises

1. Swap the bi-encoder model to `"BAAI/bge-small-en-v1.5"`. Does the ranking order change? Log the top-3 before and after.
2. Add two deliberately misleading documents to `CORPUS` — ones that contain the exact words "cross-encoder" and "precision" but describe something unrelated (e.g., signal processing). Run the full pipeline and observe whether the cross-encoder correctly demotes them.
3. Modify `crossencoder_rerank` to accept a `top_n: int` parameter and return only the top-N results. Write a `pytest` unit test that mocks `CrossEncoder.predict` and asserts correct ranking order.

(Solutions: ../solutions/week-05/day-1.md)

## Self-check

- [ ] I can explain why bi-encoders are fast but imprecise.
- [ ] I can describe what "joint attention" means in a cross-encoder.
- [ ] I can sketch the two-stage retrieval pipeline from memory.
- [ ] I can explain why cross-encoder inference cannot be precomputed.
- [ ] I can implement the two-stage pipeline end-to-end without reference.

## Supplementary links

- [Sentence-Transformers pretrained models](https://www.sbert.net/docs/pretrained_models.html)
- [MS MARCO cross-encoder models on HuggingFace](https://huggingface.co/cross-encoder)
- [BEIR benchmark — retrieval evaluation](https://github.com/beir-cellar/beir)
