# Phase 4 · Week 05 · Day 3 — MMR: Maximal Marginal Relevance for Diversity

> Objective: Implement MMR to balance relevance and diversity in retrieval results, preventing redundant documents from flooding the top-K.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

After reranking, the top results are often *redundant*. If three documents all say "cross-encoders produce a scalar score from concatenated input", returning all three wastes context window and confuses the reader.

**Maximal Marginal Relevance (MMR)** solves this with a greedy selection loop:

```
At each step, pick the candidate that maximises:
  λ · sim(candidate, query)  −  (1 − λ) · max_sim(candidate, already_selected)
```

- `sim(candidate, query)`: relevance to the query (high = good).
- `max_sim(candidate, already_selected)`: similarity to the most similar already-chosen document (high = redundant).
- `λ` (lambda): trade-off parameter. λ=1 → pure relevance (no diversity). λ=0 → pure diversity (no relevance). λ=0.5 is a common default.

The algorithm:
1. Start with an empty selected set S and full candidate set C.
2. Always add the highest-relevance candidate first (ignores redundancy for first pick).
3. For each remaining slot: score all remaining candidates with the MMR formula; add the winner to S, remove from C.
4. Stop when |S| = k.

This is O(k · |C|) — fast enough for hundreds of candidates.

## How it works internally

MMR operates on *embedding space*, not on cross-encoder scores. You need:
- `query_emb`: the query embedding (bi-encoder).
- `doc_embs`: embeddings for each candidate.
- Cosine similarity function.

Cross-encoder scores and MMR are complementary:
- Cross-encoder: tells you *how relevant* each document is.
- MMR: selects a *diverse relevant* subset.

Common pipeline position:

```
top-K candidates (fast retriever)
       │
  cross-encoder rerank         ← Day 2
       │
  MMR diversity filter         ← Today
       │
  top-N diverse relevant docs  ─► LLM context window
```

Alternatively, run MMR directly on the bi-encoder embeddings of the reranked candidates, using the cross-encoder scores as the relevance term instead of cosine-to-query. This gives you a score-aware MMR.

### Similarity choices

- **Cosine similarity**: standard for normalised embeddings. `cos(a, b) = a · b` when ‖a‖ = ‖b‖ = 1.
- **Dot product**: equivalent to cosine for unit vectors, faster.
- **MMR never uses cross-encoder scores directly for the redundancy term** — cross-encoder cannot compare two documents without a query.

## Worked example

```python
"""
day3_mmr.py

Maximal Marginal Relevance implementation.
Uses only numpy — no model calls in the core algorithm.
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MMRResult:
    index: int        # position in original candidates list
    relevance: float  # sim(candidate, query)
    mmr_score: float  # score at selection time
    text: str


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def mmr_select(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    texts: list[str],
    k: int,
    lambda_mult: float = 0.5,
) -> list[MMRResult]:
    """
    Select k documents from candidates using Maximal Marginal Relevance.

    Parameters
    ----------
    query_emb:
        Unit-normalised query embedding, shape (D,).
    doc_embs:
        Unit-normalised document embeddings, shape (N, D).
    texts:
        Corresponding document strings, length N.
    k:
        Number of documents to select.
    lambda_mult:
        Trade-off between relevance (1.0) and diversity (0.0).

    Returns
    -------
    List of MMRResult in selection order (first = highest MMR at each step).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not (0.0 <= lambda_mult <= 1.0):
        raise ValueError(f"lambda_mult must be in [0, 1], got {lambda_mult}")

    n = len(texts)
    k = min(k, n)

    # Relevance scores: cosine sim to query (shape N,)
    relevance: np.ndarray = doc_embs @ query_emb  # works for unit vectors

    remaining: set[int] = set(range(n))
    selected_indices: list[int] = []
    results: list[MMRResult] = []

    for step in range(k):
        if step == 0:
            # First pick: pure relevance
            best_idx = int(np.argmax(relevance))
            best_score = float(relevance[best_idx])
        else:
            # Compute redundancy: max cosine sim to any already-selected doc
            selected_embs = doc_embs[selected_indices]  # shape (step, D)
            best_score = -1e9
            best_idx = -1
            for idx in remaining:
                rel = float(relevance[idx])
                # Similarity to most similar selected doc
                sims_to_selected: np.ndarray = selected_embs @ doc_embs[idx]
                redundancy = float(np.max(sims_to_selected))
                score = lambda_mult * rel - (1.0 - lambda_mult) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx

        selected_indices.append(best_idx)
        remaining.discard(best_idx)
        results.append(
            MMRResult(
                index=best_idx,
                relevance=float(relevance[best_idx]),
                mmr_score=best_score,
                text=texts[best_idx],
            )
        )
        log.debug("Step %d: selected idx=%d score=%.4f", step, best_idx, best_score)

    return results


# ---------------------------------------------------------------------------
# Convenience wrapper: MMR on raw text (loads encoder internally)
# ---------------------------------------------------------------------------
def mmr_from_texts(
    query: str,
    candidates: list[str],
    k: int,
    lambda_mult: float = 0.5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[MMRResult]:
    """Encode query + candidates then run MMR. Convenience function."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    all_texts = [query] + candidates
    embs = model.encode(all_texts, normalize_embeddings=True)
    query_emb = embs[0]
    doc_embs = embs[1:]
    return mmr_select(query_emb, doc_embs, candidates, k=k, lambda_mult=lambda_mult)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    candidates = [
        "Cross-encoders score query-document pairs jointly.",
        "Cross-encoders attend to both query and document simultaneously.",
        "A cross-encoder takes concatenated input and outputs a relevance scalar.",
        "MMR selects a diverse subset of relevant documents.",
        "Retrieval pipelines have a recall stage and a precision stage.",
        "BM25 computes term frequency and inverse document frequency.",
    ]
    query = "How does cross-encoder scoring work?"

    results = mmr_from_texts(query, candidates, k=3, lambda_mult=0.7)
    for r in results:
        log.info("relevance=%.3f mmr=%.3f  %s", r.relevance, r.mmr_score, r.text)


if __name__ == "__main__":
    main()
```

### Tests

```python
"""
tests/unit/test_mmr.py

Tests for mmr_select. Uses synthetic embeddings — no model loading.
"""

import numpy as np
import pytest

from day3_mmr import MMRResult, mmr_select


def unit_vec(v: list[float]) -> np.ndarray:
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


@pytest.fixture()
def embeddings():
    """
    Three 3-D unit vectors.
    Docs 0 and 1 are nearly identical; doc 2 is orthogonal to both.
    """
    q = unit_vec([1.0, 0.0, 0.0])
    d0 = unit_vec([1.0, 0.1, 0.0])   # very relevant, similar to d1
    d1 = unit_vec([1.0, 0.15, 0.0])  # very relevant, similar to d0
    d2 = unit_vec([0.0, 1.0, 0.0])   # less relevant, very different
    texts = ["doc0", "doc1", "doc2"]
    doc_embs = np.stack([d0, d1, d2])
    return q, doc_embs, texts


def test_first_result_most_relevant(embeddings):
    q, doc_embs, texts = embeddings
    results = mmr_select(q, doc_embs, texts, k=1, lambda_mult=0.5)
    assert len(results) == 1
    # doc0 or doc1 (both high relevance) should be first
    assert results[0].index in (0, 1)


def test_diversity_avoids_duplicate(embeddings):
    """With lambda < 0.5, second pick should be diverse doc2 not the clone."""
    q, doc_embs, texts = embeddings
    results = mmr_select(q, doc_embs, texts, k=2, lambda_mult=0.3)
    indices = [r.index for r in results]
    # First pick: one of the relevant docs (0 or 1)
    # Second pick: with high diversity weight, doc2 beats the redundant twin
    assert indices[1] == 2


def test_lambda_1_gives_pure_relevance(embeddings):
    """lambda=1 → ranking purely by relevance, ignore diversity."""
    q, doc_embs, texts = embeddings
    results = mmr_select(q, doc_embs, texts, k=3, lambda_mult=1.0)
    # All docs returned in relevance order
    relevances = [r.relevance for r in results]
    assert relevances == sorted(relevances, reverse=True)


def test_k_larger_than_n_capped(embeddings):
    q, doc_embs, texts = embeddings
    results = mmr_select(q, doc_embs, texts, k=100)
    assert len(results) == 3


def test_invalid_lambda_raises():
    q = np.array([1.0, 0.0])
    embs = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        mmr_select(q, embs, ["x"], k=1, lambda_mult=1.5)
```

## Exercises

1. Modify `mmr_select` to accept an optional `relevance_scores: list[float] | None` parameter. When provided, use those scores (e.g., cross-encoder scores) as the relevance term instead of cosine-to-query. Write a test for this path.
2. Add logging at each step that shows: which document was selected, its relevance score, its redundancy score, and its final MMR score. Verify with a test that the log messages contain the expected values using `caplog`.
3. Implement a `lambda_sweep` function that runs MMR with λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} and returns a dict mapping λ to the selected document indices. Useful for hyperparameter exploration.

(Solutions: ../solutions/week-05/day-3.md)

## Self-check

- [ ] I can write the MMR formula from memory.
- [ ] I can implement the greedy MMR loop without reference.
- [ ] I understand why λ=1 degenerates to pure relevance ranking.
- [ ] I can explain why cross-encoder scores cannot be used for the redundancy term.
- [ ] I can reason about when to place MMR in the pipeline (before or after reranking).

## Supplementary links

- [MMR original paper (Carbonell & Goldstein 1998)](https://dl.acm.org/doi/10.1145/290941.291025)
- [LangChain MMR implementation for reference](https://python.langchain.com/docs/how_to/maximal_marginal_relevance_search/)
- [FAISS similarity search docs](https://faiss.ai/index.html)
