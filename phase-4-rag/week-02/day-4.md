# Phase 4 · Week 02 · Day 4 — Embedding Models & Similarity

> Objective: Understand what an embedding is, how transformer encoders produce them, key quality metrics, and how to compute similarity correctly.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

An **embedding** is a dense floating-point vector that represents the *meaning* of a piece of text. Two texts with similar meanings have vectors that are close in high-dimensional space; unrelated texts are far apart.

Embeddings are produced by **encoder models** (not the generative decoder you chat with). The encoder reads the full input in one pass using bidirectional attention and outputs a fixed-length vector.

### Key dimensions

| Model | Dims | Max tokens | Notes |
|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | 1536 (or 512 truncated) | 8191 | Fast, cheap |
| `text-embedding-3-large` (OpenAI) | 3072 | 8191 | Higher quality |
| `text-embedding-ada-002` (OpenAI) | 1536 | 8191 | Legacy, still common |
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 | 256 | Local, free |
| `all-mpnet-base-v2` (sentence-transformers) | 768 | 384 | Local, higher quality |
| `embed-english-v3.0` (Cohere) | 1024 | 512 | Good for reranking |

**Dimensionality tradeoff:** higher dims → richer representation, more storage, slower index.  
**Max token truncation:** text exceeding the model's limit is silently truncated — your chunker must guarantee chunks fit.

## How it works internally

### Mean pooling

Transformers produce one vector per token. To get a single sentence vector, **mean pool**: average all token vectors. Some models use the `[CLS]` token instead, but mean pooling is more common in sentence-transformers.

```
sentence_vector = mean(token_vectors)  # shape: (hidden_dim,)
```

### Normalisation

Normalise to unit length before storage:

```
v_norm = v / ||v||
```

After normalisation, **cosine similarity = dot product**. Vector databases assume normalised vectors; always normalise unless you're storing raw for a model that does it internally.

### Cosine vs. Euclidean

- **Cosine similarity**: ignores magnitude, measures angle. Best for semantic similarity (two short and one long paraphrase should match; raw Euclidean would penalise length).
- **Euclidean (L2)**: measures geometric distance. Works well only on unit-normalised vectors (where it is monotonically equivalent to cosine distance).

Always use cosine for text embedding retrieval.

## Worked example

```python
# embeddings/similarity.py
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

Vector = np.ndarray  # shape (D,), dtype float32


def normalise(v: np.ndarray) -> np.ndarray:
    """Return the L2-normalised version of *v*. Safe for zero vectors."""
    norm = np.linalg.norm(v)
    if norm == 0.0:
        logger.warning("Attempted to normalise zero vector — returning as-is.")
        return v
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D arrays. Both should be unit-normalised."""
    return float(np.dot(a, b))


def cosine_similarity_matrix(
    query: np.ndarray,
    corpus: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query vector and a corpus matrix.

    Args:
        query: shape (D,) — single query vector, unit-normalised.
        corpus: shape (N, D) — N corpus vectors, unit-normalised.

    Returns:
        shape (N,) similarities.
    """
    return corpus @ query  # matrix-vector product == batch dot products


def top_k(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    """Return the top-k (index, similarity) pairs from *corpus* for *query*.

    Both query and corpus vectors must be unit-normalised.
    """
    sims = cosine_similarity_matrix(query, corpus)
    indices = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in indices]
```

```python
# embeddings/demo_similarity.py
"""Demonstrates similarity computation with deterministic random embeddings."""
from __future__ import annotations

import logging

import numpy as np
from embeddings.similarity import normalise, top_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rng = np.random.default_rng(42)
DIM = 384


def fake_embed(text: str) -> np.ndarray:
    """Return a deterministic pseudo-embedding based on text hash."""
    seed = hash(text) % (2**32)
    r = np.random.default_rng(seed)
    raw = r.standard_normal(DIM).astype(np.float32)
    return normalise(raw)


corpus_texts = [
    "Photosynthesis converts sunlight to chemical energy.",
    "Chlorophyll is the green pigment in plant cells.",
    "Neural networks learn by adjusting weights via backpropagation.",
    "Gradient descent minimises the loss function iteratively.",
    "The Amazon River has the largest discharge of any river.",
    "South America hosts tropical rainforests with high biodiversity.",
]

corpus_matrix = np.stack([fake_embed(t) for t in corpus_texts])  # (N, D)

query = fake_embed("How do plants capture energy from the sun?")
results = top_k(query, corpus_matrix, k=3)

for rank, (idx, sim) in enumerate(results, 1):
    logger.info("rank %d | sim=%.4f | %s", rank, sim, corpus_texts[idx])
```

```python
# tests/unit/test_similarity.py
import numpy as np
import pytest
from embeddings.similarity import normalise, cosine_similarity, cosine_similarity_matrix, top_k


def test_normalise_unit_vector():
    v = np.array([3.0, 4.0])
    n = normalise(v)
    assert abs(np.linalg.norm(n) - 1.0) < 1e-6


def test_normalise_zero_vector():
    v = np.zeros(4)
    n = normalise(v)
    assert np.allclose(n, v)


def test_cosine_similarity_identical():
    v = normalise(np.array([1.0, 2.0, 3.0]))
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = normalise(np.array([1.0, 0.0]))
    b = normalise(np.array([0.0, 1.0]))
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_top_k_returns_correct_count():
    rng = np.random.default_rng(0)
    corpus = np.stack([normalise(rng.standard_normal(8).astype(np.float32)) for _ in range(20)])
    query = normalise(rng.standard_normal(8).astype(np.float32))
    result = top_k(query, corpus, k=5)
    assert len(result) == 5


def test_top_k_sorted_descending():
    rng = np.random.default_rng(1)
    corpus = np.stack([normalise(rng.standard_normal(8).astype(np.float32)) for _ in range(10)])
    query = normalise(rng.standard_normal(8).astype(np.float32))
    result = top_k(query, corpus, k=10)
    sims = [s for _, s in result]
    assert sims == sorted(sims, reverse=True)
```

## Exercises

1. Implement `pairwise_similarity_matrix(vectors: np.ndarray) -> np.ndarray` that returns the full N×N similarity matrix. Use matrix multiplication; avoid a Python loop.
2. Add a `threshold` parameter to `top_k` that filters out results below it.
3. Explain in a comment why `corpus @ query` gives cosine similarities when both are unit-normalised.
4. Given embedding dimension 1536, compute the storage in MB for 1 million chunks (float32). Show your working.
5. Write a test that verifies `cosine_similarity(a, b) == cosine_similarity(b, a)` (symmetry).

(Solutions: ../solutions/week-02/day-4.md)

## Self-check

- [ ] I can explain what mean pooling is and why normalisation is required before dot-product similarity.
- [ ] I can compute cosine similarity by hand for a 2-D example.
- [ ] I can implement `top_k` retrieval against a corpus matrix without a Python loop.
- [ ] I can estimate embedding storage cost for a large corpus.
- [ ] My tests cover identical, orthogonal, and opposite vector cases.

## Supplementary links

- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [sentence-transformers model cards](https://www.sbert.net/docs/pretrained_models.html)
- [MTEB leaderboard — embedding benchmarks](https://huggingface.co/spaces/mteb/leaderboard)
- [Understanding embeddings (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/)
