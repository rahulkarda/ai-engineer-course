# Phase 4 · Week 02 · Day 3 — Semantic Chunking

> Objective: Implement semantic chunking using embedding similarity to detect topic shifts, and understand when to choose it over structural approaches.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

Every strategy so far splits on **form** (characters, punctuation, headings). Semantic chunking splits on **meaning**: when consecutive sentences discuss different topics, that's a chunk boundary.

The algorithm:
1. Split text into sentences.
2. Embed each sentence (or a small sliding window of sentences).
3. Compute cosine similarity between adjacent sentence embeddings.
4. Where similarity **drops below a threshold** (a "valley"), insert a chunk boundary.
5. Collect sentences between boundaries into chunks.

This produces chunks with high **intra-chunk cohesion** and low **inter-chunk similarity** — the ideal property for a retrieval index where each chunk encodes one idea.

**When to use it:**

| Scenario | Best strategy |
|---|---|
| Dense technical docs, mixed topics | Semantic |
| Markdown / structured docs | Structure-aware |
| Uniform prose (books, articles) | Sentence or recursive |
| Code files | Structure-aware (AST) |
| Quick prototype, no GPU | Fixed-size |

## How it works internally

### Cosine similarity

Given two embedding vectors **a** and **b**:

```
cos_sim(a, b) = dot(a, b) / (||a|| * ||b||)
```

Values range from -1 (opposite) to 1 (identical). Topic boundaries show up as local minima in the similarity time series across sentences.

### Breakpoint detection

Two common strategies:

- **Percentile threshold**: find the Pth percentile of all similarity scores; treat scores below it as breakpoints. Adaptive: always finds *some* structure.
- **Fixed threshold**: any drop below a fixed value (e.g., 0.75) is a breakpoint. Predictable but needs tuning per domain.

### Sliding window smoothing

A sentence-by-sentence similarity is noisy. Smooth by embedding a window of 2–3 sentences together, then compare adjacent windows. This amortizes noisy single-sentence embeddings.

## Worked example

The embedding calls are mocked so this runs without any API key or model download.

```python
# chunking/semantic.py
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Vector = list[float]


class EmbedFn(Protocol):
    """Anything callable that maps a list of strings → list of vectors."""
    def __call__(self, texts: list[str]) -> list[Vector]: ...


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: Vector) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Return cosine similarity in [-1, 1]. Returns 0.0 for zero vectors."""
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


# ---------------------------------------------------------------------------
# Sentence splitting (reuse from day-1 approach)
# ---------------------------------------------------------------------------

import re

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(text.strip()) if s.strip()]


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _window_texts(sentences: list[str], window: int) -> list[str]:
    """Return sliding-window sentence groups for embedding."""
    result: list[str] = []
    for i in range(len(sentences)):
        lo = max(0, i - window // 2)
        hi = min(len(sentences), i + window // 2 + 1)
        result.append(" ".join(sentences[lo:hi]))
    return result


def _find_breakpoints(similarities: list[float], percentile: float) -> list[int]:
    """Return indices where similarity drops below the Pth percentile."""
    if not similarities:
        return []
    sorted_sims = sorted(similarities)
    idx = max(0, int(len(sorted_sims) * percentile / 100) - 1)
    threshold = sorted_sims[idx]
    return [i for i, s in enumerate(similarities) if s < threshold]


def semantic_chunks(
    text: str,
    embed_fn: EmbedFn,
    breakpoint_percentile: float = 25.0,
    window_size: int = 1,
    min_chunk_sentences: int = 2,
) -> list[Chunk]:
    """Split *text* into semantically coherent chunks.

    Args:
        text: Input text.
        embed_fn: Callable mapping list[str] → list[Vector].
        breakpoint_percentile: Percentile threshold for breakpoint detection.
            Lower values → more (smaller) chunks.
        window_size: Sentences on each side of current sentence to include
            in the embedding window.
        min_chunk_sentences: Minimum sentences per chunk (prevents 1-sentence chunks).

    Returns:
        Ordered list of Chunk objects with ``similarity_at_break`` metadata.
    """
    sentences = _sentences(text)
    if len(sentences) < 2:
        return [Chunk(text=text, index=0)]

    window_texts = _window_texts(sentences, window_size)
    embeddings = embed_fn(window_texts)

    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)
        logger.debug("sim[%d→%d]=%.4f", i, i + 1, sim)

    break_indices = set(_find_breakpoints(similarities, breakpoint_percentile))

    chunks: list[Chunk] = []
    current: list[str] = []
    chunk_idx = 0

    for i, sent in enumerate(sentences):
        current.append(sent)
        is_break = i in break_indices and len(current) >= min_chunk_sentences
        is_last = i == len(sentences) - 1
        if is_break or is_last:
            chunk_text = " ".join(current)
            sim_val = similarities[i] if i < len(similarities) else None
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=chunk_idx,
                    metadata={"similarity_at_break": sim_val},
                )
            )
            chunk_idx += 1
            current = []

    logger.info("semantic_chunks: %d breakpoints → %d chunks", len(break_indices), len(chunks))
    return chunks
```

```python
# Demo with a deterministic mock embedder
if __name__ == "__main__":
    import logging
    import math
    from chunking.semantic import semantic_chunks, Vector

    logging.basicConfig(level=logging.INFO)

    # Mock embedder: each sentence gets a "topic hash" vector.
    # Sentences starting with the same word cluster together.
    def mock_embed(texts: list[str]) -> list[Vector]:
        vecs: list[Vector] = []
        for t in texts:
            # 4-dim hash: simulate topic shift by flipping dimensions.
            first_word = t.split()[0].lower() if t.split() else "x"
            h = hash(first_word) % 256 / 256.0
            vecs.append([math.cos(h), math.sin(h), math.cos(2 * h), math.sin(2 * h)])
        return vecs

    corpus = (
        "Photosynthesis converts sunlight into sugar. "
        "Chlorophyll absorbs red and blue light. "
        "Leaves are green because chlorophyll reflects green. "
        "Machine learning optimises a loss function. "
        "Gradient descent iteratively adjusts weights. "
        "Backpropagation computes gradients efficiently. "
        "The Amazon river is the largest by discharge. "
        "It flows through nine countries in South America."
    )

    for c in semantic_chunks(corpus, mock_embed, breakpoint_percentile=30, window_size=1):
        print(f"[chunk {c.index}] {c.text[:80]}")
```

## Exercises

1. Replace `_find_breakpoints` with a **fixed threshold** variant: take a `threshold: float` parameter instead of a percentile. Write a unit test comparing both variants on the same input.
2. Add a `max_chunk_chars: int` guard to `semantic_chunks`. If a semantically detected chunk exceeds it, sub-split using the sentence chunker from Day 1.
3. Implement `cosine_similarity` using `numpy` dot and norm operations. Benchmark it against the pure-Python version on 1,000-vector pairs.
4. Write `tests/unit/test_semantic.py`. Use a deterministic mock embedder that returns pre-defined vectors so tests are reproducible. Verify: (a) single-sentence input returns one chunk; (b) a clear topic shift produces two chunks; (c) `min_chunk_sentences` is respected.
5. (Stretch) Implement `_window_texts` with `window_size=2` and verify that the number of detected breakpoints decreases vs. `window_size=0` on the same corpus.

(Solutions: ../solutions/week-02/day-3.md)

## Self-check

- [ ] I can derive the cosine similarity formula and explain why normalisation is needed.
- [ ] I can explain percentile-based breakpoint detection and its adaptive advantage.
- [ ] I can implement `semantic_chunks` without reference, wiring in any callable embedder.
- [ ] My tests use a mock embedder — no real API calls.
- [ ] I understand when semantic chunking outperforms structural chunking and vice versa.

## Supplementary links

- [Greg Kamradt — semantic chunking notebook](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [LangChain SemanticChunker source](https://github.com/langchain-ai/langchain-experimental/blob/main/libs/experimental/langchain_experimental/text_splitter.py)
- [Cosine similarity explained (3Blue1Brown style)](https://en.wikipedia.org/wiki/Cosine_similarity)
