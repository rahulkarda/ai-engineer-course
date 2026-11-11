# Phase 4 · Week 02 · Day 6 — Build: Configurable Chunking + Embedding Module

> Objective: Build a production-quality, configurable chunking and embedding pipeline with multiple strategies, a mockable embedder, and a strategy comparison harness.  ·  Time: ~4 hr

## What you're building

A `rag_pipeline` Python package with:

```
rag_pipeline/
├── __init__.py
├── chunking/
│   ├── __init__.py
│   ├── base.py          # ChunkerProtocol, Chunk dataclass
│   ├── fixed.py         # FixedSizeChunker
│   ├── sentence.py      # SentenceChunker
│   ├── recursive.py     # RecursiveChunker
│   ├── markdown.py      # MarkdownChunker
│   └── semantic.py      # SemanticChunker
├── embeddings/
│   ├── __init__.py
│   ├── providers.py     # EmbeddingProvider protocol + Mock/OpenAI/ST providers
│   ├── similarity.py    # cosine_similarity, top_k, normalise
│   └── cost.py          # estimate_cost
├── pipeline.py          # ChunkEmbedPipeline — runs chunker + embedder end-to-end
└── compare.py           # strategy_comparison — quality metrics across chunkers
tests/
├── unit/
│   ├── test_fixed.py
│   ├── test_sentence.py
│   ├── test_recursive.py
│   ├── test_markdown.py
│   ├── test_semantic.py
│   ├── test_similarity.py
│   └── test_providers.py
└── integration/
    └── test_pipeline.py
pyproject.toml
.env.example
```

## Step 1 — Project scaffold

```bash
mkdir rag_pipeline && cd rag_pipeline
uv init --name rag_pipeline --python 3.12
uv add numpy pytest
# When you want real providers:
# uv add openai sentence-transformers
```

`pyproject.toml` — add to `[tool.pytest.ini_options]`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

`.env.example`:
```
OPENAI_API_KEY=your-key-here
```

## Step 2 — Core types (`chunking/base.py`)

```python
# rag_pipeline/chunking/base.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Chunk:
    """A single text chunk with positional index and arbitrary metadata."""

    text: str
    index: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def token_estimate(self, chars_per_token: int = 4) -> int:
        """Heuristic token count: 1 token ≈ chars_per_token characters."""
        return len(self.text) // chars_per_token


@runtime_checkable
class ChunkerProtocol(Protocol):
    """All chunkers must be callable: chunker(text) → list[Chunk]."""

    def __call__(self, text: str) -> list[Chunk]: ...
```

## Step 3 — Chunkers

Copy the implementations from Days 1–3, wrapping each in a class that implements `ChunkerProtocol`:

```python
# rag_pipeline/chunking/fixed.py
from __future__ import annotations

import logging

from rag_pipeline.chunking.base import Chunk, ChunkerProtocol

logger = logging.getLogger(__name__)


class FixedSizeChunker:
    """Split text into fixed-character chunks with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not (0 <= overlap < chunk_size):
            raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def __call__(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        start, idx = 0, 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(Chunk(text=text[start:end], index=idx))
            start += step
            idx += 1
        logger.info("FixedSizeChunker: %d chunks", len(chunks))
        return chunks
```

Follow the same pattern for `SentenceChunker`, `RecursiveChunker`, `MarkdownChunker`. For `SemanticChunker`, inject the embedder at construction:

```python
# rag_pipeline/chunking/semantic.py  (class wrapper)
from __future__ import annotations

import logging
from rag_pipeline.chunking.base import Chunk
from rag_pipeline.embeddings.providers import EmbeddingProvider
# ... import helpers from day-3 implementation

logger = logging.getLogger(__name__)


class SemanticChunker:
    def __init__(
        self,
        provider: EmbeddingProvider,
        breakpoint_percentile: float = 25.0,
        window_size: int = 1,
        min_chunk_sentences: int = 2,
    ) -> None:
        self.provider = provider
        self.breakpoint_percentile = breakpoint_percentile
        self.window_size = window_size
        self.min_chunk_sentences = min_chunk_sentences

    def __call__(self, text: str) -> list[Chunk]:
        # Delegate to the functional implementation from day-3,
        # passing self.provider.embed as the embed_fn.
        from rag_pipeline.chunking._semantic_impl import semantic_chunks
        return semantic_chunks(
            text,
            embed_fn=self.provider.embed,
            breakpoint_percentile=self.breakpoint_percentile,
            window_size=self.window_size,
            min_chunk_sentences=self.min_chunk_sentences,
        )
```

## Step 4 — Pipeline (`pipeline.py`)

```python
# rag_pipeline/pipeline.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from rag_pipeline.chunking.base import Chunk, ChunkerProtocol
from rag_pipeline.embeddings.providers import EmbeddingProvider
from rag_pipeline.embeddings.similarity import normalise

logger = logging.getLogger(__name__)


@dataclass
class IndexedChunk:
    chunk: Chunk
    embedding: np.ndarray  # shape (D,), unit-normalised


class ChunkEmbedPipeline:
    """Run a chunker and an embedder end-to-end to produce an indexed chunk list."""

    def __init__(
        self,
        chunker: ChunkerProtocol,
        provider: EmbeddingProvider,
        embed_batch_size: int = 100,
    ) -> None:
        self.chunker = chunker
        self.provider = provider
        self.embed_batch_size = embed_batch_size

    def run(self, text: str, metadata: dict | None = None) -> list[IndexedChunk]:
        """Chunk *text* and embed each chunk.

        Args:
            text: Source document text.
            metadata: Optional metadata merged into every chunk's metadata.

        Returns:
            List of IndexedChunk with chunk text, original metadata, and embedding.
        """
        chunks = self.chunker(text)
        if metadata:
            for c in chunks:
                c.metadata.update(metadata)

        texts = [c.text for c in chunks]
        raw_vectors = self.provider.embed_batch(texts, batch_size=self.embed_batch_size)

        indexed: list[IndexedChunk] = []
        for chunk, vec in zip(chunks, raw_vectors):
            arr = np.array(vec, dtype=np.float32)
            indexed.append(IndexedChunk(chunk=chunk, embedding=normalise(arr)))

        logger.info("ChunkEmbedPipeline: %d chunks → %d embeddings", len(chunks), len(indexed))
        return indexed
```

## Step 5 — Strategy comparison (`compare.py`)

```python
# rag_pipeline/compare.py
"""Compare chunking strategies on quality metrics."""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass

import numpy as np

from rag_pipeline.chunking.base import Chunk, ChunkerProtocol
from rag_pipeline.embeddings.providers import EmbeddingProvider
from rag_pipeline.embeddings.similarity import cosine_similarity_matrix
from rag_pipeline.pipeline import ChunkEmbedPipeline

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    strategy_name: str
    num_chunks: int
    mean_chars: float
    stdev_chars: float
    min_chars: int
    max_chars: int
    mean_intra_sim: float   # average cosine sim between adjacent chunks (lower = better separation)
    coverage_ratio: float   # total chars in chunks / original chars (should be ~1)


def evaluate_strategy(
    name: str,
    chunker: ChunkerProtocol,
    provider: EmbeddingProvider,
    text: str,
) -> StrategyMetrics:
    """Run *chunker* on *text* and compute quality metrics."""
    pipeline = ChunkEmbedPipeline(chunker, provider)
    indexed = pipeline.run(text)

    chunks = [ic.chunk for ic in indexed]
    embeddings = np.stack([ic.embedding for ic in indexed])  # (N, D)

    char_counts = [len(c) for c in chunks]
    total_chars = sum(char_counts)

    # Intra-chunk similarity: adjacent chunk pairs.
    adj_sims: list[float] = []
    for i in range(len(embeddings) - 1):
        s = float(np.dot(embeddings[i], embeddings[i + 1]))
        adj_sims.append(s)

    return StrategyMetrics(
        strategy_name=name,
        num_chunks=len(chunks),
        mean_chars=statistics.mean(char_counts),
        stdev_chars=statistics.stdev(char_counts) if len(char_counts) > 1 else 0.0,
        min_chars=min(char_counts),
        max_chars=max(char_counts),
        mean_intra_sim=statistics.mean(adj_sims) if adj_sims else 0.0,
        coverage_ratio=total_chars / max(len(text), 1),
    )


def compare_strategies(
    strategies: dict[str, ChunkerProtocol],
    provider: EmbeddingProvider,
    text: str,
) -> list[StrategyMetrics]:
    """Evaluate multiple strategies and return sorted metrics."""
    results = [evaluate_strategy(name, chunker, provider, text) for name, chunker in strategies.items()]
    results.sort(key=lambda m: m.mean_intra_sim)  # lower adjacent similarity = better separation
    for m in results:
        logger.info(
            "%s: chunks=%d mean_chars=%.0f stdev=%.0f adj_sim=%.4f coverage=%.3f",
            m.strategy_name, m.num_chunks, m.mean_chars, m.stdev_chars,
            m.mean_intra_sim, m.coverage_ratio,
        )
    return results
```

## Step 6 — Integration test

```python
# tests/integration/test_pipeline.py
import pytest
from rag_pipeline.chunking.fixed import FixedSizeChunker
from rag_pipeline.chunking.sentence import SentenceChunker
from rag_pipeline.embeddings.providers import MockEmbeddingProvider
from rag_pipeline.pipeline import ChunkEmbedPipeline
from rag_pipeline.compare import compare_strategies
import numpy as np


SAMPLE_TEXT = """
Retrieval-Augmented Generation improves LLM accuracy by injecting retrieved context.
The retrieval step depends critically on how well documents are chunked and embedded.
Fixed-size chunking is simple but can cut sentences in half.
Sentence-aware chunking respects natural language boundaries.
Semantic chunking groups sentences by topic similarity.
Each strategy has distinct precision-recall tradeoffs that depend on document type.
""".strip()


@pytest.fixture
def provider():
    return MockEmbeddingProvider()


def test_pipeline_produces_indexed_chunks(provider):
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    pipeline = ChunkEmbedPipeline(chunker, provider)
    result = pipeline.run(SAMPLE_TEXT)
    assert len(result) > 0
    for ic in result:
        assert len(ic.embedding) == provider.dimensions
        assert abs(np.linalg.norm(ic.embedding) - 1.0) < 1e-5


def test_pipeline_merges_metadata(provider):
    chunker = FixedSizeChunker(chunk_size=200, overlap=0)
    pipeline = ChunkEmbedPipeline(chunker, provider)
    result = pipeline.run(SAMPLE_TEXT, metadata={"source": "test_doc"})
    for ic in result:
        assert ic.chunk.metadata.get("source") == "test_doc"


def test_compare_strategies(provider):
    strategies = {
        "fixed": FixedSizeChunker(chunk_size=150, overlap=30),
        "sentence": SentenceChunker(max_chars=200, sentence_overlap=1),
    }
    metrics = compare_strategies(strategies, provider, SAMPLE_TEXT)
    assert len(metrics) == 2
    names = {m.strategy_name for m in metrics}
    assert names == {"fixed", "sentence"}
```

## Exercises

1. Add `RecursiveChunker` and `MarkdownChunker` to the `compare_strategies` call in the integration test. Observe how `stdev_chars` differs.
2. Add a `query_top_k(query_text: str, indexed: list[IndexedChunk], k: int, provider: EmbeddingProvider)` method to `ChunkEmbedPipeline` that embeds the query and returns the top-k chunks.
3. Add a `coverage_ratio` assertion in `test_pipeline.py` — verify the total embedded chars are within 10% of the original text length.
4. Implement `__repr__` for `StrategyMetrics` that prints a one-line summary suitable for a comparison table.
5. Profile `ChunkEmbedPipeline.run` on a 50,000-character text with `FixedSizeChunker`. Which step takes the most time: chunking or embedding?

(Solutions: ../solutions/week-02/day-6.md)

## Self-check

- [ ] I can scaffold a Python package with `uv init`, separate modules, and `pyproject.toml`.
- [ ] My `ChunkerProtocol` is satisfied by all five chunker classes.
- [ ] `ChunkEmbedPipeline` uses `MockEmbeddingProvider` in tests — no real API calls.
- [ ] `compare_strategies` produces metrics that clearly differ between fixed and sentence chunking.
- [ ] Integration tests pass with `pytest tests/`.

## Supplementary links

- [uv project management docs](https://docs.astral.sh/uv/guides/projects/)
- [Python Protocol typing](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [pytest fixtures docs](https://docs.pytest.org/en/stable/how-to/fixtures.html)
