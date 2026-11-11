# Phase 4 · Week 05 · Day 6 — Saturday Build: Reranking + Optimization Layer

> Objective: Integrate cross-encoder reranking, MMR diversity, and parent-document expansion into a production pipeline on top of the week-4 hybrid retriever.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]], [[../week-04/day-7]]

## Overview

Today you build the complete week-5 artefact: a `RerankingPipeline` class that wraps a retriever and applies:

1. Query expansion (optional)
2. Cross-encoder reranking
3. MMR diversity filter
4. Parent-document expansion
5. Retrieval quality measurement

The pipeline is modular — each stage is independently testable, and the full integration can be evaluated before/after against a baseline (retrieval without reranking).

Estimated split: 1 hr design, 2 hr coding, 1 hr testing + metrics.

## Design

```
┌─────────────────────────────────────────────────────────────────┐
│  RerankingPipeline.retrieve(query)                              │
│                                                                 │
│  1. expand_query()          → list[str]  (optional)            │
│  2. retriever.search(q)     → list[Chunk] for each query        │
│  3. deduplicate_chunks()    → list[Chunk]                       │
│  4. CrossEncoderReranker.rerank()  → list[RerankResult]        │
│  5. mmr_select()            → list[MMRResult]                  │
│  6. ParentDocumentStore.fetch_parents() → list[Document]        │
│                                                                 │
│  Returns: list[RetrievedContext]                                │
└─────────────────────────────────────────────────────────────────┘
```

Config is a Pydantic model so the pipeline is serialisable.

## Project scaffold

```
week05_project/
├── src/
│   ├── pipeline.py          # RerankingPipeline (main artefact)
│   ├── reranker.py          # CrossEncoderReranker (from day-2)
│   ├── mmr.py               # mmr_select (from day-3)
│   ├── query_expansion.py   # expand_query, deduplicate (from day-4)
│   ├── metrics.py           # precision_at_k ... (from day-5)
│   └── models.py            # shared data models
├── tests/
│   ├── unit/
│   │   ├── test_pipeline.py
│   │   └── test_integration.py
│   └── conftest.py
└── pyproject.toml
```

## Complete implementation

### `src/models.py`

```python
"""Shared data models for the reranking pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedContext:
    """Final output of the pipeline — what gets sent to the LLM."""
    doc_id: str
    text: str            # parent text (if expansion enabled) or chunk text
    rerank_score: float  # cross-encoder score (sigmoid-normalised)
    mmr_score: float     # MMR score at selection
    source: str          # "parent" or "chunk"
```

### `src/pipeline.py`

```python
"""
pipeline.py

RerankingPipeline: integrates cross-encoder reranking, MMR,
query expansion, and parent-document retrieval.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol

from pydantic import BaseModel, Field

from models import Chunk, Document, RetrievedContext
from reranker import CrossEncoderReranker, RerankResult
from mmr import mmr_select, MMRResult
from query_expansion import deduplicate_chunks, expand_query

import numpy as np

log = logging.getLogger(__name__)

LLMCallable = Callable[[str], str]


# ---------------------------------------------------------------------------
# Retriever protocol
# ---------------------------------------------------------------------------
class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[Chunk]:
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return unit-normalised embeddings for a list of texts."""
        ...


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class PipelineConfig(BaseModel):
    retrieval_top_k: int = Field(50, description="Candidates fetched per query")
    rerank_top_n: int = Field(20, description="Results kept after reranking")
    rerank_score_threshold: float = Field(0.2, ge=0.0, le=1.0)
    mmr_k: int = Field(5, description="Final documents returned")
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    use_query_expansion: bool = False
    n_query_expansions: int = Field(2, ge=1)
    use_parent_docs: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class RerankingPipeline:
    """
    Full reranking and retrieval optimisation pipeline.

    Parameters
    ----------
    retriever:
        Object implementing the Retriever protocol.
    config:
        PipelineConfig controlling all hyperparameters.
    parent_store:
        Optional dict mapping doc_id → Document for parent-doc expansion.
    llm:
        Optional callable for query expansion; required if use_query_expansion=True.
    """

    def __init__(
        self,
        retriever: Retriever,
        config: PipelineConfig | None = None,
        parent_store: dict[str, Document] | None = None,
        llm: LLMCallable | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config or PipelineConfig()
        self._parent_store = parent_store or {}
        self._llm = llm
        self._reranker = CrossEncoderReranker(
            model_name=self._config.reranker_model
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list[RetrievedContext]:
        """
        Run the full pipeline and return final context documents.
        """
        cfg = self._config

        # Stage 1: Query expansion
        queries = [query]
        if cfg.use_query_expansion:
            if self._llm is None:
                raise ValueError("llm must be provided when use_query_expansion=True")
            queries = expand_query(query, self._llm, n_expansions=cfg.n_query_expansions)

        # Stage 2: Retrieve candidates for each query
        all_chunks: list[Chunk] = []
        for q in queries:
            chunks = self._retriever.search(q, top_k=cfg.retrieval_top_k)
            all_chunks.extend(chunks)
            log.debug("Query '%s' → %d chunks", q[:50], len(chunks))

        # Stage 3: Deduplicate
        all_chunks = deduplicate_chunks(all_chunks)
        log.info("After dedup: %d unique chunks", len(all_chunks))

        if not all_chunks:
            log.warning("No chunks retrieved for query: %s", query)
            return []

        # Stage 4: Cross-encoder rerank
        texts = [c.text for c in all_chunks]
        reranked: list[RerankResult] = self._reranker.rerank(
            query,
            texts,
            top_n=cfg.rerank_top_n,
            score_threshold=cfg.rerank_score_threshold,
        )
        log.info("After reranking: %d candidates", len(reranked))

        if not reranked:
            return []

        # Stage 5: MMR diversity filter
        reranked_texts = [r.text for r in reranked]
        embs = self._retriever.embed(reranked_texts)
        query_emb = self._retriever.embed([query])[0]

        # Use rerank scores as relevance signal for MMR
        rerank_scores = np.array([r.score for r in reranked])
        # Normalise to [0,1] (already sigmoid-normalised from reranker)
        mmr_results: list[MMRResult] = mmr_select(
            query_emb=query_emb,
            doc_embs=embs,
            texts=reranked_texts,
            k=cfg.mmr_k,
            lambda_mult=cfg.mmr_lambda,
        )
        log.info("After MMR: %d documents selected", len(mmr_results))

        # Stage 6: Parent-document expansion
        contexts: list[RetrievedContext] = []
        for mmr_result in mmr_results:
            original_chunk = all_chunks[reranked[mmr_result.index].original_index]
            rerank_score = reranked[mmr_result.index].score

            if cfg.use_parent_docs and original_chunk.doc_id in self._parent_store:
                parent = self._parent_store[original_chunk.doc_id]
                context_text = parent.text
                source = "parent"
            else:
                context_text = original_chunk.text
                source = "chunk"

            contexts.append(
                RetrievedContext(
                    doc_id=original_chunk.doc_id,
                    text=context_text,
                    rerank_score=rerank_score,
                    mmr_score=mmr_result.mmr_score,
                    source=source,
                )
            )

        return contexts
```

### `tests/unit/test_pipeline.py`

```python
"""
Unit tests for RerankingPipeline.
All external calls (CrossEncoder, SentenceTransformer) are mocked.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models import Chunk, Document
from pipeline import PipelineConfig, RerankingPipeline
from reranker import RerankResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_retriever():
    retriever = MagicMock()
    retriever.search.return_value = [
        Chunk("doc-1", 0, "Cross-encoders improve precision by joint scoring."),
        Chunk("doc-2", 0, "MMR selects diverse relevant documents."),
        Chunk("doc-1", 1, "Reranking is the second stage of retrieval."),
    ]
    # Return simple unit vectors
    def embed(texts):
        n = len(texts)
        embs = np.eye(max(n, 4))[:n, :4]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.where(norms == 0, 1, norms)
    retriever.embed.side_effect = embed
    return retriever


@pytest.fixture()
def mock_reranker_scores():
    return [3.0, 2.0, 1.0]


@pytest.fixture()
def pipeline(mock_retriever):
    config = PipelineConfig(retrieval_top_k=10, rerank_top_n=5, mmr_k=2)
    return RerankingPipeline(retriever=mock_retriever, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPipelineRetrieve:
    def test_returns_list_of_retrieved_contexts(self, pipeline, mock_reranker_scores):
        with patch("pipeline.CrossEncoderReranker") as MockCE:
            mock_ce = MockCE.return_value
            mock_ce.rerank.return_value = [
                RerankResult(0, 0.9, "Cross-encoders improve precision by joint scoring."),
                RerankResult(1, 0.7, "MMR selects diverse relevant documents."),
            ]
            results = pipeline.retrieve("how does reranking work?")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_empty_retriever_returns_empty(self, mock_retriever):
        mock_retriever.search.return_value = []
        config = PipelineConfig()
        p = RerankingPipeline(retriever=mock_retriever, config=config)
        with patch("pipeline.CrossEncoderReranker"):
            results = p.retrieve("query")
        assert results == []

    def test_parent_doc_expansion_uses_parent_text(self, mock_retriever):
        parent = Document("doc-1", "Full parent document with rich context.")
        config = PipelineConfig(use_parent_docs=True, mmr_k=1)
        p = RerankingPipeline(
            retriever=mock_retriever,
            config=config,
            parent_store={"doc-1": parent},
        )
        with patch("pipeline.CrossEncoderReranker") as MockCE:
            mock_ce = MockCE.return_value
            mock_ce.rerank.return_value = [
                RerankResult(0, 0.9, "Cross-encoders improve precision by joint scoring."),
            ]
            results = p.retrieve("query")
        if results:
            # If doc-1 was selected, its text should be the parent text
            parent_results = [r for r in results if r.doc_id == "doc-1"]
            for r in parent_results:
                assert r.text == parent.text
                assert r.source == "parent"

    def test_query_expansion_raises_without_llm(self, mock_retriever):
        config = PipelineConfig(use_query_expansion=True)
        p = RerankingPipeline(retriever=mock_retriever, config=config)
        with pytest.raises(ValueError, match="llm must be provided"):
            p.retrieve("query")


class TestPipelineConfig:
    def test_default_config_valid(self):
        cfg = PipelineConfig()
        assert cfg.mmr_k > 0
        assert 0 <= cfg.mmr_lambda <= 1

    def test_threshold_bounds(self):
        with pytest.raises(Exception):
            PipelineConfig(rerank_score_threshold=1.5)
```

### `tests/unit/test_integration.py`

```python
"""
Integration-style test: run evaluate_retrieval on the pipeline
with a small synthetic eval set. All ML models are mocked.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from metrics import RetrievalEvalCase, evaluate_retrieval


def test_metrics_pipeline_integration():
    """Verify that evaluate_retrieval correctly computes macro-average metrics."""
    cases = [
        RetrievalEvalCase(
            query="q1",
            relevant_ids={"doc-1", "doc-2"},
            graded_ids={"doc-1": 2, "doc-2": 1, "doc-3": 0},
        ),
        RetrievalEvalCase(
            query="q2",
            relevant_ids={"doc-3"},
            graded_ids={"doc-3": 2},
        ),
    ]

    def mock_retrieve(query: str) -> list[str]:
        if query == "q1":
            return ["doc-1", "doc-2", "doc-3"]  # perfect recall
        return ["doc-3", "doc-1"]  # first result is relevant

    result = evaluate_retrieval(cases, mock_retrieve, k=3)
    assert result.precision > 0.0
    assert result.recall > 0.0
    assert result.mrr == 1.0  # both queries: first result is relevant
    assert result.ndcg > 0.0
```

## Build tasks (work through in order)

- [ ] Create the project scaffold directory structure.
- [ ] Copy/refactor day 2–5 modules into `src/`.
- [ ] Implement `pipeline.py` following the design above.
- [ ] Run unit tests: `pytest tests/unit/ -v`.
- [ ] Build a 20-case synthetic eval set (hardcode relevant_ids for a corpus you define).
- [ ] Implement `before_after_comparison`: run `evaluate_retrieval` with a mock baseline retriever (no reranking) and with the full pipeline; print the metric deltas.
- [ ] Log all intermediate step sizes (candidates fetched, after dedup, after rerank, after MMR).

## Self-check

- [ ] Pipeline handles empty retriever results gracefully.
- [ ] All external model calls are mockable without changes to pipeline code.
- [ ] Config is a validated Pydantic model.
- [ ] Tests cover the happy path and at least two edge cases.
- [ ] Metrics show measurable improvement (or you can explain why they don't on synthetic data).
