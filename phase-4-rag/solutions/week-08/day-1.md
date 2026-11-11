# Solutions — Week 08 · Day 1

## Exercise 1: Dependency Graph

The pipeline is a directed acyclic graph where each stage transforms one data representation into
another. Annotating edges with data types makes the contract between components explicit — any
component swap must honour the same types on both sides.

```
                     ┌─────────────────────────────────────────────────────────────────┐
                     │              DOCUMENT INTELLIGENCE PLATFORM                     │
                     └─────────────────────────────────────────────────────────────────┘

  Raw File / URL
       │
       ▼
 ┌─────────────┐    List[Document]     ┌──────────────────┐
 │  ingestion  │ ─────────────────────▶│    chunking      │
 └─────────────┘                       └──────────────────┘
                                               │
                                    List[Chunk]│  (text + metadata)
                                               ▼
                                       ┌──────────────────┐
                                       │   embeddings     │
                                       └──────────────────┘
                                               │
                              List[EmbeddedChunk]│  (Chunk + vector[float])
                                               ▼
                                       ┌──────────────────┐
                                       │   vector_store   │◀── UPSERT path
                                       └──────────────────┘
                                               │
  Query string ─────────────────────────────▶ │
                                               ▼
                                       ┌──────────────────┐
                                       │ hybrid_retrieval │  (dense + sparse fusion)
                                       └──────────────────┘
                                               │
                              List[ScoredChunk]│  (Chunk + score: float)
                                               ▼
                                       ┌──────────────────┐
                                       │   reranking      │
                                       └──────────────────┘
                                               │
                           List[RankedChunk]   │  (ScoredChunk + rank: int)
                                               ▼
                                       ┌──────────────────┐
                                       │  agentic_rag     │  (LLM synthesis)
                                       └──────────────────┘
                                               │
                              CitedResponse    │  (answer: str + citations: List[Citation])
                                               ▼
                                       ┌──────────────────┐
                                       │  eval_harness    │
                                       └──────────────────┘
                                               │
                               EvalReport      │  (faithfulness, relevancy, scores)
                                               ▼
                                           Results

Edge data types summary:
  ingestion       → chunking          : List[Document]
  chunking        → embeddings        : List[Chunk]
  embeddings      → vector_store      : List[EmbeddedChunk]
  vector_store    → hybrid_retrieval  : List[ScoredChunk]  (query-time retrieval)
  hybrid_retrieval→ reranking         : List[ScoredChunk]
  reranking       → agentic_rag       : List[RankedChunk]
  agentic_rag     → eval_harness      : CitedResponse
  eval_harness    →  caller           : EvalReport
```

---

## Exercise 2: ComponentRegistry Stub

The registry uses Python ABCs to enforce the interface contract at class-definition time rather
than at runtime. A dataclass holds one typed instance of each component, making injection and
testing trivial — swap any field for a mock.

```python
# src/document_platform/registry.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data models (minimal — expand in later days)
# ---------------------------------------------------------------------------

class Document:
    """Raw document with source path and text content."""
    def __init__(self, doc_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        self.doc_id = doc_id
        self.content = content
        self.metadata: dict[str, Any] = metadata or {}


class Chunk:
    """Text chunk derived from a Document."""
    def __init__(self, chunk_id: str, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.metadata: dict[str, Any] = metadata or {}


class EmbeddedChunk:
    """Chunk with its dense vector representation."""
    def __init__(self, chunk: Chunk, vector: list[float]) -> None:
        self.chunk = chunk
        self.vector = vector


class ScoredChunk:
    """Chunk with a retrieval relevance score."""
    def __init__(self, chunk: Chunk, score: float) -> None:
        self.chunk = chunk
        self.score = score


class RankedChunk:
    """Chunk with both score and final rank after reranking."""
    def __init__(self, chunk: Chunk, score: float, rank: int) -> None:
        self.chunk = chunk
        self.score = score
        self.rank = rank


class CitedResponse:
    """LLM answer with supporting citations."""
    def __init__(self, answer: str, citations: list[dict[str, Any]]) -> None:
        self.answer = answer
        self.citations = citations


class EvalReport:
    """Evaluation metrics for a CitedResponse."""
    def __init__(self, faithfulness: float, relevancy: float, scores: dict[str, float]) -> None:
        self.faithfulness = faithfulness
        self.relevancy = relevancy
        self.scores = scores


# ---------------------------------------------------------------------------
# Abstract base classes — the component interfaces
# ---------------------------------------------------------------------------

class DocumentIngester(ABC):
    """Converts raw files / URLs into Documents."""

    @abstractmethod
    def ingest(self, source: str) -> list[Document]:
        """Ingest a source (file path or URL) and return Documents."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the ingester is operational."""


class ChunkEmbedder(ABC):
    """Chunks Documents and produces EmbeddedChunks."""

    @abstractmethod
    def chunk_and_embed(self, documents: list[Document]) -> list[EmbeddedChunk]:
        """Chunk documents and compute dense embeddings."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the embedding service is reachable."""


class VectorStore(ABC):
    """Persistent store for EmbeddedChunks with nearest-neighbour search."""

    @abstractmethod
    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks."""

    @abstractmethod
    def query(self, vector: list[float], top_k: int = 10) -> list[ScoredChunk]:
        """Return top_k chunks nearest to the query vector."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the vector store is reachable."""


class HybridRetriever(ABC):
    """Fuses dense (vector) and sparse (BM25/keyword) retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Run hybrid retrieval and return fused results."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if both dense and sparse indexes are available."""


class Reranker(ABC):
    """Cross-encoder reranker that re-scores ScoredChunks for a query."""

    @abstractmethod
    def rerank(self, query: str, chunks: list[ScoredChunk]) -> list[RankedChunk]:
        """Rerank chunks with a cross-encoder; return in descending relevance order."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the reranker model is loaded."""


class AgenticRAG(ABC):
    """LLM agent that synthesises a cited answer from ranked chunks."""

    @abstractmethod
    def answer(self, query: str, ranked_chunks: list[RankedChunk]) -> CitedResponse:
        """Generate an answer grounded in ranked_chunks, with citations."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the LLM endpoint is reachable."""


class EvalHarness(ABC):
    """Evaluation harness that scores CitedResponses."""

    @abstractmethod
    def evaluate(self, response: CitedResponse, ground_truth: str) -> EvalReport:
        """Compute faithfulness, relevancy, and other metrics."""

    @abstractmethod
    def health(self) -> bool:
        """Return True if the eval harness dependencies are available."""


# ---------------------------------------------------------------------------
# ComponentRegistry
# ---------------------------------------------------------------------------

@dataclass
class ComponentRegistry:
    """
    Central registry holding one live instance of every pipeline component.

    Design rationale: a dataclass gives us free __repr__ and easy field
    access, while ABCs ensure every swap respects the same interface.
    Inject mocks in tests by replacing individual fields.
    """

    ingester: DocumentIngester
    embedder: ChunkEmbedder
    vector_store: VectorStore
    retriever: HybridRetriever
    reranker: Reranker
    rag: AgenticRAG
    evaluator: EvalHarness

    # ---------------------------------------------------------------------------
    # Exercise 3 placeholder — see Exercise 3 below for full implementation
    # ---------------------------------------------------------------------------
    def health_check(self) -> dict[str, bool]:  # implemented in Exercise 3
        raise NotImplementedError
```

---

## Exercise 3: health_check() Method

`health_check` logs each failure immediately (structured, so it survives log aggregators) and
returns a dict so callers can branch on individual component failures rather than all-or-nothing.
The try/except isolates a crashing component so the others still get checked.

```python
# Extend the ComponentRegistry dataclass above — replace the placeholder method.

    def health_check(self) -> dict[str, bool]:
        """
        Ping every component and return a mapping of component_name -> is_healthy.

        A component is considered healthy if its health() method returns True
        without raising an exception.  Any exception is caught, logged at ERROR
        level, and treated as unhealthy — so one broken component does not prevent
        the others from being checked.

        Returns:
            dict[str, bool]: e.g. {"ingester": True, "vector_store": False, ...}
        """
        checks: dict[str, Any] = {
            "ingester": self.ingester,
            "embedder": self.embedder,
            "vector_store": self.vector_store,
            "retriever": self.retriever,
            "reranker": self.reranker,
            "rag": self.rag,
            "evaluator": self.evaluator,
        }
        results: dict[str, bool] = {}

        for name, component in checks.items():
            try:
                ok = component.health()
                results[name] = bool(ok)
                if not ok:
                    logger.warning(
                        "Component unhealthy",
                        extra={"component": name},
                    )
            except Exception as exc:
                logger.error(
                    "Component health check raised an exception",
                    extra={"component": name, "error": str(exc)},
                    exc_info=True,
                )
                results[name] = False

        healthy_count = sum(results.values())
        logger.info(
            "Health check complete",
            extra={"healthy": healthy_count, "total": len(results)},
        )
        return results
```

### Usage example

```python
# tests/unit/test_registry.py
import pytest
from unittest.mock import MagicMock
from src.document_platform.registry import ComponentRegistry


def _make_mock_component(healthy: bool = True) -> MagicMock:
    m = MagicMock()
    m.health.return_value = healthy
    return m


def test_health_check_all_healthy():
    registry = ComponentRegistry(
        ingester=_make_mock_component(),
        embedder=_make_mock_component(),
        vector_store=_make_mock_component(),
        retriever=_make_mock_component(),
        reranker=_make_mock_component(),
        rag=_make_mock_component(),
        evaluator=_make_mock_component(),
    )
    result = registry.health_check()
    assert all(result.values()), "All components should be healthy"
    assert set(result.keys()) == {
        "ingester", "embedder", "vector_store",
        "retriever", "reranker", "rag", "evaluator",
    }


def test_health_check_one_unhealthy():
    registry = ComponentRegistry(
        ingester=_make_mock_component(healthy=False),
        embedder=_make_mock_component(),
        vector_store=_make_mock_component(),
        retriever=_make_mock_component(),
        reranker=_make_mock_component(),
        rag=_make_mock_component(),
        evaluator=_make_mock_component(),
    )
    result = registry.health_check()
    assert result["ingester"] is False
    assert result["embedder"] is True


def test_health_check_exception_treated_as_unhealthy():
    bad = MagicMock()
    bad.health.side_effect = RuntimeError("connection refused")
    registry = ComponentRegistry(
        ingester=bad,
        embedder=_make_mock_component(),
        vector_store=_make_mock_component(),
        retriever=_make_mock_component(),
        reranker=_make_mock_component(),
        rag=_make_mock_component(),
        evaluator=_make_mock_component(),
    )
    result = registry.health_check()
    assert result["ingester"] is False   # exception -> unhealthy, no re-raise
    assert result["embedder"] is True    # others still checked
```

---

## Key concepts demonstrated

- **ABC contracts** — abstract base classes fail at definition time if a subclass skips an
  abstract method, catching errors before any code runs.
- **Dataclass registry** — free `__repr__`, easy field swap for testing, no boilerplate
  constructor.
- **Defensive health checks** — per-component try/except means a crashing component is isolated;
  the rest still report.
- **Structured logging** — `extra={}` dict fields survive JSON log formatters (e.g. structlog,
  python-json-logger) without polluting the message string.
- **Edge type annotations** — making the data type on each pipeline edge explicit is a form of
  lightweight interface documentation that IDEs and mypy can verify.
