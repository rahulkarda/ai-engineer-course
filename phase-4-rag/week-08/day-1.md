# Phase 4 · Week 08 · Day 1 — Architecture & Wiring: Designing the Document Intelligence Platform

> Objective: Design the full Document Intelligence Platform by defining component interfaces, a dependency graph, and a `ComponentRegistry` that composes all seven RAG subsystems into one coherent service.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-5]]

## Concept

A production RAG system is not a single script — it is seven cooperating subsystems. Each can be swapped, scaled, or replaced independently. The key architectural decision is defining **stable interfaces between components** rather than coupling implementations.

The seven components:

| # | Component | Responsibility |
|---|---|---|
| 1 | **Ingestion** | Load raw documents (PDF, DOCX, HTML, plain text) into structured form |
| 2 | **Chunking** | Split documents into retrieval-sized pieces with overlap |
| 3 | **Embeddings** | Map text chunks to dense vectors |
| 4 | **Vector Store** | Index and retrieve vectors by similarity |
| 5 | **Hybrid Retrieval** | Combine dense (vector) and sparse (BM25/keyword) retrieval |
| 6 | **Reranking** | Re-score a candidate set for precision |
| 7 | **Agentic RAG** | Multi-step reasoning, citation attachment, answer synthesis |

An **Eval Harness** runs orthogonally — it observes inputs and outputs at each component boundary.

---

### Dependency Graph

```
Raw file
    │
    ▼
[Ingestion] ──► Document (text + metadata)
    │
    ▼
[Chunking] ──► list[Chunk]
    │
    ▼
[Embeddings] ──► list[EmbeddedChunk]
    │
    ▼
[Vector Store] ◄──── index()  /  search() ────┐
                                               │
User query                                     │
    │                                          │
    ▼                                          │
[Hybrid Retrieval] ──── calls Vector Store + BM25 index
    │
    ▼
[Reranking] ──► sorted list[Chunk] (top-k)
    │
    ▼
[Agentic RAG] ──► GroundedAnswer (text + citations)
    │
    ▼
[Eval Harness] ──► metrics (faithfulness, relevance, citation coverage)
```

---

### Why Abstract Base Classes

Each component is expressed as an abstract base class (ABC) with a single required method. This enforces the contract at import time (via `abstractmethod`) and lets you swap implementations — a mock for tests, a fast local version for development, a production vendor for deployment — without changing calling code.

---

## How It Works Internally

Python's `abc.ABCMeta` prevents instantiating a class that hasn't implemented all `@abstractmethod` methods. The registry stores concrete instances; callers ask the registry for a component by name. This is the **Service Locator** pattern — simple, explicit, no dependency injection framework needed.

`pydantic-settings` reads configuration from environment variables at startup. Each component receives only the config slice it needs — a `ChunkingConfig` doesn't know about the vector store URL. This limits the blast radius of misconfiguration.

The health-check method iterates all registered components and calls a lightweight `ping()` method. If any component is unavailable (network error, missing model), the service starts degraded rather than crashing entirely — the response signals which component failed.

---

## Worked Example

```python
"""
day_1_component_registry.py

Defines abstract base classes for all seven RAG components and
a ComponentRegistry that composes them into a single injectable object.

Run:
    python day_1_component_registry.py
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data models
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """Raw document loaded from a file."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    """A retrieval-sized piece of a document."""

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = {}


class EmbeddedChunk(BaseModel):
    """Chunk paired with its dense vector representation."""

    chunk: Chunk
    vector: list[float]


class GroundedAnswer(BaseModel):
    """LLM answer with attached source citations."""

    answer: str
    citations: list[dict[str, Any]] = []
    model_id: str = ""
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Configuration (pydantic-settings reads from environment)
# ---------------------------------------------------------------------------


class IngestionConfig(BaseSettings):
    max_file_size_mb: int = 20
    supported_extensions: list[str] = [".pdf", ".txt", ".docx", ".html"]

    model_config = {"env_prefix": "INGEST_"}


class ChunkingConfig(BaseSettings):
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = {"env_prefix": "CHUNK_"}


class EmbeddingConfig(BaseSettings):
    model_name: str = "text-embedding-3-small"
    batch_size: int = 64
    api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))

    model_config = {"env_prefix": "EMBED_"}


class VectorStoreConfig(BaseSettings):
    url: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_URL", ""))
    collection_name: str = "documents"
    top_k: int = 20

    model_config = {"env_prefix": "VS_"}


class LLMConfig(BaseSettings):
    model_id: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    )
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    max_tokens: int = 1024

    model_config = {"env_prefix": "LLM_"}


# ---------------------------------------------------------------------------
# Abstract base classes — one per RAG component
# ---------------------------------------------------------------------------


class BaseIngestion(ABC):
    """Load a raw file into a Document."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """Parse file at *path* and return a Document."""

    @abstractmethod
    def ping(self) -> bool:
        """Return True if the component is operational."""


class BaseChunker(ABC):
    """Split a Document into retrieval-sized Chunks."""

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]:
        """Return chunks derived from *document*."""

    @abstractmethod
    def ping(self) -> bool: ...


class BaseEmbedder(ABC):
    """Embed a list of Chunks into dense vectors."""

    @abstractmethod
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Return EmbeddedChunks for each input Chunk."""

    @abstractmethod
    def ping(self) -> bool: ...


class BaseVectorStore(ABC):
    """Index and search EmbeddedChunks."""

    @abstractmethod
    def index(self, embedded: list[EmbeddedChunk]) -> None:
        """Persist *embedded* chunks to the store."""

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 20) -> list[Chunk]:
        """Return up to *top_k* Chunks nearest to *query_vector*."""

    @abstractmethod
    def ping(self) -> bool: ...


class BaseRetriever(ABC):
    """Hybrid dense + sparse retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 20) -> list[Chunk]:
        """Return candidate chunks for *query* using hybrid strategy."""

    @abstractmethod
    def ping(self) -> bool: ...


class BaseReranker(ABC):
    """Re-score a candidate set to improve precision."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[Chunk], top_n: int = 5) -> list[Chunk]:
        """Return the *top_n* most relevant chunks from *candidates*."""

    @abstractmethod
    def ping(self) -> bool: ...


class BaseAgenticRAG(ABC):
    """Multi-step reasoning with citation attachment."""

    @abstractmethod
    def answer(self, query: str, chunks: list[Chunk]) -> GroundedAnswer:
        """Generate a grounded answer with citations from *chunks*."""

    @abstractmethod
    def ping(self) -> bool: ...


# ---------------------------------------------------------------------------
# Health status model
# ---------------------------------------------------------------------------


@dataclass
class ComponentHealth:
    name: str
    healthy: bool
    error: str = ""


@dataclass
class PlatformHealth:
    overall: bool
    components: list[ComponentHealth] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Platform healthy: {self.overall}"]
        for c in self.components:
            status = "OK" if c.healthy else f"FAIL ({c.error})"
            lines.append(f"  {c.name}: {status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComponentRegistry — composes all seven subsystems
# ---------------------------------------------------------------------------


class ComponentRegistry:
    """
    Central composition point for the Document Intelligence Platform.

    Holds one concrete instance of each RAG subsystem.
    Call health_check() at startup to validate all components before
    accepting traffic.
    """

    def __init__(
        self,
        ingestion: BaseIngestion,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        agentic_rag: BaseAgenticRAG,
    ) -> None:
        self.ingestion = ingestion
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.agentic_rag = agentic_rag
        logger.info("ComponentRegistry initialised with %d components.", 7)

    def health_check(self) -> PlatformHealth:
        """
        Ping each component and return an aggregated health report.

        Does NOT raise on failure — reports degraded state so the API
        can respond with 503 rather than crashing on startup.
        """
        pairs: list[tuple[str, Any]] = [
            ("ingestion", self.ingestion),
            ("chunker", self.chunker),
            ("embedder", self.embedder),
            ("vector_store", self.vector_store),
            ("retriever", self.retriever),
            ("reranker", self.reranker),
            ("agentic_rag", self.agentic_rag),
        ]
        results: list[ComponentHealth] = []
        for name, component in pairs:
            try:
                ok = component.ping()
                results.append(ComponentHealth(name=name, healthy=ok))
                if not ok:
                    logger.warning("Component %r reported unhealthy.", name)
            except Exception as exc:
                logger.error("Component %r ping raised: %s", name, exc)
                results.append(ComponentHealth(name=name, healthy=False, error=str(exc)))

        overall = all(c.healthy for c in results)
        return PlatformHealth(overall=overall, components=results)

    def run_ingestion_pipeline(self, file_path: str) -> list[EmbeddedChunk]:
        """
        Execute the offline pipeline: ingest -> chunk -> embed -> index.

        Returns the list of EmbeddedChunks that were indexed.
        """
        logger.info("Starting ingestion pipeline for %r.", file_path)

        doc = self.ingestion.load(file_path)
        logger.info("Loaded document %r (%d chars).", doc.doc_id, len(doc.text))

        chunks = self.chunker.split(doc)
        logger.info("Split into %d chunks.", len(chunks))

        embedded = self.embedder.embed(chunks)
        logger.info("Embedded %d chunks.", len(embedded))

        self.vector_store.index(embedded)
        logger.info("Indexed %d chunks into vector store.", len(embedded))

        return embedded

    def run_query_pipeline(self, query: str, top_n: int = 5) -> GroundedAnswer:
        """
        Execute the online pipeline: retrieve -> rerank -> answer.

        Returns a GroundedAnswer with citations.
        """
        logger.info("Query pipeline start: %r", query)

        candidates = self.retriever.retrieve(query)
        logger.info("Hybrid retrieval returned %d candidates.", len(candidates))

        ranked = self.reranker.rerank(query, candidates, top_n=top_n)
        logger.info("Reranked to top %d chunks.", len(ranked))

        answer = self.agentic_rag.answer(query, ranked)
        logger.info(
            "Answer generated (%d chars, %d citations).",
            len(answer.answer),
            len(answer.citations),
        )

        return answer


# ---------------------------------------------------------------------------
# Stub implementations for local demo (all ping() return True)
# ---------------------------------------------------------------------------


class StubIngestion(BaseIngestion):
    def load(self, path: str) -> Document:
        return Document(doc_id=path, text=f"stub text from {path}")

    def ping(self) -> bool:
        return True


class StubChunker(BaseChunker):
    def split(self, document: Document) -> list[Chunk]:
        return [
            Chunk(chunk_id=f"{document.doc_id}::0", doc_id=document.doc_id, text=document.text)
        ]

    def ping(self) -> bool:
        return True


class StubEmbedder(BaseEmbedder):
    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk=c, vector=[0.1] * 8) for c in chunks]

    def ping(self) -> bool:
        return True


class StubVectorStore(BaseVectorStore):
    _store: list[EmbeddedChunk] = []

    def index(self, embedded: list[EmbeddedChunk]) -> None:
        self._store.extend(embedded)

    def search(self, query_vector: list[float], top_k: int = 20) -> list[Chunk]:
        return [e.chunk for e in self._store[:top_k]]

    def ping(self) -> bool:
        return True


class StubRetriever(BaseRetriever):
    def __init__(self, vector_store: StubVectorStore) -> None:
        self._vs = vector_store

    def retrieve(self, query: str, top_k: int = 20) -> list[Chunk]:
        return self._vs.search([0.1] * 8, top_k=top_k)

    def ping(self) -> bool:
        return True


class StubReranker(BaseReranker):
    def rerank(self, query: str, candidates: list[Chunk], top_n: int = 5) -> list[Chunk]:
        return candidates[:top_n]

    def ping(self) -> bool:
        return True


class StubAgenticRAG(BaseAgenticRAG):
    def answer(self, query: str, chunks: list[Chunk]) -> GroundedAnswer:
        return GroundedAnswer(
            answer=f"[MOCK] Answer to: {query}",
            citations=[{"chunk_id": c.chunk_id, "text": c.text[:60]} for c in chunks],
            model_id=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        )

    def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    vs = StubVectorStore()

    registry = ComponentRegistry(
        ingestion=StubIngestion(),
        chunker=StubChunker(),
        embedder=StubEmbedder(),
        vector_store=vs,
        retriever=StubRetriever(vs),
        reranker=StubReranker(),
        agentic_rag=StubAgenticRAG(),
    )

    health = registry.health_check()
    print(health.summary())

    registry.run_ingestion_pipeline("sample-doc.txt")
    answer = registry.run_query_pipeline("What is in the document?")
    print(f"\nAnswer: {answer.answer}")
    print(f"Citations: {len(answer.citations)}")
```

**Run it:**
```bash
python day_1_component_registry.py
```

Expected: all components report OK, ingestion pipeline runs, query returns a mock answer with citations.

---

## Exercises

1. **Draw the dependency graph.** On paper or in a text diagram, draw the full dependency graph including data types flowing between each arrow (e.g., `Document`, `list[Chunk]`, `list[EmbeddedChunk]`). Label each edge with the method name called (e.g., `split()`, `embed()`).

2. **Write the ComponentRegistry stub.** Starting from the ABCs above, write a `ComponentRegistry.__init__` that accepts all seven components and stores them. Do not look at the worked example first — write it from the interface contracts alone.

3. **Add a health-check method.** Extend the `ComponentRegistry` with a `health_check()` method that calls `ping()` on each component, catches exceptions, and returns a `PlatformHealth` dataclass. Write a pytest test that injects one broken component (whose `ping()` raises) and asserts `PlatformHealth.overall` is `False`.

4. **Configuration isolation.** Create a `RerankerConfig(BaseSettings)` with `model_name: str`, `top_n: int = 5`, and `env_prefix = "RERANK_"`. Demonstrate it reads `RERANK_TOP_N=10` from the environment.

(Solutions: ../solutions/week-08/day-1.md)

---

## Self-check

- [ ] I can name all seven RAG components and the data type each produces.
- [ ] I can explain why abstract base classes enforce interface contracts at import time.
- [ ] I can implement a `ComponentRegistry` that runs the offline and online pipelines end-to-end.
- [ ] I can write a `health_check()` that reports per-component status without raising.
- [ ] I can configure each component via `pydantic-settings` without hardcoding any credentials.

---

## Supplementary Links

- [Python `abc` module docs](https://docs.python.org/3/library/abc.html)
- [pydantic-settings env variable docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Twelve-Factor App: Config](https://12factor.net/config)
- [RAG survey — architectural patterns](https://arxiv.org/abs/2312.10997)
