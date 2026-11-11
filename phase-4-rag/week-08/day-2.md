# Phase 4 · Week 08 · Day 2 — FastAPI Endpoints: Upload, Ingest, Query

> Objective: Design and implement the three core REST endpoints of the Document Intelligence Platform — file upload, pipeline ingestion, and grounded query — with background tasks and Pydantic request/response models.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-1]]

## Concept

A RAG service exposes three natural operations to callers:

1. **Upload** — receive a raw file (multipart/form-data) and store it.
2. **Ingest** — trigger the offline pipeline (chunking, embedding, indexing) for a stored file.
3. **Query** — accept a question and return a grounded answer with citations.

Ingestion is expensive (seconds to minutes per document). It must run as a **background task** so the HTTP response returns immediately with a job ID. The caller polls for status separately. Query is online (sub-second) and returns synchronously.

---

### Request / Response Design Principles

- Every request and response is a Pydantic `BaseModel` — no raw dicts in route signatures.
- IDs are `str` (UUID4 as string) — simple, URL-safe, no database dependency.
- Error responses use FastAPI's `HTTPException` with a structured detail dict.
- Background tasks use FastAPI's built-in `BackgroundTasks` — no Celery needed for moderate scale.
- Status is tracked in an in-memory dict for this lesson; in production, replace with Redis or a DB.

---

## How It Works Internally

FastAPI's `BackgroundTasks` appends a callable to a list that is executed **after** the HTTP response is sent. The event loop continues processing other requests while the background task runs. This works well for I/O-bound work (network calls to embedding APIs, vector store writes) but not for CPU-heavy work — use a task queue (Celery, ARQ) for the latter.

`UploadFile` is a Starlette abstraction over the multipart body. It streams the file into memory (or a temp file) without blocking. Always `await file.read()` rather than accessing `.file` directly in async routes.

The in-memory status dict (`dict[str, IngestStatus]`) is a deliberate simplification. It is not thread-safe across multiple workers. The exercise asks you to reason about this boundary — acceptable for single-worker dev, must be replaced with shared state in production.

---

## Worked Example

```python
"""
day_2_fastapi_app.py

Document Intelligence Platform — core FastAPI application.
Three endpoints: upload, ingest, query.

Run:
    pip install fastapi uvicorn python-multipart pydantic-settings
    uvicorn day_2_fastapi_app:app --reload
"""

from __future__ import annotations

import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class AppConfig(BaseSettings):
    upload_dir: str = "/tmp/doc_intel_uploads"
    max_upload_mb: int = 20
    anthropic_model: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    )

    model_config = {"env_prefix": "APP_"}


config = AppConfig()
Path(config.upload_dir).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Status tracking (in-memory — replace with Redis in production)
# ---------------------------------------------------------------------------


class IngestState(str, Enum):
    pending = "pending"
    running = "running"
    complete = "complete"
    failed = "failed"


class IngestStatus(BaseModel):
    job_id: str
    doc_id: str
    state: IngestState = IngestState.pending
    chunks_indexed: int = 0
    error: str = ""


# Global in-memory store.  NOT safe for multi-worker deployments.
_status_store: dict[str, IngestStatus] = {}
_document_store: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    size_bytes: int
    message: str


class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="ID returned by /documents/upload")
    chunk_size: int = Field(512, ge=64, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=512)


class IngestResponse(BaseModel):
    job_id: str
    doc_id: str
    state: IngestState
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    model_id: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    )


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    text_excerpt: str
    score: float = 0.0


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    model_id: str
    tokens_used: int = 0


# ---------------------------------------------------------------------------
# Stub pipeline (replace with ComponentRegistry from day-1)
# ---------------------------------------------------------------------------


def _stub_run_ingestion(doc_id: str, file_path: str, job_id: str, chunk_size: int) -> None:
    """
    Stand-in for the real ingestion pipeline.
    Replace this body with registry.run_ingestion_pipeline(file_path).
    """
    import time

    logger.info("Ingestion job %r starting for doc %r.", job_id, doc_id)
    status = _status_store[job_id]
    status.state = IngestState.running

    try:
        # Simulate work
        time.sleep(0.5)
        status.chunks_indexed = 4  # real: count of embedded chunks
        status.state = IngestState.complete
        logger.info("Ingestion job %r complete — %d chunks indexed.", job_id, status.chunks_indexed)
    except Exception as exc:
        status.state = IngestState.failed
        status.error = str(exc)
        logger.error("Ingestion job %r failed: %s", job_id, exc)


def _stub_run_query(question: str, top_k: int, model_id: str) -> QueryResponse:
    """
    Stand-in for the real query pipeline.
    Replace with registry.run_query_pipeline(question, top_n=top_k).
    """
    # In production: retrieve -> rerank -> agentic_rag.answer()
    citations = [
        Citation(
            chunk_id=f"chunk-{i}",
            doc_id="doc-stub-001",
            text_excerpt=f"Stub context excerpt {i} relevant to: {question[:40]}",
            score=round(0.95 - i * 0.05, 2),
        )
        for i in range(min(top_k, 2))
    ]
    return QueryResponse(
        question=question,
        answer=f"[MOCK] Grounded answer for: {question}",
        citations=citations,
        model_id=model_id,
        tokens_used=0,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Document Intelligence Platform",
    version="0.1.0",
    description="Upload, ingest, and query documents with grounded RAG.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


# --- Endpoint 1: Upload ---


@app.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile) -> UploadResponse:
    """
    Accept a document file via multipart/form-data.

    Validates extension and size, saves to upload_dir, returns a doc_id
    that the caller uses to trigger ingestion.
    """
    allowed = {".pdf", ".txt", ".docx", ".html"}
    suffix = Path(file.filename or "unknown").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=422,
            detail={"error": "unsupported_extension", "allowed": list(allowed)},
        )

    content = await file.read()
    size_bytes = len(content)
    max_bytes = config.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "max_mb": config.max_upload_mb},
        )

    doc_id = str(uuid.uuid4())
    dest = Path(config.upload_dir) / f"{doc_id}{suffix}"
    dest.write_bytes(content)

    _document_store[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "path": str(dest),
        "size_bytes": size_bytes,
    }

    logger.info("Uploaded doc %r (%d bytes) -> %s", doc_id, size_bytes, dest)
    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename or "",
        size_bytes=size_bytes,
        message="Upload successful. Call POST /documents/ingest to index.",
    )


# --- Endpoint 2: Ingest ---


@app.post("/documents/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """
    Trigger the ingestion pipeline for an uploaded document.

    Returns immediately with a job_id.  Poll GET /documents/{doc_id}/status
    to track progress.
    """
    doc = _document_store.get(request.doc_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "doc_not_found", "doc_id": request.doc_id},
        )

    job_id = str(uuid.uuid4())
    status = IngestStatus(job_id=job_id, doc_id=request.doc_id)
    _status_store[job_id] = status

    background_tasks.add_task(
        _stub_run_ingestion,
        doc_id=request.doc_id,
        file_path=doc["path"],
        job_id=job_id,
        chunk_size=request.chunk_size,
    )

    logger.info("Enqueued ingestion job %r for doc %r.", job_id, request.doc_id)
    return IngestResponse(
        job_id=job_id,
        doc_id=request.doc_id,
        state=IngestState.pending,
        message="Ingestion started. Poll /documents/{doc_id}/status for progress.",
    )


@app.get("/documents/{job_id}/status", response_model=IngestStatus)
async def get_ingest_status(job_id: str) -> IngestStatus:
    """Return the current state of an ingestion job."""
    status = _status_store.get(job_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "job_not_found", "job_id": job_id},
        )
    return status


# --- Endpoint 3: Query ---


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Accept a natural language question and return a grounded answer.

    Runs the online pipeline synchronously: retrieve -> rerank -> answer.
    Returns the answer with source citations.
    """
    logger.info("Query received: %r (top_k=%d)", request.question, request.top_k)
    result = _stub_run_query(
        question=request.question,
        top_k=request.top_k,
        model_id=request.model_id,
    )
    logger.info("Query answered (%d citations).", len(result.citations))
    return result
```

**Run it:**
```bash
uvicorn day_2_fastapi_app:app --reload
```

Test the upload endpoint:
```bash
echo "Hello from the platform." > /tmp/test.txt
curl -X POST http://localhost:8000/documents/upload \
     -F "file=@/tmp/test.txt"
```

Test the query endpoint:
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic?", "top_k": 3}'
```

---

## Exercises

1. **Add GET /documents list endpoint.** Implement `GET /documents` that returns all documents in `_document_store` as a `list[DocumentSummary]` (a Pydantic model with `doc_id`, `filename`, `size_bytes`). Add `limit: int = 10` and `offset: int = 0` query parameters for pagination. Validate that `limit <= 100`.

2. **Add GET /documents/{doc_id}/status endpoint.** The current `/documents/{job_id}/status` uses the job ID. Add a second route `GET /documents/{doc_id}/jobs` that returns all ingestion jobs for a given `doc_id` — useful when a document is re-ingested multiple times.

3. **Write pytest tests mocking the pipeline.** Using `pytest` and `httpx.AsyncClient` (via `httpx` and `pytest-asyncio`), write tests for:
   - Upload rejects a `.exe` file with HTTP 422.
   - Upload rejects a file exceeding `max_upload_mb` with HTTP 413.
   - Query returns a `QueryResponse` with at least one citation.
   - `GET /documents/{job_id}/status` returns 404 for an unknown job ID.
   Mock `_stub_run_query` to avoid any real network calls.

4. **Thread-safety reflection.** The `_status_store` and `_document_store` dicts are module-level globals. Under a multi-worker deployment (`uvicorn --workers 4`), what goes wrong? Write a one-paragraph explanation and propose a replacement (Redis, SQLite with WAL mode, or a DB) with reasoning.

(Solutions: ../solutions/week-08/day-2.md)

---

## Self-check

- [ ] I can explain why ingestion runs as a background task and query runs synchronously.
- [ ] I can write Pydantic request/response models for all three endpoints.
- [ ] I can implement `BackgroundTasks.add_task()` and return a job ID immediately.
- [ ] I can write pytest tests for FastAPI routes using `httpx.AsyncClient`.
- [ ] I can articulate the thread-safety limitation of in-memory status tracking.

---

## Supplementary Links

- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [FastAPI File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/)
- [httpx AsyncClient for testing FastAPI](https://www.starlette.io/testclient/)
- [Pydantic v2 Field validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [FastAPI pagination patterns](https://fastapi.tiangolo.com/tutorial/query-params/)
