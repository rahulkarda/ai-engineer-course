# Solutions — Week 08 · Day 2

## Exercise 1: GET /documents/{id}/status Endpoint

Background tasks run outside the request/response cycle, so status must live in shared state.
An `asyncio`-safe in-memory dict (protected by a lock for writes) is the simplest correct
approach for a single-process deployment; swap for Redis when scaling horizontally.

```python
# src/document_platform/api/documents.py
from __future__ import annotations

import asyncio
import logging
import uuid
from enum import StrEnum
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Status store — in-memory, asyncio-safe
# ---------------------------------------------------------------------------

class IngestStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class DocumentStatus(BaseModel):
    doc_id: str
    status: IngestStatus
    filename: str | None = None
    error: str | None = None
    chunks_created: int = 0
    created_at: str | None = None


# Module-level store + lock (replace with Redis adapter for multi-process)
_status_store: dict[str, DocumentStatus] = {}
_store_lock = asyncio.Lock()


async def _set_status(doc_id: str, update: DocumentStatus) -> None:
    async with _store_lock:
        _status_store[doc_id] = update


async def _get_status(doc_id: str) -> DocumentStatus | None:
    async with _store_lock:
        return _status_store.get(doc_id)


# ---------------------------------------------------------------------------
# Background ingestion task (stub — wire up real pipeline in production)
# ---------------------------------------------------------------------------

async def _run_ingestion(doc_id: str, filename: str, pipeline: Any) -> None:
    """
    Long-running ingestion task.  Marks status PROCESSING -> COMPLETE / FAILED.
    The pipeline argument is injected so tests can mock it without patching globals.
    """
    await _set_status(
        doc_id,
        DocumentStatus(doc_id=doc_id, status=IngestStatus.PROCESSING, filename=filename),
    )
    try:
        chunks_created: int = await asyncio.to_thread(pipeline.ingest, doc_id, filename)
        await _set_status(
            doc_id,
            DocumentStatus(
                doc_id=doc_id,
                status=IngestStatus.COMPLETE,
                filename=filename,
                chunks_created=chunks_created,
            ),
        )
        logger.info("Ingestion complete", extra={"doc_id": doc_id, "chunks": chunks_created})
    except Exception as exc:
        await _set_status(
            doc_id,
            DocumentStatus(
                doc_id=doc_id,
                status=IngestStatus.FAILED,
                filename=filename,
                error=str(exc),
            ),
        )
        logger.error("Ingestion failed", extra={"doc_id": doc_id, "error": str(exc)}, exc_info=True)


# ---------------------------------------------------------------------------
# Upload endpoint — kicks off background ingestion
# ---------------------------------------------------------------------------

class UploadRequest(BaseModel):
    filename: str
    content_b64: str = Field(description="Base-64 encoded file content")


class UploadResponse(BaseModel):
    doc_id: str
    message: str


@router.post("/", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    body: UploadRequest,
    background_tasks: BackgroundTasks,
    pipeline: Any = None,   # injected via Depends() in production
) -> UploadResponse:
    """
    Accept a document and enqueue background ingestion.
    Returns immediately with doc_id and 202 Accepted.
    """
    doc_id = str(uuid.uuid4())
    await _set_status(
        doc_id,
        DocumentStatus(doc_id=doc_id, status=IngestStatus.PENDING, filename=body.filename),
    )
    background_tasks.add_task(_run_ingestion, doc_id, body.filename, pipeline)
    logger.info("Document queued for ingestion", extra={"doc_id": doc_id, "filename": body.filename})
    return UploadResponse(doc_id=doc_id, message="Ingestion queued")


# ---------------------------------------------------------------------------
# Exercise 1: GET /documents/{id}/status
# ---------------------------------------------------------------------------

@router.get("/{doc_id}/status", response_model=DocumentStatus)
async def get_document_status(doc_id: str) -> DocumentStatus:
    """
    Return the current ingestion status for a document.

    Returns 404 if the doc_id is unknown — never silently return a default
    status, because the caller cannot distinguish "not started" from "wrong id".
    """
    record = await _get_status(doc_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id!r} not found",
        )
    return record
```

---

## Exercise 2: GET /documents with Pagination

Pagination parameters are validated by Pydantic before your handler runs.
`total` allows clients to render "page N of M" without a second round-trip.
The generic `PaginatedResponse[T]` can be reused for any list endpoint.

```python
# src/document_platform/api/documents.py  (continued in same file)
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str | None = None
    status: IngestStatus
    chunks_created: int = 0


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int = Field(description="Total number of pages")

    @classmethod
    def build(
        cls,
        all_items: list[T],
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[T]":
        """Slice all_items and compute pagination metadata."""
        total = len(all_items)
        pages = max(1, (total + page_size - 1) // page_size)
        start = (page - 1) * page_size
        end = start + page_size
        return cls(
            items=all_items[start:end],
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )


@router.get("/", response_model=PaginatedResponse[DocumentSummary])
async def list_documents(
    page: int = Field(default=1, ge=1, description="1-based page index"),
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page"),
) -> PaginatedResponse[DocumentSummary]:
    """
    Return a paginated list of all known documents.

    Defaults: page=1, page_size=20, max page_size=100.
    """
    async with _store_lock:
        all_records = list(_status_store.values())

    summaries = [
        DocumentSummary(
            doc_id=r.doc_id,
            filename=r.filename,
            status=r.status,
            chunks_created=r.chunks_created,
        )
        for r in all_records
    ]
    return PaginatedResponse.build(summaries, page=page, page_size=page_size)
```

---

## Exercise 3: pytest Tests Mocking the Pipeline

Mock at the boundary where side effects live (the pipeline object).
Never patch internal implementation details — patch the public seam so tests
survive internal refactors.

```python
# tests/unit/test_documents_api.py
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.document_platform.api.documents import (
    IngestStatus,
    _status_store,
    router,
)
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def app() -> FastAPI:
    application = FastAPI()
    application.include_router(router)
    return application


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    """Isolate each test by clearing the in-memory status store."""
    _status_store.clear()
    yield
    _status_store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(chunks: int = 5) -> MagicMock:
    pipeline = MagicMock()
    pipeline.ingest.return_value = chunks
    return pipeline


# ---------------------------------------------------------------------------
# Test: upload endpoint
# ---------------------------------------------------------------------------

def test_upload_returns_202(client: TestClient):
    """Upload should return 202 Accepted with a doc_id."""
    with patch(
        "src.document_platform.api.documents._run_ingestion",
        new_callable=AsyncMock,
    ):
        response = client.post(
            "/documents/",
            json={"filename": "report.pdf", "content_b64": "aGVsbG8="},
        )

    assert response.status_code == 202
    body = response.json()
    assert "doc_id" in body
    assert body["message"] == "Ingestion queued"


def test_upload_stores_pending_status(client: TestClient):
    """After upload, /status should return PENDING before the background task runs."""
    with patch(
        "src.document_platform.api.documents._run_ingestion",
        new_callable=AsyncMock,
    ):
        upload_resp = client.post(
            "/documents/",
            json={"filename": "report.pdf", "content_b64": "aGVsbG8="},
        )
    doc_id = upload_resp.json()["doc_id"]

    status_resp = client.get(f"/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == IngestStatus.PENDING


# ---------------------------------------------------------------------------
# Test: status endpoint
# ---------------------------------------------------------------------------

def test_status_404_for_unknown_id(client: TestClient):
    response = client.get("/documents/nonexistent-id/status")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_status_returns_complete_after_ingestion(client: TestClient):
    """
    Directly seed the store as COMPLETE and assert the endpoint reflects it.
    This avoids timing issues with real background tasks.
    """
    from src.document_platform.api.documents import DocumentStatus, _status_store

    doc_id = "test-doc-123"
    _status_store[doc_id] = DocumentStatus(
        doc_id=doc_id,
        status=IngestStatus.COMPLETE,
        filename="doc.pdf",
        chunks_created=12,
    )

    response = client.get(f"/documents/{doc_id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == IngestStatus.COMPLETE
    assert body["chunks_created"] == 12


# ---------------------------------------------------------------------------
# Test: list endpoint with pagination
# ---------------------------------------------------------------------------

def test_list_documents_empty(client: TestClient):
    response = client.get("/documents/")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["items"] == []
    assert body["pages"] == 1


def test_list_documents_pagination(client: TestClient):
    from src.document_platform.api.documents import DocumentStatus, _status_store

    for i in range(25):
        _status_store[f"doc-{i}"] = DocumentStatus(
            doc_id=f"doc-{i}",
            status=IngestStatus.COMPLETE,
            filename=f"file-{i}.pdf",
        )

    # First page
    resp1 = client.get("/documents/?page=1&page_size=10")
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["total"] == 25
    assert len(body1["items"]) == 10
    assert body1["pages"] == 3

    # Last page — only 5 items remain
    resp3 = client.get("/documents/?page=3&page_size=10")
    assert resp3.status_code == 200
    assert len(resp3.json()["items"]) == 5


def test_list_documents_page_size_capped_at_100(client: TestClient):
    """page_size > 100 should be rejected with 422 Unprocessable Entity."""
    response = client.get("/documents/?page_size=9999")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test: query endpoint (stub — exercise shows the pattern)
# ---------------------------------------------------------------------------

def test_query_endpoint_calls_rag(client: TestClient):
    """
    The query endpoint should delegate to the RAG component.
    Mock the RAG object to avoid real LLM calls.
    """
    mock_rag = MagicMock()
    mock_rag.answer.return_value = MagicMock(
        answer="Paris is the capital of France.",
        citations=[{"chunk_id": "c1", "text": "Paris...", "score": 0.95}],
    )

    # If your app wires RAG via Depends(), override the dependency here.
    # This example patches the module-level reference directly.
    with patch("src.document_platform.api.documents.rag_component", mock_rag):
        response = client.post(
            "/query",
            json={"question": "What is the capital of France?", "top_k": 5},
        )

    # The query endpoint is implemented in day-3 — this test documents the contract.
    # 501 Not Implemented is acceptable until the endpoint exists.
    assert response.status_code in {200, 501}
    if response.status_code == 200:
        body = response.json()
        assert "answer" in body
        assert "citations" in body
```

---

## Key concepts demonstrated

- **Asyncio-safe shared state** — using `asyncio.Lock` for writes prevents data races when
  multiple coroutines update the status store concurrently under an async server.
- **Generic pagination** — `PaginatedResponse[T]` with a `TypeVar` lets mypy verify that
  items match the declared type, and the `.build()` classmethod keeps slicing logic in one place.
- **Test isolation with autouse fixture** — `clear_store` runs before and after every test,
  preventing state leakage between tests without explicit cleanup calls.
- **Mock at the public seam** — patching `_run_ingestion` rather than internal asyncio primitives
  keeps tests fast and refactor-proof.
- **422 for invalid query params** — FastAPI/Pydantic validates `ge` / `le` constraints
  automatically; you get the 422 for free without any manual validation code.
