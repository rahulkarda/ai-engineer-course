# Phase 4 · Week 08 · Day 6 — Saturday: Full Platform Build

> Objective: Assemble the complete Document Intelligence Platform from scratch in a single guided session.  ·  Time: ~4-5 hrs  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]]

## Concept

This is an integration day. Each prior day delivered one isolated component. Today you wire them together into a single deployable service. The skill being practiced is not learning new primitives — it is the systems-thinking required to make five independent modules compose cleanly: agreeing on data contracts, handling failures at each boundary, and validating end-to-end behavior before declaring victory.

A platform is not the sum of its parts. Bugs that are invisible in unit tests — state leaks between requests, async timing issues, embedding dimension mismatches, middleware ordering errors — only surface when the full system runs. Build the scaffold first, add components one at a time, run smoke tests after each addition.

## How it works internally

The architecture you are building has three concentric layers:

**Transport layer** (`src/api/`) — FastAPI routers, Pydantic request/response schemas, middleware chain (auth, input validation, rate limiting). This layer knows nothing about chunking or embeddings. It translates HTTP into domain objects.

**Domain layer** (`src/services/`) — `IngestionService`, `RetrievalService`, `GenerationService`. These are pure Python classes that hold no HTTP state. They accept domain objects, call infrastructure adapters, and return domain objects. This layer is where the RAG pipeline logic lives.

**Infrastructure layer** (`src/adapters/`) — thin wrappers around external systems: ChromaDB client, Anthropic client, Redis client. Each adapter owns one external dependency. Tests mock at this boundary.

The main application wires all three layers together at startup via dependency injection (FastAPI `Depends`).

## Build Checklist

Work through the phases in order. Mark each checkbox when the corresponding validation command passes.

---

### Phase 1 — Project scaffold and config (30 min)

**Goal**: A running FastAPI app that returns 200 from `/health`, reading all config from environment variables.

Tasks:
- [ ] Create the directory tree below
- [ ] Write `pyproject.toml` with dependencies: `fastapi`, `uvicorn[standard]`, `anthropic`, `chromadb-client`, `redis`, `pydantic-settings`, `structlog`, `pytest`, `pytest-asyncio`, `httpx`
- [ ] Copy/adapt `src/config.py` from day-5 (Settings class with all required fields)
- [ ] Write `src/main.py` with middleware chain and `/health` endpoint
- [ ] Write `.env.example` with all variable names, no values

Directory tree:

```
document-intelligence/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── ingest.py
│   │   │   ├── query.py
│   │   │   └── eval.py
│   │   └── schemas.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py
│   │   ├── retrieval.py
│   │   └── generation.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── chroma.py
│   │   ├── anthropic_client.py
│   │   └── redis_client.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── security.py
│   └── eval/
│       ├── __init__.py
│       ├── harness.py
│       └── golden_dataset.py
├── tests/
│   ├── unit/
│   └── integration/
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── .gitignore
```

**Acceptance criteria for Phase 1:**

```bash
# Start the app with minimal env vars
export ANTHROPIC_API_KEY=$(python -c "import os; print(os.getenv('ANTHROPIC_API_KEY', 'not-set'))")
export API_SECRET_KEY=dev-only-not-real
uvicorn src.main:app --port 8080 &

# Must return 200
curl -s http://localhost:8080/health | python -m json.tool
```

---

### Phase 2 — Ingestion pipeline integration (60 min)

**Goal**: `POST /ingest` accepts a PDF URL or raw text, runs chunking + embedding, stores in ChromaDB, returns a document ID.

Tasks:
- [ ] Write `src/adapters/chroma.py` — wrap `chromadb.HttpClient`, expose `upsert_chunks(doc_id, chunks, embeddings)` and `collection_info()`
- [ ] Write `src/services/ingestion.py` — `IngestionService.ingest(source: IngestRequest) -> IngestResult`
  - If source is a URL: fetch with `httpx.AsyncClient`, extract text (use `pypdf` for PDFs)
  - Chunk with `RecursiveCharacterTextSplitter` (chunk_size=512, overlap=64)
  - Embed with Anthropic or a local model (configurable via `Settings.embedding_provider`)
  - Upsert to ChromaDB with metadata: `doc_id`, `chunk_index`, `source_url`, `ingested_at`
- [ ] Write `src/api/routes/ingest.py` — the FastAPI router
- [ ] Write `src/api/schemas.py` — `IngestRequest`, `IngestResult` Pydantic models
- [ ] Wire the router into `src/main.py`

Key schema contracts:

```python
# src/api/schemas.py
from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field


class IngestRequest(BaseModel):
    source_url: HttpUrl | None = None
    raw_text: str | None = None
    collection: str = Field(default="default", max_length=64)
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {"example": {
        "source_url": "https://example.com/paper.pdf",
        "collection": "research",
    }}}


class IngestResult(BaseModel):
    doc_id: str
    chunks_created: int
    collection: str
    ingested_at: datetime
```

**Acceptance criteria for Phase 2:**

```bash
# ChromaDB must be running (via docker-compose or local)
# Ingest a document
curl -s -X POST http://localhost:8080/ingest \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"raw_text": "LangGraph is a framework for building stateful multi-actor applications with LLMs.", "collection": "test"}' \
  | python -m json.tool

# Must return doc_id and chunks_created > 0
```

---

### Phase 3 — FastAPI endpoints + auth (60 min)

**Goal**: All endpoints documented in `project.md` are implemented and protected by auth middleware.

Tasks:
- [ ] Write `src/api/routes/query.py` — `POST /query` that accepts a question and collection name, returns answer + citations
- [ ] Write `src/api/routes/eval.py` — `POST /eval/run` that accepts a golden dataset path and returns eval results
- [ ] Add `GET /collections` — list all collections in ChromaDB
- [ ] Add `DELETE /documents/{doc_id}` — remove a document and its chunks
- [ ] Verify SecurityMiddleware is applied to all routes except `/health`
- [ ] Write integration tests in `tests/integration/test_api.py` using `httpx.AsyncClient`

Key schema:

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2048)
    collection: str = Field(default="default")
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True)
    stream: bool = Field(default=False)


class Citation(BaseModel):
    doc_id: str
    chunk_index: int
    source_url: str | None
    excerpt: str = Field(..., max_length=500)
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    model: str
    tokens_used: int
    latency_ms: float
```

**Acceptance criteria for Phase 3:**

```bash
# Auth must be enforced
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}' | python -m json.tool
# Must return 401

# Auth must pass through
curl -s -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph?", "collection": "test"}' \
  | python -m json.tool
# Must return answer + citations list
```

---

### Phase 4 — Streaming + citations (60 min)

**Goal**: When `stream: true`, `POST /query` returns a Server-Sent Events stream. Each event is a JSON chunk with partial answer text. The final event includes citations.

Tasks:
- [ ] Write `src/services/generation.py` — `GenerationService.generate_streaming(...)` using `anthropic.AsyncAnthropic().messages.stream()`
- [ ] Update `src/api/routes/query.py` — when `request.stream` is true, return `EventSourceResponse` (install `sse-starlette`)
- [ ] Ensure citations are extracted from the streamed response and appended as the final SSE event
- [ ] Write `tests/unit/test_generation.py` — mock the Anthropic client stream with `unittest.mock.AsyncMock`

Streaming event format:

```
data: {"type": "delta", "text": "LangGraph is"}
data: {"type": "delta", "text": " a framework"}
data: {"type": "citations", "citations": [...]}
data: {"type": "done", "tokens_used": 142}
```

**Acceptance criteria for Phase 4:**

```bash
curl -s -X POST http://localhost:8080/query \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangGraph?", "collection": "test", "stream": true}' \
  --no-buffer
# Must emit SSE events, final event must contain citations
```

---

### Phase 5 — Observability + cost tracking (60 min)

**Goal**: Every LLM call is logged with model, tokens_in, tokens_out, latency_ms, and estimated cost. A `GET /metrics` endpoint returns aggregated stats.

Tasks:
- [ ] Add `structlog` structured logging to all service methods (replace any remaining `print` calls)
- [ ] Write `src/adapters/redis_client.py` — expose `increment_cost(tokens_in, tokens_out, model)` and `get_cost_summary()`
- [ ] After every Anthropic API call: log to structlog + update Redis cost counters
- [ ] Write `GET /metrics` endpoint — returns total requests, total tokens, estimated cost (use published token prices), average latency
- [ ] Add a request ID to every log line using a middleware that sets `request_id` in `contextvars`

Cost calculation (approximate, update when prices change):

```python
COST_PER_1K_INPUT_TOKENS = {
    "claude-haiku-5-5": 0.00025,
    "claude-sonnet-5-5": 0.003,
    "claude-opus-5-5": 0.015,
}
COST_PER_1K_OUTPUT_TOKENS = {
    "claude-haiku-5-5": 0.00125,
    "claude-sonnet-5-5": 0.015,
    "claude-opus-5-5": 0.075,
}
```

**Acceptance criteria for Phase 5:**

```bash
# Make a query, then check metrics
curl -s -X GET http://localhost:8080/metrics \
  -H "Authorization: Bearer $API_SECRET_KEY" | python -m json.tool
# Must show total_requests >= 1, estimated_cost_usd > 0
```

---

### Phase 6 — Eval gate + golden dataset (60 min)

**Goal**: `POST /eval/run` runs the evaluation harness against a golden dataset and returns pass/fail per question plus aggregate scores.

Tasks:
- [ ] Write `src/eval/golden_dataset.py` — load a JSON file of `{question, expected_answer, source_doc_id}` records
- [ ] Write `src/eval/harness.py` — for each record, run the full RAG pipeline, score with faithfulness + answer relevance metrics (LLM-as-judge using claude-haiku-5-5, mocked in tests)
- [ ] Write `src/api/routes/eval.py` — `POST /eval/run` accepts `{dataset_path: str, collection: str}`, returns eval results
- [ ] Create `tests/golden/sample_dataset.json` with at least 3 question/answer pairs about your test documents
- [ ] Write `tests/unit/test_harness.py` — mock all LLM calls, test scoring logic in isolation

Golden dataset format:

```json
[
  {
    "id": "q1",
    "question": "What does LangGraph help you build?",
    "expected_answer": "Stateful multi-actor applications with LLMs",
    "source_doc_id": "doc_abc123",
    "tags": ["factual"]
  }
]
```

**Acceptance criteria for Phase 6:**

```bash
curl -s -X POST http://localhost:8080/eval/run \
  -H "Authorization: Bearer $API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "tests/golden/sample_dataset.json", "collection": "test"}' \
  | python -m json.tool
# Must return per-question scores and aggregate faithfulness/relevance
```

---

### Phase 7 — Docker + smoke tests (30 min)

**Goal**: `docker-compose up` starts all services, smoke tests pass against the containerized stack.

Tasks:
- [ ] Copy/adapt `docker/Dockerfile` and `docker-compose.yml` from day-5
- [ ] Write `tests/smoke/test_smoke.py` — uses `httpx` to hit the running container stack (skip if `SMOKE_TEST_BASE_URL` env var is not set)
- [ ] Verify the full pipeline in Docker: ingest → query → eval
- [ ] Check logs: no secrets printed, PII scrubbed, request IDs present on all lines

**Acceptance criteria for Phase 7:**

```bash
docker-compose up -d --build
sleep 15  # wait for services

export SMOKE_TEST_BASE_URL=http://localhost:8080
pytest tests/smoke/ -v

docker-compose logs app | grep "ANTHROPIC_API_KEY"
# Must return nothing — key must not appear in logs
```

---

## Validation commands (full suite)

Run these in order once all phases are complete:

```bash
# 1. Unit tests (no external services required)
pytest tests/unit/ -v --tb=short

# 2. Integration tests (requires ChromaDB + Redis)
docker-compose up -d chromadb redis
pytest tests/integration/ -v --tb=short

# 3. Smoke tests (requires full stack)
docker-compose up -d
sleep 15
SMOKE_TEST_BASE_URL=http://localhost:8080 pytest tests/smoke/ -v

# 4. Linting
ruff check src/ tests/

# 5. Type check
mypy src/ --strict --ignore-missing-imports
```

All five command groups must exit 0 before you consider the build complete.

## Self-check

- [ ] I can explain why the domain layer (services/) has no FastAPI imports
- [ ] I can trace a single request from HTTP ingress through all layers to the ChromaDB write and back
- [ ] I can explain what happens when the Anthropic API is down — which layer catches the error and what the API returns
- [ ] I can run the full test suite and interpret a failing test correctly
- [ ] I can start the full stack with a single `docker-compose up` command

## Supplementary links

- [FastAPI dependency injection](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [sse-starlette — Server-Sent Events for FastAPI](https://github.com/sysid/sse-starlette)
- [structlog quickstart](https://www.structlog.org/en/stable/getting-started.html)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/en/latest/)
- [httpx AsyncClient for testing](https://www.python-httpx.org/async/)
