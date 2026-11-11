# Solutions — Week 08 · Project Review

This file is a self-check guide for the capstone project. No single reference implementation exists — use these criteria to assess your own submission before peer review.

---

## Project Completion Checklist

Check each deliverable against the pass criteria. A single "Fail" in any item marked "Required" blocks the project from passing.

| # | Deliverable | Required | Pass Criteria | Fail Signals |
|---|---|---|---|---|
| 1 | Document ingestion endpoint | Yes | `POST /ingest` accepts text + metadata, returns `document_id` and `chunk_count` | Returns 500 on valid input; chunk_count is always 0 |
| 2 | Chunking implementation | Yes | Configurable `CHUNK_SIZE` and `CHUNK_OVERLAP` via env vars; at least one strategy (recursive split) tested | Hardcoded sizes; no overlap; no tests |
| 3 | Embeddings stored in vector store | Yes | Chunks appear in ChromaDB with `chunk_id` in metadata after ingestion | Collection is empty after ingest; metadata missing source field |
| 4 | Hybrid retrieval | Yes | Both dense (vector) and sparse (BM25 or keyword) passes run; results merged with RRF or weighted sum | Only dense or only sparse; no merge logic |
| 5 | Reranker applied | Yes | Retrieved candidates are reranked before passing to LLM; top-N is smaller than top-K from retrieval | Reranker present but not called; all K chunks passed to LLM |
| 6 | Citations in response | Yes | `POST /query` response includes `citations` array with `chunk_id` and `source`; citations are non-empty when relevant docs are ingested | Citations always empty; citation IDs not present in vector store |
| 7 | Agentic RAG loop | Yes | LangGraph graph with at least: retrieve node, generate node, and a conditional edge that can re-retrieve | Linear pipeline with no LangGraph; no conditional retry |
| 8 | FastAPI service runs | Yes | `docker-compose up` starts the service; all endpoints respond; no import errors | Service crashes on startup; endpoints return 500 |
| 9 | API key auth | Yes | Requests without `Authorization: Bearer <key>` return 401; health endpoint is exempt | No auth; auth blocks health check |
| 10 | Test suite passes | Yes | `pytest tests/ -v` runs with no failures; all LLM calls mocked | Tests import errors; tests call real Anthropic API |
| 11 | PII scrubber active | No (bonus) | Logging middleware scrubs email/phone/SSN before log emission | PII appears raw in log output |
| 12 | Eval endpoint functional | Yes | `POST /eval/batch` returns per-question scores and aggregate stats | Endpoint missing; scores always 0 or 1 |

---

## Architecture Review

### What a strong submission looks like

All 7 RAG components are present and integrated into a single cohesive service:

- **Ingestion** — `/ingest` calls the chunker, the embedder, and upserts to ChromaDB in sequence
- **Chunking** — recursive character splitting; chunk size and overlap configurable via env; tested
- **Embeddings** — a consistent embedding model used at both ingest and query time (same model = comparable vectors)
- **Vector store** — ChromaDB (or equivalent) with metadata (chunk_id, source, document_id) stored per chunk
- **Hybrid retrieval** — dense + sparse run concurrently; RRF merges results; retrieval is a service function, not inline in the endpoint
- **Reranking** — cross-encoder or equivalent; top-K from retrieval shrinks to top-N for generation; reranker is mockable in tests
- **Agentic RAG** — LangGraph StateGraph with typed state; at least a retrieve → generate flow with a conditional edge; citations extracted in a post-process node

Additional production signals that distinguish strong submissions:
- Streaming endpoint (`/query/stream`) with SSE
- Rate limiting middleware with configurable `RATE_LIMIT_PER_MINUTE`
- Structured logging with request IDs
- `GET /health` checking ChromaDB and Redis with 503 on degraded

### Common gaps in weaker submissions

| Gap | Impact | Lesson to revisit |
|---|---|---|
| Missing streaming | Functional but incomplete spec | Day 3 — streaming and agentic loop |
| No PII scrubbing | Security concern; PII in logs | Day 5 — security hardening |
| Hardcoded secrets in `.env` committed or in Dockerfile | Critical security violation | Day 5 — config and secrets |
| Citations always empty | Core feature broken; retrieval not storing chunk_id in metadata | Day 2 — embeddings and vector store |
| Reranker declared but never called | Top-K raw results go to LLM; quality loss | Day 3 — retrieval and reranking |
| Tests call real Anthropic API | Tests require live key; break in CI without env; non-deterministic | Day 4 — evaluation harness |
| Single god-file implementation | Untestable; violates separation of concerns | Day 1 — architecture walkthrough |

---

## 5-Point Self-Evaluation Rubric

Score your submission 1-5 on each dimension. Be honest — this is for your own calibration.

### Dimension 1: RAG Pipeline Completeness

- **5** — All 7 components present, integrated, and working end-to-end. Streaming works. Eval gate passes at threshold >= 0.70.
- **4** — 6 of 7 components working. One gap (e.g., no streaming, or eval endpoint missing).
- **3** — Core path (ingest → query → answer with citations) works. Reranker or agentic loop missing.
- **2** — Basic retrieval works but citations empty, reranking absent, no agentic loop.
- **1** — Service starts but ingestion or query endpoint broken.

### Dimension 2: Code Quality and Architecture

- **5** — Separation of concerns across `agents/`, `tools/`, `services/`, `routers/`, `models/`, `prompts/`. Type hints everywhere. Pydantic models for all request/response shapes. No god files.
- **4** — Most separation in place. 1-2 modules too large or mixed concerns.
- **3** — Routers and services separated. Models partially typed. Some business logic in route handlers.
- **2** — Some modularization. Many untyped dicts. Logic spread across files without clear boundaries.
- **1** — Single-file implementation or near-equivalent.

### Dimension 3: Security Posture

- **5** — No hardcoded secrets anywhere. All secrets via `os.getenv`. `.env` in `.gitignore`. Docker uses `${VAR}` syntax. API key auth on all non-health endpoints. PII scrubbing on log paths.
- **4** — Secrets correctly externalized. Auth working. Minor gap (e.g., PII scrubber missing, or health not exempt from auth).
- **3** — Secrets externalized. Auth present but incomplete (e.g., only some endpoints protected).
- **2** — Secrets in `.env.example` with fake literal values (violates no-fake-creds rule). Or auth absent.
- **1** — Hardcoded secret in any committed file.

### Dimension 4: Testing

- **5** — >80% coverage. All LLM and HTTP calls mocked. Unit + integration split. `pytest.mark.parametrize` used. Tests run offline in <30s.
- **4** — 70-80% coverage. All LLM calls mocked. Most paths tested.
- **3** — 50-70% coverage. LLM calls mocked. Missing edge cases.
- **2** — Tests present but call real APIs or coverage <50%.
- **1** — No tests or tests always pass trivially (no assertions).

### Dimension 5: Operability

- **5** — `docker-compose up` starts everything. Health endpoint returns 200 with per-service status. Structured JSON logs. Rate limiting active. Prometheus metrics at `/metrics`.
- **4** — Docker works. Health endpoint present. Logs are structured. Rate limiting or metrics missing.
- **3** — Docker works but health endpoint absent or always returns 200. Standard (unstructured) logs.
- **2** — Docker runs but requires manual steps to start. No health endpoint.
- **1** — Docker does not start cleanly. Service only runs locally with manual setup.

**Total: __ / 25**

- 22-25: Production-ready. Submit with confidence.
- 17-21: Strong submission. Address gaps noted above.
- 12-16: Passing. Revisit weak dimensions before capstone demo.
- Below 12: Rebuild missing components. Review lesson files listed below.

---

## Lesson Pointers by Weak Area

| Weak area | Revisit |
|---|---|
| Chunking, embeddings, vector store | Week 8 Day 1 — Ingestion + Embeddings |
| Hybrid retrieval, RRF, reranker | Week 8 Day 2 — Retrieval + Reranking |
| Agentic loop, LangGraph, streaming, citations | Week 8 Day 3 — Agentic RAG + Citations |
| Eval harness, golden dataset, score thresholds | Week 8 Day 4 — Evaluation |
| Rate limiting, PII scrubbing, health checks, secrets | Week 8 Day 5 — Security + Hardening |
| Docker, docker-compose, observability | Week 8 Day 5 — Docker + Config |
| Architecture separation of concerns | Phase 4 Week 5 — RAG Architecture Patterns |
