# Solutions — Week 08 · Day 6 (Saturday Build Day)

Day 6 is a hands-on build session. There are no single "correct" code answers — this file is your self-check guide. Work through each section to verify your platform is complete and working.

---

## Complete File Tree

A finished Document Intelligence Platform should look like this:

```
document-intelligence-platform/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── golden_dataset.json
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app factory, lifespan, middleware registration
│   ├── config.py                      # pydantic-settings Settings class
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ingest.py                  # POST /ingest, POST /ingest/batch
│   │   ├── query.py                   # POST /query, POST /query/stream
│   │   ├── eval.py                    # POST /eval/batch, GET /eval/results/{run_id}
│   │   └── health.py                  # GET /health
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   └── rag_agent.py               # LangGraph agentic RAG graph with citation tool
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── retrieval.py               # hybrid_retrieve(), rerank()
│   │   └── citation.py                # extract_citations(), format_citations()
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── documents.py               # Document, Chunk, IngestRequest, IngestResponse
│   │   ├── query.py                   # QueryRequest, QueryResponse, Citation
│   │   └── eval.py                    # EvalRequest, EvalResult, EvalBatchResponse
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── rag.py                     # RAG system prompt, citation format instructions
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py               # document loading, chunking, embedding
│   │   ├── vector_store.py            # ChromaDB client wrapper
│   │   └── reranker.py                # cross-encoder reranker wrapper
│   │
│   └── middleware/
│       ├── __init__.py
│       ├── auth.py                    # API key validation
│       ├── rate_limit.py              # token-bucket middleware
│       └── pii_scrubber.py            # PIIScrubber, scrub_middleware
│
└── tests/
    ├── conftest.py                    # shared fixtures: TestClient, mock_llm, sample_docs
    ├── unit/
    │   ├── test_chunking.py
    │   ├── test_embeddings.py
    │   ├── test_retrieval.py
    │   ├── test_reranker.py
    │   ├── test_citations.py
    │   ├── test_pii_scrubber.py
    │   ├── test_rate_limit.py
    │   └── test_health.py
    └── integration/
        ├── test_ingest_endpoint.py
        ├── test_query_endpoint.py
        └── test_eval_endpoint.py
```

---

## Validation Commands

### Start the platform

```bash
# Copy and configure secrets
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY, API_KEY_SECRET, etc.

# Build and launch all services
docker-compose up --build -d

# Tail logs
docker-compose logs -f app
```

### Smoke-test each endpoint

```bash
export API_KEY="$(grep API_KEY_SECRET .env | cut -d= -f2)"
export BASE="http://localhost:8000"

# Health check
curl -s $BASE/health | python3 -m json.tool

# Ingest a document
curl -s -X POST $BASE/ingest \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text": "Retrieval-augmented generation (RAG) combines dense retrieval with generative models.", "source": "intro.txt", "metadata": {"topic": "RAG"}}' \
  | python3 -m json.tool

# Query with citations
curl -s -X POST $BASE/query \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?", "top_k": 5}' \
  | python3 -m json.tool

# Streaming query
curl -s -X POST $BASE/query/stream \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain RAG in simple terms", "top_k": 3}'

# Batch eval
curl -s -X POST $BASE/eval/batch \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @golden_dataset.json \
  | python3 -m json.tool
```

### Stop the platform

```bash
docker-compose down
# Remove volumes too (clears ChromaDB data):
docker-compose down -v
```

### Run tests

```bash
# All tests
pytest tests/ -v

# Unit only (fast, no Docker needed)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80
```

---

## Common Build Pitfalls and Fixes

### ChromaDB "Connection refused" on first startup

**Symptom:** `httpx.ConnectError: [Errno 111] Connection refused` in app logs immediately after `docker-compose up`.

**Cause:** The app starts before ChromaDB is ready.

**Fix:** Add a health-check dependency in `docker-compose.yml`:

```yaml
services:
  app:
    depends_on:
      chromadb:
        condition: service_healthy
  chromadb:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s
```

Alternatively, add retry logic in `src/services/vector_store.py` using `tenacity`.

---

### Citations list is empty

**Symptom:** `/query` returns `"citations": []` even when relevant chunks were retrieved.

**Checklist:**
1. Does the RAG prompt explicitly instruct the model to cite sources by chunk ID?
2. Is `extract_citations()` parsing the model output correctly? Log `raw_response` at DEBUG level and inspect.
3. Did the ingestion step store `chunk_id` / `source` in ChromaDB metadata? Query the collection directly:

```python
import chromadb
client = chromadb.HttpClient(host="localhost", port=8000)
col = client.get_collection("documents")
print(col.peek())  # Check metadata keys
```

4. Is the citation extraction regex matching the format the model actually outputs?

---

### Streaming not working (client receives full body at once)

**Symptom:** `POST /query/stream` waits, then dumps everything at once.

**Cause:** A response middleware (e.g., logging or Gzip) is buffering the body.

**Fix:**
- Return a `StreamingResponse` with `media_type="text/event-stream"`.
- Ensure no middleware wraps the response body. Gzip middleware should skip `text/event-stream`:

```python
from starlette.middleware.gzip import GZipMiddleware
# GZip already skips streaming responses in recent Starlette — confirm version.
```

- Test streaming with `curl -N` (no buffering) or `httpx.stream()`.

---

### Auth returns 401 on all requests

**Symptom:** Every request returns `{"detail": "Invalid or missing API key"}`.

**Checklist:**
1. Is `API_KEY_SECRET` set in `.env` and in the running container? `docker-compose exec app env | grep API_KEY`
2. Is the `Authorization: Bearer <key>` header formatted correctly? No extra whitespace.
3. Does `auth.py` compare with `secrets.compare_digest` rather than `==`? The latter is fine functionally but flag it as a timing-safe concern.
4. Is the `/health` endpoint correctly excluded from auth middleware?

---

### Evaluation scores unexpectedly low (below 0.5)

**Symptom:** Eval run returns `mean_score: 0.12` on a seemingly reasonable golden dataset.

**Checklist:**
1. Was the golden dataset ingested into the vector store before running eval? Ingest it once: `POST /ingest/batch`.
2. Are `expected_answer` strings in `golden_dataset.json` complete sentences, not single keywords?
3. Is the similarity metric correct? BM25 tokenization does not match semantic similarity — ensure the scoring function is appropriate for the answer format.
4. Lower `eval_threshold` in config temporarily to isolate which questions score low: `EVAL_SCORE_THRESHOLD=0.3`.

---

## Sample golden_dataset.json

This file is both the request body for `POST /eval/batch` and the source of truth for evaluation. Three examples covering different answer types:

```json
{
  "questions": [
    {
      "question": "What is retrieval-augmented generation?",
      "ground_truth_answer": "Retrieval-augmented generation (RAG) is a technique that combines a retrieval system with a generative language model. Given a user query, relevant documents are first retrieved from a knowledge base and then provided as context to the language model, which uses them to generate a grounded, factual response.",
      "context": "The retrieved documents should include the introductory passage about RAG from the ingested corpus.",
      "expected_source_keywords": ["retrieval", "generative", "context", "knowledge base"]
    },
    {
      "question": "What are the main chunking strategies used in RAG systems?",
      "ground_truth_answer": "Common chunking strategies include fixed-size chunking with overlap, recursive character splitting that respects natural text boundaries such as paragraphs and sentences, semantic chunking that groups sentences by meaning, and document-structure-aware chunking that splits on headings or sections.",
      "context": "Answer should reference at least two named chunking strategies.",
      "expected_source_keywords": ["fixed-size", "overlap", "recursive", "semantic"]
    },
    {
      "question": "How does hybrid retrieval combine dense and sparse methods?",
      "ground_truth_answer": "Hybrid retrieval runs both a dense vector search (using embedding similarity) and a sparse keyword search (such as BM25) in parallel, then merges the two result sets. Reciprocal Rank Fusion (RRF) is a common merging strategy that re-scores each document based on its rank in each result list, rewarding documents that rank highly in both.",
      "context": "Answer must mention both dense and sparse retrieval and a fusion method.",
      "expected_source_keywords": ["dense", "sparse", "BM25", "reciprocal rank fusion", "RRF"]
    }
  ]
}
```

---

## Expected Test Run Output

A fully passing test suite should look like this:

```
============================= test session starts ==============================
platform linux -- Python 3.12.x, pytest-8.x.x
collected 42 items

tests/unit/test_chunking.py::test_fixed_size_chunk_count PASSED
tests/unit/test_chunking.py::test_chunk_overlap PASSED
tests/unit/test_chunking.py::test_empty_document PASSED
tests/unit/test_embeddings.py::test_embedding_dimension PASSED
tests/unit/test_embeddings.py::test_embedding_is_normalized PASSED
tests/unit/test_retrieval.py::test_hybrid_retrieve_returns_top_k PASSED
tests/unit/test_retrieval.py::test_rrf_score_ordering PASSED
tests/unit/test_reranker.py::test_reranker_reorders_results PASSED
tests/unit/test_citations.py::test_extract_citation_ids PASSED
tests/unit/test_citations.py::test_format_citations_markdown PASSED
tests/unit/test_pii_scrubber.py::test_email_scrubbing[...] PASSED (5 cases)
tests/unit/test_pii_scrubber.py::test_phone_scrubbing[...] PASSED (5 cases)
tests/unit/test_pii_scrubber.py::test_ssn_scrubbing[...] PASSED (2 cases)
tests/unit/test_pii_scrubber.py::test_no_pii_in_combined_output PASSED
tests/unit/test_rate_limit.py::test_requests_under_limit_pass PASSED
tests/unit/test_rate_limit.py::test_requests_over_limit_return_429 PASSED
tests/unit/test_rate_limit.py::test_retry_after_header_present PASSED
tests/unit/test_health.py::test_health_all_healthy PASSED
tests/unit/test_health.py::test_health_redis_down_returns_503 PASSED
tests/unit/test_health.py::test_health_chromadb_down_returns_503 PASSED
tests/integration/test_ingest_endpoint.py::test_ingest_single_document PASSED
tests/integration/test_ingest_endpoint.py::test_ingest_batch PASSED
tests/integration/test_ingest_endpoint.py::test_ingest_requires_auth PASSED
tests/integration/test_query_endpoint.py::test_query_returns_answer_and_citations PASSED
tests/integration/test_query_endpoint.py::test_query_stream_returns_chunks PASSED
tests/integration/test_eval_endpoint.py::test_eval_batch_returns_scores PASSED
tests/integration/test_eval_endpoint.py::test_eval_batch_aggregate_stats PASSED

============================== 42 passed in 6.83s ==============================
```

If you see fewer than ~35 tests, check for missing test files matching the tree above. If tests are skipped with `no module named src.*`, ensure `pyproject.toml` has `[tool.pytest.ini_options] pythonpath = ["."]`.
