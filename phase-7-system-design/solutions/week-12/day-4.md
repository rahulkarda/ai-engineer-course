# Solutions — Week 12 · Day 4: Architecture Review Exercises

## Exercise 1 — Agentic AI pipeline critique

**Design under review**:
> LangGraph agent → tool calls execute sequentially → results cached in a Python dict in-memory → LLM synthesises response → response stored in SQLite with no schema migrations tool.

### Flaw 1 — In-memory dict cache (Correctness + Scale)

**Category**: Correctness (data lost on restart) + Scale (not shared across replicas)

**Problem**: A Python dict lives in a single process. On restart or crash, all cached tool results are gone. If the service scales to 2+ replicas, each has its own dict — same tool call returns different cached values.

**Fix**: Replace with a shared cache (Redis) with explicit TTL per tool result type (e.g., calendar lookups: 5 min, database reads: 30 s).

### Flaw 2 — Sequential tool calls (Scale + Operational)

**Category**: Scale

**Problem**: If the agent calls 5 tools sequentially and each takes 500 ms, total latency = 2.5 s just for tools, before the LLM synthesises. Many of these calls are independent (e.g., fetch user profile + fetch calendar + fetch documents simultaneously).

**Fix**: Identify independent tool calls in the agent graph and execute them concurrently using `asyncio.gather`. LangGraph supports parallel node execution natively.

### Flaw 3 — SQLite for response storage (Scale + Operational)

**Category**: Scale (write lock contention) + Operational (no migrations)

**Problem**: SQLite is a file-level lock — only one writer at a time. At > 10 concurrent users, write contention causes timeouts. No migration tool means schema changes require manual intervention and risk data loss.

**Fix**: Replace with Postgres. Use Alembic for schema migrations. SQLite is acceptable for single-user local development only.

### Flaw 4 — No schema migrations tool (Correctness)

**Category**: Correctness (deployment risk)

**Problem**: Without Alembic or similar, adding a column to the response table requires a `ALTER TABLE` run manually in production — error-prone, undocumented, not reproducible in staging.

**Fix**: Add Alembic to the project. Version all schema changes. Run `alembic upgrade head` in the deployment pipeline.

---

## Exercise 2 — RAG system critique

**Design under review**:
> All documents embedded at startup (blocking), stored in a Pinecone index named `prod`, no namespace separation, query uses top-k=100, all 100 chunks passed directly to LLM.

### Flaw 1 — Blocking embedding at startup (Operational)

**Problem**: Embedding 1M documents at startup takes hours. The service is unavailable until complete. On restart after a crash, the full re-embedding run must complete before the service is healthy — killing uptime.

**Fix**: Move embedding to an async ingestion pipeline (background workers, triggered by document upload events). The service starts with whatever is already indexed and processes new documents asynchronously.

### Flaw 2 — No namespace separation (Correctness — multi-tenant security)

**Problem**: All tenants' data lives in the same Pinecone index with no namespace separation. A bug in the tenant filter (or a missing filter) leaks tenant A's documents to tenant B's queries.

**Fix**: Use Pinecone namespaces (one namespace per tenant) or a dedicated index per tenant. Enforce namespace selection in the query service — no query can run without a namespace parameter.

### Flaw 3 — top-k=100 with no reranking, all chunks to LLM (Scale + Correctness)

**Problem**: 100 chunks × avg 512 tokens = 51,200 tokens per query. At $3/M tokens, that's $0.15/query. For 1,000 queries/day: $150/day → $4,500/month just in input tokens. Also, LLM context quality degrades with noise — many of those 100 chunks are irrelevant.

**Fix**: Use a two-stage retrieval: top-k=20 from vector search → cross-encoder reranker → top-5 to LLM. Typical quality improvement: faithfulness +10–20%; cost reduction: 80%.

---

## Exercise 3 — Async FastAPI endpoint with retry and streaming

```python
# phase-7-system-design/week-12/solutions/async_llm_endpoint.py
import os
import logging
import asyncio
from typing import AsyncIterator

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI()

_client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5  # seconds


class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 512


async def _stream_with_retry(prompt: str, max_tokens: int) -> AsyncIterator[str]:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with _client.messages.stream(
                model="claude-sonnet-5-5",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
            return  # success — stop retrying
        except anthropic.RateLimitError as e:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("rate limit on attempt %d, sleeping %.1fs: %s", attempt, delay, e)
            if attempt == _MAX_RETRIES:
                raise
            await asyncio.sleep(delay)
        except anthropic.APIStatusError as e:
            logger.error("API error on attempt %d: %s", attempt, e)
            if attempt == _MAX_RETRIES or e.status_code < 500:
                raise
            await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))


@app.post("/v1/query")
async def query(req: QueryRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for chunk in _stream_with_retry(req.prompt, req.max_tokens):
                yield f"data: {chunk}\n\n"
        except Exception as e:
            logger.error("stream failed: %s", e)
            yield f"data: [ERROR: {type(e).__name__}]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## Exercise 4 — When NOT to split a monolith

**Three criteria before splitting**

1. **A team (not just a component) is being blocked**: splitting makes sense when multiple teams need to deploy independently. A single-team monolith gains nothing from microservices — it gains operational complexity.
2. **A specific scalability bottleneck is proven, not hypothesised**: profile and measure first. The embedding service may need 8× GPU capacity; the admin service needs 1 replica. Split only when load characteristics diverge measurably.
3. **The service boundaries are stable**: if the data model and API between hypothetical services is still changing weekly, splitting now means rewriting inter-service contracts constantly. Wait for the interface to stabilise.

**Operational costs introduced by the split**

- **Distributed tracing overhead**: every request now requires correlation IDs and a trace aggregator (Jaeger, Tempo).
- **Network failure surface**: calls that were in-process become HTTP/gRPC calls that can timeout, rate-limit, or fail independently.
- **Deployment complexity**: each service needs its own CI/CD pipeline, health checks, rollout strategy, and monitoring dashboard.
- **Data consistency**: operations that were a single DB transaction now require saga patterns or 2PC — both add complexity and failure modes.
