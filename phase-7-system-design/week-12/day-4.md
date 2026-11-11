# Phase 7 · Week 12 · Day 4 — Architecture Review Exercises

> Objective: Critique given system designs, identify structural flaws, and propose concrete improvements.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Architecture review exercises appear in senior/staff rounds as a design critique: you are given a diagram or description and asked to find flaws, assess trade-offs, and propose improvements. This tests whether you can distinguish good design from familiar-looking design.

**Three categories of architectural flaw**

| Category | Examples |
|---|---|
| Correctness | Race conditions, missing idempotency, unbounded queue |
| Scale | Single-node bottleneck, O(N²) fanout, no cache invalidation strategy |
| Operational | No observability, no graceful degradation, no backpressure |

**Review protocol (apply to every diagram)**

```
1. Identify the system's core invariants (what must always be true?)
2. Walk each data path: what fails at 10×, 100× load?
3. Walk each failure path: what happens when component X is down?
4. Look for: missing layers (auth, rate limit, CDN), hidden coupling, god services
5. Propose exactly N improvements where N = time_remaining / 5 min
```

## How it works internally

Most architecture flaws fall into five patterns:

```
Pattern 1 — The God Service
  Symptom: one service owns storage, business logic, auth, and notifications
  Fix: extract each responsibility; communicate via events

Pattern 2 — The Synchronous Fan-out
  Symptom: one request triggers N synchronous calls (N grows with users)
  Fix: async queue + idempotent workers

Pattern 3 — The Missing Cache
  Symptom: hot read path hits DB on every request
  Fix: read-through cache with appropriate TTL and invalidation strategy

Pattern 4 — The Unbounded Growth
  Symptom: append-only log/table with no TTL, rotation, or archival policy
  Fix: retention policy + archival tier + index cleanup

Pattern 5 — The Silent Failure
  Symptom: no dead-letter queue, no alerting on error rates, no circuit breaker
  Fix: DLQ + alerting + circuit breaker with fallback
```

## Worked example

### Architecture to review

> An AI document Q&A platform for 500 enterprise users. The diagram description:
>
> - A single FastAPI monolith handles: user auth, document upload, chunking, embedding, vector search, LLM call, response logging, usage billing.
> - All embeddings stored in Postgres `BYTEA` column.
> - LLM calls made synchronously in the HTTP request handler; timeout = 30 s.
> - No retry logic on LLM calls.
> - All logs written to local disk.

**Model critique**

**Flaw 1 — God service (Pattern 1)**
The monolith owns 8 distinct responsibilities. A spike in upload traffic will degrade Q&A latency; a billing bug can crash the Q&A path.
Fix: split into at minimum: `auth-service`, `ingest-service`, `query-service`, `billing-service`. Communicate ingest completion via event (SQS/Kafka).

**Flaw 2 — Embeddings in Postgres BYTEA (correctness + scale)**
BYTEA columns cannot be indexed for approximate nearest-neighbour search. Every query is a full table scan. At 10 M chunks this is fatal.
Fix: Qdrant, Weaviate, or pgvector (`vector` type + HNSW index).

**Flaw 3 — Synchronous LLM call in HTTP handler (scale + operational)**
30-second timeout blocks a thread/worker. At 50 concurrent users, server thread pool exhausts. No retry on transient LLM errors means one 429 fails the user request.
Fix: async handler (FastAPI `async def`), exponential backoff with jitter (3 retries), streaming response for perceived latency.

**Flaw 4 — No retry + no fallback (operational)**
A single LLM provider failure takes the whole system down.
Fix: retry with exponential backoff; secondary model fallback (`claude-haiku-5-5` for degraded mode); circuit breaker.

**Flaw 5 — Local disk logging (operational)**
Logs lost on container restart. No aggregation across instances.
Fix: structured JSON logging → stdout → log aggregator (Datadog, CloudWatch, Loki).

**Proposed improved architecture**
```
Client
  │
API Gateway (auth, rate limit)
  │
  ├─ Query Service (async FastAPI)
  │    └─ Vector DB (Qdrant)
  │    └─ LLM Gateway (retry, fallback, circuit breaker)
  │    └─ Prompt Service
  │
  ├─ Ingest Service
  │    └─ Queue (SQS) ← upload events
  │    └─ Chunker workers
  │    └─ Embedder workers → Vector DB
  │
  ├─ Auth Service
  └─ Billing Service ← usage events from Query Service

All services → structured logs → log aggregator
              → metrics → Prometheus → Grafana
              → traces → OpenTelemetry
```

## Exercises

1. **Critique this design** (15 min):
   > An agentic AI pipeline: user request → LangGraph agent → tool calls execute sequentially → results cached in a Python dict in-memory → LLM synthesises response → response stored in SQLite with no schema migrations tool.
   
   Find at least 4 flaws. Categorise each as correctness, scale, or operational. Propose a fix for each.

2. **Critique this design** (15 min):
   > A RAG system: all documents embedded at startup (blocking), stored in a Pinecone index named `prod`, no namespace separation between tenants, query uses cosine similarity with top-k=100, all 100 chunks passed directly to the LLM with no reranking.
   
   Find at least 3 flaws. Propose fixes.

3. **Improvement proposal**: Take Flaw 3 from the worked example. Write a complete async FastAPI endpoint (with retry, streaming) that replaces the synchronous handler. Use `os.getenv("ANTHROPIC_API_KEY")` for credentials.

4. **Staff-level question**: A colleague proposes moving from a monolith to microservices "for scale." Under what conditions is this premature? State 3 criteria that should be met before the split. What operational cost does the split introduce?

(Solutions: ../solutions/week-12/day-4.md)

## Self-check
- [ ] I can identify the 5 common architecture flaw patterns by name
- [ ] I classify each flaw as correctness, scale, or operational before proposing a fix
- [ ] I can critique a diagram in 15 min and produce prioritised, specific improvements
- [ ] I know when NOT to split a monolith

## Supplementary
- [The Architecture of Open Source Applications](https://aosabook.org/en/)
- [Patterns of Enterprise Application Architecture — Fowler](https://martinfowler.com/books/eaa.html)
