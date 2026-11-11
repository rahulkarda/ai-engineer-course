# Solutions — Week 12 · Day 6: Capstone Design Document (Model Answer)

> This is a complete model capstone system-design document for the Enterprise AI Assistant Platform (EAAP). Use it to calibrate the depth and concreteness expected after writing your own.

---

# Enterprise AI Assistant Platform (EAAP) — System Design Document

**Version**: 1.0  
**Author**: [AI Engineer]  
**Date**: 2026-11

---

## 1. Executive Summary

The Enterprise AI Assistant Platform (EAAP) is a multi-tenant, on-premises AI assistant for 50,000 employees at a Fortune 500 company. It enables staff to query internal documents, generate code, summarise meetings, and complete tasks via tool integrations — all without data leaving the corporate network.

The platform is built on a RAG architecture with a LangGraph orchestration layer, self-hosted LLM inference via vLLM, and Qdrant for vector storage. Multi-tenancy is enforced at the namespace level in the vector DB and at the row level in Postgres. Every interaction is logged for SOC 2 compliance.

Key trade-offs: self-hosted inference introduces GPU capital cost but eliminates per-token API spend and data-residency risk. Eventual consistency in the embedding pipeline is accepted in exchange for a non-blocking ingestion path. The LangGraph typed-state approach adds implementation overhead but makes debugging production failures tractable.

Expected outcomes: 800 QPS peak capacity at < 1.5 s p50 latency, 99.9% availability, full audit coverage, and a monthly operating cost (excluding capital) of approximately $45K.

---

## 2. Requirements

### Functional requirements

| # | Capability |
|---|---|
| F1 | Conversational Q&A over internal documents (RAG) |
| F2 | Code generation and explanation |
| F3 | Meeting summarisation (async, document upload) |
| F4 | Tool use: calendar, ticketing (Jira), HR system (Workday) |
| F5 | Multi-tenant isolation (department-level) |
| F6 | Admin controls: enable/disable capabilities per department |
| F7 | Full audit log of all interactions |
| F8 | PII detection and redaction (inputs and outputs) |
| F9 | Prompt injection defence |

### Non-functional requirements

| Metric | Target |
|---|---|
| Peak QPS | 800 |
| Peak concurrent sessions | 5,000 |
| Response latency p50 | < 1.5 s |
| Response latency p99 | < 5 s |
| Availability | 99.9% (< 8.7 hr/year) |
| Document corpus | 20 M documents, 500 GB raw |
| Data residency | On-premises only |
| Compliance | SOC 2 Type II |

---

## 3. Architecture Overview

```
                      ┌───────────────────────────────────────────┐
                      │         Corporate Network (VPC)           │
                      │                                           │
  Employee ──HTTPS──► │  API Gateway (Kong)                       │
                      │  OIDC Auth · Rate limit (100 req/min/user)│
                      │  WAF · TLS termination                    │
                      └──┬──────────────┬───────────────┬─────────┘
                         │              │               │
                  ┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
                  │Chat Service │ │Ingest Svc  │ │Admin Svc   │
                  │(FastAPI)    │ │(FastAPI)   │ │(FastAPI)   │
                  └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
                         │              │               │
                  ┌──────▼──────┐ ┌─────▼──────┐       │
                  │Orchestrator │ │ Chunker +  │ ┌─────▼──────┐
                  │(LangGraph)  │ │ Embedder   │ │Feature Flag│
                  └──────┬──────┘ └─────┬──────┘ │ Store (PG) │
                         │              │         └────────────┘
              ┌──────────┼────────┐     │
       ┌──────▼──┐ ┌─────▼──┐ ┌──▼──┐  │
       │ Retriever│ │PII Grd │ │Tool │  │
       │  Tool   │ │Guard   │ │Exec │  │
       └──────┬──┘ └────────┘ └─────┘  │
              │                         │
       ┌──────▼──────────────────┐ ┌────▼──────────────┐
       │   Qdrant (Vector DB)   │ │ Postgres (meta,   │
       │   HNSW, namespaced     │ │  sessions, audit) │
       └────────────────────────┘ └───────────────────┘
              │
       ┌──────▼──────────────┐
       │  vLLM Inference     │
       │  (self-hosted GPU)  │
       │  DeepSeek-Coder-33B │
       │  + Mistral-7B       │
       └─────────────────────┘

Shared: Redis (session cache, semantic cache) · S3-compatible (documents, meeting recordings)
Observability: OpenTelemetry → Tempo · Prometheus → Grafana · Loki (logs)
```

---

## 4. Component Deep-Dives

### 4.1 Chat Service

FastAPI with `async def` handlers. Accepts user query, retrieves session context from Redis (last 20 turns, TTL 24h), invokes the LangGraph orchestrator, and streams the response to the client via SSE. Session context is compressed with zlib before Redis storage to reduce memory footprint (~4 KB compressed per session vs. ~80 KB uncompressed for 20 turns).

Technology choice: FastAPI for native async + SSE support; Python because the LangGraph orchestrator is Python-native.

### 4.2 LangGraph Orchestrator

Typed state (`AgentState` as Pydantic model) enforces that every node returns well-typed data — prevents silent state corruption in multi-step chains. Nodes: `query_analyser`, `retriever`, `pii_guardrail`, `tool_executor` (parallel), `synthesiser`. Tool calls that are independent (e.g., retrieve document + fetch calendar) are executed in parallel via LangGraph's conditional branching.

Why LangGraph vs. a plain loop: the graph structure is inspectable and traceable; each node can be unit-tested in isolation; the typed state makes production debugging deterministic.

### 4.3 Retrieval Pipeline

Two-stage: (1) dense ANN search in Qdrant (top-20, per-department namespace), (2) cross-encoder reranker (BGE-reranker-large, CPU inference) → top-5 chunks. Chunk size: 512 tokens, 50-token overlap. Re-indexing: incremental on document upload events; full re-index weekly for deleted documents.

Qdrant HNSW parameters: `m=16`, `ef_construction=100`. p99 ANN search at 20M vectors: ~12 ms measured. Namespace per department enforces tenant isolation at the index level — no query filter to forget.

### 4.4 vLLM Inference Service

Self-hosted on 4× A100 80GB (2 nodes, 2 GPUs each). Primary model: Mistral-7B-Instruct for general Q&A (low latency, lower cost); DeepSeek-Coder-33B for code tasks. Tensor parallelism (2 GPUs) for the 33B model. Continuous batching (vLLM default) maximises GPU utilisation.

Token throughput target: 4,000 tokens/s aggregate → at 700 avg output tokens/query and 800 QPS peak: 560K tokens/s needed → 4 nodes of 4× A100 would be required at sustained peak. Day-1 capacity (2 nodes) handles 400 QPS; scale horizontally on demand.

### 4.5 PII Guardrail

Runs as a stateless FastAPI microservice on CPU. Uses presidio (Microsoft) for NER-based PII detection: redacts names, emails, phone numbers, national IDs from both inputs (before LLM) and outputs (before response delivery). p99 latency: < 30 ms for 512-token chunk. Does not see model weights — purely text in/text out.

### 4.6 Audit Log Service

Immutable append-only event log in Postgres (Timescale partition by month). Schema:

```sql
CREATE TABLE audit_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_time  TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id     UUID NOT NULL,
    tenant_id   UUID NOT NULL,
    session_id  UUID NOT NULL,
    event_type  VARCHAR(50) NOT NULL,   -- 'query', 'tool_call', 'response'
    payload     JSONB NOT NULL,
    pii_redacted BOOLEAN NOT NULL DEFAULT true
);
```

Retention: 7 years (SOC 2). Archived to S3-compatible cold storage after 1 year. Partition pruning makes recent queries fast; old data is accessible but slower.

---

## 5. Data Model

```
users          (id, email, tenant_id, role, created_at)
tenants        (id, name, feature_flags JSONB, created_at)
sessions       (id, user_id, tenant_id, created_at, expires_at, message_count)
documents      (id, tenant_id, s3_key, title, ingested_at, chunk_count, status)
feature_flags  (tenant_id, capability VARCHAR, enabled BOOLEAN, updated_at)
audit_events   (id, event_time, user_id, tenant_id, session_id, event_type, payload, pii_redacted)
```

Key indexes: `sessions(user_id, expires_at)`, `documents(tenant_id, status)`, `audit_events(tenant_id, event_time)`.

---

## 6. Capacity Planning

**QPS budget**:
- 30,000 DAU × 16 queries/day avg = 480K queries/day = 5.6 queries/s avg
- Peak factor 10×: 56 QPS baseline → 800 QPS peak (burst within a 15-min window)
- At 800 QPS × 700 tokens avg output × 1 GPU generating ~3K tokens/s: need 800×700/3000 ≈ 187 parallel streams → 4× A100 at ~75% utilisation at peak

**Storage**:
- 20M docs × 25 KB avg = 500 GB raw (S3)
- 20M docs × 4 chunks × 1,536 float32 dimensions × 4 bytes = 491 GB vector data (Qdrant, with HNSW index ~2× overhead = ~1 TB)
- Audit log: 30K DAU × 20 events × 500 bytes = 300 MB/day = 105 GB/year

**Caching**:
- Redis session cache: 5,000 active sessions × 4 KB compressed = 20 MB — trivial
- Semantic cache: 200K query embeddings cached (cosine ≥ 0.97) = ~150 MB — trivial; expected 20% cache hit rate → 160 QPS avoided

---

## 7. Observability & Eval

**Metrics (Prometheus → Grafana)**:
- `eaap_query_latency_seconds` (histogram, p50/p95/p99, by tenant)
- `eaap_llm_tokens_total` (counter, by model, by tenant)
- `eaap_retrieval_chunks_returned` (histogram)
- `eaap_pii_detections_total` (counter, by type)
- `eaap_cache_hit_ratio` (gauge)

**Traces (OpenTelemetry → Tempo)**:
- Span per LangGraph node: retriever, reranker, guardrail, synthesiser
- LLM call span: model, prompt tokens, completion tokens, latency

**Eval harness (RAGAS, nightly)**:
- 500-question golden test set per tenant (human-annotated)
- Metrics: faithfulness, answer relevance, context recall, context precision
- Alert threshold: faithfulness < 0.85 triggers Slack alert + auto-creates Jira ticket
- Online monitoring: sample 1% of live queries for automated eval

---

## 8. Security & Compliance

**Authentication**: OIDC via corporate IdP (Okta or Azure AD). JWT validated at API Gateway — no service trusts unauthenticated requests.

**Tenant isolation**: Qdrant namespace per department. Postgres RLS (row-level security) enforced on `tenant_id` column for all tables. No cross-tenant query possible even with a SQL bug — RLS enforced at DB engine level.

**Prompt injection defence**: Input guardrail strips system-prompt injection patterns (`IGNORE PREVIOUS INSTRUCTIONS`, role-switching tokens). Output guardrail checks response does not contain injected instructions. LangSmith tracing enables post-hoc investigation.

**PII**: Presidio redaction on both input and output paths. Redaction log stored separately from content. Redacted content never enters audit log.

**Data residency**: vLLM and all storage on-premises. No outbound calls to external AI APIs. Verified via egress firewall rules (deny all to *.anthropic.com etc. from inference cluster).

**SOC 2 controls**: Audit log immutable (Timescale append-only partition), retained 7 years, access-controlled to security team. Change management tracked via Jira. Vulnerability scanning in CI (Trivy for containers).

---

## 9. Cost Model

| Component | Monthly cost |
|---|---|
| vLLM GPU cluster (4× A100 80GB, 2 nodes, reserved) | $18,000 |
| Qdrant (self-hosted, 3-node cluster, NVMe SSD) | $3,600 |
| Postgres (2-node HA, Timescale) | $2,400 |
| Redis cluster (3 nodes) | $900 |
| API Gateway + Kong | $1,200 |
| Object storage (S3-compatible, 500 GB + audit) | $500 |
| Observability stack (Prometheus, Grafana, Tempo, Loki) | $800 |
| Network, security tools, misc infra | $2,000 |
| Engineering ops (SRE 0.5 FTE equivalent) | $15,000 |
| **Total** | **~$44,400/month** |

Note: capital cost of GPU hardware (~$320K) amortised at 3 years ≈ $8,900/month additional. Total with amortisation: ~$53,300/month for 50,000 users = $1.07/user/month.

---

## 10. Trade-offs & Alternatives Considered

### Trade-off 1: Self-hosted vLLM vs. cloud LLM API

**Chosen**: Self-hosted vLLM with open-weight models.  
**Alternative**: Anthropic API (claude-sonnet-5-5) via VPN endpoint.  
**Why chosen**: data-residency requirement eliminates any cloud API. Even private endpoints don't guarantee data never reaches Anthropic infrastructure. Self-hosted adds GPU capex and ops overhead but is the only compliant option.  
**What we give up**: Anthropic's frontier model quality; automatic model updates.

### Trade-off 2: LangGraph typed state vs. plain asyncio loop

**Chosen**: LangGraph with `AgentState` Pydantic model.  
**Alternative**: Custom asyncio orchestration loop with dict state.  
**Why chosen**: typed state makes production bugs (missing keys, wrong types) fail loudly at node boundaries rather than silently corrupting downstream nodes. The graph structure also makes tracing individual steps trivial.  
**What we give up**: ~20% implementation overhead; LangGraph version coupling.

### Trade-off 3: Per-department Qdrant namespace vs. per-department Qdrant collection

**Chosen**: Namespace per department (single collection, multiple namespaces).  
**Alternative**: Separate Qdrant collection per department.  
**Why chosen**: 50 departments × separate collection = 50× metadata overhead, 50× HNSW index memory. Namespaces share a single index with per-namespace filtering — same isolation guarantee at 50× lower resource cost.  
**What we give up**: inability to tune HNSW parameters per department (e.g., legal may want higher recall than engineering).

---

## 11. Roadmap

### Phase 1 — Foundation (Months 1–3)
- Core RAG Q&A for 5 pilot departments
- Basic auth + OIDC
- Audit logging
- Single-model inference (Mistral-7B)
- Success criterion: 95% user satisfaction in pilot; p99 < 5 s

### Phase 2 — Full Platform (Months 4–6)
- All departments onboarded
- Code generation and meeting summarisation
- Tool use (calendar, Jira)
- PII guardrail
- Admin controls
- Multi-model routing (Mistral-7B for general, DeepSeek-Coder-33B for code)
- Success criterion: 800 QPS capacity validated; SOC 2 audit evidence collected

### Phase 3 — Intelligence & Scale (Months 7–12)
- Fine-tuning on internal data (company-specific terminology, coding style)
- Semantic cache (target 25% hit rate)
- RAGAS continuous eval with alert thresholds
- Multi-modal: document image understanding (OCR + vision model)
- Self-service department admin UI
- Success criterion: faithfulness ≥ 0.90; cost/user ≤ $1.00/month
