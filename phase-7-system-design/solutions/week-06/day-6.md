# Week 06 · Day 6 Solutions — Saturday Deep Practice: Enterprise AI Platform Design

---

## Exercise 1: Complete Enterprise AI Platform Capacity Estimation

### Given Parameters

- 10,000 active users
- 500 RAG queries/min
- 50 agent runs/min
- 2,000 LLM direct calls/min

---

### Step 1: Convert to Per-Second Rates

```
RAG queries:       500 / 60  =   8.33 QPS
Agent runs:         50 / 60  =   0.83 concurrent workflows/sec
LLM direct calls: 2,000 / 60 =  33.33 QPS
Total LLM QPS:    8.33 + 33.33 = 41.67 QPS (agents handled separately below)
```

---

### Step 2: Token Throughput Calculation

**RAG queries (claude-sonnet-5-5):**

```
Per-query token breakdown:
  System prompt:         500 tokens
  Retrieved context:   1,500 tokens  (3 chunks × 500 tokens)
  User query:             50 tokens
  ─────────────────────────────────
  Total input:         2,050 tokens/query
  Output:                300 tokens/query

At 8.33 RAG QPS:
  Input tokens/sec:   8.33 × 2,050 = 17,077 tokens/sec
  Output tokens/sec:  8.33 × 300   =  2,499 tokens/sec
```

**LLM direct calls (mixed model, assume avg 1,000 in / 500 out):**

```
At 33.33 QPS:
  Input tokens/sec:  33.33 × 1,000 = 33,330 tokens/sec
  Output tokens/sec: 33.33 × 500   = 16,665 tokens/sec
```

**Agent runs (claude-opus-5-5 planning + claude-sonnet-5-5 execution):**

```
Per-workflow token breakdown:
  Planning (opus):   500 in + 200 out = 700 tokens
  Tool calls × 4 (sonnet): 4 × (800 in + 400 out) = 4,800 tokens
  ─────────────────────────────────────────────────────────────
  Total per workflow: 5,500 tokens (5,300 in, 1,800 out over full run)

At 0.83 workflows/sec:
  Input tokens/sec:  0.83 × 5,300  = 4,399 tokens/sec
  Output tokens/sec: 0.83 × 1,800  = 1,494 tokens/sec
```

**Grand total token throughput:**

```
                 Input (tokens/sec)   Output (tokens/sec)
RAG:                  17,077               2,499
LLM direct:           33,330              16,665
Agent:                 4,399               1,494
─────────────────────────────────────────────────
TOTAL:                54,806              20,658
```

---

### Step 3: GPU Count (Self-Hosted, A100 80GB at 3,000 tokens/sec)

> Note: at this scale, Anthropic API is almost always preferred over self-hosting. This calculation illustrates the GPU math for interview contexts or sensitive-data tenants.

**Assumption**: 3,000 output tokens/sec per A100 80GB (FP16, continuous batching, 70B param model with tensor parallelism).

```
Required output throughput:  20,658 tokens/sec
Throughput per A100:          3,000 tokens/sec
─────────────────────────────────────────────
Raw GPU count:  20,658 / 3,000 = 6.89 GPUs
Add 2× headroom for spikes and serving overhead:
  6.89 × 2 = 13.77 → round up to 14 A100s

For embedding GPU (query embedding at 8.33 QPS, batch 64, ~10ms):
  8.33 / 64 × 0.010s = 0.0013 GPU-seconds/second → < 1 embedding GPU

Total GPU requirement: 14 A100s (LLM) + 1 A100 (embedding) = 15 A100s
```

**Memory check (70B model on A100 80GB):**

```
Model weights (FP16): 70B × 2 bytes = 140 GB → needs 2× A100s per replica
With tensor parallelism (TP=2): each node = 2 GPUs, holds 70GB of weights
KV cache at batch 32, seq 4096: 2 × 80 layers × 128 dim × 4096 × 32 × 2B = ~5.4 GB
Total per-node: 70 + 5.4 = 75.4 GB → fits in 80GB A100 per GPU in TP=2 config
```

---

### Step 4: Vector Store Storage (1B Chunks at 1536 Dims)

```
Embedding vectors:
  1,000,000,000 chunks × 1,536 dimensions × 4 bytes (float32)
  = 1B × 6,144 bytes
  = 6,144,000,000,000 bytes
  = 6.14 TB (raw vectors)

Metadata overhead (30%):
  6.14 TB × 1.30 = 7.98 TB

HNSW index overhead (approximately 15% of raw):
  6.14 × 0.15 = 0.92 TB

Total unreplicated: 6.14 + 1.84 (metadata) + 0.92 (index) ≈ 8.9 TB

With 3× replication:
  8.9 × 3 = 26.7 TB
  → Round to ~27 TB replicated vector store

NVMe SSD sizing: at $200/TB/month NVMe-backed cloud storage:
  27 TB × $200 = $5,400/month storage cost
```

---

### Step 5: Redis Cache Sizing

**Semantic cache entries:**

```
Assumption: cache top 100,000 query→response pairs (hot working set)
Per entry:
  Key: 1,536-dim embedding = 6,144 bytes ≈ 6 KB
  Value: avg response 300 tokens × 4 bytes = 1,200 bytes ≈ 1.2 KB
  TTL metadata: ~0.1 KB
  ─────────────────────────────────────────────────
  Total per entry: ~7.3 KB

100,000 entries × 7.3 KB = 730 MB

Cost counter state (per tenant, per minute):
  10,000 users × 200 tenants counters × 64 bytes = 128 MB

Session / JWT validation cache:
  10,000 active users × 2 KB per session = 20 MB

Total Redis memory: ~880 MB → provision 2 GB Redis for headroom
Recommendation: Redis Cluster with 2 replicas, r7g.large (13 GB each) for safety
Monthly cost: ~$300/month (managed ElastiCache r7g.large × 3 nodes)
```

---

### Step 6: Network Bandwidth

```
Inbound (requests):
  RAG:        8.33 QPS × 2 KB avg payload  = 16.7 KB/s
  LLM direct: 33.33 QPS × 2 KB avg payload = 66.7 KB/s
  Agent:      0.83 QPS × 5 KB avg payload  =  4.2 KB/s
  ─────────────────────────────────────────────────────
  Total inbound: ~88 KB/s = 0.7 Mbps  (trivial)

Outbound (responses + streaming):
  RAG output:      8.33 × 300 tokens × 4 bytes  =  10.0 KB/s
  LLM direct out: 33.33 × 500 tokens × 4 bytes  =  66.7 KB/s
  Agent out:       0.83 × 1,800 tokens × 4 bytes =   6.0 KB/s
  ─────────────────────────────────────────────────────────────
  Total outbound: ~83 KB/s = 0.66 Mbps (also trivial at this scale)

Peak burst (3× average):
  ~2 Mbps sustained — a single 10 Gbps NIC handles this with 5000× headroom.

Egress cost (cloud):
  Assume 0.66 Mbps average × 86,400 s/day × 30 days = 171 GB/month outbound
  At $0.09/GB egress: $15.39/month — negligible.
```

---

### Summary Table

| Component | Estimate |
|---|---|
| Peak input tokens/sec | 54,806 |
| Peak output tokens/sec | 20,658 |
| A100 GPUs (self-hosted LLM) | 14 + 1 embedding = 15 |
| Vector store (1B chunks, 1536 dims) | ~27 TB replicated |
| Redis memory | 2 GB (semantic cache + counters) |
| Network bandwidth | ~2 Mbps peak (negligible) |

---

## Exercise 2: Five Architecture Decision Records

---

### ADR-001: Vector Store Technology Choice

**Title**: Vector Database Technology Selection for Multi-Tenant Enterprise RAG

**Status**: Accepted

**Context**:
The platform serves 200 enterprise tenants with strict data isolation requirements (SOC2 Type II). The vector store must support 1B+ vectors at 1536 dimensions, provide namespace/collection-level isolation, handle ~42 QPS ANN search at P95 < 50ms, and be deployable in environments with data sovereignty constraints that may preclude SaaS-only solutions.

**Decision**:
Use Qdrant (self-hosted, Apache 2.0) as the vector store, with one Qdrant collection namespace per tenant.

**Rationale**:
Qdrant provides native namespace (collection) isolation enforced at the database engine level — not merely at the application query layer. A logic bug in application code cannot cause cross-tenant data reads without an explicit namespace override. Qdrant's Rust implementation delivers high throughput and low latency under concurrent load. The HNSW index provides sub-50ms ANN search at 100M+ vectors per namespace. It is self-hostable, satisfying data sovereignty requirements. Apache 2.0 license avoids vendor lock-in.

**Alternatives considered**:
- **Pinecone (managed SaaS)**: fast, namespace support, zero ops. Rejected because it requires data to leave the enterprise perimeter — fails SOC2 data residency requirements for several target enterprise verticals. Also introduces vendor lock-in with no self-host path.
- **Weaviate (self-hosted or managed)**: strong GraphQL API, built-in multi-tenancy class. Rejected because hybrid BM25+dense search configuration is more complex, and at 1B vectors the Java/JVM memory model requires more careful tuning than Qdrant's Rust runtime.
- **pgvector (PostgreSQL extension)**: familiar SQL interface, transactional guarantees, easy ops for teams already running Postgres. Rejected because HNSW performance degrades materially beyond 10M vectors per index. At 1B vectors across 200 namespaces this becomes a performance risk.
- **Chroma**: excellent for prototyping and development. Rejected because it is not production-grade at 100M+ vectors — no built-in replication, weak multi-tenancy.

**Consequences**:
- Positive: data sovereignty satisfied; isolation enforced at DB layer; high throughput; open source.
- Negative: must operate Qdrant cluster (backup, monitoring, upgrades). No managed SLA. Requires NVMe-backed storage (~27 TB replicated) — infrastructure cost ~$5,400/month.

---

### ADR-002: LLM Provider Strategy

**Title**: LLM Provider Selection and Multi-Tier Model Routing Strategy

**Status**: Accepted

**Context**:
The platform requires LLM capabilities across three tiers of task complexity: low-cost classification and eval scoring, mid-tier RAG generation and agent execution, and high-quality complex reasoning for agent planning and long-context analysis. Enterprise customers require predictable API behavior, extended context windows (200K+ tokens), and a provider with demonstrable safety and compliance properties.

**Decision**:
Use Anthropic API as the sole LLM provider, with a three-tier model routing strategy:
- `claude-haiku-5-5`: eval/judge calls, classification, routing decisions, short summaries
- `claude-sonnet-5-5`: RAG generation (default), agent tool-call execution
- `claude-opus-5-5`: agent planning, long-context document synthesis, complex multi-hop reasoning

**Rationale**:
Anthropic's Constitutional AI training approach produces models with predictable refusal behavior and low hallucination rates on enterprise knowledge base tasks — critical for trust in RAG outputs. The 200K context window handles large document queries without chunking workarounds. The three-tier pricing model ($0.10/$0.50 → $2/$10 → $4/$20 per 1M tokens) enables cost optimization by routing tasks to the cheapest capable model. Anthropic's prompt caching can reduce input token costs 50–90% for static system prompts, which is significant at this scale.

**Alternatives considered**:
- **OpenAI GPT-4o / GPT-4 Turbo**: strong benchmark performance, large ecosystem. Rejected primarily because several target enterprise customers have data processing agreements that exclude OpenAI. Secondary concern: higher per-token cost at our volume.
- **Google Vertex AI (Gemini)**: strong multimodal, 1M context window. Rejected because it tightly couples the platform to GCP — unacceptable for customers on Azure or on-prem. Vertex AI API behavior for long-context tasks has shown higher variance in our evals.
- **Self-hosted open-weight models (Llama 3, Mistral)**: full data sovereignty, lower marginal cost at scale. Rejected for this phase because: (a) ops burden of GPU fleet at 15+ A100s is significant; (b) quality gap on complex reasoning tasks is 6–12 months behind proprietary; (c) break-even with Anthropic API cost only materializes at ~$500K/month in API spend — below that, self-hosting is more expensive when amortizing GPU capex and eng ops.

**Consequences**:
- Positive: zero GPU ops; access to state-of-art models; prompt caching available; predictable API behavior.
- Negative: vendor dependency on Anthropic API availability — mitigate with circuit breaker + cached response fallback. Data sent to Anthropic — acceptable under Anthropic's enterprise data processing agreement but must be disclosed to tenants. Cost at scale is variable and can spike with traffic.

---

### ADR-003: Agent State Store

**Title**: Agent Workflow State Persistence Mechanism

**Status**: Accepted

**Context**:
Agent workflows can run 30 seconds to several minutes. They must survive LLM API rate limit errors, transient network failures, and container restarts mid-execution. The state store must support: checkpoint-and-resume semantics, human-in-the-loop interrupts (workflow pauses waiting for approval), concurrent state reads during streaming, and audit trail requirements (immutable history of every agent step for compliance).

**Decision**:
Use PostgreSQL as the durable agent state store via LangGraph's `PostgresSaver` checkpointer, with Redis for hot-path state caching of in-flight workflows.

**Rationale**:
PostgreSQL provides ACID guarantees and a queryable history log — when a compliance audit requests "show me every step agent workflow W-42 took on 2026-11-03," a SQL query returns it. LangGraph's built-in checkpointer interface abstracts the persistence layer, making it swappable. For in-flight workflows, Redis caches the current state to avoid hitting Postgres on every LLM streaming token. Redis TTL is set to 2× max workflow duration; on expiry, state is already durably in Postgres.

**Alternatives considered**:
- **Redis-only (in-memory)**: lower latency, simpler ops. Rejected because Redis is not durable by default — an OOM event or failover loses in-flight workflow state. Acceptable for a cache layer but not as the authoritative state store. Also, Redis persistence (RDB/AOF) adds complexity without the querying capabilities of Postgres.
- **Event-sourced (Kafka + event store)**: maximum auditability — every state transition is an immutable event. Rejected because the operational complexity (Kafka + event store + projection layer) is disproportionate to the requirements. Kafka message retention already handles replay; a full event-sourced architecture adds 2–3 additional components with no clear benefit over Postgres at our scale.
- **DynamoDB**: fully managed, horizontal scale. Rejected because it is AWS-specific, and several enterprise tenants require Azure or on-prem deployments. Also, DynamoDB's eventual consistency for reads requires careful version management to avoid serving stale workflow state.

**Consequences**:
- Positive: ACID guarantees; queryable audit log; LangGraph checkpointer abstraction makes it swappable; human-in-the-loop patterns supported natively.
- Negative: Postgres requires ops (replication, backups, vacuum). Write amplification if checkpointing on every LLM token — mitigate by checkpointing only at node boundaries, not within streaming.

---

### ADR-004: Message Queue for Ingestion Pipeline

**Title**: Message Queue Technology for Asynchronous Document Ingestion

**Status**: Accepted

**Context**:
Document ingestion is bursty: tenants batch-upload large document sets during business hours. Each ingestion job must complete reliably even if workers crash mid-processing (durable delivery). At peak (100 tenants uploading simultaneously), the system must process ~14,000 chunks/second through the ingestion pipeline. Failed jobs must be retryable without data loss, and the system must support replaying ingestion from a specific point for reprocessing (e.g., after a chunking algorithm change).

**Decision**:
Use Apache Kafka (managed via Confluent Cloud or AWS MSK) as the ingestion message queue, with Kafka consumer groups for parallel worker scaling.

**Rationale**:
Kafka's log-structured storage provides durable, ordered, replayable message delivery — critical for the reprocessing use case. Consumer groups allow independent horizontal scaling of ingestion workers by queue depth. At 14,000 chunks/second peak, Kafka's write throughput (1M+ messages/second on a 3-broker cluster) has 70× headroom. Retention policy (e.g., 7 days) allows replaying failed ingestion batches without re-uploading raw documents.

**Alternatives considered**:
- **AWS SQS**: fully managed, simple ops, natively integrated with AWS services. Rejected because SQS lacks native message replay — there is no "re-read from offset N" operation. DLQ replay is manual and lossy. Also SQS is AWS-specific, creating cloud lock-in.
- **RabbitMQ**: mature, excellent Python client (pika/aio-pika), low latency for small workloads. Rejected because RabbitMQ is not designed for replay/retention semantics. Messages are consumed and gone. Horizontal scaling beyond a handful of nodes requires careful federation configuration.
- **Redis Streams**: low latency, simpler ops than Kafka, native Redis ecosystem. Rejected because durability guarantees are weaker than Kafka — Redis AOF persistence adds latency and the failure modes are less well understood for high-throughput append workloads. Appropriate below 10M messages/day; our peak bursts exceed this.

**Consequences**:
- Positive: durable delivery; replay capability; consumer group auto-scaling; 70× throughput headroom.
- Negative: Kafka cluster adds ops complexity. Mitigate: use managed Kafka (Confluent Cloud or MSK) to reduce ops burden. Development and CI environments use Redpanda (Kafka-compatible, single binary, much easier local setup).

---

### ADR-005: Multi-Tenancy Isolation Level

**Title**: Vector Store and Data Isolation Level for Multi-Tenant Platform

**Status**: Accepted

**Context**:
The platform serves 200 enterprise tenants at launch, projected to 1,000 in Year 2. Tenants include financial services and healthcare companies subject to SOC2 Type II and GDPR. The primary concern is: a bug in application code must not allow Tenant A to read Tenant B's documents. Secondary concern: cost — per-tenant infrastructure costs must not grow linearly with tenant count.

**Decision**:
Implement namespace-per-tenant isolation in Qdrant (one collection per tenant), combined with PostgreSQL row-level security (RLS) for all relational metadata. The tenant namespace is enforced at query time by the API Gateway extracting the tenant ID from the validated JWT and injecting it into every downstream query.

**Rationale**:
Qdrant namespace isolation means the DB engine itself enforces the tenant boundary — a query to namespace `tenant-A` physically cannot return vectors from namespace `tenant-B`. This is stronger than filter-only isolation (where a missing `WHERE tenant_id = X` clause would return all tenants' data). RLS in PostgreSQL adds a second enforcement layer for structured metadata. The cost of this approach is O(tenants) namespaces in a shared Qdrant cluster — at 1,000 tenants with 100GB corpus each, a single cluster handles this comfortably.

**Alternatives considered**:
- **Cluster-per-tenant**: strongest isolation — a full Qdrant cluster per tenant means a compromise of one tenant cannot affect others even at the infrastructure level. Rejected because at 200 tenants, provisioning 200 Qdrant clusters is cost-prohibitive (200× infra cost). Reserved for Ultra tier customers at >$100K/month contract value where the isolation premium is commercially justified.
- **Filter-only isolation (shared namespace, metadata `tenant_id` filter)**: cheapest option — single Qdrant collection, every query adds `must: [{"key": "tenant_id", "match": {"value": "T-42"}}]`. Rejected because a missing or incorrect filter in application code produces a data breach. Does not meet SOC2 Type II isolation standard.
- **Schema-per-tenant (PostgreSQL pgvector)**: familiar PostgreSQL ops, strong schema-level isolation. Rejected because pgvector HNSW performance at 100M vectors per tenant namespace degrades unacceptably. pgvector is appropriate for tenants with <10M vectors.

**Consequences**:
- Positive: data breach requires both a DB-level exploit (bypassing namespace) AND an application-level bug — defense in depth. Individual namespace HNSW indexes are smaller → faster ANN search per tenant.
- Negative: at 10,000+ tenants, per-namespace HNSW index management (creation, rebalancing) becomes operationally complex. Revisit with a sharding strategy (e.g., bin-pack tenants into groups of 50 per shard) if tenant count reaches that scale.

---

## Exercise 3: Monthly Cost Model Worked Example

### Platform Parameters

- 500 RAG QPS peak, 30% average load = 150 QPS average
- 50 agent runs/min peak, 30% average = 15/min = 0.25/sec average
- 2,000 LLM direct calls/min peak, 30% average = 600/min = 10 QPS average
- 30-day month

---

### Step 1: Monthly Request Volume

```
Monthly seconds: 30 × 86,400 = 2,592,000 s

RAG queries:       150 QPS   × 2,592,000 = 388,800,000 queries/month
Agent workflows:   0.25 /s   × 2,592,000 =     648,000 workflows/month
LLM direct calls:  10 QPS    × 2,592,000 =  25,920,000 calls/month
```

---

### Step 2: Anthropic API Cost

**Model mix assumptions:**
- RAG: 60% haiku-5-5, 40% sonnet-5-5
- Agent planning: 100% opus-5-5 (1 planning call per workflow)
- Agent execution: 100% sonnet-5-5 (4 tool calls per workflow)
- LLM direct calls: 50% haiku-5-5, 30% sonnet-5-5, 20% opus-5-5

**Pricing (per 1M tokens):**

| Model | Input | Output |
|---|---|---|
| claude-haiku-5-5 | $0.10 | $0.50 |
| claude-sonnet-5-5 | $2.00 | $10.00 |
| claude-opus-5-5 | $4.00 | $20.00 |

---

**RAG query cost:**

```
Per query token profile: 2,050 in, 300 out

Cost per haiku RAG query:
  (2,050 / 1M) × $0.10 + (300 / 1M) × $0.50
  = $0.000205 + $0.000150 = $0.000355

Cost per sonnet RAG query:
  (2,050 / 1M) × $2.00 + (300 / 1M) × $10.00
  = $0.00410 + $0.00300 = $0.00710

Blended RAG cost per query:
  0.60 × $0.000355 + 0.40 × $0.00710
  = $0.000213 + $0.002840 = $0.003053 per query

Monthly RAG cost:
  388,800,000 × $0.003053 = $1,186,994 ≈ $1.19M/month
```

**Agent workflow cost:**

```
Per workflow:
  Planning (opus): (500/1M) × $4.00 + (200/1M) × $20.00
    = $0.002000 + $0.004000 = $0.006000
  4 tool calls (sonnet): 4 × [(800/1M) × $2.00 + (400/1M) × $10.00]
    = 4 × [$0.001600 + $0.004000]
    = 4 × $0.005600 = $0.022400
  Total per workflow: $0.006000 + $0.022400 = $0.028400

Monthly agent cost:
  648,000 × $0.028400 = $18,403 ≈ $18.4K/month
```

**LLM direct calls cost:**

```
Assume per call: 1,000 in tokens, 500 out tokens

Haiku (50%): (1,000/1M) × $0.10 + (500/1M) × $0.50 = $0.000100 + $0.000250 = $0.000350
Sonnet (30%): (1,000/1M) × $2.00 + (500/1M) × $10.00 = $0.002000 + $0.005000 = $0.007000
Opus (20%): (1,000/1M) × $4.00 + (500/1M) × $20.00 = $0.004000 + $0.010000 = $0.014000

Blended: 0.50 × $0.000350 + 0.30 × $0.007000 + 0.20 × $0.014000
        = $0.000175 + $0.002100 + $0.002800 = $0.005075 per call

Monthly LLM direct cost:
  25,920,000 × $0.005075 = $131,544 ≈ $131.5K/month
```

**Total monthly Anthropic API cost:**

```
RAG:       $1,187,000
Agents:       $18,400
LLM direct:  $131,500
─────────────────────
TOTAL:     $1,336,900 ≈ $1.34M/month
```

---

### Step 3: GPU Rental Cost (Self-Hosted Embedding)

```
Embedding GPUs: 1× A100 80GB for online query embedding + 3× for ingestion bursts
  = 4 A100s at $3.50/hr (spot/reserved mix on GCP/AWS)
  = 4 × 24 × 30 × $3.50 = $10,080/month

LLM self-hosting (NOT used — using Anthropic API): $0
(Breakeven for self-hosting LLM is ~$500K/month API spend; we're at $1.34M, so worth modeling)

If self-hosting LLM were chosen instead:
  15 A100s × $3.50/hr × 730 hr/month = $38,325/month
  Saving vs. API: $1,340,000 - $38,325 = $1,301,675/month (but requires GPU ops team)
```

---

### Step 4: Storage Costs

```
Vector store (Qdrant on NVMe SSDs):
  27 TB × $0.20/GB/month (NVMe SSD pricing)
  = 27,000 GB × $0.20 = $5,400/month

Object storage (raw documents + extracted text):
  20 TB × $0.023/GB/month (S3 Standard)
  = 20,000 GB × $0.023 = $460/month

Query log storage (90-day retention):
  Ingestion: peak 34.72 QPS × 86,400 × 30 × 4 KB = 360 GB/month
  At $0.023/GB: $8.28/month (trivial)
  Running total (90 days): 1,080 GB × $0.023 = $24.84/month

Total storage: $5,400 + $460 + $25 ≈ $5,885/month
```

---

### Step 5: Redis Cost

```
Redis Cluster (ElastiCache r7g.large × 3 nodes, 2 GB RAM each):
  3 × $0.166/hr × 730 hr = $363/month

Redis backup storage (S3, < 1 GB): ~$0.02/month → negligible
Total Redis: ~$363/month
```

---

### Step 6: Network Egress

```
Outbound API responses: ~0.66 Mbps average
Monthly: 0.66 Mbps × 3.6 Mb/GB × 86,400 × 30 = ~7.6 GB/day × 30 = 228 GB/month
At $0.09/GB: $20.52/month → negligible
```

---

### Step 7: Compute (API + Ingestion Workers)

```
Application compute (ECS/GKE, stateless containers):
  API Gateway:          2× t3.xlarge = $0.166/hr × 2 × 730 = $242/month
  RAG service:          4× c5.2xlarge = $0.34/hr × 4 × 730 = $993/month
  Agent orchestration:  4× c5.2xlarge = $993/month
  Ingestion workers:    6× c5.xlarge (auto-scaled) = $0.17/hr × 6 × 730 = $744/month
  Total compute: ~$2,972/month

Managed Kafka (MSK, 3 brokers, m5.xlarge):
  3 × $0.238/hr × 730 = $521/month

PostgreSQL (RDS Postgres, db.r6g.xlarge, Multi-AZ):
  $0.48/hr × 730 × 2 (Multi-AZ) = $701/month

Observability stack (Prometheus + Grafana managed, or Datadog):
  ~$500/month (self-hosted) to $3,000/month (Datadog at this scale)
  Using self-hosted: ~$500/month
```

---

### Step 8: Total Monthly Cost Summary

| Category | Monthly Cost |
|---|---|
| Anthropic API (RAG + agents + direct) | $1,336,900 |
| GPU (embedding, 4× A100) | $10,080 |
| Vector store storage (27 TB NVMe) | $5,400 |
| Object storage (20 TB S3) | $460 |
| Redis (ElastiCache cluster) | $363 |
| Application compute (ECS/GKE) | $2,972 |
| Managed Kafka (MSK) | $521 |
| PostgreSQL RDS Multi-AZ | $701 |
| Observability (self-hosted) | $500 |
| Network egress | $21 |
| **TOTAL** | **$1,357,918 ≈ $1.36M/month** |

---

### Step 9: Per-Tenant and Per-Query Cost

```
Per-tenant monthly cost: $1,357,918 / 200 tenants = $6,790/month
Recommended pricing (30% margin): $8,800/month per tenant

Per RAG query: $1,357,918 / 388,800,000 = $0.00349 per query
  → Well within $0.05 budget target
  → Headroom ratio: $0.05 / $0.00349 = 14.3× — significant room for growth

Per agent workflow: ($1,357,918 × proportion) ÷ workflows
  Agent proportion of cost: ($18,400 + $131,500 × agent_fraction) ≈ small
  Primary cost driver is RAG (87% of API cost).
```

---

### Step 10: Cost Optimization Potential

Applying the four optimization levers simultaneously:

```
1. Prompt caching (80% hit rate, 90% cost reduction on cached prefix):
   Cached portion of input: ~500/2050 = 24% of input tokens
   Saving: 388.8M queries × 0.80 hit rate × (500/1M) × 0.90 × blended_input_price
   Approximate saving: ~$200K/month

2. Model tiering (increase haiku fraction from 60% to 80%):
   RAG cost reduction: replacing 20% sonnet with haiku on 388.8M queries
   Additional saving: ~$180K/month

3. Semantic cache (15% hit rate removes LLM call entirely):
   Saving: 388.8M × 0.15 × $0.003053 = $178,095/month

4. Context compression (35% reduction in retrieved context tokens):
   Reduces input from 2,050 to 2,050 - 0.35×1,500 = 1,525 tokens
   Saving: proportional to input token reduction × fraction not cached
   Approximate saving: ~$80K/month

Total optimized saving: ~$638K/month
Optimized monthly cost: $1,357,918 - $638,000 ≈ $720,000/month
Savings: 47% cost reduction
```
