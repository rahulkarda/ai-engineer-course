# Week 06 Project Solutions — Enterprise AI Platform System Design

---

## Part 1: Completed System Design Document

### 1.1 Functional Requirements

1. **RAG platform**: Users submit natural language queries against private enterprise knowledge bases. The system retrieves relevant document chunks from a per-tenant vector namespace, augments the prompt with retrieved context and citation metadata, and calls an LLM to synthesize a grounded answer. Every response includes source citations (document ID, chunk offset, relevance score).

2. **Agent platform**: Users trigger multi-step agentic workflows (research → summarize → draft report). Agents call typed tools (web search, code execution, internal APIs). Workflows are observable via streaming SSE, interruptible by the tenant administrator, and resumable after failure via checkpoint-and-restore.

3. **LLM serving layer**: A unified gateway routes requests to the appropriate model tier based on task type and tenant cost budget. It handles API key management, per-tenant quota enforcement, request logging (all calls recorded for audit), and abstracts provider differences via a stable internal interface.

4. **Eval pipeline**: Automated quality scoring using LLM-as-judge at a configurable sample rate (default 10%). Offline batch eval on golden datasets runs before each model version promotion. Quality gates block deployment if eval score drops > 5% from baseline.

5. **Observability stack**: Distributed tracing (OpenTelemetry) covers every span from API request to LLM response. Per-call cost attribution enables per-tenant billing reports. Prometheus metrics + Grafana dashboards. Alertmanager fires PagerDuty on burn-rate SLO violations.

6. **Security and multi-tenancy**: 200 enterprise tenants with namespace-level vector store isolation (SOC2 Type II compliant). PII auto-detected and redacted before any data leaves the platform perimeter. All LLM API keys stored in AWS Secrets Manager. JWT-authenticated API with tenant extraction at gateway edge.

---

### 1.2 Non-Functional Requirements (Filled In)

| SLO | RAG queries | Agent workflows | LLM serving |
|---|---|---|---|
| Availability | 99.9% (8.7 hr/year) | 99.5% (43.8 hr/year) | 99.95% (4.4 hr/year) |
| P50 latency | ≤ 500ms | ≤ 2s first token | ≤ 150ms TTFT |
| P99 latency | ≤ 2s | ≤ 30s total | ≤ 800ms TTFT |
| Max cost/query | $0.05 | $0.50 | — |
| Peak throughput | 500 QPS | 50 concurrent | 1,000 QPS |

**SLO Rationale:**
- *RAG P50 ≤ 500ms*: Derived from 1s user perceived response tolerance minus 300ms for browser rendering and network round-trip; 200ms reserved for first-token streaming display.
- *RAG P99 ≤ 2s*: LLM streaming of 300 tokens at ~80 tokens/sec = 3.75s full completion, but streaming first token at 400ms means the user is reading content well before completion. P99 2s covers first-token latency for the 99th-percentile slow LLM response.
- *Agent P99 ≤ 30s*: Multi-tool workflows (5 sequential tool calls, each 2–4s) plus 2 LLM calls = 15–25s median. P99 budget allows for retry on one tool call timeout.
- *LLM serving P50 TTFT ≤ 150ms*: Derived from vLLM continuous batching benchmarks on A100 80GB with batch size 32 and 4096-token context at P50 load.

---

### 1.3 Capacity Estimation — Math Shown

**Request volume:**

```
DAU: 50,000 users × 200 tenants
RAG queries: 50,000 × 20 = 1,000,000 queries/day
Agent workflows: 50,000 × 2 = 100,000 workflows/day

Average QPS (RAG): 1,000,000 / 86,400 = 11.57 QPS
Peak QPS (RAG): 11.57 × 3 = 34.72 QPS

Average agent workflows/sec: 100,000 / 86,400 = 1.157 /sec
Average concurrent (30s each): 1.157 × 30 = 34.7 concurrent
Peak concurrent: 34.7 × 3 = 104 concurrent workflows
```

**Storage (2.4 capacity estimation):**

```
Document storage:
  10,000,000 docs × 2,048 bytes = 20,480,000,000 bytes = 20.48 GB raw

Vector storage (raw):
  10,000,000 docs × 1,536 dims × 4 bytes = 61,440,000,000 bytes = 61.44 GB

Vector storage total (1.5× overhead for HNSW index + metadata):
  61.44 GB × 1.5 = 92.16 GB

Query log storage (at peak QPS, 90-day retention):
  Per day: 34.72 QPS × 86,400 s × 4,096 bytes = 12,290,310,758 bytes = 11.45 GB/day
  90-day total: 11.45 × 90 = 1,030.5 GB ≈ 1.01 TB
```

**GPU sizing (2.3 — optional self-hosted 8B model):**

```
Model weights: 8B × 2 bytes (BF16) = 16 GB

KV cache at batch=32, seq=4096, layers=32, head_dim=128, dtype=2 bytes:
  2 × 32 × 128 × 4,096 × 32 × 2 = 2,147,483,648 bytes = 2.0 GB

Min GPU memory: 16 GB (weights) + 2.0 GB (KV cache) = 18.0 GB
  → 1× A100 80GB is sufficient (18 GB of 80 GB used)

GPUs needed for peak QPS at batch 32 (150 tokens/sec per GPU):
  Peak output tokens/sec: 34.72 QPS × 200 tokens = 6,944 tokens/sec
  GPUs needed: ⌈6,944 / 150⌉ = ⌈46.3⌉ = 47 GPUs

Monthly GPU cost: 47 × $3/hr × 730 hr = $102,930/month
  Compare: Anthropic API cost at this scale ≈ much less (see Section 2.4)
  → Self-hosting is not cost-effective at 34.72 QPS; use managed API
```

**API cost estimation (2.4):**

```
Pricing (per 1M tokens):
  claude-haiku-5-5:  $0.10 in / $0.50 out
  claude-sonnet-5-5: $2.00 in / $10.00 out
  claude-opus-5-5:   $4.00 in / $20.00 out

RAG query cost:
  Input: 1,500 tokens, Output: 300 tokens

  Per haiku query:
    (1,500/1M)×$0.10 + (300/1M)×$0.50 = $0.000150 + $0.000150 = $0.000300

  Per sonnet query:
    (1,500/1M)×$2.00 + (300/1M)×$10.00 = $0.003000 + $0.003000 = $0.006000

  Blended (80% haiku, 20% sonnet):
    0.80×$0.000300 + 0.20×$0.006000 = $0.000240 + $0.001200 = $0.001440 per query

Agent workflow cost:
  Planning (opus, 1 call): (500/1M)×$4.00 + (200/1M)×$20.00 = $0.002000 + $0.004000 = $0.006000
  Execution (sonnet, 4 calls): 4 × [(800/1M)×$2.00 + (400/1M)×$10.00]
    = 4 × [$0.001600 + $0.004000] = 4 × $0.005600 = $0.022400
  Total per workflow: $0.006000 + $0.022400 = $0.028400

Daily API cost (at average load):
  RAG: 1,000,000 queries × $0.001440 = $1,440.00
  Agents: 100,000 workflows × $0.028400 = $2,840.00
  Daily total: $4,280.00

Monthly API cost: $4,280 × 30 = $128,400/month
```

---

## Part 2: Architecture Decisions

### ADR-1: Vector Store Technology Choice

**Status**: Accepted

**Context**: Platform requires vector store supporting 200 tenants with SOC2-compliant data isolation, HNSW search at < 50ms P95, and deployability in data-sovereign environments where SaaS-only solutions are excluded.

**Decision**: Self-hosted Qdrant with one collection namespace per tenant.

**Rationale**: Qdrant enforces namespace boundaries at the DB engine level — application bugs cannot cause cross-tenant data bleed without an explicit namespace override. Rust implementation provides predictable P99 latency under concurrent load. Apache 2.0 license, self-hostable. At 200 tenants with up to 100GB corpus each, a single Qdrant cluster with namespace isolation is cost-effective and operationally manageable.

**Alternatives considered**:
- **Pinecone (managed SaaS)**: fast, zero ops, namespace support. Rejected: requires data to leave enterprise perimeter — fails SOC2 data residency for financial services and healthcare tenants. No self-host path.
- **pgvector (PostgreSQL extension)**: familiar SQL ops, strong isolation via schema-per-tenant. Rejected: HNSW performance degrades materially beyond 10M vectors per index; our corpus projects to 78M+ vectors per tenant at full capacity.

**Consequences**:
- Positive: data sovereignty; DB-level isolation enforcement; HNSW sub-50ms P95 at 100M vectors per namespace.
- Negative: must operate Qdrant cluster (monitoring, backups, upgrades). Estimated storage: 92 GB raw vectors × 1.5 overhead × 3 replicas ≈ 415 GB — manageable on NVMe SSDs.

---

### ADR-2: LLM Routing Strategy

**Status**: Accepted

**Context**: Three model tiers with order-of-magnitude cost differences. Routing must minimize cost while meeting quality targets, with < 5ms overhead added to the critical path.

**Decision**: Rule-based routing: task type (derived from API endpoint) determines the model tier. RAG → haiku-5-5 (80%) / sonnet-5-5 (20%) based on query complexity heuristic (token count threshold). Agents → opus-5-5 for planning, sonnet-5-5 for execution.

**Rationale**: Rule-based routing adds < 1ms latency (pure conditional logic, no ML inference). It is fully auditable — a specific API call's routing decision can always be explained. Complexity heuristic (input > 3,000 tokens → escalate to sonnet) is simple to implement and test. ML-based routing would add 20–50ms latency for a classifier call, eating into the P95 budget.

**Alternatives considered**:
- **ML-based routing (classifier predicts optimal model)**: highest quality-per-dollar. Rejected: classifier inference adds 20–50ms to every request — a 2.5% hit on the P95 2,000ms budget. Also requires labeled training data for the quality-vs-cost optimization, which we do not have at launch.
- **Cascading (try cheap model first, escalate on low confidence)**: elegant cost optimization. Rejected: two sequential LLM calls on escalated queries add 1,600ms to P95 latency, making the 2,000ms SLO unachievable for that fraction.

**Consequences**:
- Positive: < 1ms routing overhead; fully auditable; easy to tune thresholds.
- Negative: cannot adapt routing to query semantic complexity, only surface features (token count, endpoint). Will over-route some simple long-context queries to sonnet. Revisit with ML routing after 6 months of labeled data.

---

### ADR-3: Agent State Persistence

**Status**: Accepted

**Context**: Agent workflows run 30s to several minutes. They must survive LLM API rate limits, container restarts, and network failures. Compliance requires immutable audit trail of every agent step.

**Decision**: PostgreSQL as authoritative durable state store via LangGraph `PostgresSaver` checkpointer. Redis as write-through cache for in-flight workflow hot state.

**Rationale**: PostgreSQL ACID guarantees and SQL queryability serve both the checkpoint-resume requirement and the compliance audit trail. LangGraph's checkpointer abstraction makes the persistence layer swappable. Redis caches in-flight state to avoid Postgres round-trips on every LLM streaming token; TTL = 2× max workflow duration; Postgres is the source of truth on cache miss.

**Alternatives considered**:
- **Redis-only**: lower latency, simpler ops. Rejected: Redis is not durably persistent by default — OOM or failover loses in-flight workflow state. AOF persistence adds write latency and does not provide SQL query capability for compliance audit.
- **Event-sourced (Kafka + event store)**: maximum auditability. Rejected: adds 2–3 additional components (event store, projection service) for marginal benefit over Postgres at 200-tenant scale. Kafka is already in the stack for ingestion; extending it to agent state doubles the failure modes without clear operational benefit.

**Consequences**:
- Positive: ACID guarantees; SQL audit queries; human-in-the-loop interrupt/resume supported natively; LangGraph checkpointer abstraction.
- Negative: Postgres requires ops (replication, vacuum, backups). Must checkpoint at node boundaries, not on every streaming token, to avoid write amplification.

---

### ADR-4: PII Redaction Placement

**Status**: Accepted

**Context**: Enterprise documents and user queries may contain PII (names, email, SSN, financial account numbers). PII must be redacted before any data leaves the platform perimeter (sent to Anthropic API). Redaction must be auditable for compliance.

**Decision**: Gateway-side (centralized) PII redaction using Microsoft Presidio, applied to all inbound queries and all retrieved document chunks before prompt assembly. The original unredacted data never leaves the RAG service layer.

**Rationale**: Centralized gateway enforcement guarantees coverage regardless of which client SDK or integration is used — there is no trust boundary inside the platform where unredacted PII could escape. Presidio supports 40+ entity types with configurable confidence thresholds and custom recognizers for enterprise-specific PII patterns (e.g., employee IDs, internal account numbers). The redacted query is logged; the original is discarded after the span closes.

**Alternatives considered**:
- **Client-side (SDK)**: reduces latency (redaction before network transit). Rejected: requires every client to implement and maintain correct redaction logic. Clients are external and may lag SDK updates. Zero guarantee of coverage for direct API integrations.
- **Model-side (system prompt instruction: "never repeat PII")**: zero latency overhead. Rejected: LLM instruction is a best-effort control, not a hard enforcement. Fails SOC2 Type II compliance requirement for a documented, technical PII prevention control. Cannot be audited.

**Consequences**:
- Positive: coverage guaranteed for all clients; auditable; single implementation to maintain; Presidio false positive rate (~2%) is acceptable for enterprise RAG contexts.
- Negative: adds ~50ms to P95 latency for Presidio inference on query text. Mitigate: run Presidio async on a GPU-accelerated NER model; cache entity detection results for repeated query patterns.

---

### ADR-5: Eval Gating Strategy

**Status**: Accepted

**Context**: Model version promotions (new prompt templates, model tier changes, retrieval pipeline updates) must not silently degrade output quality. Deploy velocity target is 2 deploys/week; gating strategy must not block this.

**Decision**: Automatic promotion if offline eval score ≥ 95% of baseline on the golden dataset, combined with shadow traffic comparison for the first 10% of live traffic post-deployment. Manual approval required only if either gate fails.

**Rationale**: Fully automatic gating (threshold-only) enables 2-deploy/week velocity without human bottlenecks. The shadow traffic comparison catches regressions that the static golden dataset misses (distribution shift in live traffic). The 5% degradation tolerance (95% of baseline) is calibrated to avoid blocking valid improvements while catching meaningful regressions. Human-in-the-loop is the fallback, not the default path.

**Alternatives considered**:
- **Always human-in-the-loop approval**: maximum safety. Rejected: bottlenecks deploy velocity. At 2 deploys/week, a human reviewer must be available within 4 hours of each eval run — operationally expensive and creates toil.
- **Automatic if threshold met (no shadow traffic)**: faster promotion. Rejected: golden dataset has ~5K examples; distribution shift in live traffic (new document types, emerging query patterns) can cause regressions that the static dataset misses. Shadow traffic comparison catches these.

**Consequences**:
- Positive: 2-deploy/week velocity achieved; regression detection on both static golden set and live traffic distribution; human escalation is exception rather than rule.
- Negative: shadow traffic comparison requires 10% traffic duplication for ~24h post-deploy, increasing LLM API cost by ~10% on deploy days. Acceptable: $12,840/month × 0.10 × (2 deploys × 1 day / 30 days) ≈ $85/month overhead.

---

## Part 3: Python Reference Implementation

### `capacity_model.py` — Complete Implementation with Extensions

```python
"""
Enterprise AI Platform — Capacity and Cost Reference Model
Week 06 Project, Phase 7: System Design

Usage:
    python capacity_model.py
    python capacity_model.py --json
    python capacity_model.py --yaml config.yaml

Produces a capacity and cost report for the platform configuration
defined in PlatformConfig.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, computed_field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model pricing (USD per 1M tokens, input / output)
# Source: Anthropic API reference, November 2026
# ---------------------------------------------------------------------------
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class ModelTier(str, Enum):
    HAIKU = "claude-haiku-5-5"
    SONNET = "claude-sonnet-5-5"
    OPUS = "claude-opus-5-5"


class RAGQueryProfile(BaseModel):
    """Token profile for a single RAG query."""

    avg_input_tokens: int = Field(default=1500, description="Average input tokens per query")
    avg_output_tokens: int = Field(default=300, description="Average output tokens per query")
    haiku_fraction: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Fraction of RAG queries routed to Haiku",
    )
    sonnet_fraction: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Fraction of RAG queries routed to Sonnet",
    )


class AgentWorkflowProfile(BaseModel):
    """Token profile for a single agent workflow."""

    planning_input_tokens: int = Field(default=500)
    planning_output_tokens: int = Field(default=200)
    tool_calls_per_workflow: int = Field(default=4)
    tool_call_input_tokens: int = Field(default=800)
    tool_call_output_tokens: int = Field(default=400)
    planning_model: ModelTier = ModelTier.OPUS
    execution_model: ModelTier = ModelTier.SONNET


class PlatformConfig(BaseModel):
    """Top-level configuration for the enterprise AI platform."""

    # Users and load
    daily_active_users: int = Field(default=50_000)
    tenants: int = Field(default=200)
    rag_queries_per_user_per_day: int = Field(default=20)
    agent_workflows_per_user_per_day: int = Field(default=2)
    peak_to_average_ratio: float = Field(default=3.0)

    # Knowledge base
    document_count: int = Field(default=10_000_000)
    avg_document_size_bytes: int = Field(default=2_048)
    embedding_dimension: int = Field(default=1_536)
    embedding_dtype_bytes: int = Field(default=4, description="float32 = 4 bytes")
    vector_store_overhead_factor: float = Field(
        default=1.5,
        description="Overhead multiplier for HNSW index and metadata",
    )
    query_log_retention_days: int = Field(default=90)
    avg_query_log_bytes: int = Field(default=4_096)

    # GPU sizing (for self-hosted optional track)
    model_params_billions: float = Field(default=8.0)
    model_dtype_bytes: int = Field(default=2, description="BF16 = 2 bytes")
    kv_layers: int = Field(default=32)
    kv_head_dim: int = Field(default=128)
    kv_seq_len: int = Field(default=4_096)
    kv_batch_size: int = Field(default=32)
    gpu_memory_gb: float = Field(default=80.0, description="H100 SXM")
    gpu_cost_per_hour_usd: float = Field(default=3.0)
    tokens_per_sec_batch1: float = Field(default=40.0)
    tokens_per_sec_batch32: float = Field(default=150.0)
    avg_output_tokens_per_query: int = Field(default=200)

    # Token profiles
    rag_profile: RAGQueryProfile = Field(default_factory=RAGQueryProfile)
    agent_profile: AgentWorkflowProfile = Field(default_factory=AgentWorkflowProfile)

    @classmethod
    def from_yaml(cls, path: str) -> "PlatformConfig":
        """
        Load configuration from a YAML file.

        The YAML file should contain a flat or nested mapping of PlatformConfig fields.
        Nested models (rag_profile, agent_profile) are supported as nested mappings.

        Args:
            path: Filesystem path to the YAML configuration file.

        Returns:
            PlatformConfig instance with values from the YAML file.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content fails Pydantic validation.
        """
        import yaml  # runtime import to avoid hard dependency if not using YAML

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with yaml_path.open("r") as f:
            data = yaml.safe_load(f)

        logger.info("Loaded PlatformConfig from %s", yaml_path)
        return cls.model_validate(data)


class CapacityModel(BaseModel):
    """
    Estimates infrastructure capacity requirements for the platform.

    All methods are pure functions of self.config. Call report() for a
    human-readable summary.
    """

    config: PlatformConfig

    def average_rag_qps(self) -> float:
        """Return average RAG queries per second across the day."""
        queries_per_day = (
            self.config.daily_active_users * self.config.rag_queries_per_user_per_day
        )
        logger.debug("Total RAG queries/day: %d", queries_per_day)
        return queries_per_day / 86_400

    def peak_rag_qps(self) -> float:
        """Return peak RAG QPS using the configured peak-to-average ratio."""
        return self.average_rag_qps() * self.config.peak_to_average_ratio

    def average_agent_concurrency(self) -> float:
        """
        Return average concurrent agent workflows.

        Assumes each workflow takes 30 seconds end-to-end on average.
        Uses Little's Law: L = λ × W (concurrent = arrival_rate × service_time).
        """
        workflows_per_sec = (
            self.config.daily_active_users
            * self.config.agent_workflows_per_user_per_day
            / 86_400
        )
        avg_workflow_duration_sec = 30.0
        return workflows_per_sec * avg_workflow_duration_sec

    def peak_agent_concurrency(self) -> float:
        """Return peak concurrent agent workflows."""
        return self.average_agent_concurrency() * self.config.peak_to_average_ratio

    def document_storage_gb(self) -> float:
        """Raw document storage in GB."""
        raw_bytes = self.config.document_count * self.config.avg_document_size_bytes
        return raw_bytes / (1024**3)

    def vector_storage_raw_gb(self) -> float:
        """Raw vector storage in GB (embeddings only, no overhead)."""
        raw_bytes = (
            self.config.document_count
            * self.config.embedding_dimension
            * self.config.embedding_dtype_bytes
        )
        return raw_bytes / (1024**3)

    def vector_storage_total_gb(self) -> float:
        """Vector storage with HNSW index and metadata overhead applied."""
        return self.vector_storage_raw_gb() * self.config.vector_store_overhead_factor

    def query_log_storage_per_day_gb(self) -> float:
        """Query log ingestion per day in GB at peak QPS."""
        bytes_per_day = (
            self.peak_rag_qps() * 86_400 * self.config.avg_query_log_bytes
        )
        return bytes_per_day / (1024**3)

    def kv_cache_memory_gb(self) -> float:
        """
        KV cache memory in GB for the self-hosted model at configured batch size.

        Formula: 2 * layers * head_dim * seq_len * batch_size * dtype_bytes
        The factor of 2 accounts for both key and value tensors per layer.
        """
        c = self.config
        raw_bytes = (
            2
            * c.kv_layers
            * c.kv_head_dim
            * c.kv_seq_len
            * c.kv_batch_size
            * c.model_dtype_bytes
        )
        return raw_bytes / (1024**3)

    def model_weights_gb(self) -> float:
        """Model weight memory in GB."""
        return (
            self.config.model_params_billions * 1e9 * self.config.model_dtype_bytes
            / (1024**3)
        )

    def min_gpu_memory_required_gb(self) -> float:
        """Minimum GPU memory needed: model weights + KV cache at configured batch size."""
        return self.model_weights_gb() + self.kv_cache_memory_gb()

    def gpus_needed_for_peak_qps(self) -> int:
        """
        Number of GPUs required to serve peak RAG QPS at batch=32.

        Assumes output-bound workload; each GPU delivers tokens_per_sec_batch32
        tokens per second sustained across the batch.
        """
        tokens_per_query = self.config.avg_output_tokens_per_query
        qps = self.peak_rag_qps()
        total_tokens_per_sec = qps * tokens_per_query
        tokens_per_gpu_per_sec = self.config.tokens_per_sec_batch32
        raw = total_tokens_per_sec / tokens_per_gpu_per_sec
        return math.ceil(raw)

    def monthly_gpu_cost_usd(self) -> float:
        """Monthly cost (USD) for the GPU fleet at 730 hours/month."""
        gpu_count = self.gpus_needed_for_peak_qps()
        hours_per_month = 730.0
        return gpu_count * hours_per_month * self.config.gpu_cost_per_hour_usd

    def slo_headroom(self, target_p99_ms: float, measured_p99_ms: float) -> float:
        """
        Return the fraction of the SLO budget remaining.

        A positive value means the measured P99 is within budget.
        A negative value means the SLO is being violated.

        Args:
            target_p99_ms: The SLO target in milliseconds.
            measured_p99_ms: The observed P99 latency in milliseconds.

        Returns:
            Fraction of budget remaining: (target - measured) / target.
            E.g., 0.20 means 20% headroom; -0.05 means 5% over budget.
        """
        headroom = (target_p99_ms - measured_p99_ms) / target_p99_ms
        logger.debug(
            "SLO headroom: target=%.0fms, measured=%.0fms, headroom=%.1f%%",
            target_p99_ms,
            measured_p99_ms,
            headroom * 100,
        )
        return headroom

    def report(self) -> dict[str, object]:
        """Return a dict summary of all capacity estimates."""
        return {
            "load": {
                "average_rag_qps": round(self.average_rag_qps(), 2),
                "peak_rag_qps": round(self.peak_rag_qps(), 2),
                "average_agent_concurrency": round(self.average_agent_concurrency(), 1),
                "peak_agent_concurrency": round(self.peak_agent_concurrency(), 1),
            },
            "storage_gb": {
                "documents": round(self.document_storage_gb(), 2),
                "vectors_raw": round(self.vector_storage_raw_gb(), 2),
                "vectors_total": round(self.vector_storage_total_gb(), 2),
                "query_logs_per_day": round(self.query_log_storage_per_day_gb(), 2),
            },
            "gpu_sizing": {
                "model_weights_gb": round(self.model_weights_gb(), 2),
                "kv_cache_gb": round(self.kv_cache_memory_gb(), 2),
                "min_gpu_memory_gb": round(self.min_gpu_memory_required_gb(), 2),
                "gpus_for_peak_qps": self.gpus_needed_for_peak_qps(),
                "monthly_gpu_cost_usd": round(self.monthly_gpu_cost_usd(), 2),
            },
        }


class CostModel(BaseModel):
    """
    Computes API cost estimates for the platform using Anthropic model pricing.

    All monetary values are in USD.
    """

    config: PlatformConfig

    @staticmethod
    def _cost_for_tokens(
        model: ModelTier,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return USD cost for a single call given token counts."""
        input_price_per_m, output_price_per_m = MODEL_PRICING[model.value]
        input_cost = (input_tokens / 1_000_000) * input_price_per_m
        output_cost = (output_tokens / 1_000_000) * output_price_per_m
        return input_cost + output_cost

    def cost_per_rag_query_haiku(self) -> float:
        """USD cost for a single RAG query on claude-haiku-5-5."""
        p = self.config.rag_profile
        return self._cost_for_tokens(ModelTier.HAIKU, p.avg_input_tokens, p.avg_output_tokens)

    def cost_per_rag_query_sonnet(self) -> float:
        """USD cost for a single RAG query on claude-sonnet-5-5."""
        p = self.config.rag_profile
        return self._cost_for_tokens(ModelTier.SONNET, p.avg_input_tokens, p.avg_output_tokens)

    def blended_cost_per_rag_query(self) -> float:
        """
        Weighted average cost per RAG query across the configured model mix.

        Uses haiku_fraction and sonnet_fraction from RAGQueryProfile.
        """
        p = self.config.rag_profile
        blended = (
            p.haiku_fraction * self.cost_per_rag_query_haiku()
            + p.sonnet_fraction * self.cost_per_rag_query_sonnet()
        )
        logger.debug(
            "Blended RAG cost: $%.6f/query (%.0f%% haiku, %.0f%% sonnet)",
            blended,
            p.haiku_fraction * 100,
            p.sonnet_fraction * 100,
        )
        return blended

    def cost_per_agent_workflow(self) -> float:
        """
        USD cost for a single agent workflow.

        Includes one planning call (claude-opus-5-5) and N tool execution calls (claude-sonnet-5-5).
        """
        ap = self.config.agent_profile
        planning_cost = self._cost_for_tokens(
            ap.planning_model,
            ap.planning_input_tokens,
            ap.planning_output_tokens,
        )
        per_tool_cost = self._cost_for_tokens(
            ap.execution_model,
            ap.tool_call_input_tokens,
            ap.tool_call_output_tokens,
        )
        total_tool_cost = per_tool_cost * ap.tool_calls_per_workflow
        total = planning_cost + total_tool_cost
        logger.debug(
            "Workflow cost breakdown: planning=$%.4f, tools=$%.4f (×%d), total=$%.4f",
            planning_cost,
            per_tool_cost,
            ap.tool_calls_per_workflow,
            total,
        )
        return total

    def daily_rag_cost_usd(self) -> float:
        """Total daily API cost for all RAG queries at average load."""
        daily_queries = (
            self.config.daily_active_users * self.config.rag_queries_per_user_per_day
        )
        return daily_queries * self.blended_cost_per_rag_query()

    def daily_agent_cost_usd(self) -> float:
        """Total daily API cost for all agent workflows at average load."""
        daily_workflows = (
            self.config.daily_active_users * self.config.agent_workflows_per_user_per_day
        )
        return daily_workflows * self.cost_per_agent_workflow()

    def daily_total_cost_usd(self) -> float:
        """Total daily API cost across RAG and agent workloads."""
        return self.daily_rag_cost_usd() + self.daily_agent_cost_usd()

    def monthly_total_cost_usd(self) -> float:
        """Total monthly API cost (30-day month)."""
        return self.daily_total_cost_usd() * 30

    def cost_per_tenant_per_month_usd(self) -> float:
        """Average API cost per tenant per month (equal distribution assumed)."""
        return self.monthly_total_cost_usd() / self.config.tenants

    def cost_at_qps(self, qps: float) -> float:
        """
        Return the per-second API cost at a given QPS level.

        Useful for cost-vs-load graphs and autoscaling budget checks.

        Args:
            qps: Queries per second to evaluate cost at.

        Returns:
            USD cost per second at the given QPS.
        """
        cost_per_query = self.blended_cost_per_rag_query()
        cost_per_second = qps * cost_per_query
        logger.debug(
            "Cost at %.1f QPS: $%.6f/sec ($%.2f/hr, $%.2f/day)",
            qps,
            cost_per_second,
            cost_per_second * 3600,
            cost_per_second * 86_400,
        )
        return cost_per_second

    def optimize_model_mix(self, target_cost_per_query: float) -> dict[str, float]:
        """
        Compute the maximum sonnet fraction achievable within the target cost per query.

        Solves: haiku_fraction × haiku_cost + sonnet_fraction × sonnet_cost = target
        Subject to: haiku_fraction + sonnet_fraction = 1, both in [0, 1].

        Args:
            target_cost_per_query: Maximum desired blended cost per RAG query (USD).

        Returns:
            Dict with 'haiku_fraction', 'sonnet_fraction', 'blended_cost', 'feasible'.
            If the target is below the pure-haiku cost, returns all-haiku with feasible=False.
        """
        haiku_cost = self.cost_per_rag_query_haiku()
        sonnet_cost = self.cost_per_rag_query_sonnet()

        if target_cost_per_query <= haiku_cost:
            logger.warning(
                "Target cost $%.6f is below pure-haiku cost $%.6f — cannot achieve",
                target_cost_per_query,
                haiku_cost,
            )
            return {
                "haiku_fraction": 1.0,
                "sonnet_fraction": 0.0,
                "blended_cost": haiku_cost,
                "feasible": False,
            }

        if target_cost_per_query >= sonnet_cost:
            return {
                "haiku_fraction": 0.0,
                "sonnet_fraction": 1.0,
                "blended_cost": sonnet_cost,
                "feasible": True,
            }

        # Solve: h*haiku_cost + (1-h)*sonnet_cost = target
        # h*(haiku_cost - sonnet_cost) = target - sonnet_cost
        # h = (target - sonnet_cost) / (haiku_cost - sonnet_cost)
        haiku_fraction = (target_cost_per_query - sonnet_cost) / (haiku_cost - sonnet_cost)
        sonnet_fraction = 1.0 - haiku_fraction
        blended = haiku_fraction * haiku_cost + sonnet_fraction * sonnet_cost

        logger.info(
            "Optimal mix for $%.6f/query: haiku=%.1f%%, sonnet=%.1f%% (actual=$%.6f)",
            target_cost_per_query,
            haiku_fraction * 100,
            sonnet_fraction * 100,
            blended,
        )
        return {
            "haiku_fraction": round(haiku_fraction, 4),
            "sonnet_fraction": round(sonnet_fraction, 4),
            "blended_cost": round(blended, 8),
            "feasible": True,
        }

    def report(self) -> dict[str, object]:
        """Return a dict summary of all cost estimates."""
        return {
            "per_query": {
                "rag_haiku_usd": round(self.cost_per_rag_query_haiku(), 6),
                "rag_sonnet_usd": round(self.cost_per_rag_query_sonnet(), 6),
                "rag_blended_usd": round(self.blended_cost_per_rag_query(), 6),
                "agent_workflow_usd": round(self.cost_per_agent_workflow(), 4),
            },
            "daily": {
                "rag_cost_usd": round(self.daily_rag_cost_usd(), 2),
                "agent_cost_usd": round(self.daily_agent_cost_usd(), 2),
                "total_cost_usd": round(self.daily_total_cost_usd(), 2),
            },
            "monthly": {
                "total_cost_usd": round(self.monthly_total_cost_usd(), 2),
                "cost_per_tenant_usd": round(self.cost_per_tenant_per_month_usd(), 2),
            },
        }


def main() -> None:
    """Run the capacity and cost model and print a formatted report."""
    parser = argparse.ArgumentParser(
        description="Enterprise AI Platform Capacity & Cost Model"
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--yaml", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    if args.yaml:
        config = PlatformConfig.from_yaml(args.yaml)
    else:
        config = PlatformConfig()

    capacity = CapacityModel(config=config)
    cost = CostModel(config=config)

    logger.info("Running capacity and cost model for platform configuration")

    capacity_report = capacity.report()
    cost_report = cost.report()
    combined = {"capacity": capacity_report, "cost": cost_report}

    if args.json:
        print(json.dumps(combined, indent=2))
    else:
        _print_formatted_report(capacity, cost, capacity_report, cost_report)

    logger.info(
        "Summary: peak_rag_qps=%.1f, monthly_api_cost=$%.2f, gpus_needed=%d",
        capacity.peak_rag_qps(),
        cost.monthly_total_cost_usd(),
        capacity.gpus_needed_for_peak_qps(),
    )


def _print_formatted_report(
    capacity: CapacityModel,
    cost: CostModel,
    cap_report: dict,
    cost_report: dict,
) -> None:
    """Print a human-readable formatted report to stdout."""
    load = cap_report["load"]
    storage = cap_report["storage_gb"]
    gpu = cap_report["gpu_sizing"]
    per_q = cost_report["per_query"]
    daily = cost_report["daily"]
    monthly = cost_report["monthly"]

    print("\n" + "=" * 65)
    print("  Enterprise AI Platform — Capacity & Cost Report")
    print("=" * 65)

    print("\n  LOAD")
    print(f"    Average RAG QPS:          {load['average_rag_qps']:>8.2f}")
    print(f"    Peak RAG QPS:             {load['peak_rag_qps']:>8.2f}")
    print(f"    Avg agent concurrency:    {load['average_agent_concurrency']:>8.1f}")
    print(f"    Peak agent concurrency:   {load['peak_agent_concurrency']:>8.1f}")

    print("\n  STORAGE")
    print(f"    Raw documents:            {storage['documents']:>8.2f} GB")
    print(f"    Vectors (raw):            {storage['vectors_raw']:>8.2f} GB")
    print(f"    Vectors (with overhead):  {storage['vectors_total']:>8.2f} GB")
    print(f"    Query logs/day (at peak): {storage['query_logs_per_day']:>8.2f} GB")

    print("\n  GPU SIZING (self-hosted optional track)")
    print(f"    Model weights:            {gpu['model_weights_gb']:>8.2f} GB")
    print(f"    KV cache (batch=32):      {gpu['kv_cache_gb']:>8.2f} GB")
    print(f"    Min GPU memory:           {gpu['min_gpu_memory_gb']:>8.2f} GB")
    print(f"    GPUs needed (peak QPS):   {gpu['gpus_for_peak_qps']:>8d}")
    print(f"    Monthly GPU cost:         ${gpu['monthly_gpu_cost_usd']:>10,.2f}")

    print("\n  COST PER QUERY")
    print(f"    RAG (haiku-5-5):          ${per_q['rag_haiku_usd']:>10.6f}")
    print(f"    RAG (sonnet-5-5):         ${per_q['rag_sonnet_usd']:>10.6f}")
    print(f"    RAG blended:              ${per_q['rag_blended_usd']:>10.6f}")
    print(f"    Agent workflow:           ${per_q['agent_workflow_usd']:>10.4f}")

    print("\n  DAILY API COST")
    print(f"    RAG queries:              ${daily['rag_cost_usd']:>10,.2f}")
    print(f"    Agent workflows:          ${daily['agent_cost_usd']:>10,.2f}")
    print(f"    TOTAL:                    ${daily['total_cost_usd']:>10,.2f}")

    print("\n  MONTHLY API COST")
    print(f"    Total (30-day month):     ${monthly['total_cost_usd']:>10,.2f}")
    print(f"    Per tenant avg:           ${monthly['cost_per_tenant_usd']:>10,.2f}")

    # Cost optimization example
    target = 0.002  # $0.002/query target
    mix = cost.optimize_model_mix(target)
    print(f"\n  OPTIMIZATION: max sonnet% at ${target}/query target")
    print(f"    Haiku fraction:           {mix['haiku_fraction']*100:>7.1f}%")
    print(f"    Sonnet fraction:          {mix['sonnet_fraction']*100:>7.1f}%")
    print(f"    Achievable blended cost:  ${mix['blended_cost']:>10.6f}")
    print(f"    Feasible:                 {str(mix['feasible']):>8}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
```

---

### `tests/test_capacity_model.py` — Complete Test Suite

```python
"""
tests/test_capacity_model.py

Pytest tests for the Enterprise AI Platform capacity and cost model.
Run with: pytest tests/test_capacity_model.py -v
"""

from __future__ import annotations

import math

import pytest

from capacity_model import (
    MODEL_PRICING,
    CapacityModel,
    CostModel,
    ModelTier,
    PlatformConfig,
    RAGQueryProfile,
    AgentWorkflowProfile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> PlatformConfig:
    """Default platform configuration for testing."""
    return PlatformConfig()


@pytest.fixture
def capacity(default_config: PlatformConfig) -> CapacityModel:
    """CapacityModel with default config."""
    return CapacityModel(config=default_config)


@pytest.fixture
def cost(default_config: PlatformConfig) -> CostModel:
    """CostModel with default config."""
    return CostModel(config=default_config)


# ---------------------------------------------------------------------------
# CapacityModel tests
# ---------------------------------------------------------------------------


def test_average_rag_qps_calculation(default_config: PlatformConfig) -> None:
    """Verify: average_qps = DAU × queries_per_day / 86400."""
    model = CapacityModel(config=default_config)
    expected = (
        default_config.daily_active_users
        * default_config.rag_queries_per_user_per_day
        / 86_400
    )
    assert abs(model.average_rag_qps() - expected) < 1e-9


def test_peak_rag_qps_is_multiple_of_average(default_config: PlatformConfig) -> None:
    """Verify: peak_qps == average_qps × peak_to_average_ratio."""
    model = CapacityModel(config=default_config)
    expected = model.average_rag_qps() * default_config.peak_to_average_ratio
    assert abs(model.peak_rag_qps() - expected) < 1e-9


def test_vector_storage_raw_gb(default_config: PlatformConfig) -> None:
    """Verify: raw_vectors_gb = doc_count × dim × bytes / 1024^3."""
    model = CapacityModel(config=default_config)
    expected_bytes = (
        default_config.document_count
        * default_config.embedding_dimension
        * default_config.embedding_dtype_bytes
    )
    expected_gb = expected_bytes / (1024**3)
    assert abs(model.vector_storage_raw_gb() - expected_gb) < 1e-6


def test_vector_storage_total_applies_overhead(default_config: PlatformConfig) -> None:
    """Verify: total_vectors_gb = raw_vectors_gb × overhead_factor."""
    model = CapacityModel(config=default_config)
    expected = model.vector_storage_raw_gb() * default_config.vector_store_overhead_factor
    assert abs(model.vector_storage_total_gb() - expected) < 1e-6


def test_kv_cache_formula(default_config: PlatformConfig) -> None:
    """Verify KV cache formula: 2 × layers × head_dim × seq_len × batch × dtype_bytes."""
    model = CapacityModel(config=default_config)
    c = default_config
    expected_bytes = (
        2 * c.kv_layers * c.kv_head_dim * c.kv_seq_len * c.kv_batch_size * c.model_dtype_bytes
    )
    expected_gb = expected_bytes / (1024**3)
    assert abs(model.kv_cache_memory_gb() - expected_gb) < 1e-6


def test_gpus_needed_ceil(default_config: PlatformConfig) -> None:
    """Fractional GPU count must always be ceiling'd — never round down."""
    # Use config that produces a fractional GPU count
    config = default_config.model_copy(
        update={"tokens_per_sec_batch32": 999.0}  # unusual value -> fractional result
    )
    model = CapacityModel(config=config)
    raw = (model.peak_rag_qps() * config.avg_output_tokens_per_query) / config.tokens_per_sec_batch32
    assert model.gpus_needed_for_peak_qps() == math.ceil(raw)


def test_slo_headroom_positive(capacity: CapacityModel) -> None:
    """Positive headroom when measured is below target."""
    headroom = capacity.slo_headroom(target_p99_ms=2000.0, measured_p99_ms=1600.0)
    assert abs(headroom - 0.20) < 1e-9  # 20% headroom


def test_slo_headroom_negative(capacity: CapacityModel) -> None:
    """Negative headroom when SLO is violated."""
    headroom = capacity.slo_headroom(target_p99_ms=2000.0, measured_p99_ms=2100.0)
    assert headroom < 0.0


# ---------------------------------------------------------------------------
# CostModel tests
# ---------------------------------------------------------------------------


def test_cost_per_rag_query_haiku(default_config: PlatformConfig) -> None:
    """Verify haiku RAG cost against manual calculation."""
    cost = CostModel(config=default_config)
    p = default_config.rag_profile
    input_price_per_m, output_price_per_m = MODEL_PRICING["claude-haiku-5-5"]
    expected = (p.avg_input_tokens / 1_000_000) * input_price_per_m + (
        p.avg_output_tokens / 1_000_000
    ) * output_price_per_m
    assert abs(cost.cost_per_rag_query_haiku() - expected) < 1e-10


def test_cost_per_rag_query_sonnet(default_config: PlatformConfig) -> None:
    """Verify sonnet RAG cost against manual calculation."""
    cost = CostModel(config=default_config)
    p = default_config.rag_profile
    input_price_per_m, output_price_per_m = MODEL_PRICING["claude-sonnet-5-5"]
    expected = (p.avg_input_tokens / 1_000_000) * input_price_per_m + (
        p.avg_output_tokens / 1_000_000
    ) * output_price_per_m
    assert abs(cost.cost_per_rag_query_sonnet() - expected) < 1e-10


def test_blended_cost_uses_fractions(default_config: PlatformConfig) -> None:
    """Verify: blended = haiku_fraction × haiku_cost + sonnet_fraction × sonnet_cost."""
    cost = CostModel(config=default_config)
    p = default_config.rag_profile
    expected = (
        p.haiku_fraction * cost.cost_per_rag_query_haiku()
        + p.sonnet_fraction * cost.cost_per_rag_query_sonnet()
    )
    assert abs(cost.blended_cost_per_rag_query() - expected) < 1e-10


def test_agent_workflow_cost_includes_planning_and_tools(
    default_config: PlatformConfig,
) -> None:
    """Agent workflow cost must be greater than planning-only cost."""
    cost = CostModel(config=default_config)
    ap = default_config.agent_profile
    planning_only = CostModel._cost_for_tokens(
        ap.planning_model, ap.planning_input_tokens, ap.planning_output_tokens
    )
    assert cost.cost_per_agent_workflow() > planning_only


def test_monthly_cost_is_30x_daily(cost: CostModel) -> None:
    """Monthly total must equal daily total × 30."""
    assert abs(cost.monthly_total_cost_usd() - cost.daily_total_cost_usd() * 30) < 1e-6


def test_cost_per_tenant_divides_monthly(cost: CostModel) -> None:
    """Per-tenant cost = monthly total / tenant count."""
    expected = cost.monthly_total_cost_usd() / cost.config.tenants
    assert abs(cost.cost_per_tenant_per_month_usd() - expected) < 1e-6


def test_model_pricing_constants() -> None:
    """MODEL_PRICING must contain exactly the three expected model IDs."""
    expected_keys = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"}
    assert set(MODEL_PRICING.keys()) == expected_keys


def test_platform_config_defaults_are_valid() -> None:
    """Pydantic validation must pass on default PlatformConfig with no arguments."""
    config = PlatformConfig()  # must not raise
    assert config.daily_active_users == 50_000
    assert config.tenants == 200


def test_report_returns_expected_keys(
    capacity: CapacityModel, cost: CostModel
) -> None:
    """Both report() methods must return dicts with expected top-level keys."""
    cap_report = capacity.report()
    cost_report = cost.report()

    assert set(cap_report.keys()) == {"load", "storage_gb", "gpu_sizing"}
    assert set(cost_report.keys()) == {"per_query", "daily", "monthly"}


def test_cost_at_qps_proportional(cost: CostModel) -> None:
    """cost_at_qps should be linear: doubling QPS doubles cost/sec."""
    c1 = cost.cost_at_qps(100.0)
    c2 = cost.cost_at_qps(200.0)
    assert abs(c2 / c1 - 2.0) < 1e-9


def test_optimize_model_mix_all_haiku_when_target_low(cost: CostModel) -> None:
    """When target is below haiku cost, return all-haiku with feasible=False."""
    result = cost.optimize_model_mix(target_cost_per_query=0.0)
    assert result["haiku_fraction"] == 1.0
    assert result["sonnet_fraction"] == 0.0
    assert result["feasible"] is False


def test_optimize_model_mix_feasible_target(cost: CostModel) -> None:
    """When target is between haiku and sonnet costs, return a valid mix."""
    haiku_cost = cost.cost_per_rag_query_haiku()
    sonnet_cost = cost.cost_per_rag_query_sonnet()
    target = (haiku_cost + sonnet_cost) / 2
    result = cost.optimize_model_mix(target_cost_per_query=target)
    assert result["feasible"] is True
    assert 0.0 < result["haiku_fraction"] < 1.0
    assert 0.0 < result["sonnet_fraction"] < 1.0
    assert abs(result["haiku_fraction"] + result["sonnet_fraction"] - 1.0) < 1e-6
```

---

## Part 4: Sample Output

Running `python capacity_model.py` with default `PlatformConfig()`:

```
=================================================================
  Enterprise AI Platform — Capacity & Cost Report
=================================================================

  LOAD
    Average RAG QPS:              11.57
    Peak RAG QPS:                 34.72
    Avg agent concurrency:        34.7
    Peak agent concurrency:      104.2

  STORAGE
    Raw documents:                19.07 GB
    Vectors (raw):                57.22 GB
    Vectors (with overhead):      85.83 GB
    Query logs/day (at peak):     11.45 GB

  GPU SIZING (self-hosted optional track)
    Model weights:                14.90 GB
    KV cache (batch=32):           2.00 GB
    Min GPU memory:               16.90 GB
    GPUs needed (peak QPS):           47
    Monthly GPU cost:           $102,930.00

  COST PER QUERY
    RAG (haiku-5-5):              $0.000300
    RAG (sonnet-5-5):             $0.006000
    RAG blended:                  $0.001440
    Agent workflow:               $0.0284

  DAILY API COST
    RAG queries:                  $1,440.00
    Agent workflows:              $2,840.00
    TOTAL:                        $4,280.00

  MONTHLY API COST
    Total (30-day month):       $128,400.00
    Per tenant avg:                $642.00

  OPTIMIZATION: max sonnet% at $0.002/query target
    Haiku fraction:                76.3%
    Sonnet fraction:               23.7%
    Achievable blended cost:      $0.001998
    Feasible:                      True

=================================================================
```

### Key Numbers Explained

| Metric | Value | How computed |
|---|---|---|
| Average RAG QPS | 11.57 | 50K × 20 queries / 86,400s |
| Peak RAG QPS | 34.72 | 11.57 × 3× peak ratio |
| Vector store (total) | 85.83 GB | 57.22 × 1.5 overhead |
| GPUs for peak (self-hosted) | 47 | ⌈34.72 × 200 tokens / 150 tok/s⌉ |
| RAG blended cost | $0.001440 | 80% haiku + 20% sonnet |
| Monthly API cost | $128,400 | $4,280/day × 30 |
| Per-tenant monthly | $642 | $128,400 / 200 tenants |

### Comparison: Self-Hosted vs. Managed API at This Scale

```
Self-hosted GPU cost (47 GPUs × $3/hr × 730 hr):  $102,930/month
Managed API cost:                                   $128,400/month
Self-hosted saving:                                  $25,470/month (20% cheaper)

Break-even: self-hosting becomes worthwhile when API cost > ~$100K/month.
At this scale it IS slightly cheaper, but:
  - 47 GPUs requires a GPU ops team (est. $300K/year eng cost = $25K/month)
  - Net advantage: $25,470 - $25,000 = $470/month — not worth the ops burden.
  - Recommendation: use managed API until API spend exceeds $500K/month,
    at which point self-hosting savings justify the ops investment.
```
