# Phase 7 · Week 06 · Day 7 — Sunday Project: Capacity/Cost Model + Assessment Prep

> Objective: Implement the week's capacity/cost model in Python and prepare for the system design assessment with anti-pattern awareness and mock walk-throughs.  ·  Time: ~3 hr project + 1 hr assessment prep  ·  Prereqs: [[day-6]]

---

## Part 1: Week Project — Capacity/Cost Model Implementation (3 hours)

### What You Are Building

A Python library and CLI tool that models the capacity and cost of an enterprise AI platform. Given a set of parameters (QPS targets, model choices, retention periods, number of tenants), it outputs:

- Token throughput requirements
- GPU capacity estimates for embedding and (optionally) self-hosted LLM
- Storage sizing for vector store and object storage
- Monthly cost projection with per-component breakdown
- SLO feasibility analysis (latency budget check per component)
- Optimization scenario comparison (baseline vs. caching + tiering)

This is a practical tool you will reuse in real architecture reviews. The code structure follows the project conventions from this course.

### Project Scaffold

Study this scaffold carefully before writing your implementation. It defines the data models and interfaces; your job is to fill in the methods.

```python
"""
week-project/capacity_cost_model/models.py

Pydantic models for the capacity/cost modeling tool.
All numeric fields use SI units (seconds, bytes, tokens) unless noted.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class ModelTier(str, Enum):
    """Available LLM model tiers."""
    HAIKU = "claude-haiku-5-5"
    SONNET = "claude-sonnet-5-5"
    OPUS = "claude-opus-5-5"


class ModelPricing(BaseModel):
    """Token pricing for a single model tier (USD per token)."""
    model: ModelTier
    input_price_per_token: float   # e.g., 3e-6 for $3/M tokens
    output_price_per_token: float  # e.g., 15e-6 for $15/M tokens

    @field_validator("input_price_per_token", "output_price_per_token")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return v


class TrafficProfile(BaseModel):
    """Describes the request traffic pattern."""
    peak_qps: float = Field(..., gt=0, description="Peak queries per second")
    average_load_factor: float = Field(
        0.30, ge=0.0, le=1.0,
        description="Average QPS as fraction of peak"
    )
    days_per_month: int = Field(30, ge=28, le=31)

    @property
    def average_qps(self) -> float:
        return self.peak_qps * self.average_load_factor

    @property
    def monthly_queries(self) -> float:
        return self.average_qps * 86_400 * self.days_per_month


class RAGQueryProfile(BaseModel):
    """Token breakdown for a single RAG query."""
    system_prompt_tokens: int = Field(500, gt=0)
    retrieved_context_tokens: int = Field(1500, gt=0)
    user_query_tokens: int = Field(50, gt=0)
    output_tokens: int = Field(300, gt=0)
    model: ModelTier = ModelTier.SONNET

    @property
    def input_tokens(self) -> int:
        return self.system_prompt_tokens + self.retrieved_context_tokens + self.user_query_tokens


class CorpusProfile(BaseModel):
    """Describes the document corpus for sizing the vector store."""
    num_tenants: int = Field(..., gt=0)
    max_corpus_per_tenant_gb: float = Field(..., gt=0, description="GB of raw text per tenant")
    chunk_size_tokens: int = Field(512, gt=0)
    embedding_dimensions: int = Field(1536, gt=0)
    replication_factor: int = Field(3, ge=1)
    avg_tokens_per_gb_text: float = Field(1_000_000.0, gt=0)

    @property
    def total_raw_text_gb(self) -> float:
        return self.num_tenants * self.max_corpus_per_tenant_gb

    @property
    def total_chunks(self) -> float:
        total_tokens = self.total_raw_text_gb * self.avg_tokens_per_gb_text
        return total_tokens / self.chunk_size_tokens

    @property
    def vector_store_bytes_raw(self) -> float:
        """Size of embedding vectors before replication."""
        bytes_per_vector = self.embedding_dimensions * 4  # float32
        metadata_overhead = 1.30  # 30% metadata overhead
        return self.total_chunks * bytes_per_vector * metadata_overhead

    @property
    def vector_store_bytes_replicated(self) -> float:
        return self.vector_store_bytes_raw * self.replication_factor


class OptimizationConfig(BaseModel):
    """Configuration for cost optimization scenarios."""
    prompt_cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0)
    semantic_cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0)
    context_compression_ratio: float = Field(
        0.0, ge=0.0, le=0.9,
        description="Fraction by which input tokens are reduced via compression"
    )
    haiku_fraction: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Fraction of queries routed to haiku instead of sonnet"
    )


class CapacityReport(BaseModel):
    """Full output of the capacity/cost model."""
    # Traffic
    peak_qps: float
    average_qps: float
    monthly_queries: float

    # Token throughput
    peak_input_tokens_per_second: float
    peak_output_tokens_per_second: float

    # Storage
    vector_store_tb_replicated: float
    object_storage_tb: float

    # Cost (monthly, USD)
    llm_cost_monthly_usd: float
    eval_cost_monthly_usd: float
    infra_cost_monthly_usd: float
    total_cost_monthly_usd: float
    cost_per_query_usd: float

    # Latency budget feasibility
    latency_budget_ms: float
    latency_feasible: bool
    latency_breakdown: dict[str, float]

    # Optimization savings
    optimized_cost_monthly_usd: float | None = None
    optimization_savings_pct: float | None = None
```

```python
"""
week-project/capacity_cost_model/engine.py

Core computation engine. Implements the capacity and cost calculations.
Fill in all methods marked TODO.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .models import (
    CapacityReport,
    CorpusProfile,
    ModelPricing,
    ModelTier,
    OptimizationConfig,
    RAGQueryProfile,
    TrafficProfile,
)

logger = logging.getLogger(__name__)

# Default pricing (update when Anthropic changes rates)
DEFAULT_PRICING: dict[ModelTier, ModelPricing] = {
    ModelTier.HAIKU: ModelPricing(
        model=ModelTier.HAIKU,
        input_price_per_token=0.25e-6,
        output_price_per_token=1.25e-6,
    ),
    ModelTier.SONNET: ModelPricing(
        model=ModelTier.SONNET,
        input_price_per_token=3.0e-6,
        output_price_per_token=15.0e-6,
    ),
    ModelTier.OPUS: ModelPricing(
        model=ModelTier.OPUS,
        input_price_per_token=15.0e-6,
        output_price_per_token=75.0e-6,
    ),
}

# Latency benchmarks (ms, P95 targets)
LATENCY_BENCHMARKS: dict[str, float] = {
    "query_embedding":   30.0,
    "vector_ann_search": 50.0,
    "reranking":        100.0,
    "prompt_assembly":   10.0,
    "network_overhead":  200.0,
    # LLM latency is computed from output tokens at an assumed throughput
}
LLM_OUTPUT_TOKENS_PER_SECOND = 80.0   # approximate streaming throughput
RAG_LATENCY_SLO_MS = 2000.0


class CapacityEngine:
    """
    Computes capacity requirements and cost projections for an enterprise AI platform.
    """

    def __init__(
        self,
        pricing: dict[ModelTier, ModelPricing] | None = None,
    ) -> None:
        self._pricing = pricing or DEFAULT_PRICING

    def compute_llm_latency_ms(self, query_profile: RAGQueryProfile) -> float:
        """
        Estimate P95 LLM API latency from output token count.
        Uses LLM_OUTPUT_TOKENS_PER_SECOND as the throughput assumption.
        First-token latency is not modeled here (simplification).
        """
        # TODO: implement
        # Return: output_tokens / tokens_per_second * 1000 (convert to ms)
        raise NotImplementedError

    def compute_latency_breakdown(
        self, query_profile: RAGQueryProfile
    ) -> tuple[dict[str, float], bool]:
        """
        Return (breakdown dict, feasible bool) where breakdown maps component
        name to ms, and feasible is True iff total <= RAG_LATENCY_SLO_MS.
        """
        # TODO: implement
        # Use LATENCY_BENCHMARKS for fixed components, compute_llm_latency_ms for LLM.
        raise NotImplementedError

    def compute_token_throughput(
        self, traffic: TrafficProfile, query_profile: RAGQueryProfile
    ) -> tuple[float, float]:
        """
        Return (peak_input_tokens_per_second, peak_output_tokens_per_second).
        """
        # TODO: implement
        raise NotImplementedError

    def compute_llm_cost(
        self,
        traffic: TrafficProfile,
        query_profile: RAGQueryProfile,
        optimization: OptimizationConfig | None = None,
    ) -> float:
        """
        Compute monthly LLM cost in USD.
        Apply optimization reductions if provided:
          - prompt_cache_hit_rate reduces input tokens on cache hits by 90% (cached prefix)
          - semantic_cache_hit_rate removes the LLM call entirely on hits
          - context_compression_ratio reduces retrieved_context_tokens proportionally
          - haiku_fraction routes that fraction of queries to haiku pricing
        """
        # TODO: implement
        raise NotImplementedError

    def compute_eval_cost(
        self,
        traffic: TrafficProfile,
        sample_rate: float = 0.10,
        judge_input_tokens: int = 2500,
        judge_output_tokens: int = 200,
    ) -> float:
        """
        Compute monthly cost of LLM-as-judge eval at the given sample rate.
        Uses haiku pricing for the judge.
        """
        # TODO: implement
        raise NotImplementedError

    def compute_storage(self, corpus: CorpusProfile) -> tuple[float, float]:
        """
        Return (vector_store_tb, object_storage_tb).
        object_storage = 2 × raw text (raw + extracted text).
        """
        # TODO: implement
        raise NotImplementedError

    def run(
        self,
        traffic: TrafficProfile,
        query_profile: RAGQueryProfile,
        corpus: CorpusProfile,
        infra_cost_monthly_usd: float = 5000.0,
        eval_sample_rate: float = 0.10,
        optimization: OptimizationConfig | None = None,
    ) -> CapacityReport:
        """
        Execute the full capacity/cost model and return a CapacityReport.
        Logs key figures at INFO level.
        """
        # TODO: orchestrate all sub-computations into a CapacityReport
        raise NotImplementedError
```

```python
"""
week-project/capacity_cost_model/cli.py

CLI entry point. Run with:
  python -m capacity_cost_model.cli --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .engine import CapacityEngine
from .models import (
    CorpusProfile,
    ModelTier,
    OptimizationConfig,
    RAGQueryProfile,
    TrafficProfile,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Enterprise AI Platform Capacity & Cost Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--peak-qps", type=float, default=500.0, help="Peak QPS target")
    p.add_argument("--avg-load", type=float, default=0.30, help="Average load factor (0–1)")
    p.add_argument("--num-tenants", type=int, default=100, help="Number of tenants")
    p.add_argument(
        "--corpus-gb", type=float, default=100.0, help="Max corpus per tenant (GB)"
    )
    p.add_argument(
        "--model",
        choices=[m.value for m in ModelTier],
        default=ModelTier.SONNET.value,
        help="Primary LLM model tier",
    )
    p.add_argument(
        "--optimize",
        action="store_true",
        help="Show optimized scenario (caching + tiering)",
    )
    p.add_argument("--output-json", action="store_true", help="Output raw JSON report")
    return p


def format_report(report: object, optimized: object | None = None) -> str:
    """Format a CapacityReport for human-readable CLI output."""
    # TODO: implement a clean tabular output
    # Show: monthly queries, peak tokens/sec, storage, cost breakdown, latency feasibility
    # If optimized is provided, show side-by-side baseline vs optimized.
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = build_parser().parse_args(argv)

    traffic = TrafficProfile(
        peak_qps=args.peak_qps,
        average_load_factor=args.avg_load,
    )
    query_profile = RAGQueryProfile(model=ModelTier(args.model))
    corpus = CorpusProfile(
        num_tenants=args.num_tenants,
        max_corpus_per_tenant_gb=args.corpus_gb,
    )

    engine = CapacityEngine()
    report = engine.run(traffic=traffic, query_profile=query_profile, corpus=corpus)

    optimized_report = None
    if args.optimize:
        opt = OptimizationConfig(
            prompt_cache_hit_rate=0.80,
            semantic_cache_hit_rate=0.15,
            context_compression_ratio=0.35,
            haiku_fraction=0.40,
        )
        optimized_report = engine.run(
            traffic=traffic,
            query_profile=query_profile,
            corpus=corpus,
            optimization=opt,
        )

    if args.output_json:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print(format_report(report, optimized_report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
"""
week-project/tests/test_engine.py

Unit tests for CapacityEngine. Fill in test bodies after implementing the engine.
"""

from __future__ import annotations

import pytest

from capacity_cost_model.engine import (
    CapacityEngine,
    LLM_OUTPUT_TOKENS_PER_SECOND,
    RAG_LATENCY_SLO_MS,
)
from capacity_cost_model.models import (
    CorpusProfile,
    ModelTier,
    OptimizationConfig,
    RAGQueryProfile,
    TrafficProfile,
)


@pytest.fixture
def engine() -> CapacityEngine:
    return CapacityEngine()


@pytest.fixture
def default_traffic() -> TrafficProfile:
    return TrafficProfile(peak_qps=500.0, average_load_factor=0.30)


@pytest.fixture
def default_query() -> RAGQueryProfile:
    return RAGQueryProfile()


@pytest.fixture
def default_corpus() -> CorpusProfile:
    return CorpusProfile(num_tenants=100, max_corpus_per_tenant_gb=100.0)


class TestLLMLatency:
    def test_latency_proportional_to_output_tokens(
        self, engine: CapacityEngine
    ) -> None:
        """Doubling output tokens should double LLM latency."""
        q1 = RAGQueryProfile(output_tokens=300)
        q2 = RAGQueryProfile(output_tokens=600)
        lat1 = engine.compute_llm_latency_ms(q1)
        lat2 = engine.compute_llm_latency_ms(q2)
        assert abs(lat2 / lat1 - 2.0) < 0.01

    def test_latency_uses_benchmark_throughput(
        self, engine: CapacityEngine
    ) -> None:
        q = RAGQueryProfile(output_tokens=int(LLM_OUTPUT_TOKENS_PER_SECOND))
        latency = engine.compute_llm_latency_ms(q)
        assert abs(latency - 1000.0) < 1.0  # 1 second for 1x throughput worth of tokens


class TestLatencyFeasibility:
    def test_default_profile_is_feasible(
        self,
        engine: CapacityEngine,
        default_query: RAGQueryProfile,
    ) -> None:
        _, feasible = engine.compute_latency_breakdown(default_query)
        assert feasible is True

    def test_very_long_output_is_infeasible(
        self, engine: CapacityEngine
    ) -> None:
        q = RAGQueryProfile(output_tokens=2000)
        _, feasible = engine.compute_latency_breakdown(q)
        assert feasible is False


class TestCostModel:
    def test_haiku_cheaper_than_sonnet(
        self,
        engine: CapacityEngine,
        default_traffic: TrafficProfile,
    ) -> None:
        q_haiku = RAGQueryProfile(model=ModelTier.HAIKU)
        q_sonnet = RAGQueryProfile(model=ModelTier.SONNET)
        cost_haiku = engine.compute_llm_cost(default_traffic, q_haiku)
        cost_sonnet = engine.compute_llm_cost(default_traffic, q_sonnet)
        assert cost_haiku < cost_sonnet

    def test_semantic_cache_reduces_cost(
        self,
        engine: CapacityEngine,
        default_traffic: TrafficProfile,
        default_query: RAGQueryProfile,
    ) -> None:
        baseline = engine.compute_llm_cost(default_traffic, default_query)
        opt = OptimizationConfig(semantic_cache_hit_rate=0.20)
        optimized = engine.compute_llm_cost(default_traffic, default_query, opt)
        assert optimized < baseline
        expected_reduction = 0.20
        actual_reduction = 1.0 - optimized / baseline
        assert abs(actual_reduction - expected_reduction) < 0.01

    def test_zero_optimization_equals_baseline(
        self,
        engine: CapacityEngine,
        default_traffic: TrafficProfile,
        default_query: RAGQueryProfile,
    ) -> None:
        opt = OptimizationConfig()  # all zeros
        assert engine.compute_llm_cost(
            default_traffic, default_query, opt
        ) == pytest.approx(
            engine.compute_llm_cost(default_traffic, default_query), rel=1e-6
        )


class TestStorage:
    def test_vector_store_scales_with_tenants(
        self, engine: CapacityEngine
    ) -> None:
        c1 = CorpusProfile(num_tenants=100, max_corpus_per_tenant_gb=100.0)
        c2 = CorpusProfile(num_tenants=200, max_corpus_per_tenant_gb=100.0)
        vs1, _ = engine.compute_storage(c1)
        vs2, _ = engine.compute_storage(c2)
        assert abs(vs2 / vs1 - 2.0) < 0.01

    def test_vector_store_is_smaller_than_raw_text(
        self,
        engine: CapacityEngine,
        default_corpus: CorpusProfile,
    ) -> None:
        """
        Sanity check: a 1536-dim float32 vector is 6144 bytes.
        At 512 tokens/chunk and 1M tokens/GB, that's ~12GB of vectors per 1GB text.
        But with only 512-token chunks, vector store should be in reasonable range of raw text.
        """
        vs_tb, obj_tb = engine.compute_storage(default_corpus)
        assert vs_tb > 0
        assert obj_tb > 0
```

### Implementation Notes

**Order of implementation:**

1. `models.py` — read carefully, understand each property. No code to write here.
2. `engine.py` — implement in this order: `compute_llm_latency_ms` → `compute_latency_breakdown` → `compute_token_throughput` → `compute_llm_cost` → `compute_eval_cost` → `compute_storage` → `run`.
3. Run tests after each method: `pytest tests/test_engine.py -v -k TestLLMLatency` etc.
4. `cli.py` — implement `format_report` last; it is pure presentation logic.

**Common mistakes to avoid:**

- Forgetting to convert bytes to terabytes in `compute_storage` (divide by 1e12).
- Applying optimization factors in the wrong order (semantic cache removes entire LLM call; apply it first, then apply the remaining optimizations to the non-cached fraction).
- Using `print` instead of `logger.info` in the engine (the engine is a library; it must not write to stdout directly).

---

## Part 2: Assessment Preparation (1 hour)

### Week 6 Review Checklist

Use this checklist to verify your understanding before the assessment. For each item, explain it aloud in one or two sentences without looking at notes.

**Observability (Day 1–2):**
- [ ] Difference between metrics, traces, and logs — and when to use each.
- [ ] What a span is in distributed tracing, and how spans form a trace.
- [ ] How to calculate cost per LLM call from usage data in traces.
- [ ] What a golden signal is and why you track all four.

**Scalability (Day 3):**
- [ ] Horizontal vs. vertical scaling — definition and when each applies.
- [ ] How a load balancer decides where to route a request.
- [ ] What a semantic cache is and why it is different from a KV cache.
- [ ] How sharding works in vector databases.

**Multi-tenancy and SLOs (Day 4):**
- [ ] Three multi-tenancy patterns and their tradeoffs.
- [ ] Why namespace isolation must be enforced at query time.
- [ ] Error budget definition and how burn rate alerts work.
- [ ] Difference between P95 and P99 latency and why P99 matters more for SLOs.

**Eval and Monitoring (Day 5):**
- [ ] Online eval vs. offline eval vs. shadow scoring.
- [ ] LLM-as-judge: what faithfulness measures, what relevance measures.
- [ ] Three signals that indicate embedding drift.
- [ ] Circuit breaker states: CLOSED → OPEN → HALF-OPEN.

**Platform Design (Day 6):**
- [ ] Five phases of the system design framework.
- [ ] How to estimate monthly LLM cost from QPS and token counts.
- [ ] What an ADR contains and why it matters.
- [ ] At least three cost optimization levers and their estimated impact.

---

### Common Interview Anti-patterns for AI System Design

Interviewers at staff and senior engineer levels penalize these behaviors. Recognize them and actively avoid them.

**Anti-pattern 1: Jumping to implementation before requirements.**

Candidate starts drawing boxes immediately after hearing the problem statement. The interviewer has to stop them to ask about scale and constraints.

Why it fails: the design will be wrong for the actual requirements. Interviewers are explicitly testing whether you gather requirements first.

What to do instead: spend the first 5 minutes asking clarifying questions. Do not draw anything until you have the QPS target, the latency SLO, and the data isolation requirements.

---

**Anti-pattern 2: Ignoring failure modes.**

Candidate designs the happy path and stops. No discussion of what happens when the vector DB is slow, the LLM API times out, or a tenant uploads a malformed PDF.

Why it fails: production systems spend 80% of engineering time on failure handling. A design with no failure modes is a toy.

What to do instead: after each major component, explicitly ask "what happens when this fails?" and describe the circuit breaker, fallback, or retry strategy.

---

**Anti-pattern 3: No numbers.**

Candidate says "we need a scalable database" without computing how many bytes, how many QPS, or how many tokens/second.

Why it fails: "scalable" is meaningless without numbers. Interviewers want to see that you can estimate.

What to do instead: always do back-of-envelope math on paper during the interview. Peak QPS → token throughput → storage sizing. It does not need to be exact; it needs to be in the right order of magnitude with clear assumptions stated.

---

**Anti-pattern 4: Single-option thinking.**

Candidate proposes one approach without acknowledging alternatives or tradeoffs.

Why it fails: good engineers know there is no single correct answer. Every design decision involves tradeoffs. Presenting only one option suggests you have not thought through alternatives.

What to do instead: for every major decision, briefly state two alternatives and explain why you chose your approach. "I chose Qdrant over Pinecone because we need self-hosting for data sovereignty; if we were cloud-only I would use Pinecone for reduced ops burden."

---

**Anti-pattern 5: Ignoring cost.**

Candidate designs a system with 10 GPU clusters per tenant and never mentions cost.

Why it fails: enterprise AI systems must be cost-modeled. If your design costs $10M/month to run, it is a bad design regardless of its technical elegance.

What to do instead: include a rough cost estimate. It does not need to be precise — showing you thought about it is what matters.

---

**Anti-pattern 6: Treating LLM calls as instantaneous.**

Candidate draws a sequence diagram where the LLM call appears as a thin arrow with no latency annotation.

Why it fails: LLM calls are the dominant latency component in AI systems. Ignoring this produces infeasible designs where the total latency "works out" but the LLM call alone blows the budget.

What to do instead: explicitly budget your P95 latency target across components. Show that the LLM call fits within the remaining budget after subtracting retrieval, embedding, and network overhead.

---

### Mock Question Walk-through: "Design a Production RAG System"

**Prompt (typical wording)**: "Design a production RAG system for a Fortune 500 enterprise with 50 business units that need to query their internal knowledge base."

Work through this question using the five-phase framework. Below is a model answer outline — not a script to memorize, but a structure to internalize.

**Phase 1 — Requirements (ask these questions):**
- How many concurrent users and what is the peak QPS?
- What is the acceptable response latency (P95)?
- Must business units be isolated from each other's data?
- What document types? What is the maximum corpus size?
- Is there a compliance requirement (SOC2, GDPR, HIPAA)?

**Phase 2 — Capacity (compute these numbers):**
- Assume: 500 QPS peak, 100GB corpus per business unit.
- Token throughput: 500 × 2,050 = 1.025M input tokens/sec.
- Vector store: 50 units × 100GB × ~12GB vectors/GB text × 3x replication ≈ 18TB.
- Monthly cost at 30% avg load: ~$4M for LLM API (before optimization).

**Phase 3 — High-level design (draw these components):**
- API Gateway → Tenant Auth → Semantic Cache → Query Pipeline → LLM API.
- Ingestion: S3 → Queue → Worker (PII redact → chunk → embed → write to Qdrant namespace).
- Observability: OTel traces → Prometheus → Grafana + eval pipeline.

**Phase 4 — Deep dives (pick 2):**
- Latency budget: embedding 30ms + ANN 50ms + rerank 100ms + LLM 1,600ms + overhead 200ms = 1,980ms P95 ✓
- Multi-tenancy: Qdrant namespace per business unit, enforced at query time. RLS in PostgreSQL for metadata.

**Phase 5 — Tradeoffs:**
- Managed API vs. self-hosted: managed API wins unless GDPR/HIPAA requires on-prem.
- Namespace vs. cluster isolation: namespace at 50 units; revisit at 10,000.
- Reranker: cross-encoder improves quality 15–25%. Adds 100ms to latency. Worth it if P95 budget allows.

---

### Mock Question Walk-through: "Design an LLM Serving Layer"

**Prompt**: "Your company wants to serve open-weight LLMs internally for cost reasons. Design the LLM serving layer."

**Requirements to clarify:**
- What model sizes? (7B, 13B, 70B — determines GPU requirements)
- What latency SLO for first-token and for full completion?
- Batch or interactive workloads?
- Multi-tenant or single-tenant serving?

**Capacity math:**
```
70B FP16 model weight size:
  70B params × 2 bytes/param = 140GB
  Requires: 2× A100 80GB with tensor parallelism

Throughput per 2×A100 node (continuous batching):
  ~2,000 output tokens/second

Required nodes for 150,000 output tokens/second:
  150,000 / 2,000 = 75 nodes = 150 A100s

At $3/hr per A100 (spot pricing):
  150 × $3 × 24 × 30 = $324,000/month
  Compare: Anthropic API at same load ≈ $4M+/month
  Break-even: self-hosted is 12x cheaper at this scale —
  but requires significant GPU ops expertise.
```

**Key components:**
- vLLM or TGI as serving backend (continuous batching, PagedAttention).
- Load balancer with queue depth-aware routing (route to least-loaded replica).
- Autoscaling based on request queue depth, not CPU/memory.
- Model artifact stored in object storage; loaded into GPU memory on worker startup.
- Canary deploy: update model weights on 5% of workers first, compare P95 latency and quality.

**What good vs. great looks like:**

| Dimension            | Good answer                                | Great answer                                      |
|----------------------|--------------------------------------------|---------------------------------------------------|
| Requirements         | Asks about QPS and latency                 | Also asks about batch vs. interactive, TP/PP needs|
| Capacity             | Estimates GPU count                        | Shows FP16 memory math; mentions quantization options (INT8, AWQ) |
| Failure handling     | Describes health checks                    | Describes circuit breaker, graceful degradation to smaller model |
| Cost                 | Mentions GPU cost                          | Compares spot vs. reserved; shows break-even with API |
| Tradeoffs            | Notes self-hosted vs. managed              | Discusses latency impact of tensor parallelism overhead |
| Monitoring           | Mentions metrics                           | Specifies: TTFT (time to first token), throughput (tok/sec), KV cache utilization, GPU memory fragmentation |

---

## Exercises

1. **Complete the capacity engine**: Implement all `TODO` methods in `engine.py`. All tests in `test_engine.py` must pass. Then run the CLI: `python -m capacity_cost_model.cli --peak-qps 500 --optimize` and verify the output is reasonable (monthly cost in the range of $1–5M, vector store in the range of 1–10TB).

2. **Mock interview self-practice**: Set a 45-minute timer. Open a blank document. Write out your complete response to this prompt: "Design a multi-tenant enterprise AI assistant with RAG and agents for a financial services company subject to GDPR." Cover all five phases. After the timer stops, review against the anti-pattern list and the week-review checklist. Note which areas you rushed or skipped.

---

## Self-check

- What is the memory footprint in gigabytes of a 70B parameter model in FP16?
- Which cost optimization lever typically has the highest impact on LLM API spend?
- Name the five anti-patterns that interviewers penalize in AI system design interviews.
- What is TTFT and why is it the most user-perceptible latency metric for LLM serving?
- What does "good vs. great" look like for the failure handling dimension of an LLM serving design?
- You have 45 minutes in a system design interview. How should you allocate time across the five phases?
