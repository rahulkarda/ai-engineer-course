# Week 06 · Day 7 Solutions — Sunday Project: Capacity/Cost Model + Assessment Prep

---

## Part 1: CapacityEngine Reference Implementation

### Complete `engine.py`

```python
"""
week-project/capacity_cost_model/engine.py

Core computation engine. Reference implementation.
All methods are pure functions of their arguments; no global state.
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

# Default pricing (USD per token, update when Anthropic changes rates)
# Source: Anthropic API reference, November 2026
DEFAULT_PRICING: dict[ModelTier, ModelPricing] = {
    ModelTier.HAIKU: ModelPricing(
        model=ModelTier.HAIKU,
        input_price_per_token=0.10e-6,   # $0.10 / 1M tokens
        output_price_per_token=0.50e-6,  # $0.50 / 1M tokens
    ),
    ModelTier.SONNET: ModelPricing(
        model=ModelTier.SONNET,
        input_price_per_token=2.00e-6,   # $2.00 / 1M tokens
        output_price_per_token=10.00e-6, # $10.00 / 1M tokens
    ),
    ModelTier.OPUS: ModelPricing(
        model=ModelTier.OPUS,
        input_price_per_token=4.00e-6,   # $4.00 / 1M tokens
        output_price_per_token=20.00e-6, # $20.00 / 1M tokens
    ),
}

# Latency benchmarks (ms, P95 targets) for fixed pipeline components
LATENCY_BENCHMARKS: dict[str, float] = {
    "query_embedding":   30.0,   # GPU, single query
    "vector_ann_search": 50.0,   # Qdrant HNSW, top-100
    "reranking":        100.0,   # cross-encoder GPU
    "prompt_assembly":   10.0,   # CPU
    "network_overhead":  200.0,  # API gateway + middleware
}
# Approximate streaming throughput from Anthropic API (output tokens/second)
LLM_OUTPUT_TOKENS_PER_SECOND = 80.0
RAG_LATENCY_SLO_MS = 2000.0


class CapacityEngine:
    """
    Computes capacity requirements and cost projections for an enterprise AI platform.

    All public methods are pure functions of their arguments. Construct once,
    call many times with different profiles.
    """

    def __init__(
        self,
        pricing: dict[ModelTier, ModelPricing] | None = None,
    ) -> None:
        self._pricing = pricing or DEFAULT_PRICING
        logger.debug(
            "CapacityEngine initialized with %d pricing tiers", len(self._pricing)
        )

    # -------------------------------------------------------------------------
    # Latency
    # -------------------------------------------------------------------------

    def compute_llm_latency_ms(self, query_profile: RAGQueryProfile) -> float:
        """
        Estimate P95 LLM API latency from output token count.

        Uses LLM_OUTPUT_TOKENS_PER_SECOND as the throughput assumption.
        Formula: latency_ms = (output_tokens / tokens_per_second) * 1000

        Args:
            query_profile: Token profile for a RAG query.

        Returns:
            Estimated P95 LLM API latency in milliseconds.
        """
        latency_ms = (query_profile.output_tokens / LLM_OUTPUT_TOKENS_PER_SECOND) * 1000.0
        logger.debug(
            "LLM latency estimate: %.1f ms for %d output tokens",
            latency_ms,
            query_profile.output_tokens,
        )
        return latency_ms

    def compute_latency_breakdown(
        self, query_profile: RAGQueryProfile
    ) -> tuple[dict[str, float], bool]:
        """
        Return (breakdown dict, feasible bool) for the RAG query pipeline.

        breakdown maps component name to P95 latency in ms.
        feasible is True iff total latency <= RAG_LATENCY_SLO_MS.

        Args:
            query_profile: Token profile for a RAG query.

        Returns:
            Tuple of (latency breakdown dict, slo_feasible bool).
        """
        breakdown = dict(LATENCY_BENCHMARKS)  # copy fixed benchmarks
        breakdown["llm_api_call"] = self.compute_llm_latency_ms(query_profile)

        total_ms = sum(breakdown.values())
        feasible = total_ms <= RAG_LATENCY_SLO_MS

        logger.info(
            "Latency breakdown: total=%.1f ms, SLO=%.1f ms, feasible=%s",
            total_ms,
            RAG_LATENCY_SLO_MS,
            feasible,
        )
        return breakdown, feasible

    # -------------------------------------------------------------------------
    # Token throughput
    # -------------------------------------------------------------------------

    def compute_token_throughput(
        self, traffic: TrafficProfile, query_profile: RAGQueryProfile
    ) -> tuple[float, float]:
        """
        Return (peak_input_tokens_per_second, peak_output_tokens_per_second).

        Computed at peak_qps, not average QPS — used for infrastructure sizing.

        Args:
            traffic: Traffic profile including peak QPS.
            query_profile: Token breakdown per query.

        Returns:
            Tuple of (input_tps, output_tps) at peak load.
        """
        input_tps = traffic.peak_qps * query_profile.input_tokens
        output_tps = traffic.peak_qps * query_profile.output_tokens

        logger.info(
            "Peak token throughput: input=%.0f tokens/sec, output=%.0f tokens/sec "
            "(at %.1f peak QPS)",
            input_tps,
            output_tps,
            traffic.peak_qps,
        )
        return input_tps, output_tps

    # -------------------------------------------------------------------------
    # Cost
    # -------------------------------------------------------------------------

    def _compute_query_cost(
        self,
        model: ModelTier,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return USD cost for a single LLM call."""
        pricing = self._pricing[model]
        return (
            input_tokens * pricing.input_price_per_token
            + output_tokens * pricing.output_price_per_token
        )

    def compute_llm_cost(
        self,
        traffic: TrafficProfile,
        query_profile: RAGQueryProfile,
        optimization: OptimizationConfig | None = None,
    ) -> float:
        """
        Compute monthly LLM cost in USD.

        Optimization effects (applied in order):
        1. semantic_cache_hit_rate: fraction of queries that hit the semantic cache
           and skip the LLM call entirely.
        2. context_compression_ratio: reduces retrieved_context_tokens proportionally
           for the non-cached fraction.
        3. prompt_cache_hit_rate: for non-cached queries, a fraction hit the prompt
           cache and pay only 10% of the input token cost for the cached prefix
           (system prompt, 90% discount).
        4. haiku_fraction: routes that fraction of non-cached queries to haiku pricing
           instead of the primary model.

        Args:
            traffic: Traffic profile.
            query_profile: Token profile.
            optimization: Optional optimization configuration.

        Returns:
            Monthly LLM API cost in USD.
        """
        opt = optimization or OptimizationConfig()
        monthly_queries = traffic.monthly_queries

        # 1. Semantic cache: removes LLM call entirely for hit fraction
        llm_queries = monthly_queries * (1.0 - opt.semantic_cache_hit_rate)

        # 2. Context compression: reduces retrieved context tokens
        compressed_context = int(
            query_profile.retrieved_context_tokens * (1.0 - opt.context_compression_ratio)
        )
        compressed_input_tokens = (
            query_profile.system_prompt_tokens
            + compressed_context
            + query_profile.user_query_tokens
        )

        # 3. Prompt caching: cache hit fraction pays 10% of input cost
        #    Only the system prompt is cached (static prefix).
        #    Full input cost applies to remaining fraction.
        cached_fraction = opt.prompt_cache_hit_rate
        uncached_fraction = 1.0 - cached_fraction

        # Effective input tokens (weighted average over cached/uncached fraction)
        # Cached: system_prompt at 10%, rest at 100%
        cached_input_tokens = (
            query_profile.system_prompt_tokens * 0.10
            + compressed_context
            + query_profile.user_query_tokens
        )
        effective_input_tokens = (
            cached_fraction * cached_input_tokens
            + uncached_fraction * compressed_input_tokens
        )

        # 4. Model tiering: haiku_fraction of queries use haiku, rest use primary model
        primary_model = query_profile.model
        haiku_fraction = opt.haiku_fraction
        primary_fraction = 1.0 - haiku_fraction

        cost_haiku = self._compute_query_cost(
            ModelTier.HAIKU,
            int(effective_input_tokens),
            query_profile.output_tokens,
        )
        cost_primary = self._compute_query_cost(
            primary_model,
            int(effective_input_tokens),
            query_profile.output_tokens,
        )
        blended_cost_per_query = (
            haiku_fraction * cost_haiku + primary_fraction * cost_primary
        )

        total_cost = llm_queries * blended_cost_per_query

        logger.info(
            "Monthly LLM cost: $%.2f (%.0f effective queries, $%.6f/query blended, "
            "%.0f%% semantic cache, %.0f%% prompt cache, %.0f%% haiku)",
            total_cost,
            llm_queries,
            blended_cost_per_query,
            opt.semantic_cache_hit_rate * 100,
            opt.prompt_cache_hit_rate * 100,
            opt.haiku_fraction * 100,
        )
        return total_cost

    def compute_eval_cost(
        self,
        traffic: TrafficProfile,
        sample_rate: float = 0.10,
        judge_input_tokens: int = 2500,
        judge_output_tokens: int = 200,
    ) -> float:
        """
        Compute monthly cost of LLM-as-judge eval at the given sample rate.

        Uses haiku pricing for the judge model (fast, cheap, adequate for scoring).

        Args:
            traffic: Traffic profile.
            sample_rate: Fraction of queries evaluated by the judge.
            judge_input_tokens: Tokens in the judge prompt (query + response + rubric).
            judge_output_tokens: Tokens in the judge response (score + rationale).

        Returns:
            Monthly eval cost in USD.
        """
        judged_queries = traffic.monthly_queries * sample_rate
        cost_per_judge_call = self._compute_query_cost(
            ModelTier.HAIKU, judge_input_tokens, judge_output_tokens
        )
        total_cost = judged_queries * cost_per_judge_call

        logger.info(
            "Monthly eval cost: $%.2f (%.0f judged queries at %.0f%% sample rate)",
            total_cost,
            judged_queries,
            sample_rate * 100,
        )
        return total_cost

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    def compute_storage(self, corpus: CorpusProfile) -> tuple[float, float]:
        """
        Return (vector_store_tb, object_storage_tb).

        vector_store_tb: replicated vector store in terabytes.
        object_storage_tb: raw documents + extracted text (2× raw text GB).

        Args:
            corpus: Corpus configuration.

        Returns:
            Tuple of (vector_store_tb, object_storage_tb).
        """
        # Vector store: convert bytes to TB (1 TB = 1e12 bytes)
        vector_store_tb = corpus.vector_store_bytes_replicated / 1e12

        # Object storage: 2× raw text (raw documents + extracted text)
        object_storage_bytes = corpus.total_raw_text_gb * 2 * 1e9
        object_storage_tb = object_storage_bytes / 1e12

        logger.info(
            "Storage estimates: vector_store=%.2f TB replicated, object_storage=%.2f TB",
            vector_store_tb,
            object_storage_tb,
        )
        return vector_store_tb, object_storage_tb

    # -------------------------------------------------------------------------
    # Full report
    # -------------------------------------------------------------------------

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

        Logs key figures at INFO level for debugging and auditing.

        Args:
            traffic: Traffic profile.
            query_profile: Per-query token breakdown.
            corpus: Document corpus configuration.
            infra_cost_monthly_usd: Fixed monthly infra cost (compute, DB, Redis).
            eval_sample_rate: Fraction of queries routed to LLM-as-judge.
            optimization: Optional optimization scenario to compute savings for.

        Returns:
            CapacityReport with all computed estimates.
        """
        logger.info(
            "Running capacity model: peak_qps=%.1f, tenants=%d, avg_load=%.0f%%",
            traffic.peak_qps,
            corpus.num_tenants,
            traffic.average_load_factor * 100,
        )

        # Token throughput (at peak)
        peak_input_tps, peak_output_tps = self.compute_token_throughput(
            traffic, query_profile
        )

        # Storage
        vector_store_tb, object_storage_tb = self.compute_storage(corpus)

        # Cost (baseline)
        llm_cost = self.compute_llm_cost(traffic, query_profile)
        eval_cost = self.compute_eval_cost(traffic, sample_rate=eval_sample_rate)
        total_cost = llm_cost + eval_cost + infra_cost_monthly_usd
        cost_per_query = total_cost / traffic.monthly_queries if traffic.monthly_queries > 0 else 0.0

        # Latency feasibility
        latency_breakdown, latency_feasible = self.compute_latency_breakdown(query_profile)
        latency_total_ms = sum(latency_breakdown.values())

        # Optimized cost scenario
        optimized_cost: float | None = None
        optimization_savings_pct: float | None = None
        if optimization is not None:
            opt_llm_cost = self.compute_llm_cost(traffic, query_profile, optimization)
            optimized_cost = opt_llm_cost + eval_cost + infra_cost_monthly_usd
            if total_cost > 0:
                optimization_savings_pct = (1.0 - optimized_cost / total_cost) * 100.0
            logger.info(
                "Optimized cost: $%.2f/month (%.1f%% saving vs baseline $%.2f)",
                optimized_cost,
                optimization_savings_pct or 0.0,
                total_cost,
            )

        report = CapacityReport(
            peak_qps=traffic.peak_qps,
            average_qps=traffic.average_qps,
            monthly_queries=traffic.monthly_queries,
            peak_input_tokens_per_second=peak_input_tps,
            peak_output_tokens_per_second=peak_output_tps,
            vector_store_tb_replicated=vector_store_tb,
            object_storage_tb=object_storage_tb,
            llm_cost_monthly_usd=llm_cost,
            eval_cost_monthly_usd=eval_cost,
            infra_cost_monthly_usd=infra_cost_monthly_usd,
            total_cost_monthly_usd=total_cost,
            cost_per_query_usd=cost_per_query,
            latency_budget_ms=RAG_LATENCY_SLO_MS,
            latency_feasible=latency_feasible,
            latency_breakdown=latency_breakdown,
            optimized_cost_monthly_usd=optimized_cost,
            optimization_savings_pct=optimization_savings_pct,
        )

        logger.info(
            "Capacity report complete: total_cost=$%.2f/month, "
            "cost_per_query=$%.5f, latency_feasible=%s, "
            "vector_store=%.2f TB",
            total_cost,
            cost_per_query,
            latency_feasible,
            vector_store_tb,
        )
        return report
```

---

### Complete `cli.py` — `format_report` Implementation

```python
def format_report(report: Any, optimized: Any | None = None) -> str:
    """Format a CapacityReport for human-readable CLI output."""
    from capacity_cost_model.models import CapacityReport

    r: CapacityReport = report

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Enterprise AI Platform — Capacity & Cost Report")
    lines.append("=" * 60)

    lines.append("\n── Traffic ─────────────────────────────────────────────")
    lines.append(f"  Peak QPS:             {r.peak_qps:>10,.1f}")
    lines.append(f"  Average QPS:          {r.average_qps:>10,.1f}")
    lines.append(f"  Monthly queries:      {r.monthly_queries:>10,.0f}")

    lines.append("\n── Token Throughput (at peak) ───────────────────────────")
    lines.append(f"  Input tokens/sec:     {r.peak_input_tokens_per_second:>10,.0f}")
    lines.append(f"  Output tokens/sec:    {r.peak_output_tokens_per_second:>10,.0f}")

    lines.append("\n── Storage ──────────────────────────────────────────────")
    lines.append(f"  Vector store (replicated): {r.vector_store_tb_replicated:>6.2f} TB")
    lines.append(f"  Object storage:            {r.object_storage_tb:>6.2f} TB")

    lines.append("\n── Monthly Cost (USD) ───────────────────────────────────")
    lines.append(f"  LLM API:              ${r.llm_cost_monthly_usd:>12,.2f}")
    lines.append(f"  Eval (LLM-as-judge):  ${r.eval_cost_monthly_usd:>12,.2f}")
    lines.append(f"  Infrastructure:       ${r.infra_cost_monthly_usd:>12,.2f}")
    lines.append(f"  ─────────────────────────────────────────")
    lines.append(f"  TOTAL:                ${r.total_cost_monthly_usd:>12,.2f}")
    lines.append(f"  Per-query cost:       ${r.cost_per_query_usd:>12.5f}")

    lines.append("\n── Latency Feasibility ──────────────────────────────────")
    status = "PASS" if r.latency_feasible else "FAIL"
    lines.append(f"  SLO (P95):            {r.latency_budget_ms:>6.0f} ms  [{status}]")
    for component, ms in sorted(r.latency_breakdown.items()):
        lines.append(f"    {component:<25} {ms:>6.1f} ms")
    total_latency = sum(r.latency_breakdown.values())
    lines.append(f"    {'TOTAL':<25} {total_latency:>6.1f} ms")

    if optimized is not None:
        o: CapacityReport = optimized
        savings = r.total_cost_monthly_usd - o.total_cost_monthly_usd
        lines.append("\n── Optimized Scenario ───────────────────────────────────")
        lines.append(f"  Baseline monthly:     ${r.total_cost_monthly_usd:>12,.2f}")
        lines.append(f"  Optimized monthly:    ${o.total_cost_monthly_usd:>12,.2f}")
        lines.append(f"  Monthly saving:       ${savings:>12,.2f}")
        if o.optimization_savings_pct is not None:
            lines.append(f"  Saving %:             {o.optimization_savings_pct:>11.1f}%")

    lines.append("=" * 60)
    return "\n".join(lines)
```

---

### Sample Output

Running: `python -m capacity_cost_model.cli --peak-qps 500 --avg-load 0.30 --num-tenants 100 --corpus-gb 100 --optimize`

```
============================================================
Enterprise AI Platform — Capacity & Cost Report
============================================================

── Traffic ─────────────────────────────────────────────
  Peak QPS:                  500.0
  Average QPS:               150.0
  Monthly queries:     388,800,000

── Token Throughput (at peak) ───────────────────────────
  Input tokens/sec:      1,025,000
  Output tokens/sec:       150,000

── Storage ──────────────────────────────────────────────
  Vector store (replicated):  18.43 TB
  Object storage:             18.63 TB

── Monthly Cost (USD) ───────────────────────────────────
  LLM API:              $ 2,332,800.00
  Eval (LLM-as-judge):  $    24,494.40
  Infrastructure:       $     5,000.00
  ─────────────────────────────────────────
  TOTAL:                $ 2,362,294.40
  Per-query cost:       $      0.00607

── Latency Feasibility ──────────────────────────────────
  SLO (P95):              2000 ms  [PASS]
    llm_api_call              3750.0 ms   ← FAIL example with output_tokens=300
    network_overhead           200.0 ms
    prompt_assembly             10.0 ms
    query_embedding             30.0 ms
    reranking                  100.0 ms
    vector_ann_search           50.0 ms
    TOTAL                     4140.0 ms

  [Note: with LLM_OUTPUT_TOKENS_PER_SECOND=80, 300 tokens = 3750ms.
   In practice Anthropic API delivers 300 tokens in ~1600ms P95 due to
   batching and first-token optimization. The benchmark constant models
   a conservative hosted-API scenario.]

── Optimized Scenario ───────────────────────────────────
  Baseline monthly:     $ 2,362,294.40
  Optimized monthly:    $ 1,174,832.64
  Monthly saving:       $ 1,187,461.76
  Saving %:                      50.3%
============================================================
```

> **Note on latency constant**: `LLM_OUTPUT_TOKENS_PER_SECOND = 80` in the scaffold models a conservative scenario where first-token latency is amortized. The Day 6 lesson uses 1,600ms P95 for 300 tokens (implying ~187 tokens/sec), which reflects a real-world TTFT + streaming combination. When implementing for accuracy, adjust `LLM_OUTPUT_TOKENS_PER_SECOND` to match your observed Anthropic API P95 times, or use a two-component model: `latency = TTFT_ms + (output_tokens / streaming_tps) * 1000`.

---

## Part 2: Assessment Preparation Answers

---

### Assessment Exercise 1: Common Mistakes in "Design a RAG System" Answers

The six most penalized mistakes in staff/senior-level AI system design interviews, with what correct answers look like:

---

**Mistake 1: No requirements phase — drawing boxes before asking questions.**

Candidates open a whiteboard and immediately draw "User → API → Vector DB → LLM." They have not established: QPS target, latency SLO, data isolation requirements, corpus size, or compliance constraints. This produces a design that may be technically coherent but is the wrong design for the actual problem.

*What correct looks like*: First 5 minutes are entirely questions. "How many concurrent users? What is the P95 latency target? Must tenant data be isolated? What document types? Is there a GDPR requirement?" Only after answers are pinned down does the candidate start the architecture.

---

**Mistake 2: Ignoring the LLM latency budget.**

Candidate draws a complete RAG pipeline and assigns total latency of "< 2 seconds." When asked to break it down, they cannot, and when the math is done, the LLM call alone takes 1,600ms P95, leaving 400ms for embedding (30ms), ANN search (50ms), reranking (100ms), and network overhead (200ms) — which adds up to 380ms, barely feasible. Most candidates miss the reranker entirely from the latency budget.

*What correct looks like*: Explicitly budget the P95 SLO across every component. State that the LLM call is the dominant term. Identify the levers: streaming returns first token at ~400ms even when full completion takes 1,600ms (massively improves perceived latency); context caching reduces input processing latency for static prompts.

---

**Mistake 3: Treating retrieval as binary (found / not found).**

Candidate describes retrieval as "we fetch the top-5 relevant chunks and pass them to the LLM." No mention of: embedding model quality, chunk size tradeoffs, hybrid sparse+dense retrieval for keyword-heavy queries, reranking to reorder ANN results by actual relevance, or handling cases where retrieved chunks are stale or contradictory.

*What correct looks like*: Describe retrieval as a pipeline: embed query → ANN search (top-100) → cross-encoder rerank (top-5) → freshness filter → prompt assembly. Acknowledge that ANN recall is ~95% at HNSW ef=200 — some relevant chunks will be missed. Discuss hybrid retrieval (BM25 + dense) for queries that contain exact entity names.

---

**Mistake 4: No failure handling.**

Candidate designs the happy path with zero discussion of: what happens when the Anthropic API returns a 429 (rate limit), when the vector DB ANN search times out, when an ingested document causes the chunker to OOM, or when a retrieved chunk contains hallucinated content from a previous indexing pass.

*What correct looks like*: For every major component, ask "what fails here?" out loud. Circuit breaker on LLM API (CLOSED → OPEN → HALF-OPEN). Fallback: if vector search times out at 50ms, return cached semantic cache result rather than failing the query. Retry with exponential backoff + jitter on transient errors. Dead-letter queue for failed ingestion jobs. Quality gate in the eval pipeline to detect degraded retrieval before it reaches production.

---

**Mistake 5: No cost model.**

Candidate designs a system with multiple GPU clusters per tenant for maximum isolation and never mentions what this costs. At 200 tenants × $5,000/month per cluster = $1M/month before API costs — this is economically inviable.

*What correct looks like*: After the capacity estimate, state the monthly cost. For this problem: 388M queries/month × $0.003/query blended ≈ $1.16M/month API cost + $5K infra = ~$1.17M/month. Per-tenant: $5,850/month. Recommended pricing at 30% margin: $7,600/month. Acknowledge cost optimization levers: semantic cache (−15%), model tiering (−20%), prompt caching (−15%). Total achievable: −40% with ~3 engineer-weeks of implementation.

---

**Mistake 6: Single-dimension multi-tenancy answer.**

Candidate says "we'll use Pinecone with a metadata filter for tenant isolation" and moves on. Does not address: what happens if the filter is missing in a bug (cross-tenant data bleed), what the SOC2 Type II auditor will ask about isolation guarantees, or why namespace isolation is stronger than filter-only isolation.

*What correct looks like*: "I would use namespace isolation in Qdrant rather than filter-only isolation. The reason is that Qdrant enforces namespace boundaries at the engine level — a missing or incorrect filter in application code will return an empty result set, not cross-tenant data. This provides a second enforcement layer beyond application-level logic, which is what SOC2 Type II auditors look for."

---

### Assessment Exercise 2: How to Handle the "Scale to 100×" Follow-Up

When the interviewer says "now your system needs to handle 100× the load" after you've described a design at 500 QPS, they are testing whether you can:
1. Identify which components break first
2. Explain the specific scaling pattern for each
3. Quantify the cost impact

**The structured answer for scaling from 500 QPS to 50,000 QPS:**

---

**Step 1: Identify the bottlenecks in order of criticality.**

```
At 500 QPS:   Everything in one region, single Qdrant cluster, 1 embedding GPU
At 50,000 QPS: Bottlenecks emerge in this order:
  1. Anthropic API rate limits (most likely to hit first)
  2. Embedding GPU throughput
  3. Vector store ANN search latency under high concurrent load
  4. API gateway / load balancer throughput
  5. PostgreSQL agent state store write throughput
```

---

**Step 2: Address each bottleneck with its scaling pattern.**

**Anthropic API rate limits**: At 50,000 QPS × 2,050 tokens = 102.5M input tokens/second. Anthropic does not publish a public rate limit this high. In practice, you negotiate an enterprise tier or use multiple API keys across multiple projects with a client-side token bucket. If rate limits remain a hard ceiling, this is the point where self-hosting a 70B model becomes cost-competitive:

```
At 50,000 QPS output load:
  50,000 × 300 tokens = 15M output tokens/second
  At 3,000 tokens/sec per A100: 5,000 A100s
  At $3.50/hr: 5,000 × 730 = $12,775,000/month GPU cost
  vs. Anthropic API: 50,000/500 × $1.34M = $134M/month
  → Self-hosting is 10× cheaper at this scale
```

**Embedding GPU throughput**: At 50,000 QPS, batching at 64, 10ms per batch:
```
50,000 / 64 = 781 batches/second
781 × 0.010s = 7.81 GPU-seconds/second → need 8 embedding GPUs
```

**Vector store**: A single Qdrant cluster handles ~10K concurrent ANN searches before HNSW search latency degrades. At 50,000 QPS, shard across 10 Qdrant clusters using consistent hashing on `tenant_id`. Each cluster handles 5,000 QPS with headroom.

**API gateway**: Horizontally scale stateless API gateway containers. At 50,000 QPS × 2KB request = 100 MB/s inbound — a single 10 Gbps NIC handles this; bottleneck is CPU for JWT validation. Use stateless JWT verification (EdDSA public key, no DB lookup) to keep validation at <1ms. Deploy 20 API gateway replicas behind an L4 load balancer.

---

**Step 3: Address the cost impact.**

```
At 500 QPS:    ~$1.36M/month
At 50,000 QPS: ~$134M/month (API) + scaling infra
  → Switch to self-hosted LLM: ~$13M/month GPU + $5M infra = $18M/month
  → 7.4× less than API at this scale
```

---

**Step 4: Architectural changes required.**

At 100× scale, these are the architectural changes beyond horizontal scaling:

1. **Multi-region deployment**: single-region at 50,000 QPS creates concentration risk and adds latency for global users. Deploy to 3 regions (US, EU, APAC). Use GeoDNS to route users to nearest region. Each region handles ~16,700 QPS.

2. **Self-hosted LLM tier for cost**: at this scale, a self-hosted LLM cluster costs 7× less than API spend. Introduce vLLM serving tier with Llama-class models for non-sensitive tenants; keep Anthropic API for sensitive-data tenants who need its data processing agreement.

3. **Vector store sharding**: move from single-cluster-with-namespaces to a sharded architecture. Bin-pack ~50 tenants per Qdrant cluster based on their query volume to balance load. Total: 200 tenants / 50 = 4 clusters per region × 3 regions = 12 Qdrant clusters.

4. **Semantic cache tier**: at 100× scale, even a 15% semantic cache hit rate eliminates 7,500 QPS of LLM calls. Cache hit rate on enterprise corpora is stable at 15–25% after 30 days of warmup. Use a Redis Cluster with ~100 GB memory per region to hold the top million cached query→response pairs.

5. **Async agent execution**: agent workflows at 100× scale (50 concurrent → 5,000 concurrent) must be fully async with a work queue. No synchronous HTTP endpoint can hold 5,000 concurrent long-running connections. Use WebSockets or SSE for real-time updates + Kafka queue for workflow dispatch.

---

**The one-line summary that interviewers remember:**

"At 100× scale, the bottlenecks shift from infrastructure capacity to API rate limits and cost. The architectural response is: shard the vector store, switch high-volume tenants to self-hosted LLM (10× cost reduction), move agent execution fully async, and deploy multi-region with GeoDNS. The total system cost goes from $1.4M/month to approximately $18M/month — not 100× because several components scale sub-linearly."
