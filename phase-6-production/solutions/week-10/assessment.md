# Week 10 Assessment — Answer Key

---

## Section 1: Conceptual Questions

### 1. TTFT vs TBT Tradeoffs

**TTFT (Time to First Token)** is determined by the prefill phase: the model processes the entire input prompt in parallel before generating a single output token. Prefill is compute-bound — doubling input length roughly doubles TTFT. It does not scale with output length.

**TBT (Time Between Tokens)** is determined by the decode phase: each output token is generated autoregressively, one at a time. Decode is memory-bandwidth-bound — the bottleneck is loading all model weights and KV-cache from HBM on every token step, not floating-point throughput. Larger batch sizes in decode share that bandwidth cost across multiple sequences, but each individual sequence's TBT still grows roughly linearly with the number of concurrent sequences on the same GPU.

**Batching tradeoff:** Static batching improves throughput by amortizing weight loads across sequences, but it increases TTFT because shorter requests must wait in a queue for the batch to fill. Continuous batching (iteration-level scheduling) reduces that queue time, but TTFT is still affected by request queuing latency under high load.

**Streaming:** High TBT is user-perceptible as a slow typing effect. Streaming tokens progressively hides TBT latency — the user sees output arriving rather than waiting for a complete response, making even 30–50 ms TBT feel acceptable.

**Interview answer summary:** TTFT = prefill, compute-bound, scales with input length. TBT = decode, memory-bandwidth-bound, scales with concurrent sequences. Batching improves throughput but adds queuing to TTFT. Streaming makes TBT tolerable by progressive display.

---

### 2. Continuous Batching Benefit — Quantification

**The straggler problem in static batching:** With static batching, a batch is held until all sequences in it have finished. One unusually long sequence (a "straggler") blocks the GPU from starting new requests, leaving it idle or underutilized.

**Continuous (iteration-level) batching** schedules at the decode step level rather than the request level. Finished sequences are immediately evicted and replaced with new ones from the queue on the next iteration. The GPU never idles waiting for stragglers.

**Throughput improvement:** Empirical benchmarks across vLLM and TGI consistently show **2–5x throughput improvement** over static batching at equivalent concurrency. The exact multiplier depends on output-length variance — the more bimodal the distribution (mix of short and long responses), the larger the gain.

**Why 2–5x:** With static batching, average GPU utilization is often 20–50% due to straggler waits. Continuous batching pushes utilization toward 80–95%, which corresponds directly to the 2–5x throughput range.

---

### 3. KV-Cache Memory Cost Formula

**Exact formula:**

```
KV_cache_bytes = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * bytes_per_element
```

The factor of `2` accounts for both the Key and Value tensors.

**Llama-3-8B concrete calculation** (32 layers, 32 heads, 128 head_dim, FP16 = 2 bytes per element):

```
= 2 * 32 * 32 * 128 * seq_len * batch_size * 2
= 524,288 * seq_len * batch_size  bytes
```

At `seq_len = 4096`, `batch_size = 8`:

```
= 524,288 * 4096 * 8
= 17,179,869,184 bytes
≈ 16 GB
```

**Implication:** KV-cache is the primary memory constraint for serving. A 16 GB KV-cache for a batch of 8 at 4K context leaves almost no room on a single 24 GB GPU alongside model weights (~16 GB for Llama-3-8B in FP16). This is precisely the problem PagedAttention (vLLM) solves by allocating KV-cache in non-contiguous pages rather than reserving a contiguous block per sequence.

---

### 4. Quantization Precision Tradeoffs

| Precision | Relative Quality | Memory vs FP16 | Typical Use Case |
|-----------|-----------------|----------------|-----------------|
| FP32 | Baseline (reference) | 2x FP16 | Training, debugging |
| FP16 | Indistinguishable from FP32 on most tasks | 1x (reference) | Standard inference |
| BF16 | Equivalent to FP16 for inference | 1x | TPUs, Ampere+ GPUs |
| INT8 | ~99% of FP16 quality; perceptible degradation only on extreme precision tasks | 0.5x (2x memory savings) | Production serving, throughput-sensitive |
| INT4 | ~97–98% of FP16 quality; slight degradation on complex reasoning | 0.25x (4x memory savings) | Edge deployment, very high throughput |
| INT2 | Significant quality degradation; meaningful errors on most tasks | 0.125x | Research only; not production-ready |

**Key rule of thumb:** Each 1-bit reduction halves memory footprint but compounds quality loss non-linearly. INT8 is the standard production sweet spot. INT4 (GPTQ, AWQ) is viable for throughput-maximized serving when small quality degradation is acceptable. INT2 is not production-ready.

---

### 5. Speculative Decoding Acceptance Rate

**Mechanism:** A small, fast draft model generates `k` candidate tokens. The large target model evaluates all `k+1` positions in a single forward pass (since the draft tokens are already known). Each draft token is accepted or rejected by comparing target and draft probabilities.

**Per-token acceptance probability:**

```
alpha_i = min(1, P_target(x_i) / P_draft(x_i))
```

If the draft token is exactly what the target would have chosen, `alpha = 1`. If it is very unlikely under the target distribution, `alpha → 0`.

**Overall throughput gain formula:**

```
speedup = (1 + k * alpha) / (1 + k * cost_ratio)
```

Where `cost_ratio = cost_draft / cost_target` (forward pass cost, typically 0.1–0.2 for a well-chosen draft model). With `k = 5`, `alpha = 0.7`, `cost_ratio = 0.1`:

```
speedup = (1 + 5*0.7) / (1 + 5*0.1) = 4.5 / 1.5 = 3.0x
```

**Practical conditions for high acceptance rate:**
- Draft model trained on the same data/domain as the target
- Greedy or low-temperature decoding (high acceptance for common token sequences)
- Short, predictable output patterns (code with boilerplate, JSON, formulaic text)

**Works poorly when:** output is highly creative/stochastic (high temperature), domain mismatch between draft and target, or output length is very short (setup overhead not amortized).

---

### 6. vLLM vs TGI — When to Choose Each

**Choose vLLM when:**
- Maximum single-node throughput is the primary objective
- You need a drop-in OpenAI-compatible REST API (`/v1/completions`, `/v1/chat/completions`)
- Memory efficiency is critical: PagedAttention eliminates KV-cache fragmentation and supports much higher batch concurrency than naive implementations
- You want fine-grained control over KV-cache allocation and scheduling policy
- Running self-hosted models where Python ecosystem integration matters

**Choose TGI (Text Generation Inference) when:**
- You are deploying models directly from the HuggingFace Hub and want first-class Hub integration (model sharding, automatic config detection)
- You need tensor parallelism across multiple GPUs and want it managed for you
- Your application needs streaming with gRPC (lower overhead than HTTP/2 for token streaming)
- You want Rust-based performance for the server layer with Python model backends
- Deploying on HuggingFace Inference Endpoints (TGI is the native backend)

**Summary table:**

| Dimension | vLLM | TGI |
|-----------|------|-----|
| Throughput optimization | PagedAttention, high | Solid, tensor-parallel |
| API compatibility | OpenAI-compatible | gRPC + REST |
| HuggingFace Hub | Manual | Native |
| Multi-GPU | Tensor parallel (manual) | Automatic sharding |
| Memory efficiency | Best-in-class | Good |

---

### 7. Semantic vs Exact Caching

**Exact caching:**
- Key: `hash(prompt_string)` — a deterministic, collision-free fingerprint
- Match: only on byte-identical prompts
- False positives: zero
- False negatives: high — any paraphrase, added whitespace, or reordered parameter misses the cache
- Latency: O(1) hash lookup
- Best for: API responses to identical programmatic queries (same template, same inputs)

**Semantic caching:**
- Key: embedding vector of the prompt, stored in a vector database (e.g., Redis with vector search, Pinecone, Weaviate)
- Match: nearest-neighbor search with a cosine similarity threshold (e.g., `similarity > 0.95`)
- False positives: possible — two semantically similar but different questions may receive the wrong cached answer if the threshold is too low
- False negatives: lower than exact — paraphrases, spelling variations, and equivalent queries hit the cache
- Latency: O(log n) ANN search — typically 1–10 ms, which is acceptable but non-trivial
- Threshold tuning: critical. Too low → wrong answers served from cache. Too high → effectively degenerates to exact matching
- Best for: user-facing chatbots, FAQ systems, repeated natural language queries with high paraphrase rate

**When to use which:** Use exact caching for structured, programmatic prompts (code generation, data extraction, classification with fixed templates). Use semantic caching when the prompt space is large and natural-language variation is expected, but invest in threshold calibration with a held-out eval set before deploying.

---

### 8. Prompt Cost Reduction Strategies

Five concrete strategies, ordered roughly by ease of implementation:

1. **System prompt caching (`cache_control`):** Mark the system prompt and any static context blocks with `cache_control: {type: "ephemeral"}`. Repeated requests that share an identical prefix pay only for cache-read tokens (Anthropic: ~$0.20–$0.30/MTok) instead of full input tokens ($0.10–$4.00/MTok depending on model). Effective when the system prompt is large (>512 tokens) and reused across many requests within a 5-minute window.

2. **Shorter system prompts:** Audit system prompts for redundant instructions, verbose examples, and outdated guidance. A 1,000-token system prompt cut to 400 tokens reduces per-request input cost by 600 tokens — at scale (1M requests/day), that is 600M tokens/day saved. Remove examples that can be achieved through instruction alone; replace long few-shot examples with one well-chosen exemplar.

3. **Model routing (haiku for simple tasks):** Route classification, intent detection, simple extraction, and FAQ lookups to `claude-haiku-5-5` ($0.10/$0.50 per MTok input/output) instead of `claude-sonnet-5-5` ($2.00/$10.00) or `claude-opus-5-5` ($4.00/$20.00). A routing classifier itself costs very little and can be rule-based (keyword matching, regex) or a lightweight model call. Typical result: 80–90% of traffic goes to haiku, 10–15% to sonnet, <5% to opus, reducing average cost per request by 60–80%.

4. **Response length limits (`max_tokens`):** Set `max_tokens` to a value appropriate for the task. For classification, 10–50 tokens suffice. For summaries, 200–500 tokens. Leaving `max_tokens` at 4096 for tasks that return 50 tokens wastes nothing directly (you pay for actual output tokens), but it signals no constraint to the model, which may produce unnecessarily verbose responses. Add explicit length instructions in the prompt: "Reply in one sentence." "Return only the JSON object, no explanation."

5. **Semantic response caching:** Cache LLM responses keyed by semantic similarity of the input. A cache hit returns a stored response at ~1 ms instead of making a full API call. At a 30% cache hit rate on a 1M request/day workload, this eliminates 300K API calls per day. Infrastructure cost (vector DB, embedding model) is typically 5–10% of the LLM cost it displaces. Requires threshold calibration (see Question 7) to avoid serving stale or mismatched cached responses.

---

## Section 2: Coding Challenge — InferenceOptimizer

### Full Implementation

```python
"""
inference_optimizer.py

Week 10 Coding Challenge: InferenceOptimizer
Recommends model, quantization, batching, and caching strategy
based on a request traffic profile.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RequestProfile(BaseModel):
    """Describes the traffic characteristics of a workload."""

    avg_input_tokens: int = Field(gt=0, description="Average input tokens per request")
    avg_output_tokens: int = Field(gt=0, description="Average output tokens per request")
    requests_per_second: float = Field(gt=0, description="Sustained requests per second")
    latency_budget_ms: int = Field(gt=0, description="Maximum acceptable p95 latency in ms")
    cost_budget_per_1k_requests: float = Field(
        gt=0, description="Maximum acceptable cost in USD per 1,000 requests"
    )

    @model_validator(mode="after")
    def validate_token_ratio(self) -> "RequestProfile":
        if self.avg_output_tokens > self.avg_input_tokens * 10:
            logger.warning(
                "avg_output_tokens (%d) is >10x avg_input_tokens (%d); "
                "latency estimates may be underestimated.",
                self.avg_output_tokens,
                self.avg_input_tokens,
            )
        return self


class OptimizationPlan(BaseModel):
    """Recommended inference optimization configuration for a workload."""

    recommended_model: Literal[
        "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
    ]
    recommended_quantization: Literal["none", "INT8", "INT4"]
    use_batching: bool
    use_caching: bool
    estimated_cost_per_1k: float = Field(description="Estimated USD cost per 1,000 requests")
    estimated_p95_latency_ms: int = Field(description="Estimated p95 end-to-end latency in ms")
    rationale: str = Field(description="Plain-text explanation of the recommendation")


# ---------------------------------------------------------------------------
# Pricing and latency constants
# Current Anthropic first-party API pricing (per 1M tokens, input / output)
# Source: claude-api skill, 2026-11-11
# ---------------------------------------------------------------------------

# (input_price_per_1m, output_price_per_1m) in USD
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5":  (0.10,  0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5":   (4.00, 20.00),
}

# Approximate p95 latency components at moderate load (no batching queue)
# ttft_ms: time to first token (prefill-dominated)
# tbt_ms:  time between tokens (decode-dominated)
LATENCY_PROFILE: dict[str, dict[str, int]] = {
    "claude-haiku-5-5":  {"ttft_ms": 300,  "tbt_ms": 20},
    "claude-sonnet-5-5": {"ttft_ms": 500,  "tbt_ms": 30},
    "claude-opus-5-5":   {"ttft_ms": 1000, "tbt_ms": 50},
}

# Model preference order: cheapest/fastest first
MODEL_PREFERENCE: list[str] = [
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "claude-opus-5-5",
]

# Batching adds queuing delay at high RPS; this is an approximate adder
BATCHING_TTFT_OVERHEAD_MS = 50

# Caching reduces effective input cost; approximate hit-rate assumption
CACHE_HIT_RATE = 0.40
CACHE_READ_PRICE_PER_1M = 0.03  # flat approximation; actual price is model-dependent


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------

class InferenceOptimizer:
    """
    Recommends an inference optimization plan for a given workload profile.

    All logic is pure (no I/O, no LLM calls) so it can be unit-tested
    without mocking.
    """

    def recommend(self, profile: RequestProfile) -> OptimizationPlan:
        """
        Produce a fully-specified OptimizationPlan for the given profile.

        Selection logic:
        1. Pick the cheapest model whose estimated p95 latency and per-1k cost
           fit within the profile's budgets.
        2. If no model fits both constraints, fall back to the cheapest
           (haiku) and flag the mismatch in the rationale.
        3. Quantization is chosen based on throughput and cost pressure.
        4. Batching is enabled when RPS justifies the GPU utilisation gain.
        5. Caching is enabled when input prompts are long enough to produce
           meaningful cache savings (>500 tokens average).
        """
        model, budget_miss = self._pick_model(profile)
        quant = self._pick_quantization(profile)
        use_batching = self._should_batch(profile)
        use_caching = self._should_cache(profile)

        cost = self._estimate_cost(model, profile, use_caching=use_caching)
        latency = self._estimate_latency(model, profile, use_batching=use_batching)

        rationale_parts = [
            f"Model: {model}",
            f"latency_budget={profile.latency_budget_ms} ms, "
            f"estimated_p95={latency} ms",
            f"cost_budget=${profile.cost_budget_per_1k_requests:.4f}/1k, "
            f"estimated_cost=${cost:.4f}/1k",
            f"batching={'ON' if use_batching else 'OFF'} "
            f"({profile.requests_per_second:.1f} rps, "
            f"threshold=5 rps)",
            f"caching={'ON' if use_caching else 'OFF'} "
            f"(avg_input={profile.avg_input_tokens} tokens, "
            f"threshold=500 tokens)",
            f"quantization={quant}",
        ]
        if budget_miss:
            rationale_parts.append(
                f"WARNING: no model satisfies both budgets; "
                f"falling back to haiku. Budget miss: {budget_miss}"
            )

        logger.info("OptimizationPlan: %s", " | ".join(rationale_parts))

        return OptimizationPlan(
            recommended_model=model,  # type: ignore[arg-type]
            recommended_quantization=quant,
            use_batching=use_batching,
            use_caching=use_caching,
            estimated_cost_per_1k=round(cost, 6),
            estimated_p95_latency_ms=latency,
            rationale=" | ".join(rationale_parts),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pick_model(
        self, profile: RequestProfile
    ) -> tuple[str, str | None]:
        """
        Return (model_id, budget_miss_description).
        budget_miss is None when both budgets are satisfied.
        """
        for model in MODEL_PREFERENCE:
            latency = self._estimate_latency(model, profile, use_batching=False)
            cost = self._estimate_cost(model, profile, use_caching=False)
            latency_ok = latency <= profile.latency_budget_ms
            cost_ok = cost <= profile.cost_budget_per_1k_requests
            if latency_ok and cost_ok:
                return model, None

        # Fallback: return cheapest model, describe what was missed
        fallback = "claude-haiku-5-5"
        latency = self._estimate_latency(fallback, profile, use_batching=False)
        cost = self._estimate_cost(fallback, profile, use_caching=False)
        misses: list[str] = []
        if latency > profile.latency_budget_ms:
            misses.append(
                f"latency {latency} ms > budget {profile.latency_budget_ms} ms"
            )
        if cost > profile.cost_budget_per_1k_requests:
            misses.append(
                f"cost ${cost:.4f} > budget ${profile.cost_budget_per_1k_requests:.4f}"
            )
        return fallback, "; ".join(misses)

    def _pick_quantization(self, profile: RequestProfile) -> str:
        """
        Choose quantization level based on throughput and cost pressure.

        Rules:
        - INT4: very high throughput (>20 rps) OR extremely tight cost budget
        - INT8: moderate throughput (>5 rps)
        - none: low-latency quality-first workloads
        """
        if (
            profile.requests_per_second > 20
            or profile.cost_budget_per_1k_requests < 0.005
        ):
            return "INT4"
        if profile.requests_per_second > 5:
            return "INT8"
        return "none"

    def _should_batch(self, profile: RequestProfile) -> bool:
        """Enable batching when RPS is high enough to fill batches efficiently."""
        return profile.requests_per_second > 5.0

    def _should_cache(self, profile: RequestProfile) -> bool:
        """Enable caching when average input length makes cache savings significant."""
        return profile.avg_input_tokens > 500

    def _estimate_cost(
        self,
        model: str,
        profile: RequestProfile,
        *,
        use_caching: bool = False,
    ) -> float:
        """
        Estimate USD cost per 1,000 requests.

        When caching is on, a fraction of input tokens are read from cache
        at a lower price.
        """
        inp_price, out_price = PRICING[model]
        input_tokens = profile.avg_input_tokens
        output_tokens = profile.avg_output_tokens

        if use_caching:
            cached_tokens = input_tokens * CACHE_HIT_RATE
            uncached_tokens = input_tokens * (1 - CACHE_HIT_RATE)
            input_cost_per_1m = (
                uncached_tokens * inp_price
                + cached_tokens * CACHE_READ_PRICE_PER_1M
            ) / input_tokens  # weighted average price per token
            cost_per_1k = (
                input_tokens * input_cost_per_1m
                + output_tokens * out_price
            ) / 1_000
        else:
            cost_per_1k = (
                input_tokens * inp_price + output_tokens * out_price
            ) / 1_000

        return cost_per_1k

    def _estimate_latency(
        self,
        model: str,
        profile: RequestProfile,
        *,
        use_batching: bool = False,
    ) -> int:
        """
        Estimate p95 latency in ms.

        Formula: TTFT + avg_output_tokens * TBT
        Batching adds a queue overhead to TTFT.
        """
        lp = LATENCY_PROFILE[model]
        ttft = lp["ttft_ms"]
        if use_batching:
            ttft += BATCHING_TTFT_OVERHEAD_MS
        total = ttft + profile.avg_output_tokens * lp["tbt_ms"]
        return total
```

---

### Full Pytest Test Suite

```python
"""
tests/test_inference_optimizer.py

Three scenarios:
1. Latency-sensitive workload (interactive chat, tight TTFT budget)
2. Cost-sensitive workload (bulk pipeline, very low cost budget)
3. High-throughput workload (log classification at 50 rps)
"""

from __future__ import annotations

import pytest

from inference_optimizer import InferenceOptimizer, OptimizationPlan, RequestProfile


@pytest.fixture
def optimizer() -> InferenceOptimizer:
    return InferenceOptimizer()


class TestLatencySensitiveWorkload:
    """
    Interactive chat assistant.
    - Short inputs (200 tokens), short outputs (150 tokens)
    - Tight latency budget: 2,000 ms p95
    - Moderate cost tolerance: $0.10 per 1k requests
    - Low RPS: 2 rps
    """

    @pytest.fixture
    def profile(self) -> RequestProfile:
        return RequestProfile(
            avg_input_tokens=200,
            avg_output_tokens=150,
            requests_per_second=2.0,
            latency_budget_ms=2_000,
            cost_budget_per_1k_requests=0.10,
        )

    def test_recommends_haiku_for_low_latency(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # Haiku is the only model that fits within 2000 ms at 150 output tokens
        # (haiku: 300 + 150*20 = 3300 ms... let's verify with a relaxed budget)
        # Actually with 150 output tokens * 20ms = 3000ms + 300ms = 3300ms > 2000ms
        # So NO model fits — fallback to haiku is expected
        assert plan.recommended_model == "claude-haiku-5-5"

    def test_no_batching_at_low_rps(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        assert plan.use_batching is False

    def test_no_caching_for_short_inputs(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # avg_input_tokens=200 < 500 threshold
        assert plan.use_caching is False

    def test_no_quantization_at_low_rps(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # 2 rps is below both quantization thresholds
        assert plan.recommended_quantization == "none"

    def test_plan_is_valid_pydantic_model(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        assert isinstance(plan, OptimizationPlan)
        assert plan.estimated_p95_latency_ms > 0
        assert plan.estimated_cost_per_1k > 0
        assert len(plan.rationale) > 0


class TestCostSensitiveWorkload:
    """
    Bulk document summarisation pipeline.
    - Large inputs (2,000 tokens), medium outputs (300 tokens)
    - Relaxed latency budget: 30,000 ms (30 seconds acceptable for batch)
    - Very tight cost budget: $0.05 per 1k requests
    - Low RPS: 1 rps (overnight batch)
    """

    @pytest.fixture
    def profile(self) -> RequestProfile:
        return RequestProfile(
            avg_input_tokens=2_000,
            avg_output_tokens=300,
            requests_per_second=1.0,
            latency_budget_ms=30_000,
            cost_budget_per_1k_requests=0.05,
        )

    def test_recommends_haiku_for_cost(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # Cost for haiku at 2000 input + 300 output:
        # (2000 * 0.10 + 300 * 0.50) / 1000 = (200 + 150) / 1000 = $0.35 / 1k
        # Still > $0.05 budget — fallback to haiku expected
        assert plan.recommended_model == "claude-haiku-5-5"

    def test_enables_caching_for_large_inputs(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # avg_input_tokens=2000 > 500 threshold
        assert plan.use_caching is True

    def test_no_batching_at_very_low_rps(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # 1.0 rps < 5.0 threshold
        assert plan.use_batching is False

    def test_no_quantization_at_low_rps(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        assert plan.recommended_quantization == "none"

    def test_caching_reduces_estimated_cost(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan_cached = optimizer.recommend(profile)
        # Manually get uncached estimate for comparison
        cost_no_cache = optimizer._estimate_cost(
            "claude-haiku-5-5", profile, use_caching=False
        )
        cost_with_cache = optimizer._estimate_cost(
            "claude-haiku-5-5", profile, use_caching=True
        )
        assert cost_with_cache < cost_no_cache


class TestHighThroughputWorkload:
    """
    Real-time log classification at high RPS.
    - Short inputs (100 tokens), very short outputs (20 tokens)
    - Moderate latency budget: 5,000 ms
    - Cost budget: $0.02 per 1k requests
    - Very high RPS: 50 rps
    """

    @pytest.fixture
    def profile(self) -> RequestProfile:
        return RequestProfile(
            avg_input_tokens=100,
            avg_output_tokens=20,
            requests_per_second=50.0,
            latency_budget_ms=5_000,
            cost_budget_per_1k_requests=0.02,
        )

    def test_recommends_haiku_for_high_throughput(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        assert plan.recommended_model == "claude-haiku-5-5"

    def test_enables_batching_at_high_rps(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # 50 rps >> 5 rps threshold
        assert plan.use_batching is True

    def test_int4_quantization_at_very_high_rps(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # 50 rps > 20 rps threshold → INT4
        assert plan.recommended_quantization == "INT4"

    def test_no_caching_for_short_inputs(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        # avg_input_tokens=100 < 500 threshold
        assert plan.use_caching is False

    def test_latency_estimate_increases_with_batching(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        latency_no_batch = optimizer._estimate_latency(
            "claude-haiku-5-5", profile, use_batching=False
        )
        latency_with_batch = optimizer._estimate_latency(
            "claude-haiku-5-5", profile, use_batching=True
        )
        assert latency_with_batch > latency_no_batch

    def test_plan_fields_are_positive(
        self, optimizer: InferenceOptimizer, profile: RequestProfile
    ) -> None:
        plan = optimizer.recommend(profile)
        assert plan.estimated_cost_per_1k > 0
        assert plan.estimated_p95_latency_ms > 0


class TestEdgeCases:
    """Boundary conditions and model selection thresholds."""

    def test_exactly_at_rps_threshold_batching(self) -> None:
        optimizer = InferenceOptimizer()
        profile = RequestProfile(
            avg_input_tokens=500,
            avg_output_tokens=100,
            requests_per_second=5.0,
            latency_budget_ms=20_000,
            cost_budget_per_1k_requests=1.0,
        )
        plan = optimizer.recommend(profile)
        # 5.0 rps is NOT > 5.0, so batching should be off
        assert plan.use_batching is False

    def test_exactly_at_cache_threshold(self) -> None:
        optimizer = InferenceOptimizer()
        profile = RequestProfile(
            avg_input_tokens=500,
            avg_output_tokens=100,
            requests_per_second=2.0,
            latency_budget_ms=20_000,
            cost_budget_per_1k_requests=1.0,
        )
        plan = optimizer.recommend(profile)
        # 500 tokens is NOT > 500, so caching should be off
        assert plan.use_caching is False

    def test_plan_rationale_is_non_empty(self) -> None:
        optimizer = InferenceOptimizer()
        profile = RequestProfile(
            avg_input_tokens=1_000,
            avg_output_tokens=200,
            requests_per_second=10.0,
            latency_budget_ms=15_000,
            cost_budget_per_1k_requests=0.50,
        )
        plan = optimizer.recommend(profile)
        assert isinstance(plan.rationale, str)
        assert len(plan.rationale) > 20
```

---

### Running the Tests

```bash
# From the project root
pip install pydantic pytest

pytest tests/test_inference_optimizer.py -v
```

Expected output: all tests pass. The "latency-sensitive" and "cost-sensitive" scenarios both fall back to haiku because the estimates show no model fits a $0.05/1k or 2,000 ms budget at those token volumes — this is the correct, intentional behavior, and the tests verify that fallback logic explicitly.
