# Phase 6 · Week 10 · Day 6 — Build: Inference Optimization Toolkit

> Objective: Build a production-ready toolkit: cost calculator, streaming client, load-test harness, and decision guide — all integrated, tested, mockable.  ·  Time: ~4 hr  ·  Prereqs: [[day-1.md]] through [[day-5.md]]

## Concept

This session synthesizes the week into four composable modules:

1. **TokenCostCalculator** — pure math; no I/O; takes model name + token counts + cache hits → dollar cost.
2. **StreamingClient** — wraps Anthropic streaming; tracks TTFT/TBT; injectable mock for tests.
3. **LoadTestHarness** — fires N concurrent async requests; collects p50/p95/p99 latency + throughput.
4. **DecisionGuide** — given latency/cost/throughput constraints → recommends model size, quantization, batching strategy.

All components:
- Type-annotated with Pydantic models for all data at boundaries.
- Mockable: every LLM call behind a protocol/interface.
- Tested: full pytest suite, no real API calls.
- Logging not print.

**Connection to week-9 gateway**: In production, these four modules wrap a gateway — the StreamingClient points at the gateway URL instead of `api.anthropic.com`, the LoadTestHarness load-tests the gateway, and the TokenCostCalculator reads pricing from the gateway's billing metadata.

## How it works internally

```
                    ┌──────────────────────────────────┐
                    │        Inference Toolkit          │
                    │                                   │
  Input prompt  ──► │  StreamingClient                  │──► token stream
                    │    (mock | real Anthropic)        │
                    │                                   │
  N, concurrency ──►│  LoadTestHarness                  │──► LatencySummary
                    │    asyncio.gather() N tasks       │
                    │    each task → StreamingClient    │
                    │                                   │
  model+tokens   ──►│  TokenCostCalculator              │──► CostBreakdown
                    │    pricing dict → Pydantic model  │
                    │                                   │
  constraints    ──►│  DecisionGuide                    │──► Recommendation
                    │    pure conditional logic         │
                    └──────────────────────────────────┘
```

**TokenCostCalculator**: Pricing loaded from a hard-coded dict keyed by model ID. Uses Pydantic for schema validation. Returns a `CostBreakdown` model.

**StreamingClient**: Uses the `LLMBackend` protocol — two implementations: `AnthropicBackend` (real API) and `MockBackend` (configurable delays, fixed responses). The harness always uses `MockBackend` to avoid API costs and flakiness.

**LoadTestHarness**: `asyncio.gather()` with a semaphore to bound concurrency. Each worker records start/end time. After all tasks complete, sorts latencies and computes percentiles.

**DecisionGuide**: Pure function — no I/O. Decision tree over latency/cost/throughput thresholds returns a `Recommendation` dataclass.

## Worked example

```python
"""
day6_inference_toolkit.py

Production inference optimization toolkit.
Four modules: TokenCostCalculator, StreamingClient, LoadTestHarness, DecisionGuide.

Run all components:
    python day6_inference_toolkit.py

Run tests:
    pytest day6_inference_toolkit.py -v
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

import anthropic
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ============================================================================
# MODULE 1: TokenCostCalculator
# ============================================================================

class ModelPricing(BaseModel):
    """Pricing per million tokens (USD)."""
    model_id: str
    input_per_mtok: float = Field(gt=0)
    output_per_mtok: float = Field(gt=0)
    cache_read_per_mtok: float = Field(ge=0)
    cache_creation_per_mtok: float = Field(ge=0)

    @model_validator(mode="after")
    def cache_read_cheaper_than_input(self) -> "ModelPricing":
        if self.cache_read_per_mtok > self.input_per_mtok:
            raise ValueError("cache_read must be cheaper than input")
        return self


class CostBreakdown(BaseModel):
    """Full cost breakdown for a single LLM call."""
    model_id: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    cache_read_cost_usd: float
    cache_creation_cost_usd: float
    total_cost_usd: float
    savings_vs_no_cache_usd: float


# Pricing as of 2025 — update from docs.anthropic.com/pricing
_PRICING_TABLE: dict[str, dict] = {
    "claude-haiku-5-5": {
        "input_per_mtok": 0.80,
        "output_per_mtok": 4.00,
        "cache_read_per_mtok": 0.08,
        "cache_creation_per_mtok": 1.00,
    },
    "claude-sonnet-5-5": {
        "input_per_mtok": 3.00,
        "output_per_mtok": 15.00,
        "cache_read_per_mtok": 0.30,
        "cache_creation_per_mtok": 3.75,
    },
    "claude-opus-5-5": {
        "input_per_mtok": 15.00,
        "output_per_mtok": 75.00,
        "cache_read_per_mtok": 1.50,
        "cache_creation_per_mtok": 18.75,
    },
}


class TokenCostCalculator:
    """
    Compute LLM call cost from token counts.
    No I/O — pure arithmetic.
    """

    def __init__(self) -> None:
        self._pricing: dict[str, ModelPricing] = {
            model_id: ModelPricing(model_id=model_id, **pricing)
            for model_id, pricing in _PRICING_TABLE.items()
        }

    def supported_models(self) -> list[str]:
        return list(self._pricing.keys())

    def compute(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> CostBreakdown:
        """
        Compute cost for a single API call.

        Tokens are mutually exclusive buckets:
          - input_tokens: normal (non-cached) input
          - cache_read_tokens: served from cache (cheap)
          - cache_creation_tokens: written to cache (slightly expensive)
        """
        if model_id not in self._pricing:
            raise ValueError(
                f"Unknown model '{model_id}'. Supported: {self.supported_models()}"
            )
        p = self._pricing[model_id]

        def tok_cost(tokens: int, rate_per_mtok: float) -> float:
            return tokens * rate_per_mtok / 1_000_000

        input_cost = tok_cost(input_tokens, p.input_per_mtok)
        output_cost = tok_cost(output_tokens, p.output_per_mtok)
        cache_read_cost = tok_cost(cache_read_tokens, p.cache_read_per_mtok)
        cache_creation_cost = tok_cost(cache_creation_tokens, p.cache_creation_per_mtok)
        total = input_cost + output_cost + cache_read_cost + cache_creation_cost

        # Savings = what cache_read_tokens would have cost at full input price
        savings = tok_cost(cache_read_tokens, p.input_per_mtok - p.cache_read_per_mtok)

        return CostBreakdown(
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            cache_read_cost_usd=round(cache_read_cost, 8),
            cache_creation_cost_usd=round(cache_creation_cost, 8),
            total_cost_usd=round(total, 8),
            savings_vs_no_cache_usd=round(savings, 8),
        )

    def compare_models(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
    ) -> list[CostBreakdown]:
        """Return cost for all models, sorted cheapest first."""
        results = [
            self.compute(m, input_tokens, output_tokens, cache_read_tokens)
            for m in self.supported_models()
        ]
        return sorted(results, key=lambda r: r.total_cost_usd)


# ============================================================================
# MODULE 2: StreamingClient
# ============================================================================

class StreamResult(BaseModel):
    """Result of a completed streaming call."""
    request_id: str
    full_text: str
    ttft_ms: float
    total_ms: float
    token_count: int
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol: any backend that can stream tokens."""

    async def stream(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        ...

    async def complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> StreamResult:
        ...


class MockBackend:
    """
    Mock LLM backend for testing and load simulation.
    Configurable TTFT, token delay, and response text.
    """

    def __init__(
        self,
        ttft_ms: float = 50.0,
        token_delay_ms: float = 20.0,
        response: str = "This is a mock response with ten tokens total here.",
    ) -> None:
        self.ttft_ms = ttft_ms
        self.token_delay_ms = token_delay_ms
        self.response = response

    async def stream(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        await asyncio.sleep(self.ttft_ms / 1000)
        tokens = self.response.split()
        for token in tokens[:max_tokens]:
            yield token + " "
            await asyncio.sleep(self.token_delay_ms / 1000)

    async def complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> StreamResult:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        tokens: list[str] = []
        first_token_time: float | None = None

        async for token in self.stream(prompt, system, max_tokens):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = (now - start) * 1000
            tokens.append(token)

        total_ms = (time.perf_counter() - start) * 1000
        return StreamResult(
            request_id=request_id,
            full_text="".join(tokens),
            ttft_ms=first_token_time or 0.0,
            total_ms=total_ms,
            token_count=len(tokens),
            input_tokens=len(prompt.split()),  # approximation for mock
            output_tokens=len(tokens),
        )


class AnthropicBackend:
    """Real Anthropic streaming backend."""

    def __init__(self, model: str = "claude-haiku-5-5") -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def stream(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        # Note: anthropic SDK streaming is sync; wrap in thread for async context
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_stream():
            msgs = [{"role": "user", "content": prompt}]
            kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
            if system:
                kwargs["system"] = system
            chunks = []
            with self._client.messages.stream(**kwargs) as s:
                for chunk in s.text_stream:
                    chunks.append(chunk)
            return chunks

        chunks = await loop.run_in_executor(None, _sync_stream)
        for chunk in chunks:
            yield chunk

    async def complete(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> StreamResult:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        tokens: list[str] = []
        first_token_time: float | None = None

        async for token in self.stream(prompt, system, max_tokens):
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = (now - start) * 1000
            tokens.append(token)

        total_ms = (time.perf_counter() - start) * 1000
        return StreamResult(
            request_id=request_id,
            full_text="".join(tokens),
            ttft_ms=first_token_time or 0.0,
            total_ms=total_ms,
            token_count=len(tokens),
        )


class StreamingClient:
    """Facade over LLMBackend — use MockBackend for tests."""

    def __init__(self, backend: LLMBackend | None = None) -> None:
        self.backend: LLMBackend = backend or MockBackend()

    async def call(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 256,
    ) -> StreamResult:
        return await self.backend.complete(prompt, system, max_tokens)


# ============================================================================
# MODULE 3: LoadTestHarness
# ============================================================================

class LatencySummary(BaseModel):
    """Aggregate latency stats from a load test."""
    n_requests: int
    n_success: int
    n_error: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    throughput_rps: float
    total_duration_s: float
    errors: list[str] = Field(default_factory=list)


def _percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    floor_k = int(math.floor(k))
    ceil_k = int(math.ceil(k))
    if floor_k == ceil_k:
        return sorted_data[floor_k]
    return sorted_data[floor_k] + (sorted_data[ceil_k] - sorted_data[floor_k]) * (k - floor_k)


@dataclass
class LoadTestHarness:
    """
    Fires N concurrent requests via StreamingClient and reports latency stats.
    Uses MockBackend by default — inject real backend for production tests.
    """
    client: StreamingClient = field(default_factory=StreamingClient)
    max_concurrency: int = 10

    async def run(
        self,
        prompt: str,
        n_requests: int,
        system: str = "",
        max_tokens: int = 64,
    ) -> LatencySummary:
        semaphore = asyncio.Semaphore(self.max_concurrency)
        latencies: list[float] = []
        errors: list[str] = []

        async def single_request(_: int) -> None:
            async with semaphore:
                try:
                    result = await self.client.call(prompt, system, max_tokens)
                    latencies.append(result.total_ms)
                except Exception as exc:
                    errors.append(str(exc))

        wall_start = time.perf_counter()
        await asyncio.gather(*[single_request(i) for i in range(n_requests)])
        wall_total_s = time.perf_counter() - wall_start

        n_success = len(latencies)
        throughput = n_success / wall_total_s if wall_total_s > 0 else 0.0

        return LatencySummary(
            n_requests=n_requests,
            n_success=n_success,
            n_error=len(errors),
            p50_ms=round(_percentile(latencies, 50), 2),
            p95_ms=round(_percentile(latencies, 95), 2),
            p99_ms=round(_percentile(latencies, 99), 2),
            min_ms=round(min(latencies, default=0.0), 2),
            max_ms=round(max(latencies, default=0.0), 2),
            mean_ms=round(statistics.mean(latencies) if latencies else 0.0, 2),
            throughput_rps=round(throughput, 2),
            total_duration_s=round(wall_total_s, 3),
            errors=errors[:10],  # cap error list
        )


# ============================================================================
# MODULE 4: DecisionGuide
# ============================================================================

class DeploymentConstraints(BaseModel):
    """User-defined constraints for model selection."""
    latency_budget_ms: float = Field(gt=0, description="Max acceptable p95 latency")
    cost_budget_per_1k_usd: float = Field(gt=0, description="Max cost per 1000 requests")
    throughput_req_per_s: float = Field(gt=0, description="Required requests per second")
    avg_input_tokens: int = Field(gt=0, default=500)
    avg_output_tokens: int = Field(gt=0, default=200)


class Recommendation(BaseModel):
    """Decision guide output."""
    model_id: str
    quantization: str  # "none" | "int8" | "int4"
    batching_strategy: str  # "continuous" | "static" | "speculative"
    prompt_caching: bool
    rationale: str
    estimated_cost_per_1k_usd: float
    warnings: list[str] = Field(default_factory=list)


# Approximate latency reference values (ms, p95) at typical throughput
_MODEL_LATENCY: dict[str, float] = {
    "claude-haiku-5-5":  800.0,
    "claude-sonnet-5-5": 2500.0,
    "claude-opus-5-5":   6000.0,
}


class DecisionGuide:
    """
    Pure logic: given deployment constraints, recommend model + config.
    No I/O. No LLM calls. Fully testable.
    """

    def __init__(self) -> None:
        self._calc = TokenCostCalculator()

    def recommend(self, constraints: DeploymentConstraints) -> Recommendation:
        warnings: list[str] = []

        # Step 1: select model by latency budget
        model_id = self._select_model_by_latency(
            constraints.latency_budget_ms, warnings
        )

        # Step 2: estimate cost for selected model
        cost_breakdown = self._calc.compute(
            model_id,
            input_tokens=constraints.avg_input_tokens,
            output_tokens=constraints.avg_output_tokens,
        )
        cost_per_1k = cost_breakdown.total_cost_usd * 1000

        # Step 3: check cost budget, downgrade or warn
        if cost_per_1k > constraints.cost_budget_per_1k_usd:
            warnings.append(
                f"Estimated cost ${cost_per_1k:.4f}/1k exceeds budget "
                f"${constraints.cost_budget_per_1k_usd:.4f}/1k"
            )
            # Try downgrading to haiku
            if model_id != "claude-haiku-5-5":
                haiku_cost = self._calc.compute(
                    "claude-haiku-5-5",
                    constraints.avg_input_tokens,
                    constraints.avg_output_tokens,
                ).total_cost_usd * 1000
                if haiku_cost <= constraints.cost_budget_per_1k_usd:
                    model_id = "claude-haiku-5-5"
                    cost_per_1k = haiku_cost
                    warnings.append("Downgraded to claude-haiku-5-5 to meet cost budget")
                else:
                    warnings.append(
                        "Even claude-haiku-5-5 exceeds cost budget — reduce token counts"
                    )

        # Step 4: quantization recommendation
        quantization = self._recommend_quantization(
            constraints.throughput_req_per_s, model_id
        )

        # Step 5: batching strategy
        batching = self._recommend_batching(
            constraints.latency_budget_ms,
            constraints.throughput_req_per_s,
        )

        # Step 6: prompt caching
        prompt_caching = constraints.avg_input_tokens > 512

        rationale = (
            f"Selected {model_id}: p95 latency ~{_MODEL_LATENCY[model_id]:.0f}ms "
            f"vs budget {constraints.latency_budget_ms:.0f}ms. "
            f"Cost ~${cost_per_1k:.4f}/1k requests. "
            f"Batching={batching}, quant={quantization}."
        )

        return Recommendation(
            model_id=model_id,
            quantization=quantization,
            batching_strategy=batching,
            prompt_caching=prompt_caching,
            rationale=rationale,
            estimated_cost_per_1k_usd=round(cost_per_1k, 6),
            warnings=warnings,
        )

    def _select_model_by_latency(
        self, budget_ms: float, warnings: list[str]
    ) -> str:
        # Tightest budget → smallest model
        if budget_ms < 1500:
            return "claude-haiku-5-5"
        elif budget_ms < 4000:
            return "claude-sonnet-5-5"
        else:
            return "claude-opus-5-5"

    def _recommend_quantization(self, rps: float, model_id: str) -> str:
        # High throughput → quantization to fit more in GPU memory
        if rps > 50:
            return "int8"
        elif rps > 100:
            return "int4"
        return "none"

    def _recommend_batching(self, latency_budget_ms: float, rps: float) -> str:
        if latency_budget_ms < 1000 and rps < 10:
            return "speculative"   # low-latency, low-throughput: speculative decoding
        elif rps > 20:
            return "continuous"    # high throughput: continuous batching (vLLM-style)
        else:
            return "static"        # moderate: static batches


# ============================================================================
# Integration demo
# ============================================================================

async def main() -> None:
    log.info("=== Inference Optimization Toolkit Demo ===")

    # 1. Cost calculator
    log.info("--- TokenCostCalculator ---")
    calc = TokenCostCalculator()
    for model in calc.supported_models():
        cost = calc.compute(model, input_tokens=1000, output_tokens=500, cache_read_tokens=800)
        log.info(
            "%s: total=$%.6f (savings=$%.6f from cache)",
            model,
            cost.total_cost_usd,
            cost.savings_vs_no_cache_usd,
        )

    # 2. Streaming client (mock)
    log.info("--- StreamingClient (mock) ---")
    client = StreamingClient(backend=MockBackend(ttft_ms=50, token_delay_ms=10))
    result = await client.call("What is the speed of light?")
    log.info(
        "TTFT=%.1fms | total=%.1fms | tokens=%d",
        result.ttft_ms,
        result.total_ms,
        result.token_count,
    )

    # 3. Load test harness (mock backend, no real API calls)
    log.info("--- LoadTestHarness (20 requests, concurrency=5) ---")
    harness = LoadTestHarness(
        client=StreamingClient(backend=MockBackend(ttft_ms=50, token_delay_ms=5)),
        max_concurrency=5,
    )
    summary = await harness.run("Tell me a fact.", n_requests=20, max_tokens=32)
    log.info(
        "p50=%.1fms | p95=%.1fms | p99=%.1fms | throughput=%.1f rps | errors=%d",
        summary.p50_ms,
        summary.p95_ms,
        summary.p99_ms,
        summary.throughput_rps,
        summary.n_error,
    )

    # 4. Decision guide
    log.info("--- DecisionGuide ---")
    guide = DecisionGuide()
    scenarios = [
        DeploymentConstraints(
            latency_budget_ms=1000,
            cost_budget_per_1k_usd=0.01,
            throughput_req_per_s=5.0,
        ),
        DeploymentConstraints(
            latency_budget_ms=5000,
            cost_budget_per_1k_usd=0.50,
            throughput_req_per_s=100.0,
            avg_input_tokens=2000,
            avg_output_tokens=1000,
        ),
    ]
    for i, constraints in enumerate(scenarios, 1):
        rec = guide.recommend(constraints)
        log.info(
            "Scenario %d: model=%s | quant=%s | batching=%s | cache=%s | cost=$%.4f/1k",
            i,
            rec.model_id,
            rec.quantization,
            rec.batching_strategy,
            rec.prompt_caching,
            rec.estimated_cost_per_1k_usd,
        )
        for w in rec.warnings:
            log.warning("  Warning: %s", w)


if __name__ == "__main__":
    asyncio.run(main())


# ============================================================================
# Tests
# ============================================================================

import pytest  # noqa: E402


class TestTokenCostCalculator:
    def setup_method(self):
        self.calc = TokenCostCalculator()

    def test_haiku_basic_cost(self):
        cost = self.calc.compute("claude-haiku-5-5", input_tokens=1_000_000, output_tokens=0)
        assert abs(cost.input_cost_usd - 0.80) < 0.001

    def test_output_cost(self):
        cost = self.calc.compute("claude-haiku-5-5", input_tokens=0, output_tokens=1_000_000)
        assert abs(cost.output_cost_usd - 4.00) < 0.001

    def test_cache_read_cheaper_than_input(self):
        cost = self.calc.compute(
            "claude-haiku-5-5",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1_000_000,
        )
        # 0.08/MTok, not 0.80
        assert abs(cost.cache_read_cost_usd - 0.08) < 0.001

    def test_savings_calculation(self):
        cost = self.calc.compute(
            "claude-haiku-5-5",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1_000_000,
        )
        # Savings = (0.80 - 0.08) * 1 = 0.72
        assert abs(cost.savings_vs_no_cache_usd - 0.72) < 0.001

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            self.calc.compute("gpt-999", input_tokens=100, output_tokens=100)

    def test_compare_models_sorted(self):
        results = self.calc.compare_models(input_tokens=1000, output_tokens=500)
        costs = [r.total_cost_usd for r in results]
        assert costs == sorted(costs)

    def test_zero_tokens_zero_cost(self):
        cost = self.calc.compute("claude-sonnet-5-5", input_tokens=0, output_tokens=0)
        assert cost.total_cost_usd == 0.0

    def test_all_models_supported(self):
        for model in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]:
            cost = self.calc.compute(model, input_tokens=100, output_tokens=100)
            assert cost.total_cost_usd > 0


class TestMockBackend:
    @pytest.mark.asyncio
    async def test_ttft_within_tolerance(self):
        backend = MockBackend(ttft_ms=100, token_delay_ms=0)
        client = StreamingClient(backend=backend)
        result = await client.call("hello", max_tokens=5)
        # Allow 2x tolerance for test environment overhead
        assert result.ttft_ms < 100 * 3

    @pytest.mark.asyncio
    async def test_full_text_returned(self):
        backend = MockBackend(response="hello world test")
        client = StreamingClient(backend=backend)
        result = await client.call("hi", max_tokens=3)
        assert "hello" in result.full_text

    @pytest.mark.asyncio
    async def test_max_tokens_respected(self):
        backend = MockBackend(response="one two three four five six seven eight nine ten")
        client = StreamingClient(backend=backend)
        result = await client.call("hi", max_tokens=3)
        assert result.token_count <= 3

    @pytest.mark.asyncio
    async def test_request_id_unique(self):
        backend = MockBackend()
        client = StreamingClient(backend=backend)
        r1 = await client.call("a")
        r2 = await client.call("b")
        assert r1.request_id != r2.request_id


class TestLoadTestHarness:
    @pytest.mark.asyncio
    async def test_n_requests_fulfilled(self):
        harness = LoadTestHarness(
            client=StreamingClient(backend=MockBackend(ttft_ms=1, token_delay_ms=1)),
            max_concurrency=5,
        )
        summary = await harness.run("hello", n_requests=10, max_tokens=5)
        assert summary.n_success == 10
        assert summary.n_error == 0

    @pytest.mark.asyncio
    async def test_percentile_ordering(self):
        harness = LoadTestHarness(
            client=StreamingClient(backend=MockBackend(ttft_ms=10, token_delay_ms=2)),
        )
        summary = await harness.run("hi", n_requests=20, max_tokens=5)
        assert summary.p50_ms <= summary.p95_ms <= summary.p99_ms

    @pytest.mark.asyncio
    async def test_throughput_positive(self):
        harness = LoadTestHarness(
            client=StreamingClient(backend=MockBackend(ttft_ms=1, token_delay_ms=1)),
            max_concurrency=10,
        )
        summary = await harness.run("hello", n_requests=5, max_tokens=3)
        assert summary.throughput_rps > 0

    @pytest.mark.asyncio
    async def test_summary_counts_match(self):
        harness = LoadTestHarness(
            client=StreamingClient(backend=MockBackend()),
        )
        summary = await harness.run("x", n_requests=7, max_tokens=2)
        assert summary.n_success + summary.n_error == summary.n_requests


class TestDecisionGuide:
    def setup_method(self):
        self.guide = DecisionGuide()

    def test_tight_latency_selects_haiku(self):
        rec = self.guide.recommend(
            DeploymentConstraints(
                latency_budget_ms=500,
                cost_budget_per_1k_usd=1.0,
                throughput_req_per_s=1.0,
            )
        )
        assert rec.model_id == "claude-haiku-5-5"

    def test_loose_latency_allows_opus(self):
        rec = self.guide.recommend(
            DeploymentConstraints(
                latency_budget_ms=10_000,
                cost_budget_per_1k_usd=10.0,
                throughput_req_per_s=1.0,
                avg_input_tokens=5000,
                avg_output_tokens=2000,
            )
        )
        assert rec.model_id == "claude-opus-5-5"

    def test_cost_budget_exceeded_produces_warning(self):
        rec = self.guide.recommend(
            DeploymentConstraints(
                latency_budget_ms=10_000,
                cost_budget_per_1k_usd=0.0001,  # impossibly low
                throughput_req_per_s=1.0,
            )
        )
        assert len(rec.warnings) > 0

    def test_large_input_enables_caching(self):
        rec = self.guide.recommend(
            DeploymentConstraints(
                latency_budget_ms=5000,
                cost_budget_per_1k_usd=1.0,
                throughput_req_per_s=1.0,
                avg_input_tokens=2048,
            )
        )
        assert rec.prompt_caching is True

    def test_small_input_disables_caching(self):
        rec = self.guide.recommend(
            DeploymentConstraints(
                latency_budget_ms=1000,
                cost_budget_per_1k_usd=1.0,
                throughput_req_per_s=1.0,
                avg_input_tokens=100,
                avg_output_tokens=50,
            )
        )
        assert rec.prompt_caching is False

    def test_recommendation_fields_valid(self):
        rec = self.guide.recommend(
            DeploymentConstraints(
                latency_budget_ms=2000,
                cost_budget_per_1k_usd=0.10,
                throughput_req_per_s=5.0,
            )
        )
        assert rec.model_id in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
        assert rec.quantization in ["none", "int8", "int4"]
        assert rec.batching_strategy in ["continuous", "static", "speculative"]
        assert rec.estimated_cost_per_1k_usd >= 0
```

## Exercises

1. **Extend the calculator**: Add a `monthly_cost_estimate(requests_per_day: int, **kwargs) -> float` method to `TokenCostCalculator`. Include a `cache_hit_rate: float` parameter — for each request, `cache_hit_rate` fraction of input tokens are `cache_read_tokens`. Write 3 pytest tests covering 0%, 50%, and 100% cache hit rates.

2. **Percentile reporting**: Add a `percentile(data: list[float], p: float) -> float` function that handles edge cases: empty list, single element, p=0, p=100. Add it to `LatencySummary` as a method. Write parametrized pytest tests for all edge cases using `@pytest.mark.parametrize`.

3. **Cache-hit simulation in load test**: Add a `cache_hit_rate: float = 0.0` parameter to `MockBackend`. When a request hits the cache, set `ttft_ms = 5` (simulating a fast cache hit). Run the load test with `cache_hit_rate=0.5` and verify that mean latency is roughly 50% of the no-cache case.

4. **Gateway integration**: The `StreamingClient` currently hardcodes `AnthropicBackend`. Add a `GatewayBackend` that accepts a `base_url: str` and sends HTTP POST requests to `{base_url}/v1/messages` using `httpx.AsyncClient`. The backend must implement `LLMBackend` protocol. Write a pytest test using `respx` (or `unittest.mock`) to mock the HTTP call.

5. **Decision guide refinement**: The current `_recommend_quantization` has a bug — the `int4` branch requires `rps > 100` but also `rps > 50`, so `int4` is never reached (dead code). Fix the logic, add a pytest test that triggers the `int4` path, and write a docstring explaining the corrected thresholds.

(Solutions: ../solutions/week-10/day-6.md)

## Self-check

- [ ] I can implement `TokenCostCalculator` from memory with correct cache savings math.
- [ ] I can explain the `LLMBackend` protocol pattern and why it makes the harness testable without API calls.
- [ ] I can write an async load test with `asyncio.gather()` and a semaphore for concurrency limiting.
- [ ] I can implement p50/p95/p99 percentile computation correctly for edge cases.
- [ ] I can articulate when `DecisionGuide` should recommend each model tier and batching strategy.

## References

- Anthropic pricing: https://docs.anthropic.com/en/docs/about-claude/models
- vLLM: https://docs.vllm.ai
- asyncio semaphore patterns: https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore
