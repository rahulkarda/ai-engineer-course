# Week 10 · Day 6 Solutions — Inference Optimization Toolkit

---

## Overview

This day assembles four reusable components into a single `inference_toolkit` package:

```
src/inference_toolkit/
    __init__.py
    cost.py          # TokenCostCalculator
    streaming.py     # StreamingClient
    load_test.py     # LoadTestHarness
    decision.py      # DecisionGuide
tests/week10/day6/
    test_cost.py
    test_streaming.py
    test_load_test.py
    test_decision.py
```

All components are importable from `inference_toolkit` and designed to work together.

---

## Component 1: TokenCostCalculator

### Reference Implementation

```python
"""
src/inference_toolkit/cost.py

Token cost calculator for Anthropic models.

Pricing (as of 2025 — verify against official pricing page before billing):
  claude-haiku-5-5   : $0.10 / $0.50  per 1M input/output tokens
  claude-sonnet-5-5  : $2.00 / $10.00 per 1M
  claude-opus-5-5    : $4.00 / $20.00 per 1M

Cache reads: 10% of input price (ephemeral cache read tier).
"""

import logging
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

DOLLARS_PER_MILLION = 1_000_000


class PricingConfig(BaseModel):
    """Per-model pricing in dollars per token."""

    input_price_per_token: Annotated[float, Field(gt=0)]
    output_price_per_token: Annotated[float, Field(gt=0)]
    cache_read_price_per_token: float  # derived: 10% of input

    @model_validator(mode="before")
    @classmethod
    def derive_cache_price(cls, data: dict) -> dict:
        if "cache_read_price_per_token" not in data:
            data["cache_read_price_per_token"] = (
                data["input_price_per_token"] * 0.10
            )
        return data


# Canonical model -> pricing table
MODEL_PRICING: dict[str, PricingConfig] = {
    "claude-haiku-5-5": PricingConfig(
        input_price_per_token=0.10 / DOLLARS_PER_MILLION,
        output_price_per_token=0.50 / DOLLARS_PER_MILLION,
    ),
    "claude-sonnet-5-5": PricingConfig(
        input_price_per_token=2.00 / DOLLARS_PER_MILLION,
        output_price_per_token=10.00 / DOLLARS_PER_MILLION,
    ),
    "claude-opus-5-5": PricingConfig(
        input_price_per_token=4.00 / DOLLARS_PER_MILLION,
        output_price_per_token=20.00 / DOLLARS_PER_MILLION,
    ),
}


class CostResult(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int
    input_cost: float      # USD
    output_cost: float     # USD
    cache_read_cost: float # USD
    total_cost: float      # USD


class TokenCostCalculator:
    """
    Compute USD cost for a single LLM request.

    Parameters
    ----------
    model          : one of the keys in MODEL_PRICING
    input_tokens   : non-cache input tokens billed at full input price
    output_tokens  : generated output tokens
    cache_hit_tokens: tokens served from prompt cache (billed at 10% of input price)
    """

    def __init__(self, pricing: dict[str, PricingConfig] | None = None) -> None:
        self._pricing = pricing or MODEL_PRICING

    def compute_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
    ) -> CostResult:
        if model not in self._pricing:
            raise ValueError(
                f"Unknown model {model!r}. Supported: {list(self._pricing)}"
            )
        pricing = self._pricing[model]

        input_cost = input_tokens * pricing.input_price_per_token
        output_cost = output_tokens * pricing.output_price_per_token
        cache_read_cost = cache_hit_tokens * pricing.cache_read_price_per_token
        total = input_cost + output_cost + cache_read_cost

        logger.debug(
            "cost model=%s in=%d out=%d cache=%d total=%.6f",
            model, input_tokens, output_tokens, cache_hit_tokens, total,
        )
        return CostResult(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_read_cost,
            total_cost=total,
        )
```

### Tests

```python
"""tests/week10/day6/test_cost.py"""

import pytest
from inference_toolkit.cost import CostResult, TokenCostCalculator


@pytest.fixture
def calc() -> TokenCostCalculator:
    return TokenCostCalculator()


def test_haiku_zero_tokens_zero_cost(calc: TokenCostCalculator) -> None:
    result = calc.compute_cost("claude-haiku-5-5", 0, 0)
    assert result.total_cost == pytest.approx(0.0)


def test_haiku_input_cost_correct(calc: TokenCostCalculator) -> None:
    """1M input tokens at $0.10/1M should cost $0.10."""
    result = calc.compute_cost("claude-haiku-5-5", 1_000_000, 0)
    assert result.input_cost == pytest.approx(0.10, rel=1e-6)


def test_opus_output_cost_correct(calc: TokenCostCalculator) -> None:
    """1M output tokens at $20.00/1M should cost $20.00."""
    result = calc.compute_cost("claude-opus-5-5", 0, 1_000_000)
    assert result.output_cost == pytest.approx(20.00, rel=1e-6)


def test_cache_hit_is_10_percent_of_input(calc: TokenCostCalculator) -> None:
    """Cache-read tokens should cost 10% of the full input price."""
    full = calc.compute_cost("claude-sonnet-5-5", 1_000_000, 0, cache_hit_tokens=0)
    cached = calc.compute_cost("claude-sonnet-5-5", 0, 0, cache_hit_tokens=1_000_000)
    assert cached.cache_read_cost == pytest.approx(full.input_cost * 0.10, rel=1e-6)


def test_unknown_model_raises(calc: TokenCostCalculator) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        calc.compute_cost("claude-made-up-99-9", 100, 100)


def test_total_is_sum_of_parts(calc: TokenCostCalculator) -> None:
    result = calc.compute_cost("claude-sonnet-5-5", 500, 200, cache_hit_tokens=100)
    expected = result.input_cost + result.output_cost + result.cache_read_cost
    assert result.total_cost == pytest.approx(expected, rel=1e-9)
```

---

## Component 2: StreamingClient

### Reference Implementation

```python
"""
src/inference_toolkit/streaming.py

Streaming LLM client with TTFT and per-token latency tracking.

LLMBackend is a Protocol — implementations include MockBackend (for tests)
and AnthropicBackend (for production). All tests use MockBackend.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class StreamEvent(BaseModel):
    token: str
    token_index: int          # 0-indexed
    elapsed_ms: float         # ms since stream start
    inter_token_ms: float     # ms since previous token (0 for first token = TTFT)


class StreamMetrics(BaseModel):
    ttft_ms: float            # time to first token
    mean_tbt_ms: float        # mean time between tokens
    total_tokens: int
    total_elapsed_ms: float


@runtime_checkable
class LLMBackend(Protocol):
    async def stream(
        self, prompt: str, model: str
    ) -> AsyncGenerator[str, None]: ...


class MockBackend:
    """
    Deterministic backend for testing.

    Yields `tokens` with a `delay_ms` pause between each.
    """

    def __init__(
        self,
        tokens: list[str] | None = None,
        delay_ms: float = 10.0,
    ) -> None:
        self._tokens = tokens or ["Hello", " world", "!"]
        self._delay_ms = delay_ms

    async def stream(
        self, prompt: str, model: str
    ) -> AsyncGenerator[str, None]:
        for token in self._tokens:
            await asyncio.sleep(self._delay_ms / 1000.0)
            yield token


class AnthropicBackend:
    """
    Production backend using the official Anthropic SDK.

    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self) -> None:
        import os
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    async def stream(
        self, prompt: str, model: str
    ) -> AsyncGenerator[str, None]:
        import anthropic

        # Run blocking SDK call in a thread pool to preserve async interface
        loop = asyncio.get_event_loop()

        def _sync_stream():
            with self._client.messages.stream(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as s:
                for text in s.text_stream:
                    yield text

        # Collect synchronously, then yield asynchronously
        # For production: consider anyio or httpx async client instead
        tokens: list[str] = await loop.run_in_executor(
            None, lambda: list(_sync_stream())
        )
        for token in tokens:
            yield token


class StreamingClient:
    """
    Wraps any LLMBackend and collects streaming metrics (TTFT, TBT).
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def stream_with_metrics(
        self,
        prompt: str,
        model: str = "claude-haiku-5-5",
    ) -> tuple[list[StreamEvent], StreamMetrics]:
        events: list[StreamEvent] = []
        start = time.perf_counter()
        last_ts = start

        async for token in self._backend.stream(prompt, model):
            now = time.perf_counter()
            elapsed_ms = (now - start) * 1000.0
            inter_token_ms = (now - last_ts) * 1000.0
            last_ts = now

            events.append(
                StreamEvent(
                    token=token,
                    token_index=len(events),
                    elapsed_ms=elapsed_ms,
                    inter_token_ms=inter_token_ms,
                )
            )

        if not events:
            return events, StreamMetrics(
                ttft_ms=0.0, mean_tbt_ms=0.0, total_tokens=0, total_elapsed_ms=0.0
            )

        ttft_ms = events[0].elapsed_ms
        tbt_values = [e.inter_token_ms for e in events[1:]]
        mean_tbt_ms = sum(tbt_values) / len(tbt_values) if tbt_values else 0.0

        metrics = StreamMetrics(
            ttft_ms=ttft_ms,
            mean_tbt_ms=mean_tbt_ms,
            total_tokens=len(events),
            total_elapsed_ms=events[-1].elapsed_ms,
        )
        logger.debug("stream metrics: %s", metrics)
        return events, metrics
```

### Tests

```python
"""tests/week10/day6/test_streaming.py"""

import asyncio
import pytest
from inference_toolkit.streaming import MockBackend, StreamingClient


@pytest.fixture
def client() -> StreamingClient:
    backend = MockBackend(tokens=["Hello", " world", "!"], delay_ms=5.0)
    return StreamingClient(backend)


def test_events_count_matches_tokens(client: StreamingClient) -> None:
    events, _ = asyncio.get_event_loop().run_until_complete(
        client.stream_with_metrics("hi")
    )
    assert len(events) == 3


def test_ttft_is_positive(client: StreamingClient) -> None:
    _, metrics = asyncio.get_event_loop().run_until_complete(
        client.stream_with_metrics("hi")
    )
    assert metrics.ttft_ms > 0.0


def test_token_order_preserved(client: StreamingClient) -> None:
    events, _ = asyncio.get_event_loop().run_until_complete(
        client.stream_with_metrics("hi")
    )
    assert [e.token for e in events] == ["Hello", " world", "!"]


def test_empty_backend_returns_zero_metrics() -> None:
    backend = MockBackend(tokens=[], delay_ms=0.0)
    client = StreamingClient(backend)
    events, metrics = asyncio.get_event_loop().run_until_complete(
        client.stream_with_metrics("hi")
    )
    assert len(events) == 0
    assert metrics.total_tokens == 0
    assert metrics.ttft_ms == 0.0
```

---

## Component 3: LoadTestHarness

### Reference Implementation

```python
"""
src/inference_toolkit/load_test.py

Asyncio-based concurrent load tester for LLM inference endpoints.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Awaitable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LoadTestConfig(BaseModel):
    n_requests: int = 50
    concurrency: int = 10
    prompt: str = "Hello, respond in exactly 5 words."
    model: str = "claude-haiku-5-5"


class RequestResult(BaseModel):
    request_id: int
    latency_ms: float
    tokens_generated: int
    success: bool
    error: str | None = None


class LoadTestResult(BaseModel):
    total_requests: int
    successful: int
    failed: int
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    throughput_rps: float         # requests per second
    throughput_tokens_s: float    # output tokens per second
    total_elapsed_s: float


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# Type alias for the callable each request sends
RequestFn = Callable[[str, str], Awaitable[int]]  # (prompt, model) -> tokens_generated


class LoadTestHarness:
    """
    Send N_REQUESTS concurrently (bounded by semaphore to `concurrency`)
    and collect latency + throughput statistics.

    Parameters
    ----------
    request_fn : async callable (prompt, model) -> tokens_generated
                 Use a mock in tests; an AnthropicBackend wrapper in production.
    """

    def __init__(self, request_fn: RequestFn) -> None:
        self._request_fn = request_fn

    async def _single_request(
        self,
        sem: asyncio.Semaphore,
        request_id: int,
        config: LoadTestConfig,
    ) -> RequestResult:
        async with sem:
            start = time.perf_counter()
            try:
                tokens = await self._request_fn(config.prompt, config.model)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return RequestResult(
                    request_id=request_id,
                    latency_ms=elapsed_ms,
                    tokens_generated=tokens,
                    success=True,
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                logger.warning("Request %d failed: %s", request_id, exc)
                return RequestResult(
                    request_id=request_id,
                    latency_ms=elapsed_ms,
                    tokens_generated=0,
                    success=False,
                    error=str(exc),
                )

    async def run(self, config: LoadTestConfig) -> LoadTestResult:
        sem = asyncio.Semaphore(config.concurrency)
        wall_start = time.perf_counter()

        tasks = [
            self._single_request(sem, i, config)
            for i in range(config.n_requests)
        ]
        results: list[RequestResult] = await asyncio.gather(*tasks)

        total_elapsed = time.perf_counter() - wall_start

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        latencies = [r.latency_ms for r in successful]
        total_tokens = sum(r.tokens_generated for r in successful)

        return LoadTestResult(
            total_requests=len(results),
            successful=len(successful),
            failed=len(failed),
            error_rate=len(failed) / len(results) if results else 0.0,
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            p99_latency_ms=_percentile(latencies, 99),
            mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            throughput_rps=len(results) / total_elapsed if total_elapsed else 0.0,
            throughput_tokens_s=total_tokens / total_elapsed if total_elapsed else 0.0,
            total_elapsed_s=total_elapsed,
        )
```

### Tests

```python
"""tests/week10/day6/test_load_test.py"""

import asyncio
import pytest
from inference_toolkit.load_test import LoadTestConfig, LoadTestHarness, _percentile


async def mock_request_fn(prompt: str, model: str) -> int:
    """Simulates a fast LLM response returning 10 tokens."""
    await asyncio.sleep(0.005)  # 5 ms simulated latency
    return 10


async def failing_request_fn(prompt: str, model: str) -> int:
    raise RuntimeError("simulated failure")


def test_percentile_median() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == pytest.approx(3.0)


def test_percentile_empty_returns_zero() -> None:
    assert _percentile([], 95) == 0.0


def test_load_test_all_succeed() -> None:
    harness = LoadTestHarness(mock_request_fn)
    config = LoadTestConfig(n_requests=20, concurrency=5)
    result = asyncio.get_event_loop().run_until_complete(harness.run(config))
    assert result.successful == 20
    assert result.failed == 0
    assert result.error_rate == pytest.approx(0.0)


def test_load_test_records_latency() -> None:
    harness = LoadTestHarness(mock_request_fn)
    config = LoadTestConfig(n_requests=10, concurrency=5)
    result = asyncio.get_event_loop().run_until_complete(harness.run(config))
    assert result.p50_latency_ms > 0
    assert result.p50_latency_ms <= result.p95_latency_ms
    assert result.p95_latency_ms <= result.p99_latency_ms


def test_load_test_counts_failures() -> None:
    harness = LoadTestHarness(failing_request_fn)
    config = LoadTestConfig(n_requests=5, concurrency=2)
    result = asyncio.get_event_loop().run_until_complete(harness.run(config))
    assert result.failed == 5
    assert result.error_rate == pytest.approx(1.0)


def test_throughput_rps_positive() -> None:
    harness = LoadTestHarness(mock_request_fn)
    config = LoadTestConfig(n_requests=10, concurrency=10)
    result = asyncio.get_event_loop().run_until_complete(harness.run(config))
    assert result.throughput_rps > 0
    assert result.throughput_tokens_s > 0
```

---

## Component 4: DecisionGuide

### Reference Implementation

```python
"""
src/inference_toolkit/decision.py

Constraint-based inference optimization decision guide.

Given a workload profile (latency budget, cost budget, RPS requirement),
returns an OptimizationPlan with model choice and configuration recommendations.
"""

import logging
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

ModelTier = Literal["haiku", "sonnet", "opus"]
QuantizationType = Literal["none", "int8", "int4"]
BatchingStrategy = Literal["none", "dynamic", "static"]


class RequestProfile(BaseModel):
    latency_budget_ms: float          # max acceptable p95 latency
    cost_budget_per_1m: float         # max USD per 1M output tokens
    target_rps: float                 # requests per second
    avg_input_tokens: int = 500
    avg_output_tokens: int = 200
    requires_streaming: bool = False
    task_complexity: Literal["simple", "moderate", "complex"] = "moderate"


class OptimizationPlan(BaseModel):
    recommended_model: str
    model_tier: ModelTier
    quantization: QuantizationType
    batching_strategy: BatchingStrategy
    use_prompt_caching: bool
    use_speculative_decoding: bool
    reasoning: list[str]             # ordered list of decision rationale


class DecisionGuide:
    """
    Four distinct recommendation paths based on constraint priorities.

    Path 1: Ultra-low latency (<200 ms)      -> Haiku, no batching, streaming
    Path 2: Cost-constrained (<$1/1M)        -> Haiku + int8 quantization + caching
    Path 3: High-throughput (>100 RPS)       -> Sonnet + dynamic batching
    Path 4: Complex reasoning (complex task) -> Opus, high quality, caching
    Default: balanced Sonnet configuration
    """

    def recommend(self, profile: RequestProfile) -> OptimizationPlan:
        reasoning: list[str] = []

        # Path 1: Ultra-low latency
        if profile.latency_budget_ms < 200:
            reasoning.append(
                f"Latency budget {profile.latency_budget_ms} ms < 200 ms: "
                "use smallest model to minimize TTFT."
            )
            reasoning.append(
                "Haiku has the lowest TTFT (~100-300 ms) of the Claude family."
            )
            reasoning.append("Streaming required for sub-200 ms perceived latency.")
            return OptimizationPlan(
                recommended_model="claude-haiku-5-5",
                model_tier="haiku",
                quantization="none",
                batching_strategy="none",
                use_prompt_caching=True,
                use_speculative_decoding=True,
                reasoning=reasoning,
            )

        # Path 2: Strict cost constraint
        if profile.cost_budget_per_1m < 1.0:
            reasoning.append(
                f"Cost budget ${profile.cost_budget_per_1m:.2f}/1M tokens < $1: "
                "aggressive cost reduction required."
            )
            reasoning.append(
                "Haiku at $0.50/1M output is the only model within budget."
            )
            reasoning.append(
                "INT8 quantization reduces memory footprint by ~50% with <2% quality loss."
            )
            reasoning.append(
                "Prompt caching on repeated system prompts can reduce effective "
                "input costs by 80-90%."
            )
            return OptimizationPlan(
                recommended_model="claude-haiku-5-5",
                model_tier="haiku",
                quantization="int8",
                batching_strategy="dynamic",
                use_prompt_caching=True,
                use_speculative_decoding=False,
                reasoning=reasoning,
            )

        # Path 3: High throughput
        if profile.target_rps > 100:
            reasoning.append(
                f"Target RPS {profile.target_rps} > 100: throughput is the binding constraint."
            )
            reasoning.append(
                "Dynamic batching amortizes KV-cache setup across concurrent requests."
            )
            reasoning.append(
                "Sonnet balances quality and throughput better than Opus at high concurrency."
            )
            return OptimizationPlan(
                recommended_model="claude-sonnet-5-5",
                model_tier="sonnet",
                quantization="none",
                batching_strategy="dynamic",
                use_prompt_caching=True,
                use_speculative_decoding=False,
                reasoning=reasoning,
            )

        # Path 4: Complex reasoning tasks
        if profile.task_complexity == "complex":
            reasoning.append(
                "Task complexity is 'complex': use Opus for maximum reasoning capability."
            )
            reasoning.append(
                "Prompt caching is essential at Opus pricing to amortize long system prompts."
            )
            reasoning.append(
                "Static batching provides predictable latency for complex queries."
            )
            return OptimizationPlan(
                recommended_model="claude-opus-5-5",
                model_tier="opus",
                quantization="none",
                batching_strategy="static",
                use_prompt_caching=True,
                use_speculative_decoding=False,
                reasoning=reasoning,
            )

        # Default: balanced Sonnet
        reasoning.append(
            "No binding constraint detected: defaulting to balanced Sonnet configuration."
        )
        reasoning.append(
            "Sonnet provides strong quality/cost/latency balance for moderate workloads."
        )
        return OptimizationPlan(
            recommended_model="claude-sonnet-5-5",
            model_tier="sonnet",
            quantization="none",
            batching_strategy="dynamic",
            use_prompt_caching=True,
            use_speculative_decoding=False,
            reasoning=reasoning,
        )
```

### Tests

```python
"""tests/week10/day6/test_decision.py"""

import pytest
from inference_toolkit.decision import DecisionGuide, RequestProfile


@pytest.fixture
def guide() -> DecisionGuide:
    return DecisionGuide()


def test_ultra_low_latency_selects_haiku(guide: DecisionGuide) -> None:
    plan = guide.recommend(
        RequestProfile(
            latency_budget_ms=150.0,
            cost_budget_per_1m=10.0,
            target_rps=10.0,
        )
    )
    assert plan.model_tier == "haiku"
    assert plan.recommended_model == "claude-haiku-5-5"
    assert plan.use_speculative_decoding is True


def test_cost_constrained_uses_quantization(guide: DecisionGuide) -> None:
    plan = guide.recommend(
        RequestProfile(
            latency_budget_ms=500.0,
            cost_budget_per_1m=0.50,
            target_rps=5.0,
        )
    )
    assert plan.model_tier == "haiku"
    assert plan.quantization == "int8"
    assert plan.use_prompt_caching is True


def test_high_throughput_selects_sonnet_dynamic_batch(guide: DecisionGuide) -> None:
    plan = guide.recommend(
        RequestProfile(
            latency_budget_ms=800.0,
            cost_budget_per_1m=15.0,
            target_rps=200.0,
        )
    )
    assert plan.model_tier == "sonnet"
    assert plan.batching_strategy == "dynamic"


def test_complex_task_selects_opus(guide: DecisionGuide) -> None:
    plan = guide.recommend(
        RequestProfile(
            latency_budget_ms=2000.0,
            cost_budget_per_1m=50.0,
            target_rps=5.0,
            task_complexity="complex",
        )
    )
    assert plan.model_tier == "opus"
    assert plan.recommended_model == "claude-opus-5-5"
    assert plan.use_prompt_caching is True


def test_default_selects_sonnet(guide: DecisionGuide) -> None:
    plan = guide.recommend(
        RequestProfile(
            latency_budget_ms=1000.0,
            cost_budget_per_1m=20.0,
            target_rps=50.0,
            task_complexity="moderate",
        )
    )
    assert plan.model_tier == "sonnet"


def test_all_plans_have_reasoning(guide: DecisionGuide) -> None:
    profiles = [
        RequestProfile(latency_budget_ms=100.0, cost_budget_per_1m=10.0, target_rps=5.0),
        RequestProfile(latency_budget_ms=500.0, cost_budget_per_1m=0.30, target_rps=5.0),
        RequestProfile(latency_budget_ms=800.0, cost_budget_per_1m=15.0, target_rps=200.0),
        RequestProfile(latency_budget_ms=2000.0, cost_budget_per_1m=50.0, target_rps=5.0, task_complexity="complex"),
    ]
    for profile in profiles:
        plan = guide.recommend(profile)
        assert len(plan.reasoning) >= 1
```

---

## Integrated Module Structure

```python
"""
src/inference_toolkit/__init__.py

Public API for the inference optimization toolkit.
"""

from inference_toolkit.cost import (
    CostResult,
    PricingConfig,
    TokenCostCalculator,
    MODEL_PRICING,
)
from inference_toolkit.streaming import (
    LLMBackend,
    MockBackend,
    AnthropicBackend,
    StreamEvent,
    StreamMetrics,
    StreamingClient,
)
from inference_toolkit.load_test import (
    LoadTestConfig,
    LoadTestResult,
    RequestResult,
    LoadTestHarness,
)
from inference_toolkit.decision import (
    RequestProfile,
    OptimizationPlan,
    DecisionGuide,
)

__all__ = [
    # cost
    "CostResult",
    "PricingConfig",
    "TokenCostCalculator",
    "MODEL_PRICING",
    # streaming
    "LLMBackend",
    "MockBackend",
    "AnthropicBackend",
    "StreamEvent",
    "StreamMetrics",
    "StreamingClient",
    # load test
    "LoadTestConfig",
    "LoadTestResult",
    "RequestResult",
    "LoadTestHarness",
    # decision
    "RequestProfile",
    "OptimizationPlan",
    "DecisionGuide",
]
```

### Integration Example

```python
"""
Demonstrate all four components working together.
All using MockBackend — no external API calls required.
"""

import asyncio
from inference_toolkit import (
    TokenCostCalculator,
    MockBackend,
    StreamingClient,
    LoadTestHarness,
    LoadTestConfig,
    DecisionGuide,
    RequestProfile,
)


async def demo() -> None:
    # 1. Cost calculation
    calc = TokenCostCalculator()
    cost = calc.compute_cost("claude-sonnet-5-5", 1000, 500, cache_hit_tokens=200)
    print(f"Request cost: ${cost.total_cost:.6f}")

    # 2. Streaming with metrics
    backend = MockBackend(tokens=["Hello", " from", " streaming"], delay_ms=20.0)
    client = StreamingClient(backend)
    _, metrics = await client.stream_with_metrics("hi", "claude-haiku-5-5")
    print(f"TTFT: {metrics.ttft_ms:.1f} ms, tokens: {metrics.total_tokens}")

    # 3. Load test
    async def mock_fn(prompt: str, model: str) -> int:
        await asyncio.sleep(0.01)
        return 50

    harness = LoadTestHarness(mock_fn)
    result = await harness.run(LoadTestConfig(n_requests=20, concurrency=5))
    print(f"P95 latency: {result.p95_latency_ms:.1f} ms, RPS: {result.throughput_rps:.1f}")

    # 4. Decision guide
    guide = DecisionGuide()
    plan = guide.recommend(
        RequestProfile(latency_budget_ms=300.0, cost_budget_per_1m=5.0, target_rps=50.0)
    )
    print(f"Recommended: {plan.recommended_model} ({plan.batching_strategy} batching)")


if __name__ == "__main__":
    asyncio.run(demo())
```

---

## Key Takeaways

1. **Separation of concerns**: each component has a single responsibility and a clean interface.
   Cost is pure math; streaming handles I/O; load testing handles concurrency; decisions handle logic.

2. **Protocol-based backends**: `LLMBackend` as a `Protocol` means all tests run with
   `MockBackend` — no API keys, no network, deterministic timing.

3. **Pydantic models at boundaries**: every input and output crossing module boundaries is
   a typed Pydantic model, making serialization, validation, and documentation automatic.

4. **DecisionGuide is constraint-based, not ML-based**: explicit priority order (latency first,
   then cost, then throughput, then complexity) makes reasoning transparent and testable.
