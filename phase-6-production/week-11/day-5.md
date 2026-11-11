# Phase 6 · Week 11 · Day 5 — LLM and Agent Observability: Token Usage, Cost, and Latency

> Objective: Build a typed observer that tracks token usage, USD cost, and latency for every LLM call and exposes the data as Prometheus metrics.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

### Why LLM observability is different

Standard service observability asks: "Is the server up? How fast does it respond?" LLM observability adds a financial dimension and a *capacity* dimension that traditional services lack.

| Traditional service | LLM service |
|---|---|
| CPU / memory usage | Token usage (input + output) |
| Request count | LLM call count per model |
| P99 latency | Time-to-first-token (TTFT) + total latency |
| Error rate | Refusal rate, truncation rate |
| Cost ≈ infra | Cost = tokens × per-token price (varies by model) |

Without LLM-specific observability, you will:

- Overpay without knowing which workflow is the most expensive.
- Hit context window limits silently (responses truncated, no error raised).
- Blame "the model" for slowness when the real issue is large prompts.
- Lose money to runaway retry loops.

### Token tracking

The Anthropic API response includes `usage.input_tokens` and `usage.output_tokens`. Track both:

- **Input tokens**: driven by prompt length + system prompt + conversation history.
- **Output tokens**: driven by `max_tokens` cap and the actual response length.
- **Context window pressure**: `input_tokens / context_window_size` — signals when you are close to the limit.

### Cost tracking

Each model has a fixed price per million tokens (as of the model IDs used in this course):

| Model | Input $/M | Output $/M |
|---|---|---|
| `claude-opus-5-5` | 15.00 | 75.00 |
| `claude-sonnet-5-5` | 3.00 | 15.00 |
| `claude-haiku-5-5` | 0.80 | 4.00 |

Always store cost as `float` USD and accumulate it per session or per workflow run. A `$0.001` individual call becomes meaningful when multiplied by 10,000 daily runs.

### Latency metrics

- **TTFT (Time-to-First-Token)**: how long until the first token streams. Relevant for interactive UX. Measured by starting a timer before the API call and recording when `stream.text_stream.__anext__()` first returns.
- **Total latency**: wall-clock time from request start to response complete.
- **Per-step latency** in agents: each tool call + LLM call measured independently. A slow step compounds in multi-turn agents.

### Agent-specific metrics

Agents introduce additional dimensions beyond single LLM calls:

- **Steps per run**: high step counts signal loops or over-planning.
- **Tool call count**: tracks which tools are used and how often.
- **Retry count**: excessive retries indicate brittle tool interfaces or bad prompts.
- **Branching factor**: how many paths the agent explored (for tree-search agents).

### LangSmith (concept)

LangSmith is LangChain's hosted tracing and evaluation platform. It auto-instruments LangGraph agents to record:

- Full message trees (each LLM call, tool call, and result)
- Token counts and cost estimates per run
- Latency per node
- Replay and comparison of run versions

To use it, set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in your environment. Zero code changes needed if you use LangChain/LangGraph. This lesson builds the *same concepts* from scratch so you understand what LangSmith is measuring.

---

## How it works internally

1. **Before each LLM call**: record `start_time = time.perf_counter()`.
2. **After the call returns**: extract `usage.input_tokens`, `usage.output_tokens`, `model` from the response.
3. **Compute cost**: `cost = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out`.
4. **Record**: create an `LLMCallRecord` Pydantic model and append it to an in-memory list (or write to a time-series DB in production).
5. **Expose to Prometheus**: use `prometheus_client` counters/histograms so Grafana can scrape the data.
6. **Summarise**: compute totals + averages on demand.

Prometheus metric types used:

| Metric | Type | Labels |
|---|---|---|
| `llm_tokens_total` | Counter | `model`, `token_type` (input/output) |
| `llm_cost_usd_total` | Counter | `model` |
| `llm_request_duration_seconds` | Histogram | `model` |

---

## Worked example

Install dependencies:

```bash
uv add prometheus-client anthropic pydantic
```

```python
# llm_observer.py
"""
LLM observability: token tracking, cost accounting, latency measurement,
and Prometheus metric exposure.

Tests must MOCK all LLM calls — no real API calls in test suite.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import anthropic
from prometheus_client import Counter, Histogram, start_http_server
from pydantic import BaseModel, Field, computed_field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── 1. Cost table ──────────────────────────────────────────────────────────────

# Prices in USD per 1,000,000 tokens (check Anthropic pricing page for updates)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {
        "input_per_million": 15.00,
        "output_per_million": 75.00,
        "context_window": 200_000,
    },
    "claude-sonnet-5-5": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "context_window": 200_000,
    },
    "claude-haiku-5-5": {
        "input_per_million": 0.80,
        "output_per_million": 4.00,
        "context_window": 200_000,
    },
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute USD cost for one LLM call.

    Args:
        model: Model ID string (must be in MODEL_PRICING).
        input_tokens: Tokens consumed by the prompt.
        output_tokens: Tokens generated in the response.

    Returns:
        Cost in USD as a float.

    Raises:
        KeyError: If model is not in MODEL_PRICING.
    """
    pricing = MODEL_PRICING[model]
    cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    cost += (output_tokens / 1_000_000) * pricing["output_per_million"]
    return round(cost, 8)


# ── 2. Pydantic models ─────────────────────────────────────────────────────────

class LLMCallRecord(BaseModel):
    """Immutable record of a single LLM API call."""

    model: str = Field(description="Model ID used for this call")
    input_tokens: int = Field(ge=0, description="Prompt token count")
    output_tokens: int = Field(ge=0, description="Response token count")
    cost_usd: float = Field(ge=0.0, description="USD cost for this call")
    latency_ms: float = Field(ge=0.0, description="Wall-clock time from request to response")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_preview: str = Field(
        default="",
        max_length=200,
        description="First 200 chars of the prompt (for debugging — never log full prompts in prod)",
    )
    success: bool = Field(default=True)
    error_message: str | None = Field(default=None)

    @computed_field  # type: ignore[misc]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @computed_field  # type: ignore[misc]
    @property
    def context_window_pct(self) -> float:
        """Percentage of the model's context window consumed by the input."""
        pricing = MODEL_PRICING.get(self.model)
        if not pricing:
            return 0.0
        return round((self.input_tokens / pricing["context_window"]) * 100, 2)


class SummaryStats(BaseModel):
    """Aggregated statistics over a list of LLMCallRecords."""

    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    p99_latency_ms: float
    calls_by_model: dict[str, int]
    cost_by_model: dict[str, float]
    error_count: int


# ── 3. Prometheus metrics ──────────────────────────────────────────────────────

# Counters accumulate forever — Grafana computes rates with rate()
_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens processed by the LLM",
    labelnames=["model", "token_type"],  # token_type: "input" | "output"
)

_COST_TOTAL = Counter(
    "llm_cost_usd_total",
    "Cumulative USD cost of LLM calls",
    labelnames=["model"],
)

# Histograms track distribution — used for P50/P95/P99 latency
_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM call wall-clock latency",
    labelnames=["model"],
    # Buckets tuned for LLM latency (seconds)
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)


# ── 4. Observer class ──────────────────────────────────────────────────────────

class LLMObserver:
    """
    Wraps LLM calls to record token usage, cost, and latency.

    Usage:
        observer = LLMObserver()
        record = observer.call(model="claude-haiku-5-5", prompt="Hello")
        print(observer.summary())
    """

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        logger.info("LLMObserver initialised")

    def _record_metrics(self, record: LLMCallRecord) -> None:
        """Push one call's data into Prometheus counters/histograms."""
        _TOKENS_TOTAL.labels(model=record.model, token_type="input").inc(record.input_tokens)
        _TOKENS_TOTAL.labels(model=record.model, token_type="output").inc(record.output_tokens)
        _COST_TOTAL.labels(model=record.model).inc(record.cost_usd)
        _REQUEST_DURATION.labels(model=record.model).observe(record.latency_ms / 1000)

    def call(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        system: str = "You are a helpful assistant.",
    ) -> LLMCallRecord:
        """
        Make a single-turn LLM call and return a full observability record.

        Args:
            model: One of claude-opus-5-5, claude-sonnet-5-5, claude-haiku-5-5.
            prompt: User message.
            max_tokens: Maximum tokens in the response.
            system: System prompt.

        Returns:
            LLMCallRecord with tokens, cost, latency, and the response text.
        """
        if model not in MODEL_PRICING:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODEL_PRICING)}")

        start = time.perf_counter()
        error_msg: str | None = None
        success = True
        input_tokens = 0
        output_tokens = 0

        try:
            logger.info("Calling %s (max_tokens=%d)", model, max_tokens)
            response = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            logger.info(
                "Response: input=%d output=%d tokens",
                input_tokens,
                output_tokens,
            )

        except anthropic.APIError as exc:
            success = False
            error_msg = str(exc)
            logger.error("LLM call failed: %s", exc)

        latency_ms = (time.perf_counter() - start) * 1000
        cost = compute_cost(model, input_tokens, output_tokens)

        record = LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round(latency_ms, 2),
            prompt_preview=prompt[:200],
            success=success,
            error_message=error_msg,
        )

        self._records.append(record)
        self._record_metrics(record)

        logger.info(
            "Recorded: model=%s tokens=%d cost=$%.6f latency=%.1fms",
            model,
            record.total_tokens,
            cost,
            latency_ms,
        )
        return record

    def summary(self) -> SummaryStats:
        """
        Compute aggregate statistics over all recorded calls.

        Returns:
            SummaryStats with totals, averages, and per-model breakdowns.
        """
        if not self._records:
            return SummaryStats(
                total_calls=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cost_usd=0.0,
                avg_latency_ms=0.0,
                p99_latency_ms=0.0,
                calls_by_model={},
                cost_by_model={},
                error_count=0,
            )

        latencies = sorted(r.latency_ms for r in self._records)
        p99_idx = max(0, int(len(latencies) * 0.99) - 1)

        calls_by_model: dict[str, int] = {}
        cost_by_model: dict[str, float] = {}
        for r in self._records:
            calls_by_model[r.model] = calls_by_model.get(r.model, 0) + 1
            cost_by_model[r.model] = round(
                cost_by_model.get(r.model, 0.0) + r.cost_usd, 8
            )

        return SummaryStats(
            total_calls=len(self._records),
            total_input_tokens=sum(r.input_tokens for r in self._records),
            total_output_tokens=sum(r.output_tokens for r in self._records),
            total_cost_usd=round(sum(r.cost_usd for r in self._records), 8),
            avg_latency_ms=round(sum(latencies) / len(latencies), 2),
            p99_latency_ms=latencies[p99_idx],
            calls_by_model=calls_by_model,
            cost_by_model=cost_by_model,
            error_count=sum(1 for r in self._records if not r.success),
        )

    def records(self) -> list[LLMCallRecord]:
        """Return a copy of all recorded calls (immutable access)."""
        return list(self._records)


# ── 5. Test-ready mock LLM caller ──────────────────────────────────────────────

@dataclass
class MockLLMResponse:
    """Minimal stand-in for anthropic.types.Message — used in tests."""
    input_tokens: int
    output_tokens: int
    text: str = "mock response"


def make_mock_observer(
    fake_responses: list[MockLLMResponse] | None = None,
) -> LLMObserver:
    """
    Return an LLMObserver whose _client.messages.create is monkey-patched
    to return fake responses without any network calls.

    Intended for use in tests:
        observer = make_mock_observer([MockLLMResponse(input_tokens=100, output_tokens=50)])
        record = observer.call("claude-haiku-5-5", "test prompt")
    """
    import unittest.mock as mock

    responses = fake_responses or [MockLLMResponse(input_tokens=100, output_tokens=50)]
    response_iter = iter(responses)

    def _fake_create(**kwargs: Any) -> Any:  # noqa: ANN401
        resp = next(response_iter, MockLLMResponse(input_tokens=10, output_tokens=10))
        # Build a minimal object that looks like anthropic.types.Message
        usage_mock = mock.Mock()
        usage_mock.input_tokens = resp.input_tokens
        usage_mock.output_tokens = resp.output_tokens
        msg_mock = mock.Mock()
        msg_mock.usage = usage_mock
        return msg_mock

    observer = LLMObserver.__new__(LLMObserver)
    observer._records = []  # noqa: SLF001
    mock_client = mock.Mock()
    mock_client.messages.create.side_effect = _fake_create
    observer._client = mock_client  # noqa: SLF001
    return observer


# ── 6. Example usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Start Prometheus metrics HTTP server on port 8001
    # Grafana/Prometheus scrapes http://localhost:8001/metrics
    start_http_server(8001)
    logger.info("Prometheus metrics available at http://localhost:8001/metrics")

    # Use the mock observer for the demo (no real API key needed)
    fake = [
        MockLLMResponse(input_tokens=512, output_tokens=128),
        MockLLMResponse(input_tokens=1024, output_tokens=256),
        MockLLMResponse(input_tokens=300, output_tokens=75),
    ]
    observer = make_mock_observer(fake)

    models = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    prompts = [
        "What is observability?",
        "Explain distributed tracing in detail.",
        "How do I reduce LLM costs?",
    ]

    for model, prompt in zip(models, prompts, strict=True):
        record = observer.call(model=model, prompt=prompt)
        logger.info(
            "Call complete: total_tokens=%d cost=$%.6f context_window=%.2f%%",
            record.total_tokens,
            record.cost_usd,
            record.context_window_pct,
        )

    stats = observer.summary()
    logger.info("=== Summary ===")
    logger.info("Total calls:   %d", stats.total_calls)
    logger.info("Total tokens:  %d", stats.total_input_tokens + stats.total_output_tokens)
    logger.info("Total cost:    $%.6f", stats.total_cost_usd)
    logger.info("Avg latency:   %.1f ms", stats.avg_latency_ms)
    logger.info("Cost by model: %s", stats.cost_by_model)
```

### Test file

```python
# tests/unit/test_llm_observer.py
"""
Unit tests for LLMObserver.
All LLM calls are MOCKED — no real API calls, no ANTHROPIC_API_KEY required.
"""

from __future__ import annotations

import pytest

from llm_observer import (
    LLMCallRecord,
    LLMObserver,
    MockLLMResponse,
    SummaryStats,
    compute_cost,
    make_mock_observer,
)


class TestComputeCost:
    def test_haiku_cost(self) -> None:
        cost = compute_cost("claude-haiku-5-5", input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 4.80) < 0.01

    def test_opus_cost(self) -> None:
        cost = compute_cost("claude-opus-5-5", input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 90.00) < 0.01

    def test_zero_tokens(self) -> None:
        assert compute_cost("claude-haiku-5-5", 0, 0) == 0.0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(KeyError):
            compute_cost("gpt-99-turbo", 100, 100)


class TestLLMCallRecord:
    def test_total_tokens_computed(self) -> None:
        record = LLMCallRecord(
            model="claude-haiku-5-5",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0001,
            latency_ms=250.0,
        )
        assert record.total_tokens == 150

    def test_context_window_pct(self) -> None:
        record = LLMCallRecord(
            model="claude-haiku-5-5",
            input_tokens=20_000,
            output_tokens=0,
            cost_usd=0.0,
            latency_ms=100.0,
        )
        # 20_000 / 200_000 = 10.0%
        assert record.context_window_pct == 10.0


class TestLLMObserver:
    def test_single_call_records_are_stored(self) -> None:
        observer = make_mock_observer([MockLLMResponse(input_tokens=200, output_tokens=100)])
        record = observer.call(model="claude-haiku-5-5", prompt="hello")

        assert isinstance(record, LLMCallRecord)
        assert record.input_tokens == 200
        assert record.output_tokens == 100
        assert record.model == "claude-haiku-5-5"
        assert record.success is True
        assert len(observer.records()) == 1

    def test_cost_computed_correctly(self) -> None:
        observer = make_mock_observer([MockLLMResponse(input_tokens=1_000_000, output_tokens=0)])
        record = observer.call(model="claude-haiku-5-5", prompt="big prompt")
        assert abs(record.cost_usd - 0.80) < 0.001

    def test_summary_over_multiple_calls(self) -> None:
        observer = make_mock_observer(
            [
                MockLLMResponse(input_tokens=100, output_tokens=50),
                MockLLMResponse(input_tokens=200, output_tokens=100),
            ]
        )
        observer.call(model="claude-haiku-5-5", prompt="first")
        observer.call(model="claude-sonnet-5-5", prompt="second")

        stats = observer.summary()
        assert stats.total_calls == 2
        assert stats.total_input_tokens == 300
        assert stats.total_output_tokens == 150
        assert stats.calls_by_model["claude-haiku-5-5"] == 1
        assert stats.calls_by_model["claude-sonnet-5-5"] == 1

    def test_empty_summary(self) -> None:
        observer = make_mock_observer()
        stats = observer.summary()
        assert stats.total_calls == 0
        assert stats.total_cost_usd == 0.0

    def test_unknown_model_raises(self) -> None:
        observer = make_mock_observer()
        with pytest.raises(ValueError, match="Unknown model"):
            observer.call(model="gpt-99", prompt="test")

    def test_prompt_preview_truncated(self) -> None:
        observer = make_mock_observer([MockLLMResponse(input_tokens=10, output_tokens=5)])
        long_prompt = "x" * 500
        record = observer.call(model="claude-haiku-5-5", prompt=long_prompt)
        assert len(record.prompt_preview) <= 200
```

---

## Exercises

1. Add a `ttft_ms` (time-to-first-token) field to `LLMCallRecord`. Modify `LLMObserver.call` to use the Anthropic streaming API (`stream=True`) and record the time until the first chunk arrives. Add a `llm_ttft_seconds` Prometheus histogram.

2. Extend `LLMObserver` to support per-session tracking: add a `session_id: str` parameter to `call()`, store it in the record, and add a `session_summary(session_id: str) -> SummaryStats` method. Write tests that verify isolation between sessions.

3. Add a `budget_guard` decorator (or context manager) that raises a `BudgetExceededError` if the cumulative cost for a session exceeds a configurable `max_usd` threshold. Write a test that verifies it fires at the right call.

4. Write a function `cost_forecast(records: list[LLMCallRecord], daily_call_volume: int) -> dict[str, float]` that takes historical records, computes average cost per call per model, and returns a projected daily and monthly cost dict. Include edge-case tests for empty records and single-model history.

(Solutions: ../solutions/week-11/day-5.md)

---

## Self-check

- [ ] I can explain the difference between TTFT and total latency and why both matter for different LLM use cases.
- [ ] I can compute the USD cost of a Claude API call given input/output token counts and the pricing table, and explain why output tokens are more expensive.
- [ ] I can implement a Prometheus Counter vs. Histogram and explain which to use for cost vs. latency.
- [ ] I can write a pytest test that exercises `LLMObserver` without making any real network calls, using mock objects for the Anthropic client.
