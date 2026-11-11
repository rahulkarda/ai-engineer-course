# Phase 6 · Week 10 · Day 7 — Week Project: Optimization Toolkit + LLM Gateway

> Objective: Wire the day-6 toolkit into a FastAPI service with SSE streaming, cost accounting, and a /metrics endpoint — then assess the full week.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6.md]]

## Concept

All week we built in isolation: a calculator, a streaming client, a load-test harness, a decision guide. Today we wire them into a **FastAPI service** that mirrors a production LLM gateway:

```
Client
  │
  ├── POST /chat          → StreamingClient → SSE token stream back to client
  │                          + TokenCostCalculator logs cost per request
  │
  └── GET  /metrics       → LoadTestHarness results (from last background run)
                             + aggregated cost totals
```

This is the pattern week 9 introduced for gateway design: the optimization toolkit sits *inside* the gateway, not alongside it. Every request is metered, every cost is logged, and the metrics endpoint gives operators a live view of the system.

**Mockable by default**: in tests, `StreamingClient` uses `MockBackend` — no real API calls. In production, swap to `AnthropicBackend` via an environment flag.

## How it works internally

End-to-end request flow:

```
POST /chat {"prompt": "...", "max_tokens": 256}
    │
    ├── RequestValidator (Pydantic)
    │
    ├── StreamingClient.call(prompt)
    │     └── MockBackend (tests) | AnthropicBackend (production)
    │
    ├── TokenCostCalculator.compute(model, input_tokens, output_tokens)
    │     └── CostBreakdown logged to RequestLog
    │
    ├── RequestLog appended to MetricsStore (in-memory list)
    │
    └── StreamingResponse (SSE)
          "data: {token}\n\n" for each token
          "data: [DONE]\n\n"

GET /metrics
    └── MetricsStore.summarize() → MetricsSummary
          (total_requests, total_cost, p50/p95/p99 latency, cache stats)
```

**Concurrency note**: FastAPI uses an async event loop. `StreamingClient` is async-native. `MetricsStore` uses a simple `asyncio.Lock` for thread-safe appends.

**SSE format**: FastAPI `StreamingResponse` with `media_type="text/event-stream"`. Each token is sent as `data: {json}\n\n`. The `[DONE]` sentinel signals end of stream. This matches the Anthropic streaming wire format and is compatible with `EventSource` in browsers.

## Worked example

```python
"""
day7_gateway_service.py

FastAPI LLM gateway with:
  - POST /chat          SSE streaming endpoint
  - GET  /metrics       Aggregated latency + cost stats
  - POST /load-test     Trigger async load test (uses MockBackend)

Run:
    uvicorn day7_gateway_service:app --reload --port 8000

Test:
    pytest day7_gateway_service.py -v

Curl example (streaming):
    curl -N http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"prompt": "What is Python?", "max_tokens": 64}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Import toolkit modules (assumes same file or installed package)
# In a real project these would be: from inference_toolkit import ...
# For this lesson they are inline imports from day-6 code.
# We redefine the essential parts here for self-contained demo.

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inline toolkit (subset from day-6 — import from package in production)
# ---------------------------------------------------------------------------

# Pricing (same as day-6)
_PRICING: dict[str, dict] = {
    "claude-haiku-5-5":  {"input": 0.80, "output": 4.00, "cache_read": 0.08},
    "claude-sonnet-5-5": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-opus-5-5":   {"input": 15.00, "output": 75.00, "cache_read": 1.50},
}

DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")
USE_REAL_API = os.getenv("USE_REAL_API", "false").lower() == "true"


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Return total cost in USD."""
    p = _PRICING.get(model, _PRICING["claude-haiku-5-5"])
    return (
        input_tokens * p["input"] / 1_000_000
        + output_tokens * p["output"] / 1_000_000
        + cache_read_tokens * p["cache_read"] / 1_000_000
    )


# ---------------------------------------------------------------------------
# Mock streaming backend (no real API calls)
# ---------------------------------------------------------------------------

class MockStreamBackend:
    """
    Configurable mock: simulates streaming with adjustable delays.
    Inject a real Anthropic client for production use.
    """

    def __init__(
        self,
        ttft_ms: float = 80.0,
        token_delay_ms: float = 25.0,
        tokens_per_response: int = 12,
    ) -> None:
        self.ttft_ms = ttft_ms
        self.token_delay_ms = token_delay_ms
        self.tokens_per_response = tokens_per_response
        self._words = (
            "the quick brown fox jumps over lazy dog and finds treasure "
            "underneath a sprawling oak tree in the forest of inference"
        ).split()

    async def stream_tokens(
        self, prompt: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        await asyncio.sleep(self.ttft_ms / 1000)
        limit = min(max_tokens, self.tokens_per_response, len(self._words))
        for i in range(limit):
            yield self._words[i % len(self._words)] + " "
            await asyncio.sleep(self.token_delay_ms / 1000)

    def estimate_tokens(self, text: str) -> int:
        """Rough token count: ~0.75 words per token."""
        return max(1, int(len(text.split()) / 0.75))


# ---------------------------------------------------------------------------
# Metrics store
# ---------------------------------------------------------------------------

class RequestLog(BaseModel):
    """One logged request."""
    request_id: str
    prompt_preview: str  # first 60 chars
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: float


class MetricsSummary(BaseModel):
    """Aggregated metrics for GET /metrics."""
    total_requests: int
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_model: dict[str, int]
    last_n_errors: list[str] = Field(default_factory=list)


import math
import statistics as _stats


def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


class MetricsStore:
    """Thread-safe in-memory log of requests."""

    def __init__(self) -> None:
        self._logs: list[RequestLog] = []
        self._lock = asyncio.Lock()
        self._errors: list[str] = []

    async def append(self, log_entry: RequestLog) -> None:
        async with self._lock:
            self._logs.append(log_entry)

    async def append_error(self, error: str) -> None:
        async with self._lock:
            self._errors.append(error)
            if len(self._errors) > 50:
                self._errors = self._errors[-50:]

    async def summarize(self) -> MetricsSummary:
        async with self._lock:
            logs = list(self._logs)
            errors = list(self._errors)

        if not logs:
            return MetricsSummary(
                total_requests=0,
                total_cost_usd=0.0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cache_read_tokens=0,
                mean_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                requests_per_model={},
                last_n_errors=errors[-5:],
            )

        latencies = [r.latency_ms for r in logs]
        model_counts: dict[str, int] = {}
        for r in logs:
            model_counts[r.model] = model_counts.get(r.model, 0) + 1

        return MetricsSummary(
            total_requests=len(logs),
            total_cost_usd=round(sum(r.cost_usd for r in logs), 8),
            total_input_tokens=sum(r.input_tokens for r in logs),
            total_output_tokens=sum(r.output_tokens for r in logs),
            total_cache_read_tokens=sum(r.cache_read_tokens for r in logs),
            mean_latency_ms=round(_stats.mean(latencies), 2),
            p50_latency_ms=round(_pct(latencies, 50), 2),
            p95_latency_ms=round(_pct(latencies, 95), 2),
            p99_latency_ms=round(_pct(latencies, 99), 2),
            requests_per_model=model_counts,
            last_n_errors=errors[-5:],
        )


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

@dataclass
class AppState:
    backend: MockStreamBackend = field(default_factory=MockStreamBackend)
    metrics: MetricsStore = field(default_factory=MetricsStore)


_state = AppState()


@asynccontextmanager
async def lifespan(application: FastAPI):
    log.info("Starting gateway (model=%s, real_api=%s)", DEFAULT_MODEL, USE_REAL_API)
    yield
    log.info("Shutting down gateway")


app = FastAPI(title="LLM Inference Gateway", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8192)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    model: str = Field(default=DEFAULT_MODEL)
    cache_read_tokens: int = Field(default=0, ge=0)

    def model_post_init(self, __context) -> None:
        if self.model not in _PRICING:
            raise ValueError(f"Unsupported model: {self.model}")


class LoadTestRequest(BaseModel):
    n_requests: int = Field(default=20, ge=1, le=500)
    max_concurrency: int = Field(default=5, ge=1, le=50)
    prompt: str = Field(default="What is the speed of light?")
    max_tokens: int = Field(default=32, ge=1, le=256)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    Stream LLM response as SSE.
    Logs cost and latency to MetricsStore after stream completes.
    """
    request_id = str(uuid.uuid4())[:8]
    backend = _state.backend
    metrics = _state.metrics

    async def event_stream() -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        output_tokens = 0
        error_msg: str | None = None

        try:
            async for token in backend.stream_tokens(request.prompt, request.max_tokens):
                output_tokens += 1
                payload = json.dumps({"token": token, "request_id": request_id})
                yield f"data: {payload}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as exc:
            error_msg = str(exc)
            log.error("Stream error for %s: %s", request_id, exc)
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            input_tokens = backend.estimate_tokens(request.prompt)
            cost = compute_cost(
                request.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=request.cache_read_tokens,
            )

            log_entry = RequestLog(
                request_id=request_id,
                prompt_preview=request.prompt[:60],
                model=request.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=request.cache_read_tokens,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost, 8),
                timestamp=time.time(),
            )
            await metrics.append(log_entry)

            if error_msg:
                await metrics.append_error(f"[{request_id}] {error_msg}")

            log.info(
                "Request %s: latency=%.1fms cost=$%.8f tokens=%d+%d",
                request_id,
                latency_ms,
                cost,
                input_tokens,
                output_tokens,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Request-ID": request_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/metrics", response_model=MetricsSummary)
async def get_metrics() -> MetricsSummary:
    """Return aggregated latency and cost metrics."""
    return await _state.metrics.summarize()


@app.post("/load-test", response_model=MetricsSummary)
async def run_load_test(request: LoadTestRequest) -> MetricsSummary:
    """
    Fire N concurrent mock requests, collect stats, return summary.
    Uses MockStreamBackend directly — does not hit the real API.
    Does NOT use the /chat endpoint (avoids SSE overhead in tests).
    """
    backend = MockStreamBackend(ttft_ms=50, token_delay_ms=10)
    semaphore = asyncio.Semaphore(request.max_concurrency)
    latencies: list[float] = []
    errors: list[str] = []

    async def single(i: int) -> None:
        async with semaphore:
            start = time.perf_counter()
            try:
                tokens = []
                async for token in backend.stream_tokens(request.prompt, request.max_tokens):
                    tokens.append(token)
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception as exc:
                errors.append(str(exc))

    await asyncio.gather(*[single(i) for i in range(request.n_requests)])

    n = len(latencies)
    model_counts = {DEFAULT_MODEL: n}

    if not latencies:
        return MetricsSummary(
            total_requests=request.n_requests,
            total_cost_usd=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cache_read_tokens=0,
            mean_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            requests_per_model={},
            last_n_errors=errors[:5],
        )

    # Estimate cost per request
    input_toks = backend.estimate_tokens(request.prompt)
    output_toks = request.max_tokens
    cost_per_req = compute_cost(DEFAULT_MODEL, input_toks, output_toks)

    return MetricsSummary(
        total_requests=request.n_requests,
        total_cost_usd=round(cost_per_req * n, 8),
        total_input_tokens=input_toks * n,
        total_output_tokens=output_toks * n,
        total_cache_read_tokens=0,
        mean_latency_ms=round(_stats.mean(latencies), 2),
        p50_latency_ms=round(_pct(latencies, 50), 2),
        p95_latency_ms=round(_pct(latencies, 95), 2),
        p99_latency_ms=round(_pct(latencies, 99), 2),
        requests_per_model=model_counts,
        last_n_errors=errors[:5],
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": DEFAULT_MODEL}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from httpx import AsyncClient, ASGITransport  # noqa: E402


@pytest.fixture
def reset_state():
    """Reset global state before each test."""
    _state.backend = MockStreamBackend(ttft_ms=5, token_delay_ms=2)
    _state.metrics = MetricsStore()
    yield


@pytest.fixture
def client(reset_state):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_returns_sse_content_type(self, client):
        async with client as c:
            async with c.stream(
                "POST", "/chat",
                json={"prompt": "hello", "max_tokens": 5},
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.asyncio
    async def test_streams_data_lines(self, client):
        chunks = []
        async with client as c:
            async with c.stream(
                "POST", "/chat",
                json={"prompt": "hi", "max_tokens": 4},
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        chunks.append(line)
        # Expect at least one token + [DONE]
        assert len(chunks) >= 2
        assert any("[DONE]" in c for c in chunks)

    @pytest.mark.asyncio
    async def test_metrics_updated_after_request(self, client):
        async with client as c:
            # Fire one request and consume the stream
            async with c.stream("POST", "/chat", json={"prompt": "test", "max_tokens": 3}) as r:
                async for _ in r.aiter_bytes():
                    pass
            # Allow event_stream finally block to run
            await asyncio.sleep(0.05)
            resp = await c.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 1
        assert data["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_invalid_model_rejected(self, client):
        async with client as c:
            resp = await c.post("/chat", json={"prompt": "hi", "model": "gpt-fake"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_request_id_header_present(self, client):
        async with client as c:
            async with c.stream("POST", "/chat", json={"prompt": "hello"}) as r:
                assert "x-request-id" in r.headers


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_empty_metrics(self, client):
        async with client as c:
            resp = await c.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 0
        assert data["total_cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_after_multiple_requests(self, client):
        async with client as c:
            for _ in range(3):
                async with c.stream("POST", "/chat", json={"prompt": "x", "max_tokens": 2}) as r:
                    async for _ in r.aiter_bytes():
                        pass
            await asyncio.sleep(0.1)
            resp = await c.get("/metrics")
        data = resp.json()
        assert data["total_requests"] == 3

    @pytest.mark.asyncio
    async def test_percentiles_ordered(self, client):
        async with client as c:
            for i in range(5):
                async with c.stream("POST", "/chat", json={"prompt": "test", "max_tokens": 2}) as r:
                    async for _ in r.aiter_bytes():
                        pass
            await asyncio.sleep(0.1)
            resp = await c.get("/metrics")
        data = resp.json()
        assert data["p50_latency_ms"] <= data["p95_latency_ms"] <= data["p99_latency_ms"]


class TestLoadTestEndpoint:
    @pytest.mark.asyncio
    async def test_load_test_returns_summary(self, client):
        async with client as c:
            resp = await c.post(
                "/load-test",
                json={"n_requests": 5, "max_concurrency": 3, "max_tokens": 4},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 5
        assert data["p50_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_load_test_cost_positive(self, client):
        async with client as c:
            resp = await c.post("/load-test", json={"n_requests": 3, "max_tokens": 8})
        data = resp.json()
        assert data["total_cost_usd"] > 0


class TestComputeCost:
    def test_zero_tokens(self):
        assert compute_cost("claude-haiku-5-5", 0, 0) == 0.0

    def test_haiku_1m_input(self):
        cost = compute_cost("claude-haiku-5-5", 1_000_000, 0)
        assert abs(cost - 0.80) < 0.001

    def test_cache_cheaper(self):
        normal = compute_cost("claude-haiku-5-5", 1000, 0, cache_read_tokens=0)
        cached = compute_cost("claude-haiku-5-5", 0, 0, cache_read_tokens=1000)
        assert cached < normal


class TestMetricsStore:
    @pytest.mark.asyncio
    async def test_empty_summary(self):
        store = MetricsStore()
        summary = await store.summarize()
        assert summary.total_requests == 0

    @pytest.mark.asyncio
    async def test_append_and_summarize(self):
        store = MetricsStore()
        entry = RequestLog(
            request_id="abc",
            prompt_preview="hello",
            model="claude-haiku-5-5",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            latency_ms=200.0,
            cost_usd=0.0001,
            timestamp=time.time(),
        )
        await store.append(entry)
        summary = await store.summarize()
        assert summary.total_requests == 1
        assert summary.total_cost_usd == 0.0001
        assert summary.mean_latency_ms == 200.0
```

## Exercises

1. **Persist metrics to disk**: Add a background task (using FastAPI's `BackgroundTasks` or an `asyncio.Task`) that writes `MetricsStore` contents to a JSON file every 60 seconds. The file path should come from `os.getenv("METRICS_FILE", "metrics.json")`. Write a test that verifies the file is written correctly after appending 3 log entries.

2. **Add prompt caching to the /chat endpoint**: Extend `ChatRequest` with a `system_prompt: str = ""` field. When `len(system_prompt) > 512`, construct the Anthropic API call with `cache_control: {"type": "ephemeral"}` on the system prompt content block. Update `RequestLog` to capture `cache_creation_tokens`. Write a test that mocks the Anthropic client and verifies `cache_control` appears in the request payload when the system prompt is long enough.

3. **Week integration challenge**: Wire `DecisionGuide` from day-6 into a new `GET /recommend` endpoint. Accept `latency_budget_ms`, `cost_budget_per_1k_usd`, and `throughput_req_per_s` as query parameters. Return a `Recommendation` JSON. Write 3 tests: tight latency returns haiku, loose latency with high cost budget returns opus, and an impossibly low cost budget returns warnings.

(Solutions: ../solutions/week-10/day-7.md)

## Week 10 Self-check

**Quantization (day-1/2)**
- [ ] I can explain the difference between INT8 and INT4 quantization and the quality trade-off.
- [ ] I can describe GPTQ, AWQ, and GGUF and when to use each.

**Batching and throughput (day-3)**
- [ ] I can explain why static batching wastes GPU memory and how continuous batching fixes it.
- [ ] I can implement a naive static batcher and measure its throughput vs. a request-at-a-time baseline.

**Speculative decoding + serving engines (day-4)**
- [ ] I can explain why speculative decoding has no quality loss (acceptance sampling proof sketch).
- [ ] I can describe vLLM's PagedAttention and continuous batching scheduler.
- [ ] I can state two reasons to choose TGI over vLLM.

**Streaming + caching (day-5)**
- [ ] I can implement a streaming client that measures TTFT and inter-token latency.
- [ ] I can explain exact caching vs. semantic caching and their respective failure modes.
- [ ] I can use Anthropic `cache_control` and read cache hit metrics from the response.

**Toolkit (day-6)**
- [ ] I can implement `TokenCostCalculator` with correct cache savings math from memory.
- [ ] I can write a load test harness using `asyncio.gather()` and a semaphore.
- [ ] I can identify the dead code bug in `DecisionGuide._recommend_quantization`.

**Integration (day-7)**
- [ ] I can build a FastAPI SSE endpoint that streams tokens and logs cost per request.
- [ ] I can wire `MetricsStore` into a `/metrics` endpoint returning p50/p95/p99 latency.
- [ ] I can write integration tests for all three endpoints using `httpx.AsyncClient` + `ASGITransport`.

## References

- FastAPI streaming responses: https://fastapi.tiangolo.com/advanced/custom-response/
- Anthropic streaming API: https://docs.anthropic.com/en/api/messages-streaming
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- httpx async client for testing: https://www.python-httpx.org/async/
