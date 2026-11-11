# Week 10 · Day 7 Solutions — FastAPI Inference Gateway

---

## Overview

This is the complete FastAPI application that integrates all four Day 6 toolkit components
behind a unified HTTP gateway. It is designed to slot in alongside the Week 9 gateway:
see the integration notes at the bottom of this file.

```
src/gateway/
    __init__.py
    app.py          # FastAPI app + dependency injection
    dependencies.py # FastAPI Depends providers
    schemas.py      # Request / response Pydantic models
    routers/
        chat.py
        metrics.py
        load_test.py
        cost.py
        decision.py
tests/week10/day7/
    conftest.py
    test_chat.py
    test_metrics.py
    test_load_test.py
    test_cost.py
    test_decision.py
```

---

## Pydantic Schemas

```python
"""
src/gateway/schemas.py

Request and response schemas for the Inference Gateway API.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ---- /chat ----

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=100_000)
    model: str = Field(default="claude-haiku-5-5")


class ChatTokenStats(BaseModel):
    ttft_ms: float
    total_tokens: int
    total_elapsed_ms: float


class ChatResponse(BaseModel):
    text: str
    model: str
    token_stats: ChatTokenStats


# ---- /load-test ----

class LoadTestRequest(BaseModel):
    n_requests: int = Field(default=20, ge=1, le=500)
    concurrency: int = Field(default=5, ge=1, le=50)
    prompt: str = Field(default="Hello, respond in exactly 5 words.")
    model: str = Field(default="claude-haiku-5-5")


# ---- /cost ----

class CostRequest(BaseModel):
    model: str = Field(default="claude-haiku-5-5")
    input_tokens: int = Field(default=500, ge=0)
    output_tokens: int = Field(default=200, ge=0)
    cache_hit_tokens: int = Field(default=0, ge=0)


# ---- /decision ----

class DecisionRequest(BaseModel):
    latency_budget_ms: float = Field(default=500.0, gt=0)
    cost_budget_per_1m: float = Field(default=10.0, gt=0)
    target_rps: float = Field(default=10.0, gt=0)
    avg_input_tokens: int = Field(default=500, gt=0)
    avg_output_tokens: int = Field(default=200, gt=0)
    requires_streaming: bool = False
    task_complexity: Literal["simple", "moderate", "complex"] = "moderate"
```

---

## Dependency Injection

```python
"""
src/gateway/dependencies.py

FastAPI dependency providers for all toolkit components.
Uses MockBackend by default so the app runs without an API key in development/tests.
Set ANTHROPIC_API_KEY and USE_REAL_BACKEND=1 to switch to AnthropicBackend.
"""

import asyncio
import logging
import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from inference_toolkit import (
    TokenCostCalculator,
    MockBackend,
    StreamingClient,
    LoadTestHarness,
    DecisionGuide,
    LoadTestResult,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_backend() -> MockBackend:
    """
    Returns MockBackend for development/testing.
    In production swap for AnthropicBackend when ANTHROPIC_API_KEY is set.
    """
    use_real = os.getenv("USE_REAL_BACKEND", "0") == "1"
    if use_real:
        # Deferred import to avoid hard dependency in test environments
        from inference_toolkit import AnthropicBackend
        logger.info("Using AnthropicBackend (real API calls)")
        return AnthropicBackend()  # type: ignore[return-value]
    logger.info("Using MockBackend (no API calls)")
    return MockBackend(
        tokens=["Mocked", " response", " token", "."],
        delay_ms=15.0,
    )


def get_streaming_client() -> StreamingClient:
    return StreamingClient(_get_backend())


def get_calculator() -> TokenCostCalculator:
    return TokenCostCalculator()


def get_decision_guide() -> DecisionGuide:
    return DecisionGuide()


# In-memory store for the most recent load test result
_last_load_test_result: LoadTestResult | None = None


def get_last_load_test_result() -> LoadTestResult | None:
    return _last_load_test_result


def set_last_load_test_result(result: LoadTestResult) -> None:
    global _last_load_test_result
    _last_load_test_result = result


# Type aliases for injection
StreamingClientDep = Annotated[StreamingClient, Depends(get_streaming_client)]
CalculatorDep = Annotated[TokenCostCalculator, Depends(get_calculator)]
DecisionGuideDep = Annotated[DecisionGuide, Depends(get_decision_guide)]
```

---

## Main Application

```python
"""
src/gateway/app.py

FastAPI Inference Gateway — Week 10 Day 7.

Endpoints:
    POST /chat          Stream a response and return text + streaming stats
    GET  /metrics       Return the most recent load test summary
    POST /load-test     Run a concurrent load test against the mock/real backend
    GET  /cost          Compute cost for a given token profile
    POST /decision      Return an optimization recommendation

Week 9 Integration Notes
-------------------------
This gateway is designed to sit behind the Week 9 API Gateway.
- Prefix all routes with /v1/inference (configure via the ROOT_PATH env var
  or FastAPI's root_path parameter).
- The Week 9 gateway handles authentication (API key validation) and rate limiting;
  this service trusts the forwarded X-Request-ID header for tracing.
- Health check: GET /health (returns 200 {"status": "ok"}).

Example nginx / reverse-proxy config:
    location /v1/inference/ {
        proxy_pass http://inference-gateway:8000/;
    }
"""

import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from inference_toolkit import LoadTestConfig, RequestProfile

from gateway.dependencies import (
    CalculatorDep,
    DecisionGuideDep,
    StreamingClientDep,
    get_last_load_test_result,
    set_last_load_test_result,
    get_streaming_client,
    get_calculator,
    get_decision_guide,
)
from gateway.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTokenStats,
    CostRequest,
    DecisionRequest,
    LoadTestRequest,
)
from inference_toolkit import LoadTestHarness

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inference Optimization Gateway",
    version="1.0.0",
    root_path=os.getenv("ROOT_PATH", ""),
)


# ---- Health ----

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---- /chat (SSE streaming) ----

@app.post("/chat")
async def chat(
    request: ChatRequest,
    client: StreamingClientDep,
) -> StreamingResponse:
    """
    Stream a response as Server-Sent Events.

    Each SSE event has the form:
        data: {"token": "...", "token_index": N, "elapsed_ms": X}\n\n

    A final event with type "done" carries streaming metrics:
        event: done
        data: {"ttft_ms": X, "total_tokens": N, "total_elapsed_ms": Y}\n\n
    """
    import json

    async def event_generator():
        events, metrics = await client.stream_with_metrics(request.prompt, request.model)
        for event in events:
            payload = json.dumps({
                "token": event.token,
                "token_index": event.token_index,
                "elapsed_ms": round(event.elapsed_ms, 2),
            })
            yield f"data: {payload}\n\n"
        # Final metrics event
        done_payload = json.dumps({
            "ttft_ms": round(metrics.ttft_ms, 2),
            "total_tokens": metrics.total_tokens,
            "total_elapsed_ms": round(metrics.total_elapsed_ms, 2),
        })
        yield f"event: done\ndata: {done_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---- /metrics ----

@app.get("/metrics")
async def get_metrics():
    """Return the most recent load test result summary."""
    result = get_last_load_test_result()
    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No load test has been run yet. POST /load-test first.",
        )
    return result.model_dump()


# ---- /load-test ----

@app.post("/load-test")
async def run_load_test(request: LoadTestRequest) -> dict:
    """
    Run N concurrent requests against the configured backend and store results.
    Returns the LoadTestResult summary.
    """
    backend = get_streaming_client()._backend  # type: ignore[attr-defined]

    async def request_fn(prompt: str, model: str) -> int:
        events, _ = await StreamingClient(backend).stream_with_metrics(prompt, model)
        return len(events)

    harness = LoadTestHarness(request_fn)
    config = LoadTestConfig(
        n_requests=request.n_requests,
        concurrency=request.concurrency,
        prompt=request.prompt,
        model=request.model,
    )
    result = await harness.run(config)
    set_last_load_test_result(result)
    logger.info(
        "Load test complete: %d requests, p95=%.1f ms, RPS=%.1f",
        result.total_requests,
        result.p95_latency_ms,
        result.throughput_rps,
    )
    return result.model_dump()


# ---- /cost ----

@app.get("/cost")
async def compute_cost(
    model: str = "claude-haiku-5-5",
    input_tokens: int = 500,
    output_tokens: int = 200,
    cache_hit_tokens: int = 0,
    calculator: TokenCostCalculator = None,
) -> dict:
    """
    Compute the USD cost for a given token profile.

    Query params: model, input_tokens, output_tokens, cache_hit_tokens
    """
    if calculator is None:
        calculator = get_calculator()
    try:
        result = calculator.compute_cost(model, input_tokens, output_tokens, cache_hit_tokens)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result.model_dump()


# ---- /decision ----

@app.post("/decision")
async def get_decision(
    request: DecisionRequest,
    guide: DecisionGuideDep,
) -> dict:
    """Return an optimization plan for the given workload profile."""
    profile = RequestProfile(
        latency_budget_ms=request.latency_budget_ms,
        cost_budget_per_1m=request.cost_budget_per_1m,
        target_rps=request.target_rps,
        avg_input_tokens=request.avg_input_tokens,
        avg_output_tokens=request.avg_output_tokens,
        requires_streaming=request.requires_streaming,
        task_complexity=request.task_complexity,
    )
    plan = guide.recommend(profile)
    return plan.model_dump()
```

---

## Test Suite

```python
"""
tests/week10/day7/conftest.py

Shared fixtures for Day 7 gateway tests.
All tests use ASGI transport — no real HTTP server or external API calls.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from gateway.app import app


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """Async HTTP client wired directly to the FastAPI ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
```

```python
"""
tests/week10/day7/test_chat.py
"""

import json
import pytest
import pytest_asyncio
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_streaming_returns_sse(client: AsyncClient) -> None:
    """
    POST /chat should return Content-Type: text/event-stream.
    The response body must contain at least one 'data:' line and a 'event: done' line.
    """
    response = await client.post(
        "/chat",
        json={"prompt": "Hello", "model": "claude-haiku-5-5"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    data_lines = [line for line in body.splitlines() if line.startswith("data:")]
    assert len(data_lines) >= 1

    done_lines = [line for line in body.splitlines() if line.strip() == "event: done"]
    assert len(done_lines) == 1


@pytest.mark.asyncio
async def test_chat_sse_events_are_valid_json(client: AsyncClient) -> None:
    """Each 'data:' line must carry valid JSON."""
    response = await client.post(
        "/chat",
        json={"prompt": "Test", "model": "claude-haiku-5-5"},
    )
    body = response.text
    for line in body.splitlines():
        if line.startswith("data:"):
            raw = line[len("data:"):].strip()
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_chat_done_event_has_metrics(client: AsyncClient) -> None:
    """The 'done' event payload must include ttft_ms and total_tokens."""
    response = await client.post("/chat", json={"prompt": "Test"})
    lines = response.text.splitlines()

    done_data = None
    for i, line in enumerate(lines):
        if line.strip() == "event: done" and i + 1 < len(lines):
            raw = lines[i + 1]
            if raw.startswith("data:"):
                done_data = json.loads(raw[len("data:"):].strip())
                break

    assert done_data is not None
    assert "ttft_ms" in done_data
    assert "total_tokens" in done_data
```

```python
"""
tests/week10/day7/test_metrics.py
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_metrics_404_before_load_test(client: AsyncClient) -> None:
    """Before any load test runs, GET /metrics should return 404."""
    # Reset the in-memory state (may have been set by a prior test)
    import gateway.dependencies as deps
    deps._last_load_test_result = None

    response = await client.get("/metrics")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_metrics_returns_summary_after_load_test(client: AsyncClient) -> None:
    """After running /load-test, GET /metrics should return a result summary."""
    await client.post(
        "/load-test",
        json={"n_requests": 5, "concurrency": 2},
    )
    response = await client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "p95_latency_ms" in data
    assert "throughput_rps" in data
```

```python
"""
tests/week10/day7/test_load_test.py
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_load_test_runs_concurrently(client: AsyncClient) -> None:
    """POST /load-test should return a full LoadTestResult summary."""
    response = await client.post(
        "/load-test",
        json={"n_requests": 10, "concurrency": 3, "model": "claude-haiku-5-5"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] == 10
    assert data["successful"] + data["failed"] == 10
    assert data["p50_latency_ms"] >= 0
    assert data["throughput_rps"] > 0


@pytest.mark.asyncio
async def test_load_test_stores_result_for_metrics(client: AsyncClient) -> None:
    """Running a load test should make /metrics return that result."""
    await client.post("/load-test", json={"n_requests": 3, "concurrency": 1})
    metrics_response = await client.get("/metrics")
    assert metrics_response.status_code == 200
    assert metrics_response.json()["total_requests"] == 3
```

```python
"""
tests/week10/day7/test_cost.py
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_cost_calculation_endpoint(client: AsyncClient) -> None:
    """GET /cost should return a valid CostResult for valid query params."""
    response = await client.get(
        "/cost",
        params={
            "model": "claude-haiku-5-5",
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_hit_tokens": 0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_cost" in data
    assert "input_cost" in data
    assert "output_cost" in data
    assert data["total_cost"] > 0


@pytest.mark.asyncio
async def test_cost_unknown_model_returns_400(client: AsyncClient) -> None:
    """Requesting cost for an unknown model should return HTTP 400."""
    response = await client.get(
        "/cost",
        params={"model": "claude-nonexistent-99-9", "input_tokens": 100, "output_tokens": 50},
    )
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_cost_zero_tokens_zero_cost(client: AsyncClient) -> None:
    response = await client.get(
        "/cost",
        params={"model": "claude-sonnet-5-5", "input_tokens": 0, "output_tokens": 0},
    )
    assert response.status_code == 200
    assert response.json()["total_cost"] == pytest.approx(0.0)
```

```python
"""
tests/week10/day7/test_decision.py
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_decision_guide_endpoint_returns_plan(client: AsyncClient) -> None:
    """POST /decision should return an OptimizationPlan."""
    response = await client.post(
        "/decision",
        json={
            "latency_budget_ms": 500.0,
            "cost_budget_per_1m": 5.0,
            "target_rps": 20.0,
            "task_complexity": "moderate",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "recommended_model" in data
    assert "model_tier" in data
    assert isinstance(data["reasoning"], list)
    assert len(data["reasoning"]) >= 1


@pytest.mark.asyncio
async def test_decision_ultra_low_latency_returns_haiku(client: AsyncClient) -> None:
    response = await client.post(
        "/decision",
        json={"latency_budget_ms": 100.0, "cost_budget_per_1m": 10.0, "target_rps": 5.0},
    )
    assert response.status_code == 200
    assert response.json()["model_tier"] == "haiku"


@pytest.mark.asyncio
async def test_decision_complex_returns_opus(client: AsyncClient) -> None:
    response = await client.post(
        "/decision",
        json={
            "latency_budget_ms": 3000.0,
            "cost_budget_per_1m": 100.0,
            "target_rps": 5.0,
            "task_complexity": "complex",
        },
    )
    assert response.status_code == 200
    assert response.json()["model_tier"] == "opus"
```

---

## Running the Tests

```bash
# Install dependencies
pip install fastapi uvicorn httpx pytest pytest-asyncio anyio

# Run all Day 7 tests
pytest tests/week10/day7/ -v

# Run the gateway locally (MockBackend, no API key required)
uvicorn gateway.app:app --reload --port 8000
```

---

## Week 9 Gateway Integration Notes

```python
# These are reference comments for connecting this service to the Week 9 gateway.
# No code changes are needed to the gateway itself — configure via environment variables.

# 1. Root path prefix
#    Set ROOT_PATH=/v1/inference so the OpenAPI docs and redirects work correctly
#    behind a reverse proxy.
#    os.environ["ROOT_PATH"] = "/v1/inference"

# 2. Authentication
#    The Week 9 gateway validates API keys before forwarding requests.
#    This service trusts the forwarded X-Authenticated-User header.
#    Add a middleware in app.py to read it if needed:
#
#    @app.middleware("http")
#    async def log_authenticated_user(request: Request, call_next):
#        user = request.headers.get("X-Authenticated-User", "anonymous")
#        logger.info("request from user=%s path=%s", user, request.url.path)
#        return await call_next(request)

# 3. Tracing
#    Forward the X-Request-ID header from the Week 9 gateway for distributed tracing.
#    Use a middleware to attach it to the logging context.

# 4. Production backend
#    Set USE_REAL_BACKEND=1 and ANTHROPIC_API_KEY=<key> to switch from MockBackend
#    to AnthropicBackend in the dependency injection layer (dependencies.py).

# 5. Concurrency limits
#    The /load-test endpoint should be guarded behind an admin route in the Week 9 gateway
#    to prevent accidental large-scale load tests in production.
```

---

## Key Takeaways

1. **FastAPI + Depends keeps components testable**: by injecting `StreamingClient`,
   `TokenCostCalculator`, and `DecisionGuide` through `Depends`, each endpoint can be
   tested with mocks or overridden per environment without changing application logic.

2. **SSE streaming works through ASGI**: `StreamingResponse` with `text/event-stream`
   content type flows correctly through FastAPI's ASGI stack and `httpx.ASGITransport`,
   making streaming fully testable without a real server.

3. **httpx.ASGITransport eliminates test infrastructure**: all five test modules run
   against the real FastAPI app with zero ports, zero subprocesses, and zero network calls.
   This is the canonical pattern for FastAPI testing.

4. **In-memory state is sufficient for /metrics at this scale**: a single global variable
   with a module-level setter is adequate in a single-process development environment.
   For production, swap for Redis or a database-backed store.
