# Week 10 Project — Reference Implementation Notes

This document is the grader's reference for the Week 10 project: build an inference
optimization gateway that routes LLM requests, measures latency, and applies
quantization/caching strategies. It covers grading criteria, common mistakes,
the reference file layout, key design decisions, integration guidance, and stretch
goal hints.

---

## 1. Grading Guide

### Deliverable 1 — LLMBackend Protocol and at least two concrete backends (20 points)

**Full marks require:**
- A `Protocol` class (not ABC) named `LLMBackend` in `src/gateway/backends/base.py`
  with at minimum: `def complete(self, request: CompletionRequest) -> CompletionResponse`
  and `async def acomplete(self, request: CompletionRequest) -> CompletionResponse`
- At least two implementations: `AnthropicBackend` (real API, using `anthropic` SDK)
  and `MockBackend` (deterministic, no network call, used in tests)
- `CompletionRequest` and `CompletionResponse` as Pydantic v2 `BaseModel` subclasses;
  both typed fully — no `dict` fields crossing the boundary
- All API credentials sourced via `os.getenv("ANTHROPIC_API_KEY")`; no hardcoded keys
- `logging.getLogger(__name__)` at module level; no bare `print` calls

**Partial marks:**
- ABC instead of Protocol: −4 points (see design decision below)
- Untyped `dict` for request/response: −3 points
- `print` used anywhere in non-test code: −2 points
- Missing async variant: −2 points

---

### Deliverable 2 — LoadTestHarness with asyncio concurrency (25 points)

**Full marks require:**
- `LoadTestHarness` in `src/gateway/harness.py` using `asyncio` (not `threading`
  or `multiprocessing`)
- `run(rps: float, duration_s: float) -> HarnessResult` method that drives
  `asyncio.gather` or `asyncio.Semaphore`-controlled tasks
- `HarnessResult` Pydantic model with at minimum: `p50_latency_ms`, `p95_latency_ms`,
  `p99_latency_ms`, `throughput_rps`, `error_rate`, `total_requests`
- Percentile calculation uses actual measured values (not estimated); `statistics.quantiles`
  or `numpy.percentile` acceptable
- Tests in `tests/unit/test_harness.py` that mock the backend and verify percentile
  calculation correctness

**Partial marks:**
- `threading.Thread` instead of `asyncio`: −8 points
- Percentile calculated incorrectly (mean used, or off-by-one): −5 points
- Missing `error_rate` in result: −3 points
- No tests for the harness: −5 points

---

### Deliverable 3 — DecisionGuide with pure routing logic (20 points)

**Full marks require:**
- `DecisionGuide` in `src/gateway/decision.py` with no I/O, no LLM calls,
  no external dependencies beyond Pydantic and the standard library
- `decide(profile: RequestProfile) -> RoutingDecision` is a pure function
  (given the same input, always returns the same output)
- `RoutingDecision` is a Pydantic model with: `model`, `use_cache`, `quantization`,
  `rationale`
- Decision logic documented with inline comments explaining thresholds
- Full unit test coverage (every branch path tested)

**Partial marks:**
- Any I/O or LLM call inside `DecisionGuide`: −10 points
- Missing rationale field: −3 points
- Test coverage < 80% of branches: −5 points

---

### Deliverable 4 — FastAPI gateway wiring (20 points)

**Full marks require:**
- `src/gateway/main.py` with a FastAPI app
- `POST /complete` endpoint that: validates input with Pydantic, calls `DecisionGuide`,
  selects the appropriate backend, returns `CompletionResponse`
- `GET /health` endpoint returning `{"status": "ok"}`
- Startup logging: model name, API key presence (not the value), caching enabled/disabled
- Integration test in `tests/integration/test_gateway.py` using `httpx.AsyncClient`
  with `TestClient` or `AsyncClient(app=app)`

**Partial marks:**
- No health endpoint: −3 points
- Input not validated by Pydantic (raw `dict` from request body): −4 points
- No integration test: −6 points

---

### Deliverable 5 — Tests (15 points)

**Full marks require:**
- `tests/unit/` mirrors `src/gateway/` directory structure
- Every public function/class has at least one unit test
- All external calls (HTTP, LLM API) are mocked with `unittest.mock.patch`
  or `pytest-mock`
- `pytest.mark.integration` applied to integration tests; `pytest.mark.unit`
  to unit tests
- All tests pass with `pytest --tb=short -q`

**Partial marks:**
- Missing mocks (real HTTP in unit tests): −5 points
- No `pytest.mark` categories: −2 points
- < 60% overall coverage: −5 points

---

## 2. Common Mistakes and How to Avoid Them

### Mistake 1: Storing raw dicts as state

**What it looks like:**
```python
# Wrong
def complete(self, request: dict) -> dict:
    model = request["model"]
    ...
```

**Why it fails:** Untyped dicts cross module boundaries silently. A missing key raises
`KeyError` at runtime, not at request-parse time. Pydantic catches this at the boundary.

**Fix:** Define `CompletionRequest(BaseModel)` with all fields typed. FastAPI's `Body`
parameter declaration handles parsing and returns a 422 with a clear message on bad input.

---

### Mistake 2: Using threading instead of asyncio for the load test harness

**What it looks like:**
```python
# Wrong
threads = [Thread(target=self._run_request, args=(i,)) for i in range(n)]
for t in threads: t.start()
for t in threads: t.join()
```

**Why it fails:** `Thread` per request does not scale past ~100–200 concurrent requests
before GIL contention and OS scheduler overhead dominate. The Python GIL means CPU-bound
work does not parallelise across threads. An HTTP-bound load test at 500 rps would
require 500 threads, which is impractical.

**Fix:** Use `asyncio.gather` with a `Semaphore` to cap concurrency:

```python
import asyncio
import time

sem = asyncio.Semaphore(max_concurrent)

async def one_request() -> float:
    async with sem:
        t0 = time.monotonic()
        await backend.acomplete(request)
        return (time.monotonic() - t0) * 1000  # ms

tasks = [one_request() for _ in range(total_requests)]
latencies = await asyncio.gather(*tasks, return_exceptions=True)
```

This scales to thousands of concurrent requests with a single OS thread.

---

### Mistake 3: Putting routing logic inside the FastAPI endpoint

**What it looks like:**
```python
@app.post("/complete")
async def complete(req: CompletionRequest):
    if req.input_tokens > 500:
        model = "claude-haiku-5-5"
    elif req.latency_budget < 1000:
        model = "claude-sonnet-5-5"
    ...
```

**Why it fails:** Routing logic embedded in the handler cannot be unit-tested without
spinning up the full FastAPI app. It also couples the transport layer to the business logic.

**Fix:** Move all routing logic into `DecisionGuide.decide()`. The handler becomes:

```python
@app.post("/complete")
async def complete(req: CompletionRequest) -> CompletionResponse:
    decision = guide.decide(req.to_profile())
    backend = registry.get(decision.model)
    return await backend.acomplete(req)
```

`DecisionGuide` can be tested with a plain `pytest` function — no HTTP, no mocking.

---

### Mistake 4: Hardcoding model IDs or API keys

**What it looks like:**
```python
# Wrong
client = anthropic.Anthropic(api_key="sk-ant-...")
response = client.messages.create(model="claude-3-5-sonnet-20241022", ...)
```

**Why it fails:** Hardcoded secrets will be caught by the pre-commit hook and blocked.
Hardcoded model IDs cause failures when models are updated and the string no longer maps
to a valid model.

**Fix:**
```python
import os
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# Model IDs from config, never literals in business logic
```

Use only: `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-opus-5-5`.

---

### Mistake 5: Calculating percentiles from summary statistics

**What it looks like:**
```python
# Wrong — this is the mean, not p95
p95 = sum(latencies) / len(latencies) * 1.5
```

**Fix:** Collect all raw latency values and use `statistics.quantiles` or `numpy.percentile`:

```python
import statistics

sorted_latencies = sorted(lat for lat in latencies if isinstance(lat, float))
p50 = statistics.quantiles(sorted_latencies, n=100)[49]  # index 49 = 50th percentile
p95 = statistics.quantiles(sorted_latencies, n=100)[94]
p99 = statistics.quantiles(sorted_latencies, n=100)[98]
```

---

### Mistake 6: Missing async/await on backend calls

**What it looks like:**
```python
# Wrong — blocking call inside async context
async def complete_endpoint(req: CompletionRequest):
    response = sync_backend.complete(req)  # blocks the event loop
    return response
```

**Fix:** Either use `await backend.acomplete(req)` if the backend is async-native, or
wrap a sync call in `asyncio.to_thread`:

```python
import asyncio
response = await asyncio.to_thread(sync_backend.complete, req)
```

Blocking the event loop in an async context serialises all concurrent requests, defeating
the purpose of asyncio.

---

## 3. Reference Implementation File Layout

```
week-10-project/
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── main.py                  # FastAPI app, lifespan, /complete, /health
│       ├── config.py                # pydantic-settings Settings class
│       ├── decision.py              # DecisionGuide (pure logic, no I/O)
│       ├── harness.py               # LoadTestHarness (asyncio)
│       ├── models.py                # CompletionRequest, CompletionResponse,
│       │                            #   RequestProfile, RoutingDecision,
│       │                            #   HarnessResult
│       └── backends/
│           ├── __init__.py
│           ├── base.py              # LLMBackend Protocol + BackendRegistry
│           ├── anthropic_backend.py # AnthropicBackend (real API)
│           └── mock_backend.py      # MockBackend (tests, no network)
├── tests/
│   ├── conftest.py                  # shared fixtures: mock_backend, test_profile
│   ├── unit/
│   │   ├── test_decision.py         # DecisionGuide — all branches
│   │   ├── test_harness.py          # LoadTestHarness — percentile math
│   │   ├── test_models.py           # Pydantic validation edge cases
│   │   └── backends/
│   │       └── test_mock_backend.py
│   └── integration/
│       └── test_gateway.py          # POST /complete, GET /health via AsyncClient
├── .env.example
├── .gitignore                       # must include .env
├── pyproject.toml                   # pytest config, ruff, black
└── README.md
```

**Key constraint:** `src/gateway/decision.py` must import nothing from `backends/`.
`main.py` is the only module that knows about both `decision.py` and `backends/`.
This enforces separation of concerns and keeps `DecisionGuide` independently testable.

---

## 4. Key Design Decisions

### Why Protocol over ABC for LLMBackend

```python
# base.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, request: CompletionRequest) -> CompletionResponse: ...
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse: ...
```

**Reason:** `Protocol` enables structural subtyping. Any class that implements `complete`
and `acomplete` with the correct signatures is automatically a valid `LLMBackend` without
inheriting from it. This means:

1. `MockBackend` in tests does not need to import from the production module tree —
   it just implements the right methods.
2. Third-party clients (e.g. a Bedrock wrapper, a local Ollama client) can satisfy
   the protocol without modification.
3. `isinstance(backend, LLMBackend)` works at runtime with `@runtime_checkable`,
   enabling registry validation.

With `ABC`, every backend must explicitly inherit `class AnthropicBackend(LLMBackend)`,
creating a hard coupling. `Protocol` avoids that coupling while retaining type safety.

---

### Why asyncio for LoadTestHarness (not threading)

Load testing requires sending many concurrent requests. The bottleneck is network I/O,
not CPU. For I/O-bound concurrency:

- `threading`: each thread occupies ~1 MB of stack, holds a Python GIL slot during
  Python code execution, and has non-trivial OS context-switch overhead. Practical
  limit ~200–500 concurrent threads before performance degrades.
- `asyncio`: a single OS thread runs a cooperative event loop. `await` yields control
  without a thread switch. 10,000 concurrent coroutines share one OS thread; the limit
  is memory (each coroutine ~1–2 KB) and file descriptor count, not thread overhead.

For a load test harness targeting 100–1,000 rps, asyncio is the only practical choice
in pure Python. The `asyncio.Semaphore` controls maximum concurrency to avoid overwhelming
the target service:

```python
async def run(self, rps: float, duration_s: float) -> HarnessResult:
    interval = 1.0 / rps
    sem = asyncio.Semaphore(int(rps * 2))  # allow burst of 2x rps
    tasks: list[asyncio.Task[float]] = []
    start = asyncio.get_event_loop().time()

    while asyncio.get_event_loop().time() - start < duration_s:
        tasks.append(asyncio.create_task(self._timed_request(sem)))
        await asyncio.sleep(interval)

    latencies = await asyncio.gather(*tasks, return_exceptions=True)
    return self._compute_stats(latencies)
```

---

### Why Pydantic for all cross-module data

Every value that crosses a module boundary in this project is a Pydantic `BaseModel`:

- `CompletionRequest`, `CompletionResponse` — FastAPI endpoint I/O
- `RequestProfile` — input to `DecisionGuide`
- `RoutingDecision` — output from `DecisionGuide`
- `HarnessResult` — output from `LoadTestHarness`

**Three concrete reasons:**

1. **Validation at the boundary:** A `RequestProfile` with `avg_input_tokens=-1` is
   caught immediately by `Field(gt=0)` rather than propagating into routing logic and
   producing a nonsensical recommendation.
2. **Serialisation for free:** `plan.model_dump_json()` and `HarnessResult.model_validate(data)`
   work without writing custom serialisation code. This matters when persisting test
   results or logging decisions.
3. **Self-documenting contracts:** The Pydantic model is the canonical contract between
   modules. When a field is added to `RoutingDecision`, every consumer that accesses
   a field that no longer exists fails at import time or test time, not silently at
   runtime.

The rule: no untyped `dict`, `Any`, or `TypedDict` crosses a module boundary.
Internal use of dicts within a function is fine; function signatures and return types
must use Pydantic models.

---

### Why pure logic in DecisionGuide (testable without mocking)

`DecisionGuide.decide()` takes a `RequestProfile` and returns a `RoutingDecision`.
It has no parameters that reference external state, makes no I/O calls, and has no
side effects.

**Consequence:** every test for `DecisionGuide` is a straightforward value test:

```python
def test_routes_high_rps_to_haiku():
    guide = DecisionGuide()
    profile = RequestProfile(
        avg_input_tokens=100,
        avg_output_tokens=20,
        requests_per_second=50.0,
        latency_budget_ms=5_000,
        cost_budget_per_1k_requests=0.02,
    )
    decision = guide.decide(profile)
    assert decision.model == "claude-haiku-5-5"
```

No `@patch`, no fixtures, no async. This makes the routing logic easy to reason about
and fast to test (sub-millisecond per test case).

If routing logic were inside the FastAPI handler, every test would require an HTTP
client, a running ASGI server, and a mocked backend — adding ~50–200 ms per test and
making the logic harder to cover exhaustively.

---

## 5. Integration Notes: Wiring FastAPI to Real vs Mock LLM

### Structure

```python
# src/gateway/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .backends.anthropic_backend import AnthropicBackend
from .backends.mock_backend import MockBackend
from .backends.base import BackendRegistry
from .decision import DecisionGuide
from .config import Settings

settings = Settings()  # reads ANTHROPIC_API_KEY, USE_MOCK_BACKEND, etc.

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = BackendRegistry()
    if settings.use_mock_backend:
        registry.register("claude-haiku-5-5", MockBackend(latency_ms=50))
        registry.register("claude-sonnet-5-5", MockBackend(latency_ms=100))
        registry.register("claude-opus-5-5", MockBackend(latency_ms=200))
    else:
        backend = AnthropicBackend(api_key=settings.anthropic_api_key)
        for model in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]:
            registry.register(model, backend)
    app.state.registry = registry
    app.state.guide = DecisionGuide()
    yield

app = FastAPI(lifespan=lifespan)
```

### AnthropicBackend — real API

```python
# src/gateway/backends/anthropic_backend.py
import os
import logging
from anthropic import Anthropic, AsyncAnthropic
from ..models import CompletionRequest, CompletionResponse
from .base import LLMBackend

logger = logging.getLogger(__name__)


class AnthropicBackend:
    """Real Anthropic API backend. Model ID is passed per-request."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or pass api_key= to AnthropicBackend()."
            )
        self._sync = Anthropic(api_key=key)
        self._async = AsyncAnthropic(api_key=key)
        logger.info("AnthropicBackend initialised (key present: True)")

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        msg = self._sync.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=[{"role": "user", "content": request.prompt}],
        )
        return CompletionResponse(
            content=msg.content[0].text,
            model=msg.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        msg = await self._async.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=[{"role": "user", "content": request.prompt}],
        )
        return CompletionResponse(
            content=msg.content[0].text,
            model=msg.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
```

**Only valid model IDs for this project:** `claude-haiku-5-5`, `claude-sonnet-5-5`,
`claude-opus-5-5`. Never use `-4-x` suffixed variants.

### MockBackend — tests and local development

```python
# src/gateway/backends/mock_backend.py
import asyncio
import logging
from ..models import CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)


class MockBackend:
    """
    Deterministic backend for testing.
    Returns a canned response after simulated latency.
    Does not make any network calls.
    """

    def __init__(self, latency_ms: int = 100, response_text: str = "mock response") -> None:
        self._latency_s = latency_ms / 1000.0
        self._response_text = response_text

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        import time
        time.sleep(self._latency_s)
        return self._make_response(request)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        await asyncio.sleep(self._latency_s)
        return self._make_response(request)

    def _make_response(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(
            content=self._response_text,
            model=request.model,
            input_tokens=len(request.prompt.split()),
            output_tokens=len(self._response_text.split()),
        )
```

### Switching between real and mock

Set `USE_MOCK_BACKEND=true` in `.env` for local development and CI. Set
`USE_MOCK_BACKEND=false` (or omit) for production. The `Settings` class handles this:

```python
# src/gateway/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    use_mock_backend: bool = False
    log_level: str = "INFO"

    model_config = {"env_file": ".env"}
```

Integration tests should always use `MockBackend` so they run without an API key
in CI. Reserve `AnthropicBackend` for manual smoke tests and the load test harness
when measuring real latency.

---

## 6. Stretch Goal Hints

### Prompt Caching Metrics

To measure cache hit rates against the Anthropic API, record `usage.cache_read_input_tokens`
and `usage.cache_creation_input_tokens` from each response:

```python
# In AnthropicBackend.acomplete, after the API call:
cache_read = getattr(msg.usage, "cache_read_input_tokens", 0)
cache_write = getattr(msg.usage, "cache_creation_input_tokens", 0)
logger.info(
    "cache_read=%d cache_write=%d input=%d output=%d",
    cache_read, cache_write,
    msg.usage.input_tokens, msg.usage.output_tokens,
)
```

Aggregate these in `HarnessResult` with additional fields:
```python
class HarnessResult(BaseModel):
    ...
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    cache_hit_rate: float = 0.0  # cache_read / (cache_read + cache_write + uncached)
```

To actually trigger cache hits: set `cache_control` on the system prompt in your
`CompletionRequest` and ensure `avg_input_tokens > 512` (minimum cacheable prefix).
The load test harness naturally sends the same prompt repeatedly, so cache hit rate
should approach the `CACHE_HIT_RATE` constant used in `InferenceOptimizer`.

---

### Speculative Decoding Simulation

Implement `SpeculativeDecodingSimulator` in `src/gateway/speculative.py`:

```python
import random
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpeculativeConfig(BaseModel):
    k: int = Field(default=5, gt=0, description="Draft tokens per step")
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Acceptance rate")
    draft_cost_ratio: float = Field(
        default=0.1, gt=0.0, le=1.0,
        description="Draft forward pass cost relative to target"
    )


class SpeculativeResult(BaseModel):
    tokens_generated: int
    draft_tokens_accepted: int
    draft_tokens_rejected: int
    theoretical_speedup: float
    simulated_speedup: float


class SpeculativeDecodingSimulator:
    """
    Simulates speculative decoding acceptance.

    Does not call a real model. Uses the per-token acceptance probability
    to simulate whether each draft token would be accepted.
    """

    def __init__(self, config: SpeculativeConfig | None = None) -> None:
        self.config = config or SpeculativeConfig()

    def simulate(self, target_tokens: int) -> SpeculativeResult:
        k = self.config.k
        alpha = self.config.alpha
        cost_ratio = self.config.draft_cost_ratio

        accepted = 0
        rejected = 0
        steps = 0

        remaining = target_tokens
        while remaining > 0:
            draft_count = min(k, remaining)
            step_accepted = 0
            for _ in range(draft_count):
                if random.random() < alpha:
                    step_accepted += 1
                    accepted += 1
                else:
                    rejected += 1
                    break  # rejection stops the step
            remaining -= step_accepted + 1  # +1 for the target's bonus token
            steps += 1

        theoretical = (1 + k * alpha) / (1 + k * cost_ratio)
        simulated = target_tokens / max(steps, 1) if steps > 0 else 1.0

        logger.debug(
            "speculative: tokens=%d steps=%d accepted=%d rejected=%d speedup=%.2f",
            target_tokens, steps, accepted, rejected, simulated,
        )

        return SpeculativeResult(
            tokens_generated=target_tokens,
            draft_tokens_accepted=accepted,
            draft_tokens_rejected=rejected,
            theoretical_speedup=round(theoretical, 3),
            simulated_speedup=round(simulated, 3),
        )
```

**Testing this:** run with `alpha=0.7`, `k=5`, `cost_ratio=0.1` over 10,000 tokens
and verify `simulated_speedup` converges toward `theoretical_speedup` (~3.0). Due to
random sampling, expect ±10% variance with 10,000 tokens and near-exact convergence
at 100,000 tokens.

**Connection to the gateway:** Wire `SpeculativeDecodingSimulator` into `MockBackend`
so the mock backend produces latency values that reflect the speedup. This lets the
load test harness show measurably lower TBT when speculative decoding is "enabled",
making the simulation pedagogically meaningful without requiring a real draft model.
