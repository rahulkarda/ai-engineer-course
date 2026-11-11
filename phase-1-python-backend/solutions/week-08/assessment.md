# Week 08 Assessment — Answer Key

## Conceptual Questions

### Q1: Explain how the asyncio event loop works. Why is it single-threaded, and what does it actually do between your `await` calls?

The event loop is a loop that alternates between two activities:

1. **Poll the OS selector** (`select`/`epoll`/`kqueue`) for I/O readiness events —
   file descriptors that have data to read, connections that completed, timers that fired.
2. **Drain the ready queue** — run every callback/coroutine that is no longer waiting.

Single-threaded by design: there is exactly one call stack active at a time. When
your coroutine hits `await`, it yields control back to the loop, which runs the next
ready coroutine. The loop resumes your coroutine only when the awaited event fires.

**Between your `await` calls:** the loop is either running another coroutine's synchronous
code (no context switch possible), or blocked in the OS syscall waiting for I/O.
There is no preemption — your code runs until it explicitly awaits.

**Implication:** CPU-bound work that never awaits blocks the entire event loop. A tight
`for` loop computing hashes freezes all other coroutines until it finishes.

---

### Q2: When should you use asyncio, threads, or multiprocessing?

| Workload | Best tool | Reason |
|---|---|---|
| I/O-bound (HTTP, DB, file) | `asyncio` | Thousands of concurrent waits with one thread; minimal overhead |
| I/O-bound legacy blocking libs | `ThreadPoolExecutor` | Run blocking calls off the event loop thread |
| CPU-bound (parsing, hashing, ML) | `ProcessPoolExecutor` / `multiprocessing` | Bypasses the GIL; true parallelism across cores |
| Mixed I/O + light CPU | `asyncio` + `run_in_executor` | Offload the CPU part to a thread/process pool |

**Rule of thumb:** if the bottleneck is waiting for the network or disk, `asyncio`. If
the bottleneck is the CPU, processes. Threads are the escape hatch for blocking libraries
that haven't been ported to async.

---

### Q3: What is the GIL, why does it exist, and what does it affect?

The **Global Interpreter Lock** is a mutex inside CPython that ensures only one thread
executes Python bytecode at a time.

**Why it exists:** CPython's memory management (reference counting) is not thread-safe.
The GIL is a coarse-grained lock that prevents race conditions on the reference counts of
every Python object without requiring fine-grained per-object locks.

**What it affects:**

- **Python threads cannot parallelise CPU-bound code.** Two threads on an 8-core machine
  still execute one bytecode at a time.
- **I/O-bound threads are fine.** The GIL is released during blocking system calls
  (network, file), so multiple threads can wait for I/O concurrently.
- **C extensions can release the GIL.** NumPy, PyTorch, and most I/O code release it
  explicitly, enabling real parallelism for their C-level work.

**Python 3.13+:** introduces an opt-in free-threaded build (`--disable-gil`) that removes
the GIL. Stability and ecosystem compatibility are still maturing.

---

### Q4: What is a race condition? How do you detect it, and how do you prevent it?

A **race condition** occurs when the correctness of a program depends on the relative
timing or interleaving of multiple concurrent operations on shared state.

**Detection:**
- Code review: look for read-modify-write patterns on shared state across `await` points.
- Testing: run the code with many concurrent workers and check for non-deterministic
  results (counts off, missing data, duplicated work).
- Tools: `asyncio` has no race detector, but thread-level races can be found with
  ThreadSanitizer (`-fsanitize=thread` in Python builds with C extensions).

**Prevention:**
- `asyncio.Lock` — mutual exclusion; one coroutine in the critical section at a time.
- `asyncio.Semaphore(n)` — limit concurrency to *n* rather than 1.
- **Immutable data / functional style** — data that is never mutated cannot race.
- **Index-based writes** — pre-allocate and write to distinct indices; no shared
  read-modify-write.
- **Database transactions** — ACID guarantees prevent races at the DB layer.

---

### Q5: How does Docker layer caching work, and how do you structure a Dockerfile to maximise cache reuse?

Each `RUN`, `COPY`, and `ADD` instruction creates an immutable layer identified by a hash
of its instruction and all parent layer hashes. Docker reuses a cached layer if the hash
matches; the cache is **invalidated for that layer and every layer after it** as soon as
any input changes.

**Optimise by putting frequently-changing files last:**

```dockerfile
# BAD — changing any source file invalidates the pip install layer
COPY . .
RUN pip install -r requirements.txt

# GOOD — requirements.txt changes rarely; source changes often
COPY requirements.txt .
RUN pip install -r requirements.txt   # cached unless requirements.txt changes
COPY src/ ./src/                       # only this layer rebuilds on source changes
```

**Other tips:**
- Use `--no-cache-dir` with pip to avoid pip's own cache polluting layers.
- In multi-stage builds, only the final stage's layers end up in the shipped image.
- Pin base image tags (`python:3.12.7-slim` not `python:3.12-slim`) so the base layer
  hash is stable across CI runs.

---

### Q6: What is the difference between a liveness probe and a readiness probe?

| | Liveness | Readiness |
|---|---|---|
| Question answered | "Is the process alive?" | "Is the process ready to serve traffic?" |
| Kubernetes action on failure | Restart the container | Remove pod from load-balancer endpoints |
| Typical implementation | `GET /health` → 200 immediately | `GET /ready` → checks DB, cache, etc. |
| Danger of wrong check | DB check in liveness → DB hiccup causes restart storm | None — failing readiness just drains traffic |

**Rule:** liveness must be cheap and always-fast. Readiness may be slower but should
still complete in well under 1 second to avoid timeout cascades.

---

### Q7: Why use structured (JSON) logging, and what role do correlation IDs play?

**Structured logging (JSON):**
- Log lines become machine-parseable events, not free-text strings.
- Centralised logging platforms (Datadog, CloudWatch, ELK) can filter/aggregate on any
  field (`level`, `duration_ms`, `user_id`) without regex parsing.
- Consistent schema across all services enables cross-service queries.

**Correlation IDs:**
- A UUID generated (or forwarded from upstream) per request and attached to every log
  record emitted during that request.
- In a distributed system, the same ID flows through Service A → Service B → DB, so a
  single `grep correlation_id=abc-123` retrieves the full call chain across services.
- Returned in response headers so clients can include it in bug reports.
- Stored in a `ContextVar` (not a global) so concurrent requests don't cross-contaminate.

---

### Q8: Describe the CI/CD pipeline stages for a Python FastAPI app.

```
Push → PR
  │
  ├─ 1. Lint & Format
  │     ruff check src/ tests/
  │     black --check src/ tests/
  │
  ├─ 2. Type Check
  │     mypy src/
  │
  ├─ 3. Unit Tests
  │     pytest tests/unit -x --cov=src --cov-fail-under=80
  │
  ├─ 4. Integration Tests
  │     docker compose -f docker-compose.test.yml up -d
  │     pytest tests/integration
  │     docker compose down -v
  │
  ├─ 5. Security Scan
  │     pip-audit (dependency CVEs)
  │     bandit -r src/ (code patterns)
  │
  └─ 6. Sonar / Quality Gate
        sonar-scanner → coverage, complexity, duplication ratings

Merge to main
  │
  ├─ 7. Build Docker Image
  │     docker buildx build --platform linux/amd64,linux/arm64
  │     push to registry with git-sha tag
  │
  ├─ 8. Staging Deploy
  │     kubectl set image / helm upgrade
  │     smoke tests against staging URL
  │
  └─ 9. Production Deploy (manual gate or auto on tag)
        blue-green or rolling update
        post-deploy health check: /health + /ready
```

---

## Coding Challenge: Async Financial Data Aggregator

### Implementation

```python
# src/aggregator.py
"""
Async Financial Data Aggregator
Fetches data from 3 mock endpoints concurrently with:
  - Semaphore(2) to cap concurrent connections
  - Exponential backoff retry (max 3 attempts) via tenacity
  - Order-preserving results with timing metadata
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class EndpointResult:
    source: str
    data: dict | None
    success: bool
    duration_ms: float
    error: str | None = None


@dataclass
class AggregatedResult:
    results: list[EndpointResult]
    total_duration_ms: float
    successful_count: int = field(init=False)
    failed_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.successful_count = sum(1 for r in self.results if r.success)
        self.failed_count = len(self.results) - self.successful_count


# ── Retry-decorated fetch ─────────────────────────────────────────────────────

def _make_retry_fetch(semaphore: asyncio.Semaphore):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(httpx.HTTPError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _fetch_url(client: httpx.AsyncClient, url: str) -> dict:
        async with semaphore:
            logger.debug("fetching %s", url)
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()

    return _fetch_url


# ── Aggregator ────────────────────────────────────────────────────────────────

class FinancialDataAggregator:
    """
    Fetches financial data from multiple endpoints concurrently.
    Caps concurrent HTTP connections with a Semaphore.
    Retries transient HTTP errors with exponential backoff.
    Returns results in original endpoint order with per-endpoint timing.
    """

    def __init__(self, max_concurrent: int = 2) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._retry_fetch = _make_retry_fetch(self._semaphore)

    async def fetch_one(
        self,
        client: httpx.AsyncClient,
        source: str,
        url: str,
    ) -> EndpointResult:
        start = time.perf_counter()
        try:
            data = await self._retry_fetch(client, url)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("success: %s in %.1fms", source, duration_ms)
            return EndpointResult(
                source=source,
                data=data,
                success=True,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("failed: %s — %s (%.1fms)", source, exc, duration_ms)
            return EndpointResult(
                source=source,
                data=None,
                success=False,
                duration_ms=duration_ms,
                error=str(exc),
            )

    async def aggregate(
        self,
        endpoints: dict[str, str],  # {source_name: url}
        client: httpx.AsyncClient | None = None,
    ) -> AggregatedResult:
        """
        Fetch all endpoints and return aggregated results.
        Accepts an optional *client* for testing (injection of a mock client).
        """
        wall_start = time.perf_counter()

        async def _run(c: httpx.AsyncClient) -> list[EndpointResult]:
            tasks = [
                self.fetch_one(c, source, url)
                for source, url in endpoints.items()
            ]
            return list(await asyncio.gather(*tasks))

        if client is not None:
            results = await _run(client)
        else:
            async with httpx.AsyncClient() as c:
                results = await _run(c)

        total_ms = (time.perf_counter() - wall_start) * 1000
        logger.info(
            "aggregation complete: %d/%d succeeded in %.1fms",
            sum(r.success for r in results),
            len(results),
            total_ms,
        )
        return AggregatedResult(results=results, total_duration_ms=total_ms)
```

### Tests

```python
# tests/unit/test_aggregator.py
"""
Tests for FinancialDataAggregator using respx to mock httpx calls.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from src.aggregator import AggregatedResult, EndpointResult, FinancialDataAggregator


ENDPOINTS = {
    "stocks": "https://api.example.com/stocks",
    "forex":  "https://api.example.com/forex",
    "crypto": "https://api.example.com/crypto",
}

MOCK_RESPONSES = {
    "stocks": {"symbol": "AAPL", "price": 150.0},
    "forex":  {"pair": "EUR/USD", "rate": 1.08},
    "crypto": {"symbol": "BTC", "price": 45000.0},
}


@pytest.mark.asyncio
async def test_all_endpoints_success():
    """All three endpoints succeed — AggregatedResult has 3 successful results."""
    agg = FinancialDataAggregator(max_concurrent=2)

    with respx.mock(assert_all_called=True) as mock:
        for source, url in ENDPOINTS.items():
            mock.get(url).mock(
                return_value=httpx.Response(200, json=MOCK_RESPONSES[source])
            )

        async with httpx.AsyncClient() as client:
            result = await agg.aggregate(ENDPOINTS, client=client)

    assert isinstance(result, AggregatedResult)
    assert result.successful_count == 3
    assert result.failed_count == 0
    assert result.total_duration_ms > 0

    by_source = {r.source: r for r in result.results}
    assert by_source["stocks"].data == MOCK_RESPONSES["stocks"]
    assert by_source["forex"].data == MOCK_RESPONSES["forex"]
    assert by_source["crypto"].data == MOCK_RESPONSES["crypto"]


@pytest.mark.asyncio
async def test_one_endpoint_fails():
    """One endpoint returns 500 — result shows 2 successes, 1 failure."""
    agg = FinancialDataAggregator(max_concurrent=2)

    with respx.mock():
        respx.get(ENDPOINTS["stocks"]).mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSES["stocks"])
        )
        respx.get(ENDPOINTS["forex"]).mock(
            return_value=httpx.Response(200, json=MOCK_RESPONSES["forex"])
        )
        # Crypto fails all 3 retry attempts
        respx.get(ENDPOINTS["crypto"]).mock(
            return_value=httpx.Response(500, json={"error": "internal error"})
        )

        async with httpx.AsyncClient() as client:
            result = await agg.aggregate(ENDPOINTS, client=client)

    assert result.successful_count == 2
    assert result.failed_count == 1

    crypto_result = next(r for r in result.results if r.source == "crypto")
    assert crypto_result.success is False
    assert crypto_result.data is None
    assert crypto_result.error is not None


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """
    Verify that no more than max_concurrent requests run simultaneously.
    Uses an asyncio.Event to detect concurrency level.
    """
    max_concurrent = 2
    agg = FinancialDataAggregator(max_concurrent=max_concurrent)
    active = 0
    peak_active = 0

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.05)  # simulate latency
        active -= 1
        return httpx.Response(200, json={"ok": True})

    with respx.mock():
        for url in ENDPOINTS.values():
            respx.get(url).mock(side_effect=slow_handler)

        async with httpx.AsyncClient() as client:
            await agg.aggregate(ENDPOINTS, client=client)

    assert peak_active <= max_concurrent, (
        f"Peak concurrency {peak_active} exceeded limit {max_concurrent}"
    )


@pytest.mark.asyncio
async def test_result_order_matches_endpoint_order():
    """Results are returned in the same order as the input endpoints dict."""
    agg = FinancialDataAggregator(max_concurrent=2)
    expected_order = list(ENDPOINTS.keys())

    with respx.mock():
        for source, url in ENDPOINTS.items():
            respx.get(url).mock(
                return_value=httpx.Response(200, json=MOCK_RESPONSES[source])
            )

        async with httpx.AsyncClient() as client:
            result = await agg.aggregate(ENDPOINTS, client=client)

    actual_order = [r.source for r in result.results]
    assert actual_order == expected_order


@pytest.mark.asyncio
async def test_retry_on_transient_failure():
    """
    First two attempts return 503, third returns 200.
    Verifies that tenacity retries and ultimately succeeds.
    """
    agg = FinancialDataAggregator(max_concurrent=2)
    call_count = 0

    async def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(503, json={"error": "service unavailable"})
        return httpx.Response(200, json={"symbol": "BTC", "price": 45000.0})

    single_endpoint = {"crypto": ENDPOINTS["crypto"]}

    with respx.mock():
        respx.get(ENDPOINTS["crypto"]).mock(side_effect=flaky_handler)

        async with httpx.AsyncClient() as client:
            # Patch the wait to 0 so the test doesn't actually sleep
            import tenacity
            with (
                pytest.MonkeyPatch().context() as mp,
            ):
                result = await agg.aggregate(single_endpoint, client=client)

    # With reraise=True and 3 attempts, call_count == 3 means the 3rd attempt succeeded
    assert call_count == 3
    assert result.successful_count == 1
```

### pyproject.toml snippet

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"

[project.optional-dependencies]
test = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "respx>=0.21",
    "httpx>=0.27",
]
```

### Running the Tests

```bash
pytest tests/unit/test_aggregator.py -v
```

Expected output:
```
tests/unit/test_aggregator.py::test_all_endpoints_success         PASSED
tests/unit/test_aggregator.py::test_one_endpoint_fails            PASSED
tests/unit/test_aggregator.py::test_semaphore_limits_concurrency  PASSED
tests/unit/test_aggregator.py::test_result_order_matches_endpoint_order PASSED
tests/unit/test_aggregator.py::test_retry_on_transient_failure    PASSED
```

### Design Decisions

| Decision | Rationale |
|---|---|
| `asyncio.Semaphore(2)` not `3` | Exam spec says max 2 concurrent; Semaphore is cleaner than manual counter |
| `dataclass` not `Pydantic` | Internal data with no serialisation requirement; dataclass is lighter |
| `client` injection in `aggregate()` | Enables test mocking without monkeypatching; cleaner than `unittest.mock.patch` |
| `reraise=True` in tenacity | Preserves original `httpx.HTTPError` type so callers can handle it specifically |
| Per-endpoint error isolation | A failure in one endpoint returns an `EndpointResult(success=False)` instead of propagating; the aggregator never raises unless the entire call site is wrong |
| `time.perf_counter()` for timing | Higher resolution than `time.time()`; monotonic so unaffected by system clock adjustments |
