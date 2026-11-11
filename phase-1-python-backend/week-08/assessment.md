# Week 08 Assessment

## Conceptual (interview-style)

1. **Event loop mechanics.** Describe the three phases of one iteration of CPython's asyncio event loop. What does `select`/`epoll`/`kqueue` do, and why does the event loop delegate I/O polling to the OS rather than spinning in Python?

2. **Async vs threading vs multiprocessing.** A teammate proposes rewriting a CPU-bound image-compression worker as `async def` to "speed it up." Explain why this will not help and what approach you would recommend instead. Then give a concrete example of workload that *does* benefit from `asyncio`.

3. **The GIL.** What is the Global Interpreter Lock, when does CPython release it, and what are the practical consequences for (a) I/O-bound concurrent code and (b) CPU-bound parallel code? Name one Python runtime that removes the GIL and describe the trade-off.

4. **Race conditions in async code.** An async counter is incremented with `count += 1` inside a coroutine. Two coroutines run concurrently on the same event loop. Can a race condition occur? Explain why or why not, and describe a scenario where a race *can* occur in single-threaded async Python even without threads.

5. **Docker layer caching.** You have a `Dockerfile` that copies the entire project directory before running `pip install`. CI builds take 4 minutes because pip re-downloads all packages on every push. Explain the root cause and rewrite the relevant `Dockerfile` instructions to restore caching for the dependency install step.

6. **Health check design.** Distinguish `/health` (liveness) from `/ready` (readiness) in the context of a Kubernetes deployment. What should each endpoint check? What happens if readiness returns 503 — what does the kubelet do? Give a realistic implementation that checks database connectivity.

7. **Structured logging.** A production log line reads `ERROR: failed to process request`. Explain why this is insufficient for a distributed system. Describe what fields a well-structured JSON log entry should include, and explain how correlation IDs help trace a request across multiple microservices.

8. **CI/CD pipeline stages.** Sketch a minimal GitHub Actions CI pipeline for a FastAPI service. List the stages in order, explain the purpose of each, and identify which stage should gate deployment to production. Why is it important to fail fast and run the cheapest checks first?

---

## Coding challenge

### Async Financial Data Aggregator

Build an async service (`aggregator.py`) that fetches price data from three mock HTTP endpoints concurrently, enforces a concurrency limit, retries on failure, and returns aggregated results with timing metadata.

#### Specification

```python
# src/aggregator.py
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


ENDPOINTS: list[str] = [
    os.getenv("PRICE_ENDPOINT_A", ""),
    os.getenv("PRICE_ENDPOINT_B", ""),
    os.getenv("PRICE_ENDPOINT_C", ""),
]

MAX_CONCURRENT = 2   # Semaphore limit
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.5


@dataclass
class PriceResult:
    endpoint: str
    data: dict[str, Any] | None
    error: str | None
    attempts: int
    elapsed_ms: float


@dataclass
class AggregatedPrices:
    results: list[PriceResult]
    total_elapsed_ms: float
    success_count: int
    failure_count: int


async def fetch_with_retry(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    endpoint: str,
) -> PriceResult:
    """Fetch endpoint with Semaphore-bounded concurrency and exponential backoff."""
    ...


async def aggregate_prices() -> AggregatedPrices:
    """Fetch all endpoints concurrently via asyncio.gather and return aggregated results."""
    ...
```

#### Acceptance criteria

- `fetch_with_retry` uses `asyncio.Semaphore(MAX_CONCURRENT)` so at most 2 HTTP calls run simultaneously.
- Retries up to `MAX_RETRIES` times on `httpx.HTTPError` or non-2xx status; delay between attempts is `BASE_BACKOFF_SECONDS * 2^(attempt - 1)` (exponential backoff).
- After exhausting retries, `PriceResult.error` is set to the exception message and `data` is `None`; the aggregator does **not** raise — it records the failure and continues.
- `aggregate_prices` uses `asyncio.gather(*tasks)` to run all endpoint fetches concurrently.
- `AggregatedPrices.total_elapsed_ms` is the wall-clock time for the entire gather call.
- All endpoints are read from environment variables — no hardcoded URLs.
- All functions are typed. `logging` used, not `print`.
- Tests use `pytest-asyncio` and mock `httpx.AsyncClient` to simulate success, transient failure (retried), and permanent failure (exhausted retries).
- Test file is at `tests/unit/test_aggregator.py`. Tests cover: all-success, one endpoint fails all retries, retry count verification, semaphore limits concurrent calls to ≤ 2.

#### Deliverables

- `src/aggregator.py` — implementation
- `tests/unit/test_aggregator.py` — test suite
- `.env.example` updated with `PRICE_ENDPOINT_A`, `PRICE_ENDPOINT_B`, `PRICE_ENDPOINT_C`

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-08/assessment.md
