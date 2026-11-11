# Week 08 · Day 2 — Solutions
## Coroutines & Tasks

---

### Exercise 1
**Write `run_with_timeout(coro, seconds)` that wraps any coroutine with `asyncio.wait_for` and handles `asyncio.TimeoutError` gracefully, returning `None` on timeout.**

```python
import asyncio
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def run_with_timeout(coro: Coroutine[Any, Any, T], seconds: float) -> T | None:
    """Run a coroutine with a deadline; return None if it exceeds the timeout.

    Args:
        coro:    Any awaitable coroutine.
        seconds: Maximum wall-clock seconds to wait before cancelling.

    Returns:
        The coroutine's return value, or None if it timed out.
    """
    try:
        result: T = await asyncio.wait_for(coro, timeout=seconds)
        return result
    except asyncio.TimeoutError:
        logger.warning("Coroutine timed out after %.2f seconds", seconds)
        return None


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

async def slow_task(delay: float) -> str:
    await asyncio.sleep(delay)
    return f"Finished after {delay}s"


async def demo() -> None:
    # Should succeed — completes in 0.5s, limit is 2s
    result = await run_with_timeout(slow_task(0.5), seconds=2.0)
    logger.info("Fast task result: %s", result)

    # Should return None — would take 5s but limit is 1s
    result = await run_with_timeout(slow_task(5.0), seconds=1.0)
    logger.info("Slow task result: %s", result)  # logs: None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(demo())
```

**Key points:**
- `asyncio.wait_for` raises `asyncio.TimeoutError` (a subclass of `Exception` in Python 3.11+) when the deadline expires. It also cancels the wrapped coroutine automatically.
- Catching `TimeoutError` rather than `Exception` makes the error handling explicit — we only swallow timeouts, not programming bugs.
- The `TypeVar T` preserves the coroutine's return type so callers get `T | None` instead of `Any | None`, keeping type checkers happy.
- In Python 3.11+, `asyncio.TimeoutError` is an alias for the built-in `TimeoutError`. Using `asyncio.TimeoutError` is clearer about the source.

---

### Exercise 2
**Create a `TaskGroup` usage example: run 3 async tasks, collect results, handle one task failing without cancelling others using `asyncio.gather(return_exceptions=True)`.**

```python
import asyncio
import logging

logger = logging.getLogger(__name__)


async def task_ok(name: str, delay: float) -> str:
    """Simulate a successful async task."""
    await asyncio.sleep(delay)
    logger.debug("Task %s completed", name)
    return f"result_from_{name}"


async def task_fail(name: str) -> str:
    """Simulate a task that raises an exception."""
    await asyncio.sleep(0.05)
    raise ValueError(f"Task {name} deliberately failed")


# ---------------------------------------------------------------------------
# Approach A: asyncio.gather with return_exceptions=True
# ---------------------------------------------------------------------------
# This is the recommended pattern when you want ALL tasks to run to
# completion (or failure) and then inspect each outcome individually.

async def run_with_gather() -> None:
    logger.info("--- asyncio.gather approach ---")

    results = await asyncio.gather(
        task_ok("alpha", 0.1),
        task_fail("beta"),           # this one raises
        task_ok("gamma", 0.2),
        return_exceptions=True,      # exceptions become return values, no propagation
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Task %d failed: %s", i, result)
        else:
            logger.info("Task %d succeeded: %s", i, result)


# ---------------------------------------------------------------------------
# Approach B: asyncio.TaskGroup (Python 3.11+)
# ---------------------------------------------------------------------------
# TaskGroup cancels ALL remaining tasks on the FIRST exception by default.
# Use it when partial completion is a bug, not a valid state.
# To avoid cancellation on failure, catch exceptions INSIDE each task.

async def safe_wrapper(coro, name: str) -> dict:
    """Wrap a coroutine so exceptions are captured, not propagated."""
    try:
        value = await coro
        return {"name": name, "ok": True, "value": value}
    except Exception as exc:
        return {"name": name, "ok": False, "error": str(exc)}


async def run_with_task_group() -> None:
    logger.info("--- asyncio.TaskGroup approach ---")

    task_results: list[dict] = []

    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(safe_wrapper(task_ok("alpha", 0.1), "alpha"))
        t2 = tg.create_task(safe_wrapper(task_fail("beta"), "beta"))
        t3 = tg.create_task(safe_wrapper(task_ok("gamma", 0.2), "gamma"))

    # TaskGroup waits for all tasks; results are available after the block
    for task in (t1, t2, t3):
        result = task.result()
        task_results.append(result)
        if result["ok"]:
            logger.info("Task %s: %s", result["name"], result["value"])
        else:
            logger.error("Task %s failed: %s", result["name"], result["error"])


async def main() -> None:
    await run_with_gather()
    await run_with_task_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
```

**Key points:**
- `asyncio.gather(return_exceptions=True)` is the simplest way to run tasks and collect both successes and failures — exceptions are returned as values in the results list instead of being re-raised.
- `asyncio.TaskGroup` (Python 3.11+) is more structured but cancels all tasks if one raises. Wrapping each task in a `safe_wrapper` gives per-task error isolation while keeping the `TaskGroup` API.
- Without `return_exceptions=True`, the first exception from `gather` propagates immediately and the other coroutines' results are lost (though they continue running until awaited or collected).
- `task.result()` after a `TaskGroup` block raises the task's exception if it failed without wrapping — always check or handle this.

---

### Exercise 3
**Implement `rate_limited_tasks(urls, max_concurrent=3)` that creates tasks for each URL but limits to `max_concurrent` running at once using `asyncio.Semaphore` and `asyncio.gather`.**

```python
import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)


async def _fetch_one(
    url: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Acquire semaphore slot, fetch URL, release slot.

    The semaphore limits how many requests run simultaneously without
    preventing others from being scheduled.
    """
    async with semaphore:          # blocks if max_concurrent slots are taken
        logger.debug("Fetching %s (slot acquired)", url)
        start = time.perf_counter()
        try:
            response = await client.get(url, timeout=httpx.Timeout(10.0))
            elapsed_ms = (time.perf_counter() - start) * 1000
            return {
                "url": url,
                "status": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            }
        except httpx.RequestError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Request failed for %s: %s", url, exc)
            return {"url": url, "status": -1, "elapsed_ms": round(elapsed_ms, 2)}


async def rate_limited_tasks(
    urls: list[str],
    max_concurrent: int = 3,
) -> list[dict]:
    """Fetch all URLs concurrently, but at most `max_concurrent` at a time.

    Args:
        urls:           List of URLs to fetch.
        max_concurrent: Maximum number of simultaneous in-flight requests.

    Returns:
        List of result dicts in the same order as `urls`.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:
        tasks = [
            asyncio.create_task(_fetch_one(url, client, semaphore))
            for url in urls
        ]
        results: list[dict] = await asyncio.gather(*tasks, return_exceptions=False)
    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    urls = [
        f"https://httpbin.org/delay/{i % 2}"
        for i in range(9)
    ]
    logger.info("Fetching %d URLs with max_concurrent=3", len(urls))
    wall_start = time.perf_counter()

    results = await rate_limited_tasks(urls, max_concurrent=3)

    wall_elapsed = (time.perf_counter() - wall_start) * 1000
    logger.info("Done in %.1f ms wall time", wall_elapsed)
    for r in results:
        logger.info("  %-40s  %s  %.1f ms", r["url"], r["status"], r["elapsed_ms"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
```

**Key points:**
- `asyncio.Semaphore(n)` is a counter that allows at most `n` coroutines to hold it simultaneously. `async with semaphore` decrements on entry and increments on exit, blocking if the count is zero.
- `asyncio.create_task()` schedules all coroutines immediately — they will start as soon as the event loop is free. The semaphore controls how many proceed past the `async with` block at once.
- Using `asyncio.gather(*tasks)` collects results in input order, which is important when callers need to correlate results with the original URL list.
- This pattern is preferred over batching URLs into chunks of `max_concurrent` because a semaphore allows the next URL to start as soon as any slot frees, rather than waiting for an entire batch to finish.
