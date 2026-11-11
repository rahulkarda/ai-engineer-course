# Phase 1 · Week 08 · Day 2 — Coroutines & Tasks

> Objective: Master coroutines as state machines, understand the difference between tasks and bare coroutines, and use `asyncio.gather` for concurrent I/O.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

### Coroutines

A coroutine is a function defined with `async def`. Calling it does **not** execute it — it returns a coroutine object:

```python
async def fetch(url: str) -> str:
    ...

coro = fetch("https://example.com")   # nothing runs yet
result = await coro                    # execution starts here
```

Coroutines are **generators under the hood** (they use `yield` at the bytecode level). Every `await` is a suspension point — the coroutine freezes its local state and yields a value up the chain until the event loop receives it.

### Tasks vs Awaiting Directly

| Pattern | What happens |
|---|---|
| `result = await coro` | Runs coro *inline* — current coroutine blocks until coro finishes |
| `task = asyncio.create_task(coro)` | Schedules coro on the event loop immediately; current coroutine continues |
| `results = await asyncio.gather(*coros)` | Schedules all, waits for all, returns results in order |

`asyncio.create_task()` is the key to concurrency. Without it, sequential `await` calls are just serial I/O.

### Futures

A `Future` is a lower-level primitive — a placeholder for a value that will be available later. `Task` is a subclass of `Future` that wraps a coroutine. You rarely create `Future` objects directly; the event loop and asyncio internals use them. `await future` suspends until the future has a result set.

---

## How It Works Internally

### Coroutine as a State Machine

The Python compiler transforms `async def` functions into a state machine. Each `await` becomes a state boundary. CPython stores:

- **`cr_frame`**: the suspended frame (locals, line number, bytecode offset)
- **`cr_await`**: the object currently being awaited
- **`cr_running`**: bool, prevents re-entering a running coroutine

```
Coroutine states:
  CREATED   → (first send/await) →  RUNNING
  RUNNING   → (await suspends)   →  SUSPENDED
  SUSPENDED → (event resolves)   →  RUNNING
  RUNNING   → (return/raise)     →  CLOSED
```

Inspection: `import inspect; inspect.getcoroutinestate(coro)` returns `'CORO_CREATED'`, `'CORO_SUSPENDED'`, `'CORO_RUNNING'`, `'CORO_CLOSED'`.

### Task Scheduling

When you call `asyncio.create_task(coro)`:

1. A `Task` object is created wrapping `coro`.
2. `loop.call_soon(task.__step)` is called — this queues `task.__step` on the ready queue.
3. On the next event loop iteration, `task.__step` calls `coro.send(None)`, running the coroutine until its first `await`.
4. If the `await` is on I/O, the task registers its fd with the selector and suspends. If it's `asyncio.sleep(0)`, it yields immediately and reschedules itself.

### Task Cancellation

`task.cancel()` injects a `CancelledError` into the coroutine at its next `await` point. The coroutine can catch it for cleanup via `try/finally` or `except asyncio.CancelledError`. If not caught, the task is marked cancelled.

---

## Worked Example

```python
# week08/day2_tasks.py
"""
Demonstrates: sequential await vs concurrent gather, task creation,
cancellation, and timeout handling.
Requires: pip install httpx
Run: python -m week08.day2_tasks
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data model for results
# --------------------------------------------------------------------------- #

@dataclass
class FetchResult:
    url: str
    status: int
    elapsed_s: float
    error: str | None = field(default=None)


# --------------------------------------------------------------------------- #
# Core fetch helper
# --------------------------------------------------------------------------- #

async def fetch_one(client: httpx.AsyncClient, url: str) -> FetchResult:
    """Fetch a single URL; return FetchResult (never raises)."""
    start = time.perf_counter()
    try:
        response = await client.get(url, timeout=10.0)
        response.raise_for_status()
        elapsed = time.perf_counter() - start
        logger.info("  fetched %s → %d (%.3fs)", url, response.status_code, elapsed)
        return FetchResult(url=url, status=response.status_code, elapsed_s=elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.warning("  error fetching %s: %s (%.3fs)", url, exc, elapsed)
        return FetchResult(url=url, status=0, elapsed_s=elapsed, error=str(exc))


# --------------------------------------------------------------------------- #
# Pattern 1: Sequential await (no concurrency)
# --------------------------------------------------------------------------- #

async def fetch_sequential(urls: list[str]) -> list[FetchResult]:
    """Fetches one URL at a time. Total time ≈ sum of individual times."""
    logger.info("=== Sequential fetch (%d URLs) ===", len(urls))
    results: list[FetchResult] = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            result = await fetch_one(client, url)
            results.append(result)
    return results


# --------------------------------------------------------------------------- #
# Pattern 2: asyncio.gather (true concurrency)
# --------------------------------------------------------------------------- #

async def fetch_concurrent(urls: list[str]) -> list[FetchResult]:
    """
    Fetches all URLs concurrently using asyncio.gather.
    Total time ≈ max of individual times (not sum).
    gather() preserves input order in results regardless of completion order.
    """
    logger.info("=== Concurrent fetch (%d URLs) ===", len(urls))
    async with httpx.AsyncClient() as client:
        # gather schedules all coroutines as tasks immediately
        results: list[FetchResult] = await asyncio.gather(
            *[fetch_one(client, url) for url in urls],
            return_exceptions=False,   # propagate exceptions; set True to suppress
        )
    return results


# --------------------------------------------------------------------------- #
# Pattern 3: create_task with explicit cancellation + timeout
# --------------------------------------------------------------------------- #

async def fetch_with_timeout(url: str, timeout_s: float) -> FetchResult | None:
    """
    Wraps a fetch in asyncio.wait_for to enforce a wall-clock deadline.
    asyncio.wait_for cancels the inner task if timeout expires.
    """
    logger.info("=== Fetch with %.1fs timeout: %s ===", timeout_s, url)
    try:
        async with httpx.AsyncClient() as client:
            result = await asyncio.wait_for(
                fetch_one(client, url),
                timeout=timeout_s,
            )
        return result
    except asyncio.TimeoutError:
        logger.warning("  timed out after %.1fs for %s", timeout_s, url)
        return None


# --------------------------------------------------------------------------- #
# Pattern 4: Manual task control
# --------------------------------------------------------------------------- #

async def demonstrate_task_lifecycle() -> None:
    """
    Shows how to create, inspect, and cancel a Task explicitly.
    Using asyncio.sleep as a stand-in for a long I/O operation.
    """
    logger.info("=== Task lifecycle demo ===")

    async def slow_job(name: str, duration: float) -> str:
        logger.info("  %s: starting (%.1fs)", name, duration)
        try:
            await asyncio.sleep(duration)
            logger.info("  %s: completed", name)
            return f"{name} done"
        except asyncio.CancelledError:
            logger.warning("  %s: cancelled — cleaning up", name)
            # perform cleanup here (close connections, flush buffers, etc.)
            raise   # re-raise so the task is marked cancelled

    # Schedule two tasks
    task_a: asyncio.Task[str] = asyncio.create_task(slow_job("A", 2.0), name="job-A")
    task_b: asyncio.Task[str] = asyncio.create_task(slow_job("B", 0.5), name="job-B")

    # Let task_b finish naturally, cancel task_a early
    await asyncio.sleep(1.0)            # wait 1s — task_b should be done
    logger.info("  task_b done=%s result=%s", task_b.done(), task_b.result() if task_b.done() else "pending")

    task_a.cancel()                      # injects CancelledError at next await
    try:
        await task_a
    except asyncio.CancelledError:
        logger.info("  task_a was cancelled (confirmed)")


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #

URLS: list[str] = [
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/delay/1",
]

async def main() -> None:
    # Sequential: expect ~3s total
    t0 = time.perf_counter()
    seq_results = await fetch_sequential(URLS)
    seq_elapsed = time.perf_counter() - t0
    logger.info("Sequential total: %.3fs", seq_elapsed)

    # Concurrent: expect ~1s total
    t0 = time.perf_counter()
    con_results = await fetch_concurrent(URLS)
    con_elapsed = time.perf_counter() - t0
    logger.info("Concurrent total: %.3fs  (speedup: %.1fx)", con_elapsed, seq_elapsed / con_elapsed)

    # Timeout demo (tight deadline — will likely expire)
    await fetch_with_timeout("https://httpbin.org/delay/3", timeout_s=1.0)

    # Task lifecycle
    await demonstrate_task_lifecycle()


if __name__ == "__main__":
    asyncio.run(main())
```

**Expected output:** Sequential total ~3 s, concurrent total ~1 s, ~3× speedup logged. The tight timeout will log a warning. Task A will log cancellation cleanup.

---

## Exercises

1. **gather vs as_completed.** Rewrite `fetch_concurrent` to use `asyncio.as_completed()` instead of `gather`. Log each result as it arrives (not waiting for all). When would you prefer `as_completed` over `gather`? Write a docstring explaining the trade-off.

2. **Task exception handling.** Modify `demonstrate_task_lifecycle` to launch five tasks where task index 3 raises `ValueError("simulated failure")`. Use `asyncio.gather(*tasks, return_exceptions=True)`. Iterate the results and log which tasks succeeded and which failed. Do not let any exception propagate uncaught.

3. **Coroutine state inspection.** Write a coroutine `async def lazy(n: int) -> int` that loops `n` times, `await asyncio.sleep(0)`-ing each iteration. Before running it, inspect `inspect.getcoroutinestate(coro)` — it should be `CORO_CREATED`. Wrap it in `asyncio.create_task()`, then `await asyncio.sleep(0)` to let the event loop start it, then inspect `inspect.getcoroutinestate` again. Finally `await` it to completion and inspect once more. Log each state transition.

*(Solutions: ../solutions/week-08/day-2.md)*

---

## Self-check

- [ ] I can explain the difference between `await coro`, `asyncio.create_task(coro)`, and `asyncio.gather(*coros)` — when to use each.
- [ ] I can describe the four coroutine states (`CREATED`, `SUSPENDED`, `RUNNING`, `CLOSED`) and what transition triggers each.
- [ ] I can implement task cancellation with cleanup using `try/except asyncio.CancelledError` and explain why re-raising is necessary.

---

## Supplementary

- asyncio tasks and coroutines docs: <https://docs.python.org/3/library/asyncio-task.html>
- PEP 492 — Coroutines with async/await: <https://peps.python.org/pep-0492/>
- asyncio internals deep-dive: <https://tenthousandmeters.com/blog/python-behind-the-scenes-12-how-asyncio-works/>
