# Phase 1 · Week 08 · Day 1 — Async & the Event Loop

> Objective: Understand why async I/O exists, how CPython's event loop works, and write your first async FastAPI endpoint.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-5]]

---

## Concept

### Sync vs Async I/O

Synchronous code blocks: when you call `requests.get(url)`, the Python thread sits idle waiting for the network response. The OS has paused the thread, and no other Python code in that thread runs until the response arrives.

Async I/O inverts the model. Instead of blocking, you register interest in an event ("notify me when the socket is readable") and yield control back to a scheduler. The scheduler runs other work while waiting, then resumes you when the data arrives.

**Why it matters for I/O-bound work:** A web server handling 1 000 simultaneous requests each taking 100 ms of network I/O would need 1 000 threads under a synchronous model (expensive — each thread costs ~8 MB stack + OS context switch overhead). Under async, a single thread handles all 1 000 concurrently, interleaving work at each `await` point.

**When async does NOT help:** CPU-bound work (e.g., numpy, ML inference). Async keeps one thread occupied; the GIL still prevents true parallel CPU execution. Use `multiprocessing` or `concurrent.futures.ProcessPoolExecutor` for CPU parallelism.

---

## How It Works Internally

### CPython's Event Loop

The event loop is a loop — literally a `while True:` — that:

1. **Polls I/O** via `select` (Unix portable) or `epoll` (Linux) or `kqueue` (macOS). These are OS syscalls that return when any registered file descriptor becomes ready.
2. **Drains the ready queue**: for each descriptor that's ready, find the coroutine waiting on it and schedule it to resume.
3. **Runs scheduled callbacks**: microtasks (e.g., `asyncio.ensure_future`) run before the next I/O poll.

```
┌─────────────────────────────────────┐
│           Event Loop Tick           │
│                                     │
│  1. Run all ready callbacks         │
│  2. epoll(fd_list, timeout=0.001)   │
│     → returns list of ready fds     │
│  3. Resume coroutines waiting on    │
│     those fds                       │
│  4. Repeat                          │
└─────────────────────────────────────┘
```

### Call Stack vs Event Queue

- **Call stack**: the normal Python execution stack. Synchronous calls push/pop frames here.
- **Event queue / ready queue**: coroutines that are ready to run. `asyncio.create_task()` adds to this queue.
- **Suspension**: `await` suspends the current coroutine and hands control back to the event loop. The coroutine is resumed later when its awaitable completes.

### asyncio in CPython

`asyncio` ships with CPython. The default event loop on Unix uses `SelectorEventLoop` (wraps `selectors.DefaultSelector`, which picks `epoll`/`kqueue` per OS). `uvloop` is a drop-in replacement built on `libuv` — 2–4× faster because libuv avoids extra Python layers.

FastAPI uses `uvicorn` (ASGI server) which runs the event loop for you. You only need `async def` route handlers.

---

## Worked Example

```python
# week08/day1_async_fastapi.py
"""
Compare sync vs async FastAPI endpoints.
Requires: pip install fastapi uvicorn httpx
Run: uvicorn week08.day1_async_fastapi:app --reload
"""

import asyncio
import logging
import time
from typing import Annotated

import httpx
from fastapi import FastAPI, Query

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Week 08 Day 1 — Async Demo")

MOCK_LATENCY_SECONDS: float = 0.5  # simulates network I/O


# --- Sync endpoint (blocks the thread) ---

@app.get("/sync")
def sync_endpoint(n: Annotated[int, Query(ge=1, le=10)] = 3) -> dict[str, float]:
    """
    Calls a slow operation `n` times sequentially.
    Under uvicorn's threadpool each sync handler runs in a thread,
    but threads are limited — concurrency ceiling is hit fast.
    """
    start = time.perf_counter()
    results: list[float] = []

    for i in range(n):
        logger.info("sync call %d/%d — blocking %.2fs", i + 1, n, MOCK_LATENCY_SECONDS)
        time.sleep(MOCK_LATENCY_SECONDS)          # blocks the OS thread
        results.append(MOCK_LATENCY_SECONDS)

    elapsed = time.perf_counter() - start
    logger.info("sync total: %.3fs", elapsed)
    return {"calls": n, "elapsed_s": round(elapsed, 3), "mode": "sync"}


# --- Async endpoint (non-blocking) ---

@app.get("/async")
async def async_endpoint(n: Annotated[int, Query(ge=1, le=10)] = 3) -> dict[str, float]:
    """
    Awaits a slow operation `n` times sequentially (still sequential here —
    see day-2.md for concurrent gather patterns).
    Runs in the event loop thread; does NOT block other requests.
    """
    start = time.perf_counter()
    results: list[float] = []

    for i in range(n):
        logger.info("async call %d/%d — awaiting %.2fs", i + 1, n, MOCK_LATENCY_SECONDS)
        await asyncio.sleep(MOCK_LATENCY_SECONDS)  # yields to event loop
        results.append(MOCK_LATENCY_SECONDS)

    elapsed = time.perf_counter() - start
    logger.info("async total: %.3fs", elapsed)
    return {"calls": n, "elapsed_s": round(elapsed, 3), "mode": "async"}


# --- Async endpoint calling a real external service ---

@app.get("/fetch")
async def fetch_endpoint(url: str = "https://httpbin.org/delay/1") -> dict[str, object]:
    """
    Uses httpx.AsyncClient to make a non-blocking HTTP call.
    httpx is the async-native replacement for requests.
    """
    logger.info("fetching %s", url)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    elapsed = time.perf_counter() - start
    logger.info("fetch done: status=%d elapsed=%.3fs", response.status_code, elapsed)
    return {
        "url": url,
        "status": response.status_code,
        "elapsed_s": round(elapsed, 3),
    }


# --- Minimal standalone runner (without uvicorn CLI) ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("week08.day1_async_fastapi:app", host="127.0.0.1", port=8000, reload=False)
```

**Key observations to test:**
- Hit `/sync?n=5` and `/async?n=5` simultaneously from two browser tabs. The async version lets the second request interleave; the sync version queues it.
- Watch the logs — async logs "awaiting" in interleaved order across concurrent requests; sync logs are strictly sequential per thread.

---

## Exercises

1. **Timing comparison.** Write a script using `asyncio.run()` and `httpx.AsyncClient` that fetches `https://httpbin.org/delay/1` five times. First implement it sequentially with a `for` loop of `await client.get(...)`. Measure total time. Then switch to `asyncio.gather()` (preview — covered in day-2). How do the times compare? Explain the difference in terms of event loop suspension points.

2. **Blocking detection.** Add a new FastAPI route `/cpu` that computes `sum(i * i for i in range(10_000_000))` inside an `async def` handler. Start the server with two workers (`--workers 2`). Hit `/cpu` from one tab and `/async?n=1` from another simultaneously. What do you observe? Fix it using `asyncio.get_event_loop().run_in_executor(None, ...)` or `fastapi.concurrency.run_in_threadpool`. Explain why CPU work in `async def` blocks the event loop.

3. **Event loop inspection.** Write a short script (not FastAPI) that:
   - Creates a coroutine `async def greet(name: str) -> str` that awaits `asyncio.sleep(0.1)` then returns `f"Hello, {name}"`.
   - Runs it with `asyncio.run()`.
   - Inspects the return value and logs it with `logging`.
   - Then tries calling the coroutine *without* `await` or `asyncio.run()` — what does Python return? What does that tell you about coroutine objects?

*(Solutions: ../solutions/week-08/day-1.md)*

---

## Self-check

- [ ] I can explain why async I/O improves throughput for I/O-bound work but not CPU-bound work.
- [ ] I can describe the three phases of one event loop tick (poll, schedule callbacks, resume coroutines) without referring to notes.
- [ ] I can implement an async FastAPI endpoint using `async def` and `await asyncio.sleep()`, and articulate when `time.sleep()` inside an `async def` is dangerous.

---

## Supplementary

- Python asyncio docs: <https://docs.python.org/3/library/asyncio.html>
- uvloop (faster event loop): <https://github.com/MagicStack/uvloop>
- FastAPI async docs: <https://fastapi.tiangolo.com/async/>
