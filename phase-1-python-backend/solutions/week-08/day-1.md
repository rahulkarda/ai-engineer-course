# Week 08 · Day 1 — Solutions
## Async & the Event Loop

---

### Exercise 1
**Write an async function `fetch_status(url: str) -> dict` using `httpx.AsyncClient` that returns `{"url": url, "status": code, "elapsed_ms": float}`. Add a 5-second timeout.**

```python
import logging
import time

import httpx

logger = logging.getLogger(__name__)


async def fetch_status(url: str) -> dict:
    """Fetch a URL and return status code with elapsed time in milliseconds.

    Uses a 5-second timeout. On network or timeout errors the status is -1
    and elapsed_ms reflects how long the attempt took.
    """
    timeout = httpx.Timeout(5.0)
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.debug("GET %s -> %d (%.1f ms)", url, response.status_code, elapsed_ms)
            return {
                "url": url,
                "status": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            }
    except httpx.TimeoutException:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Timeout fetching %s after %.1f ms", url, elapsed_ms)
        return {"url": url, "status": -1, "elapsed_ms": round(elapsed_ms, 2)}
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Request error for %s: %s", url, exc)
        return {"url": url, "status": -1, "elapsed_ms": round(elapsed_ms, 2)}
```

**Key points:**
- `httpx.Timeout(5.0)` applies the 5-second limit to connect, read, write, and pool phases. Pass it at client construction so every request in the `async with` block inherits it.
- `time.perf_counter()` gives sub-millisecond resolution; `time.time()` would also work but has lower resolution on some platforms.
- Returning a sentinel status `-1` instead of re-raising keeps callers simple — they can check `result["status"] == -1` rather than wrapping every call in try/except.
- `httpx.TimeoutException` and `httpx.RequestError` cover timeouts and network-level failures respectively; HTTP 4xx/5xx come through normally as status codes, not exceptions.

---

### Exercise 2
**Rewrite a sync FastAPI route `GET /items/{item_id}` that calls `time.sleep(0.1)` as a proper async route using `asyncio.sleep(0.1)`.**

Sync version (what NOT to do):

```python
# BAD — blocks the event loop thread
import time
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def get_item_sync(item_id: int) -> dict:
    time.sleep(0.1)          # blocks the entire event loop
    return {"item_id": item_id}
```

Async version (correct):

```python
import asyncio
import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)
app = FastAPI()


@app.get("/items/{item_id}")
async def get_item(item_id: int) -> dict:
    """Return item details after a simulated async delay.

    asyncio.sleep yields control back to the event loop so other
    requests can be handled during the wait.
    """
    logger.debug("Processing item %d", item_id)
    await asyncio.sleep(0.1)   # yields; event loop can serve other requests
    return {"item_id": item_id, "name": f"Item #{item_id}"}
```

**Key points:**
- `time.sleep()` inside an `async def` route blocks the entire event loop for the duration — no other coroutine can run. This is one of the most common async anti-patterns.
- `await asyncio.sleep()` suspends the current coroutine and hands control back to the event loop, allowing other requests to be processed concurrently.
- For truly blocking work that cannot be made async (e.g., a CPU-bound library), use `await asyncio.get_event_loop().run_in_executor(None, blocking_fn)` instead of `asyncio.sleep`.

---

### Exercise 3
**Write a script that uses `asyncio.run(main())` where `main()` gathers 5 concurrent `fetch_status` calls and prints results.**

```python
import asyncio
import logging
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

URLS = [
    "https://httpbin.org/status/200",
    "https://httpbin.org/status/404",
    "https://httpbin.org/delay/1",
    "https://httpbin.org/status/500",
    "https://httpbin.org/get",
]


async def fetch_status(url: str, client: httpx.AsyncClient) -> dict:
    """Fetch a URL using a shared client; return status metadata."""
    timeout = httpx.Timeout(5.0)
    start = time.perf_counter()
    try:
        response = await client.get(url, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "url": url,
            "status": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
        }
    except httpx.TimeoutException:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"url": url, "status": -1, "elapsed_ms": round(elapsed_ms, 2)}


async def main() -> None:
    """Run 5 fetch_status coroutines concurrently, print results."""
    wall_start = time.perf_counter()

    # Share one client across all concurrent requests (connection pooling)
    async with httpx.AsyncClient() as client:
        results: list[dict] = await asyncio.gather(
            *[fetch_status(url, client) for url in URLS]
        )

    wall_elapsed = (time.perf_counter() - wall_start) * 1000
    logger.info("All requests finished in %.1f ms (wall clock)", wall_elapsed)

    for r in results:
        logger.info("  %-45s  status=%-4s  elapsed=%.1f ms", r["url"], r["status"], r["elapsed_ms"])


if __name__ == "__main__":
    # asyncio.run() creates a fresh event loop, runs main(), then closes it.
    # Equivalent to the old asyncio.get_event_loop().run_until_complete(main())
    # but cleaner and handles loop cleanup automatically.
    asyncio.run(main())
```

**Key points:**
- `asyncio.run()` is the canonical entry point since Python 3.7+. It creates a new event loop, runs the coroutine, and closes the loop — avoiding the "no current event loop" issues of the older `get_event_loop()` pattern.
- `asyncio.gather(*coros)` schedules all coroutines concurrently. Total wall time approaches the slowest request, not the sum of all requests.
- Sharing one `httpx.AsyncClient` across all concurrent requests allows connection reuse (HTTP keep-alive), which reduces latency compared to opening a new client per call.
- `asyncio.gather` returns results in the same order as the input coroutines, regardless of completion order.
