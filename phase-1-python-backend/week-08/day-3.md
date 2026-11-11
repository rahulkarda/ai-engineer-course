# Phase 1 · Week 08 · Day 3 — Async HTTP & I/O Patterns

> Objective: Use `httpx` and `aiofiles` for production-grade async I/O; implement connection pooling, streaming, and retry with exponential backoff.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

### Async HTTP Clients

`requests` is synchronous — it blocks the OS thread waiting for each response. For async code, you need a client built on non-blocking sockets.

Two leading choices:

| Library | Notes |
|---|---|
| `httpx` | Drop-in `requests`-compatible API, built-in async support, HTTP/2, type-safe. Preferred. |
| `aiohttp` | Older, more battle-tested at scale, slightly more low-level API. |

Both provide:
- **AsyncClient / ClientSession**: long-lived objects with a connection pool. Never create a new client per request.
- **Streaming**: iterate response body chunks without buffering the full response in memory.
- **Timeout controls**: connect timeout, read timeout, total timeout.

### Connection Pools

A connection pool keeps TCP connections open and reused across requests to the same host. Without pooling, every request pays:

1. TCP 3-way handshake (~1 RTT)
2. TLS 1.3 handshake (~1 RTT)
3. HTTP request/response

With pooling, connections 2–N skip steps 1–2 entirely. `httpx.AsyncClient` maintains a pool per host; pool size is configurable via `httpx.Limits`.

### Async File I/O

Standard `open()` + `read()` is blocking — it calls into the OS kernel and may stall the event loop waiting on disk. For correctness under async, use `aiofiles` which wraps file operations in a thread executor, keeping the event loop free.

For large files, always stream: read and process in chunks rather than loading the whole file into memory.

### Backpressure

When a producer generates data faster than a consumer can process it, the intermediate buffer grows unboundedly — this is the backpressure problem. In streaming HTTP:
- The producer is the server sending bytes.
- The consumer is your code processing chunks.
- Backpressure is managed by flow control at the TCP level (receive window) and at the application level by not reading ahead faster than you can process.

`httpx` handles TCP-level flow control automatically. Application-level: process each chunk before reading the next (`async for chunk in response.aiter_bytes()`).

---

## How It Works Internally

### Non-blocking Socket I/O

A socket in non-blocking mode returns immediately (raising `BlockingIOError` / `EAGAIN`) if no data is available instead of sleeping the thread. The event loop registers the socket fd with `epoll`/`kqueue` and resumes the coroutine only when data arrives. The flow:

```
1. coroutine calls await client.get(url)
2. httpx attempts recv() on the socket → EAGAIN (no data yet)
3. httpx registers fd with event loop selector, suspends coroutine
4. event loop runs other work
5. epoll returns: fd is readable
6. event loop resumes httpx internals, which call recv() — data available
7. httpx builds Response, resolves the awaitable
8. coroutine resumes with the Response object
```

### Connection Pool Internals (httpx)

`httpx.AsyncClient` uses `httpcore` under the hood. `httpcore.AsyncConnectionPool` maintains:
- A dict of `(scheme, host, port) → list[Connection]`
- `max_connections` (total) and `max_keepalive_connections` (idle)
- Idle connections expire after `keepalive_expiry` seconds

When all connections to a host are in use and `max_connections` is reached, `acquire()` awaits until one is released — natural rate limiting.

### Exponential Backoff

When a request fails transiently (5xx, network error), you retry with increasing delays:

```
delay = base * (factor ** attempt) + jitter
```

Jitter (a random component) prevents the thundering herd problem: without it, all clients that failed at the same time would retry simultaneously, re-causing the overload.

---

## Worked Example

```python
# week08/day3_async_http.py
"""
Production patterns for async HTTP:
  - Shared AsyncClient with connection pool
  - Parallel requests with gather
  - Streaming large response
  - Retry with exponential backoff + jitter
  - Async file I/O with aiofiles

Requires: pip install httpx aiofiles
Run: python -m week08.day3_async_http
"""

import asyncio
import logging
import math
import random
import time
from pathlib import Path

import aiofiles
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared client — create once, reuse everywhere
# --------------------------------------------------------------------------- #

def build_client() -> httpx.AsyncClient:
    """
    Build a shared AsyncClient with tuned limits and timeouts.
    Use as an async context manager or call aclose() explicitly.
    """
    limits = httpx.Limits(
        max_connections=100,           # total open connections across all hosts
        max_keepalive_connections=20,  # idle connections kept alive
        keepalive_expiry=30.0,         # seconds before idle conn is closed
    )
    timeout = httpx.Timeout(
        connect=5.0,     # time to establish TCP + TLS
        read=30.0,       # time waiting for response bytes
        write=10.0,      # time sending request body
        pool=5.0,        # time waiting for a free connection from the pool
    )
    return httpx.AsyncClient(limits=limits, timeout=timeout, http2=True)


# --------------------------------------------------------------------------- #
# Pattern 1: Parallel requests — shared client, gather
# --------------------------------------------------------------------------- #

async def fetch_all_parallel(urls: list[str]) -> list[dict[str, object]]:
    """
    Fetch all URLs concurrently, sharing one connection pool.
    The client is created once; connections to the same host are reused.
    """
    logger.info("Fetching %d URLs in parallel", len(urls))
    async with build_client() as client:
        async def _one(url: str) -> dict[str, object]:
            t0 = time.perf_counter()
            r = await client.get(url)
            return {
                "url": url,
                "status": r.status_code,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "bytes": len(r.content),
            }

        results: list[dict[str, object]] = await asyncio.gather(
            *[_one(url) for url in urls]
        )
    return results


# --------------------------------------------------------------------------- #
# Pattern 2: Streaming a large response
# --------------------------------------------------------------------------- #

async def stream_to_file(url: str, dest: Path, chunk_size: int = 65_536) -> int:
    """
    Download a large file without buffering the whole body in memory.
    Writes chunks to disk via aiofiles as they arrive.
    Returns total bytes written.
    """
    logger.info("Streaming %s → %s", url, dest)
    total_bytes = 0

    async with build_client() as client:
        # stream=True keeps the connection open and streams the body
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length", "unknown")
            logger.info("  content-length: %s", content_length)

            async with aiofiles.open(dest, "wb") as fp:
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    await fp.write(chunk)
                    total_bytes += len(chunk)

    logger.info("  streamed %d bytes to %s", total_bytes, dest)
    return total_bytes


# --------------------------------------------------------------------------- #
# Pattern 3: Retry with exponential backoff + jitter
# --------------------------------------------------------------------------- #

class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    retryable_statuses: frozenset[int] = frozenset({429, 500, 502, 503, 504}),
) -> httpx.Response:
    """
    GET `url` with exponential backoff retry on transient errors.

    Delay formula:  min(base * factor^attempt, max_delay) + random jitter [0, 1)
    Jitter prevents thundering herd when many clients retry simultaneously.
    """
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        try:
            response = await client.get(url)

            if response.status_code not in retryable_statuses:
                return response

            # Retryable HTTP error
            logger.warning(
                "attempt %d/%d: HTTP %d for %s",
                attempt + 1, max_attempts, response.status_code, url,
            )

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            logger.warning("attempt %d/%d: %s — %s", attempt + 1, max_attempts, type(exc).__name__, exc)
            last_exc = exc

        if attempt < max_attempts - 1:
            delay = min(base_delay * math.pow(backoff_factor, attempt), max_delay)
            delay += random.random()   # jitter
            logger.info("  retrying in %.2fs", delay)
            await asyncio.sleep(delay)

    raise RetryError(
        f"All {max_attempts} attempts failed for {url}"
    ) from last_exc


# --------------------------------------------------------------------------- #
# Pattern 4: Async file read with aiofiles
# --------------------------------------------------------------------------- #

async def read_lines_async(path: Path) -> list[str]:
    """
    Read a text file line by line without blocking the event loop.
    aiofiles delegates to a thread executor under the hood.
    """
    logger.info("Reading %s", path)
    lines: list[str] = []
    async with aiofiles.open(path, "r", encoding="utf-8") as fp:
        async for line in fp:
            lines.append(line.rstrip("\n"))
    logger.info("  read %d lines", len(lines))
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main() -> None:
    # 1. Parallel fetch
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/uuid",
        "https://httpbin.org/ip",
    ]
    results = await fetch_all_parallel(urls)
    for r in results:
        logger.info("  %s → status=%s bytes=%s elapsed=%ss", r["url"], r["status"], r["bytes"], r["elapsed_s"])

    # 2. Streaming download
    tmp_path = Path("/tmp/week08_stream_test.bin")
    await stream_to_file("https://httpbin.org/bytes/102400", tmp_path)
    logger.info("  download size check: %d bytes on disk", tmp_path.stat().st_size)
    tmp_path.unlink(missing_ok=True)

    # 3. Retry demo (using a URL that returns 500 to trigger retries)
    logger.info("=== Retry demo ===")
    async with build_client() as client:
        try:
            resp = await fetch_with_retry(client, "https://httpbin.org/status/500", max_attempts=3, base_delay=0.2)
            logger.info("  final status: %d", resp.status_code)
        except RetryError as exc:
            logger.error("  %s", exc)

    # 4. Async file I/O
    sample_path = Path("/tmp/week08_sample.txt")
    async with aiofiles.open(sample_path, "w") as fp:
        await fp.write("line one\nline two\nline three\n")
    lines = await read_lines_async(sample_path)
    logger.info("  lines: %s", lines)
    sample_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
```

**Key points to notice:**

- `build_client()` is called **once** per logical operation group, not per request. Sharing the client reuses TCP connections.
- Streaming with `client.stream(...)` as an async context manager keeps the connection open; the body is not loaded into memory.
- Retry `delay` grows as `0.5 → 1.0 → 2.0 → 4.0 s` (+ jitter). Jitter randomizes so concurrent retriers don't synchronize.
- `aiofiles.open` has the same API as `open` but all methods are coroutines.

---

## Exercises

1. **Connection pool observability.** Modify `fetch_all_parallel` to launch 50 concurrent requests to `https://httpbin.org/delay/1`. Set `max_connections=5` in the limits. Log the start and end time of each request. Observe in the logs that only 5 run simultaneously while the rest wait for a pool slot. Explain in a comment how `httpcore`'s pool queue enforces this.

2. **Streaming with progress.** Extend `stream_to_file` to accept a `progress_callback: Callable[[int, int | None], None]` parameter (bytes_so_far, total_bytes_or_None). Call it after each chunk. Write a callback that logs a percentage bar when total size is known, and just logs bytes when content-length is absent. Test against a 1 MB URL (`https://httpbin.org/bytes/1048576`).

3. **Retry with respect-Retry-After header.** Extend `fetch_with_retry` to check the `Retry-After` response header on 429 responses. If the header is present and numeric, use that value as the delay instead of the computed backoff. Write a unit test using `unittest.mock.AsyncMock` to simulate a 429 with `Retry-After: 2` followed by a 200 success, and assert the sleep was called with `2.0`.

*(Solutions: ../solutions/week-08/day-3.md)*

---

## Self-check

- [ ] I can explain why a shared `httpx.AsyncClient` is more efficient than creating a new client per request, in terms of TCP and TLS handshakes.
- [ ] I can implement streaming HTTP download using `client.stream()` and `aiter_bytes()` without loading the full response body into memory.
- [ ] I can implement exponential backoff with jitter and explain why jitter prevents the thundering herd problem.

---

## Supplementary

- httpx async docs: <https://www.python-httpx.org/async/>
- aiofiles: <https://github.com/Tinche/aiofiles>
- httpcore connection pool: <https://www.encode.io/httpcore/connection-pools/>
- Exponential backoff — AWS blog: <https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/>
