# Week 08 · Day 3 — Solutions
## Async HTTP & I/O Patterns

---

### Exercise 1
**Write `download_files(urls: list[str], output_dir: Path) -> list[Path]` using `httpx.AsyncClient` that downloads files concurrently to disk using `aiofiles`, returns list of written paths.**

```python
import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import httpx

logger = logging.getLogger(__name__)


async def _download_one(
    url: str,
    output_dir: Path,
    client: httpx.AsyncClient,
) -> Path:
    """Download a single URL and write it to output_dir.

    Filename is derived from the URL path; falls back to a hash of the URL
    if the path has no filename component.
    """
    # Derive filename from URL path; fall back to url hash
    parsed = urlparse(url)
    filename = Path(parsed.path).name or f"{abs(hash(url))}.bin"
    dest = output_dir / filename

    logger.debug("Starting download: %s -> %s", url, dest)

    async with client.stream("GET", url) as response:
        response.raise_for_status()
        async with aiofiles.open(dest, "wb") as fh:
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                await fh.write(chunk)

    logger.info("Saved %s (%d bytes)", dest.name, dest.stat().st_size)
    return dest


async def download_files(
    urls: list[str],
    output_dir: Path,
) -> list[Path]:
    """Download all URLs concurrently to output_dir.

    Args:
        urls:       List of HTTP/HTTPS URLs to download.
        output_dir: Directory to write files into (must exist).

    Returns:
        List of Path objects for each written file, in the same order as urls.

    Raises:
        httpx.HTTPStatusError: If any URL returns a non-2xx status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
    ) as client:
        paths: list[Path] = await asyncio.gather(
            *[_download_one(url, output_dir, client) for url in urls],
            return_exceptions=False,
        )
    return paths


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    urls = [
        "https://httpbin.org/bytes/1024",
        "https://httpbin.org/bytes/2048",
        "https://httpbin.org/bytes/4096",
    ]
    output_dir = Path("/tmp/async_downloads")
    paths = await download_files(urls, output_dir)
    for p in paths:
        logger.info("Downloaded: %s  (%d bytes)", p, p.stat().st_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
```

**Key points:**
- `client.stream("GET", url)` streams the response body in chunks rather than buffering it entirely in memory — critical for large files.
- `aiofiles.open(dest, "wb")` gives an async file handle whose `write` method yields to the event loop, so disk I/O does not block other downloads.
- `output_dir.mkdir(parents=True, exist_ok=True)` is idempotent — safe to call even if the directory already exists.
- All downloads run concurrently because `asyncio.gather` schedules them simultaneously; total time approaches the slowest single download, not the sum.

---

### Exercise 2
**Implement `fetch_with_retry(client, url, max_retries=3)` with exponential backoff (1s, 2s, 4s) + jitter, retrying on 429 or 5xx status codes.**

```python
import asyncio
import logging
import random

import httpx

logger = logging.getLogger(__name__)

# Status codes that warrant a retry
RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    max_retries: int = 3,
) -> httpx.Response:
    """GET a URL, retrying on 429 or 5xx with exponential backoff + jitter.

    Backoff schedule: 1s, 2s, 4s (base * 2**attempt) with ±50% jitter.
    Raises the last exception / returns the last response on exhaustion.

    Args:
        client:      Shared AsyncClient (caller manages lifetime).
        url:         URL to fetch.
        max_retries: Maximum number of retry attempts (not including the first).

    Returns:
        httpx.Response — may be an error response if all retries are exhausted.
    """
    base_delay = 1.0
    last_response: httpx.Response | None = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(url)
            last_response = response

            if response.status_code not in RETRYABLE_STATUSES:
                # Success or a non-retryable error (e.g. 400, 401, 404)
                return response

            if attempt == max_retries:
                logger.error(
                    "Exhausted %d retries for %s — last status %d",
                    max_retries, url, response.status_code,
                )
                return response

            # Calculate backoff with full jitter
            delay = base_delay * (2 ** attempt)
            jitter = delay * random.uniform(-0.5, 0.5)
            sleep_for = max(0.0, delay + jitter)

            logger.warning(
                "Attempt %d/%d: %s returned %d — retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                url,
                response.status_code,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)

        except (httpx.TransportError, httpx.TimeoutException) as exc:
            if attempt == max_retries:
                logger.error("Network error on final attempt for %s: %s", url, exc)
                raise

            delay = base_delay * (2 ** attempt)
            jitter = delay * random.uniform(-0.5, 0.5)
            sleep_for = max(0.0, delay + jitter)

            logger.warning(
                "Attempt %d/%d: %s raised %s — retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                url,
                type(exc).__name__,
                sleep_for,
            )
            await asyncio.sleep(sleep_for)

    # Unreachable, but keeps mypy happy
    assert last_response is not None
    return last_response


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # httpbin returns the status code you ask for
        response = await fetch_with_retry(client, "https://httpbin.org/status/503")
        logger.info("Final status: %d", response.status_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
```

**Key points:**
- Exponential backoff (base × 2^attempt) prevents thundering herd when a server is overloaded.
- Jitter (`±50%` of the base delay) randomises retry timing so concurrent clients don't all retry at the same moment.
- Separating `RETRYABLE_STATUSES` into a constant makes it easy to adjust without touching the function logic.
- Network-level errors (`TransportError`, `TimeoutException`) are also retried since they often indicate transient infrastructure issues.
- 429 (Too Many Requests) ideally uses the `Retry-After` header — check `response.headers.get("Retry-After")` for a more precise delay in production code.

---

### Exercise 3
**Build a connection pool demo: create one `httpx.AsyncClient` at application startup (FastAPI lifespan), reuse it across 10 concurrent requests to the same host, log actual connection reuse.**

```python
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state: one shared client for the lifetime of the process
# ---------------------------------------------------------------------------

class AppState:
    http_client: httpx.AsyncClient


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create the shared AsyncClient on startup; close it on shutdown.

    A single client maintains a connection pool. Concurrent requests to the
    same host reuse existing TCP connections (HTTP keep-alive), reducing
    latency and OS resource usage.
    """
    logger.info("Creating shared httpx.AsyncClient")
    state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=20,        # global pool cap
            max_keepalive_connections=10,  # idle connections to keep open
            keepalive_expiry=30.0,     # idle connection TTL in seconds
        ),
        timeout=httpx.Timeout(10.0),
    )
    yield
    logger.info("Closing shared httpx.AsyncClient")
    await state.http_client.aclose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Route that fans out 10 concurrent requests using the shared client
# ---------------------------------------------------------------------------

async def _single_fetch(index: int, client: httpx.AsyncClient) -> dict:
    url = "https://httpbin.org/get"
    response = await client.get(url)
    # httpx exposes connection details via response.extensions
    conn_info = response.extensions.get("network_stream")
    logger.info(
        "Request %d: status=%d  reused_connection=%s",
        index,
        response.status_code,
        conn_info is not None,
    )
    return {"index": index, "status": response.status_code}


@app.get("/pool-demo")
async def pool_demo(request: Request) -> dict:
    """Fire 10 concurrent requests using the shared client connection pool."""
    client: httpx.AsyncClient = state.http_client

    results = await asyncio.gather(
        *[_single_fetch(i, client) for i in range(10)]
    )

    return {"requests": results, "total": len(results)}


# ---------------------------------------------------------------------------
# Standalone script equivalent (without FastAPI)
# ---------------------------------------------------------------------------

async def standalone_demo() -> None:
    """Run the pool demo outside FastAPI for local testing."""
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=5),
        timeout=httpx.Timeout(10.0),
    ) as client:
        results = await asyncio.gather(
            *[_single_fetch(i, client) for i in range(10)]
        )
        logger.info("All done: %d results", len(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(standalone_demo())
```

**Key points:**
- The FastAPI `lifespan` context manager (replacing the deprecated `@app.on_event` hooks) is the correct place to initialise and tear down shared resources.
- One `httpx.AsyncClient` per application, not one per request — creating a client per request defeats connection pooling and causes socket exhaustion under load.
- `httpx.Limits` controls pool size explicitly. `max_keepalive_connections` limits idle connections held open; `max_connections` caps total concurrent connections.
- In the standalone demo, `async with httpx.AsyncClient()` ensures `aclose()` is called even if an exception occurs, preventing connection leaks.
- Connection reuse is observable via `response.extensions` in httpx 0.24+ or by monitoring OS-level metrics (`ss -s` / `netstat`) — the connection count stays flat across 10 requests when pooling works correctly.
