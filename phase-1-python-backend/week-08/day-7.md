# Phase 1 · Week 08 · Day 7 — Production Engineering: Logging, Health Checks & Error Handling

> Objective: Make a FastAPI app production-ready with structured logging, correlation IDs, health endpoints, graceful shutdown, and robust retry logic.  ·  Time: ~4 hr  ·  Prereqs: [[day-6]]

---

## Concept

### Structured Logging (JSON Logs)

Plain text logs like `"ERROR: database connection failed"` are hard to query in aggregation tools (Datadog, Splunk, CloudWatch). Structured logging emits each event as a JSON object:

```json
{"timestamp": "2026-11-11T10:15:03Z", "level": "ERROR", "message": "db connection failed",
 "correlation_id": "c3f1a2b9", "service": "my-api", "host": "pod-abc123"}
```

Every field is queryable. You can filter by `correlation_id` to trace a single request across microservices, or by `level` to alert on errors.

### Log Levels

| Level | Use |
|-------|-----|
| `DEBUG` | Detailed internal state — off in production |
| `INFO` | Normal operational events (startup, request completed) |
| `WARNING` | Unexpected but recoverable (cache miss, retry attempt) |
| `ERROR` | Operation failed, needs attention |
| `CRITICAL` | System cannot continue |

Production default: `INFO`. Never log secrets, tokens, or PII at any level.

### Correlation IDs

A correlation ID (also called trace ID or request ID) is a UUID generated at the start of each request. Every log line for that request includes the same ID. Middleware injects it into the request context; the logger picks it up automatically via Python's `logging` context variables or `contextvars`.

### Health Check Endpoints

Two endpoints serve different audiences:

- **`/health`** (liveness): "Is the process alive?" — responds 200 as long as the process is running. Kubernetes uses this to decide whether to restart the pod.
- **`/ready`** (readiness): "Is this instance ready to serve traffic?" — checks DB connectivity, external service reachability, etc. Kubernetes uses this to decide whether to send traffic.

A pod can be alive (healthy) but not ready (e.g., still warming up cache). Splitting them prevents routing traffic to an instance that can't serve it.

### Graceful Shutdown

When a container receives `SIGTERM` (from `docker stop`, Kubernetes scaling down, deploy rollover), it should:

1. Stop accepting new requests
2. Finish in-flight requests
3. Close DB connections and release resources
4. Exit cleanly

FastAPI's `lifespan` context manager handles startup and shutdown in one function, replacing the deprecated `@app.on_event` decorators.

### Retry Strategies with tenacity

External calls (HTTP to third-party APIs, DB queries during transient network blips) fail intermittently. A retry strategy specifies:

- How many times to retry
- How long to wait between attempts (fixed, exponential backoff, jitter)
- Which exceptions to retry vs. which to raise immediately

`tenacity` provides a clean decorator API for this. Exponential backoff with jitter avoids thundering-herd problems where many services all retry simultaneously after an outage.

---

## How It Works Internally

### Python Logging Hierarchy

Python's `logging` module uses a tree of loggers named by dotted paths: `myapp`, `myapp.db`, `myapp.routers.items`. A log record propagates up the tree until a handler processes it. Setting the level on the root logger (`logging.getLogger()`) affects all loggers below it. Setting it on `myapp` only affects that subtree.

```
root
└── myapp
    ├── myapp.db
    └── myapp.routers
        └── myapp.routers.items
```

### contextvars for Request-Scoped State

`contextvars.ContextVar` is Python's mechanism for storing values that are scoped to the current async task (or OS thread). Middleware sets `correlation_id_var.set(new_id)` at request start. A log filter reads `correlation_id_var.get()` when formatting each record. Because asyncio tasks inherit a copy of the context from their parent, the ID flows through all `await` calls within one request without explicit passing.

### FastAPI Lifespan Events

The `@asynccontextmanager lifespan` function runs startup code (before `yield`) when uvicorn initializes the app, and teardown code (after `yield`) when uvicorn shuts down. Uvicorn catches `SIGTERM` and `SIGINT`, cancels the running event loop tasks, and calls the lifespan teardown — giving you a window to close connections gracefully.

### SIGTERM vs SIGINT

- `SIGTERM`: polite shutdown request — the default signal from `docker stop`, Kubernetes, `kill <pid>`. Catchable; uvicorn handles it.
- `SIGINT`: interrupt from Ctrl+C — also catchable, also handled by uvicorn.
- `SIGKILL`: immediate kill — not catchable. Sent by Docker after the stop timeout (default 10s) if SIGTERM is not handled.

Set `docker stop --time 30` (or `stop_grace_period: 30s` in compose) to give the app enough time for graceful shutdown.

---

## Worked Example

### `app/logging_config.py` — Structured JSON Logger

```python
import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Module-level ContextVar: holds the correlation ID for the current request
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default="-"
)


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON output to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
```

### `app/middleware.py` — Correlation ID Middleware

```python
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import correlation_id_var

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a correlation ID to every request and add it to the response headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Honour upstream ID if provided (e.g. from an API gateway)
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        token = correlation_id_var.set(cid)
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = cid
            return response
        finally:
            correlation_id_var.reset(token)
```

### `app/health.py` — Health and Readiness Endpoints

```python
import logging
import os
from typing import Any

import asyncpg
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict[str, str]:
    """Liveness probe — is the process alive?"""
    return {"status": "ok"}


@router.get("/ready")
async def readiness() -> dict[str, Any]:
    """
    Readiness probe — can this instance serve traffic?
    Checks: PostgreSQL connectivity.
    Returns 503 if any dependency is unavailable.
    """
    checks: dict[str, str] = {}
    healthy = True

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        checks["postgres"] = "DATABASE_URL not configured"
        healthy = False
    else:
        try:
            conn = await asyncpg.connect(database_url, timeout=3)
            await conn.execute("SELECT 1")
            await conn.close()
            checks["postgres"] = "ok"
        except Exception as exc:
            logger.warning("Readiness check: postgres unreachable: %s", exc)
            checks["postgres"] = "unreachable"
            healthy = False

    status_code = 200 if healthy else 503
    body = {"status": "ready" if healthy else "not_ready", "checks": checks}

    if not healthy:
        raise HTTPException(status_code=503, detail=body)
    return body
```

### `app/retry.py` — Tenacity Retry Decorator

```python
import logging
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> Callable[[F], F]:
    """
    Decorator: retry on any Exception with exponential backoff.

    Usage:
        @with_retry(max_attempts=4, min_wait=0.5, max_wait=8.0)
        async def call_external_api(url: str) -> dict: ...
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
```

### `app/exception_handlers.py` — Global Exception Handler

```python
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_config import correlation_id_var

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI app."""

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        cid = correlation_id_var.get()
        logger.error(
            "Unhandled exception on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
            extra={"correlation_id": cid},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "correlation_id": cid,
                "detail": "An unexpected error occurred.",
            },
        )
```

### `app/main.py` — Wiring It All Together

```python
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
from fastapi import FastAPI

from app.exception_handlers import register_exception_handlers
from app.health import router as health_router
from app.logging_config import configure_logging
from app.middleware import CorrelationIdMiddleware

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _db_pool

    # ── Startup ──────────────────────────────────────────────
    logger.info("Starting up")
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            _db_pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
            logger.info("Database pool created")
        except Exception as exc:
            logger.error("Failed to create DB pool: %s", exc)
    else:
        logger.warning("DATABASE_URL not set — DB pool skipped")

    yield  # app is live here

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down — closing DB pool")
    if _db_pool:
        await _db_pool.close()
    logger.info("Shutdown complete")


app = FastAPI(title="My API", version="1.0.0", lifespan=lifespan)

app.add_middleware(CorrelationIdMiddleware)
app.include_router(health_router)
register_exception_handlers(app)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok"}
```

---

## Exercises

> Solutions: ../solutions/week-08/day-7.md

**Exercise 1 — Correlation ID Middleware**

Add `CorrelationIdMiddleware` to a FastAPI app:

1. Generate a UUID per request if `X-Correlation-ID` is not in the incoming headers; otherwise use the provided value
2. Store it in a `ContextVar`
3. Include it in every log line by writing a `logging.Filter` subclass that reads the `ContextVar` and injects `correlation_id` into each `LogRecord`
4. Return it as `X-Correlation-ID` in the response headers

Test: call `GET /` twice — each response should have a different `X-Correlation-ID`; logs for each request should share its ID.

---

**Exercise 2 — `/health` and `/ready` Endpoints**

Implement both endpoints:

- `GET /health` returns `{"status": "ok"}` with HTTP 200 always (liveness)
- `GET /ready` returns HTTP 200 with `{"status": "ready", "checks": {...}}` when all deps are up, or HTTP 503 with `{"status": "not_ready", "checks": {...}}` when any dep is down

Checks to include:
- PostgreSQL: attempt `SELECT 1` with a 3-second timeout
- Redis: attempt `PING` with a 1-second timeout

Each check key should be `"ok"` or `"unreachable"` in the `checks` dict. All connection strings must come from `os.getenv()`. Log a `WARNING` (not `ERROR`) for each unreachable dependency.

---

**Exercise 3 — Retry with Backoff Using tenacity**

Write a `with_retry` decorator factory using `tenacity`:

1. Parameters: `max_attempts: int`, `min_wait: float`, `max_wait: float`
2. Uses `wait_exponential` with the given min/max
3. Logs the exception and attempt number before each sleep (use `before_sleep_log`)
4. Re-raises the final exception after exhausting attempts

Apply it to a function that calls an external HTTP endpoint using `httpx.AsyncClient`. The function should accept a URL and return a `dict`. Write a unit test using `pytest` and `unittest.mock.AsyncMock` that verifies the function retries exactly 3 times before raising.

---

**Exercise 4 — Global Exception Handler with Request Context**

Add a global exception handler to a FastAPI app:

1. Catches all unhandled `Exception` types
2. Logs at `ERROR` level with: method, path, correlation ID, and full traceback (`exc_info=True`)
3. Returns JSON `{"error": "internal_server_error", "correlation_id": "...", "detail": "An unexpected error occurred."}`  with HTTP 500
4. Does NOT expose internal exception messages or stack traces to the client

Add a test route `GET /kaboom` that raises `RuntimeError("intentional test error")`. Write a `pytest` test using `TestClient` that:
- Calls `/kaboom`
- Asserts status code is 500
- Asserts the response body contains `"correlation_id"`
- Asserts the response body does NOT contain `"intentional test error"`

---

## Self-Check

1. **What is the difference between `/health` and `/ready`?** Give a concrete scenario where a pod is healthy but not ready.

2. **Why use `contextvars.ContextVar` rather than a global variable for the correlation ID in an async FastAPI app?**

3. **tenacity's `wait_exponential` with `jitter`** avoids a specific failure mode — what is it called, and why does random jitter help?

4. **What happens if `lifespan` teardown (after `yield`) raises an exception?** Does the app still shut down? What should you do to ensure cleanup always runs?

---

## Supplementary Resources

- Python logging HOWTO: https://docs.python.org/3/library/logging.html
- tenacity docs: https://tenacity.readthedocs.io/
- FastAPI lifespan events: https://fastapi.tiangolo.com/advanced/events/
- asyncpg docs: https://magicstack.github.io/asyncpg/current/
