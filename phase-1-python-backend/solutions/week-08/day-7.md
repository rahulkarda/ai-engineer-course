# Week 08 Day 7 — Production Engineering: Answer Key

## Exercise 1: Correlation ID Middleware

```python
# src/middleware/correlation_id.py
import logging
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Module-level ContextVar — one value per async task (per request)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

HEADER_NAME = "X-Correlation-ID"

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Per-request correlation ID middleware.

    On each incoming request:
    1. Reads X-Correlation-ID header from the client (allows trace propagation
       from upstream services / API gateways).
    2. Generates a new UUID4 if the header is absent.
    3. Stores the ID in a ContextVar — safe for async because each task gets
       its own copy; no shared mutable state.
    4. Adds the ID to the response header so clients can correlate logs.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Honour forwarded ID from upstream; generate new one if absent
        cid = request.headers.get(HEADER_NAME) or str(uuid.uuid4())
        token = correlation_id_var.set(cid)

        try:
            response = await call_next(request)
        finally:
            # Always restore context even if the handler raises
            correlation_id_var.reset(token)

        response.headers[HEADER_NAME] = cid
        return response


class CorrelationIdFilter(logging.Filter):
    """
    Injects correlation_id into every LogRecord so formatters can include it.
    Attach to root logger or specific handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get() or "no-context"
        return True


def configure_logging() -> None:
    """
    Set up JSON-friendly structured logging with correlation IDs.
    In production replace this with python-json-logger or structlog.
    """
    fmt = "%(asctime)s %(levelname)s [%(correlation_id)s] %(name)s: %(message)s"
    handler = logging.StreamHandler()
    handler.addFilter(CorrelationIdFilter())
    handler.setFormatter(logging.Formatter(fmt))
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
```

```python
# src/main.py — wiring the middleware
from fastapi import FastAPI
from src.middleware.correlation_id import CorrelationIdMiddleware, configure_logging

configure_logging()
app = FastAPI()
app.add_middleware(CorrelationIdMiddleware)
```

### Why `ContextVar` and Not a Global?

A plain `global cid = ""` would be overwritten by every concurrent request.
`ContextVar` gives each `asyncio` Task its own isolated slot — the event loop copies
the current context when it creates a new task, so coroutines spawned inside a request
handler automatically inherit the same correlation ID without any explicit passing.

---

## Exercise 2: Health and Readiness Endpoints

```python
# src/routers/health.py
import logging
import os

from fastapi import APIRouter, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["observability"])

# Engine created lazily from environment — no hardcoded DSN
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL is not set")
        _engine = create_async_engine(dsn, pool_pre_ping=True)
    return _engine


@router.get("/health")
async def health() -> dict:
    """
    Liveness probe.
    Returns 200 immediately — if the process is alive enough to handle HTTP,
    it's alive. Kubernetes uses this to decide whether to restart the container.
    Never put slow checks (DB, Redis) here.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready(response: Response) -> dict:
    """
    Readiness probe.
    Returns 200 only when the app is ready to serve traffic (DB reachable).
    Kubernetes removes the pod from the load-balancer until this returns 200.
    Returns 503 with a detail message on failure so ops can diagnose quickly.
    """
    try:
        engine = _get_engine()
        async with AsyncSession(engine) as session:
            await session.execute(text("SELECT 1"))
        logger.debug("/ready: database OK")
        return {"status": "ready", "db": "ok"}
    except Exception as exc:
        logger.error("/ready: database check failed: %s", exc)
        response.status_code = 503
        return {"status": "unavailable", "detail": str(exc)}
```

### Liveness vs Readiness

| Probe | When to fail | Effect |
|---|---|---|
| `/health` (liveness) | Process deadlocked / crashed | Container restarted |
| `/ready` (readiness) | DB unreachable, migrations pending, warming up | Pod removed from load balancer |

Never put DB checks in liveness — a slow DB would cause infinite restart loops.

---

## Exercise 3: Retry with Tenacity

```python
# src/utils/retry.py
import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,  # re-raise the original exception after all attempts fail
)
async def fetch_with_retry(url: str, client: httpx.AsyncClient) -> httpx.Response:
    """
    Fetch *url* with exponential backoff retry on httpx.HTTPError.

    Attempt 1 → immediate
    Attempt 2 → wait 1s  (min)
    Attempt 3 → wait 2s  (multiplier * 2^1)
    After 3 failures → raises the last httpx.HTTPError

    Backoff formula: min(max, multiplier * 2^(attempt - 1))
    With multiplier=1, min=1, max=10: 1s, 2s, 4s, 8s, 10s, 10s, ...
    """
    logger.debug("GET %s", url)
    response = await client.get(url, timeout=10.0)
    response.raise_for_status()  # raises httpx.HTTPStatusError (subclass of HTTPError)
    return response
```

### What Each Parameter Does

| Parameter | Value | Effect |
|---|---|---|
| `stop_after_attempt(3)` | 3 | Give up after the 3rd failure total |
| `wait_exponential(multiplier=1, min=1, max=10)` | — | Wait 1s, 2s, 4s… capped at 10s between attempts |
| `retry_if_exception_type(httpx.HTTPError)` | — | Only retry on HTTP errors; don't retry `ValueError` etc. |
| `before_sleep` | log at WARNING | Logs "Retrying in Xs after exception: …" before each sleep |
| `reraise=True` | — | After all attempts fail, raise original exception (not `RetryError`) |

### Retry on Status Code (not just transport errors)

```python
from tenacity import retry_if_result

def is_server_error(response: httpx.Response) -> bool:
    return response.status_code >= 500

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_result(is_server_error),
)
async def fetch_with_status_retry(url: str, client: httpx.AsyncClient) -> httpx.Response:
    return await client.get(url, timeout=10.0)
```

---

## Exercise 4: Global Exception Handler

```python
# src/exception_handlers.py
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.middleware.correlation_id import correlation_id_var

logger = logging.getLogger(__name__)


def _correlation_id() -> str:
    return correlation_id_var.get() or "no-context"


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all global exception handlers on *app*.
    Call this after creating the FastAPI instance, before adding routes.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Pydantic / FastAPI request validation failure → 422 Unprocessable Entity.
        Log at WARNING — this is a client error, not a server fault.
        """
        cid = _correlation_id()
        logger.warning(
            "validation error",
            extra={
                "correlation_id": cid,
                "path": str(request.url),
                "errors": exc.errors(),
            },
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": exc.errors(),
                "correlation_id": cid,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        Explicit HTTPException raised in route handlers → pass status code through.
        4xx are client errors (WARNING); 5xx are server errors (ERROR).
        """
        cid = _correlation_id()
        level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
        logger.log(
            level,
            "http exception %d: %s",
            exc.status_code,
            exc.detail,
            extra={"correlation_id": cid, "path": str(request.url)},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "correlation_id": cid,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for any unhandled exception → 500 Internal Server Error.
        Log at ERROR with full traceback (exc_info=True).
        Never leak internal details (stack traces, SQL) to the client.
        """
        cid = _correlation_id()
        logger.error(
            "unhandled exception",
            exc_info=exc,
            extra={
                "correlation_id": cid,
                "path": str(request.url),
                "method": request.method,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "correlation_id": cid,  # client includes this in bug reports
            },
        )
```

```python
# src/main.py — complete wiring example
from fastapi import FastAPI
from src.exception_handlers import register_exception_handlers
from src.middleware.correlation_id import CorrelationIdMiddleware, configure_logging
from src.routers import health

configure_logging()
app = FastAPI()
app.add_middleware(CorrelationIdMiddleware)
register_exception_handlers(app)
app.include_router(health.router)
```

### Why Include Correlation ID in Error Responses?

When the client receives a 500, they call support with a ticket. Support searches the
logs for `correlation_id = "abc-123"` and finds the exact request, the traceback, the
DB query that failed — everything, in order. Without a correlation ID, finding the right
log line in a high-traffic service is guesswork.

### Handler Registration Order

FastAPI evaluates exception handlers from most-specific to most-general. Register
`RequestValidationError` before `Exception` — if you register `Exception` first it
shadows everything else. The `register_exception_handlers` function groups them in one
place to avoid ordering bugs.
