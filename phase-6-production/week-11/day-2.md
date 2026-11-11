# Phase 6 · Week 11 · Day 2 — Structured Logging and Correlation IDs

> Objective: Build production-grade structured logging with JSON formatting, log context propagation, and correlation ID threading across async call stacks.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

### Why human-readable logs fail at scale

Plain-text logs are optimised for a single developer reading a terminal. At production scale — thousands of requests per second, dozens of services, multiple replicas — you need to:

- **Filter** logs by field value (`status_code == 500`)
- **Aggregate** across fields (`COUNT(*) GROUP BY service, endpoint`)
- **Correlate** a log line with a trace span or metric data point

None of these are possible when fields are embedded in a free-form string. Parsing regexes are brittle and expensive. JSON logs make every field a first-class citizen queryable by any log platform (Loki, OpenSearch, CloudWatch Logs Insights, Splunk).

### JSON log format

A well-structured JSON log line contains:

```json
{
  "timestamp": "2026-11-11T09:00:00.123Z",
  "level": "INFO",
  "logger": "myapp.orders",
  "message": "order_placed",
  "correlation_id": "f47ac10b-...",
  "user_id": 42,
  "order_id": "ORD-9981",
  "duration_ms": 12.4
}
```

Key conventions:
- `timestamp`: ISO 8601 with millisecond precision, UTC.
- `level`: uppercase string, not an integer.
- `message` / `event`: short snake_case identifier, not a sentence. Sentences go in a separate `detail` field if needed.
- All business identifiers (`user_id`, `order_id`, etc.) are top-level fields, not embedded in the message string.

### Log levels — semantics

| Level | Meaning | Alert? |
|-------|---------|--------|
| `DEBUG` | Developer trace, never in production | No |
| `INFO` | Normal operational events | No |
| `WARNING` | Unexpected but recoverable; SLA at risk | Sometimes |
| `ERROR` | Operation failed; user impact likely | Yes |
| `CRITICAL` | System stability at risk | Always |

Rule: never use `print()`. Every diagnostic output must go through `logging.getLogger(...)` so it carries level, logger name, and can be routed, sampled, or silenced centrally.

### Correlation IDs in a distributed system

A correlation ID (also called a request ID) is a UUID that:

1. Is generated at the edge (API gateway, load balancer, or the first service the request hits).
2. Is propagated in HTTP headers (`X-Correlation-ID`, `X-Request-ID`, or the W3C `traceparent` header).
3. Is attached to every log line emitted during that request's lifetime.
4. Is stored in the response headers so clients can reference it in support tickets.

Without correlation IDs, when Service A calls Service B which calls Service C, the three services produce log lines with no common field. You cannot reconstruct the full request timeline.

---

## How it works internally

### `contextvars.ContextVar`

Python's `contextvars` module (PEP 567, Python 3.7+) provides variables whose values are local to an execution context. In `asyncio`, each `Task` gets its own copy of the context, initialised from the parent task's context at the time the task is created.

```
Request 1 Task:  correlation_id_var = "aaa-..."
Request 2 Task:  correlation_id_var = "bbb-..."
```

Both tasks run in the same event loop thread, but their `ContextVar` values never bleed across. This is precisely what middleware needs — thread-local semantics without threads.

`ContextVar.set(value)` returns a `Token`. `ContextVar.reset(token)` restores the previous value. Always call `reset` in a `finally` block to ensure cleanup, especially when the same task might be reused.

### Log record injection

Python's `logging.Filter` interface provides a hook to modify `LogRecord` objects before they are formatted. A filter added to a handler or logger can inject the current correlation ID from the `ContextVar` into every record automatically — no caller has to remember to pass `extra={"correlation_id": ...}`.

```python
class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("")
        return True  # True = keep the record
```

This is preferable to injecting the ID in every `logger.info(...)` call — it is automatic and cannot be forgotten.

### Log sampling and rate limiting

High-volume endpoints (health checks, metrics scrapes) can flood a log aggregation system. Two strategies:

1. **Probabilistic sampling**: emit a log with probability `p`. Simple, stateless.
2. **Rate limiting**: emit at most `N` logs per second for a given logger or key. Requires a token bucket or sliding window counter.

For structured logging at scale, sampling decisions should be stored in the log record itself (`sampled: true`) so downstream processors know the log was sampled and can extrapolate counts.

---

## Worked example

Install dependencies:

```bash
uv add fastapi uvicorn[standard]
```

```python
# structured_logging/logging_setup.py
"""
Structured logging configuration for a FastAPI service.

Provides:
- JSONLogFormatter: emits every LogRecord as a JSON line
- CorrelationFilter: auto-injects correlation_id from ContextVar into every record
- configure_logging(): one-call setup for application startup
"""

import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Any

# ---------------------------------------------------------------------------
# Correlation ID context variable
# ---------------------------------------------------------------------------
# Exported so middleware and tests can read/write it.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


# ---------------------------------------------------------------------------
# Filter: auto-inject correlation ID into every log record
# ---------------------------------------------------------------------------
class CorrelationFilter(logging.Filter):
    """Inject the active correlation_id from ContextVar into each LogRecord.

    Add this filter to a handler (not a logger) so it applies to all
    loggers that route through that handler.

    Example::

        handler = logging.StreamHandler()
        handler.addFilter(CorrelationFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.correlation_id = correlation_id_var.get("")  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Formatter: emit JSON
# ---------------------------------------------------------------------------
class JSONLogFormatter(logging.Formatter):
    """Convert a LogRecord to a single-line JSON string.

    All ``extra`` fields passed to the logger are promoted to top-level
    JSON keys. Internal Python logging bookkeeping fields are excluded.
    """

    _SKIP: frozenset[str] = frozenset(
        {
            "args",
            "exc_info",
            "exc_text",
            "msg",
            "message",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
            "processName",
            "process",
            "msecs",
            "relativeCreated",
            "filename",
            "module",
            "funcName",
            "created",
            "levelno",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Ensure getMessage() is called so %s-style formatting is resolved.
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": self._utc_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "pathname": f"{record.pathname}:{record.lineno}",
        }

        # Correlation ID injected by CorrelationFilter (may be absent in tests).
        cid: str = getattr(record, "correlation_id", "")
        if cid:
            payload["correlation_id"] = cid

        # Merge extra fields, avoiding clobber of core keys.
        for key, value in record.__dict__.items():
            if key not in self._SKIP and not key.startswith("_") and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)

    @staticmethod
    def _utc_iso(created: float) -> str:
        """Return ISO 8601 UTC timestamp with millisecond precision."""
        ms = int((created % 1) * 1000)
        base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(created))
        return f"{base}.{ms:03d}Z"


# ---------------------------------------------------------------------------
# One-call setup
# ---------------------------------------------------------------------------
def configure_logging(level: int | None = None) -> None:
    """Configure the root logger with JSON output and correlation ID injection.

    Reads ``LOG_LEVEL`` from the environment (default ``INFO``).
    Call once at application startup before creating any logger instances.
    """
    env_level = os.getenv("LOG_LEVEL", "INFO").upper()
    resolved_level = getattr(logging, env_level, logging.INFO) if level is None else level

    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())
    handler.addFilter(CorrelationFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
```

```python
# structured_logging/middleware.py
"""
LoggingMiddleware: ASGI middleware for FastAPI.

Responsibilities:
1. Read or generate a correlation ID for each request.
2. Store it in correlation_id_var so all log lines in this request carry it.
3. Forward it as a response header.
4. Log request start and end with structured fields.
5. Propagate correlation ID to outgoing HTTP calls via httpx event hooks (optional).
"""

import logging
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request

from .logging_setup import correlation_id_var

logger = logging.getLogger(__name__)

# Header names — centralised so they are changed in one place.
CORRELATION_HEADER = "X-Correlation-ID"


class LoggingMiddleware:
    """ASGI middleware that instruments every HTTP request with structured logging.

    Usage::

        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

    Note: ``add_middleware`` wraps the ASGI app; the class receives ``app`` as
    its first argument automatically.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(
        self,
        scope: dict,
        receive: Callable,
        send: Callable,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # ------------------------------------------------------------------
        # 1. Resolve correlation ID
        # ------------------------------------------------------------------
        incoming = request.headers.get(CORRELATION_HEADER)
        cid: str = incoming if incoming else str(uuid.uuid4())

        # Store in ContextVar — isolated to this asyncio Task.
        token = correlation_id_var.set(cid)

        # ------------------------------------------------------------------
        # 2. Log request start
        # ------------------------------------------------------------------
        logger.info(
            "http_request_start",
            extra={
                "event": "http_request_start",
                "http_method": request.method,
                "http_path": request.url.path,
                "http_query": str(request.url.query) or None,
                "user_agent": request.headers.get("user-agent"),
                "remote_addr": request.client.host if request.client else None,
            },
        )

        start_ns = time.perf_counter_ns()
        status_holder: list[int] = []

        # ------------------------------------------------------------------
        # 3. Intercept send to capture status and inject response header
        # ------------------------------------------------------------------
        async def send_intercepted(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder.append(message["status"])
                headers = list(message.get("headers", []))
                headers.append((CORRELATION_HEADER.lower().encode(), cid.encode()))
                message = {**message, "headers": headers}
            await send(message)

        # ------------------------------------------------------------------
        # 4. Run the downstream application
        # ------------------------------------------------------------------
        try:
            await self.app(scope, receive, send_intercepted)
        except Exception:
            logger.exception(
                "http_request_unhandled_exception",
                extra={
                    "event": "http_request_unhandled_exception",
                    "http_method": request.method,
                    "http_path": request.url.path,
                },
            )
            raise
        finally:
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            status = status_holder[0] if status_holder else 0

            log_fn = logger.warning if status >= 400 else logger.info
            log_fn(
                "http_request_end",
                extra={
                    "event": "http_request_end",
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status_code": status,
                    "duration_ms": round(duration_ms, 3),
                },
            )

            # Always restore context, even on exception.
            correlation_id_var.reset(token)
```

```python
# structured_logging/app.py
"""
FastAPI application demonstrating log propagation across function calls.

Key point: every function in the call stack uses the same logger (or a child
logger).  Because CorrelationFilter is attached to the root handler, the
correlation_id flows into every log record regardless of which function emits
it — no argument threading required.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .logging_setup import configure_logging
from .middleware import LoggingMiddleware

configure_logging()

app = FastAPI(title="Structured Logging Demo")
app.add_middleware(LoggingMiddleware)

logger = logging.getLogger(__name__)

# Child loggers — messages propagate to the root handler automatically.
db_logger = logging.getLogger(f"{__name__}.db")
cache_logger = logging.getLogger(f"{__name__}.cache")


# ---------------------------------------------------------------------------
# Simulated internal functions — each emits its own log, each carries the
# same correlation_id without receiving it as an argument.
# ---------------------------------------------------------------------------

def _check_cache(item_id: int) -> dict | None:
    """Return cached item or None.  Logs its own trace line."""
    cache_logger.debug(
        "cache_lookup",
        extra={"event": "cache_lookup", "item_id": item_id},
    )
    # Simulate a cache miss for odd IDs.
    if item_id % 2 == 1:
        cache_logger.debug(
            "cache_miss",
            extra={"event": "cache_miss", "item_id": item_id},
        )
        return None
    cache_logger.debug(
        "cache_hit",
        extra={"event": "cache_hit", "item_id": item_id},
    )
    return {"item_id": item_id, "name": f"Widget {item_id}", "source": "cache"}


def _fetch_from_db(item_id: int) -> dict:
    """Simulate a database fetch.  Logs its own trace line."""
    db_logger.info(
        "db_query",
        extra={"event": "db_query", "item_id": item_id, "table": "items"},
    )
    return {"item_id": item_id, "name": f"Widget {item_id}", "source": "db"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/items/{item_id}")
async def get_item(item_id: int) -> JSONResponse:
    """Fetch an item, checking cache first.

    Notice: ``item_id`` and ``correlation_id`` both appear on every log line
    emitted in this request — including those from _check_cache and
    _fetch_from_db — without any explicit argument passing.
    """
    logger.info(
        "get_item_start",
        extra={"event": "get_item_start", "item_id": item_id},
    )

    result = _check_cache(item_id)
    if result is None:
        result = _fetch_from_db(item_id)

    logger.info(
        "get_item_end",
        extra={"event": "get_item_end", "item_id": item_id, "source": result["source"]},
    )
    return JSONResponse(result)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("structured_logging.app:app", host="0.0.0.0", port=8000, reload=True)
```

**Observing correlation ID propagation** — run the app, then:

```bash
curl -H "X-Correlation-ID: my-test-id-123" http://localhost:8000/items/3
```

Every log line emitted — from the middleware, from `get_item`, from `_check_cache`, from `_fetch_from_db` — will carry `"correlation_id": "my-test-id-123"`. No function received it as an argument.

**Log sampling example** (simple probabilistic):

```python
# structured_logging/sampling.py
"""Probabilistic log sampler using a logging.Filter."""

import logging
import random


class SamplingFilter(logging.Filter):
    """Drop log records probabilistically.

    Args:
        rate: Fraction of records to keep, in [0.0, 1.0].
              rate=1.0 keeps everything; rate=0.1 keeps 10 %.
    """

    def __init__(self, rate: float = 1.0, name: str = "") -> None:
        super().__init__(name)
        if not 0.0 <= rate <= 1.0:
            msg = f"rate must be in [0.0, 1.0], got {rate}"
            raise ValueError(msg)
        self.rate = rate

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return random.random() < self.rate
```

Attach to a specific logger to sample health-check noise:

```python
health_logger = logging.getLogger("structured_logging.app.health")
health_logger.addFilter(SamplingFilter(rate=0.1))
```

---

## Exercises

1. **Request body logging**: Extend `LoggingMiddleware` to log the request body for `POST` and `PUT` requests. The body must be read with `await request.body()`. Handle the case where the body is not valid UTF-8 by logging its byte length instead. Ensure the body is not logged when it exceeds 4 KB (truncate to the first 200 bytes and add `truncated: true` to the log record).

2. **Rate-limited logger**: Implement a `RateLimitedFilter` that allows at most `max_per_second` records per second for a given logger name. Use a sliding window with `time.monotonic()`. Write a unit test that calls the filter 100 times in a tight loop and asserts that fewer than `max_per_second + 1` records pass through.

3. **Child logger hierarchy**: Refactor `app.py` so that `_check_cache` and `_fetch_from_db` live in separate modules (`cache.py`, `db.py`). Use `logging.getLogger(__name__)` in each module. Verify that the `CorrelationFilter` (attached to the root handler) still injects the correlation ID into records emitted by both child loggers — explain in a comment why this works without attaching the filter to each child logger individually.

4. **Log sanitisation**: Add a `SanitisationFilter` that redacts sensitive fields from log records before they are emitted. Define a set of field names to redact: `{"password", "token", "api_key", "secret"}`. For any matching key in the record's `__dict__`, replace the value with `"***REDACTED***"`. Write a test that passes a record with a `password` field and asserts the formatter output does not contain the original value.

(Solutions: ../solutions/week-11/day-2.md)

---

## Self-check

- [ ] I can explain why JSON-structured logs are preferable to plain-text logs for production systems and give two concrete operations that are impossible (or prohibitively expensive) on plain-text logs.
- [ ] I can implement a `logging.Filter` subclass that injects a `ContextVar` value into every log record, and explain why attaching it to the handler rather than individual loggers is the correct approach.
- [ ] I can describe how `contextvars.ContextVar` provides per-request isolation in an `asyncio`-based server and contrast it with `threading.local`.
- [ ] I can implement probabilistic log sampling and explain the trade-off between sampling rate, log volume, and diagnostic fidelity.

---

## Supplementary reading

- [Python `contextvars` — PEP 567](https://peps.python.org/pep-0567/)
- [Python `logging` — Filter Objects](https://docs.python.org/3.12/library/logging.html#filter-objects)
- [OpenTelemetry — Correlation](https://opentelemetry.io/docs/concepts/context-propagation/)
- [W3C Trace Context — `traceparent` header](https://www.w3.org/TR/trace-context/)
- [Grafana Loki — Log queries](https://grafana.com/docs/loki/latest/query/)
