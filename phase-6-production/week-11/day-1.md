# Phase 6 · Week 11 · Day 1 — The Three Pillars of Observability: Logs, Metrics, Traces

> Objective: Understand the three pillars of observability and implement structured logging with correlation IDs in a FastAPI service.  ·  Time: ~1 hr  ·  Prereqs: [[phase-6-week-10-day-5]]

---

## Concept

**Monitoring** answers: "Is the system working?" You define thresholds in advance and alert when crossed. It's reactive — you know what to look for.

**Observability** answers: "Why is the system behaving this way?" You can ask arbitrary questions about internal state from external outputs alone. It handles *unknown unknowns* — failure modes you never anticipated.

The three pillars each capture a different dimension of system behaviour:

| Pillar | What it captures | Granularity | Storage |
|--------|-----------------|-------------|---------|
| Logs | Discrete events with context | Per-event | Append-only (Loki, CloudWatch, ELK) |
| Metrics | Numeric aggregations over time | Per-interval | Time-series (Prometheus, InfluxDB) |
| Traces | Request flow across service boundaries | Per-request | Span store (Jaeger, Tempo, Zipkin) |

### Logs

A log is a timestamped record of a discrete event. Two formats exist:

- **Unstructured**: `2026-01-01 12:00:00 INFO User 42 logged in` — human-readable, machine-hostile.
- **Structured**: `{"timestamp": "2026-01-01T12:00:00Z", "level": "INFO", "event": "user_login", "user_id": 42}` — machine-queryable, filterable, aggregatable.

At scale, unstructured logs are a liability. You cannot GROUP BY or filter on fields that aren't fields.

### Metrics

Metrics are numeric measurements sampled over time: `http_requests_total{method="POST", status="200"} 1042`. They are cheap to store (one number per interval per label set) and fast to query. They answer "how many" and "how long" questions at aggregate level.

### Traces

A trace is a directed acyclic graph of **spans**. A span is a unit of work with a start time, end time, and metadata. Spans are linked by a shared `trace_id`. Traces answer: "For this specific request, what happened, in what order, with what latency, in which service?"

### The Unknown Unknowns Problem

- Metrics alone: you know requests are slow, but not which user, path, or downstream call caused it.
- Logs alone: you find the error event, but cannot correlate it to the upstream call that triggered it.
- Traces alone: you see the span tree, but without logs on each span you lack event context.

All three together: a metric alert fires → you drill into traces for that time window → you open logs for the failing span. This is the *observability loop*.

---

## How it works internally

### Structured logging in Python

Python's `logging` module emits `LogRecord` objects. A `Formatter` converts a `LogRecord` to a string. To emit JSON, write a custom `Formatter` that serialises `record.__dict__` (plus any extra fields) to JSON.

Key fields on a `LogRecord`:
- `name`, `levelname`, `message`, `pathname`, `lineno`, `created` (Unix timestamp)
- `extra` dict: any keyword arguments passed via `logging.getLogger().info("msg", extra={...})`

### Correlation IDs

A correlation ID (also called a request ID or trace ID) is a UUID generated at the entry point of a request. It flows through every log line, every downstream call, and every span. It is the thread you pull to reconstruct a request's full story across services.

In HTTP systems, it travels in a header: `X-Correlation-ID` or `X-Request-ID`. Middleware reads it on ingress (or generates one if absent), stores it in thread-local or async context, and injects it into every log record for the duration of that request.

In Python async code, `contextvars.ContextVar` provides per-task isolation — the correct primitive for FastAPI's async request handling.

---

## Worked example

Install dependencies:

```bash
uv add fastapi uvicorn[standard] httpx
```

```python
# observability_demo/main.py
"""
Day 1 worked example: structured logging + correlation IDs in FastAPI.

Run with:
    uvicorn observability_demo.main:app --reload
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Context variable — stores the correlation ID for the current async task.
# ContextVar is copy-on-write per asyncio Task, so concurrent requests never
# share state even in the same process.
# ---------------------------------------------------------------------------
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


# ---------------------------------------------------------------------------
# Custom JSON log formatter
# ---------------------------------------------------------------------------
class JSONLogFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object.

    Extra fields injected via `logging.getLogger().info("msg", extra={...})`
    are merged into the top-level JSON object, making them first-class
    queryable fields in any log aggregation platform.
    """

    # Fields that live on LogRecord but are internal / noisy to expose.
    _SKIP_FIELDS: frozenset[str] = frozenset(
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
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Build the base payload.
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pathname": record.pathname,
            "lineno": record.lineno,
        }

        # Attach the active correlation ID if one exists.
        cid = correlation_id_var.get("")
        if cid:
            payload["correlation_id"] = cid

        # Merge any `extra={}` fields the caller injected.
        for key, value in record.__dict__.items():
            if key not in self._SKIP_FIELDS and not key.startswith("_"):
                if key not in payload:
                    payload[key] = value

        # Attach exception info when present.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logger setup — apply the JSON formatter to the root handler.
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> None:
    """Replace the root logger's handlers with a JSON-emitting stream handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI middleware: injects correlation ID + request/response logging
# ---------------------------------------------------------------------------
class CorrelationMiddleware:
    """ASGI middleware that:

    1. Reads ``X-Correlation-ID`` from the incoming request header, or
       generates a fresh UUID4 if absent.
    2. Stores it in ``correlation_id_var`` so every log line emitted during
       this request automatically carries it.
    3. Adds it to the response as ``X-Correlation-ID``.
    4. Logs request start and end with method, path, status code, and
       wall-clock duration in milliseconds.
    """

    def __init__(self, app: Callable) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            # Pass through websocket / lifespan events unchanged.
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)

        # Resolve correlation ID — honour upstream header or mint a new one.
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        token = correlation_id_var.set(cid)

        method = request.method
        path = request.url.path
        start_ns = time.perf_counter_ns()

        logger.info(
            "request_start",
            extra={
                "event": "request_start",
                "http_method": method,
                "http_path": path,
            },
        )

        # Capture the status code from the response before it is sent.
        status_code_holder: list[int] = []

        async def send_with_cid(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_code_holder.append(message["status"])
                # Inject the correlation ID header into the response.
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", cid.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_cid)
        finally:
            duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            status = status_code_holder[0] if status_code_holder else 0

            logger.info(
                "request_end",
                extra={
                    "event": "request_end",
                    "http_method": method,
                    "http_path": path,
                    "http_status_code": status,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            # Restore the previous context value (good practice for middleware
            # that may be composed in ways that share the same task).
            correlation_id_var.reset(token)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="Observability Day 1 Demo")
app.middleware("http")(CorrelationMiddleware(app))  # type: ignore[arg-type]


@app.get("/health")
async def health() -> JSONResponse:
    logger.info("health check called", extra={"event": "health_check"})
    return JSONResponse({"status": "ok"})


@app.get("/items/{item_id}")
async def get_item(item_id: int) -> JSONResponse:
    logger.info(
        "fetching item",
        extra={"event": "item_fetch", "item_id": item_id},
    )
    # Simulate a downstream lookup.
    if item_id < 0:
        logger.warning(
            "invalid item_id",
            extra={"event": "item_invalid", "item_id": item_id},
        )
        return JSONResponse({"error": "item_id must be non-negative"}, status_code=400)

    return JSONResponse({"item_id": item_id, "name": f"Widget {item_id}"})


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("observability_demo.main:app", host="0.0.0.0", port=8000, reload=True)
```

**Sample output** (one log line, pretty-printed here for readability):

```json
{
  "timestamp": "2026-11-11T09:00:00",
  "level": "INFO",
  "logger": "observability_demo.main",
  "message": "request_end",
  "pathname": "/observability_demo/main.py",
  "lineno": 112,
  "correlation_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "event": "request_end",
  "http_method": "GET",
  "http_path": "/items/7",
  "http_status_code": 200,
  "duration_ms": 1.23
}
```

Every line for a single request shares the same `correlation_id`. In a log aggregation tool (Grafana Loki, Kibana, CloudWatch Insights), you filter on `correlation_id = "f47ac10b-..."` to see the full story of that one request across all services.

---

## Exercises

1. **Log sampling**: The middleware currently logs every request. Add a mechanism to log only 10 % of `/health` requests (health checks are noisy). Implement a `should_sample(path: str, sample_rate: float) -> bool` helper and gate the `request_start` log on it for the health endpoint.

2. **Downstream propagation**: Create a second FastAPI route `/call-upstream` that uses `httpx.AsyncClient` to call `http://localhost:8001/health`. Modify the `CorrelationMiddleware` (or the route handler) to forward the `X-Correlation-ID` header to the upstream call, so the correlation ID propagates across service boundaries.

3. **Log levels from environment**: Modify `configure_logging()` to read the desired level from an environment variable `LOG_LEVEL` (e.g., `DEBUG`, `INFO`, `WARNING`) via `os.getenv`. Fall back to `INFO` if the variable is absent or invalid. Add a `logging.DEBUG` line inside `get_item` that logs the full request URL, and verify it appears only when `LOG_LEVEL=DEBUG`.

4. **Error correlation**: Add an exception handler to the FastAPI app that catches unhandled exceptions, logs them at `ERROR` level with the `correlation_id` and a `traceback` field, and returns a `500` response. Test it by raising a `RuntimeError` in a route.

(Solutions: ../solutions/week-11/day-1.md)

---

## Self-check

- [ ] I can explain the difference between observability and monitoring, and give a concrete example where monitoring alone is insufficient.
- [ ] I can describe what information each of the three pillars captures and name one storage system appropriate for each.
- [ ] I can implement a `JSONLogFormatter` and explain why structured logs are preferable to plain-text logs at scale.
- [ ] I can explain how `contextvars.ContextVar` provides per-request isolation in an async FastAPI application and why it is preferable to a global variable.

---

## Supplementary reading

- [OpenTelemetry — What is Observability?](https://opentelemetry.io/docs/concepts/observability-primer/)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
- [Prometheus — Introduction](https://prometheus.io/docs/introduction/overview/)
- [Google SRE Book — Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Python `logging` HOWTO](https://docs.python.org/3.12/howto/logging.html)
