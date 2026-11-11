# Phase 6 · Week 11 · Day 3 — Metrics with Prometheus: Counters, Gauges, Histograms

> Objective: Instrument a FastAPI service with Prometheus metrics using the RED and USE methods, and expose a `/metrics` endpoint consumable by Prometheus.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

### The Prometheus data model

Prometheus stores **time series** — streams of timestamped float64 values. Each time series is identified by a **metric name** and a set of **labels** (key-value pairs):

```
http_requests_total{method="GET", endpoint="/items", status="200"} 1042
http_requests_total{method="POST", endpoint="/orders", status="500"} 3
```

Labels turn one metric into a multi-dimensional dataset. You can aggregate across all label values (`sum(http_requests_total)`) or filter to a subset (`http_requests_total{status=~"5.."}` — all 5xx errors).

Label cardinality matters: if you use `user_id` as a label on a high-traffic metric, Prometheus stores one time series per unique user ID. At millions of users this becomes a cardinality explosion. Use labels for low-cardinality dimensions only (method, endpoint, status code, region).

### Metric types

#### Counter

A **counter** is a monotonically increasing float that resets to zero on process restart. It only ever goes up (barring restarts). Use it to count events.

```
http_requests_total         # total requests since startup
errors_total                # total errors since startup
bytes_sent_total            # total bytes sent since startup
```

To compute a rate from a counter, use `rate()` in PromQL:

```promql
rate(http_requests_total[5m])   # requests per second, 5-minute window
```

Never represent something that can go down (like active connections) as a counter.

#### Gauge

A **gauge** is a float that can go up or down. It represents a current snapshot — a value at a point in time.

```
active_requests             # in-flight HTTP requests right now
queue_depth                 # items waiting in a job queue
memory_bytes_used           # current heap usage
```

Gauges are not rate-d — they are sampled directly. Use `avg_over_time` or `max_over_time` to smooth them.

#### Histogram

A **histogram** samples observations and counts them into configurable **buckets**. It also maintains a running sum and count. Histograms are the correct instrument for durations and sizes.

```
http_request_duration_seconds_bucket{le="0.1"}   # requests <= 100ms
http_request_duration_seconds_bucket{le="0.5"}   # requests <= 500ms
http_request_duration_seconds_bucket{le="1.0"}   # requests <= 1s
http_request_duration_seconds_bucket{le="+Inf"}  # all requests
http_request_duration_seconds_sum                # total duration
http_request_duration_seconds_count              # total observations
```

From histograms you compute percentiles in PromQL:

```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

This gives you the p95 latency — the request duration below which 95 % of requests fall in the last 5 minutes.

#### Summary (not used here)

Summaries pre-compute quantiles client-side. They cannot be aggregated across replicas, making them inferior to histograms for distributed services. Prefer histograms.

---

### The RED method (services)

Proposed by Tom Wilkie (Grafana). For every **service** (HTTP API, gRPC endpoint, message consumer), instrument three things:

| Signal | Meaning | Metric example |
|--------|---------|---------------|
| **R**ate | Requests per second | `rate(http_requests_total[5m])` |
| **E**rrors | Error rate or ratio | `rate(http_requests_total{status=~"5.."}[5m])` |
| **D**uration | Latency distribution | `histogram_quantile(0.99, ...)` |

RED gives you a dashboard that answers: "Is this service healthy right now?"

### The USE method (resources)

Proposed by Brendan Gregg. For every **resource** (CPU, memory, disk, network interface, connection pool):

| Signal | Meaning | Metric example |
|--------|---------|---------------|
| **U**tilisation | Fraction of time resource is busy | `rate(cpu_seconds_total[5m])` |
| **S**aturation | Amount of queued/waiting work | `queue_depth` |
| **E**rrors | Error events | `disk_errors_total` |

USE answers: "Is any resource a bottleneck?"

---

## How it works internally

### `prometheus_client` Python library

The `prometheus_client` package maintains a **global registry** of metrics. Each metric object (`Counter`, `Gauge`, `Histogram`) registers itself on first instantiation. The `/metrics` handler calls `generate_latest(REGISTRY)`, which iterates all registered metrics and serialises them into the [Prometheus text exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/).

```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",endpoint="/health",status="200"} 42.0
```

Labels are applied at **observation time** via `.labels(method="GET", ...)`, which returns a child metric. Children are cached internally — calling `.labels(method="GET")` twice returns the same object.

Histograms in `prometheus_client` require bucket boundaries at definition time:

```python
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
```

Default buckets cover 5 ms to 10 s — suitable for web services. Tune for your SLOs.

### ASGI middleware integration

The pattern is identical to Day 1 and Day 2: ASGI middleware wraps the app and intercepts request start (increment active gauge, record start time) and response end (decrement active gauge, observe histogram, increment request counter with labels).

The Gauge's `track_inprogress()` context manager handles increment/decrement automatically for synchronous code; for ASGI we manage it manually in `try/finally`.

---

## Worked example

Install dependencies:

```bash
uv add fastapi uvicorn[standard] prometheus-client
```

> `prometheus-client` is the official Python client for Prometheus.
> Add it to your `pyproject.toml` under `[project.dependencies]`.

```python
# metrics_demo/metrics.py
"""
Prometheus metric definitions.

All metrics are defined here and imported by the middleware and routes.
Define metrics at module level — creating them inside functions would
re-register them on each call and raise a ValueError from prometheus_client.
"""

from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# RED method metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    name="http_requests_total",
    documentation="Total number of HTTP requests completed.",
    labelnames=["method", "endpoint", "status"],
)
"""Counter — incremented once per completed request.

Labels:
    method:   HTTP verb (GET, POST, …)
    endpoint: normalised route path (/items/{item_id} → /items/{item_id})
    status:   HTTP status code as a string ("200", "404", "500")
"""

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    name="http_request_duration_seconds",
    documentation="HTTP request duration in seconds.",
    labelnames=["method", "endpoint"],
    # Buckets tuned for a web API with p99 SLO of 1 s.
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
"""Histogram — one observation per completed request.

Labels exclude status code to avoid high cardinality on error spikes.
"""

# ---------------------------------------------------------------------------
# Gauge — current state
# ---------------------------------------------------------------------------

ACTIVE_REQUESTS = Gauge(
    name="active_requests",
    documentation="Number of HTTP requests currently being processed.",
    labelnames=["method"],
)
"""Gauge — incremented at request start, decremented at request end.

Tracks in-flight requests per HTTP method.
"""
```

```python
# metrics_demo/middleware.py
"""
Prometheus instrumentation middleware for FastAPI.

Responsibilities:
1. Increment ACTIVE_REQUESTS at request start; decrement in finally.
2. Record HTTP_REQUEST_DURATION_SECONDS on completion.
3. Increment HTTP_REQUESTS_TOTAL with method, endpoint, status labels.
"""

import logging
import time
from collections.abc import Callable

from fastapi import Request

from .metrics import ACTIVE_REQUESTS, HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

logger = logging.getLogger(__name__)


def _normalise_path(path: str, route_path: str | None) -> str:
    """Return the matched route pattern instead of the concrete path.

    Using the concrete path (/items/42, /items/99) as a label would
    create one time series per unique item ID — a cardinality explosion.
    The route pattern (/items/{item_id}) collapses all item requests
    into a single time series.
    """
    return route_path if route_path else path


class PrometheusMiddleware:
    """ASGI middleware that records Prometheus RED metrics for every request."""

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
        method = request.method

        # The matched route template is available on scope["route"] after
        # routing completes; we fall back to the raw path for unmatched routes.
        # We resolve the endpoint label after the response to get the matched route.
        start = time.perf_counter()
        ACTIVE_REQUESTS.labels(method=method).inc()

        status_holder: list[int] = []

        async def send_intercepted(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder.append(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_intercepted)
        finally:
            duration = time.perf_counter() - start
            ACTIVE_REQUESTS.labels(method=method).dec()

            status = str(status_holder[0]) if status_holder else "0"

            # Resolve the matched route template from the ASGI scope.
            # FastAPI populates scope["route"] after routing.
            route = scope.get("route")
            endpoint = _normalise_path(
                request.url.path,
                route.path if route is not None else None,
            )

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status=status,
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

            logger.debug(
                "metrics_recorded",
                extra={
                    "event": "metrics_recorded",
                    "http_method": method,
                    "endpoint": endpoint,
                    "http_status_code": status,
                    "duration_s": round(duration, 4),
                },
            )
```

```python
# metrics_demo/app.py
"""
FastAPI application with Prometheus metrics.

Endpoints:
    GET  /health               — liveness probe
    GET  /items/{item_id}      — example business route
    GET  /metrics              — Prometheus text-format scrape endpoint

Run with:
    uvicorn metrics_demo.app:app --reload
Then scrape metrics:
    curl http://localhost:8000/metrics
"""

import logging
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

from .logging_setup import configure_logging
from .middleware import PrometheusMiddleware

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Metrics Demo")
app.add_middleware(PrometheusMiddleware)


# ---------------------------------------------------------------------------
# /metrics — Prometheus scrape endpoint
# ---------------------------------------------------------------------------

@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> PlainTextResponse:
    """Expose all registered Prometheus metrics in text exposition format.

    Prometheus scrapes this endpoint at a configurable interval (default 15 s).
    The response content type must match CONTENT_TYPE_LATEST for Prometheus to
    parse it correctly.
    """
    data = generate_latest(REGISTRY)
    return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Business routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> JSONResponse:
    logger.info("health_check", extra={"event": "health_check"})
    return JSONResponse({"status": "ok"})


@app.get("/items/{item_id}")
async def get_item(item_id: int) -> JSONResponse:
    logger.info(
        "item_fetch",
        extra={"event": "item_fetch", "item_id": item_id},
    )
    if item_id <= 0:
        return JSONResponse(
            {"error": "item_id must be positive"},
            status_code=422,
        )
    return JSONResponse({"item_id": item_id, "name": f"Widget {item_id}"})


@app.post("/items")
async def create_item(payload: dict) -> JSONResponse:
    name = payload.get("name", "Unnamed")
    logger.info(
        "item_create",
        extra={"event": "item_create", "name": name},
    )
    return JSONResponse({"item_id": 999, "name": name}, status_code=201)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("metrics_demo.app:app", host="0.0.0.0", port=8000, reload=True)
```

Note: you need a `logging_setup.py` in `metrics_demo/` (same as Day 2, or a simplified version):

```python
# metrics_demo/logging_setup.py
"""Minimal logging setup reused from Day 2."""

import logging
import os


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
```

**Verify the metrics endpoint:**

```bash
# Start the app
uvicorn metrics_demo.app:app --reload &

# Send some traffic
curl http://localhost:8000/items/1
curl http://localhost:8000/items/2
curl -X POST http://localhost:8000/items -H "Content-Type: application/json" -d '{"name":"Gadget"}'
curl http://localhost:8000/items/-1   # triggers 422

# Scrape metrics
curl -s http://localhost:8000/metrics | grep http_requests
```

**Expected output excerpt:**

```
# HELP http_requests_total Total number of HTTP requests completed.
# TYPE http_requests_total counter
http_requests_total{endpoint="/items/{item_id}",method="GET",status="200"} 2.0
http_requests_total{endpoint="/items/{item_id}",method="GET",status="422"} 1.0
http_requests_total{endpoint="/items",method="POST",status="201"} 1.0

# HELP http_request_duration_seconds HTTP request duration in seconds.
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{endpoint="/items/{item_id}",le="0.005",method="GET"} 3.0
http_request_duration_seconds_bucket{endpoint="/items/{item_id}",le="+Inf",method="GET"} 3.0
http_request_duration_seconds_count{endpoint="/items/{item_id}",method="GET"} 3.0
http_request_duration_seconds_sum{endpoint="/items/{item_id}",method="GET"} 0.0012...

# HELP active_requests Number of HTTP requests currently being processed.
# TYPE active_requests gauge
active_requests{method="GET"} 0.0
```

**PromQL queries for a RED dashboard:**

```promql
# Request rate (req/s, 5-minute window)
sum by (endpoint) (rate(http_requests_total[5m]))

# Error ratio (5xx errors as a fraction of total)
sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))

# p95 latency by endpoint
histogram_quantile(0.95,
  sum by (endpoint, le) (
    rate(http_request_duration_seconds_bucket[5m])
  )
)
```

---

## Exercises

1. **Token count histogram**: LLM APIs return token counts per call. Define a `Histogram` named `llm_tokens_total` with label `model` and buckets appropriate for token counts: `[100, 500, 1000, 2000, 4000, 8000, 16000, 32000]`. Add a route `POST /llm/complete` that accepts `{"prompt": str, "model": str}` and calls the Anthropic API (`claude-haiku-5-5` by default) via `os.getenv("ANTHROPIC_API_KEY")`. Observe the response's `usage.input_tokens + usage.output_tokens` into the histogram. Verify the metric appears in `/metrics` after a call.

2. **Error rate alert threshold**: Add a second counter `llm_errors_total` with labels `model` and `error_type` (e.g., `"rate_limit"`, `"timeout"`, `"invalid_request"`). Wrap the LLM call in a `try/except` and increment this counter on failure. Write a PromQL expression that would fire an alert when the error rate exceeds 5 % over a 1-minute window. (You do not need to configure Alertmanager — write the expression as a comment in the code.)

3. **USE method for a connection pool**: Simulate a database connection pool with a `Semaphore` of size 10. Define three metrics representing the USE method: a `Gauge` for utilisation (connections in use / pool size), a `Gauge` for saturation (requests waiting for a connection), and a `Counter` for errors (acquisition timeouts). Add a route `/db-query` that acquires a semaphore slot, sleeps for a random 10–200 ms, then releases it. Use `asyncio.wait_for` with a timeout to simulate acquisition failures.

4. **Histogram bucket tuning**: The default `prometheus_client` histogram buckets go up to 10 s. Your LLM API calls typically complete in 1–30 seconds. Define a `Histogram` for `llm_response_duration_seconds` with buckets appropriate for this range. Explain in a comment why using the default buckets for this metric would produce misleading p99 estimates, and how your chosen buckets address this.

(Solutions: ../solutions/week-11/day-3.md)

---

## Self-check

- [ ] I can explain the difference between a Counter, Gauge, and Histogram, and give a concrete example of when each is the appropriate instrument.
- [ ] I can describe the RED method and name the three PromQL query patterns (rate, error ratio, latency quantile) that implement it.
- [ ] I can explain why using high-cardinality values (user IDs, request IDs) as Prometheus labels is dangerous and how to avoid it.
- [ ] I can implement a `/metrics` endpoint with `generate_latest(REGISTRY)` and explain how Prometheus scraping works (pull model, scrape interval, text exposition format).

---

## Supplementary reading

- [Prometheus — Metric types](https://prometheus.io/docs/concepts/metric_types/)
- [Prometheus — Instrumentation best practices](https://prometheus.io/docs/practices/instrumentation/)
- [`prometheus-client` Python library](https://github.com/prometheus/client_python)
- [Tom Wilkie — The RED method](https://grafana.com/blog/2018/08/02/the-red-method-how-to-instrument-your-services/)
- [Brendan Gregg — The USE method](https://www.brendangregg.com/usemethod.html)
- [PromQL — Querying basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
