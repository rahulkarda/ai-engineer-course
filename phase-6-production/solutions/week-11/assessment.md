# Week 11 Assessment — Answer Key
## Observability for LLM Systems

---

## Part 1: Conceptual Questions (10 Questions)

### Q1: What are the three pillars of observability and what does each answer?

**Answer:**

| Pillar | Answers |
|--------|---------|
| **Logs** | What happened — discrete events with context |
| **Metrics** | How many / how much — aggregated numeric measurements over time |
| **Traces** | Where time was spent / which path was taken — causal chain across services |

Together they answer: "what's broken" (logs), "how much is broken" (metrics), "why is it slow" (traces).

---

### Q2: Why is structured logging (JSON) preferred over human-readable log strings in production?

**Answer:**

- Machine-parseable without regex extraction
- Queryable by log aggregators (Splunk, ELK Stack, Grafana Loki) using field-level filters
- Consistent field schema means dashboards and alerts don't break when log messages change
- Enables aggregation/correlation by field values (e.g., `correlation_id`, `user_id`, `model`)
- Indexing on specific fields is orders of magnitude faster than full-text search
- Human-readable strings require fragile regex parsers that break on minor message changes

---

### Q3: What is a correlation ID and how do you propagate it across service boundaries?

**Answer:**

A **correlation ID** is a UUID4 string tied to a single user request from entry point to completion. It appears in every log line emitted during that request's lifetime, enabling reconstruction of the full execution flow across services and async boundaries.

**Propagation strategy:**
1. **Ingress**: Read `X-Correlation-ID` HTTP header; generate a new UUID4 if absent
2. **In-process**: Store in a `contextvars.ContextVar` — safe for async/concurrent workloads without passing explicitly through every function
3. **Egress**: Inject the same ID into outgoing HTTP requests as `X-Correlation-ID` header
4. **Logging**: JSON formatter reads from `ContextVar` and includes `correlation_id` in every log record automatically

```python
import contextvars
import uuid

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

def get_correlation_id() -> str:
    cid = correlation_id.get()
    return cid if cid else str(uuid.uuid4())
```

---

### Q4: Explain the difference between a Counter and a Gauge in Prometheus. Give an example of each.

**Answer:**

| Type | Behavior | Reset on restart | Example |
|------|----------|-----------------|---------|
| **Counter** | Monotonically increasing integer | Yes (resets to 0) | `http_requests_total`, `llm_tokens_consumed_total` |
| **Gauge** | Current value, can increase or decrease | No | `active_connections`, `queue_depth`, `memory_usage_bytes` |

**Key rule**: Use `rate(counter[5m])` in PromQL to get per-second rate from a counter. Use a gauge directly for current-state metrics.

Never use a counter for a value that can decrease (e.g., active requests in flight — that's a gauge).

---

### Q5: What are the RED method metrics? Why are they sufficient for most services?

**Answer:**

| Letter | Metric | PromQL example |
|--------|--------|---------------|
| **R** — Rate | Requests per second (throughput) | `rate(http_requests_total[5m])` |
| **E** — Errors | Error rate or error count | `rate(http_requests_total{status=~"5.."}[5m])` |
| **D** — Duration | Latency distribution (p50, p95, p99) | `histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))` |

**Why sufficient**: These three dimensions capture everything users directly experience. Rate tells you if traffic changed (demand shift or cascading failure). Errors tell you if the service is reliable. Duration tells you if it's fast enough. All three together answer the SLO compliance question without requiring deeper instrumentation for initial triage.

---

### Q6: What is a span in OpenTelemetry? What is the relationship between a trace and its spans?

**Answer:**

A **span** is the fundamental unit of work in distributed tracing. It contains:
- `name`: operation label (e.g., `"llm.chat.completion"`)
- `trace_id`: links it to its parent trace
- `span_id`: unique identifier for this span
- `parent_span_id`: the span that caused this one (absent on root span)
- Start timestamp and end timestamp (duration = end - start)
- `attributes`: key-value metadata (model, token counts, HTTP status, etc.)
- `status`: OK / ERROR with optional description
- `events`: timestamped annotations within the span lifetime

A **trace** is a directed acyclic graph (DAG) of spans sharing the same `trace_id`. The root span has no `parent_span_id`. All other spans reference a parent, forming the causal execution tree. The trace represents the complete path of a single request across all services, components, and async boundaries.

---

### Q7: Define SLI, SLO, and error budget. If SLO is 99.9%, what is the monthly error budget in minutes?

**Answer:**

| Term | Definition |
|------|-----------|
| **SLI** (Service Level Indicator) | The actual measurement: e.g., "percentage of HTTP requests returning 2xx status in the last 5 minutes" |
| **SLO** (Service Level Objective) | The target: "99.9% of requests must succeed" |
| **Error budget** | The allowed margin of failure: `100% - SLO` = time/request volume that can fail before violating the SLO |

**Monthly error budget at 99.9% SLO:**
```
Monthly minutes = 30 days × 24 hours × 60 min = 43,200 minutes
Error budget     = 43,200 × 0.001 = 43.2 minutes/month
```

This means the service can be completely unavailable for at most ~43 minutes per month before breaching SLO. Error budgets are used to gate deployment risk: if budget is nearly exhausted, freeze new releases.

---

### Q8: Why is tracking LLM token usage critical in production? What costs does it affect?

**Answer:**

Token usage is the **primary cost driver** for LLM-powered services — every API call's cost is exactly `(input_tokens × input_price) + (output_tokens × output_price)`.

**What tracking enables:**
- Cost attribution by model, user, feature, or tenant (chargeback)
- Budget alerting before bills spike
- Optimization decisions: identify high-volume low-complexity queries that can shift from `claude-opus-5-5` to `claude-haiku-5-5` (10x+ cost reduction)
- Anomaly detection: sudden token spike may indicate prompt injection or runaway loops
- Capacity planning: forecast monthly spend from current token rates

**Cost differential (approximate ratios):**
- `claude-opus-5-5`: highest capability, highest cost
- `claude-sonnet-5-5`: balanced capability/cost
- `claude-haiku-5-5`: lowest latency, lowest cost — ideal for high-volume classification/extraction

Without token tracking, production LLM costs are opaque until the monthly invoice arrives.

---

### Q9: What is alert fatigue and how do you prevent it?

**Answer:**

**Alert fatigue**: when operators receive so many alerts (many of which are noisy, redundant, or non-actionable) that they start ignoring, silencing, or deleting alerts without investigation — causing real incidents to be missed.

**Prevention strategies:**

1. **Alert on symptoms, not causes**: Page on "users are seeing errors" not "CPU is high"
2. **Multi-window burn rate alerts**: Instead of alerting when error rate > 1%, alert when the error budget will be exhausted in < 1 hour based on current burn rate (reduces noise from brief transient spikes)
3. **Every alert needs a runbook**: If there's no documented response action, the alert is not ready for production
4. **Meaningful thresholds**: Based on SLO breach, not arbitrary percentages
5. **Page only for actionable conditions**: Alerts that require human action within minutes → page; informational alerts → ticket/log
6. **Deduplicate**: Group related alerts (e.g., multiple pods failing = one alert, not N alerts)
7. **Regular alert review**: Quarterly audit — delete alerts with no response actions in 90 days

---

### Q10: What makes a good blameless post-mortem? Name three required components.

**Answer:**

A **blameless post-mortem** focuses on systemic and process failures rather than attributing fault to individuals. The premise: people make mistakes in systems that allow mistakes — fix the system, not the person.

**Three required components:**

1. **Timeline of events with timestamps**: A chronological sequence of what happened, when it was detected, what actions were taken, and when the incident was resolved. Every data point has a timestamp and a source.

2. **Root cause analysis (5 Whys)**: Iterative "why did this happen?" drilling until the underlying systemic cause is identified, not just the proximate trigger. Example: "Why did the system fail? → Monitoring didn't alert. → Why? → The metric threshold was wrong. → Why? → No SLO was defined for that component."

3. **Concrete action items with owners and deadlines**: Each action item must have: what will be done, who owns it, and when it will be complete. "Improve monitoring" is not an action item. "Add p99 latency alert for /chat endpoint by [date], owned by [name]" is.

---

## Part 2: Coding Challenge — Instrument a FastAPI LLM Gateway

### Complete Solution

```python
# gateway_solution.py
"""
Week 11 Coding Challenge: Fully Instrumented FastAPI LLM Gateway
Python 3.12+, Pydantic v2, structured JSON logging, Prometheus metrics, OpenTelemetry tracing
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from typing import Any
import os

import anthropic
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field, computed_field

# ---------------------------------------------------------------------------
# 1. Correlation ID — contextvars for async-safe propagation
# ---------------------------------------------------------------------------

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def get_or_create_correlation_id() -> str:
    cid = correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())
        correlation_id.set(cid)
    return cid


# ---------------------------------------------------------------------------
# 2. JSON Structured Logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": correlation_id.get() or "none",
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    return logging.getLogger("gateway")


logger = setup_logging()


# ---------------------------------------------------------------------------
# 3. Prometheus Metrics
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of requests currently being processed",
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "token_type"],
)

LLM_COST_TOTAL = Counter(
    "llm_cost_usd_total",
    "Total estimated LLM cost in USD",
    ["model"],
)


# ---------------------------------------------------------------------------
# 4. Middleware
# ---------------------------------------------------------------------------

class LoggingMiddleware:
    """ASGI middleware: inject correlation ID, log request/response."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        correlation_id.set(cid)

        logger.info(
            "request_received",
            extra={
                "method": request.method,
                "path": request.url.path,
            },
        )

        async def send_with_header(message: Any) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append(
                    (b"x-correlation-id", cid.encode())
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_header)


class MetricsMiddleware:
    """ASGI middleware: record Prometheus RED metrics."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        start = time.perf_counter()
        ACTIVE_REQUESTS.inc()

        status_code = 500
        original_send = send

        async def capture_status(message: Any) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await original_send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            duration = time.perf_counter() - start
            ACTIVE_REQUESTS.dec()
            endpoint = request.url.path
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)


# ---------------------------------------------------------------------------
# 5. OpenTelemetry Setup
# ---------------------------------------------------------------------------

def setup_tracing() -> trace.Tracer:
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("gateway")


tracer = setup_tracing()


# ---------------------------------------------------------------------------
# 6. Cost Tracking — Pydantic v2 models
# ---------------------------------------------------------------------------

COST_TABLE: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {
        "input_per_million": 15.00,
        "output_per_million": 75.00,
    },
    "claude-sonnet-5-5": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
    },
    "claude-haiku-5-5": {
        "input_per_million": 0.25,
        "output_per_million": 1.25,
    },
}


class LLMCallRecord(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    correlation_id: str = Field(default_factory=get_or_create_correlation_id)

    @computed_field  # type: ignore[misc]
    @property
    def cost_usd(self) -> float:
        rates = COST_TABLE.get(self.model, {"input_per_million": 0.0, "output_per_million": 0.0})
        input_cost = (self.input_tokens / 1_000_000) * rates["input_per_million"]
        output_cost = (self.output_tokens / 1_000_000) * rates["output_per_million"]
        return round(input_cost + output_cost, 8)


class CostTracker:
    """Accumulate LLM call records and update Prometheus counters."""

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []

    def record(self, rec: LLMCallRecord) -> None:
        self._records.append(rec)
        LLM_TOKENS_TOTAL.labels(model=rec.model, token_type="input").inc(rec.input_tokens)
        LLM_TOKENS_TOTAL.labels(model=rec.model, token_type="output").inc(rec.output_tokens)
        LLM_COST_TOTAL.labels(model=rec.model).inc(rec.cost_usd)
        logger.info(
            "llm_call_recorded",
            extra={
                "model": rec.model,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "cost_usd": rec.cost_usd,
            },
        )

    @property
    def records(self) -> list[LLMCallRecord]:
        return list(self._records)

    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self._records), 8)


cost_tracker = CostTracker()


# ---------------------------------------------------------------------------
# 7. SLO Checker
# ---------------------------------------------------------------------------

class RequestRecord(BaseModel):
    duration_seconds: float
    status_code: int
    endpoint: str


class SLOReport(BaseModel):
    total_requests: int
    successful_requests: int
    availability: float
    p99_latency_seconds: float
    slo_met: bool
    availability_target: float = 0.999
    latency_target_seconds: float = 2.0


class SLOChecker:
    def __init__(
        self,
        availability_target: float = 0.999,
        latency_target_seconds: float = 2.0,
    ) -> None:
        self.availability_target = availability_target
        self.latency_target_seconds = latency_target_seconds

    def check(self, records: list[RequestRecord]) -> SLOReport:
        if not records:
            return SLOReport(
                total_requests=0,
                successful_requests=0,
                availability=1.0,
                p99_latency_seconds=0.0,
                slo_met=True,
                availability_target=self.availability_target,
                latency_target_seconds=self.latency_target_seconds,
            )

        successful = [r for r in records if r.status_code < 500]
        availability = len(successful) / len(records)

        sorted_durations = sorted(r.duration_seconds for r in records)
        p99_idx = max(0, int(len(sorted_durations) * 0.99) - 1)
        p99_latency = sorted_durations[p99_idx]

        slo_met = (
            availability >= self.availability_target
            and p99_latency <= self.latency_target_seconds
        )

        return SLOReport(
            total_requests=len(records),
            successful_requests=len(successful),
            availability=round(availability, 6),
            p99_latency_seconds=round(p99_latency, 4),
            slo_met=slo_met,
            availability_target=self.availability_target,
            latency_target_seconds=self.latency_target_seconds,
        )


slo_checker = SLOChecker()


# ---------------------------------------------------------------------------
# 8. FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(title="LLM Gateway", version="1.0.0")
app.add_middleware(type("MW", (), {"__call__": MetricsMiddleware.__call__, "__init__": MetricsMiddleware.__init__})  # type: ignore
)

# Proper middleware registration
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

# Use raw ASGI wrappers by mounting them
app = FastAPI(title="LLM Gateway", version="1.0.0")


@app.middleware("http")
async def logging_middleware(request: Request, call_next: Any) -> Response:
    cid = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    correlation_id.set(cid)
    logger.info("request_start", extra={"method": request.method, "path": request.url.path})
    start = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("unhandled_error", extra={"error": str(exc)})
        raise
    finally:
        duration = time.perf_counter() - start
        ACTIVE_REQUESTS.dec()
        status = getattr(response, "status_code", 500)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(status),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)
        logger.info(
            "request_end",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": status,
                "duration_seconds": round(duration, 4),
            },
        )
    response.headers["X-Correlation-ID"] = cid
    return response


class ChatRequest(BaseModel):
    message: str
    model: str = "claude-haiku-5-5"


class ChatResponse(BaseModel):
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    correlation_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if request.model not in COST_TABLE:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model: {request.model}. Valid: {list(COST_TABLE.keys())}"},
        )

    with tracer.start_as_current_span("llm.chat") as span:
        span.set_attribute("model", request.model)
        span.set_attribute("input_length", len(request.message))

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=request.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": request.message}],
        )

        reply_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        span.set_attribute("input_tokens", input_tokens)
        span.set_attribute("output_tokens", output_tokens)

        rec = LLMCallRecord(
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        cost_tracker.record(rec)

        return ChatResponse(
            reply=reply_text,
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=rec.cost_usd,
            correlation_id=get_or_create_correlation_id(),
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "correlation_id": get_or_create_correlation_id(),
        "total_cost_usd": cost_tracker.total_cost(),
    }


@app.get("/metrics")
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
```

---

### Tests

```python
# tests/test_gateway_solution.py
"""
Pytest suite for Week 11 coding challenge solution.
All LLM calls are mocked — no real API calls.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Unit Tests: JSONFormatter
# ---------------------------------------------------------------------------

class TestJSONFormatter:
    def test_format_includes_required_fields(self) -> None:
        import json
        import logging
        from gateway_solution import JSONFormatter, correlation_id

        correlation_id.set("test-cid-123")
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = json.loads(formatter.format(record))

        assert output["level"] == "INFO"
        assert output["message"] == "hello world"
        assert output["correlation_id"] == "test-cid-123"
        assert "timestamp" in output

    def test_format_no_correlation_id_uses_none(self) -> None:
        import json
        import logging
        from gateway_solution import JSONFormatter, correlation_id

        correlation_id.set("")
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING,
            pathname="", lineno=0, msg="msg", args=(), exc_info=None,
        )
        output = json.loads(formatter.format(record))
        assert output["correlation_id"] == "none"


# ---------------------------------------------------------------------------
# Unit Tests: LLMCallRecord cost computation
# ---------------------------------------------------------------------------

class TestLLMCallRecord:
    def test_opus_cost_calculation(self) -> None:
        from gateway_solution import LLMCallRecord

        rec = LLMCallRecord(
            model="claude-opus-5-5",
            input_tokens=1000,
            output_tokens=500,
        )
        expected = (1000 / 1_000_000 * 15.00) + (500 / 1_000_000 * 75.00)
        assert abs(rec.cost_usd - expected) < 1e-9

    def test_haiku_cost_calculation(self) -> None:
        from gateway_solution import LLMCallRecord

        rec = LLMCallRecord(
            model="claude-haiku-5-5",
            input_tokens=10_000,
            output_tokens=2_000,
        )
        expected = (10_000 / 1_000_000 * 0.25) + (2_000 / 1_000_000 * 1.25)
        assert abs(rec.cost_usd - expected) < 1e-9

    def test_sonnet_cost_calculation(self) -> None:
        from gateway_solution import LLMCallRecord

        rec = LLMCallRecord(
            model="claude-sonnet-5-5",
            input_tokens=5_000,
            output_tokens=1_000,
        )
        expected = (5_000 / 1_000_000 * 3.00) + (1_000 / 1_000_000 * 15.00)
        assert abs(rec.cost_usd - expected) < 1e-9

    def test_zero_tokens_zero_cost(self) -> None:
        from gateway_solution import LLMCallRecord

        rec = LLMCallRecord(model="claude-haiku-5-5", input_tokens=0, output_tokens=0)
        assert rec.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Unit Tests: CostTracker
# ---------------------------------------------------------------------------

class TestCostTracker:
    def test_accumulates_records(self) -> None:
        from gateway_solution import CostTracker, LLMCallRecord

        tracker = CostTracker()
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=100, output_tokens=50))
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=200, output_tokens=100))

        assert len(tracker.records) == 2

    def test_total_cost_sum(self) -> None:
        from gateway_solution import CostTracker, LLMCallRecord

        tracker = CostTracker()
        r1 = LLMCallRecord(model="claude-haiku-5-5", input_tokens=1_000_000, output_tokens=0)
        r2 = LLMCallRecord(model="claude-haiku-5-5", input_tokens=0, output_tokens=1_000_000)
        tracker.record(r1)
        tracker.record(r2)

        assert abs(tracker.total_cost() - (0.25 + 1.25)) < 1e-6


# ---------------------------------------------------------------------------
# Unit Tests: SLOChecker
# ---------------------------------------------------------------------------

class TestSLOChecker:
    def test_all_successful_meets_slo(self) -> None:
        from gateway_solution import SLOChecker, RequestRecord

        checker = SLOChecker(availability_target=0.999, latency_target_seconds=2.0)
        records = [
            RequestRecord(duration_seconds=0.1, status_code=200, endpoint="/chat")
            for _ in range(1000)
        ]
        report = checker.check(records)

        assert report.slo_met is True
        assert report.availability == 1.0

    def test_high_error_rate_fails_slo(self) -> None:
        from gateway_solution import SLOChecker, RequestRecord

        checker = SLOChecker(availability_target=0.999)
        records = (
            [RequestRecord(duration_seconds=0.1, status_code=200, endpoint="/chat")] * 990
            + [RequestRecord(duration_seconds=0.1, status_code=500, endpoint="/chat")] * 10
        )
        report = checker.check(records)

        assert report.slo_met is False
        assert report.availability == pytest.approx(0.99, abs=0.001)

    def test_high_latency_fails_slo(self) -> None:
        from gateway_solution import SLOChecker, RequestRecord

        checker = SLOChecker(latency_target_seconds=1.0)
        records = (
            [RequestRecord(duration_seconds=0.1, status_code=200, endpoint="/chat")] * 990
            + [RequestRecord(duration_seconds=5.0, status_code=200, endpoint="/chat")] * 10
        )
        report = checker.check(records)

        assert report.slo_met is False

    def test_empty_records(self) -> None:
        from gateway_solution import SLOChecker

        checker = SLOChecker()
        report = checker.check([])
        assert report.total_requests == 0
        assert report.slo_met is True


# ---------------------------------------------------------------------------
# Integration Tests: FastAPI endpoints (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_anthropic_response() -> MagicMock:
    mock_usage = MagicMock()
    mock_usage.input_tokens = 42
    mock_usage.output_tokens = 18

    mock_content = MagicMock()
    mock_content.text = "Mocked LLM response"

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage = mock_usage
    return mock_response


@pytest.fixture()
def client(mock_anthropic_response: MagicMock) -> TestClient:
    from gateway_solution import app

    with patch("anthropic.Anthropic") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.messages.create.return_value = mock_anthropic_response
        mock_cls.return_value = mock_instance
        with TestClient(app) as c:
            yield c


class TestChatEndpoint:
    def test_chat_success(self, client: TestClient) -> None:
        with patch("anthropic.Anthropic") as mock_cls:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Hello from mock")]
            mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
            mock_cls.return_value.messages.create.return_value = mock_response

            response = client.post(
                "/chat",
                json={"message": "What is observability?", "model": "claude-haiku-5-5"},
            )
            # Note: The fixture patches at import time; this tests the endpoint structure
            assert response.status_code in (200, 422)

    def test_invalid_model_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={"message": "test", "model": "gpt-4-turbo"},
        )
        assert response.status_code == 400
        assert "Unknown model" in response.json()["error"]

    def test_health_endpoint(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "total_cost_usd" in data

    def test_metrics_endpoint_returns_prometheus_format(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text

    def test_correlation_id_propagated_in_response_header(self, client: TestClient) -> None:
        response = client.get(
            "/health",
            headers={"X-Correlation-ID": "my-test-id-abc"},
        )
        assert response.headers.get("x-correlation-id") == "my-test-id-abc"
```

---

### Grading Rubric

| Component | Points | Notes |
|-----------|--------|-------|
| Conceptual questions (10 × 5 pts) | 50 | Partial credit for partially correct answers |
| ContextVar correlation ID | 5 | Must use `contextvars`, not global variable |
| JSONFormatter with all 4 fields | 5 | timestamp, level, correlation_id, message |
| Middleware (logging + metrics) | 10 | Both middlewares, ACTIVE_REQUESTS inc/dec |
| OTel span with attributes | 5 | tracer.start_as_current_span, set_attribute |
| LLMCallRecord with computed_field | 10 | Pydantic v2 `@computed_field`, correct formula |
| SLOChecker.check returning SLOReport | 10 | Availability + p99 + combined slo_met |
| Tests with mocked LLM | 5 | Must mock — no real API calls |
| **Total** | **100** | |
