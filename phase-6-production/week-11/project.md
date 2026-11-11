# Week 11 Project — Full Observability Stack for the LLM Gateway

**Time**: Saturday ~4 hours (implementation) + Sunday (assessment)

---

## Overview

Instrument the LLM gateway built in Weeks 9–10 — an API gateway with routing and rate limiting — with a complete, production-grade observability stack. By the end of the week you will have structured logs with correlation IDs flowing through every layer, Prometheus metrics scraped by a local collector, OpenTelemetry traces linkable across services, per-model LLM cost tracking backed by Pydantic v2 models, and SLO compliance checks with auto-generated Prometheus alerting rules.

All LLM API calls are **mocked** in the test suite. Real Anthropic calls use `os.getenv("ANTHROPIC_API_KEY")`.

---

## Learning Objectives

- Implement the three pillars of observability (logs, metrics, traces) in a single FastAPI service using production-ready Python libraries.
- Propagate correlation IDs through async middleware using `contextvars.ContextVar` without polluting function signatures.
- Define, instrument, and expose RED-method Prometheus metrics and write PromQL expressions against them.
- Build a complete OpenTelemetry tracing pipeline with parent/child spans and semantic span attributes.
- Model LLM cost and token usage with Pydantic v2 and integrate the results into Prometheus counters.
- Define SLOs programmatically with Pydantic v2 models and generate Prometheus alerting YAML from them.

---

## Prerequisites and Setup

### Install dependencies

```bash
uv add prometheus-client \
    opentelemetry-api \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-http \
    fastapi uvicorn anthropic pydantic
```

### Environment variables

```bash
# .env  (never commit this file)
ANTHROPIC_API_KEY=...   # loaded via os.getenv("ANTHROPIC_API_KEY")
```

```python
# config.py
import os
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
```

Keep `.env` in `.gitignore`. Maintain `.env.example` with placeholder values.

### Suggested directory structure

```
gateway/
├── src/
│   └── gateway/
│       ├── __init__.py
│       ├── main.py              # FastAPI app factory, middleware registration
│       ├── config.py            # pydantic-settings or os.getenv config
│       ├── middleware/
│       │   ├── __init__.py
│       │   ├── logging.py       # LoggingMiddleware, JSONFormatter
│       │   ├── metrics.py       # MetricsMiddleware, metric definitions
│       │   └── tracing.py       # OTelMiddleware, setup_tracing()
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── cost.py          # LLMCallRecord, CostTracker, COST_PER_1K_TOKENS
│       │   └── slo.py           # SLOConfig, RequestRecord, SLOReport, compute_slo_compliance
│       └── routers/
│           ├── chat.py          # /chat endpoint
│           └── health.py        # /health endpoint
├── tests/
│   ├── conftest.py
│   ├── test_logging.py
│   ├── test_metrics.py
│   ├── test_slo.py
│   └── test_cost.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

---

## Implementation Phases

### Phase 1 — Structured Logging Layer (45 min)

**Goal**: every log line is machine-parseable JSON; every line carries the `correlation_id` of the in-flight request.

#### 1.1 ContextVar for correlation ID

```python
# middleware/logging.py
from contextvars import ContextVar
import uuid

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

def get_correlation_id() -> str:
    return correlation_id_var.get()
```

#### 1.2 Custom JSON log formatter

Implement a `logging.Formatter` subclass that serialises each `LogRecord` to a JSON object. Required output fields:

| Field | Value |
|---|---|
| `timestamp` | ISO 8601 UTC string |
| `level` | `record.levelname` |
| `correlation_id` | `correlation_id_var.get()` |
| `message` | formatted message |
| `logger` | `record.name` |
| extra fields | any extra key/values passed via `extra={}` |

```python
import json
import logging
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "correlation_id": correlation_id_var.get(),
            "message": record.getMessage(),
            "logger": record.name,
        }
        # merge any extra= fields the caller provided
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload)
```

#### 1.3 LoggingMiddleware

ASGI middleware that runs before every request:

1. Reads `X-Correlation-ID` from request headers; generates `uuid.uuid4().hex` if absent.
2. Stores the ID in `correlation_id_var` via `token = correlation_id_var.set(cid)`.
3. Logs request start: `method`, `path`.
4. Awaits `call_next(request)`.
5. Logs request end: `status_code`, `duration_ms` (wall-clock difference).
6. Resets the ContextVar: `correlation_id_var.reset(token)`.

```python
import time
import uuid
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("gateway.access")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        token = correlation_id_var.set(cid)
        start = time.perf_counter()
        logger.info("request.start", extra={"method": request.method, "path": request.url.path})
        try:
            response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request.end",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            correlation_id_var.reset(token)
        return response
```

---

### Phase 2 — Prometheus Metrics (45 min)

**Goal**: RED-method metrics exposed at `/metrics` and scrapeable by Prometheus.

#### 2.1 Metric definitions

Define at module level in `middleware/metrics.py`:

```python
from prometheus_client import Counter, Gauge, Histogram

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens processed",
    labelnames=["model", "token_type"],  # token_type: input | output
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Cumulative LLM cost in USD",
    labelnames=["model"],
)

active_connections = Gauge(
    "active_connections",
    "Number of requests currently in flight",
)
```

#### 2.2 MetricsMiddleware

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        active_connections.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            duration = time.perf_counter() - start
            active_connections.dec()
            http_request_duration_seconds.observe(duration)
            http_requests_total.labels(
                method=request.method,
                path=request.url.path,
                status_code=str(response.status_code),
            ).inc()
        return response
```

#### 2.3 `/metrics` endpoint

```python
from fastapi import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

---

### Phase 3 — OpenTelemetry Tracing (45 min)

**Goal**: every request produces a trace with at least one root span and one LLM child span, exportable to any OTLP-compatible backend.

#### 3.1 TracerProvider setup

```python
# middleware/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

def setup_tracing(service_name: str = "llm-gateway") -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider
```

To export to an OTLP collector instead of stdout, swap `ConsoleSpanExporter` for `OTLPSpanExporter` from `opentelemetry-exporter-otlp-proto-http`.

#### 3.2 OTelMiddleware

```python
from opentelemetry import trace as otel_trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

tracer = otel_trace.get_tracer("gateway")

class OTelMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        with tracer.start_as_current_span("http.request") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
        return response
```

#### 3.3 Child spans inside handlers

In `routers/chat.py`, wrap the LLM call in a child span. The `start_as_current_span` context manager automatically attaches the child to the active parent span:

```python
from opentelemetry import trace as otel_trace

tracer = otel_trace.get_tracer("gateway.chat")

async def call_llm(model: str, prompt: str) -> dict:
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        # --- actual or mocked Anthropic call here ---
        result = {"input_tokens": 120, "output_tokens": 80, "content": "..."}
        span.set_attribute("llm.input_tokens", result["input_tokens"])
        span.set_attribute("llm.output_tokens", result["output_tokens"])
    return result
```

Required span attributes:

| Span | Attribute | Example value |
|---|---|---|
| `http.request` | `http.method` | `"POST"` |
| `http.request` | `http.route` | `"/chat"` |
| `http.request` | `http.status_code` | `200` |
| `llm.call` | `llm.model` | `"claude-sonnet-5-5"` |
| `llm.call` | `llm.input_tokens` | `120` |
| `llm.call` | `llm.output_tokens` | `80` |

---

### Phase 4 — LLM Cost Tracking (30 min)

**Goal**: every LLM call produces an `LLMCallRecord`; a `CostTracker` aggregates totals per model and integrates with Prometheus counters.

#### 4.1 Models and pricing table

```python
# observability/cost.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "claude-opus-5-5":   {"input": 0.015,  "output": 0.075},
    "claude-sonnet-5-5": {"input": 0.003,  "output": 0.015},
    "claude-haiku-5-5":  {"input": 0.00025,"output": 0.00125},
}

class LLMCallRecord(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

#### 4.2 Cost calculation

```python
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = COST_PER_1K_TOKENS[model]
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
```

#### 4.3 CostTracker

```python
class CostTracker:
    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []

    def record(self, rec: LLMCallRecord) -> None:
        self._records.append(rec)
        # mirror into Prometheus
        llm_tokens_total.labels(model=rec.model, token_type="input").inc(rec.input_tokens)
        llm_tokens_total.labels(model=rec.model, token_type="output").inc(rec.output_tokens)
        llm_cost_usd_total.labels(model=rec.model).inc(rec.cost_usd)

    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self._records)

    def summary(self) -> dict[str, Any]:
        by_model: dict[str, dict[str, Any]] = {}
        for rec in self._records:
            entry = by_model.setdefault(rec.model, {"calls": 0, "cost_usd": 0.0,
                                                     "input_tokens": 0, "output_tokens": 0})
            entry["calls"] += 1
            entry["cost_usd"] += rec.cost_usd
            entry["input_tokens"] += rec.input_tokens
            entry["output_tokens"] += rec.output_tokens
        return {
            "total_cost_usd": self.total_cost(),
            "call_count": len(self._records),
            "by_model": by_model,
        }
```

---

### Phase 5 — SLO Definitions (30 min)

**Goal**: compute SLO compliance from request records and generate Prometheus alerting rules automatically.

#### 5.1 Pydantic models

```python
# observability/slo.py
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class SLOConfig(BaseModel):
    name: str
    target: float = Field(gt=0.0, le=1.0)   # e.g. 0.999
    window_seconds: int = Field(gt=0)         # e.g. 2592000 for 30 days

class RequestRecord(BaseModel):
    success: bool
    duration_ms: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SLOReport(BaseModel):
    slo_name: str
    target: float
    actual: float
    compliant: bool
    sample_size: int
```

#### 5.2 Compliance computation

```python
def compute_slo_compliance(
    records: list[RequestRecord],
    slo_name: str,
    target: float,
) -> SLOReport:
    if not records:
        return SLOReport(slo_name=slo_name, target=target,
                         actual=0.0, compliant=False, sample_size=0)
    actual = sum(1 for r in records if r.success) / len(records)
    return SLOReport(
        slo_name=slo_name,
        target=target,
        actual=actual,
        compliant=actual >= target,
        sample_size=len(records),
    )
```

#### 5.3 Pre-defined SLOs

```python
latency_slo = SLOConfig(
    name="latency_p99_under_2s",
    target=0.99,          # 99% of requests complete within 2000ms
    window_seconds=86400, # 24-hour window
)

availability_slo = SLOConfig(
    name="availability_99_9",
    target=0.999,          # 99.9% success rate
    window_seconds=2592000, # 30-day window
)
```

#### 5.4 Alert rule generation

```python
def generate_alert_rules(slos: list[SLOConfig]) -> str:
    """Return a Prometheus YAML alert rules string for the given SLOs."""
    rules = ["groups:", "  - name: slo_alerts", "    rules:"]
    for slo in slos:
        error_budget = 1.0 - slo.target
        rules.append(f"""    - alert: SLOBreach_{slo.name}
      expr: >
        (
          sum(rate(http_requests_total{{status_code=~"5.."}}[{slo.window_seconds}s]))
          /
          sum(rate(http_requests_total[{slo.window_seconds}s]))
        ) > {error_budget}
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "SLO breach: {slo.name}"
        description: "Error rate exceeds SLO target of {slo.target*100:.2f}%"
""")
    return "\n".join(rules)
```

---

### Phase 6 — Dashboard Spec (15 min)

Define the RED dashboard as a Pydantic model so it can be serialised to JSON and consumed by a dashboard-as-code tool (e.g. Grafana provisioning or grafanalib):

```python
from pydantic import BaseModel

class DashboardPanel(BaseModel):
    title: str
    panel_type: str       # "graph" | "stat" | "gauge"
    promql: str
    unit: str
    description: str

class DashboardSpec(BaseModel):
    title: str
    refresh: str
    panels: list[DashboardPanel]

RED_DASHBOARD = DashboardSpec(
    title="LLM Gateway — RED Dashboard",
    refresh="30s",
    panels=[
        DashboardPanel(
            title="Request Rate",
            panel_type="graph",
            promql='sum(rate(http_requests_total[5m])) by (path)',
            unit="req/s",
            description="Requests per second per endpoint (RED: Rate)",
        ),
        DashboardPanel(
            title="Error Rate",
            panel_type="stat",
            promql=(
                'sum(rate(http_requests_total{status_code=~"5.."}[5m])) '
                '/ sum(rate(http_requests_total[5m])) * 100'
            ),
            unit="%",
            description="Percentage of 5xx responses over 5-minute window (RED: Errors)",
        ),
        DashboardPanel(
            title="p99 Latency",
            panel_type="graph",
            promql='histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
            unit="s",
            description="99th-percentile request duration (RED: Duration)",
        ),
        DashboardPanel(
            title="LLM Cost / hr",
            panel_type="stat",
            promql='sum(increase(llm_cost_usd_total[1h]))',
            unit="USD",
            description="Cumulative LLM API cost in the last hour",
        ),
    ],
)
```

---

## Tests

All LLM calls are mocked via `unittest.mock.patch`. Never import or call `anthropic.Anthropic` directly in tests.

### `tests/test_logging.py`

```python
import json
import logging
from gateway.middleware.logging import JSONFormatter, correlation_id_var

def test_json_formatter_includes_correlation_id():
    correlation_id_var.set("test-cid-123")
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="hello", args=(), exc_info=None,
    )
    output = json.loads(formatter.format(record))
    assert output["correlation_id"] == "test-cid-123"
    assert output["message"] == "hello"
    assert "timestamp" in output
    assert output["level"] == "INFO"
```

### `tests/test_metrics.py`

```python
from prometheus_client import REGISTRY
from gateway.middleware.metrics import http_requests_total, http_request_duration_seconds

def test_counter_increments():
    before = http_requests_total.labels(method="GET", path="/health", status_code="200")._value.get()
    http_requests_total.labels(method="GET", path="/health", status_code="200").inc()
    after = http_requests_total.labels(method="GET", path="/health", status_code="200")._value.get()
    assert after == before + 1

def test_histogram_observation():
    http_request_duration_seconds.observe(0.123)
    # verify no exception raised and bucket exists
    metrics = list(REGISTRY.get_sample_value(
        "http_request_duration_seconds_count"
    ) or [])  # smoke test — just checking no error
```

### `tests/test_slo.py`

```python
from datetime import datetime, timezone
from gateway.observability.slo import RequestRecord, compute_slo_compliance

def _make_records(n: int, success: bool) -> list[RequestRecord]:
    return [
        RequestRecord(success=success, duration_ms=100.0,
                      timestamp=datetime.now(timezone.utc))
        for _ in range(n)
    ]

def test_full_compliance():
    records = _make_records(1000, success=True)
    report = compute_slo_compliance(records, "availability", 0.999)
    assert report.compliant is True
    assert report.actual == 1.0
    assert report.sample_size == 1000

def test_breach_detected():
    records = _make_records(990, True) + _make_records(10, False)  # 99.0% < 99.9%
    report = compute_slo_compliance(records, "availability", 0.999)
    assert report.compliant is False

def test_empty_records():
    report = compute_slo_compliance([], "availability", 0.999)
    assert report.compliant is False
    assert report.sample_size == 0
```

### `tests/test_cost.py`

```python
from unittest.mock import patch, MagicMock
from gateway.observability.cost import (
    CostTracker, LLMCallRecord, calculate_cost, COST_PER_1K_TOKENS,
)

def test_calculate_cost_sonnet():
    cost = calculate_cost("claude-sonnet-5-5", input_tokens=1000, output_tokens=500)
    expected = 1.0 * 0.003 + 0.5 * 0.015
    assert abs(cost - expected) < 1e-9

def test_cost_tracker_summary():
    tracker = CostTracker()
    rec = LLMCallRecord(
        model="claude-haiku-5-5",
        input_tokens=500,
        output_tokens=200,
        cost_usd=calculate_cost("claude-haiku-5-5", 500, 200),
        latency_ms=350.0,
    )
    tracker.record(rec)
    summary = tracker.summary()
    assert summary["call_count"] == 1
    assert summary["total_cost_usd"] > 0
    assert "claude-haiku-5-5" in summary["by_model"]

@patch("anthropic.Anthropic")
def test_llm_call_mocked(mock_anthropic_cls):
    """Verify that the LLM client is never called in unit tests."""
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        usage=MagicMock(input_tokens=100, output_tokens=50),
        content=[MagicMock(text="mocked response")],
    )
    # import and call your gateway's LLM wrapper here
    # assert mock_client.messages.create.called
```

---

## Deliverables Checklist

- [ ] `LoggingMiddleware` sets `correlation_id` in `ContextVar` and logs JSON with that ID on every request.
- [ ] `JSONFormatter` output is valid JSON parseable with `json.loads()`.
- [ ] `GET /metrics` returns Prometheus text with `http_requests_total`, `http_request_duration_seconds`, `active_connections`, `llm_tokens_total`, `llm_cost_usd_total`.
- [ ] `OTelMiddleware` creates a root span with `http.method` and `http.route` attributes.
- [ ] `/chat` handler creates a child `llm.call` span with `llm.model`, `llm.input_tokens`, `llm.output_tokens`.
- [ ] `LLMCallRecord` and `RequestRecord` are Pydantic v2 `BaseModel` subclasses with appropriate field validators.
- [ ] `COST_PER_1K_TOKENS` covers all three models: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`.
- [ ] `CostTracker.record()` increments `llm_tokens_total` and `llm_cost_usd_total` Prometheus counters.
- [ ] `compute_slo_compliance` returns `compliant=True` for 1000 successful records at a 99.9% target.
- [ ] `generate_alert_rules()` returns a string containing valid YAML with `alert:` and `expr:` keys.
- [ ] `RED_DASHBOARD` spec serialises to JSON with four panels.
- [ ] `pytest tests/` passes with zero failures; all Anthropic API calls are mocked.
- [ ] `.env` in `.gitignore`; `.env.example` present with placeholder values.
- [ ] No hardcoded API keys anywhere in the codebase.

---

## Stretch Goals

- **Jaeger / Tempo integration**: swap `ConsoleSpanExporter` for `OTLPSpanExporter` and run Jaeger locally via Docker (`docker run -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one`). Verify traces appear in the Jaeger UI.
- **Alertmanager webhook**: write a minimal FastAPI endpoint that receives Alertmanager webhook payloads and logs them as structured JSON; wire it up in `alertmanager.yml`.
- **Custom OTel exporter**: implement a `SpanExporter` subclass that writes spans to a local SQLite database for offline analysis.
- **Exemplars**: attach a trace ID as a Prometheus Exemplar on `http_request_duration_seconds` observations so Grafana can link from a histogram bucket to the corresponding trace.
- **Token budget middleware**: add a middleware that rejects requests exceeding a configurable per-minute token budget, returning HTTP 429 with a `Retry-After` header.
