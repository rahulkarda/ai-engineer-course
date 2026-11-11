# Week 11 Project — Reference Solution
## Full Observability Stack for the LLM Gateway

This is the reference implementation for the Week 11 project. It integrates all six observability phases into a production-ready LLM gateway:

1. Structured JSON logging with correlation ID propagation
2. Prometheus RED metrics and Grafana dashboard spec
3. OpenTelemetry distributed tracing
4. LLM token cost tracking
5. SLO compliance checking with Prometheus alert rules
6. Dashboard definition builder

All LLM calls use the Anthropic SDK with `os.getenv("ANTHROPIC_API_KEY")`. Tests mock all external calls.

---

## Project Structure

```
gateway/
├── gateway.py                    # FastAPI application entry point
├── observability/
│   ├── __init__.py
│   ├── logging.py                # ContextVar, JSONFormatter, LoggingMiddleware
│   ├── metrics.py                # Prometheus metrics, MetricsMiddleware
│   ├── tracing.py                # OTel TracerProvider, OTelMiddleware
│   ├── cost.py                   # LLMCallRecord, CostTracker, COST_TABLE
│   ├── slo.py                    # SLOConfig, RequestRecord, SLOReport, alert rules
│   └── dashboard.py              # DashboardPanel, DashboardSpec, build_red_dashboard
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Shared fixtures and mock helpers
│   ├── test_logging.py
│   ├── test_metrics.py
│   ├── test_tracing.py
│   ├── test_cost.py
│   ├── test_slo.py
│   ├── test_dashboard.py
│   └── test_gateway.py           # Integration tests with mocked LLM
├── pyproject.toml
└── .env.example
```

---

## observability/logging.py

```python
"""
Structured JSON logging with correlation ID propagation via contextvars.
"""

from __future__ import annotations

import contextvars
import json
import logging
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Correlation ID — async-safe via contextvars
# ---------------------------------------------------------------------------

correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def get_or_create_correlation_id() -> str:
    """Return existing correlation ID or generate a new UUID4."""
    cid = correlation_id.get()
    if not cid:
        cid = str(uuid.uuid4())
        correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Explicitly set the correlation ID for the current context."""
    correlation_id.set(cid)


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """
    Emit log records as single-line JSON.

    Required fields in every record:
      - timestamp  (ISO-8601 from formatTime)
      - level      (CRITICAL / ERROR / WARNING / INFO / DEBUG)
      - logger     (logger name)
      - correlation_id
      - message

    Optional extras injected by callers via extra={} are merged at top level.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": correlation_id.get() or "none",
            "message": record.getMessage(),
        }

        # Merge caller-supplied extras (skip internal LogRecord attrs)
        _skip = {
            "args", "created", "exc_info", "exc_text", "filename", "funcName",
            "levelname", "levelno", "lineno", "message", "module", "msecs",
            "msg", "name", "pathname", "process", "processName", "relativeCreated",
            "stack_info", "taskName", "thread", "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in _skip and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a logger with JSONFormatter attached to its stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def configure_root_logging(level: int = logging.INFO) -> None:
    """Replace root logger handlers with a single JSONFormatter handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


# ---------------------------------------------------------------------------
# LoggingMiddleware — ASGI
# ---------------------------------------------------------------------------

class LoggingMiddleware:
    """
    ASGI middleware that:
    1. Reads X-Correlation-ID from incoming request (generates if absent)
    2. Sets correlation_id ContextVar for the duration of the request
    3. Injects X-Correlation-ID into response headers
    4. Logs request start and request end (with status code + duration)
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self._logger = get_logger("gateway.access")

    async def __call__(
        self, scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import time
        from starlette.requests import Request

        request = Request(scope, receive)
        cid = request.headers.get("x-correlation-id") or str(uuid.uuid4())
        set_correlation_id(cid)

        self._logger.info(
            "request_start",
            extra={"method": request.method, "path": request.url.path},
        )
        start = time.perf_counter()
        status_holder: list[int] = [500]

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder[0] = message.get("status", 500)
                headers = list(message.get("headers", []))
                headers.append((b"x-correlation-id", cid.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            self._logger.info(
                "request_end",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_holder[0],
                    "duration_seconds": round(duration, 4),
                },
            )
```

---

## observability/metrics.py

```python
"""
Prometheus metrics definitions and MetricsMiddleware.

RED metrics:
  - http_requests_total          (Counter)  — Rate + Errors
  - http_request_duration_seconds (Histogram) — Duration
  - active_requests              (Gauge)    — concurrency

LLM-specific:
  - llm_tokens_total             (Counter)  — by model + token_type
  - llm_cost_usd_total           (Counter)  — by model
  - llm_request_duration_seconds (Histogram) — by model
"""

from __future__ import annotations

import time
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests by method, endpoint, and status code",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of HTTP requests currently in-flight",
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed, by model and token type (input/output)",
    ["model", "token_type"],
)

LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Cumulative estimated LLM API cost in USD, by model",
    ["model"],
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM API call latency in seconds, by model",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)


# ---------------------------------------------------------------------------
# MetricsMiddleware — ASGI
# ---------------------------------------------------------------------------

class MetricsMiddleware:
    """
    ASGI middleware recording:
    - ACTIVE_REQUESTS gauge (inc on entry, dec on exit)
    - HTTP_REQUESTS_TOTAL counter (labeled by method + endpoint + status)
    - HTTP_REQUEST_DURATION histogram (labeled by method + endpoint)

    Skips the /metrics endpoint itself to avoid self-referential noise.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self, scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        endpoint = request.url.path

        # Skip Prometheus scrape endpoint
        if endpoint == "/metrics":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        ACTIVE_REQUESTS.inc()
        status_holder: list[int] = [500]

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder[0] = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start
            ACTIVE_REQUESTS.dec()
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(status_holder[0]),
            ).inc()
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)


# ---------------------------------------------------------------------------
# /metrics route handler
# ---------------------------------------------------------------------------

async def metrics_endpoint(_request: Request) -> Response:
    """Expose Prometheus metrics in text format."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
```

---

## observability/tracing.py

```python
"""
OpenTelemetry tracing setup.

Exports spans to console by default (swap ConsoleSpanExporter for
OTLPSpanExporter in production pointing at Jaeger/Tempo).
"""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)


def build_tracer_provider(service_name: str = "llm-gateway") -> TracerProvider:
    """
    Construct and register a TracerProvider.

    Uses ConsoleSpanExporter for local development.
    Set OTEL_EXPORTER_OTLP_ENDPOINT to enable OTLP export.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        except ImportError:
            # Fall back to console if OTLP exporter not installed
            provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "gateway") -> trace.Tracer:
    """Return the application tracer from the registered provider."""
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# OTelMiddleware — ASGI wrapper
# ---------------------------------------------------------------------------

class OTelMiddleware:
    """
    ASGI middleware creating a root span for each HTTP request.

    Span attributes:
      - http.method
      - http.url
      - http.status_code
      - correlation_id (from ContextVar)
    """

    def __init__(self, app: Any, tracer_name: str = "gateway.http") -> None:
        self.app = app
        self._tracer = get_tracer(tracer_name)

    async def __call__(
        self, scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from observability.logging import get_or_create_correlation_id
        from starlette.requests import Request

        request = Request(scope, receive)
        span_name = f"{request.method} {request.url.path}"

        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("correlation_id", get_or_create_correlation_id())

            status_holder: list[int] = [500]

            async def send_wrapper(message: dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    status_holder[0] = message.get("status", 500)
                    span.set_attribute("http.status_code", status_holder[0])
                    if status_holder[0] >= 500:
                        span.set_status(
                            trace.StatusCode.ERROR,
                            f"HTTP {status_holder[0]}",
                        )
                await send(message)

            await self.app(scope, receive, send_wrapper)
```

---

## observability/cost.py

```python
"""
LLM token cost tracking.

COST_TABLE holds per-million-token prices for all supported models.
LLMCallRecord uses Pydantic v2 @computed_field to derive cost_usd.
CostTracker accumulates records and updates Prometheus counters.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import BaseModel, Field, computed_field

from observability.logging import get_or_create_correlation_id, get_logger
from observability.metrics import LLM_COST_USD_TOTAL, LLM_TOKENS_TOTAL

logger = get_logger("gateway.cost")


# ---------------------------------------------------------------------------
# Cost Table — USD per million tokens
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

SUPPORTED_MODELS: frozenset[str] = frozenset(COST_TABLE.keys())


# ---------------------------------------------------------------------------
# LLMCallRecord
# ---------------------------------------------------------------------------

class LLMCallRecord(BaseModel):
    """
    Immutable record of a single LLM API call.

    cost_usd is derived automatically from token counts and COST_TABLE rates.
    """

    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    correlation_id: str = Field(default_factory=get_or_create_correlation_id)

    model_config = {"frozen": True}

    @computed_field  # type: ignore[misc]
    @property
    def cost_usd(self) -> float:
        rates = COST_TABLE.get(
            self.model,
            {"input_per_million": 0.0, "output_per_million": 0.0},
        )
        input_cost = (self.input_tokens / 1_000_000) * rates["input_per_million"]
        output_cost = (self.output_tokens / 1_000_000) * rates["output_per_million"]
        return round(input_cost + output_cost, 8)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Thread-safe (asyncio-safe) accumulator for LLMCallRecords.

    Records are kept in memory and Prometheus counters are updated on each
    record() call. Use cost_by_model() for per-model breakdown reports.
    """

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []

    def record(self, rec: LLMCallRecord) -> None:
        """Append a call record and update Prometheus counters."""
        self._records.append(rec)
        LLM_TOKENS_TOTAL.labels(model=rec.model, token_type="input").inc(
            rec.input_tokens
        )
        LLM_TOKENS_TOTAL.labels(model=rec.model, token_type="output").inc(
            rec.output_tokens
        )
        LLM_COST_USD_TOTAL.labels(model=rec.model).inc(rec.cost_usd)
        logger.info(
            "llm_call_recorded",
            extra={
                "model": rec.model,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "cost_usd": rec.cost_usd,
                "correlation_id": rec.correlation_id,
            },
        )

    @property
    def records(self) -> list[LLMCallRecord]:
        return list(self._records)

    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self._records), 8)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._records)

    def cost_by_model(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for rec in self._records:
            result[rec.model] = round(result.get(rec.model, 0.0) + rec.cost_usd, 8)
        return result

    def token_usage_by_model(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for rec in self._records:
            if rec.model not in result:
                result[rec.model] = {"input": 0, "output": 0}
            result[rec.model]["input"] += rec.input_tokens
            result[rec.model]["output"] += rec.output_tokens
        return result

    def clear(self) -> None:
        """Reset in-memory records (does not reset Prometheus counters)."""
        self._records.clear()
```

---

## observability/slo.py

```python
"""
SLO compliance tracking and Prometheus alerting rules generator.

SLOConfig defines targets.
RequestRecord captures per-request measurement.
SLOReport summarizes compliance over a window.
compute_slo_compliance() processes a list of RequestRecords.
generate_alert_rules() outputs Prometheus YAML alert rules.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class SLOConfig(BaseModel):
    """SLO targets for a service endpoint or globally."""

    availability_target: float = Field(
        default=0.999,
        ge=0.0,
        le=1.0,
        description="Fraction of requests that must succeed (non-5xx)",
    )
    p99_latency_target_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="99th-percentile latency threshold",
    )
    window_seconds: int = Field(
        default=3600,
        gt=0,
        description="Measurement window in seconds",
    )

    @field_validator("availability_target")
    @classmethod
    def validate_availability(cls, v: float) -> float:
        if v < 0.9:
            raise ValueError("availability_target below 0.9 is unusual — check units")
        return v

    @property
    def error_budget_fraction(self) -> float:
        return 1.0 - self.availability_target

    def monthly_error_budget_minutes(self) -> float:
        return round(30 * 24 * 60 * self.error_budget_fraction, 2)


class RequestRecord(BaseModel):
    """Single request measurement."""

    duration_seconds: float = Field(ge=0.0)
    status_code: int = Field(ge=100, le=599)
    endpoint: str
    model: str | None = None

    @property
    def is_success(self) -> bool:
        return self.status_code < 500

    @property
    def is_error(self) -> bool:
        return self.status_code >= 500


class SLOReport(BaseModel):
    """Summary of SLO compliance over a measurement window."""

    total_requests: int
    successful_requests: int
    error_requests: int
    availability: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float
    slo_availability_met: bool
    slo_latency_met: bool
    slo_met: bool
    error_budget_consumed: float = Field(
        description="Fraction of error budget consumed (0.0–1.0+)"
    )
    config: SLOConfig


# ---------------------------------------------------------------------------
# compute_slo_compliance
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Calculate p-th percentile from pre-sorted list."""
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(len(sorted_values) * p) - 1)
    return sorted_values[idx]


def compute_slo_compliance(
    records: list[RequestRecord],
    config: SLOConfig | None = None,
) -> SLOReport:
    """
    Compute SLO compliance from a list of RequestRecords.

    Args:
        records: Request measurements from the observation window.
        config: SLO targets; defaults to SLOConfig() if not provided.

    Returns:
        SLOReport with availability, latency percentiles, and compliance flags.
    """
    cfg = config or SLOConfig()

    if not records:
        return SLOReport(
            total_requests=0,
            successful_requests=0,
            error_requests=0,
            availability=1.0,
            p50_latency_seconds=0.0,
            p95_latency_seconds=0.0,
            p99_latency_seconds=0.0,
            slo_availability_met=True,
            slo_latency_met=True,
            slo_met=True,
            error_budget_consumed=0.0,
            config=cfg,
        )

    successful = [r for r in records if r.is_success]
    errors = [r for r in records if r.is_error]
    availability = len(successful) / len(records)

    sorted_durations = sorted(r.duration_seconds for r in records)
    p50 = _percentile(sorted_durations, 0.50)
    p95 = _percentile(sorted_durations, 0.95)
    p99 = _percentile(sorted_durations, 0.99)

    slo_availability_met = availability >= cfg.availability_target
    slo_latency_met = p99 <= cfg.p99_latency_target_seconds

    # Error budget consumed: how much of allowed failure fraction was used
    error_rate = 1.0 - availability
    budget_fraction = cfg.error_budget_fraction
    error_budget_consumed = (
        error_rate / budget_fraction if budget_fraction > 0 else 0.0
    )

    return SLOReport(
        total_requests=len(records),
        successful_requests=len(successful),
        error_requests=len(errors),
        availability=round(availability, 6),
        p50_latency_seconds=round(p50, 4),
        p95_latency_seconds=round(p95, 4),
        p99_latency_seconds=round(p99, 4),
        slo_availability_met=slo_availability_met,
        slo_latency_met=slo_latency_met,
        slo_met=slo_availability_met and slo_latency_met,
        error_budget_consumed=round(error_budget_consumed, 4),
        config=cfg,
    )


# ---------------------------------------------------------------------------
# generate_alert_rules — Prometheus YAML
# ---------------------------------------------------------------------------

def generate_alert_rules(config: SLOConfig | None = None) -> str:
    """
    Generate Prometheus alerting rules YAML for the given SLO config.

    Implements multi-window burn rate alerts per the Google SRE workbook:
    - Critical (page): 14x burn over 1h + 5m window
    - Warning (ticket): 6x burn over 6h + 30m window
    - Latency: p99 exceeds target over 5m window
    """
    cfg = config or SLOConfig()
    avail = cfg.availability_target
    error_budget = cfg.error_budget_fraction
    # critical burn = 14x; budget burned in 1h if sustained
    critical_threshold = 14 * error_budget
    # warning burn = 6x; budget burned in ~7h if sustained
    warning_threshold = 6 * error_budget
    latency = cfg.p99_latency_target_seconds

    return f"""# Auto-generated Prometheus alert rules
# SLO: availability={avail*100:.2f}%, p99<={latency}s
# Monthly error budget: {cfg.monthly_error_budget_minutes()} minutes

groups:
  - name: llm_gateway_slo
    rules:

      # -----------------------------------------------------------------------
      # Availability SLO — Multi-window burn rate
      # -----------------------------------------------------------------------

      - alert: LLMGatewayHighErrorBurnRate
        expr: |
          (
            rate(http_requests_total{{status_code=~"5.."}}[1h])
            / rate(http_requests_total[1h])
          ) > {critical_threshold:.6f}
          and
          (
            rate(http_requests_total{{status_code=~"5.."}}[5m])
            / rate(http_requests_total[5m])
          ) > {critical_threshold:.6f}
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "LLM Gateway error burn rate critical (14x SLO budget)"
          description: >
            Error rate is burning the error budget 14x faster than allowed.
            At this rate, the monthly error budget will be exhausted in < 2 hours.
          runbook: "https://runbooks.internal/llm-gateway/high-error-rate"

      - alert: LLMGatewayElevatedErrorBurnRate
        expr: |
          (
            rate(http_requests_total{{status_code=~"5.."}}[6h])
            / rate(http_requests_total[6h])
          ) > {warning_threshold:.6f}
          and
          (
            rate(http_requests_total{{status_code=~"5.."}}[30m])
            / rate(http_requests_total[30m])
          ) > {warning_threshold:.6f}
        for: 15m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "LLM Gateway error burn rate elevated (6x SLO budget)"
          description: >
            Error rate is burning the error budget 6x faster than allowed.
            Error budget will be exhausted in < 1 day if sustained.
          runbook: "https://runbooks.internal/llm-gateway/high-error-rate"

      # -----------------------------------------------------------------------
      # Latency SLO
      # -----------------------------------------------------------------------

      - alert: LLMGatewayHighP99Latency
        expr: |
          histogram_quantile(
            0.99,
            rate(http_request_duration_seconds_bucket{{endpoint="/chat"}}[5m])
          ) > {latency}
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "LLM Gateway /chat p99 latency exceeds SLO ({latency}s)"
          description: >
            99th-percentile latency for /chat has been above {latency}s for 5 minutes.
          runbook: "https://runbooks.internal/llm-gateway/high-latency"

      # -----------------------------------------------------------------------
      # Cost / Token Anomaly
      # -----------------------------------------------------------------------

      - alert: LLMTokenSpike
        expr: |
          rate(llm_tokens_total[5m]) > 10000
        for: 3m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "LLM token consumption spike detected"
          description: >
            Token consumption rate exceeds 10k tokens/second for 3+ minutes.
            Possible prompt injection, runaway loop, or traffic surge.
          runbook: "https://runbooks.internal/llm-gateway/token-spike"

      # -----------------------------------------------------------------------
      # Availability: no traffic (dead man's switch)
      # -----------------------------------------------------------------------

      - alert: LLMGatewayNoTraffic
        expr: |
          rate(http_requests_total{{endpoint="/chat"}}[5m]) == 0
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "LLM Gateway /chat receiving no traffic for 10+ minutes"
          description: >
            Either the service is down or all traffic has been routed away.
          runbook: "https://runbooks.internal/llm-gateway/no-traffic"
"""
```

---

## observability/dashboard.py

```python
"""
Dashboard specification builder.

Produces a structured DashboardSpec that can be serialized to JSON
and imported into Grafana (via the HTTP API or provisioning).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PanelType(str, Enum):
    TIMESERIES = "timeseries"
    STAT = "stat"
    GAUGE = "gauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    TEXT = "text"


class GridPos(BaseModel):
    x: int = 0
    y: int = 0
    w: int = 12
    h: int = 8


class DashboardPanel(BaseModel):
    """Single Grafana panel definition."""

    title: str
    panel_type: PanelType
    description: str = ""
    promql: str = ""
    unit: str = "short"
    grid_pos: GridPos = Field(default_factory=GridPos)
    thresholds: list[dict[str, Any]] = Field(default_factory=list)
    legend_display_mode: str = "list"

    def to_grafana_dict(self) -> dict[str, Any]:
        """Serialize to Grafana panel JSON schema."""
        panel: dict[str, Any] = {
            "title": self.title,
            "type": self.panel_type.value,
            "description": self.description,
            "gridPos": self.grid_pos.model_dump(),
            "fieldConfig": {
                "defaults": {
                    "unit": self.unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": self.thresholds or [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80},
                        ],
                    },
                }
            },
            "options": {"legend": {"displayMode": self.legend_display_mode}},
        }
        if self.promql:
            panel["targets"] = [{"expr": self.promql, "legendFormat": "{{model}}"}]
        return panel


class DashboardSpec(BaseModel):
    """Complete Grafana dashboard specification."""

    title: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    refresh_interval: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"
    panels: list[DashboardPanel] = Field(default_factory=list)

    def to_grafana_dict(self) -> dict[str, Any]:
        """Serialize to full Grafana dashboard JSON."""
        panel_dicts = []
        for i, panel in enumerate(self.panels):
            d = panel.to_grafana_dict()
            d["id"] = i + 1
            panel_dicts.append(d)

        return {
            "dashboard": {
                "title": self.title,
                "description": self.description,
                "tags": self.tags,
                "refresh": self.refresh_interval,
                "time": {"from": self.time_from, "to": self.time_to},
                "panels": panel_dicts,
                "schemaVersion": 38,
            },
            "overwrite": True,
        }


# ---------------------------------------------------------------------------
# Dashboard Builders
# ---------------------------------------------------------------------------

def build_red_dashboard(service_name: str = "llm-gateway") -> DashboardSpec:
    """
    Build a Grafana dashboard covering RED metrics + LLM-specific panels.

    Panels:
      Row 1 (RED):
        - Request Rate (timeseries)
        - Error Rate (timeseries)
        - p99 Latency (timeseries)
        - Active Requests (stat)
      Row 2 (LLM):
        - Token Consumption Rate by Model (timeseries)
        - Estimated Cost/hr by Model (timeseries)
        - LLM Request Duration p99 (timeseries)
      Row 3 (SLO):
        - Availability (gauge)
        - Error Budget Remaining (gauge)
    """
    panels: list[DashboardPanel] = [
        # --- RED Row ---
        DashboardPanel(
            title="Request Rate",
            panel_type=PanelType.TIMESERIES,
            description="Requests per second across all endpoints",
            promql='rate(http_requests_total{job="' + service_name + '"}[5m])',
            unit="reqps",
            grid_pos=GridPos(x=0, y=0, w=8, h=8),
        ),
        DashboardPanel(
            title="Error Rate",
            panel_type=PanelType.TIMESERIES,
            description="Fraction of requests returning 5xx status",
            promql=(
                'rate(http_requests_total{job="' + service_name + '",status_code=~"5.."}[5m])'
                ' / rate(http_requests_total{job="' + service_name + '"}[5m])'
            ),
            unit="percentunit",
            grid_pos=GridPos(x=8, y=0, w=8, h=8),
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 0.001},
                {"color": "red", "value": 0.01},
            ],
        ),
        DashboardPanel(
            title="p99 Latency (/chat)",
            panel_type=PanelType.TIMESERIES,
            description="99th-percentile request duration for /chat",
            promql=(
                'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket'
                '{job="' + service_name + '",endpoint="/chat"}[5m]))'
            ),
            unit="s",
            grid_pos=GridPos(x=16, y=0, w=8, h=8),
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 1.0},
                {"color": "red", "value": 2.0},
            ],
        ),
        DashboardPanel(
            title="Active Requests",
            panel_type=PanelType.STAT,
            description="Requests currently in-flight",
            promql='active_requests{job="' + service_name + '"}',
            unit="short",
            grid_pos=GridPos(x=0, y=8, w=4, h=4),
        ),

        # --- LLM Row ---
        DashboardPanel(
            title="Token Consumption Rate by Model",
            panel_type=PanelType.TIMESERIES,
            description="Tokens/second consumed, split by model and type",
            promql='rate(llm_tokens_total{job="' + service_name + '"}[5m])',
            unit="short",
            grid_pos=GridPos(x=0, y=12, w=12, h=8),
        ),
        DashboardPanel(
            title="Estimated Cost / hour by Model",
            panel_type=PanelType.TIMESERIES,
            description="Projected LLM spend per hour from current token rate",
            promql=(
                'rate(llm_cost_usd_total{job="' + service_name + '"}[5m]) * 3600'
            ),
            unit="currencyUSD",
            grid_pos=GridPos(x=12, y=12, w=12, h=8),
        ),
        DashboardPanel(
            title="LLM Request Duration p99 by Model",
            panel_type=PanelType.TIMESERIES,
            description="99th-percentile latency of LLM API calls by model",
            promql=(
                "histogram_quantile(0.99, "
                'rate(llm_request_duration_seconds_bucket{job="' + service_name + '"}[5m]))'
            ),
            unit="s",
            grid_pos=GridPos(x=0, y=20, w=24, h=8),
        ),

        # --- SLO Row ---
        DashboardPanel(
            title="Availability (1h window)",
            panel_type=PanelType.GAUGE,
            description="Fraction of non-5xx requests over the past hour",
            promql=(
                '1 - (rate(http_requests_total{job="' + service_name + '",status_code=~"5.."}[1h])'
                ' / rate(http_requests_total{job="' + service_name + '"}[1h]))'
            ),
            unit="percentunit",
            grid_pos=GridPos(x=0, y=28, w=12, h=8),
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 0.99},
                {"color": "green", "value": 0.999},
            ],
        ),
        DashboardPanel(
            title="Error Budget Remaining (monthly)",
            panel_type=PanelType.GAUGE,
            description="Fraction of monthly error budget not yet consumed (SLO=99.9%)",
            promql=(
                "1 - ("
                'rate(http_requests_total{job="' + service_name + '",status_code=~"5.."}[30d])'
                ' / rate(http_requests_total{job="' + service_name + '"}[30d])'
                ") / 0.001"
            ),
            unit="percentunit",
            grid_pos=GridPos(x=12, y=28, w=12, h=8),
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 0.25},
                {"color": "green", "value": 0.5},
            ],
        ),
    ]

    return DashboardSpec(
        title=f"{service_name} — Observability",
        description="RED metrics + LLM cost tracking + SLO dashboard",
        tags=["llm", "observability", "slo", service_name],
        refresh_interval="30s",
        panels=panels,
    )
```

---

## gateway.py

```python
"""
LLM Gateway — production-ready FastAPI application.

Integrates all six observability phases:
  1. Structured JSON logging + correlation ID
  2. Prometheus RED metrics
  3. OpenTelemetry tracing
  4. LLM cost tracking
  5. SLO compliance
  6. Dashboard spec

Run:
    uvicorn gateway:app --host 0.0.0.0 --port 8000

Environment variables:
    ANTHROPIC_API_KEY  (required)
    OTEL_EXPORTER_OTLP_ENDPOINT  (optional; defaults to console exporter)
    LOG_LEVEL          (optional; defaults to INFO)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace
from pydantic import BaseModel

from observability.cost import COST_TABLE, SUPPORTED_MODELS, CostTracker, LLMCallRecord
from observability.logging import (
    configure_root_logging,
    get_logger,
    get_or_create_correlation_id,
    set_correlation_id,
)
from observability.metrics import (
    ACTIVE_REQUESTS,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    LLM_REQUEST_DURATION,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from observability.slo import (
    RequestRecord,
    SLOConfig,
    compute_slo_compliance,
    generate_alert_rules,
)
from observability.tracing import build_tracer_provider, get_tracer

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

configure_root_logging()
logger = get_logger("gateway")

_tracer_provider = build_tracer_provider("llm-gateway")
tracer = get_tracer("gateway")

cost_tracker = CostTracker()
_request_records: list[RequestRecord] = []
_slo_config = SLOConfig()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Gateway",
    description="Observability-instrumented LLM proxy",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def observability_middleware(request: Request, call_next: Any) -> Response:
    """
    Unified middleware applying all observability concerns in one pass:
      - Correlation ID injection
      - Prometheus RED metrics
      - Request logging
    """
    # Correlation ID
    cid = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    set_correlation_id(cid)

    # Skip /metrics self-instrumentation
    is_metrics = request.url.path == "/metrics"

    start = time.perf_counter()
    if not is_metrics:
        ACTIVE_REQUESTS.inc()

    logger.info(
        "request_start",
        extra={"method": request.method, "path": request.url.path},
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("unhandled_exception", extra={"error": str(exc)})
        if not is_metrics:
            ACTIVE_REQUESTS.dec()
        raise

    duration = time.perf_counter() - start

    if not is_metrics:
        ACTIVE_REQUESTS.dec()
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        # Accumulate for SLO tracking
        _request_records.append(
            RequestRecord(
                duration_seconds=duration,
                status_code=response.status_code,
                endpoint=request.url.path,
            )
        )

    logger.info(
        "request_end",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_seconds": round(duration, 4),
        },
    )

    response.headers["x-correlation-id"] = cid
    return response


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    model: str = "claude-haiku-5-5"
    max_tokens: int = 1024


class ChatResponse(BaseModel):
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    correlation_id: str


class HealthResponse(BaseModel):
    status: str
    total_requests: int
    total_cost_usd: float
    correlation_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | Response:
    """
    Send a message to an Anthropic LLM and return the response.

    Instruments:
    - OTel span with model + token attributes
    - LLM call record (cost tracking + Prometheus counters)
    - LLM request duration histogram
    """
    if request.model not in SUPPORTED_MODELS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Unsupported model: {request.model!r}",
                "supported_models": sorted(SUPPORTED_MODELS),
            },
        )

    with tracer.start_as_current_span("llm.chat.completion") as span:
        span.set_attribute("llm.model", request.model)
        span.set_attribute("llm.max_tokens", request.max_tokens)
        span.set_attribute("correlation_id", get_or_create_correlation_id())

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        llm_start = time.perf_counter()

        try:
            api_response = client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.message}],
            )
        except anthropic.APIError as exc:
            span.set_status(trace.StatusCode.ERROR, str(exc))
            logger.exception("llm_api_error", extra={"model": request.model, "error": str(exc)})
            return JSONResponse(
                status_code=502,
                content={"error": "Upstream LLM API error", "detail": str(exc)},
            )
        finally:
            llm_duration = time.perf_counter() - llm_start
            LLM_REQUEST_DURATION.labels(model=request.model).observe(llm_duration)

        reply_text = api_response.content[0].text
        input_tokens = api_response.usage.input_tokens
        output_tokens = api_response.usage.output_tokens

        span.set_attribute("llm.input_tokens", input_tokens)
        span.set_attribute("llm.output_tokens", output_tokens)

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


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe with cost summary."""
    return HealthResponse(
        status="healthy",
        total_requests=len(_request_records),
        total_cost_usd=cost_tracker.total_cost(),
        correlation_id=get_or_create_correlation_id(),
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/slo/report")
async def slo_report() -> dict[str, Any]:
    """Current SLO compliance report over all recorded requests."""
    report = compute_slo_compliance(_request_records, _slo_config)
    return report.model_dump()


@app.get("/slo/alert-rules")
async def slo_alert_rules() -> Response:
    """Prometheus YAML alert rules for current SLO config."""
    return Response(
        content=generate_alert_rules(_slo_config),
        media_type="text/plain",
    )


@app.get("/cost/report")
async def cost_report() -> dict[str, Any]:
    """Cost breakdown by model."""
    return {
        "total_cost_usd": cost_tracker.total_cost(),
        "total_tokens": cost_tracker.total_tokens(),
        "by_model": cost_tracker.cost_by_model(),
        "token_usage_by_model": cost_tracker.token_usage_by_model(),
    }
```

---

## tests/conftest.py

```python
"""
Shared pytest fixtures and mock helpers.
All LLM calls are mocked — no real Anthropic API calls in tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def make_mock_anthropic_response(
    reply_text: str = "Mock LLM response",
    input_tokens: int = 42,
    output_tokens: int = 18,
) -> MagicMock:
    """Build a mock anthropic.Message response."""
    mock_content = MagicMock()
    mock_content.text = reply_text

    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage = mock_usage
    return mock_response


@pytest.fixture()
def mock_llm_response() -> MagicMock:
    return make_mock_anthropic_response()


@pytest.fixture()
def patched_anthropic(mock_llm_response: MagicMock):
    """Patch anthropic.Anthropic class for all tests using this fixture."""
    with patch("anthropic.Anthropic") as mock_cls:
        instance = MagicMock()
        instance.messages.create.return_value = mock_llm_response
        mock_cls.return_value = instance
        yield mock_cls


@pytest.fixture()
def gateway_client(patched_anthropic) -> TestClient:
    """FastAPI test client with LLM mocked."""
    from gateway import app
    with TestClient(app) as client:
        yield client
```

---

## tests/test_logging.py

```python
"""Tests for observability/logging.py"""

from __future__ import annotations

import json
import logging

import pytest

from observability.logging import (
    JSONFormatter,
    correlation_id,
    get_or_create_correlation_id,
    set_correlation_id,
)


class TestJSONFormatter:
    def _make_record(
        self, msg: str = "test", level: int = logging.INFO
    ) -> logging.LogRecord:
        return logging.LogRecord(
            name="test", level=level, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None,
        )

    def test_required_fields_present(self) -> None:
        set_correlation_id("abc-123")
        formatter = JSONFormatter()
        output = json.loads(formatter.format(self._make_record("hello")))

        assert output["message"] == "hello"
        assert output["level"] == "INFO"
        assert output["correlation_id"] == "abc-123"
        assert "timestamp" in output
        assert "logger" in output

    def test_no_correlation_id_defaults_to_none_string(self) -> None:
        set_correlation_id("")
        formatter = JSONFormatter()
        output = json.loads(formatter.format(self._make_record()))
        assert output["correlation_id"] == "none"

    def test_extra_fields_merged(self) -> None:
        set_correlation_id("cid")
        formatter = JSONFormatter()
        record = self._make_record("msg")
        record.__dict__["model"] = "claude-haiku-5-5"
        output = json.loads(formatter.format(record))
        assert output["model"] == "claude-haiku-5-5"

    def test_exception_info_serialized(self) -> None:
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = self._make_record("error")
            record.exc_info = sys.exc_info()
        output = json.loads(formatter.format(record))
        assert "exception" in output
        assert "ValueError" in output["exception"]

    def test_warning_level(self) -> None:
        formatter = JSONFormatter()
        output = json.loads(formatter.format(self._make_record("w", logging.WARNING)))
        assert output["level"] == "WARNING"


class TestCorrelationId:
    def test_set_and_get(self) -> None:
        set_correlation_id("test-uuid")
        assert correlation_id.get() == "test-uuid"

    def test_get_or_create_generates_uuid_when_empty(self) -> None:
        set_correlation_id("")
        cid = get_or_create_correlation_id()
        assert len(cid) == 36  # UUID4 format
        assert cid.count("-") == 4

    def test_get_or_create_reuses_existing(self) -> None:
        set_correlation_id("existing-id")
        assert get_or_create_correlation_id() == "existing-id"
```

---

## tests/test_cost.py

```python
"""Tests for observability/cost.py"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from observability.cost import COST_TABLE, CostTracker, LLMCallRecord, SUPPORTED_MODELS


class TestLLMCallRecord:
    @pytest.mark.parametrize("model,input_t,output_t,expected_cost", [
        ("claude-opus-5-5", 1_000_000, 0, 15.00),
        ("claude-opus-5-5", 0, 1_000_000, 75.00),
        ("claude-sonnet-5-5", 1_000_000, 0, 3.00),
        ("claude-sonnet-5-5", 0, 1_000_000, 15.00),
        ("claude-haiku-5-5", 1_000_000, 0, 0.25),
        ("claude-haiku-5-5", 0, 1_000_000, 1.25),
        ("claude-haiku-5-5", 0, 0, 0.0),
    ])
    def test_cost_formula(
        self, model: str, input_t: int, output_t: int, expected_cost: float
    ) -> None:
        rec = LLMCallRecord(model=model, input_tokens=input_t, output_tokens=output_t)
        assert abs(rec.cost_usd - expected_cost) < 1e-6

    def test_total_tokens_property(self) -> None:
        rec = LLMCallRecord(model="claude-haiku-5-5", input_tokens=100, output_tokens=50)
        assert rec.total_tokens == 150

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMCallRecord(model="claude-haiku-5-5", input_tokens=-1, output_tokens=0)

    def test_unknown_model_zero_cost(self) -> None:
        rec = LLMCallRecord(model="unknown-model", input_tokens=1_000_000, output_tokens=1_000_000)
        assert rec.cost_usd == 0.0

    def test_record_is_immutable(self) -> None:
        rec = LLMCallRecord(model="claude-haiku-5-5", input_tokens=100, output_tokens=50)
        with pytest.raises(Exception):
            rec.model = "claude-opus-5-5"  # type: ignore[misc]


class TestCostTracker:
    def test_empty_tracker(self) -> None:
        tracker = CostTracker()
        assert tracker.records == []
        assert tracker.total_cost() == 0.0
        assert tracker.total_tokens() == 0

    def test_record_and_accumulate(self) -> None:
        tracker = CostTracker()
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=1_000_000, output_tokens=0))
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=0, output_tokens=1_000_000))
        assert len(tracker.records) == 2
        assert abs(tracker.total_cost() - 1.50) < 1e-6

    def test_cost_by_model_breakdown(self) -> None:
        tracker = CostTracker()
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=1_000_000, output_tokens=0))
        tracker.record(LLMCallRecord(model="claude-opus-5-5", input_tokens=1_000_000, output_tokens=0))
        breakdown = tracker.cost_by_model()
        assert abs(breakdown["claude-haiku-5-5"] - 0.25) < 1e-6
        assert abs(breakdown["claude-opus-5-5"] - 15.00) < 1e-6

    def test_token_usage_by_model(self) -> None:
        tracker = CostTracker()
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=100, output_tokens=50))
        usage = tracker.token_usage_by_model()
        assert usage["claude-haiku-5-5"]["input"] == 100
        assert usage["claude-haiku-5-5"]["output"] == 50

    def test_clear_resets_records(self) -> None:
        tracker = CostTracker()
        tracker.record(LLMCallRecord(model="claude-haiku-5-5", input_tokens=100, output_tokens=0))
        tracker.clear()
        assert tracker.records == []
        assert tracker.total_cost() == 0.0
```

---

## tests/test_slo.py

```python
"""Tests for observability/slo.py"""

from __future__ import annotations

import pytest

from observability.slo import (
    RequestRecord,
    SLOConfig,
    SLOReport,
    compute_slo_compliance,
    generate_alert_rules,
)


def make_records(
    n_success: int,
    n_error: int = 0,
    success_duration: float = 0.1,
    error_duration: float = 0.1,
) -> list[RequestRecord]:
    records = [
        RequestRecord(duration_seconds=success_duration, status_code=200, endpoint="/chat")
        for _ in range(n_success)
    ]
    records += [
        RequestRecord(duration_seconds=error_duration, status_code=500, endpoint="/chat")
        for _ in range(n_error)
    ]
    return records


class TestSLOConfig:
    def test_default_config(self) -> None:
        cfg = SLOConfig()
        assert cfg.availability_target == 0.999
        assert cfg.p99_latency_target_seconds == 2.0

    def test_monthly_error_budget_minutes(self) -> None:
        cfg = SLOConfig(availability_target=0.999)
        budget = cfg.monthly_error_budget_minutes()
        assert abs(budget - 43.2) < 0.1

    def test_error_budget_fraction(self) -> None:
        cfg = SLOConfig(availability_target=0.999)
        assert abs(cfg.error_budget_fraction - 0.001) < 1e-9

    def test_invalid_availability_below_0_9_raises(self) -> None:
        with pytest.raises(Exception):
            SLOConfig(availability_target=0.5)


class TestComputeSLOCompliance:
    def test_empty_records_slo_met(self) -> None:
        report = compute_slo_compliance([])
        assert report.slo_met is True
        assert report.total_requests == 0

    def test_all_success_meets_slo(self) -> None:
        records = make_records(n_success=1000)
        report = compute_slo_compliance(records)
        assert report.slo_met is True
        assert report.availability == 1.0

    def test_high_error_rate_breaks_availability_slo(self) -> None:
        records = make_records(n_success=990, n_error=10)
        report = compute_slo_compliance(records)
        assert report.slo_availability_met is False
        assert report.slo_met is False

    def test_high_latency_breaks_latency_slo(self) -> None:
        records = (
            make_records(n_success=990, success_duration=0.1)
            + make_records(n_success=10, success_duration=5.0)
        )
        cfg = SLOConfig(p99_latency_target_seconds=1.0)
        report = compute_slo_compliance(records, cfg)
        assert report.slo_latency_met is False
        assert report.slo_met is False

    def test_availability_below_target_non_zero_error_budget_consumed(self) -> None:
        records = make_records(n_success=990, n_error=10)
        cfg = SLOConfig(availability_target=0.999)
        report = compute_slo_compliance(records, cfg)
        assert report.error_budget_consumed > 1.0  # budget exceeded

    def test_p99_latency_calculation(self) -> None:
        # 99 fast + 1 slow → p99 should be the slow one
        records = (
            [RequestRecord(duration_seconds=0.1, status_code=200, endpoint="/chat")] * 99
            + [RequestRecord(duration_seconds=9.9, status_code=200, endpoint="/chat")]
        )
        report = compute_slo_compliance(records)
        assert report.p99_latency_seconds == pytest.approx(9.9, abs=0.01)

    def test_single_request(self) -> None:
        records = [RequestRecord(duration_seconds=1.0, status_code=200, endpoint="/chat")]
        report = compute_slo_compliance(records)
        assert report.total_requests == 1
        assert report.successful_requests == 1


class TestGenerateAlertRules:
    def test_returns_valid_yaml_string(self) -> None:
        rules = generate_alert_rules()
        assert "groups:" in rules
        assert "LLMGatewayHighErrorBurnRate" in rules
        assert "LLMGatewayElevatedErrorBurnRate" in rules
        assert "LLMGatewayHighP99Latency" in rules

    def test_thresholds_reflect_config(self) -> None:
        cfg = SLOConfig(availability_target=0.995, p99_latency_target_seconds=5.0)
        rules = generate_alert_rules(cfg)
        assert "5.0" in rules  # latency target


class TestRequestRecord:
    def test_is_success_for_2xx(self) -> None:
        r = RequestRecord(duration_seconds=0.1, status_code=200, endpoint="/chat")
        assert r.is_success is True
        assert r.is_error is False

    def test_is_error_for_5xx(self) -> None:
        r = RequestRecord(duration_seconds=0.1, status_code=503, endpoint="/chat")
        assert r.is_error is True
        assert r.is_success is False

    def test_4xx_counts_as_success_for_slo(self) -> None:
        # 4xx = client error, not service failure — success for availability SLO
        r = RequestRecord(duration_seconds=0.1, status_code=404, endpoint="/chat")
        assert r.is_success is True
```

---

## tests/test_dashboard.py

```python
"""Tests for observability/dashboard.py"""

from __future__ import annotations

import pytest

from observability.dashboard import (
    DashboardPanel,
    DashboardSpec,
    GridPos,
    PanelType,
    build_red_dashboard,
)


class TestDashboardPanel:
    def test_to_grafana_dict_structure(self) -> None:
        panel = DashboardPanel(
            title="Test Panel",
            panel_type=PanelType.TIMESERIES,
            promql='rate(http_requests_total[5m])',
            unit="reqps",
        )
        d = panel.to_grafana_dict()
        assert d["title"] == "Test Panel"
        assert d["type"] == "timeseries"
        assert len(d["targets"]) == 1
        assert d["targets"][0]["expr"] == 'rate(http_requests_total[5m])'

    def test_panel_without_promql_has_no_targets(self) -> None:
        panel = DashboardPanel(title="Text Panel", panel_type=PanelType.TEXT)
        d = panel.to_grafana_dict()
        assert "targets" not in d

    def test_custom_thresholds_passed_through(self) -> None:
        panel = DashboardPanel(
            title="T",
            panel_type=PanelType.GAUGE,
            thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 0.5}],
        )
        d = panel.to_grafana_dict()
        steps = d["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert steps[1]["color"] == "green"


class TestDashboardSpec:
    def test_to_grafana_dict_assigns_panel_ids(self) -> None:
        panels = [
            DashboardPanel(title=f"Panel {i}", panel_type=PanelType.TIMESERIES)
            for i in range(3)
        ]
        spec = DashboardSpec(title="Test Dashboard", panels=panels)
        d = spec.to_grafana_dict()
        ids = [p["id"] for p in d["dashboard"]["panels"]]
        assert ids == [1, 2, 3]

    def test_metadata_fields(self) -> None:
        spec = DashboardSpec(
            title="My Dashboard",
            tags=["llm", "prod"],
            refresh_interval="1m",
        )
        d = spec.to_grafana_dict()
        assert d["dashboard"]["title"] == "My Dashboard"
        assert d["dashboard"]["tags"] == ["llm", "prod"]
        assert d["dashboard"]["refresh"] == "1m"


class TestBuildREDDashboard:
    def test_returns_dashboard_spec(self) -> None:
        spec = build_red_dashboard("llm-gateway")
        assert isinstance(spec, DashboardSpec)

    def test_has_required_panels(self) -> None:
        spec = build_red_dashboard()
        titles = [p.title for p in spec.panels]
        assert any("Request Rate" in t for t in titles)
        assert any("Error Rate" in t for t in titles)
        assert any("Latency" in t for t in titles)
        assert any("Cost" in t for t in titles)
        assert any("Availability" in t for t in titles)

    def test_all_panels_have_grid_pos(self) -> None:
        spec = build_red_dashboard()
        for panel in spec.panels:
            assert panel.grid_pos is not None
            assert panel.grid_pos.w > 0
            assert panel.grid_pos.h > 0

    def test_service_name_in_promql(self) -> None:
        spec = build_red_dashboard("my-service")
        for panel in spec.panels:
            if panel.promql:
                assert "my-service" in panel.promql
```

---

## tests/test_gateway.py

```python
"""
Integration tests for gateway.py endpoints.
All LLM calls are mocked via conftest.py fixtures.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestChatEndpoint:
    def test_chat_success_with_mocked_llm(self, gateway_client, patched_anthropic) -> None:
        patched_anthropic.return_value.messages.create.return_value.content[0].text = "Paris"
        patched_anthropic.return_value.messages.create.return_value.usage.input_tokens = 10
        patched_anthropic.return_value.messages.create.return_value.usage.output_tokens = 5

        response = gateway_client.post(
            "/chat",
            json={"message": "What is the capital of France?", "model": "claude-haiku-5-5"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert data["model"] == "claude-haiku-5-5"
        assert data["input_tokens"] == 10
        assert data["output_tokens"] == 5
        assert data["cost_usd"] >= 0.0
        assert "correlation_id" in data

    def test_chat_invalid_model_returns_400(self, gateway_client) -> None:
        response = gateway_client.post(
            "/chat",
            json={"message": "test", "model": "gpt-4o"},
        )
        assert response.status_code == 400
        assert "supported_models" in response.json()

    def test_chat_all_supported_models(self, gateway_client, patched_anthropic) -> None:
        for model in ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]:
            patched_anthropic.return_value.messages.create.return_value.content[0].text = "ok"
            patched_anthropic.return_value.messages.create.return_value.usage.input_tokens = 5
            patched_anthropic.return_value.messages.create.return_value.usage.output_tokens = 3
            response = gateway_client.post(
                "/chat",
                json={"message": "ping", "model": model},
            )
            assert response.status_code == 200, f"Failed for {model}"

    def test_correlation_id_echoed_in_response_header(self, gateway_client) -> None:
        response = gateway_client.get(
            "/health",
            headers={"X-Correlation-ID": "test-cid-xyz"},
        )
        assert response.headers.get("x-correlation-id") == "test-cid-xyz"


class TestHealthEndpoint:
    def test_health_returns_200(self, gateway_client) -> None:
        response = gateway_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "total_cost_usd" in data
        assert "correlation_id" in data

    def test_health_reflects_request_count(self, gateway_client) -> None:
        response = gateway_client.get("/health")
        assert response.status_code == 200


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, gateway_client) -> None:
        response = gateway_client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text
        assert "active_requests" in response.text

    def test_metrics_content_type(self, gateway_client) -> None:
        response = gateway_client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]


class TestSLOEndpoints:
    def test_slo_report_returns_dict(self, gateway_client) -> None:
        response = gateway_client.get("/slo/report")
        assert response.status_code == 200
        data = response.json()
        assert "slo_met" in data
        assert "availability" in data

    def test_slo_alert_rules_returns_yaml(self, gateway_client) -> None:
        response = gateway_client.get("/slo/alert-rules")
        assert response.status_code == 200
        assert "groups:" in response.text


class TestCostEndpoint:
    def test_cost_report_structure(self, gateway_client) -> None:
        response = gateway_client.get("/cost/report")
        assert response.status_code == 200
        data = response.json()
        assert "total_cost_usd" in data
        assert "by_model" in data
        assert "token_usage_by_model" in data
```

---

## pyproject.toml

```toml
[tool.poetry]
name = "llm-gateway"
version = "1.0.0"
description = "Observability-instrumented LLM gateway — Week 11 reference implementation"
authors = ["AI Engineering Course"]
python = "^3.12"

[tool.poetry.dependencies]
python = "^3.12"
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.34"}
anthropic = "^0.40"
pydantic = "^2.10"
prometheus-client = "^0.21"
opentelemetry-api = "^1.29"
opentelemetry-sdk = "^1.29"
opentelemetry-exporter-otlp-proto-grpc = {version = "^1.29", optional = true}

[tool.poetry.dev-dependencies]
pytest = "^8.3"
pytest-asyncio = "^0.25"
httpx = "^0.28"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

---

## .env.example

```dotenv
# Anthropic API key — required for /chat endpoint
ANTHROPIC_API_KEY=your_api_key_here

# Optional: OTLP endpoint for exporting traces to Jaeger/Tempo
# OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# Optional: override log level (DEBUG / INFO / WARNING / ERROR)
# LOG_LEVEL=INFO
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single `observability_middleware` in `gateway.py` | Avoids double-counting; one pass through the ASGI stack per request |
| `contextvars.ContextVar` for correlation ID | Async-safe; no parameter threading; works with `asyncio` task isolation |
| `@computed_field` on `LLMCallRecord` | Pydantic v2 idiomatic; cost derived at validation time, immutable after construction |
| `frozen=True` on `LLMCallRecord` | Records are append-only facts; prevents accidental mutation |
| `CostTracker.clear()` resets memory only | Prometheus counters are monotonic — reset would corrupt `rate()` calculations |
| Multi-window burn rate alerts | Single-threshold alerts on raw error rate cause alert fatigue from transient spikes |
| `ConsoleSpanExporter` default | Zero-dependency local dev; swap `OTLPSpanExporter` via env var without code changes |
| `generate_alert_rules()` returns YAML string | Grafana provisioning and Prometheus Operator both accept raw YAML; no extra dep |
