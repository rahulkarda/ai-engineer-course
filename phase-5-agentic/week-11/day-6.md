# Phase 5 · Week 11 · Day 6 — Observability: Tracing, Spans, Token & Cost Tracking

> Objective: Instrument agent runs with traces and spans; track token usage and cost; understand LangSmith and OpenTelemetry integration points.  ·  Time: ~4 hr (Saturday deep practice)

## Concept

Production agents fail silently without observability. You need answers to:
- Which step took the most time / tokens?
- Why did the agent call `search` four times instead of once?
- What was the total cost of this user session?
- Which tool call caused the 30-second latency spike?

**Observability for agents = traces + spans + metrics + logs, all correlated by run ID.**

### Three layers

| Layer | What | How |
|-------|------|-----|
| **Traces** | Full agent run, identified by `run_id` | Start/end time, metadata |
| **Spans** | Individual steps inside a run | Tool calls, model calls, validation |
| **Metrics** | Aggregated signals | Tokens/run, cost/run, latency p95 |

### Key concepts

**Span** — a named, timed unit of work. A span has:
- `name` (e.g., `"tool.search"`)
- `start_time`, `end_time`
- `attributes` (arbitrary key-value metadata)
- `parent_span_id` (enabling the tree structure of a trace)

**Trace** — a tree of spans forming one complete agent run.

**OpenTelemetry (OTel)** — vendor-neutral standard for traces, metrics, and logs. Export to any backend: Jaeger, Honeycomb, LangSmith, Datadog.

**Logfire** — Pydantic's first-party observability layer, built on OTel. Pydantic AI has native Logfire integration: `agent.run()` automatically emits spans when Logfire is configured.

**LangSmith** — LangChain's tracing backend. Supports runs/spans from LangChain, but also accepts traces via OTel OTLP endpoint.

## How it works internally

### Pydantic AI + Logfire

```
pydantic_ai.Agent.run()
    ├─ logfire.span("agent run")
    │       ├─ logfire.span("model call #1")
    │       ├─ logfire.span("tool.search")
    │       └─ logfire.span("model call #2")
    └─ result.usage() → {requests: N, tokens: M}
```

When `logfire.configure()` is called once at startup, all Pydantic AI agent runs emit structured spans automatically — no code changes to the agent required.

### Manual OTel spans (framework-agnostic)

```python
from opentelemetry import trace
tracer = trace.get_tracer("my_agent")

with tracer.start_as_current_span("tool.search") as span:
    span.set_attribute("search.query", query)
    result = search(query)
    span.set_attribute("search.result_count", len(result))
```

## Worked example — custom token/cost tracker

```python
# src/observability/cost_tracker.py
"""Token usage and cost tracking for Pydantic AI agent runs."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai.usage import Usage

logger = logging.getLogger(__name__)


# ── Pricing table (USD per 1M tokens — update as needed) ─────────────────────
_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 15.0, "output": 75.0},
    "claude-sonnet-5-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-5-5": {"input": 0.25, "output": 1.25},
}


def cost_usd(model: str, usage: Usage) -> float:
    """Compute USD cost from token usage for a given model."""
    pricing = _PRICING.get(model)
    if not pricing:
        logger.warning("No pricing data for model: %s — cost unknown", model)
        return 0.0
    input_cost = (usage.request_tokens or 0) / 1_000_000 * pricing["input"]
    output_cost = (usage.response_tokens or 0) / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 6)


# ── Run record ────────────────────────────────────────────────────────────────
class RunRecord(BaseModel):
    run_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    tool_calls: int
    success: bool


# ── Tracker ───────────────────────────────────────────────────────────────────
class CostTracker:
    """Accumulate per-run cost/token data and report aggregates."""

    def __init__(self) -> None:
        self._records: list[RunRecord] = []

    def record(
        self,
        run_id: str,
        model: str,
        usage: Usage,
        latency_ms: float,
        tool_calls: int,
        success: bool,
    ) -> RunRecord:
        record = RunRecord(
            run_id=run_id,
            model=model,
            input_tokens=usage.request_tokens or 0,
            output_tokens=usage.response_tokens or 0,
            total_tokens=(usage.request_tokens or 0) + (usage.response_tokens or 0),
            cost_usd=cost_usd(model, usage),
            latency_ms=latency_ms,
            tool_calls=tool_calls,
            success=success,
        )
        self._records.append(record)
        logger.info(
            "Run %s: tokens=%d cost=$%.4f latency=%.0fms",
            run_id,
            record.total_tokens,
            record.cost_usd,
            record.latency_ms,
        )
        return record

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {}
        total_cost = sum(r.cost_usd for r in self._records)
        total_tokens = sum(r.total_tokens for r in self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / len(self._records)
        return {
            "runs": len(self._records),
            "total_cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "success_rate": sum(r.success for r in self._records) / len(self._records),
        }
```

```python
# src/observability/tracer.py
"""Lightweight span tracer — wraps agent runs with timing and metadata."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

    def finish(self, error: str | None = None) -> None:
        self.end_time = time.perf_counter()
        self.error = error
        logger.debug(
            "Span %s/%s finished in %.1fms%s",
            self.name,
            self.span_id,
            self.duration_ms,
            f" (error: {error})" if error else "",
        )


class Tracer:
    """In-process span collector. Replace with OTel exporter in production."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    @asynccontextmanager
    async def span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: str | None = None,
        **attributes: Any,
    ) -> AsyncGenerator[Span, None]:
        s = Span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=dict(attributes),
        )
        try:
            yield s
            s.finish()
        except Exception as exc:
            s.finish(error=str(exc))
            raise
        finally:
            self._spans.append(s)

    def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def export_json(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
                "error": s.error,
            }
            for s in self._spans
        ]
```

### Wiring it all together

```python
# src/observability/instrumented_runner.py
"""Instrumented agent runner — wraps a Pydantic AI agent with tracing + cost."""
from __future__ import annotations

import time
import uuid
import logging
from typing import Any

from src.observability.cost_tracker import CostTracker
from src.observability.tracer import Tracer

logger = logging.getLogger(__name__)
_tracer = Tracer()
_tracker = CostTracker()


async def run_instrumented(
    agent: Any,
    user_input: str,
    deps: Any,
    *,
    model: str,
) -> Any:
    """Run a Pydantic AI agent and record trace + cost."""
    trace_id = uuid.uuid4().hex
    t0 = time.perf_counter()

    async with _tracer.span("agent.run", trace_id, model=model, input_len=len(user_input)) as root:
        result = await agent.run(user_input, deps=deps)

    latency_ms = (time.perf_counter() - t0) * 1000
    usage = result.usage()

    # Count tool calls from messages
    from pydantic_ai.messages import ToolCallPart
    tool_calls = sum(
        1
        for msg in result.all_messages()
        for part in getattr(msg, "parts", [])
        if isinstance(part, ToolCallPart)
    )

    _tracker.record(
        run_id=trace_id,
        model=model,
        usage=usage,
        latency_ms=latency_ms,
        tool_calls=tool_calls,
        success=True,
    )
    logger.info("Run %s complete. Summary: %s", trace_id, _tracker.summary())
    return result


def get_summary() -> dict[str, Any]:
    return _tracker.summary()
```

### Logfire integration (when available)

```python
# src/main.py
import logfire

# Call once at application startup
logfire.configure(
    service_name="my-agent",
    # token read from LOGFIRE_TOKEN env var automatically
)

# That's it — all pydantic_ai.Agent.run() calls now emit spans automatically
```

### OpenTelemetry export to LangSmith (via OTLP)

```bash
# Set env vars — never hardcode
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.smith.langchain.com/otel"
export OTEL_EXPORTER_OTLP_HEADERS="x-api-key=${LANGSMITH_API_KEY}"
export OTEL_SERVICE_NAME="my-agent"
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
```

## Exercises

1. Extend `CostTracker` with a `budget_alert(max_cost_usd: float) -> bool` method that returns True if total cost exceeds the budget. Write a test.

2. Add a `tool_latency` attribute to spans in `Tracer`. Modify `run_instrumented` to emit a separate span for each tool call (parse tool calls from `result.all_messages()`).

3. Implement `Tracer.waterfall(trace_id) -> str` that returns a text waterfall diagram of the spans for a trace (indent children under parents by depth).

4. Write a pytest fixture `instrumented_agent(agent, model)` that wraps an agent with `run_instrumented`, runs it against 3 test inputs, and asserts `get_summary()["success_rate"] == 1.0`. Use `TestModel` to avoid real API calls.

5. Read the Logfire docs. List the 3 things you must set up to get automatic Pydantic AI tracing. Note which env var holds the API token.

(Solutions: ../solutions/week-11/day-6.md)

## Self-check

- [ ] I can explain traces, spans, and how they relate to an agent run.
- [ ] I can implement a `CostTracker` that computes USD cost from Pydantic AI `Usage`.
- [ ] I can build a `Tracer` that records spans with timing and attributes.
- [ ] I understand how `logfire.configure()` enables automatic Pydantic AI tracing.
- [ ] I know how to configure OTel OTLP export to send traces to an external backend.

## Supplementary links

- [Pydantic AI — Logfire](https://ai.pydantic.dev/logfire/)
- [Logfire docs](https://logfire.pydantic.dev/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [LangSmith OTel integration](https://docs.smith.langchain.com/observability/how_to_guides/tracing/trace_with_opentelemetry)
