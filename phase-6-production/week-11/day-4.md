# Phase 6 · Week 11 · Day 4 — Distributed Tracing with OpenTelemetry

> Objective: Instrument a FastAPI service with OpenTelemetry traces, propagate context across HTTP boundaries, and visualize spans in stdout.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]]

---

## Concept

Traditional logging records *what* happened on a single machine. In a distributed system — where a single user request fans out across API gateways, LLM orchestrators, tool-call services, and databases — logs on each node show fragments. You can't tell which log lines belong to the same user request, or how long each hop took.

**Distributed tracing** solves this by threading a unique identifier (a *trace ID*) through every service involved in a request.

### Core vocabulary

| Term | Meaning |
|---|---|
| **Trace** | The complete journey of one request through all services. Identified by a `trace_id` (16-byte hex). |
| **Span** | One unit of work within a trace (e.g., "parse request", "call LLM", "query DB"). Has a `span_id`, a `parent_span_id`, start time, and duration. |
| **Parent span** | The span that spawned a child span. Forms a tree (DAG in complex cases). |
| **Root span** | The top-level span with no parent — usually the entry point (e.g., HTTP request handler). |
| **Context propagation** | Passing the `trace_id` + `span_id` from caller to callee so the callee's spans attach to the same trace. |

### Why traditional logging fails

- No causal link between log lines across services.
- No timing hierarchy — you see "error in service B" but not that service A triggered it.
- Log correlation requires matching timestamps + request IDs by hand.
- Aggregated latency hides which *step* is slow.

### OpenTelemetry (OTel)

OpenTelemetry is a CNCF vendor-neutral standard for **telemetry signals**: traces, metrics, and logs. It defines:

- **API**: language-agnostic interfaces (`Tracer`, `Span`, `Context`).
- **SDK**: the concrete implementation you configure (exporters, samplers, processors).
- **Exporters**: ship data to Jaeger, Zipkin, OTLP-compatible backends (Grafana Tempo, Honeycomb, Datadog, etc.).

The SDK is plugged into your code; the backend is swappable without code changes.

### W3C Trace-Context header

The W3C `traceparent` HTTP header carries context between services:

```
traceparent: 00-{trace_id}-{parent_span_id}-{flags}
# e.g.
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

- `00` — version
- `4bf9…` — 32-hex trace ID
- `00f0…` — 16-hex parent span ID
- `01` — sampled flag

The `tracestate` and `baggage` headers carry vendor-specific data and user-defined key-value pairs (e.g., `baggage: user_id=42,session=abc`).

### Span anatomy

```
Span {
  name:        "POST /generate"          # human-readable
  trace_id:    "4bf92f3577b34da6a3ce929d0e0e4736"
  span_id:     "00f067aa0ba902b7"
  parent_id:   None  # root span
  kind:        SpanKind.SERVER           # SERVER | CLIENT | INTERNAL | PRODUCER | CONSUMER
  start_time:  1731312000.000
  end_time:    1731312001.234
  attributes:  {"http.method": "POST", "http.route": "/generate", "http.status_code": 200}
  events:      [SpanEvent("llm.called", {"model": "claude-haiku-5-5"})]
  status:      StatusCode.OK
}
```

---

## How it works internally

1. **Bootstrap**: configure a `TracerProvider` with a `SpanProcessor` (e.g., `BatchSpanProcessor`) and an `Exporter` (e.g., `ConsoleSpanExporter` or `OTLPSpanExporter`).
2. **Acquire a tracer**: `tracer = trace.get_tracer("my_service")`. Tracers are cheap — one per module is idiomatic.
3. **Create spans**: `with tracer.start_as_current_span("operation") as span:` — the span is automatically set as the *current* context for anything inside the `with` block.
4. **Child spans**: if you call `tracer.start_as_current_span` inside another span's block, OTel automatically sets the outer span as the parent. No manual wiring needed.
5. **Propagation**: when making an outgoing HTTP call, call `inject(carrier)` to write the `traceparent` header into your request headers dict. On the receiving side, call `extract(carrier)` to restore the context, then start a span with that context as parent.
6. **Export**: the processor batches completed spans and ships them to the exporter.

Context propagation diagram:

```
Service A                          Service B
─────────────────────────────      ──────────────────────────────
[root span: handle_request]   →    traceparent header →
  [child: call_service_b]           [span: handle_request, parent=A]
    inject(headers)                   extract(headers)
                                        [child: query_db]
```

---

## Worked example

Install dependencies (uv):

```bash
uv add opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc fastapi uvicorn httpx
# For console-only (no gRPC collector needed):
uv add opentelemetry-api opentelemetry-sdk fastapi uvicorn httpx
```

```python
# tracing_demo.py
"""
Distributed tracing demo with OpenTelemetry + FastAPI.

Demonstrates:
- TracerProvider + ConsoleSpanExporter setup
- Automatic + manual span creation
- Span attributes, events, and status
- Context propagation via inject/extract across a simulated HTTP call
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── 1. Bootstrap OpenTelemetry ────────────────────────────────────────────────

def build_tracer_provider(service_name: str, use_otlp: bool = False) -> TracerProvider:
    """
    Create a TracerProvider with a ConsoleSpanExporter (dev) or OTLP exporter (prod).

    Args:
        service_name: Identifies this service in trace UIs.
        use_otlp: If True, also export to an OTLP gRPC collector on localhost:4317.
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENV", "development"),
        }
    )
    provider = TracerProvider(resource=resource)

    # Console exporter: prints completed spans as JSON to stdout — useful for dev/tests
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))

    if use_otlp:
        # OTLP exporter sends spans to Jaeger/Tempo/Honeycomb etc.
        otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    return provider


# Register globally so all trace.get_tracer() calls use this provider
provider = build_tracer_provider("llm-gateway")
trace.set_tracer_provider(provider)

# One tracer per logical component
tracer = trace.get_tracer("llm-gateway.api", "1.0.0")


# ── 2. Pydantic models ─────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "claude-haiku-5-5"


class GenerateResponse(BaseModel):
    text: str
    trace_id: str
    latency_ms: float


# ── 3. FastAPI application ─────────────────────────────────────────────────────

app = FastAPI(title="LLM Gateway with OTel tracing")


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, request: Request) -> GenerateResponse:
    """
    Root span: SERVER kind — represents the inbound HTTP request.

    We extract any incoming trace context from HTTP headers so that if
    this service is called from another traced service, spans link correctly.
    """
    # Extract upstream context from incoming headers (enables distributed tracing)
    upstream_ctx = extract(dict(request.headers))

    start = time.perf_counter()

    # start_as_current_span sets this span as the active context for the block
    with tracer.start_as_current_span(
        "POST /generate",
        context=upstream_ctx,
        kind=SpanKind.SERVER,
    ) as root_span:
        # Semantic convention attributes from OTel spec:
        # https://opentelemetry.io/docs/specs/semconv/http/http-spans/
        root_span.set_attribute("http.method", "POST")
        root_span.set_attribute("http.route", "/generate")
        root_span.set_attribute("http.url", str(request.url))
        root_span.set_attribute("llm.model_requested", req.model)

        # Record an event — a timestamped annotation on the span
        root_span.add_event("request.received", {"prompt_length": len(req.prompt)})

        try:
            result_text = await _orchestrate(req.prompt, req.model)

            elapsed_ms = (time.perf_counter() - start) * 1000
            root_span.set_attribute("http.status_code", 200)
            root_span.set_attribute("llm.latency_ms", round(elapsed_ms, 2))
            root_span.set_status(Status(StatusCode.OK))

            # Retrieve the trace ID to return in the response (useful for debugging)
            span_ctx = root_span.get_span_context()
            trace_id_hex = format(span_ctx.trace_id, "032x")

            return GenerateResponse(
                text=result_text,
                trace_id=trace_id_hex,
                latency_ms=round(elapsed_ms, 2),
            )

        except Exception as exc:
            root_span.set_status(Status(StatusCode.ERROR, str(exc)))
            root_span.record_exception(exc)  # records type + message + stack
            logger.exception("generate failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _orchestrate(prompt: str, model: str) -> str:
    """
    INTERNAL span: represents business logic, not an HTTP boundary.

    Child spans automatically get the current span as their parent.
    """
    with tracer.start_as_current_span("orchestrate", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("llm.model", model)
        span.add_event("orchestration.started")

        # Validate → enrich → call LLM (three child spans)
        validated_prompt = _validate_prompt(prompt)
        enriched = _enrich_prompt(validated_prompt)
        result = await _call_llm(enriched, model)

        span.add_event("orchestration.completed", {"result_length": len(result)})
        return result


def _validate_prompt(prompt: str) -> str:
    """Child span: synchronous sub-operation."""
    with tracer.start_as_current_span("validate_prompt", kind=SpanKind.INTERNAL) as span:
        if not prompt.strip():
            span.set_status(Status(StatusCode.ERROR, "empty prompt"))
            raise ValueError("Prompt must not be empty")
        span.set_attribute("prompt.length", len(prompt))
        return prompt.strip()


def _enrich_prompt(prompt: str) -> str:
    """Child span: adds system context to the prompt."""
    with tracer.start_as_current_span("enrich_prompt", kind=SpanKind.INTERNAL) as span:
        system_prefix = "You are a helpful assistant.\n\n"
        enriched = system_prefix + prompt
        span.set_attribute("enriched_length", len(enriched))
        return enriched


async def _call_llm(prompt: str, model: str) -> str:
    """
    CLIENT span: represents an outgoing call to an external service (the LLM API).

    We demonstrate context propagation: inject trace context into outgoing
    headers so the downstream service can link its spans to ours.
    """
    with tracer.start_as_current_span(
        f"llm.call/{model}",
        kind=SpanKind.CLIENT,
    ) as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_length", len(prompt))

        # Inject current trace context into outgoing HTTP headers
        headers: dict[str, str] = {}
        inject(headers)  # writes 'traceparent' (and optionally 'tracestate') into headers
        logger.info("Outgoing headers with trace context: %s", headers)

        # --- In production this would be a real httpx call to the Anthropic API ---
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         "https://api.anthropic.com/v1/messages",
        #         headers={
        #             **headers,
        #             "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        #             "anthropic-version": "2023-06-01",
        #             "content-type": "application/json",
        #         },
        #         json={"model": model, "max_tokens": 256, "messages": [{"role": "user", "content": prompt}]},
        #     )
        #     data = response.json()
        #     output_text = data["content"][0]["text"]
        # -------------------------------------------------------------------------

        # Simulated response for this worked example
        output_text = f"[simulated response from {model}] Echo: {prompt[:40]}..."

        span.set_attribute("llm.output_length", len(output_text))
        span.set_attribute("llm.simulated", True)
        span.add_event("llm.response_received")

        return output_text


# ── 4. Context propagation demo (standalone, no server needed) ─────────────────

def demonstrate_context_propagation() -> None:
    """
    Shows inject/extract without HTTP: simulates what happens when service A
    calls service B by passing headers through a dict (same as real HTTP headers).
    """
    logger.info("=== Context propagation demo ===")

    with tracer.start_as_current_span("service_a.handle_request", kind=SpanKind.SERVER) as span_a:
        span_a.set_attribute("service", "A")

        # Simulate outgoing call: inject context into headers dict
        outgoing_headers: dict[str, str] = {}
        inject(outgoing_headers)
        logger.info("Service A injects headers: %s", outgoing_headers)

        # ---- Network boundary (headers travel over HTTP) ----

        # Service B receives the headers and extracts the context
        incoming_ctx = extract(outgoing_headers)

        with tracer.start_as_current_span(
            "service_b.handle_request",
            context=incoming_ctx,  # parent = Service A's span
            kind=SpanKind.SERVER,
        ) as span_b:
            span_b.set_attribute("service", "B")
            logger.info(
                "Service B span parent_id matches Service A span_id: %s",
                format(span_b.get_span_context().span_id, "016x"),
            )

    logger.info("=== Demo complete — check stdout for JSON spans ===")


if __name__ == "__main__":
    # Run the propagation demo standalone (no HTTP server needed)
    demonstrate_context_propagation()

    # To start the server:
    # uvicorn tracing_demo:app --reload
    # Then: curl -X POST http://localhost:8000/generate \
    #   -H "Content-Type: application/json" \
    #   -d '{"prompt": "Explain observability", "model": "claude-haiku-5-5"}'
```

### Running and reading the output

```bash
# Run standalone demo (propagation + console spans)
python tracing_demo.py

# Start server
uvicorn tracing_demo:app --reload

# Call it
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain observability in one sentence", "model": "claude-haiku-5-5"}'
```

The `ConsoleSpanExporter` prints each completed span as JSON. Look for:

- `name` — the span's operation name
- `parent_id` — links child to parent (forms the tree)
- `start_time` / `end_time` — compute duration
- `attributes` — key-value metadata
- `events` — timestamped annotations

---

## Exercises

1. Add a `baggage` item `user_id` to the context in `_orchestrate` using `opentelemetry.baggage` and read it back in `_call_llm`. Log its value. Confirm it is propagated via the carrier headers.

2. Add a `SpanKind.PRODUCER` span that simulates publishing the LLM result to a message queue (use a dict as the carrier). Add a corresponding `SpanKind.CONSUMER` span that extracts the context from the dict, as if a separate worker process received the message.

3. Replace `ConsoleSpanExporter` with `OTLPSpanExporter` pointed at a local Jaeger all-in-one instance (`docker run -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one`). View the trace waterfall in the Jaeger UI at `http://localhost:16686`.

4. Write a pytest test that calls `demonstrate_context_propagation()` and asserts that the child span's `parent_id` equals the parent span's `span_id`. Use `opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter` to capture spans without console noise.

(Solutions: ../solutions/week-11/day-4.md)

---

## Self-check

- [ ] I can explain the difference between a trace, a span, and a parent span, and draw the tree for a three-service call chain.
- [ ] I can set up a `TracerProvider` with `ConsoleSpanExporter` and create nested spans using `start_as_current_span`.
- [ ] I can explain what the W3C `traceparent` header contains and why `inject`/`extract` are needed at service boundaries.
- [ ] I can add `SpanKind`, attributes, and events to a span and explain when to use `StatusCode.ERROR` vs `StatusCode.OK`.

---

## Further reading

- [OpenTelemetry Python docs](https://opentelemetry-python.readthedocs.io/en/latest/)
- [OTel Trace specification](https://opentelemetry.io/docs/specs/otel/trace/api/)
- [W3C Trace Context spec](https://www.w3.org/TR/trace-context/)
- [OTel semantic conventions — HTTP spans](https://opentelemetry.io/docs/specs/semconv/http/http-spans/)
- [Jaeger quickstart](https://www.jaegertracing.io/docs/latest/getting-started/)
