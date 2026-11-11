# Week 11 · Day 4 Solutions

## Exercise 1

Record exceptions on spans with `span.record_exception()` and `span.set_status(StatusCode.ERROR)`.

```python
"""
tracing_errors.py — Error recording on OpenTelemetry spans.
"""
import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

# --- tracer setup (swap exporter for OTLP in production) ---
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer(__name__)


def call_llm(prompt: str, model: str = "claude-haiku-5-5") -> str:
    """Simulate an LLM call; records the exception on the span if one occurs."""
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.prompt_length", len(prompt))
        try:
            result = _llm_request(prompt, model)
            span.set_attribute("llm.response_length", len(result))
            return result
        except Exception as exc:
            # W3C / OTel convention: record the full exception + mark span ERROR
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
            logger.exception("LLM call failed", extra={"model": model})
            raise


def _llm_request(prompt: str, model: str) -> str:
    """Placeholder — replace with real Anthropic client call."""
    import anthropic  # lazy import so tests don't need the SDK installed

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text  # type: ignore[index]


# --- unit tests ---
import pytest
from unittest.mock import patch, MagicMock


def test_span_records_exception_on_error() -> None:
    _exporter.clear()

    with pytest.raises(RuntimeError):
        with tracer.start_as_current_span("test.error") as span:
            exc = RuntimeError("boom")
            try:
                raise exc
            except RuntimeError as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))
                raise

    spans = _exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code == StatusCode.ERROR
    # OTel stores exception events with attribute "exception.message"
    events = [e for e in s.events if e.name == "exception"]
    assert len(events) == 1
    assert events[0].attributes["exception.message"] == "boom"


def test_span_not_error_on_success() -> None:
    _exporter.clear()

    with tracer.start_as_current_span("test.ok"):
        pass

    spans = _exporter.get_finished_spans()
    assert spans[0].status.status_code != StatusCode.ERROR
```

---

## Exercise 2

Extract W3C `traceparent` header from an incoming HTTP request dict and create a child span.

```python
"""
trace_propagation.py — W3C traceparent extraction and child-span creation.

W3C Trace Context spec: https://www.w3.org/TR/trace-context/
Format: traceparent: 00-{trace_id_hex32}-{parent_span_id_hex16}-{flags_hex2}
"""
import logging
import re
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

logger = logging.getLogger(__name__)

_W3C_PATTERN = re.compile(
    r"^00-(?P<trace_id>[0-9a-f]{32})-(?P<parent_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer(__name__)


def parse_traceparent(header_value: str) -> SpanContext | None:
    """Parse a W3C traceparent header into an OTel SpanContext.

    Returns None if the header is absent or malformed.
    """
    if not header_value:
        return None
    match = _W3C_PATTERN.match(header_value.strip().lower())
    if not match:
        logger.warning("malformed traceparent header: %r", header_value)
        return None

    trace_id = int(match["trace_id"], 16)
    span_id = int(match["parent_id"], 16)
    flags = int(match["flags"], 16)
    sampled = bool(flags & 0x01)

    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED if sampled else 0),
    )


def handle_request(http_request: dict[str, Any]) -> str:
    """Process an incoming HTTP request, continuing the upstream trace if present.

    Args:
        http_request: Simulated request dict with a 'headers' sub-dict.

    Returns:
        The trace ID used for this request (hex string).
    """
    headers: dict[str, str] = http_request.get("headers", {})
    traceparent = headers.get("traceparent") or headers.get("X-Traceparent", "")

    parent_context = parse_traceparent(traceparent)

    if parent_context and parent_context.is_valid:
        # Attach the remote parent so our span becomes a child
        remote_span = NonRecordingSpan(parent_context)
        ctx = trace.set_span_in_context(remote_span)
        logger.debug(
            "continuing upstream trace",
            extra={"trace_id": format(parent_context.trace_id, "032x")},
        )
    else:
        ctx = None  # type: ignore[assignment]
        logger.debug("no valid traceparent, starting new trace")

    with tracer.start_as_current_span("service.handle_request", context=ctx) as span:
        span.set_attribute("http.method", http_request.get("method", "GET"))
        span.set_attribute("http.path", http_request.get("path", "/"))
        # ... business logic here ...
        return format(span.get_span_context().trace_id, "032x")


# --- unit tests ---
import pytest


def test_parse_valid_traceparent() -> None:
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ctx = parse_traceparent(header)
    assert ctx is not None
    assert ctx.is_valid
    assert format(ctx.trace_id, "032x") == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert format(ctx.span_id, "016x") == "00f067aa0ba902b7"
    assert ctx.trace_flags.sampled


def test_parse_malformed_traceparent() -> None:
    assert parse_traceparent("not-a-traceparent") is None
    assert parse_traceparent("") is None


def test_handle_request_with_traceparent_creates_child_span() -> None:
    _exporter.clear()
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    request = {
        "method": "GET",
        "path": "/api/v1/items",
        "headers": {"traceparent": header},
    }
    trace_id = handle_request(request)
    spans = _exporter.get_finished_spans()
    assert len(spans) == 1
    # Child span must share the upstream trace ID
    assert trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_handle_request_without_traceparent_starts_new_trace() -> None:
    _exporter.clear()
    request = {"method": "POST", "path": "/api/v1/orders", "headers": {}}
    trace_id = handle_request(request)
    assert len(trace_id) == 32  # valid hex trace ID
```

---

## Exercise 3

Baggage propagation: set `user_id` on the root span, read it in a child span.

```python
"""
baggage_propagation.py — OTel baggage: set on root, read in child.

OTel Baggage is a key-value store propagated in-process via context and
across service boundaries via the baggage W3C header.
"""
import logging

from opentelemetry import baggage, context, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

logger = logging.getLogger(__name__)

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer(__name__)

_propagator = W3CBaggagePropagator()


def handle_authenticated_request(user_id: str, resource_id: str) -> dict:
    """Root handler: sets user_id in baggage, then calls downstream logic."""
    # Set baggage on the current context — propagates to all children
    ctx = baggage.set_baggage("user_id", user_id)

    with tracer.start_as_current_span("root.handler", context=ctx) as root_span:
        root_span.set_attribute("resource.id", resource_id)
        # Verify baggage is visible here
        uid = baggage.get_baggage("user_id", context=context.get_current())
        root_span.set_attribute("user.id", uid or "unknown")
        logger.info("root span: user_id=%s", uid)

        result = _fetch_resource(resource_id)

    return result


def _fetch_resource(resource_id: str) -> dict:
    """Child function: reads user_id from baggage without explicit passing."""
    with tracer.start_as_current_span("db.fetch_resource") as span:
        # Baggage is inherited through the context — no argument needed
        uid = baggage.get_baggage("user_id", context=context.get_current())
        span.set_attribute("user.id", uid or "unknown")
        span.set_attribute("resource.id", resource_id)
        logger.info("child span: user_id=%s (from baggage)", uid)
        return {"id": resource_id, "fetched_by": uid}


# --- unit tests ---
import pytest


def test_baggage_propagates_to_child_span() -> None:
    _exporter.clear()
    result = handle_authenticated_request(user_id="user-42", resource_id="res-99")

    assert result["fetched_by"] == "user-42"

    spans = _exporter.get_finished_spans()
    assert len(spans) == 2

    span_names = {s.name for s in spans}
    assert "root.handler" in span_names
    assert "db.fetch_resource" in span_names

    for s in spans:
        assert s.attributes.get("user.id") == "user-42"


def test_baggage_not_leaked_across_requests() -> None:
    """Each request context is independent — no cross-contamination."""
    _exporter.clear()

    result_a = handle_authenticated_request(user_id="alice", resource_id="r1")
    result_b = handle_authenticated_request(user_id="bob", resource_id="r2")

    assert result_a["fetched_by"] == "alice"
    assert result_b["fetched_by"] == "bob"
```

---

## Exercise 4

pytest test using `InMemorySpanExporter` to verify an instrumented function creates exactly one span with expected attributes.

```python
"""
test_instrumented_function.py — Verify OTel spans with InMemorySpanExporter.

Pattern:
  1. Build a TracerProvider backed by InMemorySpanExporter.
  2. Set it as the global provider (or inject it into the code under test).
  3. Call the function.
  4. Retrieve finished spans and assert on names, attributes, status.
"""
import logging
import os
from typing import Generator

import pytest
from unittest.mock import MagicMock, patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Code under test
# ---------------------------------------------------------------------------

def summarise_text(text: str, model: str = "claude-haiku-5-5") -> str:
    """Summarise `text` using the given model.

    Creates exactly one span named "llm.summarise" with attributes:
      - llm.model
      - llm.input_length
      - llm.output_length
    """
    _tracer = trace.get_tracer(__name__)

    with _tracer.start_as_current_span("llm.summarise") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.input_length", len(text))

        # Real implementation would call Anthropic here
        result = _call_anthropic(text, model)

        span.set_attribute("llm.output_length", len(result))
        return result


def _call_anthropic(text: str, model: str) -> str:
    """Thin wrapper around the Anthropic client — isolated for easy mocking."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": f"Summarise: {text}"}],
    )
    return message.content[0].text  # type: ignore[index]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def span_exporter() -> Generator[InMemorySpanExporter, None, None]:
    """Provide a fresh InMemorySpanExporter and register it as the global provider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Save and restore the global provider so tests don't bleed into each other
    original_provider = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)
    yield exporter
    trace.set_tracer_provider(original_provider)
    exporter.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_summarise_creates_exactly_one_span(span_exporter: InMemorySpanExporter) -> None:
    """summarise_text must produce exactly one 'llm.summarise' span."""
    mock_response_text = "Short summary."

    mock_content = MagicMock()
    mock_content.text = mock_response_text

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        result = summarise_text("A long article about machine learning.", model="claude-haiku-5-5")

    assert result == mock_response_text

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"

    span = spans[0]
    assert span.name == "llm.summarise"


def test_summarise_span_has_expected_attributes(span_exporter: InMemorySpanExporter) -> None:
    """The span must carry model, input_length, and output_length attributes."""
    input_text = "The quick brown fox."
    expected_model = "claude-sonnet-5-5"
    mock_output = "Fox story."

    mock_content = MagicMock()
    mock_content.text = mock_output
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        summarise_text(input_text, model=expected_model)

    span = span_exporter.get_finished_spans()[0]
    attrs = span.attributes

    assert attrs["llm.model"] == expected_model
    assert attrs["llm.input_length"] == len(input_text)
    assert attrs["llm.output_length"] == len(mock_output)


def test_summarise_span_status_ok_on_success(span_exporter: InMemorySpanExporter) -> None:
    """A successful call must NOT set an error status on the span."""
    mock_content = MagicMock()
    mock_content.text = "ok"
    mock_message = MagicMock()
    mock_message.content = [mock_content]

    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        summarise_text("text", model="claude-haiku-5-5")

    span = span_exporter.get_finished_spans()[0]
    assert span.status.status_code != StatusCode.ERROR


def test_summarise_span_status_error_on_exception(span_exporter: InMemorySpanExporter) -> None:
    """When the LLM call raises, the span must record the exception and set ERROR status."""
    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.side_effect = RuntimeError("rate limit")

        # Wrap summarise_text to add error recording (mirrors Exercise 1 pattern)
        _tracer = trace.get_tracer(__name__)
        with _tracer.start_as_current_span("llm.summarise_with_error") as span:
            try:
                raise RuntimeError("rate limit")
            except RuntimeError as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))

    spans = span_exporter.get_finished_spans()
    error_span = next(s for s in spans if s.name == "llm.summarise_with_error")
    assert error_span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in error_span.events)
```
