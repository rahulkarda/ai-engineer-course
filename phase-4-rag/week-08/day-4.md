# Phase 4 · Week 08 · Day 4 — Observability: Tracing, Structured Logging, and Cost Tracking

> Objective: Instrument the Document Intelligence Platform with OpenTelemetry tracing, structured JSON logging, LLM cost tracking, and a golden-dataset eval gate that fails CI when answer quality drops.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-3]]

## Concept

Observability answers three questions you cannot answer with unit tests alone:

| Question | Tool |
|---|---|
| Why did this query return a bad answer? | Distributed tracing (spans, context propagation) |
| What happened inside request #4821? | Structured logs (JSON, correlated by trace ID) |
| How much did last week's queries cost? | Token accounting (from API response metadata) |

Without observability, a RAG system is a black box. You see the input and output but not _which retrieval step returned the wrong chunks_, _which reranker score was miscalibrated_, or _which prompt template caused the LLM to ignore the context_.

---

### OpenTelemetry Basics

OpenTelemetry (OTel) is a vendor-neutral standard for telemetry data. Three signal types matter here:

- **Traces** — a tree of **spans**, each representing a unit of work (one HTTP request, one retrieval call, one LLM call). Spans have a start time, duration, and key/value attributes.
- **Metrics** — numeric aggregations (request count, latency histogram, token total).
- **Logs** — structured events. OTel can correlate logs to traces via `trace_id`.

For RAG, the most useful span tree looks like:

```
POST /query  [root span]
├── hybrid_retrieval  [child span]
│   ├── vector_search  [grandchild]
│   └── bm25_search    [grandchild]
├── reranking          [child span]
└── llm_generate       [child span]  ← attach token counts here
```

---

### Structured Logging

`print()` logs are unqueryable. Structured JSON logs can be shipped to Elasticsearch, Loki, or CloudWatch and filtered with a single query:

```json
{"timestamp": "2026-11-11T09:12:34Z", "level": "INFO", "trace_id": "abc123",
 "event": "query_complete", "latency_ms": 342, "tokens_used": 512, "cost_usd": 0.00041}
```

Python's built-in `logging` module with a `logging.Formatter` subclass that emits JSON is sufficient — no extra dependency. `structlog` is a popular alternative with a cleaner API; the pattern is the same.

---

### LLM Cost Tracking

The Anthropic API returns `usage.input_tokens` and `usage.output_tokens` in every response. Cost is:

```
cost = (input_tokens * input_price_per_token) + (output_tokens * output_price_per_token)
```

Prices vary by model. Store them in a config dict keyed by model ID. Accumulate totals in a thread-safe counter (Python `threading.Lock` or an atomic increment library). Expose totals via a `/metrics` endpoint.

---

### Eval Gate

A **golden dataset** is a fixed set of (question, expected_answer) pairs whose quality is known. The eval runner:

1. Runs each question through the live pipeline.
2. Scores each answer (RAGAS faithfulness, or simpler: LLM-as-judge with a binary pass/fail prompt).
3. Asserts the aggregate score meets a threshold (e.g., faithfulness >= 0.8).
4. Returns exit code 1 if the gate fails — CI sees a failing step.

This is a regression test for answer quality. Run it before merging changes to the retrieval pipeline or prompt templates.

---

## How It Works Internally

OTel spans work via **context propagation**: a `Tracer` object creates spans that are parented to the current active span. Python's `opentelemetry-sdk` injects the current span into a context variable (a `contextvars.ContextVar`). Each child span reads that variable to find its parent. Across async boundaries, Python's `asyncio` propagates `contextvars` automatically.

The FastAPI middleware pattern: a `BaseHTTPMiddleware` subclass calls `await call_next(request)` and wraps it with a span. All handlers called during that `await` inherit the span as their parent.

The `TokenCostTracker` uses a `threading.Lock` to protect its running totals. In an async application, a `asyncio.Lock` is safer — the exercise asks you to make this change.

---

## Worked Example

```python
"""
day_4_observability.py

Observability layer for the Document Intelligence Platform:
  - OTel tracing middleware
  - Structured JSON logging
  - LLM cost tracking
  - Golden-dataset eval runner

Run (no external services needed — uses console exporter):
    pip install fastapi uvicorn opentelemetry-sdk opentelemetry-api pydantic
    uvicorn day_4_observability:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware import Middleware
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """
    Emit log records as single-line JSON objects.

    Attaches the current OTel trace_id when available so logs can be
    correlated with traces in your observability backend.
    """

    def format(self, record: logging.LogRecord) -> str:
        current_span = trace.get_current_span()
        ctx = current_span.get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else ""

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id,
            "event": record.getMessage(),
        }

        # Attach any extra fields the caller passed via `extra={}`
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit structured JSON."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(level)


configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenTelemetry setup
# ---------------------------------------------------------------------------


def configure_tracing(service_name: str = "doc-intel-platform") -> trace.Tracer:
    """
    Initialise OTel with a console exporter (prints spans as JSON to stdout).

    In production replace ConsoleSpanExporter with OTLPSpanExporter pointing
    at your collector (Jaeger, Tempo, Honeycomb, etc.).
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


tracer = configure_tracing()


# ---------------------------------------------------------------------------
# Token cost tracker
# ---------------------------------------------------------------------------


# Prices in USD per token (as of late 2025 — verify current pricing).
# Source: https://www.anthropic.com/pricing
_PRICE_TABLE: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 15e-6, "output": 75e-6},
    "claude-sonnet-5-5": {"input": 3e-6, "output": 15e-6},
    "claude-haiku-5-5": {"input": 0.25e-6, "output": 1.25e-6},
}


class TokenCostTracker:
    """
    Thread-safe accumulator for LLM token usage and cost.

    Call record() after each LLM response.  Read totals with snapshot().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._cost_usd: float = 0.0
        self._call_count: int = 0

    def record(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record token usage for one LLM call and return the call cost in USD.

        Args:
            model_id:      Anthropic model identifier.
            input_tokens:  Tokens in the prompt (from response.usage.input_tokens).
            output_tokens: Tokens in the completion (from response.usage.output_tokens).

        Returns:
            Cost in USD for this single call.
        """
        prices = _PRICE_TABLE.get(model_id, _PRICE_TABLE["claude-haiku-5-5"])
        call_cost = (input_tokens * prices["input"]) + (output_tokens * prices["output"])

        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cost_usd += call_cost
            self._call_count += 1

        logger.info(
            "LLM call recorded.",
            extra={
                "model_id": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "call_cost_usd": round(call_cost, 6),
            },
        )
        return call_cost

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of current totals (safe to call from any thread)."""
        with self._lock:
            return {
                "call_count": self._call_count,
                "input_tokens_total": self._input_tokens,
                "output_tokens_total": self._output_tokens,
                "cost_usd_total": round(self._cost_usd, 6),
            }


cost_tracker = TokenCostTracker()


# ---------------------------------------------------------------------------
# OTel tracing middleware
# ---------------------------------------------------------------------------


class TracingMiddleware:
    """
    ASGI middleware that wraps every HTTP request in an OTel span.

    Attaches:
      - http.method, http.route, http.status_code as span attributes.
      - Propagates the trace context into response headers for downstream
        services (X-Trace-Id).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        span_name = f"{method} {path}"

        with tracer.start_as_current_span(span_name) as span:
            start = time.perf_counter()
            span.set_attribute("http.method", method)
            span.set_attribute("http.route", path)

            # Capture the status code from the response
            status_code_holder: list[int] = [200]

            async def send_wrapper(message: Any) -> None:
                if message["type"] == "http.response.start":
                    status_code_holder[0] = message.get("status", 200)
                    span.set_attribute("http.status_code", status_code_holder[0])
                await send(message)

            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("latency_ms", round(latency_ms, 2))
                logger.info(
                    "Request complete.",
                    extra={
                        "method": method,
                        "path": path,
                        "status_code": status_code_holder[0],
                        "latency_ms": round(latency_ms, 2),
                    },
                )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Document Intelligence Platform — Observability", version="0.1.0")
app.add_middleware(TracingMiddleware)  # type: ignore[arg-type]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """
    Expose accumulated token usage and cost totals.

    In production, also expose Prometheus-format metrics via
    opentelemetry-exporter-prometheus or the prometheus_client library.
    """
    return cost_tracker.snapshot()


@app.post("/query")
async def query(request: Request) -> JSONResponse:
    """
    Stub query endpoint — instruments a simulated LLM call.

    Replace the stub with the real query pipeline from day-2 / day-3.
    """
    body = await request.json()
    question = body.get("question", "")

    with tracer.start_as_current_span("query_pipeline") as pipeline_span:
        # --- Retrieval span ---
        with tracer.start_as_current_span("hybrid_retrieval") as ret_span:
            time.sleep(0.02)  # simulate I/O
            ret_span.set_attribute("chunks_returned", 5)
            logger.info("Retrieval complete.", extra={"chunks": 5})

        # --- Reranking span ---
        with tracer.start_as_current_span("reranking") as rerank_span:
            time.sleep(0.01)
            rerank_span.set_attribute("top_n", 3)

        # --- LLM generation span ---
        model_id = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        with tracer.start_as_current_span("llm_generate") as llm_span:
            time.sleep(0.05)  # simulate network
            input_tokens = 512
            output_tokens = 128
            llm_span.set_attribute("model_id", model_id)
            llm_span.set_attribute("input_tokens", input_tokens)
            llm_span.set_attribute("output_tokens", output_tokens)

            call_cost = cost_tracker.record(model_id, input_tokens, output_tokens)
            llm_span.set_attribute("cost_usd", round(call_cost, 6))

        pipeline_span.set_attribute("question_length", len(question))

    return JSONResponse(
        {
            "answer": f"[MOCK] Answer to: {question}",
            "model_id": model_id,
            "tokens_used": input_tokens + output_tokens,
            "cost_usd": round(call_cost, 6),
        }
    )


# ---------------------------------------------------------------------------
# Golden-dataset eval runner
# ---------------------------------------------------------------------------


class EvalSample(BaseModel):
    question: str
    expected_keywords: list[str] = Field(
        ...,
        description="Keywords that must appear in a faithful answer.",
    )


class EvalResult(BaseModel):
    question: str
    answer: str
    passed: bool
    missing_keywords: list[str]


def _keyword_faithfulness(answer: str, keywords: list[str]) -> tuple[float, list[str]]:
    """
    Proxy for RAGAS faithfulness: fraction of expected keywords present.

    Returns (score 0.0–1.0, list of missing keywords).
    """
    answer_lower = answer.lower()
    missing = [kw for kw in keywords if kw.lower() not in answer_lower]
    score = 1.0 - (len(missing) / max(len(keywords), 1))
    return score, missing


def run_eval_gate(
    samples: list[EvalSample],
    query_fn: Callable[[str], str],
    min_faithfulness: float = 0.8,
) -> tuple[float, list[EvalResult]]:
    """
    Run the golden-dataset eval gate.

    Args:
        samples:           List of EvalSample with question + expected keywords.
        query_fn:          Callable that takes a question and returns an answer string.
                           In production: lambda q: registry.run_query_pipeline(q).answer
        min_faithfulness:  Threshold below which the gate fails.

    Returns:
        (aggregate_score, list[EvalResult])

    Raises:
        AssertionError if aggregate_score < min_faithfulness (use in CI).
    """
    results: list[EvalResult] = []

    for sample in samples:
        with tracer.start_as_current_span("eval_sample") as span:
            span.set_attribute("question", sample.question)
            answer = query_fn(sample.question)
            score, missing = _keyword_faithfulness(answer, sample.expected_keywords)
            span.set_attribute("faithfulness", score)
            span.set_attribute("passed", score >= min_faithfulness)

            results.append(
                EvalResult(
                    question=sample.question,
                    answer=answer,
                    passed=score >= min_faithfulness,
                    missing_keywords=missing,
                )
            )
            logger.info(
                "Eval sample.",
                extra={
                    "question": sample.question[:60],
                    "faithfulness": round(score, 3),
                    "passed": score >= min_faithfulness,
                },
            )

    aggregate = sum(1 for r in results if r.passed) / max(len(results), 1)
    logger.info("Eval gate complete.", extra={"aggregate_faithfulness": round(aggregate, 3)})

    assert aggregate >= min_faithfulness, (
        f"Eval gate FAILED: faithfulness {aggregate:.2f} < {min_faithfulness:.2f}. "
        f"Failing questions: {[r.question for r in results if not r.passed]}"
    )

    return aggregate, results


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate two LLM calls to populate the cost tracker.
    cost_tracker.record("claude-haiku-5-5", 400, 100)
    cost_tracker.record("claude-sonnet-5-5", 600, 200)
    print("Cost snapshot:", json.dumps(cost_tracker.snapshot(), indent=2))

    # Run the eval gate with a stub query function that always returns a keyword match.
    golden_set = [
        EvalSample(
            question="What file formats does the platform support?",
            expected_keywords=["pdf", "docx", "html"],
        ),
        EvalSample(
            question="How does hybrid retrieval work?",
            expected_keywords=["vector", "bm25", "retrieval"],
        ),
    ]

    def stub_query(question: str) -> str:
        return (
            "The platform supports PDF, DOCX, and HTML files. "
            "Hybrid retrieval uses vector search and BM25 keyword matching."
        )

    score, results = run_eval_gate(golden_set, stub_query, min_faithfulness=0.8)
    print(f"\nEval gate passed — aggregate faithfulness: {score:.2f}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.question[:50]}")
```

**Run it:**
```bash
python day_4_observability.py
```

Expected: structured JSON log lines to stdout, cost snapshot printed, eval gate passes.

```bash
uvicorn day_4_observability:app --reload
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "How does retrieval work?"}'
```

---

## Exercises

1. **Add a /metrics endpoint exposing token_total and cost_usd.** The current `/metrics` already returns a snapshot. Extend it to also return `cost_usd_by_model: dict[str, float]` — the cost broken down per model ID. Update `TokenCostTracker.record()` to accumulate per-model costs. Write a pytest test that records two calls on different models and asserts the per-model breakdown is correct.

2. **Write a golden-dataset eval runner that asserts RAGAS faithfulness >= 0.8.** Replace the keyword-based `_keyword_faithfulness` proxy with a real RAGAS scorer: `from ragas.metrics import faithfulness` and use `ragas.evaluate()`. Your `query_fn` should call the real pipeline (mock the LLM in tests). The runner should exit with code 1 on failure (already done via `AssertionError` — wire it into a `pytest` test or a `scripts/eval_gate.py` CLI script).

3. **Add a span for each retrieval step.** Modify the `/query` handler to emit separate spans for `vector_search` and `bm25_search` as children of `hybrid_retrieval`. Each span should carry attributes: `chunks_returned` (int) and `index_latency_ms` (float). Write a test using OTel's `InMemorySpanExporter` to assert the span tree has the correct parent/child relationships.

4. **Asyncio safety.** The `TokenCostTracker` uses `threading.Lock`. In an async FastAPI application, blocking a `threading.Lock` inside a coroutine can starve the event loop if the lock is ever held for more than a few microseconds. Replace `threading.Lock` with `asyncio.Lock`, make `record()` an `async def`, and update all callers. Write a pytest-asyncio test that calls `record()` concurrently from two coroutines and asserts the total is correct.

(Solutions: ../solutions/week-08/day-4.md)

---

## Self-check

- [ ] I can explain what a span is and how parent/child spans form a trace.
- [ ] I can configure OTel tracing with a console exporter and identify the output.
- [ ] I can write a `JSONFormatter` that emits structured logs with a `trace_id`.
- [ ] I can implement `TokenCostTracker` with thread-safe accumulation and a `/metrics` endpoint.
- [ ] I can write a golden-dataset eval runner that fails CI when answer quality drops.
- [ ] I can articulate why `threading.Lock` in async code is a footgun and how to fix it.

---

## Supplementary Links

- [OpenTelemetry Python SDK quickstart](https://opentelemetry.io/docs/languages/python/getting-started/)
- [OTel InMemorySpanExporter (for tests)](https://opentelemetry-python.readthedocs.io/en/latest/sdk/trace.export.html#opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter)
- [RAGAS evaluation framework](https://docs.ragas.io/)
- [Anthropic API usage object](https://docs.anthropic.com/en/api/messages)
- [structlog — structured logging for Python](https://www.structlog.org/en/stable/)
- [Python logging HOWTO](https://docs.python.org/3/howto/logging.html)
- [asyncio.Lock vs threading.Lock](https://docs.python.org/3/library/asyncio-sync.html)
