# Week 11 Assessment

## Conceptual (interview-style)

1. **Three Pillars of Observability**: Name the three pillars of observability and explain what each pillar tells you that the others cannot. Give a concrete example of when you would reach for each one to diagnose a production incident.

2. **Structured Logging**: What is the difference between unstructured and structured logging? Why does structured (JSON) logging with correlation IDs matter in a distributed LLM gateway handling thousands of concurrent requests? What fields should every log line include at minimum?

3. **Correlation IDs**: Explain how a correlation ID flows through a system: from the HTTP client, through middleware, into background tasks, and into log lines. What Python primitive would you use to propagate a correlation ID across async function calls without passing it explicitly as an argument?

4. **Prometheus Metric Types**: Describe the four core Prometheus metric types (Counter, Gauge, Histogram, Summary). For each, give one real example relevant to an LLM API gateway. When would you choose a Histogram over a Summary for latency?

5. **RED and USE Methods**: Compare the RED method (Rate, Errors, Duration) with the USE method (Utilization, Saturation, Errors). Which is better suited to characterising an LLM gateway service, and which is better suited to the underlying GPU/compute resources? Why?

6. **OpenTelemetry Traces and Spans**: Explain the relationship between a trace, a span, and span attributes in OpenTelemetry. How does a child span attach to a parent span? What is a `SpanContext`, and why is it needed for cross-service trace propagation?

7. **SLO, SLI, and Error Budgets**: Define SLO, SLI, and error budget. A service has a 99.9% availability SLO over a 30-day window. How many minutes of downtime does the error budget permit? If the service has already consumed 80% of its error budget with 20 days remaining, what actions should the team consider?

8. **LLM Token Tracking**: Why is token-level tracking important for LLM gateways beyond simple request counts? Describe a Pydantic v2 model you would use to capture a single LLM call's cost. How would you calculate cost for `claude-sonnet-5-5` given input and output token counts?

9. **Alerting Strategy**: Describe the difference between symptom-based alerting and cause-based alerting. Give an example of each for an LLM gateway. Why do observability experts recommend favouring symptom-based alerts for PagerDuty/on-call pages?

10. **Post-Mortems**: What are the key components of a blameless post-mortem? After a 45-minute LLM gateway outage caused by a token-limit misconfiguration, what sections would you include, and what questions should the timeline of events answer?

---

## Coding challenge

### Instrument a FastAPI LLM Gateway

**Setup**

```bash
uv add fastapi uvicorn prometheus-client \
    opentelemetry-api opentelemetry-sdk \
    pydantic anthropic
```

**Skeleton** (`gateway.py`)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    # TODO: call LLM and return response
    return JSONResponse({"response": "placeholder"})
```

You must implement the following five components:

**1. Structured JSON logging with correlation IDs**
- Use `contextvars.ContextVar` to store a `correlation_id` per request.
- Write a custom `logging.Formatter` subclass (`JSONFormatter`) that emits log records as JSON with at minimum: `timestamp` (ISO 8601), `level`, `correlation_id`, `message`.
- Write an ASGI middleware class (`LoggingMiddleware`) that reads `X-Correlation-ID` from the request header or generates a UUID4, stores it in the `ContextVar`, logs request start (`method`, `path`) and request end (`status_code`, `duration_ms`).

**2. Prometheus metrics**
- Define these three metrics at module level:
  - `requests_total`: `Counter` with labels `method`, `path`, `status_code`
  - `request_duration_seconds`: `Histogram` with buckets `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]`
  - `active_requests`: `Gauge` (no labels)
- Write `MetricsMiddleware` that increments/decrements these metrics around every request.
- Add a `GET /metrics` endpoint that returns Prometheus text format (`generate_latest()`).

**3. OpenTelemetry tracing**
- Write a `setup_tracing()` function that configures a `TracerProvider` with a `ConsoleSpanExporter` and `BatchSpanProcessor`.
- Write `OTelMiddleware` that creates a root span named `"http.request"` for every request, setting span attributes `http.method` and `http.route`.
- In the `/chat` handler, add a child span named `"llm.call"` with attributes `llm.model`, `llm.input_tokens`, `llm.output_tokens`. The child span must be created from the same tracer with the parent context propagated correctly.

**4. LLM token and cost tracking**
- Define a Pydantic v2 model `LLMCallRecord` with fields: `model: str`, `input_tokens: int`, `output_tokens: int`, `cost_usd: float`, `latency_ms: float`, `timestamp: datetime`.
- Define `COST_PER_1K_TOKENS` as a `dict[str, dict[str, float]]` covering `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` with `input` and `output` keys (use any reasonable placeholder pricing).
- Implement `calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float`.
- Implement `CostTracker` with methods `record(rec: LLMCallRecord) -> None`, `total_cost() -> float`, and `summary() -> dict[str, Any]`.
- **LLM calls must be mocked in tests** — never call the real Anthropic API in the test suite.

**5. SLO checker**
- Define a Pydantic v2 model `RequestRecord` with fields: `success: bool`, `duration_ms: float`, `timestamp: datetime`.
- Define a Pydantic v2 model `SLOReport` with fields: `slo_name: str`, `target: float`, `actual: float`, `compliant: bool`, `sample_size: int`.
- Implement `compute_slo_compliance(records: list[RequestRecord], slo_name: str, target: float) -> SLOReport` — computes the fraction of successful requests and compares to `target`.

**Acceptance criteria**

- [ ] `pytest tests/` passes with no failures (all LLM calls mocked).
- [ ] `GET /metrics` returns HTTP 200 with `Content-Type: text/plain; version=0.0.4` and contains the strings `requests_total`, `request_duration_seconds`, `active_requests`.
- [ ] Every log line is valid JSON and contains a non-empty `correlation_id` field.
- [ ] OTel spans written to the console exporter include attributes `http.method` and `http.route`; LLM child spans include `llm.model`.
- [ ] `CostTracker.summary()` returns a dict with at least keys `total_cost_usd` and `call_count`.
- [ ] `compute_slo_compliance` returns `compliant=True` for a list of 1000 successful records against a 99.9% target.

> **Dependencies**: `prometheus-client` requires `uv add prometheus-client`; OpenTelemetry requires `uv add opentelemetry-api opentelemetry-sdk`.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: `../solutions/week-11/assessment.md`
