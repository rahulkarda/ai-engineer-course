# Phase 7 · Week 09 · Day 1 — Systematic Debugging Method & Reading Logs/Traces/Metrics

> Objective: Apply a repeatable, evidence-driven debugging method to production incidents and extract signal from logs, distributed traces, and metrics dashboards.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-7]]

## Concept

Debugging in production is not guesswork — it is the scientific method applied under time pressure.

**The five-step method:**

1. **Observe** — Gather raw signals without touching anything. Screenshots, log snippets, metric graphs, alert payloads.
2. **Hypothesize** — Form the smallest set of hypotheses that explain the observed symptoms. Rank by probability.
3. **Isolate** — Find the boundary where good behavior ends and bad behavior begins. Binary-search through time, components, and requests.
4. **Verify** — Confirm or falsify each hypothesis with a targeted probe (log query, metric drill-down, test request).
5. **Fix & Validate** — Apply the minimal change, verify the symptom is gone, check for regressions.

The method prevents two failure modes: *thrashing* (changing random things and hoping) and *tunnel vision* (fixating on the first hypothesis).

## How it works internally

### Log anatomy

A structured log line carries:

```
timestamp  level  service  trace_id  span_id  message  key=value pairs
```

The `trace_id` is the join key across all services for a single request. Without it you are reading fragments; with it you reconstruct the full causal chain.

### Distributed traces

A trace is a tree of spans. Each span records:
- `start_ts`, `end_ts` → latency contribution
- `parent_span_id` → causal parent
- `status` → ok / error
- `attributes` → HTTP method, DB query, etc.

The flame graph visualization maps spans onto a timeline. The *critical path* — the longest chain of dependent spans — is where latency lives.

### Metrics taxonomy

| Type | What it measures | Example |
|------|-----------------|---------|
| Counter | Cumulative events | `http_requests_total` |
| Gauge | Current value | `memory_bytes` |
| Histogram | Distribution | `request_duration_seconds` |
| Summary | Quantiles | p50, p95, p99 latency |

**USE/RED frameworks** cover complementary angles:

- **USE** (infrastructure): Utilization, Saturation, Errors
- **RED** (services): Rate, Errors, Duration

Start with RED on the service layer, then USE on the infrastructure layer when you find resource pressure.

## Worked example

The following script simulates a log ingestor, injects a realistic incident, and demonstrates the isolation technique.

```python
"""
day1_debug_method.py — systematic debugging walkthrough

Scenario: An HTTP service starts returning 500s.
We walk through the five steps using structured logs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── structured logger ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
log = logging.getLogger("debugger")


def structured_log(
    level: str,
    service: str,
    trace_id: str,
    message: str,
    **kwargs: Any,
) -> str:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "service": service,
        "trace_id": trace_id,
        "msg": message,
        **kwargs,
    }
    line = json.dumps(record)
    getattr(log, level.lower(), log.info)(line)
    return line


# ── simulated trace store ───────────────────────────────────────────────────
@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_ms: float
    end_ms: float
    status: str  # "ok" | "error"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def simulate_request(trace_id: str, inject_fault: bool = False) -> list[Span]:
    """Simulate a three-tier request: gateway → app → database."""
    now = time.monotonic() * 1000

    db_ok = not inject_fault
    db_latency = 2400.0 if inject_fault else 12.0  # slow query when faulted

    gateway = Span(
        trace_id=trace_id,
        span_id="s-gw",
        parent_span_id=None,
        service="gateway",
        operation="HTTP GET /api/orders",
        start_ms=now,
        end_ms=now + 5 + db_latency,
        status="error" if inject_fault else "ok",
        attributes={"http.status_code": 504 if inject_fault else 200},
    )
    app = Span(
        trace_id=trace_id,
        span_id="s-app",
        parent_span_id="s-gw",
        service="app",
        operation="fetch_orders",
        start_ms=now + 1,
        end_ms=now + 3 + db_latency,
        status="error" if inject_fault else "ok",
    )
    db = Span(
        trace_id=trace_id,
        span_id="s-db",
        parent_span_id="s-app",
        service="postgres",
        operation="SELECT * FROM orders WHERE ...",
        start_ms=now + 2,
        end_ms=now + 2 + db_latency,
        status="ok",
        attributes={"db.rows_examined": 9_800_000 if inject_fault else 42},
    )
    return [gateway, app, db]


def analyze_trace(spans: list[Span]) -> None:
    """Step 3 — isolation: find the critical-path span."""
    critical = max(spans, key=lambda s: s.duration_ms)
    print("\n--- Trace analysis ---")
    for s in spans:
        marker = " ← CRITICAL" if s.span_id == critical.span_id else ""
        print(f"  [{s.service:10s}] {s.operation[:40]:<40} "
              f"{s.duration_ms:8.1f} ms  status={s.status}{marker}")
    print(f"\nRoot cause points to: {critical.service}.{critical.operation}")
    if "db.rows_examined" in critical.attributes:
        rows = critical.attributes["db.rows_examined"]
        if rows > 100_000:
            print(f"  db.rows_examined={rows:,} — missing index or full table scan")


def main() -> None:
    print("=== Step 1: OBSERVE — emit structured logs ===")
    for i in range(5):
        tid = f"trace-{i:04d}"
        fault = i == 3  # request #3 is the bad one
        status = "ERROR" if fault else "INFO"
        structured_log(
            status if status == "INFO" else "error",
            "gateway",
            tid,
            "request processed" if not fault else "upstream timeout",
            http_status=200 if not fault else 504,
            duration_ms=14 if not fault else 2405,
        )

    print("\n=== Step 2: HYPOTHESIZE ===")
    print("  H1: database query missing index (explains latency spike)")
    print("  H2: downstream service overloaded")
    print("  H3: network partition")

    print("\n=== Step 3+4: ISOLATE + VERIFY via trace ===")
    # Reproduce the bad trace
    bad_spans = simulate_request("trace-0003", inject_fault=True)
    analyze_trace(bad_spans)

    print("\n=== Step 5: FIX proposal ===")
    print("  Add composite index: CREATE INDEX ON orders(customer_id, created_at DESC);")
    print("  Verify: run EXPLAIN ANALYZE — rows_examined should drop to O(1).")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python day1_debug_method.py
```

Expected output shows the database span as critical with 2400 ms and `db.rows_examined=9,800,000`.

## Exercises

1. Extend `simulate_request()` to add a fourth span: a Redis cache lookup that happens before the database. Modify `analyze_trace()` so that when the cache is a hit (latency < 1 ms), it marks the DB span as *unreachable* and the critical path skips it.

2. Write a function `parse_trace_id(log_line: str) -> str | None` that extracts `trace_id` from a JSON log line. Return `None` if the field is absent. Add three pytest unit tests covering: (a) valid JSON with trace_id, (b) valid JSON without trace_id, (c) malformed JSON.

3. Implement a `MetricBucket` class with methods `record(value: float)` and `percentile(p: float) -> float`. Use it to compute p50/p95/p99 over the simulated durations and print them. No external libraries — implement percentile via sorted list.

(Solutions: ../solutions/week-09/day-1.md)

## Self-check

- [ ] I can describe the five-step debugging method and explain why each step exists
- [ ] I can read a structured log line and identify timestamp, trace_id, and payload fields
- [ ] I can interpret a trace flame graph and find the critical path
- [ ] I can apply USE vs RED depending on whether I am looking at infra or service metrics
- [ ] I can implement a script that reproduces a slow-query scenario and localizes the root cause

## Supplementary links

- [Google SRE Book — Chapter 12: Being On-Call](https://sre.google/sre-book/being-on-call/)
- [OpenTelemetry Trace Spec](https://opentelemetry.io/docs/concepts/signals/traces/)
- [Brendan Gregg — USE Method](https://www.brendangregg.com/usemethod.html)
