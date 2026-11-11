# Solutions — Week 09 · Day 1

## Exercise 1 — Redis cache span with hit/miss logic

```python
def simulate_request_with_cache(
    trace_id: str,
    inject_fault: bool = False,
    cache_hit: bool = True,
) -> list[Span]:
    now = time.monotonic() * 1000
    db_latency = 2400.0 if inject_fault else 12.0

    gateway = Span(
        trace_id=trace_id,
        span_id="s-gw",
        parent_span_id=None,
        service="gateway",
        operation="HTTP GET /api/orders",
        start_ms=now,
        end_ms=now + (0.5 if cache_hit else 5 + db_latency),
        status="ok",
        attributes={"http.status_code": 200},
    )
    app = Span(
        trace_id=trace_id,
        span_id="s-app",
        parent_span_id="s-gw",
        service="app",
        operation="fetch_orders",
        start_ms=now + 0.2,
        end_ms=now + 0.4 if cache_hit else now + 3 + db_latency,
        status="ok",
    )
    cache = Span(
        trace_id=trace_id,
        span_id="s-cache",
        parent_span_id="s-app",
        service="redis",
        operation="GET orders:*",
        start_ms=now + 0.3,
        end_ms=now + 0.35,  # always fast
        status="ok",
        attributes={"cache.hit": cache_hit},
    )
    spans = [gateway, app, cache]

    if not cache_hit:
        db = Span(
            trace_id=trace_id,
            span_id="s-db",
            parent_span_id="s-app",
            service="postgres",
            operation="SELECT * FROM orders WHERE ...",
            start_ms=now + 0.5,
            end_ms=now + 0.5 + db_latency,
            status="ok",
            attributes={"db.rows_examined": 9_800_000 if inject_fault else 42},
        )
        spans.append(db)

    return spans


def analyze_trace_with_cache(spans: list[Span]) -> None:
    cache_spans = [s for s in spans if s.service == "redis"]
    is_hit = cache_spans[0].attributes.get("cache.hit", False) if cache_spans else False

    if is_hit:
        # DB is unreachable — cache short-circuited it
        reachable = [s for s in spans if s.service != "postgres"]
        critical = max(reachable, key=lambda s: s.duration_ms)
        print("Cache HIT — DB span unreachable (not on critical path)")
    else:
        critical = max(spans, key=lambda s: s.duration_ms)

    print(f"Critical span: {critical.service}.{critical.operation} ({critical.duration_ms:.1f} ms)")
```

## Exercise 2 — parse_trace_id with tests

```python
import json
import pytest


def parse_trace_id(log_line: str) -> str | None:
    try:
        data = json.loads(log_line)
        return data.get("trace_id")
    except (json.JSONDecodeError, TypeError):
        return None


# tests/test_day1.py
class TestParseTraceId:
    def test_valid_with_trace_id(self):
        line = '{"ts": "2026-01-01T00:00:00Z", "trace_id": "abc-123", "msg": "ok"}'
        assert parse_trace_id(line) == "abc-123"

    def test_valid_without_trace_id(self):
        line = '{"ts": "2026-01-01T00:00:00Z", "msg": "no trace"}'
        assert parse_trace_id(line) is None

    def test_malformed_json(self):
        assert parse_trace_id("{not json}") is None
```

## Exercise 3 — MetricBucket

```python
import math


class MetricBucket:
    def __init__(self) -> None:
        self._values: list[float] = []

    def record(self, value: float) -> None:
        self._values.append(value)

    def percentile(self, p: float) -> float:
        """p in [0, 100]."""
        if not self._values:
            raise ValueError("No observations recorded")
        sorted_vals = sorted(self._values)
        idx = (p / 100) * (len(sorted_vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        fraction = idx - lower
        return sorted_vals[lower] * (1 - fraction) + sorted_vals[upper] * fraction


# Usage:
# bucket = MetricBucket()
# for d in [14, 14, 15, 2405, 14]:
#     bucket.record(d)
# print(f"p50={bucket.percentile(50):.0f}  p95={bucket.percentile(95):.0f}  p99={bucket.percentile(99):.0f}")
```
