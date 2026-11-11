# Phase 7 · Week 09 · Day 7 (Sunday) — Week Project: Production Incident Lab

> Objective: Build a self-contained "incident lab" — a suite of reproducible buggy services, diagnostic tooling, and automated tests — that demonstrates mastery of the week's debugging skills.  ·  Time: ~3 hr  ·  Prereqs: [[day-1]] through [[day-6]]

## Project spec

Build `incident_lab/` — a Python package with three simulated services, each containing a seeded production bug, plus a diagnostic CLI and test suite.

### Directory layout

```
incident_lab/
├── __init__.py
├── services/
│   ├── __init__.py
│   ├── order_processor.py   # Bug: connection pool exhaustion
│   ├── cache_service.py     # Bug: thundering herd / missing TTL cap
│   └── agent_service.py     # Bug: unbounded token accumulation
├── diagnostics/
│   ├── __init__.py
│   ├── tracer.py            # Lightweight span tracker
│   ├── metrics.py           # Counter/histogram helpers
│   └── circuit_breaker.py   # Reusable circuit breaker
├── cli.py                   # Click CLI: run-incident / diagnose / fix
└── tests/
    ├── test_order_processor.py
    ├── test_cache_service.py
    └── test_agent_service.py
```

### Acceptance criteria

- [ ] Each service has a `run_buggy()` and `run_fixed()` function
- [ ] `diagnostics/tracer.py` records spans with `start`, `end`, `service`, `operation`, `status`
- [ ] `diagnostics/metrics.py` exposes `Counter` and `Histogram` with `p50()`/`p99()` methods
- [ ] `diagnostics/circuit_breaker.py` implements the three-state machine from Day 3
- [ ] All three bugs are reproduced by tests (the buggy version fails/leaks, the fixed version passes)
- [ ] `cli.py` has commands: `run-incident <name>` (runs buggy), `diagnose <name>` (prints root cause), `fix <name>` (runs fixed + asserts clean)
- [ ] `pytest tests/` passes with ≥ 90% of test cases green
- [ ] All code: type hints, `logging`, no hardcoded secrets, Python 3.12+

### Required tests (minimum)

| Test | Asserts |
|------|---------|
| `test_pool_exhaustion_buggy` | Pool size drops to 0 after N concurrent requests |
| `test_pool_exhaustion_fixed` | Pool size returns to max_size after N concurrent requests |
| `test_thundering_herd_buggy` | Backend called > 10 times for 50 concurrent cache misses |
| `test_thundering_herd_fixed` | Backend called ≤ 2 times for 50 concurrent cache misses |
| `test_token_accumulation_buggy` | `agent._history` length > 1000 after 500 summarize() calls |
| `test_token_accumulation_fixed` | `agent._history` length == 0 after 500 summarize() calls |
| `test_circuit_breaker_opens` | CB opens after `failure_threshold` consecutive failures |
| `test_circuit_breaker_half_open` | CB transitions to HALF_OPEN after `recovery_timeout` |
| `test_tracer_critical_path` | `find_critical_path(spans)` returns the span with max duration |
| `test_histogram_percentiles` | `p99()` on known distribution within 5% of expected |

### Scaffolding to get started

The following is a starting skeleton — **you must complete all gaps marked `# TODO`**.

```python
# incident_lab/diagnostics/tracer.py
"""Lightweight distributed tracer — no external deps."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    service: str
    operation: str
    start_ms: float
    end_ms: float = 0.0
    status: str = "ok"  # "ok" | "error"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def finish(self, status: str = "ok") -> None:
        self.end_ms = time.monotonic() * 1000
        self.status = status


class Tracer:
    def __init__(self) -> None:
        self._spans: list[Span] = []

    def start_span(
        self,
        service: str,
        operation: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Span:
        span = Span(
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=parent_span_id,
            service=service,
            operation=operation,
            start_ms=time.monotonic() * 1000,
        )
        self._spans.append(span)
        return span

    def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def find_critical_path(self, spans: list[Span]) -> Span:
        # TODO: return the span with maximum duration_ms
        raise NotImplementedError


# incident_lab/diagnostics/metrics.py
"""Counter and Histogram — no external deps."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Counter:
    name: str
    _value: int = 0

    def inc(self, delta: int = 1) -> None:
        self._value += delta

    @property
    def value(self) -> int:
        return self._value


@dataclass
class Histogram:
    name: str
    _observations: list[float] = field(default_factory=list)

    def observe(self, value: float) -> None:
        self._observations.append(value)

    def p50(self) -> float:
        # TODO: implement
        raise NotImplementedError

    def p99(self) -> float:
        # TODO: implement
        raise NotImplementedError

    def mean(self) -> float:
        if not self._observations:
            return 0.0
        return sum(self._observations) / len(self._observations)
```

```python
# incident_lab/services/order_processor.py
"""Order processor with seeded pool exhaustion bug."""

from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

log = logging.getLogger("order_processor")


@dataclass
class DbConnection:
    conn_id: int


@dataclass
class ConnectionPool:
    max_size: int = 5
    timeout_s: float = 2.0
    _pool: queue.Queue = field(default_factory=queue.Queue)

    def __post_init__(self) -> None:
        for i in range(self.max_size):
            self._pool.put(DbConnection(i))

    def acquire_raw(self) -> DbConnection:
        """BUG path — raw acquire without guaranteed release."""
        return self._pool.get(timeout=self.timeout_s)

    @contextmanager
    def acquire(self) -> Generator[DbConnection, None, None]:
        """FIXED path — context manager guarantees release."""
        # TODO: implement with try/finally
        raise NotImplementedError


def process_order_buggy(pool: ConnectionPool, order_id: str) -> str:
    """BUG: drops connection on failure path."""
    conn = pool.acquire_raw()
    time.sleep(0.02)
    if hash(order_id) % 4 == 0:
        # BUG: returns without releasing conn
        return f"{order_id}: declined"
    pool._pool.put(conn)
    return f"{order_id}: ok"


def process_order_fixed(pool: ConnectionPool, order_id: str) -> str:
    """FIXED: context manager ensures release."""
    # TODO: use pool.acquire() context manager
    raise NotImplementedError


def run_buggy(n: int = 20) -> dict:
    pool = ConnectionPool(max_size=5)
    results = []
    errors = []

    def task(i: int) -> None:
        try:
            r = process_order_buggy(pool, f"ORD-{i:04d}")
            results.append(r)
        except queue.Empty:
            errors.append(f"ORD-{i:04d}: pool timeout")

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "pool_remaining": pool._pool.qsize(),
        "timeout_errors": len(errors),
        "processed": len(results),
    }


def run_fixed(n: int = 20) -> dict:
    pool = ConnectionPool(max_size=5)
    results = []
    errors = []

    def task(i: int) -> None:
        try:
            r = process_order_fixed(pool, f"ORD-{i:04d}")
            results.append(r)
        except queue.Empty:
            errors.append(f"ORD-{i:04d}: pool timeout")

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return {
        "pool_remaining": pool._pool.qsize(),
        "timeout_errors": len(errors),
        "processed": len(results),
    }
```

## Deliverables

Submit (or self-check) the following:

1. All source files under `incident_lab/` with no `# TODO` remaining
2. `pytest tests/ -v` output showing ≥ 90% passing
3. A brief `NOTES.md` (max 1 page) with:
   - One sentence per bug: what it was and what the fix was
   - The single most surprising thing you discovered while debugging

## Rubric

| Dimension | Weight | Pass bar |
|-----------|--------|----------|
| All TODOs implemented and tests pass | 40% | ≥ 90% tests green |
| Circuit breaker state machine correct | 20% | All three state transitions covered |
| Tracer + metrics diagnostics usable | 20% | `find_critical_path` and `p99` pass tests |
| Code quality (types, logging, no secrets) | 10% | ruff clean, no `print`, no hardcoded creds |
| NOTES.md clarity | 10% | Each bug described in ≤ 2 sentences |

(Solutions: ../solutions/week-09/project.md)
