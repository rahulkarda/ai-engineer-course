# Solutions — Week 09 · Day 7 (Project)

Full reference implementation for `incident_lab/`.

## diagnostics/tracer.py — complete implementation

```python
"""incident_lab/diagnostics/tracer.py"""

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
    status: str = "ok"
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
        if not spans:
            raise ValueError("No spans provided")
        return max(spans, key=lambda s: s.duration_ms)
```

## diagnostics/metrics.py — complete implementation

```python
"""incident_lab/diagnostics/metrics.py"""

from __future__ import annotations

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

    def _quantile(self, p: float) -> float:
        if not self._observations:
            raise ValueError("No observations")
        sorted_vals = sorted(self._observations)
        idx = (p / 100) * (len(sorted_vals) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        frac = idx - lower
        return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac

    def p50(self) -> float:
        return self._quantile(50)

    def p99(self) -> float:
        return self._quantile(99)

    def mean(self) -> float:
        if not self._observations:
            return 0.0
        return sum(self._observations) / len(self._observations)
```

## diagnostics/circuit_breaker.py — complete implementation

```python
"""incident_lab/diagnostics/circuit_breaker.py"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitOpenError(Exception):
    pass


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout_s: float = 10.0
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _last_failure: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure > self.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise CircuitOpenError(f"Circuit {self.name} is OPEN")

        try:
            result = fn()
        except Exception:
            with self._lock:
                self._failures += 1
                self._last_failure = time.monotonic()
                if (self._state == CircuitState.HALF_OPEN or
                        self._failures >= self.failure_threshold):
                    self._state = CircuitState.OPEN
            raise
        else:
            with self._lock:
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                self._failures = 0
            return result
```

## services/order_processor.py — complete implementation

```python
"""incident_lab/services/order_processor.py"""

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
        return self._pool.get(timeout=self.timeout_s)

    @contextmanager
    def acquire(self) -> Generator[DbConnection, None, None]:
        try:
            conn = self._pool.get(timeout=self.timeout_s)
        except queue.Empty:
            raise TimeoutError(f"Pool exhausted ({self.max_size} connections)")
        try:
            yield conn
        finally:
            self._pool.put(conn)


def process_order_buggy(pool: ConnectionPool, order_id: str) -> str:
    conn = pool.acquire_raw()
    time.sleep(0.01)
    if hash(order_id) % 4 == 0:
        return f"{order_id}: declined"  # BUG: conn not returned
    pool._pool.put(conn)
    return f"{order_id}: ok"


def process_order_fixed(pool: ConnectionPool, order_id: str) -> str:
    with pool.acquire() as conn:
        time.sleep(0.01)
        if hash(order_id) % 4 == 0:
            return f"{order_id}: declined"
        return f"{order_id}: ok"


def run_buggy(n: int = 20) -> dict:
    pool = ConnectionPool(max_size=5)
    results, errors = [], []

    def task(i: int) -> None:
        try:
            results.append(process_order_buggy(pool, f"ORD-{i:04d}"))
        except queue.Empty:
            errors.append(i)

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"pool_remaining": pool._pool.qsize(), "timeout_errors": len(errors), "processed": len(results)}


def run_fixed(n: int = 20) -> dict:
    pool = ConnectionPool(max_size=5)
    results, errors = [], []

    def task(i: int) -> None:
        try:
            results.append(process_order_fixed(pool, f"ORD-{i:04d}"))
        except TimeoutError:
            errors.append(i)

    threads = [threading.Thread(target=task, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"pool_remaining": pool._pool.qsize(), "timeout_errors": len(errors), "processed": len(results)}
```

## tests/test_order_processor.py

```python
"""tests/test_order_processor.py"""

import pytest
from incident_lab.services.order_processor import ConnectionPool, run_buggy, run_fixed


def test_pool_exhaustion_buggy():
    result = run_buggy(n=20)
    # Buggy version loses connections — pool will not be full
    assert result["pool_remaining"] < 5 or result["timeout_errors"] > 0


def test_pool_exhaustion_fixed():
    result = run_fixed(n=20)
    # Fixed version — all connections returned, no timeouts
    assert result["pool_remaining"] == 5
    assert result["timeout_errors"] == 0
```

## tests/test_diagnostics.py

```python
"""tests/test_diagnostics.py"""

import time
import pytest
from incident_lab.diagnostics.tracer import Tracer, Span
from incident_lab.diagnostics.metrics import Histogram, Counter
from incident_lab.diagnostics.circuit_breaker import CircuitBreaker, CircuitState, CircuitOpenError


def test_find_critical_path():
    tracer = Tracer()
    now = time.monotonic() * 1000
    spans = [
        Span("t1", "s1", None, "svc-a", "op-a", now, now + 10),
        Span("t1", "s2", "s1", "svc-b", "op-b", now, now + 250),
        Span("t1", "s3", "s1", "svc-c", "op-c", now, now + 50),
    ]
    critical = tracer.find_critical_path(spans)
    assert critical.span_id == "s2"
    assert critical.duration_ms == pytest.approx(250, abs=1)


def test_histogram_p99():
    h = Histogram("latency")
    for i in range(1, 101):
        h.observe(float(i))  # 1..100
    # p99 of 1..100 should be near 99
    assert 97 <= h.p99() <= 100


def test_histogram_p50():
    h = Histogram("latency")
    for i in range(1, 101):
        h.observe(float(i))
    assert 49 <= h.p50() <= 51


def test_circuit_breaker_opens():
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout_s=60)
    for _ in range(3):
        with pytest.raises(Exception):
            cb.call(lambda: (_ for _ in ()).throw(ConnectionError("fail")))
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_fast_fail():
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout_s=60)
    with pytest.raises(Exception):
        cb.call(lambda: (_ for _ in ()).throw(ConnectionError("fail")))
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "ok")


def test_circuit_breaker_half_open():
    import time
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout_s=0.05)
    with pytest.raises(Exception):
        cb.call(lambda: (_ for _ in ()).throw(ConnectionError("fail")))
    assert cb.state == CircuitState.OPEN
    time.sleep(0.1)
    result = cb.call(lambda: "recovered")
    assert result == "recovered"
    assert cb.state == CircuitState.CLOSED
```
