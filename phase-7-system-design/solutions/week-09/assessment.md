# Solutions — Week 09 Assessment

## Conceptual answers

**1. Five-step debugging method:**
- **Observe**: Gather raw evidence before touching anything — prevents premature action on incomplete data
- **Hypothesize**: Form ranked hypotheses — prevents tunnel vision by keeping multiple candidates live
- **Isolate**: Find the boundary between good and bad behavior — narrows the search space to one component
- **Verify**: Confirm or falsify each hypothesis with a targeted probe — prevents false fixes
- **Fix & Validate**: Apply minimal change and confirm symptom resolution — prevents regression

**2. USE vs RED:**
- USE (Utilization, Saturation, Errors) applies to *infrastructure* resources: CPU, memory, network, disk
- RED (Rate, Errors, Duration) applies to *service endpoints*: requests, failures, latency
- Apply RED first when p99 latency rises — it shows which service endpoints are affected. Then drill into USE on the infrastructure layers of those services to find resource pressure.

**3. Circuit breaker HALF_OPEN:**
- Trigger: circuit was OPEN for `recovery_timeout_s` seconds — it transitions to HALF_OPEN to probe
- Next transitions: (a) the probe call succeeds → CLOSED (service recovered); (b) the probe call fails → OPEN again (service still down), resetting the timeout

**4. 8 GB RSS, 200 MB active data — three patterns:**
- Unbounded cache dict accumulating unique keys (diagnose: `tracemalloc` top type = user-defined class held by dict values)
- Event listeners / callbacks registering without deregistering (diagnose: `gc.get_referrers` on the dict)
- LRU cache with unlimited size on a high-cardinality key (diagnose: dict key type and distribution)

**5. First three metrics for cost spike (3 → 31 mUSD):**
- `input_tokens` per request — context bloat is the most common cause of 10× cost increase
- `output_tokens` per request — runaway generation or tool loop
- `request_count` — unexpected volume spike

**6. "Engineer forgot" is not a valid root cause:**
Unacceptable — it stops at a person, not a system gap, which means no action item can prevent recurrence (you cannot stop humans from forgetting). Better endpoint: "The migration CI pipeline has no check that prevents index drops without a query-dependency audit." Action item targets the process, not the person.

**7. Thundering herd and double-checked locking:**
A thundering herd is N concurrent clients all attempting to recompute an expired cache value simultaneously, multiplying backend load by N. Double-checked locking prevents it: the first check (outside the lock) short-circuits the common case (still valid); the second check (inside the lock) ensures only one thread recomputes after taking the mutex — subsequent threads find the value already refreshed.

**8. Mitigation before fix:**
Mitigation stops customer impact; a fix addresses root cause. Example: during a database full-table-scan incident, the mitigation is to turn off the feature flag (stops the bleeding in seconds) while the fix is to create the missing index (takes minutes to build CONCURRENTLY without locking). Applying the fix first means customers experience 5+ minutes of 504s while the index builds; applying the mitigation first restores service in < 1 minute.

---

## Coding challenge — reference implementation

```python
"""
assessment_solution.py — ProductionDiagnosticSuite

All LLM calls mocked. No ANTHROPIC_API_KEY required.
Run: pytest assessment_solution.py -v
"""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Generator, TypeVar

log = logging.getLogger("diagnostic_suite")
T = TypeVar("T")

# API key placeholder — never hardcoded
_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── Connection Pool ────────────────────────────────────────────────────────

@dataclass
class FakeConn:
    conn_id: int


@dataclass
class Pool:
    max_size: int = 10
    timeout_s: float = 2.0
    _pool: queue.Queue = field(default_factory=queue.Queue)

    def __post_init__(self) -> None:
        for i in range(self.max_size):
            self._pool.put(FakeConn(i))

    @contextmanager
    def acquire(self) -> Generator[FakeConn, None, None]:
        try:
            conn = self._pool.get(timeout=self.timeout_s)
        except queue.Empty:
            raise TimeoutError("Pool exhausted")
        try:
            yield conn
        finally:
            self._pool.put(conn)

    @property
    def available(self) -> int:
        return self._pool.qsize()


# ── Circuit Breaker ────────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CBOpenError(Exception):
    pass


@dataclass
class CB:
    threshold: int = 5
    recovery_s: float = 10.0
    _state: CBState = field(default=CBState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _last_fail: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CBState:
        return self._state

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            if self._state == CBState.OPEN:
                if time.monotonic() - self._last_fail > self.recovery_s:
                    self._state = CBState.HALF_OPEN
                else:
                    raise CBOpenError("circuit open")
        try:
            result = fn()
        except CBOpenError:
            raise
        except Exception:
            with self._lock:
                self._failures += 1
                self._last_fail = time.monotonic()
                if self._failures >= self.threshold or self._state == CBState.HALF_OPEN:
                    self._state = CBState.OPEN
            raise
        else:
            with self._lock:
                self._state = CBState.CLOSED
                self._failures = 0
            return result


# ── Histogram ─────────────────────────────────────────────────────────────

class Histogram:
    def __init__(self) -> None:
        self._data: list[float] = []

    def record(self, v: float) -> None:
        self._data.append(v)

    def _q(self, p: float) -> float:
        if not self._data:
            return 0.0
        s = sorted(self._data)
        idx = (p / 100) * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)

    def p99(self) -> float:
        return self._q(99)


# ── Main Suite ────────────────────────────────────────────────────────────

@dataclass
class ProductionDiagnosticSuite:
    pool_size: int = 10
    cb_threshold: int = 5
    _pool: Pool = field(init=False)
    _cb: CB = field(init=False)
    _latency: Histogram = field(default_factory=Histogram, init=False)
    _total_requests: int = field(default=0, init=False)
    _error_count: int = field(default=0, init=False)
    _fast_fail_count: int = field(default=0, init=False)
    _total_cost: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._pool = Pool(max_size=self.pool_size)
        self._cb = CB(threshold=self.cb_threshold)

    def _mock_llm_cost(self) -> float:
        """Simulate LLM cost — uses env key pattern."""
        _ = _API_KEY  # referenced but not used in mock
        tokens = random.randint(100, 500)
        return tokens * 3e-6

    def submit_request(self, request_id: str, should_fail: bool = False) -> dict:
        start = time.monotonic()
        status = "ok"
        cost = 0.0

        try:
            with self._pool.acquire():
                def work() -> None:
                    if should_fail:
                        raise ConnectionError("downstream down")
                    time.sleep(random.uniform(0.005, 0.02))

                self._cb.call(work)
                cost = self._mock_llm_cost()

        except CBOpenError:
            status = "fast-fail"
            with self._lock:
                self._fast_fail_count += 1
        except Exception:
            status = "error"
            with self._lock:
                self._error_count += 1

        latency_ms = (time.monotonic() - start) * 1000
        self._latency.record(latency_ms)
        self._total_cost += cost
        with self._lock:
            self._total_requests += 1

        return {
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2),
            "status": status,
            "cost": round(cost, 6),
        }

    def get_metrics(self) -> dict:
        total = max(self._total_requests, 1)
        return {
            "error_rate": (self._error_count + self._fast_fail_count) / total,
            "p99_latency_ms": round(self._latency.p99(), 2),
            "pool_available": self._pool.available,
            "circuit_state": self._cb.state.name,
            "total_cost_usd": round(self._total_cost, 6),
        }

    def run_rca(self) -> str:
        metrics = self.get_metrics()
        parts = [
            f"Diagnostic summary: circuit={metrics['circuit_state']}, "
            f"error_rate={metrics['error_rate']:.1%}, "
            f"p99_latency={metrics['p99_latency_ms']:.0f}ms, "
            f"pool_available={metrics['pool_available']}/{self.pool_size}."
        ]
        if metrics["circuit_state"] == "OPEN":
            parts.append(
                "Circuit is OPEN — downstream dependency is failing; "
                "investigate dependency health and consider rollback."
            )
        if metrics["error_rate"] > 0.05:
            parts.append(
                f"Error rate {metrics['error_rate']:.1%} exceeds 5% SLO — "
                "check downstream logs for root cause."
            )
        if metrics["p99_latency_ms"] > 1000:
            parts.append(
                "p99 latency > 1 s — possible connection pool contention or slow query."
            )
        return " ".join(parts)


# ── Tests ─────────────────────────────────────────────────────────────────

import pytest


def test_pool_invariant():
    suite = ProductionDiagnosticSuite(pool_size=5)
    for i in range(20):
        suite.submit_request(f"R-{i:04d}")
    metrics = suite.get_metrics()
    assert metrics["pool_available"] == 5


def test_circuit_opens():
    suite = ProductionDiagnosticSuite(pool_size=5, cb_threshold=3)
    for i in range(5):
        suite.submit_request(f"R-{i:04d}", should_fail=True)
    metrics = suite.get_metrics()
    assert metrics["circuit_state"] == "OPEN"


def test_fast_fail_after_open():
    suite = ProductionDiagnosticSuite(pool_size=5, cb_threshold=2)
    for i in range(2):
        suite.submit_request(f"R-{i:04d}", should_fail=True)
    result = suite.submit_request("R-ff")
    assert result["status"] == "fast-fail"


def test_p99_accuracy():
    suite = ProductionDiagnosticSuite(pool_size=10)
    for i in range(100):
        suite.submit_request(f"R-{i:04d}")
    metrics = suite.get_metrics()
    # p99 should be > 0 and < 500 ms (mock is 5–20 ms + overhead)
    assert 0 < metrics["p99_latency_ms"] < 500


def test_rca_non_empty():
    suite = ProductionDiagnosticSuite(pool_size=5, cb_threshold=2)
    for i in range(5):
        suite.submit_request(f"R-{i}", should_fail=True)
    rca = suite.run_rca()
    assert len(rca) > 0
    assert "circuit" in rca.lower() or "error" in rca.lower()
```
