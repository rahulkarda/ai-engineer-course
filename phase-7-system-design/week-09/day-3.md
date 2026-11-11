# Phase 7 · Week 09 · Day 3 — Cascading Failures & Debugging Distributed Systems

> Objective: Understand how failures propagate across service boundaries, and apply circuit breakers, bulkheads, and distributed tracing to stop the cascade.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

A cascading failure is a failure in one component that causes failure in dependent components, which causes further failure — an amplifying chain. It is the dominant cause of large-scale outages.

**Propagation mechanisms:**

| Mechanism | Description | Example |
|-----------|-------------|---------|
| Retry storms | Failed callers retry, multiplying load on the struggling service | 100 clients retry 3×: 300× load spike |
| Timeout congestion | Slow upstream holds threads/goroutines; resource pool exhausts | DB slow → all app threads blocked → gateway queue fills |
| Error amplification | Partial failure at a leaf propagates as full failure at root | One of 10 shards down → 100% of queries fail |
| Resource contention | Degraded service uses CPU/memory to fail slowly, starving healthy co-tenants | OOM-triggered GC pauses block entire process |

**Three primary mitigations:**

1. **Circuit breaker**: After N failures, *open* the circuit — fail fast instead of waiting. After a timeout, probe with one request; close if it succeeds.
2. **Bulkhead**: Isolate thread pools / connection pools by downstream service so one slow dependency cannot exhaust resources shared with others.
3. **Timeout + retry budget**: Every network call has an explicit deadline. Retries are budget-constrained (e.g., 3 retries with exponential backoff + jitter).

## How it works internally

### Circuit breaker state machine

```
CLOSED ──(N consecutive failures)──► OPEN ──(timeout)──► HALF-OPEN
  ▲                                                            │
  └─────────────────(success)─────────────────────────────────┘
```

In `OPEN` state, calls return immediately with a `CircuitOpenError`. This prevents timeout congestion from propagating. The critical parameter is the *error threshold* — too low and you trip on transient errors; too high and the breaker doesn't protect.

### Distributed debugging without a trace backend

Even without Jaeger/Zipkin, you can reconstruct causality from logs by:
1. Correlating on `trace_id` (propagated in HTTP headers, e.g., `X-Trace-Id`)
2. Sorting by `timestamp` across services
3. Building a dependency graph: service A called B at T1; B returned error at T2; A logged error at T3

The gap `T3 - T1` = network + processing time. If `T2 - T1` is small but `T3 - T1` is large, the caller is retrying silently.

## Worked example

```python
"""
day3_cascading.py — circuit breaker + bulkhead + cascading failure demo

Run: python day3_cascading.py
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, TypeVar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("cascading")

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────

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
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def call(self, fn: Callable[[], T]) -> T:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time > self.recovery_timeout_s:
                    log.info("[%s] Circuit → HALF_OPEN (probing)", self.name)
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise CircuitOpenError(f"Circuit {self.name} is OPEN — failing fast")

        try:
            result = fn()
        except Exception as exc:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if (self._state == CircuitState.HALF_OPEN or
                    self._failure_count >= self.failure_threshold):
                self._state = CircuitState.OPEN
                log.warning("[%s] Circuit OPENED after %d failures", self.name, self._failure_count)

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                log.info("[%s] Circuit → CLOSED (probe succeeded)", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0


# ─────────────────────────────────────────────────────────────────────────────
# Bulkhead — separate thread pools per downstream
# ─────────────────────────────────────────────────────────────────────────────

class BulkheadExhaustedError(Exception):
    pass


@dataclass
class Bulkhead:
    name: str
    max_concurrent: int = 3
    _semaphore: threading.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self._semaphore = threading.Semaphore(self.max_concurrent)

    def call(self, fn: Callable[[], T], timeout: float = 1.0) -> T:
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise BulkheadExhaustedError(
                f"Bulkhead {self.name} exhausted ({self.max_concurrent} max concurrent)"
            )
        try:
            return fn()
        finally:
            self._semaphore.release()


# ─────────────────────────────────────────────────────────────────────────────
# Simulated services
# ─────────────────────────────────────────────────────────────────────────────

_failing = threading.Event()  # set → downstream is broken


def downstream_service(latency_s: float = 0.05) -> str:
    if _failing.is_set():
        time.sleep(0.5)  # slow fail
        raise ConnectionError("downstream: connection refused")
    time.sleep(latency_s)
    return "ok"


def client_without_breaker(client_id: int, results: list) -> None:
    """BUG: no circuit breaker — clients block on slow failures."""
    try:
        result = downstream_service()
        results.append(("ok", client_id))
    except Exception as exc:
        results.append(("error", client_id, str(exc)))


def client_with_breaker(
    client_id: int,
    cb: CircuitBreaker,
    results: list,
) -> None:
    """FIX: fails fast once circuit opens."""
    try:
        result = cb.call(downstream_service)
        results.append(("ok", client_id))
    except CircuitOpenError:
        results.append(("fast-fail", client_id))
    except Exception as exc:
        results.append(("error", client_id, str(exc)))


def demo_without_protection(n_clients: int = 20) -> None:
    log.info("=== Without circuit breaker ===")
    _failing.set()
    start = time.monotonic()
    results: list = []
    threads = [
        threading.Thread(target=client_without_breaker, args=(i, results))
        for i in range(n_clients)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    errors = sum(1 for r in results if r[0] == "error")
    log.info("Done in %.2f s — %d errors, %d ok", elapsed, errors, n_clients - errors)
    _failing.clear()


def demo_with_circuit_breaker(n_clients: int = 20) -> None:
    log.info("=== With circuit breaker ===")
    cb = CircuitBreaker("downstream", failure_threshold=3, recovery_timeout_s=30)
    _failing.set()
    start = time.monotonic()
    results: list = []
    threads = [
        threading.Thread(target=client_with_breaker, args=(i, cb, results))
        for i in range(n_clients)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start
    fast_fails = sum(1 for r in results if r[0] == "fast-fail")
    errors = sum(1 for r in results if r[0] == "error")
    log.info(
        "Done in %.2f s — %d fast-fail, %d actual errors, state=%s",
        elapsed, fast_fails, errors, cb.state.name,
    )
    _failing.clear()


def demo_bulkhead() -> None:
    log.info("=== Bulkhead isolation ===")
    db_bulkhead = Bulkhead("postgres", max_concurrent=3)
    cache_bulkhead = Bulkhead("redis", max_concurrent=10)

    results: list[str] = []
    lock = threading.Lock()

    def slow_db_call(req_id: int) -> None:
        try:
            def _work() -> str:
                time.sleep(0.2)  # simulate slow DB
                return "db-ok"
            res = db_bulkhead.call(_work, timeout=0.1)
            with lock:
                results.append(f"db-ok:{req_id}")
        except BulkheadExhaustedError:
            with lock:
                results.append(f"db-shed:{req_id}")

    threads = [threading.Thread(target=slow_db_call, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    shed = sum(1 for r in results if "shed" in r)
    ok = sum(1 for r in results if "ok" in r)
    log.info("DB bulkhead: %d passed, %d shed (max_concurrent=3)", ok, shed)


if __name__ == "__main__":
    demo_without_protection()
    demo_with_circuit_breaker()
    demo_bulkhead()
```

Key observations when you run it:
- Without the breaker: 20 clients × 0.5 s slow-fail ≈ 10 s total wall time (threads wait in parallel but each timeout is real)
- With the breaker: only 3 failures are real; remaining 17 fast-fail in microseconds
- The bulkhead: only 3 DB requests pass; 7 are shed immediately, protecting the rest of the system

## Exercises

1. Add a `half_open_probe_count` parameter to `CircuitBreaker` so it sends N probes before deciding to close. Currently it closes on the first success — this is too optimistic for flaky services.

2. Implement `retry_with_backoff(fn, max_retries=3, base_delay_s=0.1, jitter=True)` that retries with exponential backoff. Add a test that counts the actual number of calls to `fn` when it fails 2 times then succeeds.

3. Write a `trace_log_correlator(log_lines: list[str]) -> dict[str, list[dict]]` that groups structured JSON log lines by `trace_id` and sorts each group by `ts`. Test with a mixed list of 10 log lines from 3 different traces.

(Solutions: ../solutions/week-09/day-3.md)

## Self-check

- [ ] I can draw the circuit breaker state machine from memory
- [ ] I can explain how retry storms amplify load during a cascading failure
- [ ] I can implement a thread-safe circuit breaker in Python
- [ ] I understand bulkhead isolation and can implement it with a semaphore
- [ ] I can correlate distributed logs manually using trace_id

## Supplementary links

- [Netflix Hystrix concepts (circuit breaker)](https://github.com/Netflix/Hystrix/wiki/How-it-Works)
- [Release It! Design Patterns — Nygard](https://pragprog.com/titles/mnee2/release-it-second-edition/)
- [Google SRE — Handling Overload](https://sre.google/sre-book/handling-overload/)
