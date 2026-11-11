# Phase 7 · Week 09 · Day 2 — Common Failure Patterns: Memory Leaks, Connection Pool Exhaustion, Thundering Herd

> Objective: Recognize, reproduce, and fix the three most common resource-exhaustion failure patterns in production Python services.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Most production outages are not exotic bugs — they are predictable resource-exhaustion patterns that emerge under load. Understanding their signatures lets you diagnose them in minutes instead of hours.

### Memory leaks

A memory leak in Python usually means an **unbounded accumulating reference** — a cache, list, dict, or closure that grows without a ceiling and is never freed. Because CPython uses reference-counting with a cyclic GC, objects with circular references survive until a GC cycle, but long-lived containers that hold references prevent the GC from ever collecting the held objects.

Key signals: resident memory (`RSS`) climbs monotonically, GC collection counts rise, `tracemalloc` shows a single type dominating allocations.

### Connection pool exhaustion

Every database, HTTP, or message-broker client opens a TCP connection. Connection pools cap the number of open sockets. When all connections are checked out and not returned (due to missing `finally` blocks, exception swallowing, or long-running transactions), new requests block waiting for a slot. The pool queue fills; requests time out; the service appears to hang.

Key signals: `waiting_for_connection` metric > 0, request latency rises with a step function when concurrency hits `pool_size`, DB shows many `idle in transaction` connections.

### Thundering herd

When a cache expires (or a service restarts), every waiting client fires simultaneously. The backend receives a burst N× the steady-state rate for the cache TTL window. If the backend cannot absorb the burst, it falls over; the cache never warms; the burst repeats.

Key signals: periodic latency spikes aligned with TTL, error rate spike immediately after deploys or cache flushes, backend CPU spikes in sawtooth pattern.

## How it works internally

### tracemalloc mechanics

`tracemalloc` hooks into CPython's memory allocator at the C level. Each allocation records the Python call stack at a configurable depth. `take_snapshot()` captures the current live allocation table. Comparing two snapshots reveals net new allocations — the delta is your leak.

### Pool queue dynamics

A pool with `max_size=10` and `queue_timeout=5s` has a service capacity of `10 / avg_query_time`. If `avg_query_time = 0.5 s`, max throughput is 20 RPS. At 21 RPS, the queue grows without bound, and requests time out after 5 s. The pool is a *queuing system* — Little's Law: `L = λW` (queue length = arrival rate × wait time).

### Cache stampede probability

For a cache with TTL `T` and `N` clients polling every `t` seconds, the expected number of simultaneous misses after expiry = `min(N, T/t)`. Mitigations: *probabilistic early expiry* (refresh before expiry with probability `1 - e^{-β·(remaining/T)}`), *background refresh*, *staggered TTLs*.

## Worked example

```python
"""
day2_failure_patterns.py — reproduce and detect three failure patterns

Run with: python day2_failure_patterns.py
Each section can be commented out independently.
"""

from __future__ import annotations

import gc
import logging
import queue
import random
import threading
import time
import tracemalloc
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("patterns")


# ─────────────────────────────────────────────────────────────────────────────
# 1. MEMORY LEAK — unbounded cache
# ─────────────────────────────────────────────────────────────────────────────

class LeakyCache:
    """BUG: cache grows without bound."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._store.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._store[key] = value  # no eviction policy


class BoundedLRUCache:
    """FIX: evict oldest entries when capacity exceeded."""

    def __init__(self, max_size: int = 1000) -> None:
        from collections import OrderedDict
        self._store: OrderedDict[str, bytes] = OrderedDict()
        self._max = max_size

    def get(self, key: str) -> bytes | None:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, key: str, value: bytes) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._max:
            self._store.popitem(last=False)


def demo_memory_leak() -> None:
    log.info("=== Memory leak demo ===")
    tracemalloc.start(depth=5)

    snap1 = tracemalloc.take_snapshot()
    cache = LeakyCache()

    # Simulate 10 000 unique keys — each holds 1 KB
    for i in range(10_000):
        cache.set(f"key:{i}", b"x" * 1024)

    snap2 = tracemalloc.take_snapshot()
    top = snap2.compare_to(snap1, "lineno")[:3]
    log.info("Top allocation deltas:")
    for stat in top:
        log.info("  %s", stat)

    rss_mb = len(cache._store) * 1024 / (1024 ** 2)
    log.info("LeakyCache size: %.1f MB (%d entries)", rss_mb, len(cache._store))

    # Fix
    fixed = BoundedLRUCache(max_size=500)
    for i in range(10_000):
        fixed.set(f"key:{i}", b"x" * 1024)
    log.info("BoundedLRUCache size: %d entries (capped at 500)", len(fixed._store))
    tracemalloc.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONNECTION POOL EXHAUSTION
# ─────────────────────────────────────────────────────────────────────────────

class FakeDB:
    """Simulates a DB connection with configurable query time."""

    def __init__(self, conn_id: int) -> None:
        self.conn_id = conn_id
        self.in_use = False

    def query(self, sql: str, sleep_s: float = 0.1) -> list[dict]:
        time.sleep(sleep_s)
        return [{"row": 1}]


@dataclass
class ConnectionPool:
    max_size: int = 5
    acquire_timeout: float = 2.0
    _pool: queue.Queue = field(default_factory=queue.Queue)
    _checked_out: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        for i in range(self.max_size):
            self._pool.put(FakeDB(i))

    @contextmanager
    def acquire(self) -> Generator[FakeDB, None, None]:
        try:
            conn = self._pool.get(timeout=self.acquire_timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Pool exhausted: all {self.max_size} connections in use"
            )
        with self._lock:
            self._checked_out += 1
        try:
            yield conn
        finally:
            with self._lock:
                self._checked_out -= 1
            self._pool.put(conn)  # always returned


def buggy_request(pool: ConnectionPool, request_id: int) -> None:
    """BUG: connection never returned on exception."""
    try:
        conn = pool._pool.get(timeout=pool.acquire_timeout)  # raw get — no context mgr
        conn.query("SELECT 1", sleep_s=0.3)
        # If an exception occurs here, conn is never returned to the pool
        if request_id % 3 == 0:
            raise ValueError("simulated business error")
        pool._pool.put(conn)
    except ValueError:
        log.warning("Request %d: business error — connection LEAKED", request_id)


def fixed_request(pool: ConnectionPool, request_id: int) -> None:
    """FIX: context manager guarantees return."""
    try:
        with pool.acquire() as conn:
            conn.query("SELECT 1", sleep_s=0.05)
            if request_id % 3 == 0:
                raise ValueError("simulated business error")
    except (ValueError, TimeoutError) as exc:
        log.warning("Request %d: %s", request_id, exc)


def demo_pool_exhaustion() -> None:
    log.info("=== Pool exhaustion demo ===")
    pool = ConnectionPool(max_size=5)

    # Buggy path — will exhaust pool after ~15 requests
    threads = [
        threading.Thread(target=buggy_request, args=(pool, i))
        for i in range(12)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    remaining = pool._pool.qsize()
    log.info("After buggy requests: %d/%d connections in pool", remaining, pool.max_size)

    # Refill for the fixed demo
    while pool._pool.qsize() < pool.max_size:
        pool._pool.put(FakeDB(99))

    fixed_pool = ConnectionPool(max_size=5)
    threads2 = [
        threading.Thread(target=fixed_request, args=(fixed_pool, i))
        for i in range(20)
    ]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()
    log.info("After fixed requests: %d/%d connections in pool", fixed_pool._pool.qsize(), fixed_pool.max_size)


# ─────────────────────────────────────────────────────────────────────────────
# 3. THUNDERING HERD
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCache:
    def __init__(self, ttl_s: float) -> None:
        self._value: str | None = None
        self._expires_at: float = 0.0
        self._ttl = ttl_s
        self._lock = threading.Lock()
        self.backend_calls = 0

    def _is_expired(self) -> bool:
        return time.monotonic() > self._expires_at

    def get_buggy(self) -> str:
        """BUG: no stampede protection — all threads hit backend simultaneously."""
        if self._is_expired():
            self.backend_calls += 1
            time.sleep(0.05)  # simulate slow backend
            self._value = f"data-{random.randint(0, 9999)}"
            self._expires_at = time.monotonic() + self._ttl
        return self._value  # type: ignore[return-value]

    def get_fixed(self) -> str:
        """FIX: mutex prevents stampede; only first thread recomputes."""
        if not self._is_expired():
            return self._value  # type: ignore[return-value]
        with self._lock:
            # Double-check inside lock — another thread may have refreshed
            if self._is_expired():
                self.backend_calls += 1
                time.sleep(0.05)
                self._value = f"data-{random.randint(0, 9999)}"
                self._expires_at = time.monotonic() + self._ttl
        return self._value  # type: ignore[return-value]


def demo_thundering_herd() -> None:
    log.info("=== Thundering herd demo ===")

    buggy_cache = SimpleCache(ttl_s=0.2)
    buggy_cache._expires_at = 0  # force expired

    calls_before = buggy_cache.backend_calls
    threads = [
        threading.Thread(target=buggy_cache.get_buggy)
        for _ in range(50)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("Buggy cache: %d backend calls for 50 concurrent misses", buggy_cache.backend_calls)

    fixed_cache = SimpleCache(ttl_s=0.2)
    fixed_cache._expires_at = 0
    threads2 = [
        threading.Thread(target=fixed_cache.get_fixed)
        for _ in range(50)
    ]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()
    log.info("Fixed cache:  %d backend call(s) for 50 concurrent misses", fixed_cache.backend_calls)


if __name__ == "__main__":
    demo_memory_leak()
    demo_pool_exhaustion()
    demo_thundering_herd()
```

## Exercises

1. **Memory leak detector**: Write a pytest fixture that wraps a function call with `tracemalloc`, runs the function 100 times in a loop, and asserts that the net allocation delta is below 1 MB. Use it to test `LeakyCache` (should fail) and `BoundedLRUCache` (should pass).

2. **Pool metrics**: Extend `ConnectionPool` to track `peak_checked_out` (maximum simultaneous connections ever in use). Add a property `utilization_pct -> float`. Write a test that drives 20 concurrent requests and asserts `peak_checked_out <= max_size`.

3. **Probabilistic early expiry**: Implement the beta-refresh formula — instead of expiring exactly at TTL, refresh with probability `1 - e^{-β*(remaining_ttl/ttl)}` where `β=5`. Show that with 50 concurrent clients the backend call count stays ≤ 3 vs. up to 50 with the hard-expiry approach.

(Solutions: ../solutions/week-09/day-2.md)

## Self-check

- [ ] I can identify memory leak symptoms from tracemalloc output
- [ ] I can explain why a missing `finally` block exhausts a connection pool
- [ ] I can implement double-checked locking to prevent cache stampede
- [ ] I know what Little's Law says about pool queue depth under load
- [ ] I can write tests that reproduce each failure pattern in isolation

## Supplementary links

- [Python tracemalloc docs](https://docs.python.org/3/library/tracemalloc.html)
- [SQLAlchemy pool configuration](https://docs.sqlalchemy.org/en/20/core/pooling.html)
- [XFetch — optimal cache stampede prevention](https://cseweb.ucsd.edu/~avattani/papers/cache_stampede.pdf)
