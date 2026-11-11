# Solutions — Week 09 · Day 2

## Exercise 1 — Memory leak detector fixture

```python
import tracemalloc
import pytest


@pytest.fixture
def memory_leak_detector():
    """Fixture: run fn 100× and assert net allocation < 1 MB."""
    def check(fn, args=(), kwargs=None, max_delta_mb=1.0):
        kwargs = kwargs or {}
        tracemalloc.start(depth=3)
        snap1 = tracemalloc.take_snapshot()
        for _ in range(100):
            fn(*args, **kwargs)
        snap2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        stats = snap2.compare_to(snap1, "lineno")
        net_bytes = sum(s.size_diff for s in stats if s.size_diff > 0)
        net_mb = net_bytes / (1024 ** 2)
        assert net_mb < max_delta_mb, (
            f"Memory leak: net {net_mb:.2f} MB allocated over 100 iterations "
            f"(limit {max_delta_mb} MB)"
        )
    return check


def test_leaky_cache_leaks(memory_leak_detector):
    cache = LeakyCache()
    # This SHOULD fail because LeakyCache grows without bound
    with pytest.raises(AssertionError):
        memory_leak_detector(
            lambda: cache.set(f"key:{id(object())}", b"x" * 1024)
        )


def test_bounded_cache_does_not_leak(memory_leak_detector):
    cache = BoundedLRUCache(max_size=50)
    memory_leak_detector(
        lambda: cache.set(f"key:{id(object())}", b"x" * 1024)
    )
```

## Exercise 2 — Pool metrics

```python
from dataclasses import dataclass, field
import queue
import threading


@dataclass
class InstrumentedPool(ConnectionPool):
    _peak_checked_out: int = field(default=0, init=False)

    @contextmanager
    def acquire(self):
        try:
            conn = self._pool.get(timeout=self.acquire_timeout)
        except queue.Empty:
            raise TimeoutError(f"Pool exhausted")
        with self._lock:
            self._checked_out += 1
            if self._checked_out > self._peak_checked_out:
                self._peak_checked_out = self._checked_out
        try:
            yield conn
        finally:
            with self._lock:
                self._checked_out -= 1
            self._pool.put(conn)

    @property
    def peak_checked_out(self) -> int:
        return self._peak_checked_out

    @property
    def utilization_pct(self) -> float:
        return (self._peak_checked_out / self.max_size) * 100


def test_peak_checked_out():
    import time
    pool = InstrumentedPool(max_size=5)
    results = []

    def task():
        with pool.acquire() as conn:
            time.sleep(0.05)
            results.append("ok")

    threads = [threading.Thread(target=task) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert pool.peak_checked_out <= pool.max_size
    assert pool._pool.qsize() == pool.max_size  # all returned
```

## Exercise 3 — Probabilistic early expiry

```python
import math
import random
import threading
import time


class ProbabilisticCache:
    def __init__(self, ttl_s: float, beta: float = 5.0) -> None:
        self._value: str | None = None
        self._expires_at: float = 0.0
        self._ttl = ttl_s
        self._beta = beta
        self._lock = threading.Lock()
        self.backend_calls = 0

    def _should_refresh_early(self) -> bool:
        remaining = self._expires_at - time.monotonic()
        if remaining <= 0:
            return True
        probability = 1 - math.exp(-self._beta * (1 - remaining / self._ttl))
        return random.random() < probability

    def get(self) -> str:
        if not self._should_refresh_early():
            return self._value  # type: ignore
        with self._lock:
            if self._should_refresh_early():
                self.backend_calls += 1
                time.sleep(0.02)
                self._value = "refreshed"
                self._expires_at = time.monotonic() + self._ttl
        return self._value  # type: ignore


def test_probabilistic_vs_hard_expiry():
    # Hard expiry: up to 50 calls for 50 concurrent misses
    hard = SimpleCache(ttl_s=10)
    hard._expires_at = 0
    threads = [threading.Thread(target=hard.get_buggy) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert hard.backend_calls > 3, "Expected stampede with hard expiry"

    # Probabilistic: ≤ 3 calls for same workload
    prob = ProbabilisticCache(ttl_s=10, beta=5)
    prob._expires_at = 0
    threads2 = [threading.Thread(target=prob.get) for _ in range(50)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()
    assert prob.backend_calls <= 3, f"Expected ≤ 3 backend calls, got {prob.backend_calls}"
```
