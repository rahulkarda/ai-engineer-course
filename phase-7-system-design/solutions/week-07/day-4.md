# Solutions — Phase 7 · Week 07 · Day 4

## Exercise 1 — Thread-safe Counter

Race condition: `counter += 1` compiles to three bytecode operations (LOAD_FAST, BINARY_ADD,
STORE_FAST).  The GIL can switch threads between any two operations, causing lost increments.
Fix: wrap the read-modify-write with a `threading.Lock`.

```python
"""
thread_safe_counter.py — Exercise 1 solution.

Demonstrates a thread-safe Counter class and a stress-test that would fail
on the unsafe version.
"""

import logging
import threading

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Counter:
    """Thread-safe counter using a plain Lock."""

    def __init__(self, initial: int = 0) -> None:
        self._value: int = initial
        self._lock: threading.Lock = threading.Lock()

    def increment(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def decrement(self, amount: int = 1) -> None:
        with self._lock:
            self._value -= amount

    def value(self) -> int:
        with self._lock:
            return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0


def _stress_test() -> None:
    """10 threads x 100 000 increments; assert final value == 1 000 000."""
    counter = Counter()
    n_threads = 10
    ops_per_thread = 100_000

    threads = [
        threading.Thread(
            target=lambda: [counter.increment() for _ in range(ops_per_thread)],
            name=f"T{i}",
        )
        for i in range(n_threads)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * ops_per_thread
    actual = counter.value()
    assert actual == expected, f"Expected {expected}, got {actual}"
    log.info("Stress test passed: counter = %d", actual)


if __name__ == "__main__":
    _stress_test()
```

**Key points:**
- `threading.Lock` (non-reentrant) is sufficient here; the lock is never re-acquired by the
  same thread within a single call.
- Reading `_value` also acquires the lock — a reader racing with a writer could observe a
  partially-committed value on architectures without strong memory ordering.
- `with self._lock:` is idiomatic; guarantees release even if an exception is raised.

---

## Exercise 2 — Bounded Buffer with `threading.Condition`

`threading.Condition` wraps a lock and adds `wait()` / `notify()` / `notify_all()`.  The
pattern is always: acquire, loop-wait on the predicate, mutate, notify, release.

```python
"""
bounded_buffer.py — Exercise 2 solution.

BoundedBuffer[T]: producers block when full; consumers block when empty.
Uses threading.Condition — NOT queue.Queue.
"""

import logging
import threading
import time
from typing import Generic, TypeVar

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

T = TypeVar("T")


class BoundedBuffer(Generic[T]):
    """
    Fixed-capacity thread-safe FIFO buffer.

    put(item) — blocks when buffer is full.
    get()     — blocks when buffer is empty.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._buffer: list[T] = []
        self._cond: threading.Condition = threading.Condition()

    def put(self, item: T) -> None:
        """Enqueue item.  Blocks until space is available."""
        with self._cond:
            while len(self._buffer) >= self._capacity:
                log.debug("put: buffer full (%d/%d), waiting", len(self._buffer), self._capacity)
                self._cond.wait()
            self._buffer.append(item)
            log.debug("put: added item, size now %d", len(self._buffer))
            self._cond.notify_all()  # wake blocked consumers

    def get(self) -> T:
        """Dequeue item.  Blocks until an item is available."""
        with self._cond:
            while not self._buffer:
                log.debug("get: buffer empty, waiting")
                self._cond.wait()
            item = self._buffer.pop(0)
            log.debug("get: removed item, size now %d", len(self._buffer))
            self._cond.notify_all()  # wake blocked producers
            return item

    def size(self) -> int:
        with self._cond:
            return len(self._buffer)


def _demo() -> None:
    """Producer puts 1 000 items; consumer gets them all; assert all arrive."""
    n_items = 1_000
    buf: BoundedBuffer[int] = BoundedBuffer(capacity=10)
    received: list[int] = []

    def producer() -> None:
        for i in range(n_items):
            buf.put(i)
            time.sleep(0)  # yield to scheduler

    def consumer() -> None:
        for _ in range(n_items):
            received.append(buf.get())

    t_prod = threading.Thread(target=producer, name="producer")
    t_cons = threading.Thread(target=consumer, name="consumer")

    t_cons.start()
    t_prod.start()
    t_prod.join()
    t_cons.join()

    assert received == list(range(n_items)), "Items lost or reordered"
    log.info("BoundedBuffer demo passed: received %d items in order", len(received))


if __name__ == "__main__":
    _demo()
```

**Key points:**
- `while` (not `if`) for the wait predicate — protects against spurious wake-ups (allowed by
  the POSIX spec and Python's implementation).
- `notify_all()` is safer than `notify()` when both producers and consumers share the same
  `Condition`; `notify()` could wake another producer instead of a waiting consumer.
- The `Condition` constructor is called without an argument, so it creates its own internal
  `Lock`.  Passing an `RLock` is also valid.

---

## Exercise 3 — TTL Cache with Background Eviction

```python
"""
ttl_cache.py — Exercise 3 solution.

KVStore extended with per-key TTL and a background daemon eviction thread.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _Entry(Generic[V]):
    value: V
    expires_at: float | None = field(default=None)  # time.monotonic() epoch, or None

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.monotonic() >= self.expires_at


class TTLCache(Generic[K, V]):
    """
    Thread-safe in-memory cache with optional per-key TTL.

    A background daemon thread evicts expired keys every `eviction_interval` seconds.
    Call stop() to shut down the background thread cleanly.
    """

    def __init__(self, eviction_interval: float = 1.0) -> None:
        self._data: dict[K, _Entry[V]] = {}
        self._lock: threading.RLock = threading.RLock()
        self._stop_event: threading.Event = threading.Event()
        self._eviction_interval = eviction_interval

        self._eviction_thread = threading.Thread(
            target=self._evict_loop,
            name="ttl-cache-eviction",
            daemon=True,
        )
        self._eviction_thread.start()

    # --- public API ---

    def put(self, key: K, value: V, ttl_seconds: float | None = None) -> None:
        expires_at: float | None = None
        if ttl_seconds is not None:
            expires_at = time.monotonic() + ttl_seconds
        entry: _Entry[V] = _Entry(value=value, expires_at=expires_at)
        with self._lock:
            self._data[key] = entry
            log.debug("put key=%s ttl=%s", key, ttl_seconds)

    def get(self, key: K) -> V | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._data[key]
                log.debug("lazy eviction: key=%s expired", key)
                return None
            return entry.value

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def size(self) -> int:
        """Count of non-expired entries."""
        with self._lock:
            return sum(1 for e in self._data.values() if not e.is_expired())

    def stop(self) -> None:
        """Signal the eviction thread to stop and wait for it."""
        self._stop_event.set()

    # --- internal ---

    def _evict_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._eviction_interval):
            with self._lock:
                n = self._delete_expired()
            if n:
                log.debug("background eviction: removed %d expired keys", n)

    def _delete_expired(self) -> int:
        """Delete all expired entries.  Caller MUST hold self._lock."""
        expired = [k for k, e in self._data.items() if e.is_expired()]
        for k in expired:
            del self._data[k]
        return len(expired)


def _demo() -> None:
    cache: TTLCache[str, str] = TTLCache(eviction_interval=0.1)

    cache.put("persistent", "stays")
    cache.put("short", "gone soon", ttl_seconds=0.1)

    assert cache.get("persistent") == "stays"
    assert cache.get("short") == "gone soon"

    time.sleep(0.3)

    assert cache.get("short") is None, "short-lived key should have expired"
    assert cache.get("persistent") == "stays", "persistent key should remain"

    cache.stop()
    log.info("TTLCache demo passed")


if __name__ == "__main__":
    _demo()
```

**Key points:**
- `threading.RLock` is used instead of `Lock` because `get()` may call `delete()` internally
  (lazy eviction).  With a plain `Lock`, the same thread trying to re-acquire would deadlock.
- Eviction thread uses `threading.Event.wait(timeout=...)` instead of `time.sleep()` so that
  `stop()` unblocks it immediately rather than waiting for the next sleep cycle.
- Two eviction paths coexist: lazy (on `get`) and background (periodic).  Both are safe
  because they share the same `RLock`.

---

## Exercise 4 — Concurrent Parking Lot with Semaphore

```python
"""
parking_lot.py — Exercise 4 solution.

ParkingLot uses threading.Semaphore so that enter() blocks when all spots
are occupied.  EntrySensor / ExitSensor threads drive concurrent access.
"""

import logging
import threading
import time
import random

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ParkingLot:
    """
    Parking lot with N spots.

    enter() — blocks until a spot is available, then occupies one.
    exit()  — releases a spot (never blocks).
    occupancy() — current number of occupied spots (for monitoring).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._semaphore: threading.Semaphore = threading.Semaphore(capacity)
        self._lock: threading.Lock = threading.Lock()
        self._occupied: int = 0

    def enter(self, vehicle_id: str) -> None:
        """Occupy one spot.  Blocks if lot is full."""
        log.debug("vehicle %s waiting for spot", vehicle_id)
        self._semaphore.acquire()          # blocks when count == 0
        with self._lock:
            self._occupied += 1
        log.info("vehicle %s entered  (occupied=%d/%d)", vehicle_id, self._occupied, self._capacity)

    def exit(self, vehicle_id: str) -> None:
        """Release one spot.  Never blocks."""
        with self._lock:
            if self._occupied <= 0:
                raise RuntimeError("exit() called on empty lot")
            self._occupied -= 1
        self._semaphore.release()          # increment semaphore count
        log.info("vehicle %s exited   (occupied=%d/%d)", vehicle_id, self._occupied, self._capacity)

    def occupancy(self) -> int:
        with self._lock:
            return self._occupied


def _demo() -> None:
    lot = ParkingLot(capacity=5)
    errors: list[str] = []

    def sensor(vehicle_id: str, park_seconds: float) -> None:
        try:
            lot.enter(vehicle_id)
            time.sleep(park_seconds)
            lot.exit(vehicle_id)
        except Exception as exc:
            errors.append(str(exc))

    # Launch 20 vehicles competing for 5 spots
    vehicles = [
        threading.Thread(
            target=sensor,
            args=(f"V{i:02d}", random.uniform(0.01, 0.1)),
            name=f"sensor-{i}",
        )
        for i in range(20)
    ]

    for v in vehicles:
        v.start()
    for v in vehicles:
        v.join()

    assert errors == [], f"Errors during concurrent parking: {errors}"
    assert lot.occupancy() == 0, "Lot should be empty after all vehicles exit"
    log.info("ParkingLot stress test passed")


if __name__ == "__main__":
    _demo()
```

**Key points:**
- `threading.Semaphore(N)` starts with a count of N.  `acquire()` decrements (blocks at 0);
  `release()` increments.
- A separate `Lock` guards `_occupied` because `Semaphore` does not expose its internal count
  in a thread-safe way for external observation.
- `Semaphore` (not `BoundedSemaphore`) is used; `BoundedSemaphore` would raise on `release()`
  above the initial count, which is useful for detecting bugs but adds overhead.
- `enter()` acquires the semaphore first, then increments `_occupied`.  The order is important:
  decrementing `_occupied` before releasing the semaphore would allow a race where `occupancy()`
  shows a count below 0 momentarily.
