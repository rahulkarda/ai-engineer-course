# Phase 7 · Week 07 · Day 4 — Concurrency in LLD

> Objective: Design thread-safe classes and understand Python concurrency primitives for LLD interviews  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]]

---

## Concept

Concurrency introduces **shared mutable state** — the root cause of most threading bugs.
When two threads read-modify-write the same object without coordination, results are
non-deterministic.  Interview LLD questions often add "must handle concurrent access" —
this section gives you the tools to respond confidently.

### Core problems

| Problem | Symptom | Fix |
|---|---|---|
| Race condition | Lost updates, wrong counts | Lock around read-modify-write |
| Deadlock | Threads wait forever on each other | Consistent lock order, timeouts |
| Priority inversion | Low-priority thread blocks high-priority | Lock-free structures where possible |
| Starvation | Writer always blocked by readers | Fair read-write lock |

### Python concurrency primitives

```
threading.Lock       -- mutual exclusion (non-reentrant)
threading.RLock      -- reentrant lock (same thread can acquire N times)
threading.Condition  -- wait/notify (built on Lock)
threading.Semaphore  -- count-limited access
queue.Queue          -- thread-safe FIFO (built-in); prefer over manual list + lock
asyncio              -- single-threaded cooperative concurrency (IO-bound, no GIL concern)
```

### The GIL — what it means for you

CPython has the Global Interpreter Lock: only one thread runs Python bytecode at a time.
Implications:
- CPU-bound threads do NOT run in true parallel — use `multiprocessing` or native extensions.
- IO-bound threads (network, disk) DO benefit from `threading` because the GIL is released
  during IO syscalls.
- `asyncio` avoids threads entirely; preferred for IO-heavy LLD systems (e.g. rate limiter).
- In interviews: mention GIL when asked about performance, but model correctness first.

---

## How it works internally

### Race condition demonstrated

```python
import threading
import time

counter = 0

def increment_unsafe(n: int) -> None:
    global counter
    for _ in range(n):
        # Three bytecode ops: LOAD, ADD, STORE — GIL can switch between any two
        counter += 1

threads = [threading.Thread(target=increment_unsafe, args=(100_000,)) for _ in range(5)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Expected: 500_000.  Actual: less, non-deterministic.
print(counter)
```

### Fix: Lock around read-modify-write

```python
import threading

_lock = threading.Lock()
counter = 0

def increment_safe(n: int) -> None:
    global counter
    for _ in range(n):
        with _lock:          # acquire on enter, release on exit (even on exception)
            counter += 1
```

### Thread-safe Singleton (double-checked locking)

```python
import threading
from typing import ClassVar

class Config:
    _instance: ClassVar["Config | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "Config":
        if cls._instance is None:              # first check (no lock — fast path)
            with cls._lock:
                if cls._instance is None:      # second check (inside lock — safe)
                    cls._instance = super().__new__(cls)
        return cls._instance
```

### Producer-Consumer with `queue.Queue`

```python
import queue, threading, logging, time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def producer(q: queue.Queue[int], items: int) -> None:
    for i in range(items):
        q.put(i)
        log.info("produced %d", i)
        time.sleep(0.01)
    q.put(None)          # sentinel to signal consumer to stop

def consumer(q: queue.Queue[int]) -> None:
    while True:
        item = q.get()
        if item is None:
            break
        log.info("consumed %d", item)
        q.task_done()
```

---

## Worked example

Thread-safe in-memory key-value store with a **read-write lock** pattern.
Multiple readers may proceed in parallel; a writer gets exclusive access.

```python
"""
thread_safe_kv.py — Thread-safe in-memory key-value store.

Read-write lock implemented with RLock + Condition so that:
  - Many concurrent readers proceed without blocking each other.
  - A writer waits for all readers to finish, then gets exclusive access.
  - Readers arriving while a writer is waiting are queued (prevents starvation).
"""

import logging
import threading
import time
from typing import Generic, TypeVar

log = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class ReadWriteLock:
    """
    Fair read-write lock.

    Invariant: writers_waiting > 0 blocks new readers so writers
    do not starve behind a continuous stream of readers.
    """

    def __init__(self) -> None:
        self._cond: threading.Condition = threading.Condition(threading.Lock())
        self._readers: int = 0
        self._writer_active: bool = False
        self._writers_waiting: int = 0

    def acquire_read(self) -> None:
        with self._cond:
            # Block if a writer is active OR waiting (fairness)
            while self._writer_active or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()  # wake waiting writers

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()  # wake both readers and writers


class KVStore(Generic[K, V]):
    """
    Thread-safe in-memory key-value store.

    put() / delete() take an exclusive write lock.
    get() / contains() take a shared read lock.
    """

    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._rw: ReadWriteLock = ReadWriteLock()

    # --- write operations ---

    def put(self, key: K, value: V) -> None:
        """Insert or update a key."""
        self._rw.acquire_write()
        try:
            log.debug("put key=%s", key)
            self._data[key] = value
        finally:
            self._rw.release_write()

    def delete(self, key: K) -> bool:
        """Remove a key; return True if it existed."""
        self._rw.acquire_write()
        try:
            existed = key in self._data
            if existed:
                del self._data[key]
                log.debug("deleted key=%s", key)
            return existed
        finally:
            self._rw.release_write()

    # --- read operations ---

    def get(self, key: K) -> V | None:
        """Return value or None if absent."""
        self._rw.acquire_read()
        try:
            return self._data.get(key)
        finally:
            self._rw.release_read()

    def contains(self, key: K) -> bool:
        self._rw.acquire_read()
        try:
            return key in self._data
        finally:
            self._rw.release_read()

    def size(self) -> int:
        self._rw.acquire_read()
        try:
            return len(self._data)
        finally:
            self._rw.release_read()


# ---------------------------------------------------------------------------
# Stress test: concurrent readers + writers
# ---------------------------------------------------------------------------

def _stress_test() -> None:
    logging.basicConfig(
        level=logging.WARNING,  # reduce noise; set DEBUG to trace operations
        format="%(asctime)s %(threadName)s %(levelname)s %(message)s",
    )
    store: KVStore[str, int] = KVStore()

    errors: list[str] = []

    def writer(thread_id: int, iterations: int) -> None:
        for i in range(iterations):
            store.put(f"key-{i % 10}", i * thread_id)
            time.sleep(0)  # yield to scheduler

    def reader(iterations: int) -> None:
        for i in range(iterations):
            val = store.get(f"key-{i % 10}")
            _ = val  # consume without assertion (writers change concurrently)
            time.sleep(0)

    threads: list[threading.Thread] = [
        *[threading.Thread(target=writer, args=(tid, 500), name=f"W{tid}") for tid in range(3)],
        *[threading.Thread(target=reader, args=(1000,), name=f"R{rid}") for rid in range(5)],
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Stress test done. Final store size: {store.size()}")
    if errors:
        print(f"ERRORS: {errors}")
    else:
        print("No errors.")


if __name__ == "__main__":
    _stress_test()
```

---

## Exercises

1. **Make a class thread-safe.** Given a `Counter` class with `increment()`, `decrement()`,
   and `value()` methods that use a plain `int`, identify every race condition and add
   appropriate locking.  Verify with a stress test (10 threads, 100 000 ops each).

2. **Bounded buffer.** Implement `BoundedBuffer[T]` with `put(item)` (blocks when full) and
   `get()` (blocks when empty).  Use `threading.Condition` — do NOT use `queue.Queue`.
   Write a producer that puts 1 000 items and a consumer that gets them; assert all arrive.

3. **Atomic cache with expiry.** Extend `KVStore` to support a `ttl_seconds` per key.
   A background daemon thread should evict expired keys every second.
   Make the eviction loop thread-safe (it is a writer).

4. **Concurrent parking lot.** A `ParkingLot` has N spots.  Multiple `EntrySensor` threads
   call `enter()` concurrently; multiple `ExitSensor` threads call `exit()`.  Implement
   with a `Semaphore` so `enter()` blocks when the lot is full.

(Solutions: ../solutions/week-07/day-4.md)

---

## Self-check

- [ ] I can explain why `counter += 1` is not atomic in CPython
- [ ] I can write a thread-safe Singleton using double-checked locking
- [ ] I know when to use `Lock` vs `RLock` vs `Semaphore` vs `Condition`
- [ ] I can describe the GIL and its implications for IO-bound vs CPU-bound threads
- [ ] I can implement a bounded producer-consumer without using `queue.Queue`
- [ ] I can explain read-write lock semantics and why it improves throughput vs a plain `Lock`

---

## Supplementary

- [Python threading docs](https://docs.python.org/3/library/threading.html)
- [queue.Queue docs](https://docs.python.org/3/library/queue.html)
- [Real Python — Threading](https://realpython.com/intro-to-python-threading/)
- [Python GIL explained — David Beazley](https://www.youtube.com/watch?v=Obt-vMVdM8s)
- [Java Concurrency in Practice (Goetz) — concepts apply to Python too](https://jcip.net/)
