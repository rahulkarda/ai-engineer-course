# Phase 2 · Week 06 · Day 3 — Concurrency Primitives & Deadlock

> Objective: Master mutexes, semaphores, condition variables, and deadlock analysis.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]]

## Concept

### Why Primitives?

Multiple threads sharing memory create **race conditions**: the final state depends on the precise interleaving of instructions. Concurrency primitives enforce ordering guarantees so correctness is independent of scheduling.

### Mutex (Mutual Exclusion Lock)

A mutex has two states: **locked** and **unlocked**. Only one thread can hold it at a time. Others block on `acquire()` until the holder calls `release()`.

```
Thread A         Thread B
lock(m)          lock(m)  ← blocks here
  critical section
unlock(m)        ← unblocks, Thread B proceeds
```

Use a mutex to protect a **critical section** — any read-modify-write of shared state.

### Semaphore

A semaphore holds an integer **count**. `acquire()` decrements (blocks if 0). `release()` increments (wakes a waiter).

- **Binary semaphore** (count 0/1) ≈ mutex, but signaling semantics differ — any thread can release.
- **Counting semaphore** — limit concurrent access to a resource pool (e.g., DB connection pool of size 10).

### Condition Variable

Used to **wait for a condition** to become true, atomically releasing the lock while waiting:

```
# Producer
with lock:
    queue.append(item)
    cond.notify()

# Consumer
with lock:
    while not queue:         # re-check — spurious wakeups
        cond.wait()          # atomically releases lock, sleeps
    item = queue.pop(0)
```

`cond.wait()` releases the lock and blocks. On `notify()`, it re-acquires the lock before returning. Always re-check the condition in a `while` loop (spurious wakeups).

### Read-Write Lock

Multiple readers can hold the lock simultaneously; a writer requires exclusive access. Improves throughput when reads heavily outnumber writes.

### Deadlock

Four **necessary** conditions (Coffman, 1971) — all must hold simultaneously:

| # | Condition | Meaning |
|---|-----------|---------|
| 1 | Mutual exclusion | Resource held by at most one thread |
| 2 | Hold and wait | Thread holds one resource while waiting for another |
| 3 | No preemption | Resources cannot be forcibly taken |
| 4 | Circular wait | Cycle in the resource-allocation graph |

Classic example:
```
Thread A: lock(m1) → lock(m2)
Thread B: lock(m2) → lock(m1)
→ A waits for m2 (held by B); B waits for m1 (held by A) → deadlock
```

**Prevention strategies:**
- **Lock ordering**: always acquire locks in a fixed global order (breaks circular wait).
- **Try-lock with timeout**: if second lock unavailable, release first and retry.
- **Lock-free structures**: eliminate locks via atomic operations (compare-and-swap).
- **Single-lock design**: coarse-grained but deadlock-free.

### Python's `threading` Module

Python's `threading.Lock` is a thin wrapper over POSIX `pthread_mutex`. `threading.RLock` is reentrant — the same thread can acquire it multiple times (counts releases).

`threading.Event` — simplified flag-based signaling (no associated data).

## How It Works Internally

### Futex (Fast Userspace Mutex)

Linux mutexes use **futexes** (fast userspace locking). A shared integer is checked via atomic CAS entirely in userspace when uncontended — no syscall. Only on contention does the thread invoke `futex(FUTEX_WAIT)` to sleep in the kernel. This makes uncontended lock/unlock cost ~3 ns.

### Memory Ordering

CPUs and compilers reorder instructions for performance. The C++ memory model (and Python's GIL) impose **acquire/release semantics** at lock boundaries: all writes before `release` are visible to a thread that subsequently `acquire`s.

Without this, a thread could read a stale cache line even after another thread updated it.

### GIL as a Special Mutex

CPython's GIL is a single global mutex. It is released:
- Every 100 bytecodes (Python 3.1 and below).
- Every 5 ms (`sys.getswitchinterval()`, default, Python 3.2+).
- During any blocking I/O syscall.

This gives fair scheduling among Python threads but prevents CPU-bound parallelism.

## Worked Example

```python
"""
day3_concurrency.py — Mutex, semaphore, condition variable, deadlock demo.
Python 3.12+
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)-12s %(message)s")
log = logging.getLogger(__name__)


# ── 1. Race condition → mutex fix ─────────────────────────────────────────────

def race_condition_demo() -> None:
    counter = 0
    lock = threading.Lock()

    def increment(n: int, use_lock: bool) -> None:
        nonlocal counter
        for _ in range(n):
            if use_lock:
                with lock:
                    counter += 1
            else:
                counter += 1  # read-modify-write is NOT atomic in Python

    log.info("=== Race condition ===")
    for use_lock in (False, True):
        counter = 0
        threads = [threading.Thread(target=increment, args=(50_000, use_lock)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        log.info("use_lock=%-5s final counter=%d  (expected 200000)", use_lock, counter)


# ── 2. Semaphore — bounded resource pool ──────────────────────────────────────

def semaphore_demo(pool_size: int = 3, workers: int = 8) -> None:
    sem = threading.Semaphore(pool_size)
    active: list[int] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        with sem:
            with lock:
                active.append(wid)
                log.info("worker %d acquired slot  active=%s", wid, sorted(active))
            time.sleep(0.05)
            with lock:
                active.remove(wid)

    log.info("=== Semaphore (pool_size=%d) ===", pool_size)
    threads = [threading.Thread(target=worker, args=(i,), name=f"W{i}") for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("All workers done; active=%s", active)


# ── 3. Condition variable — producer/consumer ─────────────────────────────────

@dataclass
class BoundedQueue:
    maxsize: int
    _q: deque[int] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _not_empty: threading.Condition = field(init=False)
    _not_full: threading.Condition = field(init=False)

    def __post_init__(self) -> None:
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

    def put(self, item: int) -> None:
        with self._not_full:
            while len(self._q) >= self.maxsize:
                self._not_full.wait()
            self._q.append(item)
            self._not_empty.notify()

    def get(self) -> int:
        with self._not_empty:
            while not self._q:
                self._not_empty.wait()
            item = self._q.popleft()
            self._not_full.notify()
            return item


def producer_consumer_demo() -> None:
    log.info("=== Producer/Consumer (BoundedQueue) ===")
    q = BoundedQueue(maxsize=3)
    results: list[int] = []
    r_lock = threading.Lock()

    def producer() -> None:
        for i in range(10):
            q.put(i)
            log.info("produced %d", i)
            time.sleep(0.01)

    def consumer() -> None:
        for _ in range(10):
            item = q.get()
            with r_lock:
                results.append(item)
            log.info("consumed %d", item)
            time.sleep(0.02)

    p = threading.Thread(target=producer, name="Producer")
    c = threading.Thread(target=consumer, name="Consumer")
    p.start()
    c.start()
    p.join()
    c.join()
    log.info("All items: %s", sorted(results))


# ── 4. Deadlock demonstration (with timeout to recover) ───────────────────────

def deadlock_demo() -> None:
    log.info("=== Deadlock demo (lock ordering fix) ===")
    m1 = threading.Lock()
    m2 = threading.Lock()

    def thread_a_bad() -> None:
        with m1:
            time.sleep(0.05)
            acquired = m2.acquire(timeout=0.2)  # timeout to avoid permanent hang
            if acquired:
                log.info("Thread A acquired m2 (no deadlock this run)")
                m2.release()
            else:
                log.info("Thread A: DEADLOCK detected — could not acquire m2 in time")

    def thread_b_bad() -> None:
        with m2:
            time.sleep(0.05)
            acquired = m1.acquire(timeout=0.2)
            if acquired:
                log.info("Thread B acquired m1 (no deadlock this run)")
                m1.release()
            else:
                log.info("Thread B: DEADLOCK detected — could not acquire m1 in time")

    ta = threading.Thread(target=thread_a_bad, name="ThreadA")
    tb = threading.Thread(target=thread_b_bad, name="ThreadB")
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    log.info("Fix: always acquire m1 before m2 in both threads (lock ordering)")


if __name__ == "__main__":
    race_condition_demo()
    semaphore_demo()
    producer_consumer_demo()
    deadlock_demo()
```

## Exercises

1. Remove `use_lock=True` branch from `race_condition_demo` and run it 10 times. Record whether the counter reaches 200,000 each time. Why is the result non-deterministic?
2. Rewrite `BoundedQueue` using `threading.Condition` with a single condition (not two) and explain the trade-off.
3. Three threads A, B, C each acquire locks in order: A→{1,2}, B→{2,3}, C→{3,1}. Draw the resource allocation graph and identify whether deadlock is possible.
4. Implement a `RWLock` class (read-write lock) using `threading.Condition`. Multiple readers allowed simultaneously; writers need exclusive access. Include at least two unit tests.
5. Explain what a **livelock** is (hint: differs from deadlock). Give a real-world analogy.

(Solutions: ../solutions/week-06/day-3.md)

## Self-check

- [ ] I can implement a thread-safe counter using a mutex.
- [ ] I can implement a bounded producer/consumer queue with condition variables.
- [ ] I can list all four Coffman conditions for deadlock from memory.
- [ ] I can design a lock-ordering scheme that prevents circular wait.
- [ ] I can distinguish mutex, semaphore, and condition variable use cases.

## Supplementary

- [Python docs — threading](https://docs.python.org/3/library/threading.html)
- [Python docs — queue module](https://docs.python.org/3/library/queue.html)
- [Operating Systems: Three Easy Pieces — Locks & Condition Variables](https://pages.cs.wisc.edu/~remzi/OSTEP/)
- [Futex — Linux man page](https://man7.org/linux/man-pages/man2/futex.2.html)
