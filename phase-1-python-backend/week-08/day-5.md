# Phase 1 · Week 08 · Day 5 — Race Conditions, Locks & Semaphores

> Objective: Identify race conditions in concurrent code and fix them with the correct asyncio or threading synchronization primitive.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

Any time two concurrent tasks access shared mutable state, you have a race. The outcome depends on which task runs first — and the scheduler doesn't guarantee order.

### What a race condition looks like

```python
# BAD — two coroutines can read the same value before either writes back
counter = 0

async def increment():
    global counter
    value = counter      # read
    await asyncio.sleep(0)  # yield — another coroutine can run here
    counter = value + 1  # write — may overwrite the other coroutine's write
```

Ten calls to `increment()` launched with `asyncio.gather` will not reliably produce `counter == 10`. The final value depends on interleaving.

### Data race vs. race condition

- **Data race**: two threads/coroutines access the same memory simultaneously with at least one write and no synchronization. Undefined behavior in C; in Python, the GIL prevents true simultaneous access for threads, but asyncio coroutines can still race across `await` suspension points.
- **Race condition**: a logic flaw where outcome depends on execution order, even if no data corruption occurs. Both types require synchronization to fix.

### asyncio.Lock vs threading.Lock

| | `asyncio.Lock` | `threading.Lock` |
|---|---|---|
| Blocks | Suspends coroutine (non-blocking) | Blocks OS thread |
| Acquired with | `async with lock:` | `with lock:` or `lock.acquire()` |
| Use in | async code only | sync/thread code only |
| Deadlock risk | Yes — must not `await` inside a held lock if that await can re-enter | Yes — must not call `acquire()` twice from same thread |

**Never use `threading.Lock` in asyncio code.** It blocks the entire event loop thread.

### Semaphore for rate limiting

A `Semaphore` is a lock with a counter. It allows up to N concurrent holders. Uses:
- Limit concurrent outbound API calls (stay under rate limits)
- Cap concurrent DB connections
- Throttle file I/O

```python
sem = asyncio.Semaphore(5)  # at most 5 concurrent tasks

async def rate_limited_call():
    async with sem:
        await make_api_call()
```

---

## How it works internally

### Mutex (mutual exclusion)

A `Lock` is a binary semaphore — state is either `locked` or `unlocked`. The operation sequence:

```
coroutine A: acquire() → locked=True → [critical section] → release() → locked=False
coroutine B: acquire() → locked=True → suspend (added to waiters queue)
             [after A calls release()] → woken up → locked=True → [critical section]
```

asyncio implements this entirely in Python without OS calls. The `Lock` object holds a `collections.deque` of waiting `Future` objects. `release()` resolves the first waiter's `Future`, which schedules that coroutine to resume on the next event loop iteration.

### Lock contention and performance

High contention means many coroutines queued waiting for a single lock. Symptoms:
- Throughput drops despite I/O concurrency
- Latency spikes on operations that are individually fast

Mitigation: reduce critical section size, use separate locks per resource (sharding), or use lock-free structures (`asyncio.Queue`, which is internally synchronized).

### Semaphore internals

`asyncio.Semaphore` holds a counter `_value`. `acquire()` checks if `_value > 0` — if yes, decrements and returns immediately; if no, suspends. `release()` increments and wakes the next waiter. `BoundedSemaphore` additionally raises `ValueError` if `release()` would exceed the initial count (catches release-without-acquire bugs).

### Deadlock conditions

A deadlock requires all four of the Coffman conditions simultaneously:
1. **Mutual exclusion** — resource held exclusively
2. **Hold and wait** — holding one lock while waiting for another
3. **No preemption** — locks not forcibly released
4. **Circular wait** — A waits for B, B waits for A

Prevention in asyncio: always acquire multiple locks in the same consistent order across all coroutines. Or use `asyncio.timeout()` so a stuck `acquire()` eventually raises `TimeoutError`.

### async context manager pattern

`async with lock:` is syntactic sugar for:
```python
await lock.acquire()
try:
    ...
finally:
    lock.release()
```

Always prefer `async with` — the `finally` block guarantees release even if the critical section raises an exception. Manual `acquire/release` pairs are error-prone.

---

## Worked example

```python
"""
week08/day5_synchronization.py

Demonstrates:
  1. asyncio.Lock protecting a shared counter
  2. asyncio.Semaphore as a rate limiter for external API calls
  3. Intentional race (for comparison) vs. locked version
"""

from __future__ import annotations

import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Race condition demo, then fix with asyncio.Lock
# ---------------------------------------------------------------------------

async def increment_racy(shared: dict[str, int], task_id: int) -> None:
    """Intentionally racy — shows the problem."""
    value = shared["counter"]          # read
    await asyncio.sleep(0)             # yield — scheduler can switch here
    shared["counter"] = value + 1     # write — may stomp another task's write
    log.debug("task %d: counter now %d", task_id, shared["counter"])


async def increment_safe(
    shared: dict[str, int],
    lock: asyncio.Lock,
    task_id: int,
) -> None:
    """Lock protects the read-modify-write as an atomic unit."""
    async with lock:
        value = shared["counter"]
        await asyncio.sleep(0)             # yield inside lock — safe, still held
        shared["counter"] = value + 1
    log.debug("task %d: counter now %d", task_id, shared["counter"])


async def demo_counter() -> None:
    n_tasks = 50

    # Racy version
    state_racy: dict[str, int] = {"counter": 0}
    await asyncio.gather(*(increment_racy(state_racy, i) for i in range(n_tasks)))
    log.info("RACY result: expected %d, got %d", n_tasks, state_racy["counter"])

    # Locked version
    state_safe: dict[str, int] = {"counter": 0}
    lock = asyncio.Lock()
    await asyncio.gather(*(increment_safe(state_safe, lock, i) for i in range(n_tasks)))
    log.info("SAFE result: expected %d, got %d", n_tasks, state_safe["counter"])


# ---------------------------------------------------------------------------
# 2. asyncio.Semaphore as rate limiter
# ---------------------------------------------------------------------------

async def fake_api_call(endpoint: str, session_id: int) -> dict[str, str | int]:
    """Simulate an external HTTP call with ~100ms latency."""
    log.debug("session %d calling %s", session_id, endpoint)
    await asyncio.sleep(0.1)
    return {"endpoint": endpoint, "session": session_id, "status": 200}


async def rate_limited_fetch(
    urls: list[str],
    max_concurrent: int = 3,
) -> list[dict]:
    """
    Fetch all URLs but never exceed max_concurrent simultaneous requests.
    Semaphore ensures we stay within the API rate limit.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def fetch_one(url: str, idx: int) -> dict:
        async with sem:
            log.info("slot acquired for url %d/%d: %s", idx + 1, len(urls), url)
            result = await fake_api_call(url, idx)
            log.info("slot released for url %d", idx + 1)
            return result

    results = await asyncio.gather(*(fetch_one(url, i) for i, url in enumerate(urls)))
    return list(results)


# ---------------------------------------------------------------------------
# 3. Bounded semaphore — detect release-without-acquire bugs
# ---------------------------------------------------------------------------

async def demo_bounded_semaphore() -> None:
    """BoundedSemaphore raises if you release more than you acquired."""
    sem = asyncio.BoundedSemaphore(2)

    await sem.acquire()
    await sem.acquire()
    sem.release()
    sem.release()

    try:
        sem.release()  # over-release — programming bug
    except ValueError as exc:
        log.warning("caught over-release: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    log.info("=== Counter race condition demo ===")
    await demo_counter()

    log.info("=== Semaphore rate limiter demo ===")
    urls = [f"https://api.example.test/item/{i}" for i in range(10)]
    t0 = time.perf_counter()
    results = await rate_limited_fetch(urls, max_concurrent=3)
    elapsed = time.perf_counter() - t0
    log.info(
        "fetched %d urls in %.2fs with max_concurrent=3 "
        "(unconstrained would be ~0.1s, serial would be ~1.0s)",
        len(results), elapsed,
    )

    log.info("=== BoundedSemaphore over-release demo ===")
    await demo_bounded_semaphore()


if __name__ == "__main__":
    asyncio.run(main())
```

When you run this, observe:
- The racy counter almost never reaches 50 — the yield inside the critical section exposes the race window.
- The locked counter always reaches exactly 50.
- The semaphore demo shows at most 3 "slot acquired" log lines visible simultaneously — even though all 10 coroutines are started at once, the semaphore queues the excess.

---

## Exercises

1. **Identify the race condition**
   The following code is supposed to build a shared results list safely. Find every race condition and explain exactly when and why it can produce incorrect output.
   ```python
   results = []

   async def process_item(item: int) -> None:
       result = item * 2
       await asyncio.sleep(0)
       results.append(result)  # is this safe?
       results.sort()           # is this safe?

   await asyncio.gather(*(process_item(i) for i in range(20)))
   ```
   Hint: CPython list operations and asyncio suspension points behave differently. State whether each line is a race or not, and why.

2. **Fix a shared cache with a lock**
   You have this broken async cache:
   ```python
   _cache: dict[str, str] = {}

   async def get_or_fetch(key: str) -> str:
       if key in _cache:
           return _cache[key]
       await asyncio.sleep(0.1)  # simulate fetch
       _cache[key] = f"value-for-{key}"
       return _cache[key]
   ```
   Multiple coroutines calling `get_or_fetch("same-key")` simultaneously will each perform the fetch. Fix this using `asyncio.Lock` so the fetch happens exactly once per key. Use a `dict[str, asyncio.Lock]` (one lock per key) so unrelated keys don't block each other. Write a test that launches 10 concurrent calls for the same key and asserts the fetch was only triggered once (use a counter).

3. **Semaphore-limited fetch pool**
   Write an async function `parallel_fetch(urls: list[str], max_concurrent: int) -> list[tuple[str, int]]` that:
   - Accepts a list of URL strings and a concurrency cap
   - Uses `asyncio.Semaphore` to enforce the cap
   - For each URL, simulates an HTTP call: `await asyncio.sleep(0.05)`, returns `(url, 200)`
   - Returns results in the same order as the input list (not arrival order)
   - Handles exceptions per-URL: if the simulated call raises, store `(url, 500)` instead of propagating
   Write a test that verifies: (a) all results are returned in input order, (b) total elapsed time for 12 URLs with `max_concurrent=4` is between 0.15s and 0.35s (3 batches of 4 at 0.05s each).

*(Solutions: ../solutions/week-08/day-5.md)*

---

## Self-check

- [ ] I can explain why `asyncio.Lock` must not be replaced with `threading.Lock` in an async program, and what failure mode results from the mistake.
- [ ] I can trace through code with an `await` inside a critical section and determine whether the lock prevents the race.
- [ ] I can implement `asyncio.Semaphore` as a rate-limiting wrapper that preserves result ordering and handles per-item exceptions without stopping the pool.

---

## Supplementary

- Python docs — `asyncio` synchronization primitives: https://docs.python.org/3/library/asyncio-sync.html
- Python docs — `threading` synchronization: https://docs.python.org/3/library/threading.html#lock-objects
- Coffman conditions (deadlock theory): https://en.wikipedia.org/wiki/Deadlock#Necessary_conditions
