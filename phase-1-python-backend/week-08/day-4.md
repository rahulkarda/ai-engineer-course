# Phase 1 · Week 08 · Day 4 — Threading, Multiprocessing & the GIL

> Objective: Understand Python's concurrency models and choose the right one (threads vs processes vs async) based on workload type.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

Python offers three concurrency models. Picking the wrong one means either wasted cores or deadlocks. The deciding factor is **what your task spends time waiting on**.

### The three models

| Model | Best for | Parallelism? |
|---|---|---|
| `threading` | I/O-bound work (network, disk) | Concurrent, not parallel |
| `multiprocessing` | CPU-bound work (compute, encoding) | True parallel |
| `asyncio` | I/O-bound, high-volume, single-thread | Concurrent, not parallel |

### The GIL — why it matters

CPython (the reference interpreter) has a **Global Interpreter Lock**: a mutex that allows only one thread to execute Python bytecode at a time. This is not a bug — it protects CPython's reference-counted memory model from corruption.

Consequences:
- **Threads cannot run Python code in parallel.** They take turns.
- During a blocking I/O call (socket read, `time.sleep`, disk read), the GIL is **released** — other threads run while one thread waits. This is why threads still speed up I/O-bound code.
- CPU-bound threads fight over the GIL and can actually be *slower* than single-threaded code due to contention.
- **Processes** each have their own GIL and their own memory space — true parallel execution.

### When to use each

```
Is task I/O-bound? ──yes──► asyncio (preferred) or ThreadPoolExecutor
Is task CPU-bound? ──yes──► ProcessPoolExecutor
Need to call blocking library that has no async API? ──yes──► run_in_executor (wraps thread/process in asyncio)
```

---

## How it works internally

### Thread context switches

The OS scheduler is unaware of the GIL. CPython implements its own switching: every 5ms (configurable via `sys.setswitchinterval`), the current thread checks whether another thread is waiting for the GIL. If yes, it yields. This is cooperative at the bytecode level but preemptive at the time level.

```
Thread A: [bytecode...bytecode...GIL yield] → waiting
Thread B:                                  [acquires GIL] [bytecode...]
```

### GIL release during I/O

C extension calls (sockets, file I/O, `numpy` operations) explicitly release the GIL using `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS`. This is why `requests`, `psycopg2`, and `boto3` benefit from thread pools — the GIL is released during the actual network call.

### Multiprocessing memory model

Each `Process` is a full OS process fork. Memory is **not shared** by default. Inter-process communication uses:
- `multiprocessing.Queue` — pickled messages over a pipe
- `multiprocessing.Value` / `Array` — shared memory with a lock
- `Manager` — proxy objects over sockets (slower)

`ProcessPoolExecutor` (from `concurrent.futures`) wraps this with a cleaner API that mirrors `ThreadPoolExecutor`.

### Pool executors

Both `ThreadPoolExecutor` and `ProcessPoolExecutor` implement the same interface:
- `.submit(fn, *args)` → `Future`
- `.map(fn, iterable)` → iterator of results
- Used as context managers to ensure proper shutdown

`asyncio.get_event_loop().run_in_executor(executor, fn, *args)` bridges the sync world into an async event loop by running blocking functions in a thread (or process) pool without blocking the loop.

---

## Worked example

```python
"""
week08/day4_concurrency.py

Demonstrates ThreadPoolExecutor, ProcessPoolExecutor, and
run_in_executor for mixing sync blocking calls into async code.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s/%(processName)s] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. I/O-bound simulation — ThreadPoolExecutor
# ---------------------------------------------------------------------------

def fetch_record(record_id: int) -> dict[str, int | float]:
    """Simulate a blocking DB/HTTP call (GIL released during sleep)."""
    log.info("fetching record %d", record_id)
    time.sleep(0.1)  # simulates network latency; GIL released here
    return {"id": record_id, "value": record_id * 1.5}


def fetch_all_threaded(record_ids: list[int], max_workers: int = 8) -> list[dict]:
    """Fetch records concurrently with a thread pool."""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="db-worker") as pool:
        futures = {pool.submit(fetch_record, rid): rid for rid in record_ids}
        for future in as_completed(futures):
            rid = futures[future]
            try:
                results.append(future.result())
            except Exception:
                log.exception("failed to fetch record %d", rid)
    return results


# ---------------------------------------------------------------------------
# 2. CPU-bound work — ProcessPoolExecutor
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """CPU-bound primality check — pure computation, no I/O."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = math.isqrt(n)
    for i in range(3, limit + 1, 2):
        if n % i == 0:
            return False
    return True


def find_primes_parallel(candidates: list[int], max_workers: int = 4) -> list[int]:
    """Find primes using multiple processes (true parallelism, bypasses GIL)."""
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(is_prime, candidates))
    return [n for n, prime in zip(candidates, results) if prime]


# ---------------------------------------------------------------------------
# 3. run_in_executor — integrate blocking calls into async code
# ---------------------------------------------------------------------------

def blocking_io_call(task_id: int) -> str:
    """A third-party library function with no async API."""
    time.sleep(0.05)
    return f"result-{task_id}"


async def run_mixed_workload(task_ids: list[int]) -> list[str]:
    """
    Run blocking calls inside an async event loop without blocking the loop.
    Uses the default ThreadPoolExecutor managed by asyncio.
    """
    loop = asyncio.get_running_loop()
    # Each call runs in a thread; the event loop stays free to handle other coroutines
    tasks = [
        loop.run_in_executor(None, blocking_io_call, tid)
        for tid in task_ids
    ]
    results: list[str] = await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    record_ids = list(range(1, 21))  # 20 simulated DB records

    # --- Thread pool (I/O-bound) ---
    log.info("=== ThreadPoolExecutor: fetching %d records ===", len(record_ids))
    t0 = time.perf_counter()
    records = fetch_all_threaded(record_ids, max_workers=8)
    elapsed = time.perf_counter() - t0
    log.info("fetched %d records in %.2fs (sequential would be ~%.1fs)",
             len(records), elapsed, len(record_ids) * 0.1)

    # --- Process pool (CPU-bound) ---
    candidates = list(range(100_000, 100_200))
    log.info("=== ProcessPoolExecutor: primality check on %d numbers ===", len(candidates))
    t0 = time.perf_counter()
    primes = find_primes_parallel(candidates)
    elapsed = time.perf_counter() - t0
    log.info("found %d primes in %.3fs", len(primes), elapsed)

    # --- run_in_executor (async + blocking) ---
    task_ids = list(range(1, 11))
    log.info("=== run_in_executor: %d async tasks with blocking calls ===", len(task_ids))
    t0 = time.perf_counter()
    results = asyncio.run(run_mixed_workload(task_ids))
    elapsed = time.perf_counter() - t0
    log.info("completed %d tasks in %.2fs: %s", len(results), elapsed, results[:3])


if __name__ == "__main__":
    main()
```

Key observations when you run this:
- Thread pool fetches 20 records in ~0.1s instead of ~2.0s — threads overlap during `time.sleep` (GIL released).
- Process pool uses all available cores; each worker is a separate Python interpreter.
- `run_in_executor(None, ...)` uses the default executor — pass a custom `ThreadPoolExecutor` to control pool size.

---

## Exercises

1. **Thread pool for concurrent DB-like tasks**
   Extend `fetch_record` to randomly raise a `TimeoutError` 20% of the time. Modify `fetch_all_threaded` to retry failed records up to 2 times. Collect both successful results and permanently failed record IDs. Log each retry attempt with the record ID and attempt number.

2. **Process pool for computation**
   Write a function `compute_hashes(texts: list[str]) -> list[str]` that computes the SHA-256 hash of each string after applying 10,000 iterations of `hashlib.sha256` (simulate expensive work). Compare wall-clock time for: (a) sequential loop, (b) `ThreadPoolExecutor` with 4 workers, (c) `ProcessPoolExecutor` with 4 workers. Use `time.perf_counter` and log the results. Explain why threads don't help here.

3. **Convert blocking call to async with `run_in_executor`**
   You have this existing sync function (do not modify it):
   ```python
   import time
   def read_config_from_disk(path: str) -> dict:
       time.sleep(0.2)  # simulate slow disk
       return {"path": path, "loaded": True}
   ```
   Write an async function `load_configs(paths: list[str]) -> list[dict]` that loads all configs concurrently without blocking the event loop. Then write a second async function `main_async` that loads 5 configs and prints total elapsed time. Verify the total time is close to 0.2s, not 1.0s.

*(Solutions: ../solutions/week-08/day-4.md)*

---

## Self-check

- [ ] I can explain why threads cannot achieve true parallelism for CPU-bound Python code, and what specific CPython mechanism causes this.
- [ ] I can choose between `ThreadPoolExecutor`, `ProcessPoolExecutor`, and `run_in_executor` given a description of a task, and justify the choice.
- [ ] I can implement a `ProcessPoolExecutor` pipeline that handles worker exceptions without crashing the parent process.

---

## Supplementary

- Python docs — `concurrent.futures`: https://docs.python.org/3/library/concurrent.futures.html
- David Beazley's GIL deep dive: https://www.dabeaz.com/python/UnderstandingGIL.pdf
- PEP 703 (no-GIL CPython, experimental): https://peps.python.org/pep-0703/
