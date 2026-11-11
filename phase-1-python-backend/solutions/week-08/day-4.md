# Week 08 · Day 4 — Solutions
## Threading, Multiprocessing & the GIL

---

### Exercise 1
**Write `retry_with_thread_pool(tasks: list[Callable], max_workers=4) -> list[Any]` that runs tasks in a `ThreadPoolExecutor`, retries failed tasks once, returns results in order.**

```python
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


def retry_with_thread_pool(
    tasks: list[Callable[[], Any]],
    max_workers: int = 4,
) -> list[Any]:
    """Run callable tasks in a thread pool; retry each failure once.

    Args:
        tasks:       List of zero-argument callables.
        max_workers: Thread pool size.

    Returns:
        List of results in the same order as `tasks`.
        Items that failed on both the initial attempt and the retry
        are represented as the final Exception object.
    """
    results: list[Any] = [None] * len(tasks)
    retry_indices: list[int] = []

    # --- First pass ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index: dict[Future[Any], int] = {
            executor.submit(task): i
            for i, task in enumerate(tasks)
        }
        for future, index in future_to_index.items():
            try:
                results[index] = future.result()
                logger.debug("Task %d succeeded", index)
            except Exception as exc:
                logger.warning("Task %d failed on first attempt: %s", index, exc)
                retry_indices.append(index)

    if not retry_indices:
        return results

    # --- Retry pass ---
    logger.info("Retrying %d failed tasks", len(retry_indices))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        retry_futures: dict[Future[Any], int] = {
            executor.submit(tasks[i]): i
            for i in retry_indices
        }
        for future, index in retry_futures.items():
            try:
                results[index] = future.result()
                logger.info("Task %d succeeded on retry", index)
            except Exception as exc:
                logger.error("Task %d failed on retry too: %s", index, exc)
                results[index] = exc  # preserve the exception as the result

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

import random  # noqa: E402 — placed after function for clarity

_attempt_counts: dict[int, int] = {}


def flaky_task(task_id: int) -> str:
    """Succeed on the second call; raise on the first."""
    _attempt_counts[task_id] = _attempt_counts.get(task_id, 0) + 1
    if _attempt_counts[task_id] < 2 and task_id % 3 == 0:
        raise RuntimeError(f"Transient error in task {task_id}")
    return f"ok_{task_id}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    task_list = [lambda tid=i: flaky_task(tid) for i in range(9)]
    output = retry_with_thread_pool(task_list, max_workers=4)
    for i, r in enumerate(output):
        logger.info("Task %d: %s", i, r)
```

**Key points:**
- Futures are submitted in one batch and collected after all complete, preserving order via the `future_to_index` mapping.
- The retry pass only re-submits tasks that raised on the first attempt — successful tasks are not redundantly re-run.
- Failed retries store the `Exception` object in `results` rather than raising, so callers get a complete result list and can decide how to handle residual failures.
- Two separate `with ThreadPoolExecutor` blocks ensure all first-pass futures are collected before any retry begins.

---

### Exercise 2
**Benchmark: compare wall-clock time for computing SHA256 of 1000 random 1KB strings using (a) sequential loop, (b) `ThreadPoolExecutor(4)`, (c) `ProcessPoolExecutor(4)`. Print a comparison table.**

```python
"""
sha256_benchmark.py — compare sequential vs. thread pool vs. process pool
for a CPU-bound SHA256 workload.

Run: python sha256_benchmark.py
"""
import hashlib
import logging
import os
import random
import string
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

logger = logging.getLogger(__name__)

NUM_STRINGS = 1_000
STRING_SIZE = 1_024  # bytes


def _make_random_string(size: int) -> bytes:
    """Return `size` random ASCII bytes."""
    return "".join(random.choices(string.printable, k=size)).encode()


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA256 digest of data — CPU-bound."""
    return hashlib.sha256(data).hexdigest()


def run_sequential(corpus: list[bytes]) -> float:
    """Compute SHA256 for all items in a plain loop. Returns elapsed seconds."""
    start = time.perf_counter()
    for data in corpus:
        compute_sha256(data)
    return time.perf_counter() - start


def run_thread_pool(corpus: list[bytes], workers: int = 4) -> float:
    """Compute SHA256 using a ThreadPoolExecutor. Returns elapsed seconds.

    Note: Due to the GIL, threads do NOT run Python bytecode in parallel.
    hashlib releases the GIL during hashing, so some speedup is possible,
    but gains are limited compared to true parallelism.
    """
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(compute_sha256, corpus))
    return time.perf_counter() - start


def run_process_pool(corpus: list[bytes], workers: int = 4) -> float:
    """Compute SHA256 using a ProcessPoolExecutor. Returns elapsed seconds.

    Processes bypass the GIL entirely — each gets its own Python interpreter.
    IPC (serialisation of corpus chunks) adds overhead, but CPU-bound tasks
    see near-linear speedup up to the number of physical cores.
    """
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(compute_sha256, corpus))
    return time.perf_counter() - start


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger.info("Generating %d × %d-byte strings…", NUM_STRINGS, STRING_SIZE)
    corpus = [_make_random_string(STRING_SIZE) for _ in range(NUM_STRINGS)]

    seq_time = run_sequential(corpus)
    thr_time = run_thread_pool(corpus, workers=4)
    prc_time = run_process_pool(corpus, workers=4)

    header = f"{'Method':<30} {'Time (s)':>10} {'vs sequential':>15}"
    sep = "-" * len(header)
    rows = [
        ("Sequential (loop)", seq_time, 1.0),
        ("ThreadPoolExecutor(4)", thr_time, seq_time / thr_time),
        ("ProcessPoolExecutor(4)", prc_time, seq_time / prc_time),
    ]

    print()
    print(header)
    print(sep)
    for name, elapsed, speedup in rows:
        print(f"{name:<30} {elapsed:>10.4f} {speedup:>14.2f}×")
    print()
    print(f"CPU count: {os.cpu_count()}")
    print()
    print("Interpretation:")
    print("  Sequential:  baseline, single-threaded.")
    print("  Threads:     GIL limits true parallelism; hashlib's C extension")
    print("               releases the GIL so modest speedup is possible.")
    print("  Processes:   full parallelism, but IPC serialisation adds overhead.")
    print("               Speedup approaches CPU count for pure CPU-bound work.")


if __name__ == "__main__":
    main()
```

**Key points:**
- SHA256 is CPU-bound — the GIL means `ThreadPoolExecutor` gives little or no speedup for pure Python hashing, though `hashlib` releases the GIL in its C implementation, allowing some threading benefit.
- `ProcessPoolExecutor` bypasses the GIL entirely; each worker process runs on a separate core. Speedup should approach `min(workers, cpu_count)` for CPU-bound work once serialisation overhead is subtracted.
- `executor.map()` is preferred over `submit` + `Future.result()` for simple batch operations — it handles ordering and re-raises exceptions cleanly.
- The benchmark must use `if __name__ == "__main__":` guard because `ProcessPoolExecutor` spawns new processes that re-import the module; without the guard those workers re-run `main()` recursively.

---

### Exercise 3
**Given `def read_large_file(path: str) -> str` (blocking), wrap it as `async def read_file_async(path: Path) -> str` using `asyncio.get_event_loop().run_in_executor(None, ...)`.**

```python
import asyncio
import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Blocking function (given — do not modify)
# ---------------------------------------------------------------------------

def read_large_file(path: str) -> str:
    """Blocking file read — must not be called directly from async code."""
    logger.debug("Reading file: %s (blocking)", path)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------

async def read_file_async(path: Path) -> str:
    """Read a file asynchronously by offloading the blocking call to a thread.

    Uses the default ThreadPoolExecutor (None) managed by the event loop.
    The calling coroutine yields during the read so other async work can run.

    Args:
        path: Path to the file to read.

    Returns:
        Full file contents as a string.
    """
    loop = asyncio.get_event_loop()
    # functools.partial binds the str() path argument; run_in_executor
    # requires a zero-argument callable.
    content: str = await loop.run_in_executor(
        None,                                      # use default thread pool
        functools.partial(read_large_file, str(path)),
    )
    logger.debug("Finished reading %s (%d chars)", path, len(content))
    return content


# ---------------------------------------------------------------------------
# Alternative: asyncio.to_thread (Python 3.9+, cleaner API)
# ---------------------------------------------------------------------------

async def read_file_async_v2(path: Path) -> str:
    """Same behaviour using asyncio.to_thread — the modern idiomatic form."""
    content: str = await asyncio.to_thread(read_large_file, str(path))
    logger.debug("Finished reading %s (%d chars)", path, len(content))
    return content


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    # Write a temp file to read back
    tmp = Path("/tmp/test_async_read.txt")
    tmp.write_text("Hello from async read!\n" * 1000, encoding="utf-8")

    content_v1 = await read_file_async(tmp)
    logger.info("v1: read %d chars", len(content_v1))

    content_v2 = await read_file_async_v2(tmp)
    logger.info("v2: read %d chars", len(content_v2))

    # Concurrently read the same file 5 times
    results = await asyncio.gather(*[read_file_async(tmp) for _ in range(5)])
    logger.info("Concurrent reads: %d files, all %d chars", len(results), len(results[0]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
```

**Key points:**
- `loop.run_in_executor(None, fn)` submits `fn` to the default `ThreadPoolExecutor` that the event loop manages. This frees the event loop to run other coroutines while the blocking I/O proceeds in a thread.
- `functools.partial` is needed because `run_in_executor` requires a zero-argument callable — it does not accept `*args` directly (unlike `asyncio.to_thread`).
- `asyncio.to_thread` (Python 3.9+) is the modern, cleaner alternative — it accepts `*args` and `**kwargs` directly, making `functools.partial` unnecessary.
- This pattern is the right choice for legacy blocking libraries (database drivers, file parsers) that cannot be made natively async. For new I/O code, prefer `aiofiles`, `asyncpg`, or other native async libraries that avoid threads entirely.
- Using `None` as the executor is appropriate for I/O-bound blocking calls. For CPU-bound work, pass a `ProcessPoolExecutor` explicitly to avoid GIL contention.
