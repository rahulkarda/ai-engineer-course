# Week 08 Day 5 — Race Conditions, Locks & Semaphores: Answer Key

## Exercise 1: Identify and Fix the Race Condition

### The Problem

```python
results = []
async def fetch_and_append(url):
    data = await fetch(url)
    results.append(data)  # race condition
await asyncio.gather(*[fetch_and_append(u) for u in urls])
```

**Why it's a race condition:** Python's `asyncio` is single-threaded but concurrent. The race
is not a memory-level data race — it's a *logical* race on shared mutable state. Between the
`await fetch(url)` suspension point and the `results.append(data)` call, the event loop can
resume another coroutine. If two coroutines both check-then-modify `results` without
coordination, you get interleaved writes. With a list and simple appends CPython is *usually*
safe due to the GIL, but the pattern is wrong:

1. With `asyncio.gather`, coroutines interleave at every `await`. If `results` were replaced
   by any non-atomic operation (read-modify-write), data loss occurs.
2. The order of results is non-deterministic — coroutines complete in arrival order, not URL
   order. If callers depend on index alignment with `urls`, results will be wrong.
3. If you ever move to `asyncio` + `ThreadPoolExecutor` or use a different list implementation,
   the implicit safety disappears entirely.

### Fix with `asyncio.Lock`

```python
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

async def fetch_urls_safe(urls: list[str]) -> list[Any]:
    """
    Fetch all URLs concurrently; append results under a lock.
    Returns results in completion order (not URL order — see Exercise 3 for that).
    """
    results: list[Any] = []
    lock = asyncio.Lock()

    async def fetch_and_append(url: str) -> None:
        data = await fetch(url)           # suspend here — other coroutines may run
        async with lock:                  # only one coroutine enters this block at a time
            results.append(data)
            logger.debug("appended result for %s, total=%d", url, len(results))

    await asyncio.gather(*[fetch_and_append(u) for u in urls])
    return results
```

**Why the lock is placed *after* the await:** The expensive I/O (`fetch`) runs concurrently —
we don't hold the lock during the network call. The lock only guards the cheap, fast
`append`. This keeps concurrency high while preventing interleaved mutations.

**When you don't need the lock:** If you pre-allocate a list with `None` slots and write by
index — `results[i] = data` — each coroutine writes to a distinct slot. Index-based writes
are safe without a lock. But pre-allocation requires knowing `len(urls)` upfront.

---

## Exercise 2: Per-Key Lock Cache

```python
import asyncio
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PerKeyLockCache:
    """
    Dict cache where each key has its own asyncio.Lock.
    Prevents the "thundering herd" / cache stampede: when multiple coroutines
    concurrently request the same missing key, only one fetches — the rest wait
    and reuse the result.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()  # guards creation of per-key locks

    async def _get_key_lock(self, key: str) -> asyncio.Lock:
        """Return (and lazily create) the lock for *key*."""
        async with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            return self._locks[key]

    async def get_or_fetch(self, key: str, fetcher: Callable[[], Any]) -> Any:
        """
        Return cached value for *key*, or call *fetcher()* exactly once to
        populate it. Double-checked locking pattern:

        1. Fast path: key already in cache → return immediately (no lock).
        2. Acquire per-key lock.
        3. Re-check cache (another coroutine may have filled it while we waited).
        4. If still missing, call fetcher, store result, release lock.
        """
        # Fast path — no lock needed for a read when key is present
        if key in self._cache:
            logger.debug("cache hit (fast path): %s", key)
            return self._cache[key]

        key_lock = await self._get_key_lock(key)

        async with key_lock:
            # Double-check: another coroutine may have populated the key
            # while we were waiting for the lock
            if key in self._cache:
                logger.debug("cache hit (after lock): %s", key)
                return self._cache[key]

            logger.debug("cache miss — fetching: %s", key)
            value = await fetcher()
            self._cache[key] = value
            logger.debug("stored in cache: %s", key)
            return value

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache (lock stays, cache entry goes)."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values (locks are preserved)."""
        self._cache.clear()
```

### Usage Example

```python
cache = PerKeyLockCache()

async def expensive_fetch(user_id: str) -> dict:
    return await http_client.get(f"/users/{user_id}")

# These three concurrent calls for the same key trigger only ONE real fetch:
results = await asyncio.gather(
    cache.get_or_fetch("user-42", lambda: expensive_fetch("user-42")),
    cache.get_or_fetch("user-42", lambda: expensive_fetch("user-42")),
    cache.get_or_fetch("user-42", lambda: expensive_fetch("user-42")),
)
# All three return the same dict; fetcher was called once.
```

### Why Double-Check?

Without the second `if key in self._cache` check inside the lock:

```
Coroutine A: key not in cache (fast path fails)
Coroutine B: key not in cache (fast path fails)
Coroutine A: acquires lock → fetches → stores
Coroutine A: releases lock
Coroutine B: acquires lock → fetches AGAIN (wasted work / duplicate side-effects)
```

With double-check:

```
Coroutine A: acquires lock → fetches → stores → releases
Coroutine B: acquires lock → re-checks → key found → returns cached → releases
```

---

## Exercise 3: Order-Preserving Fetch with Semaphore

```python
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def order_preserving_fetch(
    urls: list[str],
    max_concurrent: int = 3,
) -> list[Any]:
    """
    Fetch all URLs with at most *max_concurrent* simultaneous connections.
    Returns results in the same order as *urls* regardless of completion order.

    Strategy:
    - asyncio.Semaphore caps concurrency.
    - Each coroutine is assigned its original index; results written to a
      pre-allocated list by index so ordering is guaranteed without sorting.
    """
    results: list[Any] = [None] * len(urls)  # pre-allocate — index writes are safe
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_one(index: int, url: str) -> None:
        async with semaphore:          # blocks if max_concurrent slots are taken
            logger.debug("fetching [%d] %s", index, url)
            try:
                data = await fetch(url)
                results[index] = data
                logger.debug("completed [%d] %s", index, url)
            except Exception:
                logger.exception("failed [%d] %s", index, url)
                results[index] = None  # or raise — depends on your error policy

    await asyncio.gather(*[fetch_one(i, url) for i, url in enumerate(urls)])
    return results
```

### How Order Preservation Works

`asyncio.gather` starts all coroutines immediately but they compete for the semaphore.
Completion order is driven by network latency — a slow URL at index 0 finishes last.
By writing `results[index] = data` instead of `results.append(data)`, we decouple
completion order from storage position. The list is always indexed by original URL order.

### Semaphore vs Lock Comparison

| | `asyncio.Lock` | `asyncio.Semaphore(n)` |
|---|---|---|
| Allows concurrent holders | No (1 at a time) | Yes (up to *n* at a time) |
| Use for | Mutual exclusion | Rate-limiting / connection pools |
| Typical use | Protecting shared state | Capping concurrent I/O |

### Full Runnable Demo

```python
import asyncio
import logging
import random

logging.basicConfig(level=logging.DEBUG)


async def fake_fetch(url: str) -> str:
    """Simulates variable-latency HTTP fetch."""
    delay = random.uniform(0.1, 1.0)
    await asyncio.sleep(delay)
    return f"response_from_{url}_after_{delay:.2f}s"


async def fetch(url: str) -> str:
    return await fake_fetch(url)


async def main() -> None:
    urls = [f"https://example.com/{i}" for i in range(8)]
    results = await order_preserving_fetch(urls, max_concurrent=3)
    for url, result in zip(urls, results):
        print(f"{url}: {result}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Key Takeaways

- **Lock** = mutual exclusion; only 1 coroutine in critical section at a time.
- **Semaphore(n)** = rate limiter; up to *n* coroutines proceed concurrently.
- Preserve order via **index-based writes**, not append-then-sort.
- Always place the `await` *outside* the lock to keep concurrency high.
- Double-checked locking prevents stampedes on shared caches.
