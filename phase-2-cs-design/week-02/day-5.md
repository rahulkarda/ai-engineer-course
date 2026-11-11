# Phase 2 · Week 02 · Day 5 — Caching: functools.lru_cache, Memoization, Cache Eviction Strategies

> Objective: Understand and implement memoization from scratch, use `lru_cache` correctly, and reason about LRU, LFU, and FIFO eviction policies.  ·  Time: ~1 hr  ·  Prereqs: [[phase-2-cs-design/week-02/day-4]]

---

## Concept

**Caching** trades space for time: store previously computed results so repeated calls skip the computation. Effective only when:

1. The function is **pure** (same inputs → same output, no side effects).
2. The same inputs appear more than once (temporal/spatial locality).
3. Computation cost > cache lookup cost.

### Memoization

Memoization is function-level caching: wrap a function so its first call with given arguments stores the result; subsequent calls return the stored result immediately.

```
memo: dict[args, result] = {}

def memoized(f):
    def wrapper(*args):
        if args not in memo:
            memo[args] = f(*args)
        return memo[args]
    return wrapper
```

### LRU (Least Recently Used)

When the cache is full, **evict the entry that was accessed least recently**. Rationale: recently used items are likely to be used again (temporal locality). Used by `functools.lru_cache`.

### LFU (Least Frequently Used)

Evict the entry with the **lowest access count**. Better for workloads where some items are always popular and others are one-time accesses.

### FIFO (First In, First Out)

Evict the **oldest inserted** entry regardless of usage. Simple but often outperformed by LRU. Useful when age is a better proxy for staleness than recency of access.

### TTL (Time-to-Live)

Entries expire after a fixed wall-clock duration. Common in HTTP caches, DNS, and any cache over mutable external data.

---

## How it works internally

### `functools.lru_cache` data structures

Python's `lru_cache` uses two structures:

1. **`dict` (hash map)** — `args_tuple → cache_node` for O(1) lookup.
2. **Doubly-linked list** — nodes ordered by recency; head = most recent, tail = least recent.

On a **cache hit**: move the hit node to the head → O(1).  
On a **cache miss**: insert new node at head; if over capacity, unlink and delete the tail → O(1).

This combination (hash map + doubly-linked list) is the canonical O(1) LRU design.

### Python's `lru_cache` caveats

- Arguments must be **hashable** — `list`, `dict`, `set` as arguments raises `TypeError`.
- Cache is **per function object** — two calls to the same decorated function share one cache; a new function object gets a fresh cache.
- `maxsize=None` → unlimited cache (equivalent to `@cache` in Python 3.9+), backed only by the dict.
- `maxsize` must be a power of two for optimal performance (CPython aligns to next power of two internally).
- `cache_info()` returns `CacheInfo(hits, misses, maxsize, currsize)` — use for debugging.
- `cache_clear()` empties the cache — useful in tests.

### Recursion + memoization = dynamic programming (top-down)

Recursive Fibonacci without memoization: `O(2ⁿ)` — each call spawns two more.  
With memoization: each subproblem solved once → `O(n)` time, `O(n)` space (call stack + memo).  
Bottom-up DP avoids the call stack overhead entirely.

### Thread safety

`lru_cache` uses a re-entrant lock on each cache — safe for multi-threaded reads and writes. The lock is per-cache, not global.

---

## Worked example

```python
"""
caching_patterns.py — memoization from scratch, lru_cache, manual LRU class
Python 3.12+
"""
import logging
import time
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── 1. Manual memoization decorator ─────────────────────────────────────────

def memoize(func: F) -> F:
    """
    Simple unbounded memoization decorator.
    Stores results in a dict keyed by the args tuple.
    Only works with hashable arguments.
    """
    cache: dict[tuple[Any, ...], Any] = {}

    @wraps(func)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache          # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


# ── 2. Fibonacci: naive vs memoized vs lru_cache ─────────────────────────────

def fib_naive(n: int) -> int:
    """O(2ⁿ) — exponential, no caching."""
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


@memoize
def fib_memoized(n: int) -> int:
    """O(n) time, O(n) space — top-down DP via manual memoize."""
    if n <= 1:
        return n
    return fib_memoized(n - 1) + fib_memoized(n - 2)


@lru_cache(maxsize=128)
def fib_lru(n: int) -> int:
    """O(n) time — top-down DP via functools.lru_cache."""
    if n <= 1:
        return n
    return fib_lru(n - 1) + fib_lru(n - 2)


# ── 3. Manual LRU cache class ────────────────────────────────────────────────

class LRUCache:
    """
    O(1) get/put LRU cache.
    Uses OrderedDict: move_to_end() implements the recency update.
    OrderedDict is a dict + doubly-linked list internally (same as lru_cache).
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._store: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int:
        """Return value or -1 if not present. Marks key as most recently used."""
        if key not in self._store:
            return -1
        self._store.move_to_end(key)          # O(1): update recency
        return self._store[key]

    def put(self, key: int, value: int) -> None:
        """Insert or update key. Evicts LRU entry if over capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)   # O(1): remove oldest (head)

    def __repr__(self) -> str:
        return f"LRUCache({dict(self._store)!r})"


# ── 4. TTL cache decorator ───────────────────────────────────────────────────

def ttl_cache(seconds: float) -> Callable[[F], F]:
    """
    Decorator factory: cache results for `seconds` seconds.
    Entries expire by wall-clock time, not by eviction policy.
    Not thread-safe (demo only — use cachetools.TTLCache for production).
    """
    def decorator(func: F) -> F:
        cache: dict[tuple[Any, ...], tuple[Any, float]] = {}

        @wraps(func)
        def wrapper(*args: Any) -> Any:
            now = time.monotonic()
            if args in cache:
                result, ts = cache[args]
                if now - ts < seconds:
                    log.debug("ttl_cache hit for %s", args)
                    return result
            result = func(*args)
            cache[args] = (result, now)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


@ttl_cache(seconds=2.0)
def slow_square(n: int) -> int:
    """Simulates an expensive computation."""
    time.sleep(0.1)
    return n * n


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fibonacci comparison
    t0 = time.perf_counter()
    log.info("fib_naive(30) = %d  (%.3fs)", fib_naive(30), time.perf_counter() - t0)

    t0 = time.perf_counter()
    log.info("fib_memoized(30) = %d  (%.3fs)", fib_memoized(30), time.perf_counter() - t0)

    t0 = time.perf_counter()
    log.info("fib_lru(30) = %d  (%.3fs)", fib_lru(30), time.perf_counter() - t0)
    log.info("fib_lru cache_info: %s", fib_lru.cache_info())

    # LRU cache demo
    cache = LRUCache(capacity=3)
    for k, v in [(1, 10), (2, 20), (3, 30)]:
        cache.put(k, v)
    log.info("get(1)=%d  — moves 1 to most-recent", cache.get(1))
    cache.put(4, 40)           # evicts key 2 (LRU after get(1) bumped 1)
    log.info("after put(4): %s", cache)
    log.info("get(2)=%d  — 2 was evicted", cache.get(2))

    # TTL demo
    log.info("slow_square(5) = %d  (cold)", slow_square(5))
    log.info("slow_square(5) = %d  (cached)", slow_square(5))
```

**Key observations:**

- `fib_naive(30)` takes ~0.3 s; `fib_memoized(30)` takes ~0.0001 s — orders of magnitude difference.
- `LRUCache.get` and `LRUCache.put` are both O(1) due to `OrderedDict.move_to_end`.
- `ttl_cache` shows the structure of a TTL wrapper; for production use `cachetools`.

---

## Exercises

1. **Bounded memoize with maxsize** — Extend the `memoize` decorator to accept a `maxsize: int` parameter. When the cache is full, evict the **oldest inserted** entry (FIFO policy). Use `collections.OrderedDict`.

2. **Climbing stairs** — A frog can jump 1 or 2 steps. Count the number of distinct ways to climb `n` steps. Implement `climb_ways(n: int) -> int` using `@lru_cache`. Verify for n=1→1, n=2→2, n=5→8.

3. **Coin change (minimum coins)** — Given `coins: list[int]` and `amount: int`, return the fewest coins needed to reach `amount`, or -1 if impossible. Implement `min_coins(coins: tuple[int, ...], amount: int) -> int` with `@lru_cache`. Note: use `tuple` not `list` for hashability.

4. **Cache statistics decorator** — Write a `tracked_cache` decorator that wraps any pure function and exposes a `.stats()` method returning `{"hits": int, "misses": int, "hit_rate": float}`. Do not use `lru_cache` internally — implement the dict yourself.

5. **LFU cache class** — Implement `class LFUCache` with `get(key) -> int` and `put(key, value)` methods supporting O(1) average operations. Use `dict` + `defaultdict(OrderedDict)` keyed by frequency. On a tie in frequency, evict the least recently used among the minimum-frequency group.

*(Solutions: ../solutions/week-02/day-5.md)*

---

## Self-check

- [ ] I can explain the LRU eviction policy and why it requires a hash map + doubly-linked list for O(1) operations.
- [ ] I can apply `@lru_cache` to any pure recursive function and know which argument types are forbidden.
- [ ] I can implement a basic `memoize` decorator from scratch without referencing `functools`.
- [ ] I can describe when LRU beats LFU and vice versa with a concrete access-pattern example.
- [ ] I can read `cache_info()` output and diagnose cache effectiveness from the hit rate.

---

## Supplementary links

- [Python docs — functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- [Python docs — collections.OrderedDict](https://docs.python.org/3/library/collections.html#collections.OrderedDict)
- [cachetools — TTL, LFU, RR caches](https://cachetools.readthedocs.io/)
- [LeetCode 146 — LRU Cache (canonical problem)](https://leetcode.com/problems/lru-cache/)
