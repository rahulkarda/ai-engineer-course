# Phase 2 · Week 02 · Day 5 — Solutions

## Exercise 1 — Bounded memoize with FIFO eviction

```python
"""
memoize with maxsize and FIFO eviction via OrderedDict.
O(1) amortised insert/lookup.
"""
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def memoize_fifo(maxsize: int) -> Callable[[F], F]:
    """Decorator factory: memoize with FIFO eviction when full."""
    def decorator(func: F) -> F:
        cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

        @wraps(func)
        def wrapper(*args: Any) -> Any:
            if args in cache:
                return cache[args]
            result = func(*args)
            cache[args] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)   # remove oldest inserted
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# Tests
call_count = 0

@memoize_fifo(maxsize=2)
def double(n: int) -> int:
    global call_count
    call_count += 1
    return n * 2

assert double(1) == 2 and call_count == 1
assert double(1) == 2 and call_count == 1   # cached
assert double(2) == 4 and call_count == 2
assert double(3) == 6 and call_count == 3   # evicts (1, 2) [FIFO, oldest = 1]
assert double(1) == 2 and call_count == 4   # 1 was evicted, recomputed
```

**FIFO vs LRU:** FIFO evicts the first-inserted entry regardless of how recently it was used. Here, after inserting `(1,2)` and `(2,4)` the cache is full; inserting `(3,6)` evicts `1` (first in), even though it was never accessed since insertion. LRU would evict `2` instead (least recently *accessed*).

---

## Exercise 2 — Climbing stairs

```python
"""
Distinct ways to climb n steps with 1 or 2 jumps.
Same recurrence as Fibonacci: ways(n) = ways(n-1) + ways(n-2).
"""
from functools import lru_cache


@lru_cache(maxsize=None)
def climb_ways(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    return climb_ways(n - 1) + climb_ways(n - 2)


# Tests
assert climb_ways(1) == 1
assert climb_ways(2) == 2
assert climb_ways(3) == 3    # (1+1+1), (1+2), (2+1)
assert climb_ways(5) == 8
assert climb_ways(10) == 89
```

**Pattern note:** This is Fibonacci shifted by one: `climb_ways(n) == fib(n+1)`. Base cases `n=1→1`, `n=2→2` set the offset. The `@lru_cache` turns the exponential tree into a linear scan — each of the n subproblems is solved exactly once.

---

## Exercise 3 — Coin change (minimum coins)

```python
"""
Minimum number of coins to make `amount`. -1 if impossible.
Use tuple for coins (hashable for lru_cache).
Top-down DP: min_coins(amount) = 1 + min over coins of min_coins(amount - coin).
"""
import math
from functools import lru_cache


def coin_change(coins: list[int], amount: int) -> int:
    coin_tuple = tuple(coins)

    @lru_cache(maxsize=None)
    def dp(remaining: int) -> int:
        if remaining == 0:
            return 0
        if remaining < 0:
            return math.inf  # type: ignore[return-value]
        best = math.inf
        for coin in coin_tuple:
            sub = dp(remaining - coin)
            if sub + 1 < best:
                best = sub + 1
        return best

    result = dp(amount)
    return result if result != math.inf else -1


# Tests
assert coin_change([1, 5, 10, 25], 36) == 3   # 25 + 10 + 1
assert coin_change([2], 3) == -1               # impossible
assert coin_change([1, 2, 5], 11) == 3         # 5 + 5 + 1
assert coin_change([186, 419, 83, 408], 6249) == 20
```

**Why `tuple` not `list`:** `lru_cache` hashes all arguments. `list` is mutable and unhashable; `tuple` is immutable and hashable. Passing `list` raises `TypeError: unhashable type: 'list'`.

**`math.inf` trick:** Using `math.inf` as a sentinel for "impossible" lets the `min` logic stay clean without special-casing `None`.

---

## Exercise 4 — Cache statistics decorator

```python
"""
tracked_cache: memoize + expose .stats() with hits, misses, hit_rate.
"""
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def tracked_cache(func: F) -> F:
    cache: dict[tuple[Any, ...], Any] = {}
    hits = 0
    misses = 0

    @wraps(func)
    def wrapper(*args: Any) -> Any:
        nonlocal hits, misses
        if args in cache:
            hits += 1
            return cache[args]
        misses += 1
        result = func(*args)
        cache[args] = result
        return result

    def stats() -> dict[str, Any]:
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total > 0 else 0.0,
        }

    wrapper.stats = stats          # type: ignore[attr-defined]
    wrapper.cache = cache          # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


# Tests
@tracked_cache
def expensive(n: int) -> int:
    return n ** 3

_ = expensive(2)   # miss
_ = expensive(2)   # hit
_ = expensive(3)   # miss
_ = expensive(2)   # hit

s = expensive.stats()
assert s["hits"] == 2
assert s["misses"] == 2
assert s["hit_rate"] == 0.5
```

---

## Exercise 5 — LFU cache class

```python
"""
LFU cache: O(1) get/put average using:
  - key_to_val: dict[key → value]
  - key_to_freq: dict[key → frequency]
  - freq_to_keys: dict[freq → OrderedDict[key → None]]  (ordered for LRU tiebreak)
  - min_freq: tracks current minimum frequency
"""
from collections import OrderedDict, defaultdict


class LFUCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.key_to_val: dict[int, int] = {}
        self.key_to_freq: dict[int, int] = {}
        self.freq_to_keys: dict[int, OrderedDict[int, None]] = defaultdict(OrderedDict)
        self.min_freq = 0

    def _bump_freq(self, key: int) -> None:
        """Increment frequency of key and update tracking structures."""
        freq = self.key_to_freq[key]
        self.key_to_freq[key] = freq + 1
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq = freq + 1
        self.freq_to_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.key_to_val:
            return -1
        self._bump_freq(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._bump_freq(key)
            return
        # Evict if at capacity
        if len(self.key_to_val) >= self.capacity:
            # Remove LRU key among min_freq group
            evict_key, _ = next(iter(self.freq_to_keys[self.min_freq].items()))
            del self.freq_to_keys[self.min_freq][evict_key]
            if not self.freq_to_keys[self.min_freq]:
                del self.freq_to_keys[self.min_freq]
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]
        # Insert new key with freq=1
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1


# Tests — from LeetCode 460
lfu = LFUCache(2)
lfu.put(1, 1)
lfu.put(2, 2)
assert lfu.get(1) == 1     # freq[1]=2, freq[2]=1; min_freq=1
lfu.put(3, 3)              # evict key 2 (min_freq=1, LRU=2)
assert lfu.get(2) == -1    # evicted
assert lfu.get(3) == 3
lfu.put(4, 4)              # evict key 3 (min_freq=1; 3 was accessed once)
assert lfu.get(1) == 1
assert lfu.get(3) == -1    # evicted
assert lfu.get(4) == 4
```

**Data structure rationale:**

- `key_to_val` and `key_to_freq` give O(1) value/frequency lookup.
- `freq_to_keys[f]` is an `OrderedDict` ordered by insertion time — enables LRU tiebreaking at O(1) with `popitem(last=False)` or `next(iter(...))`.
- `min_freq` is maintained incrementally: on insert it resets to 1; on frequency bump, if the bumped key was the last at `min_freq`, increment `min_freq`.
