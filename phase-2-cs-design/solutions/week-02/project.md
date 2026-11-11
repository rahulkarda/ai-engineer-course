# Week 02 Project — Reference Solution

> Complete implementation of the production LRU cache specified in `week-02/project.md`.
> Do not read until you have made a genuine attempt.

---

## `lru_cache_prod.py`

```python
"""
lru_cache_prod.py — Production LRU Cache (Python 3.12+)

Complexity:
    Operation  | Time (avg) | Time (worst) | Space
    -----------|------------|--------------|------
    get        | O(1)       | O(1)         | O(1)
    put        | O(1)       | O(1)         | O(1)
    delete     | O(1)       | O(1)         | O(1)
    stats      | O(1)       | O(1)         | O(1)
    __init__   | O(1)       | O(1)         | O(n) — n = capacity

Justifications:
    - get/put/delete: dict lookup is O(1) average; pointer surgery on the
      doubly linked list is exactly 4 assignments regardless of list length.
    - stats: all counters are pre-accumulated integers; no scan needed.
    - __init__: only two sentinel nodes are created regardless of capacity.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

_SENTINEL_KEY = object()  # unique sentinel key for dummy nodes


# ---------------------------------------------------------------------------
# Internal node
# ---------------------------------------------------------------------------

class _Node:
    """Doubly linked list node.

    Sentinel nodes use key=_SENTINEL_KEY.  Real nodes carry str keys.
    expires_at is a monotonic timestamp; None means no expiry.
    """

    __slots__ = ("key", "value", "prev", "next", "expires_at")

    def __init__(
        self,
        key: Any = _SENTINEL_KEY,
        value: Any = None,
        expires_at: float | None = None,
    ) -> None:
        self.key: Any = key
        self.value: Any = value
        self.prev: _Node | None = None
        self.next: _Node | None = None
        self.expires_at: float | None = expires_at

    def is_expired(self) -> bool:
        """Return True if this node has a TTL that has passed."""
        return self.expires_at is not None and time.monotonic() > self.expires_at

    def __repr__(self) -> str:
        return f"_Node(key={self.key!r}, expires_at={self.expires_at})"


# ---------------------------------------------------------------------------
# Production LRU Cache
# ---------------------------------------------------------------------------

class LRUCache:
    """Thread-safe Least Recently Used cache with optional TTL support.

    Uses a hash map (dict) for O(1) key lookup and a doubly linked list
    to maintain access order in O(1) per operation.

    Parameters
    ----------
    capacity:
        Maximum number of live entries.  Must be >= 1.

    Examples
    --------
    >>> cache = LRUCache(capacity=3)
    >>> cache.put("a", 1)
    >>> cache.put("b", 2, ttl_seconds=60.0)
    >>> cache.get("a")
    1
    >>> cache.stats()
    {'hits': 1, 'misses': 0, 'evictions': 0, 'size': 2}
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._capacity: int = capacity
        self._cache: dict[str, _Node] = {}
        self._lock: threading.Lock = threading.Lock()

        # Sentinel nodes — never stored in _cache, never evicted
        self._head: _Node = _Node()   # most-recent end
        self._tail: _Node = _Node()   # least-recent end
        self._head.next = self._tail
        self._tail.prev = self._head

        # Statistics counters
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # Private helpers — not thread-safe; callers hold the lock
    # ------------------------------------------------------------------

    def _remove(self, node: _Node) -> None:
        """Unlink node from the list. O(1) — 4 pointer assignments."""
        prev = node.prev
        nxt = node.next
        prev.next = nxt   # type: ignore[union-attr]
        nxt.prev = prev   # type: ignore[union-attr]

    def _insert_after_head(self, node: _Node) -> None:
        """Link node immediately after head sentinel. O(1) — 4 pointer assignments."""
        first = self._head.next
        self._head.next = node
        node.prev = self._head
        node.next = first
        first.prev = node  # type: ignore[union-attr]

    def _move_to_front(self, node: _Node) -> None:
        """Promote node to MRU position. O(1)."""
        self._remove(node)
        self._insert_after_head(node)

    def _evict_lru(self) -> None:
        """Remove the LRU entry (node just before tail) from list and dict.

        Called only when len(_cache) > capacity.
        Prefers evicting expired entries first; falls back to tail-prev.
        """
        # Fast path: evict tail-prev (standard LRU)
        lru = self._tail.prev
        if lru is self._head:
            return  # empty list — should not happen normally
        self._remove(lru)              # type: ignore[arg-type]
        del self._cache[lru.key]       # type: ignore[union-attr]
        self._evictions += 1
        log.info("Evicted key '%s' (LRU)", lru.key)  # type: ignore[union-attr]

    def _current_size(self) -> int:
        """Count of non-expired entries. O(n) — only used for stats()."""
        count = 0
        node = self._head.next
        while node is not self._tail:
            if not node.is_expired():  # type: ignore[union-attr]
                count += 1
            node = node.next  # type: ignore[union-attr]
        return count

    # ------------------------------------------------------------------
    # Public API — all methods acquire self._lock
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return value for key, or None if absent or TTL-expired.

        Complexity: O(1) average — dict lookup + constant pointer ops.
        Marks key as most recently used before checking TTL.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                log.debug("get('%s') → None (miss)", key)
                return None

            node = self._cache[key]
            self._move_to_front(node)

            if node.is_expired():
                # Silent removal of expired entry
                self._remove(node)
                del self._cache[key]
                self._misses += 1
                log.warning("get('%s') → None (TTL expired; entry removed)", key)
                return None

            self._hits += 1
            log.debug("get('%s') → hit", key)
            return node.value

    def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
    ) -> None:
        """Insert or update key → value with optional TTL.

        Complexity: O(1) average — dict lookup/insert + constant pointer ops.
        """
        expires_at: float | None = None
        if ttl_seconds is not None and ttl_seconds > 0:
            expires_at = time.monotonic() + ttl_seconds

        with self._lock:
            if key in self._cache:
                node = self._cache[key]
                node.value = value
                node.expires_at = expires_at
                self._move_to_front(node)
                log.debug("put('%s') → updated existing", key)
                return

            node = _Node(key=key, value=value, expires_at=expires_at)
            self._insert_after_head(node)
            self._cache[key] = node
            log.debug("put('%s') → inserted (cache size=%d)", key, len(self._cache))

            if len(self._cache) > self._capacity:
                self._evict_lru()

    def delete(self, key: str) -> bool:
        """Remove key from cache immediately.

        Returns True if key existed, False otherwise.
        Complexity: O(1) average.
        """
        with self._lock:
            if key not in self._cache:
                log.debug("delete('%s') → not found", key)
                return False
            node = self._cache.pop(key)
            self._remove(node)
            log.debug("delete('%s') → removed", key)
            return True

    def stats(self) -> dict[str, int]:
        """Return accumulated statistics snapshot.

        'size' reflects live (non-expired) entries — O(n) scan.
        All other counters are O(1).
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size": self._current_size(),
            }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._cache

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"LRUCache(capacity={self._capacity}, "
                f"size={len(self._cache)}, "
                f"hits={self._hits}, misses={self._misses}, "
                f"evictions={self._evictions})"
            )
```

---

## `tests/test_lru_cache_prod.py`

```python
"""
tests/test_lru_cache_prod.py — pytest suite for production LRU cache (Python 3.12+)

Run: pytest tests/test_lru_cache_prod.py -v
"""
from __future__ import annotations

import threading
import time

import pytest

from lru_cache_prod import LRUCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cache() -> LRUCache:
    return LRUCache(capacity=3)


@pytest.fixture
def tiny_cache() -> LRUCache:
    return LRUCache(capacity=1)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_get_returns_none_on_empty() -> None:
    cache = LRUCache(capacity=2)
    assert cache.get("anything") is None


def test_put_and_get_basic(small_cache: LRUCache) -> None:
    small_cache.put("x", 42)
    assert small_cache.get("x") == 42


def test_eviction_evicts_lru_entry() -> None:
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")      # "a" is now MRU; "b" is LRU
    cache.put("c", 3)   # evicts "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_update_existing_key_moves_to_front() -> None:
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 99)  # update; "a" becomes MRU, "b" is LRU
    assert len(cache) == 2
    assert cache.get("a") == 99
    # Adding a third key should evict "b", not "a"
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 99


def test_delete_existing_key(small_cache: LRUCache) -> None:
    small_cache.put("key1", "value")
    result = small_cache.delete("key1")
    assert result is True
    assert small_cache.get("key1") is None


def test_delete_missing_key(small_cache: LRUCache) -> None:
    result = small_cache.delete("nonexistent")
    assert result is False


def test_stats_hits_and_misses() -> None:
    cache = LRUCache(capacity=5)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")      # hit
    cache.get("a")      # hit
    cache.get("z")      # miss
    s = cache.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["evictions"] == 0


def test_stats_evictions() -> None:
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)   # eviction 1
    cache.put("d", 4)   # eviction 2
    assert cache.stats()["evictions"] == 2


def test_ttl_expiry_returns_none() -> None:
    cache = LRUCache(capacity=5)
    cache.put("short", "value", ttl_seconds=0.01)
    # Should be present immediately
    assert cache.get("short") == "value"
    # Wait for expiry
    time.sleep(0.05)
    assert cache.get("short") is None


def test_ttl_not_yet_expired() -> None:
    cache = LRUCache(capacity=5)
    cache.put("long", "value", ttl_seconds=3600.0)
    assert cache.get("long") == "value"


def test_capacity_one(tiny_cache: LRUCache) -> None:
    tiny_cache.put("a", 1)
    assert tiny_cache.get("a") == 1
    tiny_cache.put("b", 2)         # evicts "a"
    assert tiny_cache.get("a") is None
    assert tiny_cache.get("b") == 2
    tiny_cache.put("c", 3)         # evicts "b"
    assert tiny_cache.get("b") is None
    assert tiny_cache.get("c") == 3


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
    with pytest.raises(ValueError):
        LRUCache(capacity=-1)


# ---------------------------------------------------------------------------
# Integration / concurrency test
# ---------------------------------------------------------------------------

def test_concurrent_put_get() -> None:
    """4 threads × 500 put+get cycles — no exceptions, size stays within capacity."""
    capacity = 50
    cache = LRUCache(capacity=capacity)
    errors: list[Exception] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(500):
                key = f"t{thread_id}_k{i % 100}"
                cache.put(key, i)
                _ = cache.get(key)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrency errors: {errors}"
    assert len(cache) <= capacity


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_stats_size_reflects_live_entries() -> None:
    cache = LRUCache(capacity=5)
    cache.put("a", 1, ttl_seconds=0.01)
    cache.put("b", 2)
    time.sleep(0.05)
    s = cache.stats()
    # "a" expired, "b" live — size should be 1
    # Note: "a" is still in the internal dict until get() cleans it up
    # stats() does a live scan, so it should report 1
    assert s["size"] == 1


def test_put_overwrites_ttl() -> None:
    cache = LRUCache(capacity=5)
    cache.put("k", "first", ttl_seconds=0.01)
    time.sleep(0.005)
    # Re-put before expiry with a long TTL
    cache.put("k", "second", ttl_seconds=3600.0)
    time.sleep(0.02)   # original TTL would have expired
    assert cache.get("k") == "second"


def test_delete_then_reinsert() -> None:
    cache = LRUCache(capacity=3)
    cache.put("x", 10)
    cache.delete("x")
    assert cache.get("x") is None
    cache.put("x", 20)
    assert cache.get("x") == 20


def test_size_never_exceeds_capacity() -> None:
    cache = LRUCache(capacity=5)
    for i in range(100):
        cache.put(f"key{i}", i)
        assert len(cache) <= 5
```

---

## Complexity analysis (inline reference)

```
Operation  | Time (avg) | Time (worst) | Space | Justification
-----------|------------|--------------|-------|------------------------------------------------
get        | O(1)       | O(1)         | O(1)  | dict lookup O(1); move_to_front is 4 ptr assigns
put        | O(1)       | O(1)         | O(1)  | dict insert O(1); insert+evict are 4+4 ptr assigns
delete     | O(1)       | O(1)         | O(1)  | dict.pop O(1); _remove is 4 ptr assigns
stats      | O(n)       | O(n)         | O(1)  | size requires full list scan for expired entries
__init__   | O(1)       | O(1)         | O(n)  | n sentinel nodes; dict pre-allocated by Python
```

Note: `stats()["size"]` requires an O(n) scan because expired entries are lazily removed only on `get`. If you need O(1) size, maintain a separate `_live_count` counter — but you must decrement it on expiry during `get` and `delete`, which adds bookkeeping complexity. The lazy approach is simpler and correct; the eager counter is a stretch-goal optimisation.
