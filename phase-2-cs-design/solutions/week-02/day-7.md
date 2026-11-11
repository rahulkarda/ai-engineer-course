# Phase 2 · Week 02 · Day 7 — Solutions

> Solutions for `week-02/day-7.md`. Attempt all exercises independently before reading.

---

## Exercise 1 — Core list operations

```python
def _remove(self, node: _Node) -> None:
    """Unlink node from its current position. O(1)."""
    prev = node.prev
    nxt = node.next
    prev.next = nxt   # type: ignore[union-attr]
    nxt.prev = prev   # type: ignore[union-attr]


def _insert_after_head(self, node: _Node) -> None:
    """Link node immediately after head sentinel. O(1)."""
    first = self._head.next
    self._head.next = node
    node.prev = self._head
    node.next = first
    first.prev = node  # type: ignore[union-attr]
```

**Why this is O(1):** each operation assigns exactly four pointers. No loop, no traversal. The doubly linked structure lets us reach both neighbours directly from the node itself.

---

## Exercise 2 — Public API

```python
def get(self, key: Any) -> Any | None:
    if key not in self._cache:
        return None
    node = self._cache[key]
    self._move_to_front(node)
    return node.value


def put(self, key: Any, value: Any) -> None:
    if key in self._cache:
        node = self._cache[key]
        node.value = value
        self._move_to_front(node)
        return

    node = _Node(key, value)
    self._insert_after_head(node)
    self._cache[key] = node

    if len(self._cache) > self._capacity:
        # Evict LRU: node just before tail
        lru = self._tail.prev
        self._remove(lru)         # type: ignore[arg-type]
        del self._cache[lru.key]  # type: ignore[union-attr]
```

### Complete, runnable implementation

```python
"""
lru_cache.py — complete LRU cache (Python 3.12+)
"""
from __future__ import annotations

import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any = 0, value: Any = 0) -> None:
        self.key: Any = key
        self.value: Any = value
        self.prev: _Node | None = None
        self.next: _Node | None = None

    def __repr__(self) -> str:
        return f"_Node(key={self.key!r}, value={self.value!r})"


class LRUCache:
    """O(1) LRU cache backed by a dict and a doubly linked list."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._cache: dict[Any, _Node] = {}

        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove(self, node: _Node) -> None:
        prev = node.prev
        nxt = node.next
        prev.next = nxt   # type: ignore[union-attr]
        nxt.prev = prev   # type: ignore[union-attr]

    def _insert_after_head(self, node: _Node) -> None:
        first = self._head.next
        self._head.next = node
        node.prev = self._head
        node.next = first
        first.prev = node  # type: ignore[union-attr]

    def _move_to_front(self, node: _Node) -> None:
        self._remove(node)
        self._insert_after_head(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: Any) -> Any | None:
        if key not in self._cache:
            return None
        node = self._cache[key]
        self._move_to_front(node)
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            self._move_to_front(node)
            return

        node = _Node(key, value)
        self._insert_after_head(node)
        self._cache[key] = node

        if len(self._cache) > self._capacity:
            lru = self._tail.prev
            self._remove(lru)         # type: ignore[arg-type]
            del self._cache[lru.key]  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def keys_mru_order(self) -> list[Any]:
        result: list[Any] = []
        node = self._head.next
        while node is not self._tail:
            result.append(node.key)   # type: ignore[union-attr]
            node = node.next          # type: ignore[union-attr]
        return result

    def __repr__(self) -> str:
        order = self._head.next
        parts: list[str] = []
        while order is not self._tail:
            parts.append(f"{order.key!r}:{order.value!r}")  # type: ignore[union-attr]
            order = order.next  # type: ignore[union-attr]
        return f"LRUCache(capacity={self._capacity}, [{', '.join(parts)}])"
```

---

## Exercise 3 — Trace by hand

```
put(1, "one")    → [1]             (no eviction)
put(2, "two")    → [2, 1]          (no eviction)
put(3, "three")  → [3, 2, 1]       (no eviction — at capacity)
get(1)           → [1, 3, 2]       (key 1 moves to front; returns "one")
put(4, "four")   → [4, 1, 3]       (EVICTS key 2 — LRU was 2)
get(2)           → None            (key 2 was evicted)
put(5, "five")   → [5, 4, 1]       (EVICTS key 3 — LRU was 3)
```

Step-by-step reasoning for the tricky steps:

- After `get(1)`: key 1 accessed, moved to front. Order was [3,2,1]; becomes [1,3,2].
- `put(4, "four")`: cache is full (keys 1,3,2). LRU is the tail-prev node = key 2. Evict 2. Insert 4 at front → [4,1,3].
- `get(2)`: key 2 not in dict → None.
- `put(5, "five")`: cache is full (keys 4,1,3). LRU = tail-prev = key 3. Evict 3. Insert 5 → [5,4,1].

---

## Exercise 4 — Edge cases (pytest)

```python
"""
lru_cache_test.py — pytest suite for LRUCache (day-7 exercises)
Python 3.12+
"""
import pytest
from lru_cache import LRUCache


def test_get_empty_cache() -> None:
    cache = LRUCache(capacity=2)
    assert cache.get("missing") is None
    assert cache.get(0) is None


def test_put_duplicate_key_updates_value() -> None:
    cache = LRUCache(capacity=3)
    cache.put("x", 1)
    cache.put("y", 2)
    cache.put("x", 99)          # update; should NOT grow beyond 2 entries
    assert len(cache) == 2
    assert cache.get("x") == 99
    # "x" should now be at the front (most recently used)
    assert cache.keys_mru_order()[0] == "x"


def test_capacity_one_always_evicts() -> None:
    cache = LRUCache(capacity=1)
    cache.put("a", 1)
    assert cache.get("a") == 1
    cache.put("b", 2)           # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    cache.put("c", 3)           # should evict "b"
    assert cache.get("b") is None
    assert cache.get("c") == 3


# Bonus tests (not in the exercise but good to have)

def test_order_maintained_after_multiple_gets() -> None:
    cache = LRUCache(capacity=3)
    for k in (1, 2, 3):
        cache.put(k, k * 10)
    cache.get(1)
    cache.get(2)
    # MRU order should be [2, 1, 3]
    assert cache.keys_mru_order() == [2, 1, 3]


def test_eviction_removes_from_dict() -> None:
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)   # evicts "a"
    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
```

---

## Exercise 5 — Complexity justification

`_remove` and `_insert_after_head` each perform exactly four pointer re-assignments. Neither contains a loop, recursive call, or search — they operate only on nodes already in hand. This is possible because a **doubly linked** node carries both a `prev` and a `next` pointer, so any node can reach both its neighbours in O(1) without traversal from the head. The dict (`_cache`) provides the O(1) bridge from an arbitrary key to the corresponding node — without it, finding the node to move would require an O(n) linear scan of the list. Together, `dict[key → node]` + doubly linked list gives O(1) for the full `get` path: hash lookup (O(1)) followed by pointer surgery (O(1)).
