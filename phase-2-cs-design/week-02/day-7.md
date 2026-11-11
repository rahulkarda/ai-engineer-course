# Phase 2 · Week 02 · Day 7 — LRU Cache: Doubly Linked List + Hash Map

> Objective: Implement an LRU cache from scratch achieving O(1) get and put using a dict and a doubly linked list.  ·  Time: ~3 hr (Sunday project day)  ·  Prereqs: [[phase-2-cs-design/week-01/day-1]]

---

## Concept

**Least Recently Used (LRU)** is an eviction policy for bounded caches: when the cache is full and a new key must be inserted, the key that was accessed *least recently* is evicted first.

### Why LRU?

| Policy | Evicts | Weakness |
|---|---|---|
| FIFO | Oldest inserted | Ignores access frequency |
| LFU | Least frequently used | Cold-start problem; complex |
| LRU | Least recently *accessed* | Favors temporal locality |
| Random | Random | Unpredictable; sometimes good |

LRU works well for workloads with **temporal locality** — recently used data is likely to be used again soon. CPU L1/L2 caches, Redis, Python's `functools.lru_cache`, and browser page caches all use LRU or a variant.

### The constraint

Two operations, both **O(1) average**:

```
get(key)           → value or -1/None if not present
put(key, value)    → insert/update; evict LRU entry if over capacity
```

A naive list scan gives O(n) for finding the LRU candidate. We need a smarter structure.

---

## How it works internally

### Key insight: two data structures, one responsibility each

| Structure | Responsibility |
|---|---|
| `dict[key → node]` | O(1) lookup by key |
| Doubly linked list | O(1) insert/delete anywhere; order encodes recency |

**Recency order** is maintained by convention:
- **Head (left sentinel)** — most recently used end
- **Tail (right sentinel)** — least recently used end

```
HEAD ←→ [most recent] ←→ ... ←→ [least recent] ←→ TAIL
```

Every time a node is accessed or inserted it is **moved to just after HEAD**.  
When capacity is exceeded the node **just before TAIL** is evicted.

### Why doubly linked (not singly linked)?

Removing an arbitrary node from a singly linked list requires traversing from the head to find the predecessor — O(n).  
A doubly linked node carries `prev` and `next` pointers, so removal is O(1): just re-wire the neighbours.

### Node anatomy

```
┌──────────────────────────────────┐
│  prev │  key  │  value  │  next  │
└──────────────────────────────────┘
```

Sentinel nodes (`head`, `tail`) are dummy nodes with no real key/value.  
They simplify edge cases: the list is never truly empty, so "insert after head" and "remove before tail" always have valid neighbours.

### Operation walkthroughs

**`get(key)`**
1. Key not in dict → return None. Done.
2. Key in dict → get node.
3. Move node to front (after head): unlink from current position, re-link after head.
4. Return node.value.

**`put(key, value)`**
1. If key exists → update value, move to front. Done.
2. Create new node.
3. Insert node after head.
4. Add to dict.
5. If `len(dict) > capacity` → evict: remove node before tail, delete from dict.

### Complexity summary

| Operation | Time | Space |
|---|---|---|
| get | O(1) | O(1) |
| put | O(1) | O(1) |
| delete | O(1) | O(1) |
| Total space | — | O(capacity) |

---

## Worked example

Below is a **partial implementation**. Read every comment carefully — the `# TODO` sections are what you will complete in the exercises.

```python
"""
lru_cache_starter.py — partial LRU cache implementation (Python 3.12+)

Complete the TODO sections as part of the exercises.
Do NOT look at solutions until you have attempted every TODO.
"""
from __future__ import annotations

import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node: the building block of the doubly linked list
# ---------------------------------------------------------------------------

class _Node:
    """Internal doubly linked list node.

    Sentinel nodes (head, tail) are created with key=0, value=0.
    Real nodes carry the actual cache key and value.
    """

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Any = 0, value: Any = 0) -> None:
        self.key: Any = key
        self.value: Any = value
        self.prev: _Node | None = None   # pointer to predecessor
        self.next: _Node | None = None   # pointer to successor

    def __repr__(self) -> str:
        return f"_Node(key={self.key!r}, value={self.value!r})"


# ---------------------------------------------------------------------------
# LRUCache — starter scaffold
# ---------------------------------------------------------------------------

class LRUCache:
    """Least Recently Used cache with O(1) get and put.

    Parameters
    ----------
    capacity:
        Maximum number of key-value pairs to hold.  Must be >= 1.

    Usage
    -----
    >>> cache = LRUCache(capacity=2)
    >>> cache.put(1, "a")
    >>> cache.put(2, "b")
    >>> cache.get(1)       # returns "a"; key 1 is now most recent
    'a'
    >>> cache.put(3, "c")  # evicts key 2 (least recently used)
    >>> cache.get(2)       # returns None — evicted
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._capacity = capacity
        self._cache: dict[Any, _Node] = {}   # key → node

        # Sentinel nodes — never evicted, never stored in _cache
        self._head = _Node()   # dummy head (most-recent end)
        self._tail = _Node()   # dummy tail (least-recent end)

        # Wire sentinels together to form an empty list:
        #   HEAD ←→ TAIL
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------------
    # Private helpers — implement these first (Exercises 1 & 2)
    # ------------------------------------------------------------------

    def _remove(self, node: _Node) -> None:
        """Unlink `node` from its current position in the list.

        Does NOT delete the node — the caller decides what to do with it.

        Before:  ... ←→ prev ←→ node ←→ nxt ←→ ...
        After:   ... ←→ prev ←→ nxt ←→ ...
        """
        # TODO (Exercise 1a): implement _remove
        ...

    def _insert_after_head(self, node: _Node) -> None:
        """Link `node` immediately after the head sentinel.

        Before:  HEAD ←→ first ←→ ...
        After:   HEAD ←→ node ←→ first ←→ ...
        """
        # TODO (Exercise 1b): implement _insert_after_head
        ...

    def _move_to_front(self, node: _Node) -> None:
        """Move an existing node to the most-recently-used position."""
        self._remove(node)
        self._insert_after_head(node)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: Any) -> Any | None:
        """Return value for key, or None if not present.

        Side-effect: marks key as most recently used.
        """
        # TODO (Exercise 2a): implement get
        ...

    def put(self, key: Any, value: Any) -> None:
        """Insert or update key→value.

        If key exists: update value and move to front.
        If key is new and cache is full: evict LRU entry first, then insert.
        """
        # TODO (Exercise 2b): implement put
        ...

    # ------------------------------------------------------------------
    # Utilities (already implemented — study these)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def keys_mru_order(self) -> list[Any]:
        """Return all keys from most-recently-used to least-recently-used."""
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


# ---------------------------------------------------------------------------
# Smoke test — run to check your implementation as you build it
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    log.info("=== Smoke test ===")

    cache = LRUCache(capacity=2)
    cache.put(1, "a")
    cache.put(2, "b")
    log.info("After put(1,'a'), put(2,'b'): %s", cache)

    result = cache.get(1)
    log.info("get(1) → %r  (expected 'a')", result)
    assert result == "a"

    cache.put(3, "c")   # should evict key 2
    log.info("After put(3,'c') [evicts 2]: %s", cache)

    assert cache.get(2) is None, "key 2 should have been evicted"
    log.info("get(2) → None  ✓  (evicted)")

    assert cache.get(1) == "a"
    assert cache.get(3) == "c"
    log.info("Smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
```

**How to use the starter file:**

1. Copy it to your working directory as `lru_cache.py`.
2. Run `python lru_cache.py` — all assertions will fail until you fill in the TODOs.
3. Work through Exercises 1 → 5 in order. The smoke test passes when Exercises 1–2 are done.

---

## Exercises

**Exercise 1 — Core list operations**

Implement `_remove(node)` and `_insert_after_head(node)` in the starter above.

Constraints:
- Each method is 3–5 lines of pointer assignments. No loops.
- Do not touch `_cache` or `_capacity` inside these helpers — they are pure linked-list operations.

Draw the pointer state on paper before writing any code.

---

**Exercise 2 — Public API**

Implement `get(key)` and `put(key, value)`.

For `get`:
- Return `None` if key not in cache.
- Otherwise return the value and move the node to the front.

For `put`:
- If key exists: update `.value`, move to front. No eviction needed.
- If key is new: create a node, insert after head, add to `_cache`.
- After inserting: if `len(_cache) > _capacity`, evict the node just before `_tail` — remove from list AND delete from `_cache`.

Run the smoke test after implementing both. All assertions must pass.

---

**Exercise 3 — Trace by hand**

Given `LRUCache(capacity=3)`, trace each operation. After each step write the list state in MRU→LRU order and state whether any eviction occurred.

```
put(1, "one")
put(2, "two")
put(3, "three")
get(1)
put(4, "four")
get(2)
put(5, "five")
```

Your answer format:

```
After put(1, "one"):   [1]
After put(2, "two"):   [2, 1]
...
```

---

**Exercise 4 — Edge cases**

Write at least three pytest test functions (no answers here — write the tests and make them pass):

a) `get` on an empty cache returns `None`.
b) `put` the same key twice — the value is updated, size stays the same, node moves to front.
c) A cache of capacity 1 — every `put` of a new key evicts the previous entry.

```python
# lru_cache_test.py skeleton
import pytest
from lru_cache import LRUCache

def test_get_empty_cache() -> None:
    # TODO

def test_put_duplicate_key_updates_value() -> None:
    # TODO

def test_capacity_one_always_evicts() -> None:
    # TODO
```

---

**Exercise 5 — Complexity justification**

Write a short paragraph (4–6 sentences) explaining why each of the four pointer operations in `_remove` and `_insert_after_head` is O(1). Your answer must address:

- Why no traversal is needed.
- What property of the doubly linked list makes this possible.
- What the dict provides that makes `get` O(1) end-to-end.

*(Solutions: ../solutions/week-02/day-7.md)*

---

## Self-check

- [ ] I can draw the sentinel-node doubly linked list pattern from memory and explain why sentinels simplify edge cases.
- [ ] I can implement `_remove` and `_insert_after_head` without looking at notes — each in under 2 minutes.
- [ ] I can trace through `get` and `put` for a capacity-2 cache with 5 operations on paper with no errors.
- [ ] I can explain why both `get` and `put` are O(1) amortised — referencing the dict and the DLL separately.
- [ ] My pytest suite passes for the three edge cases in Exercise 4.

---

## Supplementary links

- [LeetCode 146 — LRU Cache](https://leetcode.com/problems/lru-cache/) (the canonical problem; do not look at solutions until you've finished the exercises)
- [Python `collections.OrderedDict` source](https://github.com/python/cpython/blob/main/Lib/collections/__init__.py) — Python's standard library implements LRU with OrderedDict; worth reading after completing this day
- [How Python's `functools.lru_cache` works](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- [Visualgo — Linked List visualizer](https://visualgo.net/en/list)
