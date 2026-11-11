# Week 02 Assessment — Answer Key

> For instructor / self-review only. Do not read until you have submitted your own answers.

---

## Conceptual answers

**1. Singly vs doubly linked list deletion**

A singly linked list node only knows its `next` pointer — to remove a node, you must re-wire its predecessor's `next` pointer to skip over it. Finding that predecessor requires traversal from the head: O(n). A doubly linked node carries both `prev` and `next`, so the predecessor is reachable directly: `node.prev.next = node.next; node.next.prev = node.prev` — four assignments, O(1). The O(1) guarantee breaks down when you do not have the node itself but only a key: you still need to find the node via a dict or scan, though the dict lookup is O(1) average, not O(n). It also breaks down if the node has been freed/invalidated and the pointer is stale — a concern in manual-memory languages like C, not in Python's garbage-collected runtime.

**2. Python dict worst-case and hash randomisation**

Python's dict is a hash table with open addressing. Average-case O(1) lookup relies on hash collisions being rare. The worst case O(n) occurs when all keys hash to the same slot, forcing the probe sequence to scan every entry. In practice, Python's hash function distributes most values well. To defend against deliberate collision attacks (e.g., from adversarial user input in a web server), Python 3.3+ introduced **hash randomisation**: `PYTHONHASHSEED` is set to a random value at interpreter start, so the hash of strings and bytes changes across processes. An attacker who cannot predict the seed cannot craft a collision-inducing input set.

**3. LRU vs LFU vs FIFO for product catalogue**

FIFO ignores access frequency entirely — a product fetched once at startup stays in cache as long as an equally old but frequently accessed one. LFU evicts the least-frequently accessed item, which is excellent for a stable "hot" set (top-20 bestsellers) but suffers from a cold-start problem: a newly launched product is evicted immediately because its count is 1. LRU evicts the least recently accessed item, which handles temporal locality well — a product accessed yesterday but not today will eventually age out. For a product catalogue, LRU is the pragmatic default: it handles both trending products (recency) and cache warming after restarts. If you know the workload has a stable long-term hot set (e.g., the same 1,000 SKUs dominate traffic for weeks), LFU would reduce churn. If access timestamps are available, an LFU-with-aging (TinyLFU, used in Caffeine/Guava) outperforms both.

**4. List-only LRU complexity**

Linear scan `get`: O(n) per access to find the key. `put` with append and truncate: O(n) to find if key exists (avoid duplicate), plus O(1) append, plus O(n) truncation in the worst case (removing from the front of a list shifts all elements). For small caches (n ≤ ~100) the constant factors of the list are low and the list approach may outperform dict+DLL due to cache-line friendliness. At n ≈ 1,000 the O(n) scans become noticeable. At n = 50,000 (the scenario in Q3) each cache access costs up to 50,000 comparisons — roughly 50 µs at 10⁹ ops/sec — versus ~100 ns for O(1) dict lookup. At that scale, list-based LRU is impractical for any high-throughput path.

**5. Sentinel nodes**

Sentinel (dummy) nodes are placeholder nodes at the head and tail of the list that never hold real data. Their purpose is to eliminate special-casing for empty-list and single-element operations. Without sentinels, `_remove` must check `if node.prev is None` (head of list) and `if node.next is None` (tail), and `_insert_after_head` must handle an empty list differently from a non-empty one. With sentinels, `head.next` is always valid (it points to either the first real node or the tail sentinel), and `tail.prev` is always valid, so the same pointer-surgery code works for every case. You could implement LRU without sentinels by adding `None` checks in `_remove` and `_insert_after_head`, but the code would be longer, more error-prone, and harder to reason about.

---

## Coding challenge — full solution

```python
"""
solution.py — Week 02 assessment LRU cache (Python 3.12+)

Implements LRUCache with:
  - get(key: int) -> int   (returns -1 if missing)
  - put(key: int, value: int) -> None

Complexity: O(1) get and put (average).
Space: O(capacity).
"""
from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class _Node:
    """Doubly linked list node."""

    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: int = 0, value: int = 0) -> None:
        self.key: int = key
        self.value: int = value
        self.prev: _Node | None = None
        self.next: _Node | None = None


class LRUCache:
    """O(1) LRU cache — dict + doubly linked list.

    Uses integer keys and values; get() returns -1 for missing keys
    to match the LeetCode 146 convention.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._cache: dict[int, _Node] = {}

        # Sentinel nodes — never evicted, never in _cache
        self._head: _Node = _Node()  # most-recent end
        self._tail: _Node = _Node()  # least-recent end
        self._head.next = self._tail
        self._tail.prev = self._head

        log.debug("LRUCache initialised with capacity=%d", capacity)

    # ------------------------------------------------------------------
    # Private helpers — O(1) pointer operations
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

    def get(self, key: int) -> int:
        """Return value for key, or -1 if not present.

        Marks key as most recently used. O(1) average.
        """
        if key not in self._cache:
            log.debug("get(%d) → -1 (miss)", key)
            return -1
        node = self._cache[key]
        self._move_to_front(node)
        log.debug("get(%d) → %d (hit)", key, node.value)
        return node.value

    def put(self, key: int, value: int) -> None:
        """Insert or update key→value.

        If key exists: update value, move to front.
        If new and at capacity: evict LRU entry first, then insert.
        O(1) average.
        """
        if key in self._cache:
            node = self._cache[key]
            node.value = value
            self._move_to_front(node)
            log.debug("put(%d, %d) → updated existing", key, value)
            return

        node = _Node(key, value)
        self._insert_after_head(node)
        self._cache[key] = node

        if len(self._cache) > self._capacity:
            lru = self._tail.prev
            self._remove(lru)          # type: ignore[arg-type]
            del self._cache[lru.key]   # type: ignore[union-attr]
            log.debug("put(%d, %d) → evicted key %d", key, value, lru.key)  # type: ignore[union-attr]
        else:
            log.debug("put(%d, %d) → inserted (size=%d)", key, value, len(self._cache))

    def __len__(self) -> int:
        return len(self._cache)
```

### Acceptance test run

```python
"""
test_solution.py — pytest suite for Week 02 assessment solution (Python 3.12+)
Run: pytest test_solution.py -v
"""
import pytest
from solution import LRUCache


def test_basic_get_put() -> None:
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1


def test_eviction_evicts_lru() -> None:
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.get(1)        # key 1 becomes most recent
    c.put(3, 3)     # evicts key 2
    assert c.get(2) == -1
    assert c.get(1) == 1
    assert c.get(3) == 3


def test_update_existing_key() -> None:
    c = LRUCache(2)
    c.put(2, 1)
    c.put(2, 2)
    assert c.get(2) == 2


def test_capacity_one() -> None:
    c = LRUCache(1)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == -1
    assert c.get(2) == 2


def test_get_moves_to_front() -> None:
    c = LRUCache(3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    c.get(1)
    c.put(4, 40)
    assert c.get(2) == -1
    assert c.get(1) == 10
    assert c.get(3) == 30
    assert c.get(4) == 40


def test_put_then_get_sequential() -> None:
    c = LRUCache(3)
    for i in range(1, 4):
        c.put(i, i * 100)
    for i in range(1, 4):
        assert c.get(i) == i * 100


# Additional edge-case tests

def test_get_on_empty_cache() -> None:
    c = LRUCache(5)
    assert c.get(999) == -1


def test_size_never_exceeds_capacity() -> None:
    c = LRUCache(3)
    for i in range(20):
        c.put(i, i)
        assert len(c) <= 3


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        LRUCache(capacity=0)
    with pytest.raises(ValueError):
        LRUCache(capacity=-5)
```
