# Phase 2 · Week 02 · Day 6 — Saturday Deep Practice

> Objective: Consolidate the week's topics (linked lists, stacks/queues, hash tables, caching) through timed problem-solving under exam-like conditions.  ·  Time: ~4 hr  ·  Prereqs: [[week-02/day-4]], [[week-02/day-5]]

---

## Instructions

Work through all problems in order. Allocate roughly 20–25 minutes per problem. Write Python 3.12+ with type hints. No `print` — use `logging` or assertions.

Problems increase in difficulty. Problems 1–3 are warm-up, 4–6 are intermediate, 7–10 require combining multiple concepts.

Do not look at solutions until you have attempted each problem. If stuck beyond 30 minutes, move on and return at the end.

*(Solutions: ../solutions/week-02/day-6.md)*

---

## Problem 1 — Reverse a Linked List (Warm-up)

**Difficulty:** Easy

Implement a singly-linked list node and a function to reverse the list in-place.

```python
from dataclasses import dataclass

@dataclass
class ListNode:
    val: int
    next: "ListNode | None" = None
```

Write:

```python
def reverse_list(head: ListNode | None) -> ListNode | None:
    ...
```

**Constraints:**
- O(n) time, O(1) space (iterative, not recursive).
- Do not allocate a new list.

**Examples:**
- `1 → 2 → 3 → 4 → 5` → `5 → 4 → 3 → 2 → 1`
- `None` → `None`
- Single node `[7]` → `[7]`

---

## Problem 2 — Valid Parentheses (Warm-up)

**Difficulty:** Easy

Given a string `s` containing only `(`, `)`, `{`, `}`, `[`, `]`, return `True` if every opener has a matching closer in correct order, `False` otherwise.

```python
def is_valid(s: str) -> bool:
    ...
```

**Constraints:** O(n) time, O(n) space.

**Examples:**
- `"()"` → `True`
- `"()[]{}"` → `True`
- `"(]"` → `False`
- `"([)]"` → `False`
- `"{[]}"` → `True`
- `""` → `True`

---

## Problem 3 — Linked List Cycle Detection (Warm-up)

**Difficulty:** Easy

Detect whether a linked list contains a cycle. Return `True` if it does.

```python
def has_cycle(head: ListNode | None) -> bool:
    ...
```

**Constraints:** O(n) time, **O(1) space** — Floyd's tortoise and hare algorithm. Do not use a `set`.

**Examples:**
- `3 → 2 → 0 → -4`, where `-4.next = 2` → `True`
- `1 → 2`, where `2.next = 1` → `True`
- `[1]` no cycle → `False`

---

## Problem 4 — Sliding Window Maximum (Intermediate)

**Difficulty:** Medium

Given `nums: list[int]` and window size `k`, return a list of the maximum value in each sliding window of size `k`.

```python
def max_sliding_window(nums: list[int], k: int) -> list[int]:
    ...
```

**Constraints:** O(n) time — use a `collections.deque` as a monotonic decreasing queue. An O(n·k) brute force exists but will not receive full credit.

**Examples:**
- `nums=[1,3,-1,-3,5,3,6,7]`, `k=3` → `[3,3,5,5,6,7]`
- `nums=[1]`, `k=1` → `[1]`
- `nums=[1,-1]`, `k=1` → `[1,-1]`

---

## Problem 5 — Group Shifted Strings (Intermediate)

**Difficulty:** Medium

Two strings are "shift-equivalent" if you can shift every character of one by the same amount (cyclically) to get the other. `"abc"` and `"xyz"` are equivalent: `a→x, b→y, c→z` all shift by 23.

Given `words: list[str]`, group shift-equivalent strings together.

```python
def group_shifted(words: list[str]) -> list[list[str]]:
    ...
```

**Hint:** Normalise each word to a canonical "difference sequence" where the first character maps to `'a'`. Use that tuple as the dict key.

**Examples:**
- `["abc","bcd","acef","xyz","az","ba","a","z"]`  
  → `[["abc","bcd","xyz"],["acef"],["az","ba"],["a","z"]]` (order within groups and order of groups may differ)

---

## Problem 6 — LRU Cache with Expiry (Intermediate)

**Difficulty:** Medium

Extend an LRU cache to support per-entry TTL (time-to-live in seconds).

```python
class TTLLRUCache:
    def __init__(self, capacity: int, ttl: float) -> None: ...
    def get(self, key: int) -> int: ...          # returns -1 if missing or expired
    def put(self, key: int, value: int) -> None: ...
```

**Requirements:**
- Expired entries must be treated as absent (return -1) and removed lazily on access.
- LRU eviction applies only to non-expired entries — expired entries do not count toward `capacity`.
- Use `time.monotonic()` for timestamps.

**Examples** (hint: mock time in tests via a parameter or monkeypatch):
- Put `(1, 10)` with ttl=1s. `get(1)` immediately → 10. Wait 1.1s. `get(1)` → -1.

---

## Problem 7 — Flatten a Multilevel Doubly Linked List (Hard)

**Difficulty:** Hard

A doubly linked list node may have a `child` pointer to another doubly linked list. Flatten the list so all nodes appear in a single-level, depth-first order.

```python
@dataclass
class DLLNode:
    val: int
    prev: "DLLNode | None" = None
    next: "DLLNode | None" = None
    child: "DLLNode | None" = None


def flatten(head: DLLNode | None) -> DLLNode | None:
    ...
```

**Constraints:** O(n) time, O(depth) space (stack). After flattening, all `child` pointers must be `None`.

**Example:**
```
Input:
  1 - 2 - 3 - 4 - 5 - 6
          |
          7 - 8 - 9 - 10
              |
              11 - 12

Output (depth-first):
  1 - 2 - 3 - 7 - 8 - 11 - 12 - 9 - 10 - 4 - 5 - 6
```

---

## Problem 8 — Top K Frequent Elements with O(n) Guarantee (Hard)

**Difficulty:** Hard

Given `nums: list[int]`, return the `k` most frequent elements. Order within the result does not matter.

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    ...
```

**Constraints:** O(n) time — use bucket sort on frequency, **not** `heapq.nlargest` (which is O(n log k)).

**Examples:**
- `[1,1,1,2,2,3]`, k=2 → `[1,2]`
- `[1]`, k=1 → `[1]`

---

## Problem 9 — Design a Hit Counter (Hard)

**Difficulty:** Hard

Design a hit counter that tracks the number of hits received in the last 5 minutes (300 seconds).

```python
class HitCounter:
    def hit(self, timestamp: int) -> None:
        """Record a hit at `timestamp` (seconds, monotonically increasing)."""
        ...

    def get_hits(self, timestamp: int) -> int:
        """Return number of hits in the window (timestamp-299, timestamp]."""
        ...
```

**Requirements:**
- Handle multiple hits at the same timestamp.
- O(1) time for `hit`, O(1) amortised for `get_hits`.
- Use a circular buffer of size 300 (`list[tuple[int, int]]` — each slot stores `(timestamp, count)`).

**Examples:**
```
hit(1), hit(1), hit(300)
get_hits(300) → 3

hit(301)
get_hits(301) → 2   # hit(1) is now outside the 300s window
```

---

## Problem 10 — LRU Cache Supporting Variable-Size Items (Hard / Challenge)

**Difficulty:** Hard / Challenge

A real cache (e.g., a CPU L1 cache or HTTP response cache) has a byte budget, not a count budget. Implement a cache evicted by total byte size rather than item count.

```python
class SizeBoundedLRUCache:
    def __init__(self, max_bytes: int) -> None: ...

    def get(self, key: str) -> bytes | None: ...

    def put(self, key: str, value: bytes) -> None:
        """
        Insert key→value. If value.size > max_bytes, do not insert.
        Evict LRU entries until the new entry fits, then insert.
        """
        ...

    @property
    def current_bytes(self) -> int: ...
```

**Requirements:**
- `get` and `put` O(1) amortised.
- After each `put`, `current_bytes <= max_bytes`.
- A single item larger than `max_bytes` is silently rejected.

**Example:**
```python
cache = SizeBoundedLRUCache(max_bytes=10)
cache.put("a", b"hello")   # 5 bytes
cache.put("b", b"world!")  # 6 bytes → evicts "a" (5 bytes used → 6 fits after evict)
assert cache.get("a") is None
assert cache.get("b") == b"world!"
```
