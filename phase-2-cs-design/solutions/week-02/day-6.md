# Phase 2 · Week 02 · Day 6 — Solutions

## Problem 1 — Reverse a Linked List

```python
from dataclasses import dataclass


@dataclass
class ListNode:
    val: int
    next: "ListNode | None" = None


def reverse_list(head: ListNode | None) -> ListNode | None:
    """
    Iterative reversal. Maintain prev pointer; walk forward advancing three pointers.
    O(n) time, O(1) space.
    """
    prev: ListNode | None = None
    current = head
    while current is not None:
        next_node = current.next   # save next before we overwrite
        current.next = prev        # reverse the pointer
        prev = current             # advance prev
        current = next_node        # advance current
    return prev                    # prev is the new head


def list_to_python(head: ListNode | None) -> list[int]:
    result: list[int] = []
    while head:
        result.append(head.val)
        head = head.next
    return result


def python_to_list(vals: list[int]) -> ListNode | None:
    dummy = ListNode(0)
    cur = dummy
    for v in vals:
        cur.next = ListNode(v)
        cur = cur.next
    return dummy.next


# Tests
assert list_to_python(reverse_list(python_to_list([1, 2, 3, 4, 5]))) == [5, 4, 3, 2, 1]
assert list_to_python(reverse_list(python_to_list([]))) == []
assert list_to_python(reverse_list(python_to_list([7]))) == [7]
```

**Pointer diagram for `1→2→3`:**
```
Step 0: prev=None, cur=1→2→3
Step 1: save next=2; 1.next=None;  prev=1, cur=2
Step 2: save next=3; 2.next=1;     prev=2, cur=3
Step 3: save next=None; 3.next=2;  prev=3, cur=None
Return: 3→2→1→None
```

---

## Problem 2 — Valid Parentheses

```python
def is_valid(s: str) -> bool:
    """
    Push openers onto a stack; on closer, check the top matches.
    O(n) time, O(n) space.
    """
    matching: dict[str, str] = {")": "(", "}": "{", "]": "["}
    stack: list[str] = []
    for ch in s:
        if ch in "({[":
            stack.append(ch)
        elif ch in ")}]":
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()
    return len(stack) == 0


# Tests
assert is_valid("()") is True
assert is_valid("()[]{}") is True
assert is_valid("(]") is False
assert is_valid("([)]") is False
assert is_valid("{[]}") is True
assert is_valid("") is True
assert is_valid("]") is False     # closer with empty stack
```

**Edge case — empty string:** The final `len(stack) == 0` handles this correctly (returns `True`).

---

## Problem 3 — Linked List Cycle Detection

```python
def has_cycle(head: ListNode | None) -> bool:
    """
    Floyd's tortoise and hare: slow moves 1, fast moves 2.
    If there is a cycle they will meet; if not, fast reaches None.
    O(n) time, O(1) space.
    """
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next           # type: ignore[union-attr]
        fast = fast.next.next
        if slow is fast:
            return True
    return False


# Build a cycle manually for testing
def make_cycle(vals: list[int], cycle_pos: int) -> ListNode | None:
    """cycle_pos=-1 means no cycle."""
    if not vals:
        return None
    nodes = [ListNode(v) for v in vals]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if cycle_pos >= 0:
        nodes[-1].next = nodes[cycle_pos]
    return nodes[0]


# Tests
assert has_cycle(make_cycle([3, 2, 0, -4], cycle_pos=1)) is True
assert has_cycle(make_cycle([1, 2], cycle_pos=0)) is True
assert has_cycle(make_cycle([1], cycle_pos=-1)) is False
assert has_cycle(None) is False
```

**Why it works:** In a cycle of length L, slow and fast meet after at most L steps — fast "laps" slow at 1 step/iteration. If no cycle, fast exits in ⌊n/2⌋ steps.

---

## Problem 4 — Sliding Window Maximum

```python
from collections import deque


def max_sliding_window(nums: list[int], k: int) -> list[int]:
    """
    Monotonic decreasing deque: stores indices.
    Front of deque is always the index of the current window max.
    O(n) time — each element is added and removed at most once.
    """
    if not nums or k == 0:
        return []

    dq: deque[int] = deque()   # stores indices, values in decreasing order
    result: list[int] = []

    for i, val in enumerate(nums):
        # Remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Remove from back all indices whose values are <= current value
        # (they can never be the max while current element is in the window)
        while dq and nums[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        # Window is full
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# Tests
assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
assert max_sliding_window([1], 1) == [1]
assert max_sliding_window([1, -1], 1) == [1, -1]
assert max_sliding_window([], 3) == []
```

**Deque invariant:** Values at stored indices are strictly decreasing from front to back. Front = max of current window. Any element smaller than the new element is useless (the new element will outlast it in every future window) → pop from back before appending.

---

## Problem 5 — Group Shifted Strings

```python
from collections import defaultdict


def group_shifted(words: list[str]) -> list[list[str]]:
    """
    Canonical key: difference sequence normalised so first char = 'a'.
    shift_key("bcd") = shift_key("xyz") = (0, 1, 2) — consecutive steps.
    O(n·L) time, O(n·L) space, where L = max word length.
    """
    def shift_key(word: str) -> tuple[int, ...]:
        if len(word) == 1:
            return (0,)
        return tuple(
            (ord(word[i]) - ord(word[0])) % 26
            for i in range(len(word))
        )

    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for word in words:
        groups[shift_key(word)].append(word)
    return list(groups.values())


# Tests
result = group_shifted(["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"])
# Sort each group and sort list of groups for deterministic comparison
normalised = sorted(sorted(g) for g in result)
expected = sorted([
    sorted(["abc", "bcd", "xyz"]),
    sorted(["acef"]),
    sorted(["az", "ba"]),
    sorted(["a", "z"]),
])
assert normalised == expected
```

**Canonical form walkthrough for `"az"`:** `(ord('a')-ord('a'))%26=0`, `(ord('z')-ord('a'))%26=25` → key `(0, 25)`. For `"ba"`: `(ord('b')-ord('b'))%26=0`, `(ord('a')-ord('b'))%26=(-1)%26=25` → key `(0, 25)`. Same key → same group.

---

## Problem 6 — LRU Cache with Expiry

```python
import time
from collections import OrderedDict


class TTLLRUCache:
    """
    LRU eviction + per-entry TTL.
    Expired entries are treated as absent and lazily removed on access.
    """

    def __init__(self, capacity: int, ttl: float) -> None:
        self.capacity = capacity
        self.ttl = ttl
        # key → (value, inserted_at)
        self._store: OrderedDict[int, tuple[int, float]] = OrderedDict()

    def _is_expired(self, key: int) -> bool:
        _, ts = self._store[key]
        return time.monotonic() - ts >= self.ttl

    def get(self, key: int) -> int:
        if key not in self._store:
            return -1
        if self._is_expired(key):
            del self._store[key]
            return -1
        self._store.move_to_end(key)
        return self._store[key][0]

    def put(self, key: int, value: int) -> None:
        # Remove and re-insert to update recency and timestamp
        if key in self._store:
            del self._store[key]
        self._store[key] = (value, time.monotonic())
        self._store.move_to_end(key)
        # Evict LRU entries until under capacity (skip expired ones)
        while len(self._store) > self.capacity:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]


# Smoke test (no sleep — just structural verification)
cache = TTLLRUCache(capacity=2, ttl=10.0)
cache.put(1, 10)
cache.put(2, 20)
assert cache.get(1) == 10
cache.put(3, 30)   # evicts 2 (LRU after get(1) bumped 1)
assert cache.get(2) == -1
assert cache.get(3) == 30
```

**TTL testing strategy:** In real tests, monkeypatch `time.monotonic` to return a controlled value rather than calling `time.sleep` — fast and deterministic.

---

## Problem 7 — Flatten a Multilevel Doubly Linked List

```python
from dataclasses import dataclass, field


@dataclass
class DLLNode:
    val: int
    prev: "DLLNode | None" = field(default=None, repr=False)
    next: "DLLNode | None" = field(default=None, repr=False)
    child: "DLLNode | None" = field(default=None, repr=False)


def flatten(head: DLLNode | None) -> DLLNode | None:
    """
    Iterative depth-first using a stack.
    When a node has a child: push next onto stack, splice child in as next.
    When current.next is None and stack non-empty: pop and link.
    O(n) time, O(depth) space.
    """
    if head is None:
        return None

    stack: list[DLLNode] = []
    current: DLLNode | None = head

    while current is not None:
        if current.child is not None:
            if current.next is not None:
                stack.append(current.next)
            # Splice child in
            current.next = current.child
            current.child.prev = current
            current.child = None

        if current.next is None and stack:
            tail = stack.pop()
            current.next = tail
            tail.prev = current

        current = current.next

    return head


def flatten_to_list(head: DLLNode | None) -> list[int]:
    result: list[int] = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# Build test case from the problem statement
n1, n2, n3, n4, n5, n6 = (DLLNode(i) for i in [1, 2, 3, 4, 5, 6])
n7, n8, n9, n10 = (DLLNode(i) for i in [7, 8, 9, 10])
n11, n12 = DLLNode(11), DLLNode(12)

for a, b in [(n1, n2), (n2, n3), (n3, n4), (n4, n5), (n5, n6)]:
    a.next = b; b.prev = a
for a, b in [(n7, n8), (n8, n9), (n9, n10)]:
    a.next = b; b.prev = a
n11.next = n12; n12.prev = n11
n3.child = n7
n8.child = n11

assert flatten_to_list(flatten(n1)) == [1, 2, 3, 7, 8, 11, 12, 9, 10, 4, 5, 6]
```

**Stack trace at key moments:**
1. Visit `3`: has child `7`. Push `4→5→6` onto stack. `3.next = 7`.
2. Visit `8`: has child `11`. Push `9→10` onto stack. `8.next = 11`.
3. Visit `12`: no child, no next. Pop `9→10`. `12.next = 9`.
4. Visit `10`: no child, no next. Pop `4→5→6`. `10.next = 4`.

---

## Problem 8 — Top K Frequent Elements (Bucket Sort, O(n))

```python
from collections import Counter


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    1. Count frequencies: O(n).
    2. Bucket by frequency — bucket[f] holds all numbers with frequency f.
       Max possible frequency is n, so we need n+1 buckets.
    3. Collect from highest-frequency bucket downward until k elements found: O(n).
    Total: O(n) time, O(n) space.
    """
    freq = Counter(nums)                         # O(n)
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)               # O(n) total

    result: list[int] = []
    for i in range(len(buckets) - 1, -1, -1):   # high freq → low freq
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result


# Tests
assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
assert top_k_frequent([1], 1) == [1]
assert sorted(top_k_frequent([1, 2], 2)) == [1, 2]
```

**Why not `heapq.nlargest`?** `heapq.nlargest(k, freq.items(), key=...)` is O(n log k), which beats O(n log n) sort but not O(n) bucket sort. For a true O(n) guarantee, bucket sort is the right tool.

---

## Problem 9 — Hit Counter (Circular Buffer)

```python
class HitCounter:
    """
    Circular buffer of 300 slots.
    Each slot: (timestamp_of_last_write, hit_count_in_that_second).
    Slot index = timestamp % 300.
    get_hits sums counts for slots whose stored timestamp is in (ts-299, ts].
    O(1) hit, O(300) = O(1) get_hits.
    """

    WINDOW = 300

    def __init__(self) -> None:
        # Each slot: (last_timestamp_written, count)
        self._slots: list[tuple[int, int]] = [(0, 0)] * self.WINDOW

    def hit(self, timestamp: int) -> None:
        idx = timestamp % self.WINDOW
        ts, count = self._slots[idx]
        if ts == timestamp:
            self._slots[idx] = (timestamp, count + 1)
        else:
            # New second — overwrite stale slot
            self._slots[idx] = (timestamp, 1)

    def get_hits(self, timestamp: int) -> int:
        total = 0
        cutoff = timestamp - self.WINDOW + 1   # inclusive lower bound
        for ts, count in self._slots:
            if ts >= cutoff:
                total += count
        return total


# Tests
hc = HitCounter()
hc.hit(1); hc.hit(1); hc.hit(300)
assert hc.get_hits(300) == 3
hc.hit(301)
assert hc.get_hits(301) == 2   # ts=1 now outside window (301-300=1 not >= 2)
hc.hit(300)
assert hc.get_hits(300) == 3   # ts=300 still in window, count=2; ts=1 gone
```

**Circular buffer key insight:** Two events at timestamp `t1` and `t2` where `t1 ≡ t2 (mod 300)` share a slot. When `t2 > t1 + 299`, `t1` is outside any future window — safely overwrite. The `if ts == timestamp` guard prevents double-counting within one second.

---

## Problem 10 — Size-Bounded LRU Cache

```python
from collections import OrderedDict


class SizeBoundedLRUCache:
    """
    LRU eviction by total byte size, not item count.
    OrderedDict for O(1) recency tracking.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._store: OrderedDict[str, bytes] = OrderedDict()
        self._current_bytes = 0

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    def get(self, key: str) -> bytes | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: bytes) -> None:
        size = len(value)
        if size > self.max_bytes:
            return   # item too large to ever fit — silently reject

        # Remove existing entry (reclaim its bytes before eviction loop)
        if key in self._store:
            self._current_bytes -= len(self._store[key])
            del self._store[key]

        # Evict LRU entries until there is room
        while self._current_bytes + size > self.max_bytes and self._store:
            lru_key, lru_val = self._store.popitem(last=False)
            self._current_bytes -= len(lru_val)

        self._store[key] = value
        self._current_bytes += size

    def __repr__(self) -> str:
        return f"SizeBoundedLRUCache(used={self._current_bytes}/{self.max_bytes})"


# Tests
cache = SizeBoundedLRUCache(max_bytes=10)
cache.put("a", b"hello")   # 5 bytes → total=5
cache.put("b", b"world!")  # 6 bytes → 5+6=11 > 10 → evict "a", total=6
assert cache.get("a") is None
assert cache.get("b") == b"world!"
assert cache.current_bytes == 6

cache.put("c", b"hi")      # 2 bytes → 6+2=8 ≤ 10 → fits
assert cache.current_bytes == 8

cache.put("d", b"x" * 20)  # 20 > 10 → rejected
assert cache.get("d") is None
assert cache.current_bytes == 8

# LRU eviction order
cache2 = SizeBoundedLRUCache(max_bytes=10)
cache2.put("x", b"12345")  # 5 bytes
cache2.put("y", b"67890")  # 5 bytes → total=10
cache2.get("x")            # access x → x is now MRU
cache2.put("z", b"ab")     # 2 bytes → needs to evict: LRU is y (not x)
assert cache2.get("y") is None
assert cache2.get("x") == b"12345"
assert cache2.get("z") == b"ab"
```

**Why evict by bytes not count:** A single 9-byte entry consumes 90% of a 10-byte cache. Count-based LRU would evict 9 one-byte entries before touching it, which is suboptimal. Real HTTP caches (Varnish, nginx) and CPU caches use size-aware eviction.
