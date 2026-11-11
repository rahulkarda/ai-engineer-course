# Phase 2 · Week 03 · Day 5 — Heaps, `heapq`, & Priority Queues

> Objective: Understand heap structure, implement heap operations, and use `heapq` for priority queues.  ·  Time: ~1 hr  ·  Prereqs: [[day-1 (binary trees)]]

## Concept

A **heap** is a complete binary tree stored in an array that satisfies the **heap property**:

- **Min-heap**: every parent ≤ both children → root is the minimum.
- **Max-heap**: every parent ≥ both children → root is the maximum.

Heaps power priority queues — data structures where `pop` always returns the most-priority element.

### Array representation

A complete binary tree maps cleanly to an array (0-indexed):
```
parent(i) = (i - 1) // 2
left(i)   = 2*i + 1
right(i)  = 2*i + 2
```

No pointers needed — the tree lives in contiguous memory, giving cache-friendly access.

```
Array: [1, 3, 6, 5, 9, 8]

Tree:
       1
      / \
     3   6
    / \ /
   5  9 8
```

### Core operations

| Operation | Time | Description |
|---|---|---|
| `peek` (min) | O(1) | `heap[0]` |
| `push` | O(log n) | append + sift-up |
| `pop` (min) | O(log n) | swap root with last, remove last, sift-down |
| `heapify` | O(n) | build heap from arbitrary array |

**Sift-up**: newly inserted element bubbles up, swapping with parent while smaller.
**Sift-down**: new root (placed after pop) sinks down, swapping with the smaller child while larger.

## How it works internally

### Why `heapify` is O(n), not O(n log n)

Inserting `n` elements one-by-one costs O(n log n). `heapify` works bottom-up: nodes at the bottom half are already leaves (trivially valid heaps), and nodes near the root do most sifting but there are few of them. The geometric series sums to O(n).

### Python's `heapq`

`heapq` is a **min-heap** only. For a max-heap: negate all values on push and negate again on pop. For priority queues with ties: use a `(priority, counter, item)` tuple — counter breaks ties deterministically.

## Worked example

```python
"""
day5_heap.py — Heap from scratch + heapq priority queue patterns.
Python 3.12+
"""

from __future__ import annotations
import heapq
import logging
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── MinHeap from scratch ─────────────────────────────────────────────────────

class MinHeap:
    """Integer min-heap backed by a list."""

    def __init__(self) -> None:
        self._data: list[int] = []

    def __len__(self) -> int:
        return len(self._data)

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def push(self, val: int) -> None:
        self._data.append(val)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty heap")
        # swap root with last element, remove last
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        val = self._data.pop()
        if self._data:
            self._sift_down(0)
        return val

    def _sift_up(self, i: int) -> None:
        data = self._data
        while i > 0:
            parent = (i - 1) // 2
            if data[i] < data[parent]:
                data[i], data[parent] = data[parent], data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        data = self._data
        n = len(data)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and data[left] < data[smallest]:
                smallest = left
            if right < n and data[right] < data[smallest]:
                smallest = right
            if smallest == i:
                break
            data[i], data[smallest] = data[smallest], data[i]
            i = smallest

    @classmethod
    def heapify(cls, values: list[int]) -> "MinHeap":
        """Build heap in O(n) by sifting down from last internal node."""
        h = cls()
        h._data = values[:]
        n = len(h._data)
        # last internal node index = (n-2)//2
        for i in range((n - 2) // 2, -1, -1):
            h._sift_down(i)
        return h


# ── Demo: MinHeap ────────────────────────────────────────────────────────────

h = MinHeap()
for v in [5, 3, 8, 1, 4]:
    h.push(v)

sorted_output: list[int] = []
while h:
    sorted_output.append(h.pop())
log.info("MinHeap sorted output: %s", sorted_output)  # [1,3,4,5,8]

h2 = MinHeap.heapify([5, 3, 8, 1, 4])
log.info("Heapify peek: %d", h2.peek())  # 1


# ── heapq — Python stdlib ────────────────────────────────────────────────────

nums = [5, 3, 8, 1, 4]
heapq.heapify(nums)
log.info("heapq after heapify: %s", nums)   # valid min-heap array

heapq.heappush(nums, 2)
log.info("After push 2: min=%d", nums[0])  # 1

log.info("heappop: %d", heapq.heappop(nums))  # 1
log.info("After pop: min=%d", nums[0])         # 2


# ── Max-heap via negation ────────────────────────────────────────────────────

max_heap: list[int] = []
for v in [5, 3, 8, 1, 4]:
    heapq.heappush(max_heap, -v)

log.info("Max-heap top: %d", -max_heap[0])    # 8
log.info("Max-heap pop: %d", -heapq.heappop(max_heap))  # 8


# ── Priority Queue with tie-breaking ────────────────────────────────────────

@dataclass(order=False)
class Task:
    name: str
    priority: int  # lower number = higher priority


pq: list[tuple[int, int, Task]] = []
counter = 0  # tie-breaker

def pq_push(task: Task) -> None:
    global counter
    heapq.heappush(pq, (task.priority, counter, task))
    counter += 1

def pq_pop() -> Task:
    _, _, task = heapq.heappop(pq)
    return task

pq_push(Task("low_priority", 10))
pq_push(Task("critical", 1))
pq_push(Task("medium", 5))

while pq:
    t = pq_pop()
    log.info("Processing: %s (priority %d)", t.name, t.priority)
# critical(1) → medium(5) → low_priority(10)


# ── nlargest / nsmallest ─────────────────────────────────────────────────────

data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
log.info("3 largest:  %s", heapq.nlargest(3, data))   # [9, 6, 5]
log.info("3 smallest: %s", heapq.nsmallest(3, data))  # [1, 1, 2]
```

### Sift-down trace — pop from `[1, 3, 6, 5, 9, 8]`

```
Swap root(1) with last(8): [8, 3, 6, 5, 9]  (removed 1)
sift_down(0): data[0]=8, left=3, right=6 → smallest=1 (val=3)
  swap 0,1: [3, 8, 6, 5, 9]
  sift_down(1): data[1]=8, left=5, right=9 → smallest=3 (val=5)
    swap 1,3: [3, 5, 6, 8, 9]
    sift_down(3): no children (8<n? left=7≥5)
result: [3, 5, 6, 8, 9]  ✓ min-heap
```

## Exercises

1. **K-th smallest** — given an unsorted list and integer `k`, return the `k`-th smallest element using `heapq`. Implement two approaches: (a) push all + pop k times; (b) maintain a max-heap of size k. Compare their time complexity.

2. **Merge k sorted lists** — given `k` sorted lists, merge them into a single sorted list using a min-heap. Time complexity?

3. **Running median** — implement a `RunningMedian` class with `add(val)` and `get_median()`. Use two heaps: a max-heap for the lower half, a min-heap for the upper half. Each `add` is O(log n); `get_median` is O(1).

4. **Heap sort** — implement `heap_sort(arr: list[int]) -> list[int]` using your `MinHeap`. What is the time and space complexity?

5. **`heapify` is O(n)** — experimentally verify: time `heapify` on random lists of sizes 1000, 10000, 100000 and plot (or print) the ratios. What do you expect if it were O(n log n)?

(Solutions: ../solutions/week-03/day-5.md)

## Self-check

- [ ] I can draw a min-heap and its array representation
- [ ] I can implement sift-up and sift-down without reference
- [ ] I know why heapify is O(n) while inserting n elements is O(n log n)
- [ ] I can use `heapq` for min-heap, max-heap, and priority queue with ties
- [ ] I can implement running median with two heaps

## Supplementary links

- [Python `heapq` docs](https://docs.python.org/3.12/library/heapq.html)
- [Heap visualization — visualgo](https://visualgo.net/en/heap)
- [LeetCode 215 — Kth Largest Element](https://leetcode.com/problems/kth-largest-element-in-an-array/)
- [LeetCode 23 — Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
