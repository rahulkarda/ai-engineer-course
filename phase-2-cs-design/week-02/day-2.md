# Phase 2 · Week 02 · Day 2 — Stacks, Queues, and `collections.deque`

> Objective: Implement stacks and queues from first principles and understand when to reach for `collections.deque`.  ·  Time: ~1 hr  ·  Prereqs: [[phase-2-cs-design/week-02/day-1]]

---

## Concept

### Stack — LIFO

A **stack** (Last In, First Out) supports two primary operations:
- `push(value)` — add to the top.
- `pop()` — remove and return the top.

```
push(3) → [1, 2, 3]
pop()   → 3, stack becomes [1, 2]
```

Classic uses: function call stack, undo/redo, expression parsing, DFS.

### Queue — FIFO

A **queue** (First In, First Out) supports:
- `enqueue(value)` — add to the back.
- `dequeue()` — remove and return the front.

```
enqueue(3) → [1, 2, 3]
dequeue()  → 1, queue becomes [2, 3]
```

Classic uses: BFS, task scheduling, producer-consumer buffers, rate-limiters.

### `collections.deque` — the right tool for both

Python's `list` has O(1) `append` (right end) but **O(n)** `pop(0)` / `insert(0, v)` because all elements shift. `collections.deque` is a doubly-linked list of fixed-size blocks that gives **O(1)** at both ends, making it the standard implementation vehicle for both stacks and queues.

| Operation | `list` | `deque` |
|---|---|---|
| Append right | O(1) amortized | O(1) |
| Pop right | O(1) | O(1) |
| Append left | O(n) | O(1) |
| Pop left | O(n) | O(1) |
| Indexed access | O(1) | O(n) |
| `maxlen` ring buffer | manual | built-in |

### Monotonic Stack

A **monotonic stack** keeps elements in strictly increasing or decreasing order. When a new element is pushed, all elements that violate the ordering are popped first. Used for "next greater element" problems in O(n).

---

## How it works internally

### `collections.deque` implementation

CPython implements `deque` as a **doubly linked list of fixed-size blocks** (64 items each on 64-bit CPython). Each block is a C array. The deque stores pointers to the leftmost and rightmost blocks plus indices into them.

- Appending to either end either fills an existing block slot (O(1)) or allocates a new block (amortized O(1)).
- Random access walks the block chain — O(n) overall but fast in practice for short deques.
- `maxlen` turns the deque into a **ring buffer**: when full, the opposite end is automatically dropped.

### Stack frames as stacks

The Python interpreter maintains a call stack. Each function call pushes a frame; `return` pops it. `RecursionError` is literally a stack overflow — Python's default limit is 1000 frames.

### Queue amortization using two stacks

An elegant O(1) amortized queue can be built from two stacks (`inbox` and `outbox`). Enqueue always pushes to `inbox`. Dequeue pops from `outbox`; when `outbox` is empty, reverse `inbox` into `outbox` (O(n) but amortized O(1) over all operations).

```
inbox [3, 4]   outbox []
dequeue → transfer: outbox [4, 3]
dequeue → 4    outbox [3]
dequeue → 3    outbox []
```

---

## Worked example

```python
"""
stacks_queues.py — stack and queue implementations + deque usage
Python 3.12+
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Generic, TypeVar

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Stack backed by a list
# ---------------------------------------------------------------------------

class Stack(Generic[T]):
    """
    LIFO stack. Uses list internally — O(1) push/pop at the right end.
    Never use list.pop(0) or list.insert(0, v): those are O(n).
    """

    def __init__(self) -> None:
        self._data: list[T] = []

    def push(self, value: T) -> None:
        """Add to top. O(1) amortized."""
        self._data.append(value)

    def pop(self) -> T:
        """Remove and return top. O(1). Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> T:
        """Return top without removing. O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Stack({self._data!r})  ← top"


# ---------------------------------------------------------------------------
# Queue backed by collections.deque
# ---------------------------------------------------------------------------

class Queue(Generic[T]):
    """
    FIFO queue. Uses deque to get O(1) at both ends.
    Enqueue to the right; dequeue from the left.
    """

    def __init__(self) -> None:
        self._data: deque[T] = deque()

    def enqueue(self, value: T) -> None:
        """Add to back. O(1)."""
        self._data.append(value)

    def dequeue(self) -> T:
        """Remove and return front. O(1). Raises IndexError if empty."""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def peek(self) -> T:
        """Return front without removing. O(1)."""
        if self.is_empty():
            raise IndexError("peek at empty queue")
        return self._data[0]

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Queue(front → {list(self._data)!r})"


# ---------------------------------------------------------------------------
# Queue using two stacks (O(1) amortized)
# ---------------------------------------------------------------------------

class TwoStackQueue(Generic[T]):
    """
    FIFO queue built from two stacks.
    Demonstrates amortized analysis: each element is moved at most once.
    """

    def __init__(self) -> None:
        self._inbox: list[T] = []    # enqueue here
        self._outbox: list[T] = []   # dequeue here

    def enqueue(self, value: T) -> None:
        """Push to inbox. O(1)."""
        self._inbox.append(value)

    def dequeue(self) -> T:
        """
        Pop from outbox. If outbox empty, reverse inbox into it first.
        Amortized O(1): each element crosses from inbox to outbox exactly once.
        """
        if not self._outbox:
            if not self._inbox:
                raise IndexError("dequeue from empty queue")
            # Transfer all at once — amortized cost
            while self._inbox:
                self._outbox.append(self._inbox.pop())
        return self._outbox.pop()

    def is_empty(self) -> bool:
        return not self._inbox and not self._outbox

    def __len__(self) -> int:
        return len(self._inbox) + len(self._outbox)


# ---------------------------------------------------------------------------
# Monotonic stack — next greater element
# ---------------------------------------------------------------------------

def next_greater_element(nums: list[int]) -> list[int]:
    """
    For each element, find the next element to its right that is greater.
    Returns -1 if no such element exists.

    Uses a monotonic decreasing stack. O(n) time, O(n) space.

    Example: [2, 1, 4, 3] → [4, 4, -1, -1]
    """
    result = [-1] * len(nums)
    stack: Stack[int] = Stack()  # stores indices

    for i, val in enumerate(nums):
        # Pop all indices whose values are smaller than current val
        while not stack.is_empty() and nums[stack.peek()] < val:
            idx = stack.pop()
            result[idx] = val     # current val is the "next greater"
        stack.push(i)

    return result


# ---------------------------------------------------------------------------
# deque as a sliding-window maximum (ring-buffer pattern)
# ---------------------------------------------------------------------------

def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """
    Return the maximum in each window of size k.
    O(n) time using a monotonic deque that stores indices.

    Invariant: deque front is always the index of the window maximum.
    """
    if not nums or k == 0:
        return []

    dq: deque[int] = deque()   # stores indices, front = max index
    result: list[int] = []

    for i, val in enumerate(nums):
        # Remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Maintain decreasing order: remove smaller elements from back
        while dq and nums[dq[-1]] < val:
            dq.pop()
        dq.append(i)
        # Record max once first window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Stack ===")
    s: Stack[int] = Stack()
    for v in [1, 2, 3, 4]:
        s.push(v)
    log.info("Stack after pushing 1-4: %s", s)
    log.info("Pop: %d", s.pop())
    log.info("Peek: %d", s.peek())

    log.info("")
    log.info("=== Queue ===")
    q: Queue[str] = Queue()
    for item in ["task-1", "task-2", "task-3"]:
        q.enqueue(item)
    log.info("Queue: %s", q)
    log.info("Dequeue: %s", q.dequeue())

    log.info("")
    log.info("=== TwoStackQueue ===")
    tsq: TwoStackQueue[int] = TwoStackQueue()
    for v in [10, 20, 30]:
        tsq.enqueue(v)
    log.info("Dequeue from TwoStackQueue: %d", tsq.dequeue())  # 10
    log.info("Dequeue: %d", tsq.dequeue())                      # 20

    log.info("")
    log.info("=== Next Greater Element ===")
    nums = [2, 1, 4, 3, 6, 5]
    log.info("Input:  %s", nums)
    log.info("Result: %s", next_greater_element(nums))  # [4, 4, 6, 6, -1, -1]

    log.info("")
    log.info("=== Sliding Window Max ===")
    nums2 = [1, 3, -1, -3, 5, 3, 6, 7]
    log.info("Input (k=3): %s", nums2)
    log.info("Max per window: %s", sliding_window_max(nums2, 3))  # [3,3,5,5,6,7]
```

**Why `TwoStackQueue` is amortized O(1):**

Total transfers across n enqueue + n dequeue operations:
- Each element enters `inbox` once: n pushes.
- Each element moves to `outbox` exactly once: n pops + n appends.
- Total work = 4n operations for 2n user operations → O(1) per operation.

---

## Exercises

1. **Valid parentheses:** Implement `is_valid_parens(s: str) -> bool` that returns `True` if every opening bracket `(`, `[`, `{` is closed in the correct order. Use a stack. Handle the string `"([{}])"` → `True` and `"([)]"` → `False`.

2. **Min-stack:** Design a `MinStack` class supporting `push`, `pop`, `peek`, and `get_min` all in O(1) time and O(n) space. `get_min` returns the current minimum value in the stack.

3. **Evaluate Reverse Polish Notation:** Given a list of tokens (e.g., `["2", "1", "+", "3", "*"]`), evaluate the expression using a stack and return the result as an integer. Operators are `+`, `-`, `*`, `/` (integer division truncating toward zero).

4. **BFS shortest path:** Given an unweighted undirected graph as `dict[str, list[str]]` (adjacency list), implement `bfs_shortest_path(graph, start, end) -> list[str] | None` that returns the shortest path between `start` and `end` using a queue. Return `None` if no path exists.

5. **Deque sliding window minimum:** Modify `sliding_window_max` to instead compute the **minimum** in each window of size k. Return the result as a list. What single change in the comparison operator achieves this?

*(Solutions: ../solutions/week-02/day-2.md)*

---

## Self-check

- [ ] I can explain why `list.pop(0)` is O(n) and why `deque.popleft()` is O(1).
- [ ] I can implement a FIFO queue using two stacks and explain the amortized O(1) cost.
- [ ] I can trace a monotonic stack on a 5-element input and explain the invariant it maintains.
- [ ] I can implement `is_valid_parens` and `MinStack` from memory.
- [ ] I know which `collections.deque` methods to use for stack operations vs. queue operations.

---

## Supplementary links

- [Python `collections.deque` docs](https://docs.python.org/3/library/collections.html#collections.deque)
- [Visualgo — Stack and Queue](https://visualgo.net/en/list)
- [LeetCode 155 — Min Stack](https://leetcode.com/problems/min-stack/)
- [LeetCode 239 — Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
