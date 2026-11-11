# Phase 7 · Week 08 · Day 2 — Binary Search Variants + Intervals + Fast/Slow Pointers

> Objective: Apply binary search beyond sorted arrays; merge/sweep intervals; detect cycles with Floyd's algorithm.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Binary Search Variants

Classic binary search finds an exact value. In interviews, most problems use a *template variant*:

| Variant | Use case | Condition |
|---|---|---|
| Find exact | sorted array, find target | `nums[mid] == target` |
| Find left boundary | first position where condition holds | `condition(mid)` shrinks right |
| Find right boundary | last position where condition holds | `condition(mid)` moves left pointer |
| Search on answer | "minimum X such that feasible(X)" | binary search the answer space |

**Left-boundary template (find first `True` in `[F,F,T,T,T]`):**

```python
def binary_search_left(nums: list[int], target: int) -> int:
    left, right = 0, len(nums)   # right is exclusive
    while left < right:
        mid = left + (right - left) // 2   # avoids overflow
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid          # mid could be answer; shrink from right
    return left                  # insertion point
```

**Search-on-answer pattern:**
```python
# "Can we do the task with capacity = cap?"
lo, hi = min_feasible, max_feasible
while lo < hi:
    mid = (lo + hi) // 2
    if feasible(mid):
        hi = mid        # mid works; try smaller
    else:
        lo = mid + 1    # mid too small
return lo
```

---

### Intervals

**Sort by start time** → O(n log n); then one O(n) sweep handles merge/insert/overlap.

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    intervals.sort(key=lambda x: x[0])
    merged: list[list[int]] = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:          # overlaps
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

For meeting-room problems (do k meetings fit?), use a **min-heap** of end times; O(n log n).

---

### Fast / Slow Pointers (Floyd's Algorithm)

**When to use:** detect cycle in linked list; find cycle entry; find middle node; find duplicate in value-range array.

```
slow moves 1 step; fast moves 2 steps.
If cycle exists: they meet inside the cycle.
```

**Cycle entry formula:** after meeting inside the cycle, reset one pointer to head; advance both one step at a time — they meet at the cycle entry.

---

## How it works internally

**Binary search correctness:** the invariant is `answer ∈ [left, right)`. Each iteration halves the range. `mid = left + (right - left) // 2` prevents integer overflow (critical in C++; Python has big integers but good habit).

**Why sort intervals?** Sorting turns an O(n²) "compare every pair" problem into a single left-to-right scan because after sorting, only adjacent intervals can overlap.

**Floyd's proof sketch:** let λ = cycle length, μ = tail length. After μ+λ steps, slow is at cycle position μ mod λ, fast is at 2(μ+λ) mod λ = μ mod λ — they meet. Resetting slow to head and advancing both by 1 step at a time brings them together after exactly μ more steps (the tail length) — that's the cycle entry.

---

## Worked example

### Problem 1: Search in Rotated Sorted Array

```python
# binary_search_variants.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def search_rotated(nums: list[int], target: int) -> int:
    """Binary search on rotated sorted array.  Time: O(log n)"""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        # Determine which half is sorted
        if nums[left] <= nums[mid]:          # left half sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:                                # right half sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

### Problem 2: Koko Eating Bananas (search on answer)

```python
import math

def min_eating_speed(piles: list[int], h: int) -> int:
    """Minimum bananas/hr so Koko finishes in h hours.  Time: O(n log max(piles))"""
    def feasible(speed: int) -> bool:
        return sum(math.ceil(p / speed) for p in piles) <= h

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### Problem 3: Linked-List Cycle II (entry point)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ListNode:
    val: int
    next: Optional["ListNode"] = None


def detect_cycle(head: Optional[ListNode]) -> Optional[ListNode]:
    """Return cycle entry node or None.  Time: O(n)  Space: O(1)"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next           # type: ignore[assignment]
        fast = fast.next.next      # type: ignore[assignment]
        if slow is fast:
            slow = head            # reset to head
            while slow is not fast:
                slow = slow.next   # type: ignore[assignment]
                fast = fast.next   # type: ignore[assignment]
            return slow
    return None
```

---

## Exercises

1. **Find Minimum in Rotated Sorted Array** — binary search; O(log n); no duplicates. State the invariant that guarantees correctness.

2. **Merge Intervals** — given a list of `[start, end]` intervals, return the merged non-overlapping list. Handle single-element input and fully overlapping intervals.

3. **Meeting Rooms II** — given intervals representing meeting times, find the minimum number of conference rooms required. Hint: sort by start; use a min-heap of end times.

4. **Find the Duplicate Number** — array of n+1 integers in range [1, n]. Find the duplicate without modifying the array and using O(1) extra space. Model as linked-list cycle detection.

5. **Capacity to Ship Packages** — given weights array and `days` D, find minimum ship capacity so all packages ship in D days. Use search-on-answer; write the `feasible` predicate first.

(Solutions: ../solutions/week-08/day-2.md)

---

## Self-check

- [ ] I can implement left-boundary binary search from memory
- [ ] I can set up a search-on-answer problem (lo/hi bounds + feasible function)
- [ ] I can merge overlapping intervals in O(n log n)
- [ ] I can detect a cycle and find its entry with O(1) space
- [ ] I can explain Floyd's meeting-point proof in one paragraph

---

## Supplementary links

- NeetCode Binary Search roadmap: https://neetcode.io/roadmap
- NeetCode Linked List (slow/fast pointers): https://neetcode.io/roadmap
