# Solutions: Week 08 · Day 2 — Binary Search Variants + Intervals + Fast/Slow Pointers

## Exercise 1: Find Minimum in Rotated Sorted Array

```python
def find_min(nums: list[int]) -> int:
    """Binary search.  Time: O(log n)  Space: O(1)

    Invariant: the minimum is always in [left, right].
    """
    left, right = 0, len(nums) - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            # min is in the right half (mid is NOT the min)
            left = mid + 1
        else:
            # min is at mid or to the left
            right = mid
    return nums[left]
```

**Invariant justification:** if `nums[mid] > nums[right]`, the right half is definitely rotated and the minimum must lie to the right of mid. Otherwise mid could be the minimum — we keep it in range by setting `right = mid` (not `mid - 1`). The loop ends when `left == right` — the single remaining candidate is the answer.

---

## Exercise 2: Merge Intervals

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Time: O(n log n)  Space: O(n)"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:              # overlaps or touches
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
```

**Fully overlapping case:** `[1,10]` + `[2,5]` — `5 <= 10`, so update to `max(10,5)=10`. Correct.

**Single element:** `[[5,7]]` — `intervals[0]` = `[5,7]`, loop body never executes, returns `[[5,7]]`. Correct.

---

## Exercise 3: Meeting Rooms II

```python
import heapq

def min_meeting_rooms(intervals: list[list[int]]) -> int:
    """Min conference rooms.  Time: O(n log n)  Space: O(n)"""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[0])
    # min-heap of end times for active meetings
    heap: list[int] = []
    for start, end in intervals:
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)   # reuse earliest-ending room
        else:
            heapq.heappush(heap, end)      # need a new room
    return len(heap)
```

**Insight:** sorting by start time, then greedily reusing the room that freed up earliest (min end time on heap) gives the minimum rooms. The heap size at the end equals the peak simultaneous meetings — that's the minimum rooms needed.

---

## Exercise 4: Find the Duplicate Number (Floyd's)

```python
def find_duplicate(nums: list[int]) -> int:
    """Array as implicit linked list.  Time: O(n)  Space: O(1)

    nums[i] is the 'next' pointer from node i.
    Duplicate value = cycle entry.
    """
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    return slow
```

**Why this models a linked list:** index i points to index `nums[i]`. Since one value is duplicated, two indices point to the same next node — creating a cycle. The cycle entry is the duplicate value.

---

## Exercise 5: Capacity to Ship Packages

```python
import math

def ship_within_days(weights: list[int], days: int) -> int:
    """Search on answer.  Time: O(n log sum(weights))  Space: O(1)"""
    def feasible(capacity: int) -> bool:
        trips = 1
        current = 0
        for w in weights:
            if current + w > capacity:
                trips += 1
                current = 0
            current += w
        return trips <= days

    lo, hi = max(weights), sum(weights)  # min=heaviest single, max=all in one day
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

**Bounds reasoning:** `lo = max(weights)` because we must fit every package on some day. `hi = sum(weights)` because one day is always feasible. The answer lies in `[lo, hi]`.

---

## Test file

```python
# tests/test_day2.py
import pytest
from day2_solutions import (
    find_min, merge_intervals, min_meeting_rooms, find_duplicate, ship_within_days
)


def test_find_min_rotated() -> None:
    assert find_min([3, 4, 5, 1, 2]) == 1
    assert find_min([4, 5, 6, 7, 0, 1, 2]) == 0
    assert find_min([1]) == 1


def test_merge_intervals() -> None:
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [
        [1, 6], [8, 10], [15, 18]
    ]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([]) == []


def test_min_meeting_rooms() -> None:
    assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
    assert min_meeting_rooms([[7, 10], [2, 4]]) == 1
    assert min_meeting_rooms([]) == 0


def test_find_duplicate() -> None:
    assert find_duplicate([1, 3, 4, 2, 2]) == 2
    assert find_duplicate([3, 1, 3, 4, 2]) == 3


def test_ship_within_days() -> None:
    assert ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
    assert ship_within_days([3, 2, 2, 4, 1, 4], 3) == 6
    assert ship_within_days([1, 2, 3, 1, 1], 4) == 3
```
