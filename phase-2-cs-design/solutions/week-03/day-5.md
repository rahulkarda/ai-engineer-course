# Solutions — Week 03 · Day 5

## Exercise 1 — K-th smallest

### Approach (a): push all, pop k times — O(n + k log n), space O(n)

```python
import heapq

def kth_smallest_a(nums: list[int], k: int) -> int:
    h = nums[:]
    heapq.heapify(h)
    for _ in range(k - 1):
        heapq.heappop(h)
    return heapq.heappop(h)
```

### Approach (b): max-heap of size k — O(n log k), space O(k)

```python
def kth_smallest_b(nums: list[int], k: int) -> int:
    """Maintain a max-heap of the k smallest seen so far (negate for max)."""
    heap: list[int] = []
    for val in nums:
        heapq.heappush(heap, -val)   # push negated (max-heap)
        if len(heap) > k:
            heapq.heappop(heap)      # remove the largest
    return -heap[0]                  # top of max-heap is k-th smallest
```

**Comparison**: approach (b) is better when `k << n` or the input is a stream (we never need to store all `n` elements). Approach (a) requires loading all elements first.

## Exercise 2 — Merge k sorted lists

```python
import heapq
from typing import Optional

def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    """
    Use a min-heap of (value, list_index, element_index).
    Each pop gives the next smallest element globally.
    """
    result: list[int] = []
    heap: list[tuple[int, int, int]] = []

    # Seed heap with the first element of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_i, elem_i = heapq.heappop(heap)
        result.append(val)
        next_i = elem_i + 1
        if next_i < len(lists[list_i]):
            heapq.heappush(heap, (lists[list_i][next_i], list_i, next_i))

    return result
```

**Complexity**: let `N` = total elements, `k` = number of lists.
- Time: O(N log k) — each of N elements is pushed/popped once; heap size ≤ k.
- Space: O(k) for the heap.

## Exercise 3 — Running median

```python
import heapq

class RunningMedian:
    """
    Two-heap approach:
    - lo_max: max-heap (negated) of the lower half
    - hi_min: min-heap of the upper half
    Invariant: len(lo) == len(hi) or len(lo) == len(hi) + 1
    """

    def __init__(self) -> None:
        self._lo_max: list[int] = []  # max-heap via negation
        self._hi_min: list[int] = []  # min-heap

    def add(self, val: int) -> None:
        # Push to lo, then balance
        heapq.heappush(self._lo_max, -val)
        # Ensure all lo values ≤ all hi values
        if self._hi_min and (-self._lo_max[0]) > self._hi_min[0]:
            moved = -heapq.heappop(self._lo_max)
            heapq.heappush(self._hi_min, moved)
        # Rebalance sizes
        if len(self._lo_max) > len(self._hi_min) + 1:
            moved = -heapq.heappop(self._lo_max)
            heapq.heappush(self._hi_min, moved)
        elif len(self._hi_min) > len(self._lo_max):
            moved = heapq.heappop(self._hi_min)
            heapq.heappush(self._lo_max, -moved)

    def get_median(self) -> float:
        if not self._lo_max:
            raise ValueError("No elements added")
        if len(self._lo_max) == len(self._hi_min):
            return (-self._lo_max[0] + self._hi_min[0]) / 2.0
        return float(-self._lo_max[0])
```

`add` is O(log n); `get_median` is O(1).

## Exercise 4 — Heap sort

```python
from src.heap import MinHeap  # or inline the class

def heap_sort(arr: list[int]) -> list[int]:
    h = MinHeap.heapify(arr)
    return [h.pop() for _ in range(len(h))]
```

**Complexity**:
- Time: O(n) for heapify + O(n log n) for n pops = O(n log n).
- Space: O(n) — the heap makes a copy of the input.

*In-place heap sort* achieves O(1) extra space using a max-heap built in the array itself, but `MinHeap.heapify` copies the input, so this version is not in-place.

## Exercise 5 — Experimental heapify is O(n)

```python
import heapq
import random
import time
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sizes = [1_000, 10_000, 100_000, 1_000_000]
times: list[float] = []

for n in sizes:
    data = [random.randint(0, 10**9) for _ in range(n)]
    t0 = time.perf_counter()
    heapq.heapify(data)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    log.info("n=%8d  time=%.4fs", n, elapsed)

# Print ratios — O(n) predicts ratio ≈ 10x for each 10x size increase
for i in range(1, len(sizes)):
    log.info("ratio sizes[%d]/sizes[%d] = %.2f  time ratio = %.2f",
             i, i-1, sizes[i]/sizes[i-1], times[i]/times[i-1])
```

**Expected**: time ratios ≈ 10x for each 10x size increase (linear). If it were O(n log n), ratios would be `10 * (log(10n) / log(n))` ≈ 11.x–12.x for large n — slightly super-linear.
