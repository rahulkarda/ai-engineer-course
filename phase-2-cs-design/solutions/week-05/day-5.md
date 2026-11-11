# Solutions — Week 05 Day 5 (Greedy & Divide and Conquer)

---

## Exercise 1 — Gas station

```python
def can_complete_circuit(gas: list[int], cost: list[int]) -> int:
    """
    Greedy: if total gas >= total cost, a solution exists.
    Start from 0; if tank goes negative, reset start to i+1.
    """
    if sum(gas) < sum(cost):
        return -1
    start = 0
    tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

**Proof**: if `sum(gas) >= sum(cost)`, a solution exists. The first index after the last negative prefix is always valid — any previous starting point would have failed earlier.

---

## Exercise 2 — Candy

```python
def candy(ratings: list[int]) -> int:
    n = len(ratings)
    candies = [1] * n
    # Left pass: give more to rising right neighbours
    for i in range(1, n):
        if ratings[i] > ratings[i-1]:
            candies[i] = candies[i-1] + 1
    # Right pass: give more to rising left neighbours (keep max)
    for i in range(n-2, -1, -1):
        if ratings[i] > ratings[i+1]:
            candies[i] = max(candies[i], candies[i+1] + 1)
    return sum(candies)

# candy([1,0,2]) → 5
# candy([1,2,2]) → 4
```

---

## Exercise 3 — Closest pair of points (D&C)

```python
import math
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

def dist(p: Point, q: Point) -> float:
    return math.hypot(p.x - q.x, p.y - q.y)

def brute_force(pts: list[Point]) -> float:
    min_d = float("inf")
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            min_d = min(min_d, dist(pts[i], pts[j]))
    return min_d

def closest_pair(pts: list[Point]) -> float:
    sorted_pts = sorted(pts, key=lambda p: p.x)

    def _closest(points: list[Point]) -> float:
        n = len(points)
        if n <= 3:
            return brute_force(points)
        mid = n // 2
        mid_x = points[mid].x
        d = min(_closest(points[:mid]), _closest(points[mid:]))
        # Strip: points within d of the midline
        strip = [p for p in points if abs(p.x - mid_x) < d]
        strip.sort(key=lambda p: p.y)
        for i in range(len(strip)):
            for j in range(i+1, len(strip)):
                if strip[j].y - strip[i].y >= d:
                    break
                d = min(d, dist(strip[i], strip[j]))
        return d

    return _closest(sorted_pts)
```

---

## Exercise 4 — Quickselect

```python
import random

def quickselect(nums: list[int], k: int) -> int:
    """Return k-th smallest (0-indexed). Average O(n), worst O(n²)."""
    def partition(lo: int, hi: int) -> int:
        pivot_idx = random.randint(lo, hi)
        nums[pivot_idx], nums[hi] = nums[hi], nums[pivot_idx]
        pivot = nums[hi]
        i = lo
        for j in range(lo, hi):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[hi] = nums[hi], nums[i]
        return i

    lo, hi = 0, len(nums) - 1
    while lo < hi:
        pivot = partition(lo, hi)
        if pivot == k:
            return nums[k]
        elif pivot < k:
            lo = pivot + 1
        else:
            hi = pivot - 1
    return nums[lo]
```

**O(n) average**: each recursion eliminates half the elements on average. Worst case O(n²) occurs when the pivot is always min/max — random pivot selection makes this extremely unlikely.

---

## Exercise 5 — Greedy coin change proof

**US coins {1, 5, 10, 25}**: greedy (always take largest) works because each denomination divides the next: 1|5, 5|25 (with 10 being the anomaly — but 2×5=10 ensures coverage). A formal proof uses exchange: if optimal uses fewer 25s than greedy, you can always swap smaller coins for a 25 with equal or fewer total coins.

**Failing on {1, 3, 4}, target=6**: greedy picks 4+1+1=3 coins. Optimal is 3+3=2 coins. Greedy fails because 4 is locally best but blocks the globally optimal (two 3s). The problem lacks the greedy-choice property: taking the largest coin doesn't always lead to the global optimum.
