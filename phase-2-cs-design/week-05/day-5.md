# Phase 2 · Week 05 · Day 5 — Greedy Algorithms & Divide and Conquer

> Objective: Know when greedy works and why, and apply divide-and-conquer correctly.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Greedy algorithms

A greedy algorithm makes the **locally optimal choice at each step** without reconsidering past decisions. It is correct only when the problem has:

1. **Greedy-choice property**: a globally optimal solution can always be built from a locally optimal choice.
2. **Optimal substructure**: an optimal solution to the problem contains optimal solutions to its sub-problems (shared with DP).

Greedy is *not* a universal strategy — it fails on 0/1 knapsack (you must sometimes skip a high-value item), coin change with non-standard denominations, and most scheduling problems without careful proof.

**Proof technique**: exchange argument — assume an optimal solution that differs from greedy; show you can swap greedy's choice in without making it worse.

### Divide and conquer

Split the problem into **independent** sub-problems, solve each recursively, then **combine** the results. The key word is *independent* — if sub-problems share state or overlap significantly, DP is better.

Master theorem gives the recurrence analysis: `T(n) = aT(n/b) + f(n)`.

| Recurrence | Example | Complexity |
|---|---|---|
| T(n) = 2T(n/2) + O(n) | merge sort | O(n log n) |
| T(n) = T(n/2) + O(1) | binary search | O(log n) |
| T(n) = 2T(n/2) + O(n²) | naive matrix multiply | O(n²) |

## How it works internally

**Interval scheduling**: sort by finish time, always pick the job that ends earliest. The exchange argument: if there is a solution that picks a job finishing later than ours, swap — the new solution is at least as good (doesn't invalidate any subsequent picks).

**Merge sort**: the combine step (merging two sorted halves) is O(n). The split step is O(1). Recurrence: T(n) = 2T(n/2) + O(n) → O(n log n) by master theorem case 2.

## Worked example

```python
"""
day5_greedy_dc.py  —  greedy algorithms & divide-and-conquer
Python 3.12+
"""
from __future__ import annotations
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── GREEDY ────────────────────────────────────────────────────────────────────

# 1. Interval scheduling: maximize number of non-overlapping intervals
def max_non_overlapping(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Sort by end time, greedily pick each interval that starts
    after the last selected one ends.
    """
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[1])
    selected = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        if start >= selected[-1][1]:
            selected.append((start, end))
    return selected


# 2. Jump game: can you reach the last index?
def can_jump(nums: list[int]) -> bool:
    """
    Greedy: track the furthest index reachable.
    At each position, update max_reach. If pos > max_reach, stuck.
    """
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
    return True


# 3. Minimum number of platforms (arrival/departure scheduling)
def min_platforms(arrivals: list[int], departures: list[int]) -> int:
    """
    Sort both. Use two pointers. Classic greedy sweep-line.
    """
    arrivals = sorted(arrivals)
    departures = sorted(departures)
    platforms = 0
    max_platforms = 0
    i = j = 0
    while i < len(arrivals):
        if arrivals[i] <= departures[j]:
            platforms += 1
            i += 1
        else:
            platforms -= 1
            j += 1
        max_platforms = max(max_platforms, platforms)
    return max_platforms


# 4. Huffman-style: task scheduler (LeetCode 621)
def least_interval(tasks: list[str], n: int) -> int:
    """
    Greedy: always schedule the most frequent remaining task.
    Formula: max(len(tasks), (max_count-1)*(n+1) + count_of_max_count_tasks).
    """
    counts = Counter(tasks)
    max_count = max(counts.values())
    num_max = sum(1 for v in counts.values() if v == max_count)
    return max(len(tasks), (max_count - 1) * (n + 1) + num_max)


# 5. Fractional knapsack — greedy WORKS here (unlike 0/1)
def fractional_knapsack(
    weights: list[float],
    values: list[float],
    capacity: float,
) -> float:
    """Sort by value/weight ratio descending; take fractions as needed."""
    items = sorted(
        zip(weights, values),
        key=lambda x: x[1] / x[0],
        reverse=True,
    )
    total = 0.0
    for w, v in items:
        if capacity <= 0:
            break
        take = min(w, capacity)
        total += take * (v / w)
        capacity -= take
    return total


# ── DIVIDE AND CONQUER ────────────────────────────────────────────────────────

# 6. Merge sort (canonical D&C)
def merge_sort(arr: list[int]) -> list[int]:
    """O(n log n). Illustrates the D&C pattern: split, recurse, combine."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: list[int], right: list[int]) -> list[int]:
    result: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# 7. Count inversions (merge sort variant — D&C + counting)
def count_inversions(arr: list[int]) -> tuple[list[int], int]:
    """
    Return (sorted_arr, num_inversions).
    Inversions counted during merge: when right element is taken before left,
    all remaining left elements form inversions.
    """
    if len(arr) <= 1:
        return arr, 0
    mid = len(arr) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])
    merged, split_inv = _merge_count(left, right)
    return merged, left_inv + right_inv + split_inv


def _merge_count(left: list[int], right: list[int]) -> tuple[list[int], int]:
    result: list[int] = []
    inversions = 0
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
            inversions += len(left) - i   # all remaining left elements > right[j]
    result.extend(left[i:])
    result.extend(right[j:])
    return result, inversions


# ── demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ivs = [(1, 3), (2, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)]
    logger.info("max_non_overlapping: %s", max_non_overlapping(ivs))

    logger.info("can_jump([2,3,1,1,4]) = %s", can_jump([2, 3, 1, 1, 4]))
    logger.info("can_jump([3,2,1,0,4]) = %s", can_jump([3, 2, 1, 0, 4]))

    logger.info(
        "min_platforms([900,940,950,1100,1500,1800],[910,1200,1120,1130,1900,2000]) = %d",
        min_platforms([900,940,950,1100,1500,1800],[910,1200,1120,1130,1900,2000]),
    )
    logger.info("least_interval(AABABC, n=2) = %d", least_interval(list("AABABC"), 2))
    logger.info(
        "fractional_knapsack([10,20,30],[60,100,120],50) = %.1f",
        fractional_knapsack([10.0, 20.0, 30.0], [60.0, 100.0, 120.0], 50.0),
    )
    logger.info("merge_sort([5,2,4,6,1,3]) = %s", merge_sort([5, 2, 4, 6, 1, 3]))
    _, inv = count_inversions([2, 4, 1, 3, 5])
    logger.info("inversions in [2,4,1,3,5] = %d", inv)
```

## Exercises

1. **Gas station** (LeetCode 134): given arrays `gas` and `cost`, find the starting station from which you can complete a circular tour. Prove the greedy solution is optimal.

2. **Candy**: `n` children in a row, each with a rating. Distribute minimum candies such that each child gets at least 1 and children with higher ratings than their neighbours get more. Two-pass greedy.

3. **Closest pair of points**: implement the O(n log n) divide-and-conquer closest-pair algorithm (split by median x, recurse, strip search). Compare against the O(n²) brute force on a random dataset.

4. **Quickselect**: implement O(n) average-case selection (k-th smallest) using the partition step from quicksort. Explain why this is O(n) on average but O(n²) worst-case.

5. **Proof exercise**: explain why the greedy coin-change algorithm (always take the largest coin ≤ remaining) works for US coin denominations {1,5,10,25} but fails for {1,3,4}.

(Solutions: ../solutions/week-05/day-5.md)

## Self-check

- [ ] I can name two conditions a problem must satisfy for greedy to work.
- [ ] I can apply the exchange argument to prove interval scheduling greedy is correct.
- [ ] I can distinguish when to use greedy vs DP (0/1 vs fractional knapsack).
- [ ] I can implement merge sort and count inversions in the merge step.
- [ ] I can apply the master theorem to common recurrences.

## Supplementary links

- Master theorem explained: https://en.wikipedia.org/wiki/Master_theorem_(analysis_of_algorithms)
- Greedy algorithms (CLRS ch 16 summary): https://web.mit.edu/6.006/www/notes/greedy.pdf
- Closest pair of points: https://cp-algorithms.com/geometry/nearest_points.html
