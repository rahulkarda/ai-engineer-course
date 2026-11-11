# Phase 2 · Week 01 · Day 1 — Big-O, Big-Theta, Big-Omega

> Objective: Read, write, and compare asymptotic complexity expressions for any algorithm.  ·  Time: ~1 hr  ·  Prereqs: [[phase-1-python-backend/week-01/day-1]]

---

## Concept

**Asymptotic analysis** measures how an algorithm's resource usage (time or space) *scales* with input size `n`, ignoring constant factors and low-order terms.  
We care about scaling because constants are hardware-dependent; the shape of the curve is universal.

### The three notations

| Notation | Reads as | Meaning |
|---|---|---|
| O(f(n)) | "big-oh" | **Upper bound** — worst case or ceiling |
| Ω(f(n)) | "big-omega" | **Lower bound** — best case or floor |
| Θ(f(n)) | "big-theta" | **Tight bound** — upper and lower match |

Formal definitions:

```
f(n) ∈ O(g(n))  iff  ∃ c > 0, n₀ ≥ 1 : f(n) ≤ c·g(n)  for all n ≥ n₀
f(n) ∈ Ω(g(n))  iff  ∃ c > 0, n₀ ≥ 1 : f(n) ≥ c·g(n)  for all n ≥ n₀
f(n) ∈ Θ(g(n))  iff  f ∈ O(g) AND f ∈ Ω(g)
```

In plain words: O gives an *at most* guarantee; Ω gives an *at least* guarantee; Θ means both bounds are tight — the function grows exactly like g(n).

### Common complexity classes (slowest → fastest growing)

```
O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
```

### Rules for deriving O from code

1. **Drop constants:** `3n` → `O(n)`
2. **Drop low-order terms:** `n² + n` → `O(n²)`
3. **Loops multiply:** nested loops each iterating `n` times → `O(n²)`
4. **Sequential blocks add:** `O(n) + O(n²)` → `O(n²)` (keep dominant)
5. **Recursion → recurrence:** solve T(n) = aT(n/b) + f(n) via Master Theorem

---

## How it works internally

Asymptotic notation is a *classification tool*, not an exact count.  
It partitions algorithms into equivalence classes:  
- `O(n²)` includes 2n², 5n² + 100n, 0.001n² — they all *bend the same way* on a log-log plot.  
- Constants matter for performance engineering but not algorithm comparison.

When a CS paper says "O(n log n) sort" it means: for any constants c₁, c₂, there exists an input size beyond which the algorithm always finishes within c₁·n·log(n) steps.

The **Master Theorem** handles divide-and-conquer recurrences:  
`T(n) = a·T(n/b) + n^c`

- If `log_b(a) > c` → `T(n) = Θ(n^(log_b(a)))`
- If `log_b(a) = c` → `T(n) = Θ(n^c · log n)`
- If `log_b(a) < c` → `T(n) = Θ(n^c)`

Example — merge sort: `T(n) = 2T(n/2) + n`  
a=2, b=2, c=1. log₂(2)=1 = c → case 2 → `Θ(n log n)`. ✓

---

## Worked example

```python
"""
complexity_examples.py — annotated Big-O derivations
Python 3.12+
"""
import logging
from typing import Sequence

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# O(1) — constant time, no loop, no recursion
def first_element(items: Sequence[int]) -> int | None:
    """Return first element. Complexity: O(1) time, O(1) space."""
    return items[0] if items else None


# O(n) — single pass
def linear_search(items: Sequence[int], target: int) -> int:
    """Return index of target or -1. Complexity: O(n) time, O(1) space."""
    for i, val in enumerate(items):           # n iterations worst case
        if val == target:
            return i
    return -1


# O(n²) — nested loops, each 0..n
def has_duplicate_naive(items: Sequence[int]) -> bool:
    """O(n²) time — compare every pair. O(1) space."""
    n = len(items)
    for i in range(n):                         # n iterations
        for j in range(i + 1, n):             # up to n-1 iterations
            if items[i] == items[j]:
                return True
    return False


# O(log n) — halving search space each iteration
def binary_search(sorted_items: Sequence[int], target: int) -> int:
    """O(log n) time, O(1) space — requires sorted input."""
    lo, hi = 0, len(sorted_items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_items[mid] == target:
            return mid
        elif sorted_items[mid] < target:
            lo = mid + 1                       # discard left half
        else:
            hi = mid - 1                       # discard right half
    return -1


# O(n log n) — divide-and-conquer merge sort
def merge_sort(items: list[int]) -> list[int]:
    """O(n log n) time, O(n) space — classic merge sort."""
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = merge_sort(items[:mid])             # T(n/2)
    right = merge_sort(items[mid:])            # T(n/2)
    return _merge(left, right)                 # O(n) merge step


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


if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    log.info("first_element: %s   O(1)", first_element(data))
    log.info("linear_search 9: %d   O(n)", linear_search(data, 9))
    log.info("has_duplicate: %s   O(n²)", has_duplicate_naive(data))
    sorted_data = sorted(data)
    log.info("binary_search 9: %d   O(log n)", binary_search(sorted_data, 9))
    log.info("merge_sort: %s   O(n log n)", merge_sort(data))
```

**Trace the complexity derivation for `has_duplicate_naive`:**

- Outer loop runs `n` times.
- Inner loop runs at most `n-1`, `n-2`, … times → total = n(n-1)/2 comparisons.
- n(n-1)/2 = (n² − n)/2 → drop constant (1/2) and low-order (−n) → **O(n²)**.

---

## Exercises

1. Classify each snippet — give Big-O for time AND space, then justify:

   ```python
   # (a)
   def sum_matrix(mat: list[list[int]]) -> int:
       total = 0
       for row in mat:
           for val in row:
               total += val
       return total
   # mat is n×n
   ```

   ```python
   # (b)
   def count_bits(n: int) -> int:
       count = 0
       while n:
           count += n & 1
           n >>= 1
       return count
   ```

   ```python
   # (c)
   def fibonacci(n: int) -> int:
       if n <= 1:
           return n
       return fibonacci(n - 1) + fibonacci(n - 2)
   ```

2. A function has recurrence `T(n) = 4T(n/2) + n`. Use the Master Theorem to find Θ.

3. Rank these functions from smallest to largest asymptotic growth:  
   `n!`, `2ⁿ`, `n³`, `n log n`, `log n`, `n`, `1`, `n²`

4. Rewrite `has_duplicate_naive` to run in O(n) time (not O(n²)). What space trade-off do you make?

5. True or False — justify each:
   - a) `O(n²)` algorithms are always slower than `O(n log n)` algorithms.
   - b) If `f(n) ∈ Θ(g(n))` then `g(n) ∈ Θ(f(n))`.
   - c) `f(n) = 5` is in `O(1)`.

*(Solutions: ../solutions/week-01/day-1.md)*

---

## Self-check

- [ ] I can define O, Ω, Θ formally and give a one-sentence intuition for each.
- [ ] I can derive Big-O from any loop-based Python snippet in under 2 minutes.
- [ ] I can apply the Master Theorem to a two-branch divide-and-conquer recurrence.
- [ ] I can explain why constants are dropped and give one real case where the constant matters anyway.

---

## Supplementary links

- [MIT 6.006 Lecture 1 — Introduction to Algorithms](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/pages/lecture-videos/)
- [Big-O Cheat Sheet](https://www.bigocheatsheet.com/)
- [Visualgo — sorting algorithm visualizer](https://visualgo.net/en/sorting)
