# Phase 2 · Week 05 · Day 6 (Saturday) — DP Problem Set (4 hr deep practice)

> Objective: Solve a curated set of DP problems end-to-end, comparing both memoized and tabulated approaches with timing and complexity notes.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Structure

Work through all six problems in order. For each:
1. Read the problem and constraints carefully.
2. Identify the DP family (knapsack / string / sequence / interval / other).
3. Write the memoized solution first (top-down).
4. Write the tabulated solution second (bottom-up).
5. Note: time complexity, space complexity, and which approach you prefer — with reasoning.

Use the test harness at the bottom to verify both solutions produce identical output.

---

## Problem 1 — Interleaving Strings

**Given** strings `s1`, `s2`, `s3`, determine if `s3` can be formed by interleaving `s1` and `s2` while preserving their relative character orders.

**Constraints**: `0 ≤ len(s1), len(s2) ≤ 100`, `len(s3) == len(s1) + len(s2)`.

**Exercises**:
1. Identify the 2-D DP state.
2. Write both memoized and tabulated versions.
3. What is the time and space complexity?

---

## Problem 2 — Burst Balloons

**Given** `n` balloons with values in `nums`. Bursting balloon `i` earns `nums[i-1] * nums[i] * nums[i+1]` coins (treat boundaries as 1). Find the maximum coins collectible.

**Constraints**: `1 ≤ n ≤ 300`, `0 ≤ nums[i] ≤ 100`.

**Exercises**:
1. Why does the naive approach fail? What DP reformulation fixes it? (Hint: think about the *last* balloon burst in a range.)
2. Implement the interval DP solution. What is its complexity?

---

## Problem 3 — Regular Expression Matching

**Given** string `s` and pattern `p` with `.` (any char) and `*` (zero or more of preceding), implement `is_match(s, p) -> bool`.

**Constraints**: `1 ≤ len(s) ≤ 20`, `1 ≤ len(p) ≤ 30`, `p` won't start with `*`.

**Exercises**:
1. Write the recurrence for the `*` case carefully — it's the tricky one.
2. Implement memoized and tabulated.

---

## Problem 4 — Maximum Rectangle in Histogram

**Given** an array of bar heights, find the largest rectangular area under the histogram.

**Constraints**: `1 ≤ n ≤ 10^5`.

**Exercises**:
1. This is a stack-based O(n) problem. Implement the stack solution.
2. Using this as a subroutine, extend to "Maximal Rectangle" in a binary matrix — a DP + histogram combination.
3. Compare with a naive O(n²) scan. When does the stack solution dominate?

---

## Problem 5 — Egg Drop Problem

**Given** `k` eggs and `n` floors, find the minimum number of trials (worst case) needed to determine the critical floor from which eggs start breaking.

**Constraints**: `1 ≤ k ≤ 100`, `1 ≤ n ≤ 10000`.

**Exercises**:
1. Implement the O(k * n²) DP (`dp[k][n]` = min trials).
2. Optimize to O(k * n log n) by binary searching over the optimal floor to try at each state.
3. What is the recurrence? Why does it have optimal substructure?

---

## Problem 6 — Longest Valid Parentheses

**Given** a string of `(` and `)`, find the length of the longest valid parentheses substring.

**Constraints**: `0 ≤ len(s) ≤ 3 × 10^4`.

**Exercises**:
1. Solve with DP: `dp[i]` = length of longest valid substring ending at index `i`.
2. Solve with a stack (no DP). Compare constant factors.
3. Can you solve in O(1) space with two passes (left→right and right→left counters)?

---

## Test harness

```python
"""
day6_dp_problemset.py  —  Saturday problem set harness
Python 3.12+
Each solution should be implemented in a separate function with a _memo and _tab variant.
"""
from __future__ import annotations
import logging
import time
from typing import Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_both(
    label: str,
    memo_fn: Callable,
    tab_fn: Callable,
    *args,
    **kwargs,
) -> None:
    """Run memo and tab versions, assert they match, log timings."""
    t0 = time.perf_counter()
    memo_result = memo_fn(*args, **kwargs)
    memo_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    tab_result = tab_fn(*args, **kwargs)
    tab_time = time.perf_counter() - t0

    match = memo_result == tab_result
    logger.info(
        "[%s] memo=%s (%.4fms)  tab=%s (%.4fms)  match=%s",
        label, memo_result, memo_time * 1000,
        tab_result, tab_time * 1000, match,
    )
    assert match, f"Solutions diverge: memo={memo_result} tab={tab_result}"


# ── Implement your solutions below this line ─────────────────────────────────

# Problem 1: Interleaving strings
def is_interleave_memo(s1: str, s2: str, s3: str) -> bool:
    raise NotImplementedError

def is_interleave_tab(s1: str, s2: str, s3: str) -> bool:
    raise NotImplementedError


# Problem 2: Burst balloons
def max_coins_memo(nums: list[int]) -> int:
    raise NotImplementedError

def max_coins_tab(nums: list[int]) -> int:
    raise NotImplementedError


# Problem 3: Regex matching
def is_match_memo(s: str, p: str) -> bool:
    raise NotImplementedError

def is_match_tab(s: str, p: str) -> bool:
    raise NotImplementedError


# Problem 4: Max rectangle in histogram (stack-based, single version acceptable)
def largest_rectangle_histogram(heights: list[int]) -> int:
    raise NotImplementedError

def maximal_rectangle(matrix: list[list[str]]) -> int:
    raise NotImplementedError


# Problem 5: Egg drop
def egg_drop_memo(k: int, n: int) -> int:
    raise NotImplementedError

def egg_drop_tab(k: int, n: int) -> int:
    raise NotImplementedError


# Problem 6: Longest valid parentheses
def longest_valid_parens_dp(s: str) -> int:
    raise NotImplementedError

def longest_valid_parens_stack(s: str) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    # Problem 1
    run_both("interleave", is_interleave_memo, is_interleave_tab, "aab", "axy", "aaxaby")
    run_both("interleave", is_interleave_memo, is_interleave_tab, "ab", "xy", "abyx")

    # Problem 2
    run_both("burst_balloons", max_coins_memo, max_coins_tab, [3, 1, 5, 8])

    # Problem 3
    run_both("regex", is_match_memo, is_match_tab, "aa", "a*")
    run_both("regex", is_match_memo, is_match_tab, "mississippi", "mis*is*p*.")

    # Problem 5
    run_both("egg_drop", egg_drop_memo, egg_drop_tab, 2, 10)
    run_both("egg_drop", egg_drop_memo, egg_drop_tab, 3, 14)

    # Problem 6
    run_both("valid_parens", longest_valid_parens_dp, longest_valid_parens_stack, ")()())")
    run_both("valid_parens", longest_valid_parens_dp, longest_valid_parens_stack, "(()")
```

(Solutions: ../solutions/week-05/day-6.md)

## Self-check

- [ ] For each problem, I can state the DP state, recurrence, base cases, and complexity before coding.
- [ ] My memoized and tabulated solutions produce identical results on all test cases.
- [ ] I can articulate one clear trade-off between the two approaches for each problem.
- [ ] I can implement the histogram stack solution from scratch.
- [ ] I can write the egg drop DP and explain its optimal substructure.

## Supplementary links

- Interleaving strings — discussion: https://leetcode.com/problems/interleaving-string/
- Burst balloons — key insight explanation: https://leetcode.com/problems/burst-balloons/solutions/
- Egg drop — full derivation: https://brilliant.org/wiki/egg-dropping/
