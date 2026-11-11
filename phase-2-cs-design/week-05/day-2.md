# Phase 2 · Week 05 · Day 2 — Memoization (Top-Down DP)

> Objective: Transform exponential recursion into polynomial time using memoization.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

**Memoization** caches the result of a function call keyed on its arguments. When the same sub-problem is encountered again, return the cached value instead of recomputing. This is the *top-down* approach to dynamic programming: start from the original problem, recurse down, cache on the way back up.

The key insight: a recursive algorithm is exponential when **sub-problems overlap** (the same (n) is computed many times). Memoization ensures each unique sub-problem is solved exactly once — turning `O(2^n)` Fibonacci into `O(n)`.

Three requirements for safe memoization:
1. **Pure function** — same inputs always produce the same output (no hidden state, I/O, side effects).
2. **Hashable arguments** — keys must be hashable (ints, strings, tuples — not lists or dicts).
3. **Finite state space** — unbounded recursion + cache = memory explosion.

## How it works internally

Python's `functools.lru_cache` / `functools.cache` implements memoization via a dict under the hood. The decorator wraps the function: before each call it checks `cache[(args, kwargs)]`; on miss it calls the function and stores the result; on hit it returns immediately.

```
fib(5)
  ├── fib(4)
  │     ├── fib(3) → computed, cached
  │     └── fib(2) → computed, cached
  └── fib(3) → CACHE HIT, O(1)
```

Space complexity of a memoized solution = number of unique sub-problems × size of each result.

`lru_cache(maxsize=None)` = unbounded; `lru_cache(maxsize=128)` evicts least-recently-used entries when the cache fills.

## Worked example

```python
"""
day2_memoization.py  —  top-down DP with memoization
Python 3.12+
"""
from __future__ import annotations
import logging
import sys
from functools import cache, lru_cache

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Fibonacci: naïve → memoized ──────────────────────────────────────────
@cache                     # functools.cache = lru_cache(maxsize=None)
def fib(n: int) -> int:
    """O(n) time, O(n) space after memoization."""
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


# ── 2. Climbing stairs (ways to reach step n using 1 or 2 steps) ─────────────
@cache
def climb_stairs(n: int) -> int:
    """Count distinct ways to climb n stairs (1 or 2 steps at a time)."""
    if n <= 1:
        return 1
    return climb_stairs(n - 1) + climb_stairs(n - 2)


# ── 3. Manual memo dict (for clarity / when @cache can't be used) ─────────────
def fib_manual(n: int, memo: dict[int, int] | None = None) -> int:
    """Fibonacci with explicit memo dict — useful when cache must be injected."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_manual(n - 1, memo) + fib_manual(n - 2, memo)
    return memo[n]


# ── 4. 0/1 Knapsack — memoized top-down ──────────────────────────────────────
def knapsack_memo(
    weights: tuple[int, ...],
    values: tuple[int, ...],
    capacity: int,
) -> int:
    """
    0/1 knapsack: given items with weights and values, find the maximum
    value achievable within the given capacity.
    State: (item_index, remaining_capacity).
    """
    memo: dict[tuple[int, int], int] = {}

    def dp(i: int, cap: int) -> int:
        if i == len(weights) or cap == 0:
            return 0
        if (i, cap) in memo:
            return memo[(i, cap)]
        # Option A: skip item i
        skip = dp(i + 1, cap)
        # Option B: take item i (only if it fits)
        take = 0
        if weights[i] <= cap:
            take = values[i] + dp(i + 1, cap - weights[i])
        memo[(i, cap)] = max(skip, take)
        return memo[(i, cap)]

    return dp(0, capacity)


# ── 5. Longest Common Subsequence — memoized ─────────────────────────────────
def lcs_memo(s: str, t: str) -> int:
    """Return length of longest common subsequence of s and t."""
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> int:
        if i == len(s) or j == len(t):
            return 0
        if s[i] == t[j]:
            return 1 + dp(i + 1, j + 1)
        return max(dp(i + 1, j), dp(i, j + 1))

    return dp(0, 0)


# ── 6. Coin change — minimum coins (memoized) ────────────────────────────────
def coin_change_memo(coins: tuple[int, ...], amount: int) -> int:
    """Return minimum number of coins to make amount, or -1 if impossible."""
    @lru_cache(maxsize=None)
    def dp(remaining: int) -> int:
        if remaining == 0:
            return 0
        if remaining < 0:
            return sys.maxsize
        best = min(dp(remaining - c) for c in coins)
        return 1 + best if best < sys.maxsize else sys.maxsize

    result = dp(amount)
    return result if result < sys.maxsize else -1


# ── demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("fib(40) = %d", fib(40))
    logger.info("climb_stairs(10) = %d", climb_stairs(10))
    logger.info(
        "knapsack([2,3,4,5], [3,4,5,6], cap=5) = %d",
        knapsack_memo((2, 3, 4, 5), (3, 4, 5, 6), 5),
    )
    logger.info("lcs('abcde','ace') = %d", lcs_memo("abcde", "ace"))
    logger.info("coin_change([1,5,6,9], 11) = %d", coin_change_memo((1, 5, 6, 9), 11))
```

## Exercises

1. **Word break**: given a string `s` and a dictionary of words `word_dict`, determine if `s` can be segmented into dictionary words. Use memoization.

2. **Unique paths**: a robot is in an `m×n` grid at top-left and can only move right or down. Count distinct paths to bottom-right using memoized recursion.

3. **Decode ways**: a string of digits can map to letters (1→A, 2→B, …, 26→Z). Count number of ways to decode a given digit string using memoization.

4. **Target sum**: given a list of integers and a target, assign each a `+` or `-` and count ways to reach the target. Memoize on `(index, current_sum)`.

5. **Cache analysis**: for `lcs_memo("AGGTAB", "GXTXAYB")`, how many unique `(i, j)` sub-problems exist? What is the time and space complexity?

(Solutions: ../solutions/week-05/day-2.md)

## Self-check

- [ ] I can explain why naïve Fibonacci is O(2^n) and memoized is O(n).
- [ ] I can apply `@functools.cache` to any pure recursive function.
- [ ] I know when to use a manual `memo` dict vs `@cache`.
- [ ] I can identify the DP state (function parameters) for knapsack and LCS.
- [ ] I can implement coin change with memoization without reference.

## Supplementary links

- `functools.lru_cache` docs: https://docs.python.org/3/library/functools.html#functools.lru_cache
- DP intro (Tushar Roy, YouTube): search "Dynamic Programming — Learn to Solve Algorithmic Problems"
- MIT 6.006 Lecture 19 — Dynamic Programming: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/
