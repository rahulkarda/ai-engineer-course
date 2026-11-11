# Phase 2 · Week 05 · Day 3 — Tabulation (Bottom-Up DP)

> Objective: Implement DP iteratively using tables, and understand when tabulation beats memoization.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

**Tabulation** (bottom-up DP) flips the top-down approach: instead of starting at the answer and recursing down, you **build up from base cases** to the answer iteratively, filling a table.

The algorithm:
1. Identify the DP state (same as memoization — what are the parameters?).
2. Allocate a table (1-D array, 2-D matrix, etc.) sized to cover all states.
3. Fill base cases explicitly.
4. Iterate in an order such that each state's dependencies are already computed.
5. Read the answer from the table.

**Memoization vs Tabulation:**

| | Memoization (top-down) | Tabulation (bottom-up) |
|---|---|---|
| Code style | Recursive + cache | Iterative + table |
| Stack overhead | Yes (risk of stack overflow) | None |
| Only needed states | Yes (lazy) | No (fills all states) |
| Space optimization | Hard | Easy (can reduce to O(1) or O(n)) |
| Cache warm-up | Automatic | Manual loop |

For large `n`, tabulation often wins on constant factors because there's no recursion overhead and the memory access pattern is cache-friendly.

## How it works internally

A 2-D DP table for LCS(`s`, `t`) stores `dp[i][j]` = LCS length of `s[:i]` and `t[:j]`. We fill it row by row, left to right. Each cell depends only on cells above, to the left, or diagonally above-left — all already filled.

**Space optimization**: if only the previous row is needed, we can keep two rows (or even one, traversed in the right order), reducing space from O(m×n) to O(min(m,n)).

## Worked example

```python
"""
day3_tabulation.py  —  bottom-up DP (tabulation)
Python 3.12+
"""
from __future__ import annotations
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Fibonacci — tabulated ──────────────────────────────────────────────────
def fib_tab(n: int) -> int:
    """O(n) time, O(1) space — only last two values needed."""
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


# ── 2. Climbing stairs — tabulated ───────────────────────────────────────────
def climb_stairs_tab(n: int) -> int:
    """Number of ways to climb n stairs (1 or 2 steps). O(1) space."""
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# ── 3. 0/1 Knapsack — tabulated ──────────────────────────────────────────────
def knapsack_tab(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> int:
    """
    Build dp[i][c] = max value using first i items, capacity c.
    dp is (n+1) × (capacity+1).
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]           # skip item i-1
            if weights[i - 1] <= c:
                dp[i][c] = max(
                    dp[i][c],
                    values[i - 1] + dp[i - 1][c - weights[i - 1]],  # take
                )
    return dp[n][capacity]


def knapsack_space_opt(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> int:
    """Space-optimised knapsack: O(capacity) space. Traverse capacity right→left."""
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for c in range(capacity, weights[i] - 1, -1):  # right→left prevents reuse
            dp[c] = max(dp[c], values[i] + dp[c - weights[i]])
    return dp[capacity]


# ── 4. Longest Common Subsequence — tabulated ────────────────────────────────
def lcs_tab(s: str, t: str) -> int:
    """O(m*n) time, O(m*n) space."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def lcs_reconstruct(s: str, t: str) -> str:
    """Recover the actual LCS string by backtracking through the table."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # backtrack
    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if s[i - 1] == t[j - 1]:
            result.append(s[i - 1])
            i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(result))


# ── 5. Coin change — tabulated ────────────────────────────────────────────────
def coin_change_tab(coins: list[int], amount: int) -> int:
    """Minimum coins to make amount. dp[a] = min coins for amount a."""
    INF = float("inf")
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return int(dp[amount]) if dp[amount] < INF else -1


# ── demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("fib_tab(40) = %d", fib_tab(40))
    logger.info("climb_stairs_tab(10) = %d", climb_stairs_tab(10))
    logger.info(
        "knapsack_tab([2,3,4,5],[3,4,5,6],5) = %d",
        knapsack_tab([2, 3, 4, 5], [3, 4, 5, 6], 5),
    )
    logger.info(
        "knapsack_space_opt([2,3,4,5],[3,4,5,6],5) = %d",
        knapsack_space_opt([2, 3, 4, 5], [3, 4, 5, 6], 5),
    )
    logger.info("lcs_tab('abcde','ace') = %d", lcs_tab("abcde", "ace"))
    logger.info("lcs_reconstruct('AGGTAB','GXTXAYB') = %s", lcs_reconstruct("AGGTAB", "GXTXAYB"))
    logger.info("coin_change_tab([1,5,6,9], 11) = %d", coin_change_tab([1, 5, 6, 9], 11))
```

## Exercises

1. **Unique paths with obstacles**: a grid has obstacles marked 1. Count paths from top-left to bottom-right (only right/down moves). Use tabulation.

2. **House robber**: given a list of non-negative integers representing money in houses, find the maximum you can rob without robbing adjacent houses. O(1) space tabulation.

3. **Longest increasing subsequence** (length only): given an array, find the length of the longest strictly increasing subsequence. `dp[i]` = LIS ending at index `i`. O(n²) tabulation.

4. **Space optimization**: reduce `lcs_tab` from O(m×n) space to O(min(m,n)) using two rolling rows. Verify the result is unchanged.

5. **Coin change II** (count ways, not min coins): given coins and amount, count the number of distinct combinations that sum to amount. Tabulate `dp[a]` = number of ways to make amount `a`.

(Solutions: ../solutions/week-05/day-3.md)

## Self-check

- [ ] I can convert any memoized solution to its tabulated equivalent.
- [ ] I understand why right-to-left iteration enables space optimization in 0/1 knapsack.
- [ ] I can implement LCS tabulation and reconstruct the sequence.
- [ ] I can explain the trade-offs between top-down and bottom-up DP.
- [ ] I can reduce a 2-D DP table to O(n) space when only the previous row is needed.

## Supplementary links

- CP-algorithms DP section: https://cp-algorithms.com/dynamic_programming/intro-to-dp.html
- "Demystifying Dynamic Programming" — Vaidehi Joshi: https://medium.com/basecs/less-repetition-more-dynamic-programming-43d29830a630
- Space optimization patterns: https://leetcode.com/discuss/study-guide/1000929/
