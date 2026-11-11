# Phase 7 · Week 08 · Day 4 — Dynamic Programming Patterns

> Objective: Recognise and apply the four major DP patterns: 1-D, 2-D, unbounded knapsack, LCS family.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Dynamic programming = recursion + memoisation (top-down) or table-fill (bottom-up). The key insight: **optimal substructure + overlapping subproblems**.

**Approach in interviews:**
1. Define `dp[i]` (or `dp[i][j]`) in English first.
2. Write the recurrence.
3. Identify base cases.
4. Choose top-down (easier to derive) or bottom-up (easier to optimise space).

---

### Pattern 1 — 1-D DP (linear sequence)

`dp[i]` = answer for prefix ending at / up to index i.

```python
# Climbing stairs: dp[i] = ways to reach step i
dp = [0] * (n + 1)
dp[0], dp[1] = 1, 1
for i in range(2, n + 1):
    dp[i] = dp[i-1] + dp[i-2]
```

Space optimise: keep only last two values → O(1) space.

---

### Pattern 2 — 2-D DP (two sequences or grid)

`dp[i][j]` = answer using first i elements of seq1 and j elements of seq2 (or for grid cell (i,j)).

```python
# Unique paths on m×n grid
dp = [[1] * n for _ in range(m)]
for i in range(1, m):
    for j in range(1, n):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
return dp[m-1][n-1]
```

---

### Pattern 3 — 0/1 Knapsack

`dp[i][w]` = max value using first i items with capacity w.

```python
for i in range(1, n + 1):
    for w in range(W + 1):
        if weights[i-1] > w:
            dp[i][w] = dp[i-1][w]
        else:
            dp[i][w] = max(dp[i-1][w],
                           dp[i-1][w - weights[i-1]] + values[i-1])
```

**Unbounded knapsack** (reuse items): loop same direction `for w in range(weights[i-1], W+1)`.

---

### Pattern 4 — LCS / Edit Distance family

```python
# Longest Common Subsequence
for i in range(1, m + 1):
    for j in range(1, n + 1):
        if text1[i-1] == text2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
```

Edit Distance adds the replace transition: `dp[i-1][j-1] + 1`.

---

## How it works internally

**Memoisation (top-down):** Python `@functools.lru_cache` or manual `dict` transforms exponential recursion to polynomial by caching results keyed on arguments. Cache hit = O(1) lookup instead of recomputation.

**Bottom-up order:** fill in dependency order — ensure `dp[i-1]` is computed before `dp[i]`. For 2-D, row-major or column-major works depending on recurrence direction.

**Space reduction:** most 2-D DPs only look at the previous row → reduce to O(n) by keeping one or two rolling rows. Iterate carefully to avoid overwriting values still needed.

---

## Worked example

### Problem 1: Coin Change (1-D unbounded knapsack)

```python
# dp_patterns.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def coin_change(coins: list[int], amount: int) -> int:
    """Fewest coins to reach amount.  dp[i] = min coins for amount i.
    Time: O(amount * len(coins))  Space: O(amount)
    """
    INF = float("inf")
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)
    return -1 if dp[amount] == INF else int(dp[amount])
```

### Problem 2: Longest Common Subsequence

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """LCS length.  Time: O(m*n)  Space: O(m*n)"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

### Problem 3: 0/1 Knapsack (space-optimised)

```python
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """Max value with weight limit.  Time: O(n*W)  Space: O(W)"""
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        # iterate right-to-left to avoid reusing same item
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]
```

---

## Exercises

1. **House Robber** — you cannot rob two adjacent houses. Given array of values, find max you can steal. Write both top-down (with `lru_cache`) and bottom-up solutions; space-optimise the bottom-up to O(1).

2. **Partition Equal Subset Sum** — determine if a set of positive integers can be partitioned into two subsets with equal sum. Model as 0/1 knapsack (boolean DP). What's the early termination condition?

3. **Edit Distance** — minimum operations (insert, delete, replace) to convert `word1` to `word2`. Derive the recurrence from the LCS table; state time and space complexity.

4. **Decode Ways** — count the number of ways to decode a digit string (A=1, B=2, …, Z=26). Handle leading zeros and double-digit decoding. Use 1-D DP.

5. **Burst Balloons** — given n balloons with values, pop them in any order; popping balloon i earns `nums[left]*nums[i]*nums[right]`. Find the maximum coins. Use interval DP (`dp[l][r]` = max coins for subarray). This is a harder problem — sketch the recurrence and base case.

(Solutions: ../solutions/week-08/day-4.md)

---

## Self-check

- [ ] I can classify a DP problem into one of the four patterns from the problem statement
- [ ] I can derive a recurrence before writing code
- [ ] I can implement coin change and LCS from memory
- [ ] I can space-optimise a 2-D DP to O(n)
- [ ] I can explain memoisation vs. tabulation trade-offs in an interview

---

## Supplementary links

- NeetCode 1-D DP: https://neetcode.io/roadmap
- NeetCode 2-D DP: https://neetcode.io/roadmap
- NeetCode Advanced Graphs + DP: https://neetcode.io/roadmap
