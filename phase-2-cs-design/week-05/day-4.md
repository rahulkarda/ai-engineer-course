# Phase 2 · Week 05 · Day 4 — Classic DP Problems: Knapsack, LCS, Edit Distance

> Objective: Master three foundational DP problem families and recognize them in new problems.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Three DP archetypes cover a huge fraction of real problems:

| Archetype | Decision per state | Classic problems |
|---|---|---|
| **0/1 Knapsack** | Take or skip an item | subset sum, partition, target sum |
| **LCS / string alignment** | Match, insert, delete | edit distance, diff, regex matching |
| **Interval / sequence** | Extend or break sequence | LIS, matrix chain, burst balloons |

Recognizing the archetype is half the solution. The other half is specifying the **state**, the **transition**, and the **base case** precisely before writing any code.

**Edit distance** (Levenshtein) is one of the most practically important DP problems — used in spell checkers, DNA alignment, fuzzy matching, and diff tools. The state is `dp[i][j]` = minimum edits to transform `s[:i]` into `t[:j]`.

## How it works internally

**Edit distance transitions:**
```
if s[i-1] == t[j-1]:
    dp[i][j] = dp[i-1][j-1]          # no edit needed
else:
    dp[i][j] = 1 + min(
        dp[i-1][j],    # delete from s
        dp[i][j-1],    # insert into s
        dp[i-1][j-1],  # replace
    )
```

**LIS (O(n log n))**: the standard O(n²) DP can be improved using binary search on a "patience sort" deck — `tails[i]` = smallest tail element of increasing subsequences of length `i+1`. This is not DP in the classical sense but produces the same result via a greedy + binary search insight.

## Worked example

```python
"""
day4_classic_dp.py  —  Knapsack, LCS, Edit Distance, LIS
Python 3.12+
"""
from __future__ import annotations
import logging
import bisect

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Subset sum (knapsack variant) ─────────────────────────────────────────
def can_partition(nums: list[int]) -> bool:
    """
    Partition Equal Subset Sum (LeetCode 416).
    Can nums be split into two subsets with equal sum?
    Reduces to: can we pick a subset summing to total//2?
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    # dp[s] = True if sum s is achievable
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for s in range(target, num - 1, -1):  # right-to-left: 0/1 property
            dp[s] = dp[s] or dp[s - num]
    return dp[target]


# ── 2. Edit distance (Levenshtein) ───────────────────────────────────────────
def edit_distance(s: str, t: str) -> int:
    """
    Minimum number of single-character edits (insert, delete, replace)
    to transform s into t.
    dp[i][j] = edit distance for s[:i] and t[:j].
    """
    m, n = len(s), len(t)
    # Space-optimized: two rows
    prev = list(range(n + 1))   # base case: transforming "" into t[:j] = j inserts
    for i in range(1, m + 1):
        curr = [i] + [0] * n   # base case: transforming s[:i] into "" = i deletes
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def edit_distance_ops(s: str, t: str) -> tuple[int, list[str]]:
    """Return edit distance AND the sequence of operations."""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Backtrack
    ops: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"replace s[{i-1}]='{s[i-1]}' with '{t[j-1]}'")
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(f"insert '{t[j-1]}' at pos {i}")
            j -= 1
        else:
            ops.append(f"delete s[{i-1}]='{s[i-1]}'")
            i -= 1
    return dp[m][n], list(reversed(ops))


# ── 3. Longest Increasing Subsequence ────────────────────────────────────────
def lis_dp(nums: list[int]) -> int:
    """O(n^2) DP. dp[i] = length of LIS ending at index i."""
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def lis_nlogn(nums: list[int]) -> int:
    """O(n log n) patience sort approach. tails[i] = smallest tail of LIS length i+1."""
    tails: list[int] = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


# ── 4. Longest Palindromic Subsequence ───────────────────────────────────────
def longest_palindromic_subseq(s: str) -> int:
    """LPS = LCS(s, reversed(s))."""
    return lcs_tab(s, s[::-1])


def lcs_tab(s: str, t: str) -> int:
    m, n = len(s), len(t)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ── demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("can_partition([1,5,11,5]) = %s", can_partition([1, 5, 11, 5]))
    logger.info("can_partition([1,2,3,5]) = %s", can_partition([1, 2, 3, 5]))

    d, ops = edit_distance_ops("intention", "execution")
    logger.info("edit_distance('intention','execution') = %d", d)
    for op in ops:
        logger.info("  %s", op)

    logger.info(
        "lis_dp([10,9,2,5,3,7,101,18]) = %d",
        lis_dp([10, 9, 2, 5, 3, 7, 101, 18]),
    )
    logger.info(
        "lis_nlogn([10,9,2,5,3,7,101,18]) = %d",
        lis_nlogn([10, 9, 2, 5, 3, 7, 101, 18]),
    )
    logger.info(
        "longest_palindromic_subseq('bbbab') = %d",
        longest_palindromic_subseq("bbbab"),
    )
```

## Exercises

1. **Minimum path sum**: given an `m×n` grid of non-negative integers, find the minimum sum path from top-left to bottom-right (right/down moves only). Return both the sum and the path.

2. **Wildcard matching**: implement `is_match(s, p)` where `p` can contain `?` (match any single char) and `*` (match any sequence including empty). Use DP.

3. **Distinct subsequences**: given strings `s` and `t`, count distinct subsequences of `s` equal to `t`. `dp[i][j]` = count using `s[:i]` matching `t[:j]`.

4. **LIS reconstruction**: modify `lis_dp` to also return the actual subsequence, not just the length. (Hint: track a `parent` array.)

5. **Palindrome min cuts**: given a string, find the minimum number of cuts to partition it into palindrome substrings. First build a `is_palindrome[i][j]` table, then DP on cuts.

(Solutions: ../solutions/week-05/day-4.md)

## Self-check

- [ ] I can derive the edit distance recurrence from scratch.
- [ ] I can explain why can_partition uses right-to-left iteration.
- [ ] I can implement LIS in O(n²) and O(n log n) and explain why both are correct.
- [ ] I can recognize "knapsack variant", "string alignment", or "LIS-like" when I see a new problem.
- [ ] I can reconstruct the solution (not just the length/cost) from a DP table.

## Supplementary links

- Edit distance visualization: https://www.educative.io/answers/what-is-levenshtein-distance
- LIS O(n log n) explanation: https://cp-algorithms.com/sequences/longest_increasing_subsequence.html
- Knapsack variations reference: https://codeforces.com/blog/entry/47247
