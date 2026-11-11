# Solutions: Week 08 · Day 4 — Dynamic Programming Patterns

## Exercise 1: House Robber

```python
from functools import lru_cache


def rob_top_down(nums: list[int]) -> int:
    """Memoised recursion.  Time: O(n)  Space: O(n)"""
    n = len(nums)

    @lru_cache(maxsize=None)
    def dp(i: int) -> int:
        if i >= n:
            return 0
        return max(nums[i] + dp(i + 2), dp(i + 1))

    return dp(0)


def rob_bottom_up(nums: list[int]) -> int:
    """Tabulation, space O(1).  Time: O(n)  Space: O(1)"""
    if not nums:
        return 0
    prev2, prev1 = 0, 0
    for amount in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + amount)
    return prev1
```

**Space optimisation:** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` — only the last two values needed, so keep `prev1` and `prev2` rolling variables.

---

## Exercise 2: Partition Equal Subset Sum

```python
def can_partition(nums: list[int]) -> bool:
    """0/1 knapsack boolean DP.  Time: O(n × sum)  Space: O(sum)"""
    total = sum(nums)
    if total % 2 != 0:
        return False        # early termination: odd total → impossible
    target = total // 2
    dp: set[int] = {0}      # reachable sums
    for num in nums:
        dp = dp | {s + num for s in dp if s + num <= target}
        if target in dp:
            return True
    return target in dp
```

**Boolean knapsack as set:** maintaining a set of reachable sums is equivalent to boolean DP table — cleaner for this variant. Alternatively:

```python
def can_partition_array(nums: list[int]) -> bool:
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):   # right-to-left: 0/1 knapsack
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
```

**Early termination:** any `num > target` → immediately return `False` (can't balance).

---

## Exercise 3: Edit Distance

```python
def min_distance(word1: str, word2: str) -> int:
    """Edit distance (Levenshtein).  Time: O(m×n)  Space: O(m×n)"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Base cases: transform prefix to empty string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]     # chars match: no operation
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # delete from word1
                    dp[i][j - 1],    # insert into word1
                    dp[i - 1][j - 1] # replace
                )
    return dp[m][n]
```

**Recurrence from LCS:** LCS only checks equality (match/skip). Edit distance adds the cost-1 replace transition and makes delete/insert explicit instead of "free skip". The table structure is identical.

**Complexity:** O(m×n) time and space. Reduce to O(min(m,n)) space by keeping only two rows.

---

## Exercise 4: Decode Ways

```python
def num_decodings(s: str) -> int:
    """1-D DP.  Time: O(n)  Space: O(1)"""
    if not s or s[0] == "0":
        return 0
    n = len(s)
    prev2, prev1 = 1, 1    # dp[-1]=1 (empty), dp[0]=1 (s[0] != '0')
    for i in range(1, n):
        curr = 0
        one_digit = int(s[i])
        two_digit = int(s[i - 1: i + 1])
        if one_digit != 0:           # s[i] can be decoded alone
            curr += prev1
        if 10 <= two_digit <= 26:    # s[i-1..i] is a valid two-digit code
            curr += prev2
        prev2, prev1 = prev1, curr
    return prev1
```

**Leading zero:** `s[i] == '0'` — the single digit 0 is invalid (no letter maps to 0), so `one_digit != 0` guard is correct. `"10"` is valid as J (10); `"30"` is invalid — `two_digit=30 > 26`, so no contribution from either branch.

---

## Exercise 5: Burst Balloons (Interval DP — sketch)

```python
def max_coins(nums: list[int]) -> int:
    """Interval DP: dp[l][r] = max coins for subarray nums[l..r].
    Time: O(n³)  Space: O(n²)
    """
    # Pad with virtual balloons of value 1
    balloons = [1] + nums + [1]
    n = len(balloons)
    dp = [[0] * n for _ in range(n)]

    # length of interval: expand from small to large
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            # k is the LAST balloon burst in subarray (left, right) exclusive
            for k in range(left + 1, right):
                dp[left][right] = max(
                    dp[left][right],
                    balloons[left] * balloons[k] * balloons[right]
                    + dp[left][k] + dp[k][right]
                )
    return dp[0][n - 1]
```

**Key insight:** think of `k` as the *last* balloon burst, not the first. When k is last, its neighbours are the boundary balloons (left, right), which haven't been burst yet. This makes the subproblems independent and the recurrence correct.

---

## Test file

```python
# tests/test_day4.py
import pytest
from day4_solutions import (
    rob_bottom_up, can_partition_array, min_distance, num_decodings, max_coins
)


def test_house_robber() -> None:
    assert rob_bottom_up([1, 2, 3, 1]) == 4
    assert rob_bottom_up([2, 7, 9, 3, 1]) == 12
    assert rob_bottom_up([]) == 0
    assert rob_bottom_up([5]) == 5


def test_can_partition() -> None:
    assert can_partition_array([1, 5, 11, 5]) is True
    assert can_partition_array([1, 2, 3, 5]) is False
    assert can_partition_array([1]) is False


def test_edit_distance() -> None:
    assert min_distance("horse", "ros") == 3
    assert min_distance("intention", "execution") == 5
    assert min_distance("", "abc") == 3
    assert min_distance("abc", "") == 3


def test_num_decodings() -> None:
    assert num_decodings("12") == 2
    assert num_decodings("226") == 3
    assert num_decodings("06") == 0
    assert num_decodings("10") == 1


def test_burst_balloons() -> None:
    assert max_coins([3, 1, 5, 8]) == 167
    assert max_coins([1, 5]) == 10
```
