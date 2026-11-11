# Solutions — Week 05 Day 3 (Tabulation / Bottom-Up DP)

---

## Exercise 1 — Unique paths with obstacles

```python
def unique_paths_with_obstacles(obstacle_grid: list[list[int]]) -> int:
    m, n = len(obstacle_grid), len(obstacle_grid[0])
    if obstacle_grid[0][0] == 1 or obstacle_grid[m-1][n-1] == 1:
        return 0
    dp = [[0] * n for _ in range(m)]
    # Base: first column
    for r in range(m):
        if obstacle_grid[r][0] == 1:
            break
        dp[r][0] = 1
    # Base: first row
    for c in range(n):
        if obstacle_grid[0][c] == 1:
            break
        dp[0][c] = 1
    # Fill
    for r in range(1, m):
        for c in range(1, n):
            if obstacle_grid[r][c] == 0:
                dp[r][c] = dp[r-1][c] + dp[r][c-1]
    return dp[m-1][n-1]
```

---

## Exercise 2 — House robber

```python
def rob(nums: list[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)
    return prev1

# rob([2,7,9,3,1]) → 12
```

O(1) space: only `prev2` and `prev1` needed.

---

## Exercise 3 — LIS (O(n²) tabulation)

```python
def lis_length(nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# lis_length([10,9,2,5,3,7,101,18]) → 4  (e.g. [2,5,7,101])
```

---

## Exercise 4 — LCS space optimization

```python
def lcs_two_rows(s: str, t: str) -> int:
    """O(min(m,n)) space by keeping only prev and curr rows."""
    if len(s) < len(t):
        s, t = t, s          # ensure s is longer for min space
    m, n = len(s), len(t)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                curr[j] = 1 + prev[j-1]
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    return prev[n]
```

Result is identical to the O(m×n) version. Space: O(min(m,n)).

---

## Exercise 5 — Coin change II (count combinations)

```python
def coin_change_count(coins: list[int], amount: int) -> int:
    """Count distinct combinations (order doesn't matter)."""
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]
    return dp[amount]

# coin_change_count([1,2,5], 5) → 4  ([5], [2+2+1], [2+1+1+1], [1+1+1+1+1])
```

**Key**: iterate coins in the outer loop to avoid counting permutations as distinct combinations. This is the unbounded knapsack (left→right) pattern.
