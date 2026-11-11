# Solutions — Week 05 Day 4 (Classic DP Problems)

---

## Exercise 1 — Minimum path sum with path reconstruction

```python
def min_path_sum(grid: list[list[int]]) -> tuple[int, list[tuple[int,int]]]:
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for c in range(1, n):
        dp[0][c] = dp[0][c-1] + grid[0][c]
    for r in range(1, m):
        dp[r][0] = dp[r-1][0] + grid[r][0]
    for r in range(1, m):
        for c in range(1, n):
            dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])
    # Reconstruct path
    path: list[tuple[int,int]] = []
    r, c = m-1, n-1
    while r > 0 or c > 0:
        path.append((r, c))
        if r == 0:
            c -= 1
        elif c == 0:
            r -= 1
        elif dp[r-1][c] < dp[r][c-1]:
            r -= 1
        else:
            c -= 1
    path.append((0, 0))
    return dp[m-1][n-1], list(reversed(path))
```

---

## Exercise 2 — Wildcard matching

```python
def is_match_wildcard(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True
    # '*' can match empty string
    for j in range(1, n+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-1]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[i][j] = dp[i-1][j] or dp[i][j-1]
                # dp[i-1][j] = '*' matches one more char from s
                # dp[i][j-1] = '*' matches zero chars
            elif p[j-1] == '?' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
    return dp[m][n]

# is_match_wildcard("adceb","*a*b") → True
# is_match_wildcard("acdcb","a*c?b") → False
```

---

## Exercise 3 — Distinct subsequences

```python
def num_distinct(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = 1  # empty t is always a subseq
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j]  # don't use s[i-1]
            if s[i-1] == t[j-1]:
                dp[i][j] += dp[i-1][j-1]  # use s[i-1] to match t[j-1]
    return dp[m][n]

# num_distinct("rabbbit","rabbit") → 3
```

---

## Exercise 4 — LIS reconstruction

```python
def lis_with_sequence(nums: list[int]) -> list[int]:
    if not nums:
        return []
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    # Find end index of LIS
    best_end = max(range(n), key=lambda i: dp[i])
    # Backtrack
    sequence: list[int] = []
    idx = best_end
    while idx != -1:
        sequence.append(nums[idx])
        idx = parent[idx]
    return list(reversed(sequence))

# lis_with_sequence([10,9,2,5,3,7,101,18]) → [2,5,7,101] (or [2,3,7,101])
```

---

## Exercise 5 — Palindrome min cuts

```python
def min_cut(s: str) -> int:
    n = len(s)
    # Precompute palindrome table — O(n²)
    is_pal = [[False]*n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_pal[i][j] = (length == 2) or is_pal[i+1][j-1]
    # dp[i] = min cuts for s[:i+1]
    dp = list(range(n))  # worst case: cut every char
    for i in range(1, n):
        if is_pal[0][i]:
            dp[i] = 0
            continue
        for j in range(1, i+1):
            if is_pal[j][i]:
                dp[i] = min(dp[i], dp[j-1] + 1)
    return dp[n-1]

# min_cut("aab") → 1  ("aa" | "b")
# min_cut("a") → 0
# min_cut("ab") → 1
```
