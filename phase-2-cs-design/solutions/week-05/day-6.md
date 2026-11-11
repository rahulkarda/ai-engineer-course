# Solutions — Week 05 Day 6 (Saturday DP Problem Set)

---

## Problem 1 — Interleaving Strings

```python
from functools import lru_cache

def is_interleave_memo(s1: str, s2: str, s3: str) -> bool:
    if len(s1) + len(s2) != len(s3):
        return False
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        if i == len(s1) and j == len(s2):
            return True
        k = i + j
        if i < len(s1) and s1[i] == s3[k] and dp(i+1, j):
            return True
        if j < len(s2) and s2[j] == s3[k] and dp(i, j+1):
            return True
        return False
    return dp(0, 0)

def is_interleave_tab(s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True
    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] and s1[i-1] == s3[i-1]
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] and s2[j-1] == s3[j-1]
    for i in range(1, m+1):
        for j in range(1, n+1):
            k = i + j - 1
            dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[k]) or \
                       (dp[i][j-1] and s2[j-1] == s3[k])
    return dp[m][n]
```

**Complexity**: O(m×n) time, O(m×n) space; reducible to O(n) with rolling row.

---

## Problem 2 — Burst Balloons

**Key insight**: instead of thinking about order of bursting, think about which balloon is burst *last* in a range. Define `dp[l][r]` = max coins from bursting all balloons in `nums[l..r]` (with padding 1s at boundaries).

```python
def max_coins_tab(nums: list[int]) -> int:
    padded = [1] + nums + [1]
    n = len(padded)
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n):
        for left in range(0, n - length):
            right = left + length
            for k in range(left+1, right):
                dp[left][right] = max(
                    dp[left][right],
                    padded[left]*padded[k]*padded[right] + dp[left][k] + dp[k][right]
                )
    return dp[0][n-1]

def max_coins_memo(nums: list[int]) -> int:
    padded = [1] + nums + [1]
    n = len(padded)
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(l: int, r: int) -> int:
        if r - l < 2:
            return 0
        return max(
            padded[l]*padded[k]*padded[r] + dp(l, k) + dp(k, r)
            for k in range(l+1, r)
        )
    return dp(0, n-1)
```

**Complexity**: O(n³) time, O(n²) space.

---

## Problem 3 — Regular Expression Matching

```python
from functools import lru_cache

def is_match_memo(s: str, p: str) -> bool:
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        if j == len(p):
            return i == len(s)
        first_match = i < len(s) and p[j] in {s[i], '.'}
        if j+1 < len(p) and p[j+1] == '*':
            # '*' matches zero occurrences OR one+more (via first_match)
            return dp(i, j+2) or (first_match and dp(i+1, j))
        return first_match and dp(i+1, j+1)
    return dp(0, 0)

def is_match_tab(s: str, p: str) -> bool:
    m, n = len(s), len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[m][n] = True
    for i in range(m, -1, -1):
        for j in range(n-1, -1, -1):
            first = i < m and p[j] in {s[i], '.'}
            if j+1 < n and p[j+1] == '*':
                dp[i][j] = dp[i][j+2] or (first and dp[i+1][j])
            else:
                dp[i][j] = first and dp[i+1][j+1]
    return dp[0][0]
```

---

## Problem 4 — Maximum Rectangle in Histogram

```python
def largest_rectangle_histogram(heights: list[int]) -> int:
    stack: list[int] = []
    max_area = 0
    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area

def maximal_rectangle(matrix: list[list[str]]) -> int:
    if not matrix:
        return 0
    n = len(matrix[0])
    heights = [0] * n
    max_area = 0
    for row in matrix:
        for c in range(n):
            heights[c] = heights[c] + 1 if row[c] == '1' else 0
        max_area = max(max_area, largest_rectangle_histogram(heights))
    return max_area
```

---

## Problem 5 — Egg Drop

```python
import sys
from functools import lru_cache

def egg_drop_memo(k: int, n: int) -> int:
    @lru_cache(maxsize=None)
    def dp(eggs: int, floors: int) -> int:
        if floors <= 1:
            return floors
        if eggs == 1:
            return floors
        lo, hi = 1, floors
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            broke = dp(eggs-1, mid-1)
            survived = dp(eggs, floors-mid)
            if broke < survived:
                lo = mid
            elif broke > survived:
                hi = mid
            else:
                lo = hi = mid
        return 1 + min(max(dp(eggs-1, f-1), dp(eggs, floors-f)) for f in [lo, hi])
    return dp(k, n)

def egg_drop_tab(k: int, n: int) -> int:
    # dp[m][e] = max floors testable with m moves and e eggs
    # Find min m such that dp[m][k] >= n
    dp = [[0]*(k+1) for _ in range(n+1)]
    m = 0
    while dp[m][k] < n:
        m += 1
        for e in range(1, k+1):
            dp[m][e] = dp[m-1][e-1] + dp[m-1][e] + 1
    return m
```

---

## Problem 6 — Longest Valid Parentheses

```python
def longest_valid_parens_dp(s: str) -> int:
    n = len(s)
    dp = [0] * n
    for i in range(1, n):
        if s[i] == ')':
            if s[i-1] == '(':
                dp[i] = (dp[i-2] if i >= 2 else 0) + 2
            elif dp[i-1] > 0:
                j = i - dp[i-1] - 1
                if j >= 0 and s[j] == '(':
                    dp[i] = dp[i-1] + 2 + (dp[j-1] if j > 0 else 0)
    return max(dp, default=0)

def longest_valid_parens_stack(s: str) -> int:
    stack = [-1]
    max_len = 0
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len
```

Two-pass O(1) space approach: count `left` and `right` going left→right (reset when right>left), then right→left (reset when left>right); max valid = `max(max from pass1, max from pass2)`.
