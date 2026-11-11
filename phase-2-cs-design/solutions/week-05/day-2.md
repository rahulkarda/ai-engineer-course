# Solutions — Week 05 Day 2 (Memoization / Top-Down DP)

---

## Exercise 1 — Word break

```python
from functools import lru_cache

def word_break(s: str, word_dict: list[str]) -> bool:
    word_set = frozenset(word_dict)

    @lru_cache(maxsize=None)
    def dp(start: int) -> bool:
        if start == len(s):
            return True
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set and dp(end):
                return True
        return False

    return dp(0)

# word_break("leetcode", ["leet","code"]) → True
# word_break("applepenapple", ["apple","pen"]) → True
# word_break("catsandog", ["cats","dog","sand","and","cat"]) → False
```

**Complexity**: O(n²) sub-problems × O(n) string slice → O(n³) time; O(n) space.

---

## Exercise 2 — Unique paths

```python
from functools import cache

def unique_paths(m: int, n: int) -> int:
    @cache
    def dp(r: int, c: int) -> int:
        if r == 0 or c == 0:
            return 1
        return dp(r - 1, c) + dp(r, c - 1)
    return dp(m - 1, n - 1)

# unique_paths(3, 7) → 28
```

---

## Exercise 3 — Decode ways

```python
from functools import lru_cache

def num_decodings(s: str) -> int:
    @lru_cache(maxsize=None)
    def dp(i: int) -> int:
        if i == len(s):
            return 1
        if s[i] == "0":
            return 0
        ways = dp(i + 1)
        if i + 1 < len(s) and 10 <= int(s[i:i+2]) <= 26:
            ways += dp(i + 2)
        return ways
    return dp(0)

# num_decodings("12") → 2
# num_decodings("226") → 3
# num_decodings("06") → 0
```

---

## Exercise 4 — Target sum

```python
from functools import lru_cache

def find_target_sum_ways(nums: list[int], target: int) -> int:
    @lru_cache(maxsize=None)
    def dp(i: int, current_sum: int) -> int:
        if i == len(nums):
            return 1 if current_sum == target else 0
        return dp(i + 1, current_sum + nums[i]) + dp(i + 1, current_sum - nums[i])
    return dp(0, 0)

# find_target_sum_ways([1,1,1,1,1], 3) → 5
```

**State**: `(index, current_sum)`. Number of unique states: `n × (2 × sum(nums) + 1)`.

---

## Exercise 5 — Cache analysis for LCS

`lcs_memo("AGGTAB", "GXTXAYB")`:
- s = "AGGTAB" (len 6), t = "GXTXAYB" (len 7)
- Unique `(i, j)` states: `(6+1) × (7+1) = 56`
- Time complexity: O(m × n) = O(6 × 7) = O(42) sub-problems, each O(1) work → O(m×n)
- Space complexity: O(m × n) for the memo table + O(m+n) call stack depth
- LCS length = 4 ("GTAB")
