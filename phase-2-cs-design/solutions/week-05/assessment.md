# Answer Key — Week 05 Assessment

> Instructor reference only. Do not share before the learner completes the assessment.

---

## Part 1 — Conceptual answers

**Q1** — Base case + recursive case. Omitting base case: `def f(n): return f(n-1)` — no termination → infinite recursion → `RecursionError`.

**Q2** — Memoization: top-down recursive, lazy (only needed states), stack overhead, harder to space-optimize. Tabulation: bottom-up iterative, fills all states, no stack risk, space optimization easy. Prefer memoization for sparse state spaces and natural recursive structure; prefer tabulation for large n (stack risk), dense tables, or when space reduction is needed.

**Q3** — `@cache` uses args as dict keys, which requires `hash()`. Lists are unhashable because they're mutable. Solution: convert list to `tuple` before calling, or use a manual `memo: dict[tuple, ...]` keyed on `tuple(list_arg)`.

**Q4** — (1) Greedy-choice property: local optimal → global optimal. (2) Optimal substructure. Works: interval scheduling (sort by finish time). Fails: 0/1 knapsack (must sometimes skip a high-value item to fit more total value).

**Q5** — Time O(n×W), space O(n×W) → O(W). Reduction: single 1-D array `dp[0..W]` traversed right-to-left per item so `dp[c - weight]` still reflects the previous item's values.

**Q6**:
```
if s[i-1] == t[j-1]:
    dp[i][j] = dp[i-1][j-1]
else:
    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
              #      delete s[i]  insert      replace
```

**Q7** — O(n²) DP: `dp[i]` = LIS length ending at `i`, scans all `j < i` → O(n) per state.
O(n log n): maintains `tails[k]` = smallest tail of any LIS of length `k+1`. For each number, binary search for its insertion position. `len(tails)` = LIS length. Key insight: `tails` is always sorted, enabling binary search.

---

## Part 2 — Coding solutions

### Challenge A — Minimum cost to cut a stick

```python
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

def min_cost_cuts_memo(n: int, cuts: list[int]) -> int:
    sorted_cuts = sorted(cuts)
    positions = [0] + sorted_cuts + [n]
    m = len(positions)

    @lru_cache(maxsize=None)
    def dp(l: int, r: int) -> int:
        if r - l <= 1:
            return 0
        min_cost = float("inf")
        for k in range(l+1, r):
            cost = positions[r] - positions[l] + dp(l, k) + dp(k, r)
            if cost < min_cost:
                min_cost = cost
        return int(min_cost)

    return dp(0, m - 1)


def min_cost_cuts_tab(n: int, cuts: list[int]) -> int:
    positions = sorted([0] + cuts + [n])
    m = len(positions)
    dp = [[0]*m for _ in range(m)]
    for length in range(2, m):
        for l in range(m - length):
            r = l + length
            dp[l][r] = min(
                positions[r] - positions[l] + dp[l][k] + dp[k][r]
                for k in range(l+1, r)
            )
    return dp[0][m-1]
```

Test cases:
```python
# min_cost_cuts_memo(7, [1,3,4,5]) == 16
# min_cost_cuts_memo(9, [5,6,1,4,2]) == 22
```

### Challenge B — Word break II

```python
from functools import lru_cache

def word_break_all(s: str, word_dict: list[str]) -> list[str]:
    word_set = frozenset(word_dict)
    n = len(s)

    @lru_cache(maxsize=None)
    def dp(start: int) -> list[str]:
        if start == n:
            return [""]
        sentences: list[str] = []
        for end in range(start + 1, n + 1):
            word = s[start:end]
            if word in word_set:
                for rest in dp(end):
                    sentences.append(word if not rest else word + " " + rest)
        return sentences

    return dp(0)
```

---

## Part 3 — Bugs

**Bug A**: `return dp[amount]` should be:
```python
return int(dp[amount]) if dp[amount] != float("inf") else -1
```
The current code returns `inf` when the amount is unreachable rather than `-1`.

**Bug B**: `result.append(path)` should be:
```python
result.append(list(path))
```
Without `list(path)`, all entries in `result` point to the same mutable object. After backtracking, they all end up as empty lists.

---

## Part 4 — Code quality checklist answers

Graders check:
1. Every function has parameter and return type hints.
2. `import logging; logger = logging.getLogger(__name__)` present; no bare `print`.
3. Pytest: `@pytest.mark.parametrize` used; test names describe input and expected output.
4. No `def f(x, memo=[])` mutable default; no module-level mutable state used as DP table.
5. Variable names: `dp` is acceptable; index `i,j` acceptable in DP context; no single-letter non-index names.
