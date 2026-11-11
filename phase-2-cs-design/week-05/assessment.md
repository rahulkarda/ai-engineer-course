# Week 05 Assessment — Recursion, DP & Greedy

> Time limit: 60 minutes (timed, closed reference)  ·  Pass gate: ≥ 75%

---

## Part 1 — Conceptual (25 points, ~15 min)

Answer in 2–4 sentences each.

1. **(3 pts)** What are the two mandatory parts of every recursive function? Give an example where omitting the base case causes a `RecursionError`.

2. **(4 pts)** Explain the difference between memoization and tabulation. When would you prefer each?

3. **(4 pts)** Why does `@functools.cache` require hashable arguments? What would you do if your DP state includes a list?

4. **(4 pts)** State the two conditions a problem must satisfy for a greedy algorithm to be correct. Give one problem where greedy works and one where it fails, with brief justification.

5. **(4 pts)** For 0/1 knapsack with capacity `W` and `n` items, what is the time and space complexity of the tabulated solution? How is space reduced to O(W)?

6. **(3 pts)** The edit distance recurrence has three operations: insert, delete, replace. Write the full recurrence for `dp[i][j]` from scratch.

7. **(3 pts)** Why is the O(n log n) LIS algorithm (patience sort) more efficient than the O(n²) DP? What is the key insight that enables binary search?

---

## Part 2 — Coding Challenge (40 points, ~35 min)

Implement the following in Python 3.12+ with full type hints. No imports beyond the standard library. Use `logging` not `print`.

### Challenge A — Minimum Cost to Cut a Stick (20 pts)

Given a stick of length `n` and a list of positions `cuts` to make, find the minimum cost to make all cuts. The cost of a cut is the current length of the stick being cut. You can make cuts in any order.

**Function signature:**
```python
def min_cost_cuts_memo(n: int, cuts: list[int]) -> int: ...
def min_cost_cuts_tab(n: int, cuts: list[int]) -> int: ...
```

**Acceptance criteria:**
- Both functions return the same integer for all inputs.
- `min_cost_cuts_memo(7, [1, 3, 4, 5])` returns `16`.
- `min_cost_cuts_memo(9, [5, 6, 1, 4, 2])` returns `22`.
- Time complexity: O(m³) where m = len(cuts). Space: O(m²).
- At least one unit test per function using `pytest`.

### Challenge B — Word Break II (20 pts)

Given a string `s` and a list of dictionary words `word_dict`, return **all** ways to segment `s` into space-separated dictionary words.

**Function signature:**
```python
def word_break_all(s: str, word_dict: list[str]) -> list[str]: ...
```

**Acceptance criteria:**
- `word_break_all("catsanddog", ["cat","cats","and","sand","dog"])` returns `["cat sand dog", "cats and dog"]` (order may vary).
- `word_break_all("pineapplepenapple", ["apple","pen","applepen","pine","pineapple"])` returns correct 3-sentence segmentations.
- Handles `word_break_all("catsandog", [...])` returning `[]` correctly.
- Uses memoization. Time: O(n² × m) where m = len(word_dict).
- At least two unit tests.

---

## Part 3 — Debugging (20 points, ~10 min)

Each snippet has one logical bug. Identify and fix it (no need to rewrite — state what line is wrong and what the fix is).

### Bug A (10 pts)
```python
def coin_change(coins: list[int], amount: int) -> int:
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount]  # BUG: what should this line be?
```

### Bug B (10 pts)
```python
def subsets(nums: list[int]) -> list[list[int]]:
    result = []
    path = []
    def backtrack(start):
        result.append(path)          # BUG: what is wrong here?
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()
    backtrack(0)
    return result
```

---

## Part 4 — Code Quality (15 points)

Your grader will evaluate your Challenge A and B submissions on:

| Criterion | Points |
|---|---|
| Type hints on all function signatures | 3 |
| `logging` used (no bare `print`) | 2 |
| Pytest tests: at least 2 per function, descriptive names | 5 |
| No mutable default arguments; no global state | 3 |
| Clear variable names; no magic numbers without comments | 2 |

---

## Rubric (pass ≥ 75 / 100)

| Dimension | Weight | Points |
|---|---|---|
| Independent implementation (Part 2) | 40% | 40 |
| Conceptual understanding (Part 1) | 25% | 25 |
| Debugging & problem-solving (Part 3) | 20% | 20 |
| Code quality & testing (Part 4) | 15% | 15 |
| **Total** | **100%** | **100** |

---

Answer key: ../solutions/week-05/assessment.md
