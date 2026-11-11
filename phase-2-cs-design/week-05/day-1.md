# Phase 2 · Week 05 · Day 1 — Recursion & Backtracking

> Objective: Build fluency with recursive thinking and systematic backtracking search.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-7]]

## Concept

**Recursion** is a function calling itself on a smaller sub-problem until it reaches a base case. Every recursive algorithm has two mandatory parts:

1. **Base case** — the condition that stops the recursion (no more self-calls).
2. **Recursive case** — the step that moves toward the base case and delegates the rest to itself.

**Backtracking** is recursion with *undoing*: explore a choice, recurse into it, then un-make that choice to try the next option. It is exhaustive search pruned by constraints.

Mental model: draw a **recursion tree**. Each node is a function call; leaf nodes are base cases. Backtracking trees have "undo" steps on the way back up.

## How it works internally

Each recursive call pushes a new **stack frame** onto the call stack — local variables, parameters, and return address are stored there. Deep recursion (large inputs without a base case) causes **stack overflow** (`RecursionError` in Python). Python's default recursion limit is 1000 (`sys.getrecursionlimit()`); you can raise it but this is rarely the right answer — prefer iteration or memoization instead.

Backtracking's "undo" step works because stack frames are isolated: modifying a shared mutable object (list, set) requires explicit revert; immutable objects (tuples, ints) are already isolated.

**Time complexity** of a recursion tree: roughly `O(branches^depth)`. Pruning cuts branches early; it does not change worst-case but dramatically improves average-case.

## Worked example

```python
"""
day1_recursion.py  —  recursion & backtracking examples
Python 3.12+
"""
from __future__ import annotations
import logging
from typing import Generator

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Classic recursion: factorial ──────────────────────────────────────────
def factorial(n: int) -> int:
    """Return n!  Base case: n == 0."""
    if n < 0:
        raise ValueError(f"factorial undefined for {n}")
    if n == 0:          # base case
        return 1
    return n * factorial(n - 1)   # recursive case


# ── 2. Fibonacci (naïve — intentionally unoptimised for Day 2) ───────────────
def fib_naive(n: int) -> int:
    """Return nth Fibonacci number (0-indexed). Exponential time — Day 2 fixes this."""
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


# ── 3. Backtracking: generate all subsets ────────────────────────────────────
def subsets(nums: list[int]) -> list[list[int]]:
    """Return all 2^n subsets of nums using backtracking."""
    result: list[list[int]] = []
    path: list[int] = []

    def backtrack(start: int) -> None:
        result.append(list(path))          # snapshot current path
        for i in range(start, len(nums)):
            path.append(nums[i])           # choose
            backtrack(i + 1)               # explore
            path.pop()                     # un-choose (backtrack)

    backtrack(0)
    return result


# ── 4. Backtracking: N-Queens ─────────────────────────────────────────────────
def solve_n_queens(n: int) -> list[list[str]]:
    """Return all valid N-Queens board configurations."""
    solutions: list[list[str]] = []
    cols: set[int] = set()
    diag1: set[int] = set()   # row - col
    diag2: set[int] = set()   # row + col
    board: list[int] = [-1] * n  # board[row] = col of queen

    def place(row: int) -> None:
        if row == n:
            # build string representation
            grid = []
            for r in range(n):
                grid.append("." * board[r] + "Q" + "." * (n - board[r] - 1))
            solutions.append(grid)
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue          # prune: constraint violated
            # choose
            board[row] = col
            cols.add(col); diag1.add(row - col); diag2.add(row + col)
            place(row + 1)        # explore
            # un-choose
            cols.remove(col); diag1.remove(row - col); diag2.remove(row + col)

    place(0)
    return solutions


# ── 5. Backtracking: permutations ────────────────────────────────────────────
def permutations(nums: list[int]) -> Generator[list[int], None, None]:
    """Yield every permutation of nums in-place (Heap-style via backtracking)."""
    n = len(nums)

    def backtrack(start: int) -> Generator[list[int], None, None]:
        if start == n:
            yield list(nums)
            return
        for i in range(start, n):
            nums[start], nums[i] = nums[i], nums[start]   # choose
            yield from backtrack(start + 1)               # explore
            nums[start], nums[i] = nums[i], nums[start]   # un-choose

    yield from backtrack(0)


# ── demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("factorial(5) = %d", factorial(5))
    logger.info("fib_naive(10) = %d", fib_naive(10))
    logger.info("subsets([1,2,3]) count = %d", len(subsets([1, 2, 3])))
    logger.info("4-queens solutions: %d", len(solve_n_queens(4)))
    logger.info("perms of [1,2,3]: %s", list(permutations([1, 2, 3])))
```

Run: `python day1_recursion.py`

## Exercises

1. **Tower of Hanoi**: implement `hanoi(n, src, dst, aux)` that logs each disk move. How many moves for `n` disks? Prove it recursively.

2. **Flatten nested list**: write `flatten(nested: list) -> list[int]` using recursion. Input can be arbitrarily deep, e.g. `[1, [2, [3, 4]], 5]`.

3. **Combination sum**: given a list of positive integers `candidates` and a target `target`, return all unique combinations that sum to `target`. Each number may be reused. Use backtracking.

4. **Phone keypad**: a phone keypad maps digits to letters (2→"abc", …). Given a string of digits, return all letter combinations using backtracking.

5. **Palindrome partitioning**: given string `s`, return all ways to partition it into palindrome substrings using backtracking with a pruning check.

(Solutions: ../solutions/week-05/day-1.md)

## Self-check

- [ ] I can identify the base case and recursive case in any recursive function.
- [ ] I can draw the recursion tree for factorial(4) and Fibonacci(4).
- [ ] I can implement a backtracking template (choose → explore → un-choose) from memory.
- [ ] I understand why immutable vs mutable state matters for the "un-choose" step.
- [ ] I can implement N-Queens without reference.

## Supplementary links

- Python `sys.setrecursionlimit` docs: https://docs.python.org/3/library/sys.html#sys.setrecursionlimit
- Backtracking pattern walkthrough (LeetCode explore): https://leetcode.com/explore/learn/card/recursion-ii/
- "Think Recursively" — Eric Roberts (classic short text, free PDF widely available)
