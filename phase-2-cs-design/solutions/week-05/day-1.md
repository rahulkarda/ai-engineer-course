# Solutions — Week 05 Day 1 (Recursion & Backtracking)

> Full solutions to all Day 1 exercises.

---

## Exercise 1 — Tower of Hanoi

```python
import logging
logger = logging.getLogger(__name__)

def hanoi(n: int, src: str, dst: str, aux: str) -> int:
    """Move n disks from src to dst using aux. Returns total moves."""
    if n == 0:
        return 0
    moves = 0
    moves += hanoi(n - 1, src, aux, dst)          # move n-1 disks to aux
    logger.info("Move disk %d from %s to %s", n, src, dst)
    moves += 1
    moves += hanoi(n - 1, aux, dst, src)          # move n-1 disks from aux to dst
    return moves
```

Number of moves for `n` disks: `T(n) = 2T(n-1) + 1 = 2^n - 1`.

**Proof**: T(0) = 0 = 2^0 - 1. Assume T(k) = 2^k - 1. Then T(k+1) = 2(2^k - 1) + 1 = 2^{k+1} - 1. QED.

---

## Exercise 2 — Flatten nested list

```python
def flatten(nested: list) -> list[int]:
    result: list[int] = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

# flatten([1, [2, [3, 4]], 5]) → [1, 2, 3, 4, 5]
```

---

## Exercise 3 — Combination sum

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    result: list[list[int]] = []
    candidates.sort()  # sort enables early termination

    def backtrack(start: int, path: list[int], remaining: int) -> None:
        if remaining == 0:
            result.append(list(path))
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # prune: sorted, so all further candidates also too large
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i, not i+1: can reuse
            path.pop()

    backtrack(0, [], target)
    return result
```

---

## Exercise 4 — Phone keypad

```python
KEYPAD: dict[str, str] = {
    "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
    "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz",
}

def letter_combinations(digits: str) -> list[str]:
    if not digits:
        return []
    result: list[str] = []

    def backtrack(idx: int, path: list[str]) -> None:
        if idx == len(digits):
            result.append("".join(path))
            return
        for ch in KEYPAD[digits[idx]]:
            path.append(ch)
            backtrack(idx + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

---

## Exercise 5 — Palindrome partitioning

```python
def partition(s: str) -> list[list[str]]:
    result: list[list[str]] = []
    path: list[str] = []

    def is_palindrome(sub: str) -> bool:
        return sub == sub[::-1]

    def backtrack(start: int) -> None:
        if start == len(s):
            result.append(list(path))
            return
        for end in range(start + 1, len(s) + 1):
            candidate = s[start:end]
            if is_palindrome(candidate):    # prune: only recurse on valid prefix
                path.append(candidate)
                backtrack(end)
                path.pop()

    backtrack(0)
    return result
```

**Optimization**: precompute `is_pal[i][j]` using DP to avoid O(n) check inside the loop — reduces backtracking inner work from O(n) to O(1) per check.
