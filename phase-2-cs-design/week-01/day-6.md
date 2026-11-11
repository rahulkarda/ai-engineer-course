# Phase 2 · Week 01 · Day 6 — Mixed Problem Set (4 hr)

> Objective: Apply all week-1 techniques (complexity analysis, two pointers, sliding window, prefix sums, string algorithms) under timed, exam-like conditions.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] – [[day-5]]

---

## Instructions

Work through all 12 problems. For each:

1. State the approach and time/space complexity **before** writing code.
2. Write a clean, type-hinted solution.
3. Add at least two test cases (use `assert` or pytest).
4. If you get stuck for more than 15 minutes, read the hint — but record that you used it.

Target pace: ~20 min per problem, with buffer for harder ones.

---

## Problem 1 — Complexity Classification (15 min)

Classify each snippet's **time** and **space** complexity. Justify in one sentence per snippet.

```python
# (a)
def f(n: int) -> int:
    result = 0
    i = 1
    while i < n:
        result += i
        i *= 2
    return result

# (b)
def g(arr: list[int]) -> list[int]:
    return [x for x in arr if x % 2 == 0]

# (c)
def h(n: int) -> int:
    if n <= 1:
        return 1
    return h(n // 2) + h(n // 2)
    # note: two recursive calls, same subproblem

# (d)
def k(arr: list[list[int]]) -> int:
    n = len(arr)
    total = 0
    for i in range(n):
        for j in range(i, n):       # inner starts at i, not 0
            total += arr[i][j]
    return total
```

---

## Problem 2 — Amortized Analysis (10 min)

A custom stack class doubles capacity on push when full, and halves capacity when size < capacity/4.

```python
class DynamicStack:
    def __init__(self) -> None:
        self._data: list[int] = [0]   # initial capacity 1
        self._size = 0

    def push(self, val: int) -> None:
        if self._size == len(self._data):
            self._data = self._data * 2   # double
        self._data[self._size] = val
        self._size += 1

    def pop(self) -> int:
        if self._size == 0:
            raise IndexError("empty stack")
        val = self._data[self._size - 1]
        self._size -= 1
        if self._size < len(self._data) // 4 and len(self._data) > 1:
            self._data = self._data[:len(self._data) // 2]   # halve
        return val
```

a) What is the worst-case time of a single `push`?  
b) What is the amortized time of `push` over n pushes?  
c) Why is the "halve at 1/4 capacity" threshold important? What goes wrong if you halve at 1/2?

---

## Problem 3 — Two Pointers: Dutch National Flag (20 min)

Implement `sort_colors(arr: list[int]) -> None` — sort an array containing only 0s, 1s, 2s in-place in O(n) time and O(1) space (the "Dutch National Flag" problem, LeetCode 75). Do NOT count elements and refill — partition in place.

Hint: maintain three regions: `[0..low-1]` = 0s, `[low..mid-1]` = 1s, `[mid..high]` = unknown, `[high+1..n-1]` = 2s.

```python
def sort_colors(arr: list[int]) -> None:
    ...

# Tests
arr = [2, 0, 2, 1, 1, 0]
sort_colors(arr)
assert arr == [0, 0, 1, 1, 2, 2]
```

---

## Problem 4 — Two Pointers: Three Sum (20 min)

Implement `three_sum(arr: list[int]) -> list[tuple[int, int, int]]` — return all unique triplets summing to zero. No duplicate triplets. Time: O(n²).

```python
def three_sum(arr: list[int]) -> list[tuple[int, int, int]]:
    ...

assert sorted(three_sum([-1, 0, 1, 2, -1, -4])) == [(-1, -1, 2), (-1, 0, 1)]
assert three_sum([0, 1, 1]) == []
assert three_sum([0, 0, 0]) == [(0, 0, 0)]
```

---

## Problem 5 — Sliding Window: Longest Substring No Repeat (20 min)

Implement `length_of_longest_substring(s: str) -> int` — LeetCode 3.  
Time: O(n)  Space: O(1) — alphabet is fixed size (at most 128 ASCII chars).

```python
def length_of_longest_substring(s: str) -> int:
    ...

assert length_of_longest_substring("abcabcbb") == 3
assert length_of_longest_substring("bbbbb") == 1
assert length_of_longest_substring("pwwkew") == 3
assert length_of_longest_substring("") == 0
```

---

## Problem 6 — Sliding Window: Minimum Window Substring (25 min)

(Harder) Implement `min_window(s: str, t: str) -> str` — minimum window in `s` containing all chars of `t`. Return `""` if impossible. Time: O(|s| + |t|).

```python
def min_window(s: str, t: str) -> str:
    ...

assert min_window("ADOBECODEBANC", "ABC") == "BANC"
assert min_window("a", "a") == "a"
assert min_window("a", "aa") == ""
```

---

## Problem 7 — Prefix Sums: Range Sum Query (15 min)

Implement a class `RangeSumQuery` with:
- `__init__(self, nums: list[int])` — O(n) preprocessing
- `sum_range(self, left: int, right: int) -> int` — O(1) per query

```python
class RangeSumQuery:
    ...

rsq = RangeSumQuery([-2, 0, 3, -5, 2, -1])
assert rsq.sum_range(0, 2) == 1
assert rsq.sum_range(2, 5) == -1
assert rsq.sum_range(0, 5) == -3
```

---

## Problem 8 — Prefix Sums: Subarray Sum Divisible by K (20 min)

Implement `subarrays_div_by_k(arr: list[int], k: int) -> int` — count subarrays whose sum is divisible by `k`. Time: O(n).

Hint: `sum(arr[l..r]) % k == 0` iff `prefix[r+1] % k == prefix[l] % k`.

```python
def subarrays_div_by_k(arr: list[int], k: int) -> int:
    ...

assert subarrays_div_by_k([4, 5, 0, -2, -3, 1], 5) == 7
assert subarrays_div_by_k([5], 9) == 0
```

---

## Problem 9 — Strings: Valid Anagram Groups with Frequency Key (15 min)

Implement `group_anagrams_fast(words: list[str]) -> list[list[str]]` using a **tuple of 26 integer counts** as the key instead of sorting. Time: O(n·L) where L = max word length.

```python
def group_anagrams_fast(words: list[str]) -> list[list[str]]:
    ...

result = group_anagrams_fast(["eat", "tea", "tan", "ate", "nat", "bat"])
# Each group may be in any order; groups may be in any order
groups_as_sets = {frozenset(g) for g in result}
assert groups_as_sets == {
    frozenset(["eat", "tea", "ate"]),
    frozenset(["tan", "nat"]),
    frozenset(["bat"]),
}
```

---

## Problem 10 — Strings: Longest Palindromic Substring (20 min)

Implement `longest_palindrome(s: str) -> str` using center expansion. If there are multiple longest palindromes, return any one. Time: O(n²)  Space: O(1).

```python
def longest_palindrome(s: str) -> str:
    ...

assert longest_palindrome("babad") in ("bab", "aba")
assert longest_palindrome("cbbd") == "bb"
assert longest_palindrome("a") == "a"
assert longest_palindrome("ac") in ("a", "c")
```

---

## Problem 11 — Combined: Sliding Window on Character Frequencies (25 min)

Implement `min_size_subarray_with_all_chars(s: str, chars: str) -> int` — find the minimum length contiguous substring of `s` that contains at least one occurrence of every character in `chars` (chars may have duplicates, requiring multiple occurrences). Return 0 if impossible. This is a generalisation of min-window from Day 5.

```python
def min_size_subarray_with_all_chars(s: str, chars: str) -> int:
    ...

assert min_size_subarray_with_all_chars("ADOBECODEBANC", "ABC") == 4
assert min_size_subarray_with_all_chars("aa", "aab") == 0
assert min_size_subarray_with_all_chars("aab", "ab") == 2
```

---

## Problem 12 — Design: Annotated Algorithm Library (30 min)

Design and implement a small Python module `algo_lib.py` with the following contract:

```python
# algo_lib.py
# Each public function must have:
#   - type hints
#   - docstring with Time/Space complexity
#   - at least one pytest test in tests/test_algo_lib.py

def two_sum_sorted(arr: list[int], target: int) -> tuple[int, int] | None: ...
def max_sum_window(arr: list[int], k: int) -> int: ...
def range_query(arr: list[int], l: int, r: int) -> int: ...   # O(1) after build
def is_anagram(s: str, t: str) -> bool: ...
```

Then write `tests/test_algo_lib.py` with pytest parametrize for each function. Ensure all tests pass.

---

## Reflection (5 min)

After finishing, write brief answers:

1. Which technique did you find most intuitive? Why?
2. Which problem took longest? What slowed you down?
3. What is the common theme between amortized analysis (Day 2) and the inner-while-loop in sliding window (Day 4)?

*(Solutions: ../solutions/week-01/day-6.md)*

---

## Self-check

- [ ] I completed all 12 problems (or know exactly which steps I need to revisit).
- [ ] I stated complexity before coding for every problem.
- [ ] My `algo_lib.py` has type hints, docstrings, and passing tests.
- [ ] I can articulate the amortized argument for dynamic arrays, stacks, and sliding windows.
