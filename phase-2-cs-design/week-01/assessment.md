# Week 01 Assessment — Complexity Analysis & Arrays/Strings

> Time limit: 60 min  ·  Pass gate: ≥ 75%  ·  No external references during timed section

---

## Part A — Conceptual (25% of total)

Answer each question concisely (2–5 sentences or a table). These mirror real interview questions.

**A1.** Define O, Ω, and Θ in your own words. Give an example function that is Θ(n log n) and explain why it is *not* O(n).

**A2.** A function has these two steps executed sequentially:
- Step 1: O(n log n)
- Step 2: O(n²)

What is the overall time complexity? Explain the rule you applied.

**A3.** Explain why Python's `list.append` is O(1) *amortized* but O(n) *worst case*. Why does the amortized bound matter for algorithm analysis?

**A4.** For each data structure operation, state the time complexity for a Python `list`:

| Operation | Time complexity |
|---|---|
| `lst[i]` | |
| `lst.append(x)` | |
| `lst.insert(0, x)` | |
| `x in lst` | |
| `lst.sort()` | |

**A5.** You need to answer 10,000 range sum queries on a static array of 1,000,000 integers. Compare two approaches: (a) compute each sum naively, (b) precompute a prefix sum array. Give the total time complexity for each approach.

**A6.** A colleague writes a function that calls `list.insert(0, x)` in a loop n times. They claim "each call is O(1) so the loop is O(n)." What is wrong, and what is the correct total complexity?

---

## Part B — Debugging (20% of total)

Each snippet below has a bug — either a correctness bug, a performance bug (wrong complexity), or both. For each:
1. Identify the bug.
2. State the incorrect complexity claim vs. the actual complexity.
3. Fix the code.

**B1.** Claimed: O(n) pair search in sorted array.
```python
def find_pair(arr: list[int], target: int) -> bool:
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == target:
                return True
    return False
```

**B2.** Claimed: O(n) sliding window for longest substring with at most 2 distinct characters.
```python
def longest_two_distinct(s: str) -> int:
    best = 0
    for i in range(len(s)):
        seen: set[str] = set()
        for j in range(i, len(s)):
            seen.add(s[j])
            if len(seen) > 2:
                break
            best = max(best, j - i + 1)
    return best
```

**B3.** Claimed: correct anagram check.
```python
def is_anagram(s: str, t: str) -> bool:
    return sorted(s) == sorted(t)

# Used as:
print(is_anagram("rat", "tar"))    # True  (correct)
print(is_anagram("rat", "tare"))   # True  (BUG — should be False)
```

---

## Part C — Coding Challenge (40% of total)

Implement the following three functions. Each must have type hints, a docstring with complexity, and pass the provided assertions.

**C1.** Two-pointer: find all pairs summing to target (return all pairs, no duplicates).

```python
def all_pairs_with_sum(arr: list[int], target: int) -> list[tuple[int, int]]:
    """
    Return all unique pairs (a, b) where a + b == target, a ≤ b.
    arr may be unsorted and may contain duplicates.
    Time: O(?)  Space: O(?)
    """
    ...

assert sorted(all_pairs_with_sum([1, 5, 3, 7, 2, 9, 3], 8)) == [(1, 7), (3, 5)]
assert all_pairs_with_sum([1, 1, 1], 2) == [(1, 1)]
assert all_pairs_with_sum([], 5) == []
```

**C2.** Sliding window + prefix sum: find the number of subarrays with product < k.

```python
def count_subarrays_product_less_than_k(arr: list[int], k: int) -> int:
    """
    Count contiguous subarrays whose product of elements < k.
    All elements are positive integers.
    Time: O(?)  Space: O(?)
    """
    ...

assert count_subarrays_product_less_than_k([10, 5, 2, 6], 100) == 8
assert count_subarrays_product_less_than_k([1, 2, 3], 0) == 0
assert count_subarrays_product_less_than_k([1, 1, 1], 2) == 6
```

**C3.** String algorithm: given a string, find the length of the longest substring that is a palindrome using a sliding window or center expansion. Return both the length and the substring.

```python
def longest_palindrome_details(s: str) -> tuple[int, str]:
    """
    Return (length, substring) of the longest palindromic substring.
    Time: O(?)  Space: O(?)
    """
    ...

length, substr = longest_palindrome_details("racecar")
assert length == 7 and substr == "racecar"

length, substr = longest_palindrome_details("abacaba")
assert length == 7

length, substr = longest_palindrome_details("abcd")
assert length == 1
```

---

## Part D — Code Quality (15% of total)

Review your Part C solutions and self-assess against these criteria (1 point each, max 6):

- [ ] All functions have accurate type hints (`list[int]`, `tuple[int, int] | None`, etc.)
- [ ] All docstrings include **Time:** and **Space:** annotations.
- [ ] No hardcoded magic constants without explanation (e.g., `26` should have a comment `# lowercase ASCII alphabet`).
- [ ] No `print()` used — logging or return values only.
- [ ] Edge cases handled: empty input, k=0, single-character string.
- [ ] Code is readable: meaningful variable names, no single-letter names except loop counters (`i`, `j`, `l`, `r`).

---

## Rubric

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation (Part C) | 40% | Correct output + correct complexity claimed |
| Conceptual understanding (Part A) | 25% | Precise definitions, correct examples |
| Debugging & problem-solving (Part B) | 20% | Bug identified + correct fix |
| Code quality (Part D) | 15% | 4+ of 6 checklist items |

**Pass gate: ≥ 75% total.**

Partial credit within each part: half credit for a correct approach with one implementation error.

Answer key: ../solutions/week-01/assessment.md
