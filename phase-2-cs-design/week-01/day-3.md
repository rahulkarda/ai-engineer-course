# Phase 2 · Week 01 · Day 3 — Two Pointers

> Objective: Recognize problems solvable with two pointers and implement them in O(n) time.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]]

---

## Concept

**Two pointers** is an array technique that maintains two indices into the same array (or two separate arrays) and moves them toward each other or in the same direction, eliminating the need for a nested loop.

### When to reach for two pointers

- Input is sorted **or** order doesn't matter for the invariant you maintain.
- You're searching for a pair/triplet satisfying a condition (sum, difference, product).
- You need to partition, remove duplicates, or merge in-place.
- The brute-force is O(n²) and you can reduce to O(n) by using ordering.

### Two canonical patterns

**Pattern 1 — Converging pointers** (left starts at 0, right at n−1, they move inward):
```
[←L ......................... R→]  squeeze toward center
```
Used for: pair sum = target, palindrome check, container with most water.

**Pattern 2 — Same-direction (fast/slow or read/write)** (both start at 0 or 0/1, move right):
```
[S→    F→                       ]  slow writes, fast reads
```
Used for: remove duplicates in-place, move zeros, linked-list cycle detection.

### Complexity

Both patterns visit each element at most once → **O(n) time, O(1) extra space**.

---

## How it works internally

**Why does converging work for "pair with target sum" on a sorted array?**

Invariant: at any point, no pair *outside* [L, R] can be the answer.

- If `arr[L] + arr[R] == target` → found.
- If `arr[L] + arr[R] < target` → moving `R` left makes the sum smaller (worse). Must move `L` right.
- If `arr[L] + arr[R] > target` → moving `L` right makes the sum bigger (worse). Must move `R` left.

This "ruling out" argument means we never miss a valid pair and never revisit elements → O(n) is provable, not just observed.

**Why does fast/slow work for remove-duplicates in-place?**

`slow` tracks the write position (boundary of the "clean" prefix). `fast` scans ahead. When `fast` finds a new (non-duplicate) value, it writes it to `slow` and advances `slow`. Elements between `slow` and `fast` are "garbage" that will be overwritten or ignored.

---

## Worked example

```python
"""
two_pointers.py — four canonical two-pointer problems
Python 3.12+
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Pattern 1: Converging ──────────────────────────────────────────────────

def pair_with_sum(arr: list[int], target: int) -> tuple[int, int] | None:
    """
    Return indices (i, j) where arr[i] + arr[j] == target, i < j.
    Requires: arr is sorted.
    Time: O(n)  Space: O(1)
    """
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return (left, right)
        elif current < target:
            left += 1      # need bigger sum → move left pointer right
        else:
            right -= 1     # need smaller sum → move right pointer left
    return None


def is_palindrome(s: str) -> bool:
    """
    Check if s is a palindrome ignoring non-alphanumeric chars and case.
    Time: O(n)  Space: O(1)
    """
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True


def container_with_most_water(heights: list[int]) -> int:
    """
    Return maximum water trapped between two vertical lines.
    Time: O(n)  Space: O(1)
    """
    left, right = 0, len(heights) - 1
    max_water = 0
    while left < right:
        width = right - left
        water = width * min(heights[left], heights[right])
        max_water = max(max_water, water)
        # move the shorter wall — it's the bottleneck
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return max_water


# ── Pattern 2: Same direction ─────────────────────────────────────────────

def remove_duplicates_inplace(arr: list[int]) -> int:
    """
    Remove duplicates from sorted arr in-place.
    Returns new length; first k elements hold unique values.
    Time: O(n)  Space: O(1)
    """
    if not arr:
        return 0
    slow = 0                     # slow = write position
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:   # found new unique value
            slow += 1
            arr[slow] = arr[fast]    # write it
    return slow + 1


def move_zeros_to_end(arr: list[int]) -> None:
    """
    Move all zeros to end of arr in-place, preserving relative order of non-zeros.
    Time: O(n)  Space: O(1)
    """
    slow = 0                          # slow = next write position for non-zeros
    for fast in range(len(arr)):
        if arr[fast] != 0:
            arr[slow] = arr[fast]
            slow += 1
    # fill remainder with zeros
    for i in range(slow, len(arr)):
        arr[i] = 0


if __name__ == "__main__":
    log.info("pair_with_sum([1,3,5,7,9], 12): %s", pair_with_sum([1, 3, 5, 7, 9], 12))
    log.info("pair_with_sum([1,2,3], 10): %s", pair_with_sum([1, 2, 3], 10))

    log.info("is_palindrome('A man a plan a canal Panama'): %s",
             is_palindrome("A man a plan a canal Panama"))
    log.info("is_palindrome('race a car'): %s", is_palindrome("race a car"))

    log.info("container_with_most_water([1,8,6,2,5,4,8,3,7]): %d",
             container_with_most_water([1, 8, 6, 2, 5, 4, 8, 3, 7]))

    arr = [0, 0, 1, 1, 2, 3, 3, 4]
    k = remove_duplicates_inplace(arr)
    log.info("remove_duplicates → len=%d, arr=%s", k, arr[:k])

    arr2 = [0, 1, 0, 3, 12]
    move_zeros_to_end(arr2)
    log.info("move_zeros: %s", arr2)
```

**Trace `pair_with_sum([1,3,5,7,9], 12)`:**

```
L=0 R=4  sum=1+9=10 < 12 → L++
L=1 R=4  sum=3+9=12 == 12 → return (1,4)
```

Two steps instead of the 10 comparisons a brute-force O(n²) would take.

---

## Exercises

1. Implement `three_sum_sorted(arr: list[int], target: int) -> list[tuple[int,int,int]]` — find all unique triplets in a sorted array that sum to `target`. Use two pointers for the inner loop. What is the overall time complexity?

2. Given a sorted array with at most one element to remove to make it non-decreasing, use two pointers to find whether it is possible:
   ```python
   def valid_after_one_removal(arr: list[int]) -> bool: ...
   ```

3. Implement `sorted_squares(arr: list[int]) -> list[int]` — given a sorted array (may include negatives), return a sorted array of their squares in O(n) time using converging pointers.

4. Implement `longest_subarray_with_sum_k` *without* a hash map, using two same-direction pointers, assuming all elements are positive. What property makes this work?

5. In `container_with_most_water`, why is moving the *taller* wall suboptimal? Write a one-paragraph proof sketch.

6. Write `remove_element_inplace(arr: list[int], val: int) -> int` that removes all occurrences of `val` in-place and returns the new length. O(n) time, O(1) space.

*(Solutions: ../solutions/week-01/day-3.md)*

---

## Self-check

- [ ] I can identify "converging" vs "same-direction" pointer setups from a problem description.
- [ ] I can implement `pair_with_sum` from scratch without reference.
- [ ] I can explain why two pointers reduces O(n²) brute force to O(n).
- [ ] I know which two-pointer problems require the input to be sorted.

---

## Supplementary links

- [LeetCode Two Pointers tag](https://leetcode.com/tag/two-pointers/)
- [Neetcode — Two Pointers playlist](https://neetcode.io/roadmap)
- [CLRS § 2.1 — Insertion sort correctness (same invariant style)](https://en.wikipedia.org/wiki/Introduction_to_Algorithms)
