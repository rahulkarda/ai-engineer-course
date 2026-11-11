# Solutions: Week 08 · Day 1 — Interview Technique + Two Pointers / Sliding Window

## Exercise 1: Valid Palindrome

```python
# valid_palindrome.py
def is_palindrome(s: str) -> bool:
    """Ignore non-alphanumeric characters, case-insensitive.  Time: O(n)  Space: O(1)"""
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
```

**Complexity:** O(n) time — each character visited at most once. O(1) space — no extra data structures; two index pointers only.

**Key insight:** skipping non-alphanumeric characters doesn't require a cleaned copy. Advance each pointer past invalid chars inline — this avoids O(n) allocation.

---

## Exercise 2: Container With Most Water

```python
def max_area(height: list[int]) -> int:
    """Two-pointer squeeze from both ends.  Time: O(n)  Space: O(1)"""
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        area = (right - left) * min(height[left], height[right])
        best = max(best, area)
        # Move the shorter line — moving the taller can only decrease width,
        # and the height is still bounded by the shorter side, so no gain.
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best
```

**Why moving the shorter line is correct (exchange argument):** The area with the current pair is `min(h[l], h[r]) × (r - l)`. If we move the taller pointer instead, width decreases by 1, and the new height is still capped at the shorter side at best — so we cannot improve. Therefore, moving the shorter line is the only chance to find a better solution.

---

## Exercise 3: Longest Subarray With At Most K Zeros

```python
def longest_subarray_k_zeros(nums: list[int], k: int) -> int:
    """Variable sliding window.  Time: O(n)  Space: O(1)"""
    left = zero_count = best = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zero_count += 1
        while zero_count > k:
            if nums[left] == 0:
                zero_count -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

**Edge case k=0:** works without special handling — `zero_count > 0` triggers the while loop immediately on any zero, shrinking the window past it.

---

## Exercise 4: Three Sum

```python
def three_sum(nums: list[int]) -> list[list[int]]:
    """All unique triplets summing to 0.  Time: O(n²)  Space: O(1) output aside"""
    nums.sort()
    result: list[list[int]] = []
    for i, a in enumerate(nums):
        if i > 0 and nums[i] == nums[i - 1]:   # skip duplicate first element
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = a + nums[left] + nums[right]
            if total == 0:
                result.append([a, nums[left], nums[right]])
                left += 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1    # skip duplicate second element
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
```

**Duplicate avoidance:** sort first so duplicates are adjacent. Skip at the `i` level (outer) and at the `left` level (inner after finding a valid triplet). Never need to skip `right` — once `left` advances past duplicates, the pair is unique.

**Complexity:** O(n²) — outer O(n) × inner O(n) two-pointer scan. O(1) extra space beyond output.

---

## Test file

```python
# tests/test_day1.py
import pytest
from day1_solutions import is_palindrome, max_area, longest_subarray_k_zeros, three_sum


@pytest.mark.parametrize("s, expected", [
    ("A man, a plan, a canal: Panama", True),
    ("race a car", False),
    (" ", True),
])
def test_is_palindrome(s: str, expected: bool) -> None:
    assert is_palindrome(s) == expected


@pytest.mark.parametrize("height, expected", [
    ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
    ([1, 1], 1),
    ([4, 3, 2, 1, 4], 16),
])
def test_max_area(height: list[int], expected: int) -> None:
    assert max_area(height) == expected


@pytest.mark.parametrize("nums, k, expected", [
    ([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2, 6),
    ([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], 3, 10),
    ([0, 0, 0], 0, 0),
])
def test_longest_subarray(nums: list[int], k: int, expected: int) -> None:
    assert longest_subarray_k_zeros(nums, k) == expected


def test_three_sum_normal() -> None:
    result = sorted(three_sum([-1, 0, 1, 2, -1, -4]))
    assert result == [[-1, -1, 2], [-1, 0, 1]]


def test_three_sum_duplicates() -> None:
    assert three_sum([0, 0, 0]) == [[0, 0, 0]]


def test_three_sum_no_result() -> None:
    assert three_sum([1, 2, 3]) == []
```
