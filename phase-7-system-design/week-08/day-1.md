# Phase 7 · Week 08 · Day 1 — Interview Technique + Two Pointers / Sliding Window

> Objective: Master the structured interview framework and two core array patterns.  ·  Time: ~1 hr  ·  Prereqs: [[phase-2-cs-design/week-02]]

## Concept

### The Interview Framework (always use this order)

Coding interviews test *process* more than raw knowledge. A wrong final answer reached methodically scores higher than a correct one reached mysteriously.

**Five-step protocol — never skip steps:**

1. **Clarify** (2–3 min) — repeat the problem in your own words; ask about constraints, edge cases, input types, expected output type.
2. **Examples** (1–2 min) — walk through 1–2 small hand-crafted inputs; draw the state.
3. **Brute force aloud** (1 min) — state the naive O(n²) solution first; don't code it.
4. **Optimise** — explain *why* the brute force is slow, then apply a pattern.
5. **Complexity + test** — state Big-O time and space; walk through edge cases after coding.

This communicates thought process, catches misunderstandings early, and prevents wasting time coding the wrong thing.

---

### Two Pointers

**When to use:** sorted array or string; pair/triplet sums; palindrome checks; merging sorted sequences.

**Template:**

```python
left, right = 0, len(arr) - 1
while left < right:
    val = compute(arr[left], arr[right])
    if val == target:
        record; left += 1; right -= 1
    elif val < target:
        left += 1
    else:
        right -= 1
```

Both pointers move inward; each element is visited once → **O(n) time, O(1) extra space**.

---

### Sliding Window

**When to use:** contiguous subarray/substring with a constraint (max sum, at most K distinct, no-repeat longest).

Two flavours:

| Flavour | When | Mechanism |
|---|---|---|
| Fixed size | window = k | advance both ends together |
| Variable / shrink | dynamic constraint | expand right, shrink left when violated |

**Variable-size template:**

```python
left = 0
state = {}          # track window contents
best = 0
for right in range(len(s)):
    # expand: add s[right] to state
    while window_invalid(state):
        # shrink: remove s[left] from state
        left += 1
    best = max(best, right - left + 1)
return best
```

---

## How it works internally

**Two pointers on sorted input:** after sorting, for any pair (i, j) where arr[i]+arr[j] < target, every j' < j also gives a sum too small — so we can skip directly by moving the left pointer right. This monotonicity property converts O(n²) search to O(n) scan.

**Sliding window invariant:** the window [left, right] always satisfies the constraint. When we advance `right` we may violate the invariant; we restore it by advancing `left`. Because each index enters and exits the window at most once, the total work is O(n) regardless of the inner while loop.

---

## Worked example

### Problem 1: Two Sum II (sorted array)
*Given sorted array `numbers`, return 1-indexed positions of two numbers that add to `target`.*

```python
# two_pointers.py
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    """Return 1-indexed [left, right] pair summing to target.

    Time: O(n)  Space: O(1)
    """
    left, right = 0, len(numbers) - 1
    while left < right:
        total = numbers[left] + numbers[right]
        logger.debug("left=%d right=%d total=%d", left, right, total)
        if total == target:
            return [left + 1, right + 1]
        elif total < target:
            left += 1
        else:
            right -= 1
    return []          # problem guarantees a solution exists
```

### Problem 2: Longest Substring Without Repeating Characters

```python
# sliding_window.py
def length_of_longest_substring(s: str) -> int:
    """Find longest substring with all unique chars.

    Time: O(n)  Space: O(min(n, alphabet))
    """
    char_index: dict[str, int] = {}
    best = left = 0
    for right, ch in enumerate(s):
        if ch in char_index and char_index[ch] >= left:
            # shrink: jump left past the previous occurrence
            left = char_index[ch] + 1
        char_index[ch] = right
        best = max(best, right - left + 1)
    return best
```

### Problem 3: Minimum Size Subarray Sum

```python
def min_subarray_len(target: int, nums: list[int]) -> int:
    """Shortest contiguous subarray with sum >= target.

    Time: O(n)  Space: O(1)
    Returns 0 if no such subarray.
    """
    left = 0
    window_sum = 0
    min_len = float("inf")
    for right in range(len(nums)):
        window_sum += nums[right]
        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1
    return 0 if min_len == float("inf") else min_len
```

---

## Exercises

1. **Valid Palindrome** — given a string with alphanumeric characters and spaces, determine if it reads the same forwards and backwards (ignore case, ignore non-alphanumeric). Use two pointers. What is the time and space complexity?

2. **Container With Most Water** — given an array `height` of n non-negative integers, find two lines that together with the x-axis form a container holding the most water. Return the maximum area. Implement with two pointers and justify why moving the shorter line inward is always correct.

3. **Longest Subarray With At Most K Zeros** — given a binary array `nums` and integer `k`, return the length of the longest subarray containing at most `k` zeros. Use the sliding window template; handle `k=0` as an edge case.

4. **Three Sum** — given array `nums`, return all unique triplets `[a, b, c]` such that `a + b + c = 0`. Sort first; use two pointers for each fixed first element. How do you avoid duplicate triplets?

(Solutions: ../solutions/week-08/day-1.md)

---

## Self-check

- [ ] I can recite the five-step interview protocol without looking
- [ ] I can implement two-pointer pair-sum on a sorted array from memory
- [ ] I can write the variable-size sliding window template from scratch
- [ ] I can state when each pattern applies and why it beats brute force
- [ ] I correctly compute O(n) time and O(1)/O(alphabet) space for each

---

## Supplementary links

- NeetCode Two Pointers playlist: https://neetcode.io/roadmap (Two Pointers section)
- NeetCode Sliding Window playlist: https://neetcode.io/roadmap (Sliding Window section)
