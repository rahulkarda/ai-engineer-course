# Phase 2 · Week 01 · Day 4 — Sliding Window

> Objective: Apply fixed and variable sliding window patterns to solve subarray/substring problems in O(n).  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

**Sliding window** maintains a contiguous subarray (or substring) that satisfies a constraint as a "window", expanding or contracting it instead of recomputing from scratch.

### Fixed-size window

Window size `k` is constant. Slide right by adding the new element and removing the leftmost.  
Useful for: moving average, max/min of every k elements, number of distinct chars in window of k.

```
arr = [2, 1, 5, 1, 3, 2]   k = 3
[2, 1, 5] → sum=8
   [1, 5, 1] → sum=7  (remove 2, add 1)
      [5, 1, 3] → sum=9  (remove 1, add 3)
```

Each element enters and exits exactly once → O(n).

### Variable-size window

Window expands when the constraint is satisfied, contracts from the left when violated.  
Classic structure:

```python
left = 0
for right in range(len(arr)):
    # include arr[right] in window state
    while window_is_invalid():
        # remove arr[left] from state
        left += 1
    # window [left..right] is valid — update answer
```

**Why O(n)?** `right` advances n times; `left` advances at most n times (never goes past `right`) → 2n total operations.

### When to use sliding window

| Signal | Pattern |
|---|---|
| "subarray/substring of size k" | Fixed window |
| "longest/shortest subarray where …" | Variable window |
| "minimum window containing all chars" | Variable window with frequency map |
| All elements are positive (or constraints are monotone) | Variable window works |

---

## How it works internally

The key insight: if you can express window validity as an **incrementally updatable state**, you can expand/contract in O(1) per step.

- Sum window: state = running sum. Add right element, subtract left element. O(1) update.
- Distinct count window: state = frequency dict + count of keys with freq > 0. O(1) update.
- Max in window: needs a **monotonic deque** for O(1) max — covered in Week 2.

For variable windows, the inner `while` loop *looks* like O(n²) but `left` only ever moves right — across the entire run it moves at most n times. This amortized argument (same as Day 2's append analysis) gives O(n) total.

---

## Worked example

```python
"""
sliding_window.py — fixed and variable window examples
Python 3.12+
"""
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Fixed-size window ────────────────────────────────────────────────────

def max_sum_subarray_k(arr: list[int], k: int) -> int:
    """
    Maximum sum of any contiguous subarray of length k.
    Time: O(n)  Space: O(1)
    """
    if len(arr) < k:
        raise ValueError(f"Array length {len(arr)} < k={k}")
    window_sum = sum(arr[:k])          # first window
    best = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]   # slide: add right, drop left
        best = max(best, window_sum)
    return best


def moving_average(nums: list[float], k: int) -> list[float]:
    """
    Return list of moving averages of window size k.
    Time: O(n)  Space: O(k) (the window)
    """
    if k > len(nums):
        return []
    result: list[float] = []
    window_sum = sum(nums[:k])
    result.append(window_sum / k)
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        result.append(window_sum / k)
    return result


# ── Variable-size window ─────────────────────────────────────────────────

def longest_subarray_sum_le_k(arr: list[int], k: int) -> int:
    """
    Longest contiguous subarray whose sum ≤ k.  (All elements ≥ 0.)
    Time: O(n)  Space: O(1)
    """
    left = 0
    window_sum = 0
    best = 0
    for right in range(len(arr)):
        window_sum += arr[right]                      # expand window
        while window_sum > k and left <= right:       # shrink if invalid
            window_sum -= arr[left]
            left += 1
        best = max(best, right - left + 1)
    return best


def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Length of longest substring with at most k distinct characters.
    Time: O(n)  Space: O(k)
    """
    freq: dict[str, int] = defaultdict(int)
    left = 0
    best = 0
    for right, ch in enumerate(s):
        freq[ch] += 1
        while len(freq) > k:                # too many distinct → shrink
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


def min_window_substring(s: str, t: str) -> str:
    """
    Minimum window in s containing all characters of t (with multiplicity).
    Time: O(|s| + |t|)  Space: O(|s| + |t|)
    LeetCode 76.
    """
    if not t:
        return ""
    need: dict[str, int] = defaultdict(int)
    for ch in t:
        need[ch] += 1
    have: dict[str, int] = defaultdict(int)
    formed = 0                                   # chars with satisfied count
    required = len(need)                         # distinct chars needed
    left = 0
    best_len = float("inf")
    best_start = 0
    for right, ch in enumerate(s):
        have[ch] += 1
        if ch in need and have[ch] == need[ch]:
            formed += 1
        while formed == required:               # valid window — try to shrink
            if right - left + 1 < best_len:
                best_len = right - left + 1
                best_start = left
            left_ch = s[left]
            have[left_ch] -= 1
            if left_ch in need and have[left_ch] < need[left_ch]:
                formed -= 1
            left += 1
    return "" if best_len == float("inf") else s[best_start: best_start + int(best_len)]


if __name__ == "__main__":
    log.info("max_sum_k=3 in [2,1,5,1,3,2]: %d",
             max_sum_subarray_k([2, 1, 5, 1, 3, 2], 3))

    log.info("moving_average k=3: %s",
             [round(x, 2) for x in moving_average([1, 3, 5, 7, 9], 3)])

    log.info("longest_subarray_sum<=7 in [3,1,2,7,4,2,1,1,5]: %d",
             longest_subarray_sum_le_k([3, 1, 2, 7, 4, 2, 1, 1, 5], 7))

    log.info("longest_substr 2-distinct 'eceba': %d",
             longest_substring_k_distinct("eceba", 2))

    log.info("min_window_substring('ADOBECODEBANC', 'ABC'): %r",
             min_window_substring("ADOBECODEBANC", "ABC"))
```

**Trace `min_window_substring("ADOBECODEBANC", "ABC")`:**

Window expands right until it contains A, B, C. Then contracts from left while still valid.  
Best window found: `"BANC"` (length 4).

---

## Exercises

1. Implement `count_subarrays_sum_equals_k(arr: list[int], k: int) -> int` using sliding window (assume all positive). Then explain why you *cannot* use plain sliding window if elements can be negative (hint: what breaks the monotone property?).

2. Implement `longest_ones_with_k_flips(arr: list[int], k: int) -> int` — given a binary array, find the longest subarray of 1s you can achieve by flipping at most `k` zeros.

3. Given `s = "pwwkew"`, find the longest substring without repeating characters. Implement it with a sliding window + set. Time and space complexity?

4. Implement `fixed_window_max(arr: list[int], k: int) -> list[int]` using a brute-force O(n·k) approach first, then explain what data structure would give O(n) (you don't need to implement the O(n) version yet — that's Week 2).

5. The inner `while` loop in `longest_subarray_sum_le_k` looks O(n) per iteration making it O(n²) total. Prove it is actually O(n) total using an amortized argument identical to the one from Day 2.

*(Solutions: ../solutions/week-01/day-4.md)*

---

## Self-check

- [ ] I can distinguish fixed-window from variable-window problems by problem statement.
- [ ] I can implement the variable-window template from memory: `left=0, for right: expand; while invalid: shrink`.
- [ ] I can prove the variable-window is O(n) total with an amortized argument.
- [ ] I can explain what "incrementally updatable state" means and why it matters.

---

## Supplementary links

- [Sliding Window — Neetcode video](https://neetcode.io/courses/advanced-algorithms/1)
- [LeetCode Sliding Window tag](https://leetcode.com/tag/sliding-window/)
- [Monotonic deque for window max (preview)](https://en.wikipedia.org/wiki/Deque)
