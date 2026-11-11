# Solutions — Day 6: Mixed Problem Set

## Problem 1 — Complexity Classification

**(a) `f(n)`:** `i` doubles each iteration → runs ⌊log₂n⌋ + 1 times.
- **Time: O(log n)** — loop halves search space each iteration.
- **Space: O(1)** — only `result` and `i`.

**(b) `g(arr)`:** list comprehension iterates all n elements.
- **Time: O(n)** — single pass.
- **Space: O(n)** — output list can hold up to n elements.

**(c) `h(n)`:** makes two recursive calls to `h(n//2)`. Recurrence: T(n) = 2T(n/2) + O(1).  
Master Theorem: a=2, b=2, c=0. log₂(2)=1 > 0 → Case 1 → **T(n) = Θ(n)**.  
Note: even though there are two calls, both solve the *same* subproblem `h(n//2)` — but Python does not memoize, so both execute. The call tree has O(n) nodes total.
- **Time: O(n)** — Θ(n) nodes in the recursion tree.
- **Space: O(log n)** — maximum call stack depth (halving).

**(d) `k(arr)`:** inner loop starts at `i`, so iterations = n + (n-1) + … + 1 = n(n+1)/2.
- **Time: O(n²)** — triangular sum.
- **Space: O(1)** — only `total`.

---

## Problem 2 — Amortized DynamicStack

**(a)** Worst-case single `push`: O(n) — occurs when doubling, copies all n existing elements.

**(b)** Amortized O(1) over n pushes — same geometric series argument as Day 2: total copies ≤ 2n.

**(c)** If you halved at 1/2 capacity (instead of 1/4), a sequence `push, pop, push, pop, ...` would trigger resize on every single operation — O(n) per operation for a sequence of n ops → O(n²) total. This is the "ping-pong" or "thrashing" problem. Using 1/4 as the shrink threshold creates a buffer: after a shrink to half-capacity, you can do n/4 more pops before the next shrink, amortizing the cost.

---

## Problem 3 — Sort Colors (Dutch National Flag)

```python
def sort_colors(arr: list[int]) -> None:
    """
    In-place partition: 0s | 1s | 2s.
    Time: O(n)  Space: O(1)
    """
    low, mid, high = 0, 0, len(arr) - 1
    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:   # arr[mid] == 2
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
            # do NOT increment mid — newly swapped value unchecked
```

---

## Problem 4 — Three Sum

```python
def three_sum(arr: list[int]) -> list[tuple[int, int, int]]:
    """Time: O(n²)  Space: O(1) output aside."""
    arr.sort()
    result: list[tuple[int, int, int]] = []
    for i in range(len(arr) - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        left, right = i + 1, len(arr) - 1
        while left < right:
            s = arr[i] + arr[left] + arr[right]
            if s == 0:
                result.append((arr[i], arr[left], arr[right]))
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                left += 1; right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result
```

---

## Problem 5 — Longest Substring No Repeat

```python
def length_of_longest_substring(s: str) -> int:
    """Time: O(n)  Space: O(1) — bounded by 128 ASCII."""
    last_seen: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1    # jump left past the duplicate
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

Optimised: instead of shrinking one step at a time, jump directly past the last occurrence. Same O(n) result, fewer iterations in practice.

---

## Problem 6 — Minimum Window Substring

See Day 5 worked example `min_window_substring` — identical algorithm. Key points:
- `need` dict counts required chars.
- `formed` counter tracks how many distinct chars have met their requirement.
- Shrink while `formed == required`; track best window.
- Time: O(|s| + |t|)  Space: O(|s| + |t|)

---

## Problem 7 — RangeSumQuery

```python
class RangeSumQuery:
    """Time: O(n) init, O(1) query.  Space: O(n)."""

    def __init__(self, nums: list[int]) -> None:
        self._prefix = [0] * (len(nums) + 1)
        for i, v in enumerate(nums):
            self._prefix[i + 1] = self._prefix[i] + v

    def sum_range(self, left: int, right: int) -> int:
        return self._prefix[right + 1] - self._prefix[left]
```

---

## Problem 8 — Subarrays Divisible by K

```python
from collections import defaultdict

def subarrays_div_by_k(arr: list[int], k: int) -> int:
    """Time: O(n)  Space: O(k)."""
    freq: dict[int, int] = defaultdict(int)
    freq[0] = 1
    prefix_sum = 0
    count = 0
    for val in arr:
        prefix_sum += val
        mod = prefix_sum % k
        # Python modulo is always non-negative for positive k
        count += freq[mod]
        freq[mod] += 1
    return count
```

---

## Problem 9 — group_anagrams_fast

```python
from collections import defaultdict

def group_anagrams_fast(words: list[str]) -> list[list[str]]:
    """Time: O(n·L)  Space: O(n·L)."""
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for word in words:
        freq = [0] * 26   # lowercase ASCII alphabet
        for ch in word:
            freq[ord(ch) - ord('a')] += 1
        groups[tuple(freq)].append(word)
    return list(groups.values())
```

---

## Problem 10 — Longest Palindromic Substring

```python
def longest_palindrome(s: str) -> str:
    """Time: O(n²)  Space: O(1)."""
    if not s:
        return ""
    best_start = best_end = 0

    def expand(l: int, r: int) -> None:
        nonlocal best_start, best_end
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l > best_end - best_start:
                best_start, best_end = l, r
            l -= 1; r += 1

    for i in range(len(s)):
        expand(i, i)
        expand(i, i + 1)
    return s[best_start: best_end + 1]
```

---

## Problem 11 — min_size_subarray_with_all_chars

```python
from collections import defaultdict

def min_size_subarray_with_all_chars(s: str, chars: str) -> int:
    """
    Min length substring containing all chars (with multiplicity).
    Returns 0 if impossible.
    Time: O(n + |chars|)  Space: O(|chars|)
    """
    if not chars:
        return 0
    need: dict[str, int] = defaultdict(int)
    for ch in chars:
        need[ch] += 1
    have: dict[str, int] = defaultdict(int)
    formed = 0
    required = len(need)
    left = 0
    best = float("inf")
    for right, ch in enumerate(s):
        have[ch] += 1
        if ch in need and have[ch] == need[ch]:
            formed += 1
        while formed == required:
            best = min(best, right - left + 1)
            lch = s[left]
            have[lch] -= 1
            if lch in need and have[lch] < need[lch]:
                formed -= 1
            left += 1
    return 0 if best == float("inf") else int(best)
```

---

## Problem 12 — algo_lib.py and tests

```python
# algo_lib.py
"""
algo_lib — small annotated algorithm library.
Python 3.12+
"""
from collections import defaultdict


def two_sum_sorted(arr: list[int], target: int) -> tuple[int, int] | None:
    """
    Find (i, j) where arr[i] + arr[j] == target. arr must be sorted.
    Time: O(n)  Space: O(1)
    """
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)
        elif s < target:
            left += 1
        else:
            right -= 1
    return None


def max_sum_window(arr: list[int], k: int) -> int:
    """
    Maximum sum of any contiguous subarray of length k.
    Time: O(n)  Space: O(1)
    """
    if len(arr) < k:
        raise ValueError("array shorter than window")
    w = sum(arr[:k])
    best = w
    for i in range(k, len(arr)):
        w += arr[i] - arr[i - k]
        best = max(best, w)
    return best


def range_query(arr: list[int], l: int, r: int) -> int:
    """
    Sum of arr[l..r] inclusive, using internal prefix build.
    Time: O(n) build + O(1) per query.  Space: O(n)
    Note: for repeated queries on the same array, build prefix once externally.
    """
    prefix = [0] * (len(arr) + 1)
    for i, v in enumerate(arr):
        prefix[i + 1] = prefix[i] + v
    return prefix[r + 1] - prefix[l]


def is_anagram(s: str, t: str) -> bool:
    """
    Check if s and t are anagrams.
    Time: O(n)  Space: O(1) — bounded by alphabet size
    """
    if len(s) != len(t):
        return False
    freq: dict[str, int] = defaultdict(int)
    for a, b in zip(s, t):
        freq[a] += 1
        freq[b] -= 1
    return all(v == 0 for v in freq.values())
```

```python
# tests/test_algo_lib.py
import pytest
from algo_lib import two_sum_sorted, max_sum_window, range_query, is_anagram


@pytest.mark.parametrize("arr,target,expected", [
    ([1, 3, 5, 7, 9], 12, (2, 4)),  # 5+7
    ([1, 2, 3], 10, None),
    ([], 5, None),
    ([2, 4], 6, (0, 1)),
])
def test_two_sum_sorted(arr, target, expected):
    assert two_sum_sorted(arr, target) == expected


@pytest.mark.parametrize("arr,k,expected", [
    ([2, 1, 5, 1, 3, 2], 3, 9),
    ([1, 1, 1], 1, 1),
    ([4, 4, 4, 4], 4, 16),
])
def test_max_sum_window(arr, k, expected):
    assert max_sum_window(arr, k) == expected


def test_max_sum_window_too_short():
    with pytest.raises(ValueError):
        max_sum_window([1], 3)


@pytest.mark.parametrize("arr,l,r,expected", [
    ([-2, 0, 3, -5, 2, -1], 0, 2, 1),
    ([-2, 0, 3, -5, 2, -1], 2, 5, -1),
    ([1, 2, 3, 4, 5], 1, 3, 9),
])
def test_range_query(arr, l, r, expected):
    assert range_query(arr, l, r) == expected


@pytest.mark.parametrize("s,t,expected", [
    ("listen", "silent", True),
    ("rat", "tar", True),
    ("rat", "tare", False),
    ("", "", True),
    ("a", "b", False),
])
def test_is_anagram(s, t, expected):
    assert is_anagram(s, t) == expected
```

## Problem 12 — Reflection answers

**1. Most intuitive:** Prefix sums — the "bank account" analogy is immediate: prefix[i] is a running total, range query is just a balance check.

**2. Slowest problem:** Usually min-window-substring — tracking `formed` requires careful bookkeeping of when a character's count is "satisfied".

**3. Common theme:** Both amortized append (Day 2) and sliding window inner-while (Day 4) use the same accounting argument: each element "pays" for its own removal when it eventually exits. The total removal cost across all elements is O(n), so the amortized per-element cost is O(1).
