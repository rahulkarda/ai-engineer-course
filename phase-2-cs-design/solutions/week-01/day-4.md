# Solutions — Day 4: Sliding Window

## Exercise 1 — count_subarrays_sum_equals_k (positive), and why negative elements break it

```python
def count_subarrays_sum_equals_k_positive(arr: list[int], k: int) -> int:
    """
    Count subarrays with sum == k, all elements positive.
    Time: O(n)  Space: O(1)
    """
    left = 0
    window_sum = 0
    count = 0
    for right in range(len(arr)):
        window_sum += arr[right]
        while window_sum > k and left <= right:
            window_sum -= arr[left]
            left += 1
        if window_sum == k:
            count += 1
    return count
```

**Why negative elements break plain sliding window:**

With negative elements, expanding the window right does not guarantee the sum increases, and shrinking left does not guarantee it decreases. The monotone property is lost. For example, `[1, -1, 1]`, k=1: window [0..2] has sum 1, but shrinking from left would wrongly bypass it. The correct approach for arbitrary elements is the **prefix sum + hash map** technique from Day 5 (`subarray_sum_equals_k`), which handles negatives in O(n).

---

## Exercise 2 — longest_ones_with_k_flips

```python
def longest_ones_with_k_flips(arr: list[int], k: int) -> int:
    """
    Longest subarray of 1s with at most k zeros flipped.
    Time: O(n)  Space: O(1)
    """
    left = 0
    zeros = 0
    best = 0
    for right in range(len(arr)):
        if arr[right] == 0:
            zeros += 1
        while zeros > k:
            if arr[left] == 0:
                zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

---

## Exercise 3 — longest substring without repeating characters

```python
def length_of_longest_substring(s: str) -> int:
    """
    Time: O(n)  Space: O(1) — at most 128 distinct ASCII chars in window
    """
    seen: set[str] = set()
    left = 0
    best = 0
    for right, ch in enumerate(s):
        while ch in seen:
            seen.discard(s[left])
            left += 1
        seen.add(ch)
        best = max(best, right - left + 1)
    return best
```

**Time:** O(n) — each char enters and exits `seen` at most once.  
**Space:** O(min(n, 128)) = O(1) — bounded by alphabet size.

---

## Exercise 4 — fixed_window_max brute force

```python
def fixed_window_max_brute(arr: list[int], k: int) -> list[int]:
    """
    Maximum of each window of size k.
    Time: O(n·k)  Space: O(1) output aside
    """
    if not arr or k > len(arr):
        return []
    result = []
    for i in range(len(arr) - k + 1):
        result.append(max(arr[i: i + k]))    # max of slice = O(k)
    return result
```

For O(n) solution: use a **monotonic deque** (collections.deque). The deque maintains indices of potentially useful elements in decreasing order of value. When the window slides, pop from front if the front index is out of window. Pop from back while back element ≤ new element (it can never be a future max). Front of deque is always the current window max. Covered in Week 2.

---

## Exercise 5 — Amortized proof for inner while loop

Claim: `left` moves at most n times total across the entire loop.

Proof (potential/accounting argument):
- `right` advances from 0 to n-1 → n steps total.
- `left` starts at 0, only ever advances (never goes backward), and never exceeds `right`.
- Therefore `left` advances at most n times total (it starts at 0 and ends at ≤ n-1).
- Each `left += 1` inside the while loop costs O(1).
- Total cost of all while-loop iterations ≤ n.

Combined with the n steps of the outer loop: **2n = O(n) total**. Same as the append analysis: assign credit to each element at entry; spend credit when it's evicted from the left.
