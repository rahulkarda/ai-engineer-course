# Solutions — Day 3: Two Pointers

## Exercise 1 — three_sum_sorted

```python
def three_sum_sorted(arr: list[int], target: int) -> list[tuple[int, int, int]]:
    """
    Find all unique triplets summing to target in a sorted array.
    Time: O(n²)  Space: O(1) output aside
    """
    arr.sort()
    result: list[tuple[int, int, int]] = []
    n = len(arr)
    for i in range(n - 2):
        if i > 0 and arr[i] == arr[i - 1]:   # skip duplicates for first element
            continue
        left, right = i + 1, n - 1
        while left < right:
            total = arr[i] + arr[left] + arr[right]
            if total == target:
                result.append((arr[i], arr[left], arr[right]))
                while left < right and arr[left] == arr[left + 1]:
                    left += 1
                while left < right and arr[right] == arr[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < target:
                left += 1
            else:
                right -= 1
    return result
```

**Complexity:** Outer loop O(n), inner two-pointer O(n) → **O(n²) total**. Sorting is O(n log n) but dominated.

---

## Exercise 2 — valid_after_one_removal

```python
def valid_after_one_removal(arr: list[int]) -> bool:
    """
    Can remove at most one element to make array non-decreasing?
    Time: O(n)  Space: O(1)
    """
    def is_nondecreasing(a: list[int], skip: int) -> bool:
        prev = -float("inf")
        for i, v in enumerate(a):
            if i == skip:
                continue
            if v < prev:
                return False
            prev = v
        return True

    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:                 # first violation at i
            return is_nondecreasing(arr, i) or is_nondecreasing(arr, i + 1)
    return True                                  # already non-decreasing
```

---

## Exercise 3 — sorted_squares

```python
def sorted_squares(arr: list[int]) -> list[int]:
    """
    Return sorted squares of a sorted (possibly negative) array.
    Time: O(n)  Space: O(n)
    """
    n = len(arr)
    result = [0] * n
    left, right = 0, n - 1
    pos = n - 1                        # fill from the end (largest first)
    while left <= right:
        lsq, rsq = arr[left] ** 2, arr[right] ** 2
        if lsq > rsq:
            result[pos] = lsq
            left += 1
        else:
            result[pos] = rsq
            right -= 1
        pos -= 1
    return result
```

**Key insight:** the largest square is always at one of the two ends of a sorted array (most negative or most positive). Converging pointers always pick the larger square and fill from the back.

---

## Exercise 4 — longest_subarray_sum_k (positive elements)

```python
def longest_subarray_sum_k(arr: list[int], k: int) -> int:
    """
    Longest subarray with sum exactly k, all elements positive.
    Time: O(n)  Space: O(1)
    """
    left = 0
    window_sum = 0
    best = 0
    for right in range(len(arr)):
        window_sum += arr[right]
        while window_sum > k and left <= right:
            window_sum -= arr[left]
            left += 1
        if window_sum == k:
            best = max(best, right - left + 1)
    return best
```

**Why positive elements only:** With positive elements, expanding right always increases the sum and shrinking left always decreases it — the sum is **monotone** in both directions. This monotonicity is what allows the `while > k: shrink` logic to be correct. With negative elements, shrinking might decrease the sum below k and miss a valid window that wrapping around a negative would create.

---

## Exercise 5 — container_with_most_water proof sketch

Moving the taller wall cannot improve the result: the area is limited by `min(heights[left], heights[right]) × width`. The current area is `min(H_L, H_R) × W`. If `H_L < H_R` (left is shorter), moving right inward decreases width but keeps the height bottleneck (left) the same or worse → area cannot increase. Only moving the *shorter* wall can potentially find a taller wall that exceeds the bottleneck. Therefore, always move the shorter wall.

---

## Exercise 6 — remove_element_inplace

```python
def remove_element_inplace(arr: list[int], val: int) -> int:
    """
    Remove all occurrences of val in-place, return new length.
    Time: O(n)  Space: O(1)
    """
    slow = 0
    for fast in range(len(arr)):
        if arr[fast] != val:
            arr[slow] = arr[fast]
            slow += 1
    return slow
```
