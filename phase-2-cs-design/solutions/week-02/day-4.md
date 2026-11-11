# Phase 2 · Week 02 · Day 4 — Solutions

## Exercise 1 — Intersection of two lists

```python
"""
Return elements present in both lists, no duplicates in output.
O(n + m) time, O(n) space.
"""
def list_intersection(a: list[int], b: list[int]) -> list[int]:
    seen_a: set[int] = set(a)          # O(n) to build
    result: list[int] = []
    added: set[int] = set()            # prevent duplicates in output
    for val in b:                      # O(m)
        if val in seen_a and val not in added:
            result.append(val)
            added.add(val)
    return result


# Test
assert sorted(list_intersection([1, 2, 2, 3], [2, 2, 3, 4])) == [2, 3]
assert list_intersection([1, 2], [3, 4]) == []
assert list_intersection([], [1, 2]) == []
```

**Why two sets?** `seen_a` gives O(1) membership for the first list; `added` prevents emitting the same value twice when `b` has repeated entries that also exist in `a`.

---

## Exercise 2 — Isomorphic strings

```python
"""
is_isomorphic: check if s and t have a consistent character bijection.
Two mappings needed: s→t AND t→s (bijection, not just injection).
O(n) time and space.
"""
def is_isomorphic(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    s_to_t: dict[str, str] = {}
    t_to_s: dict[str, str] = {}

    for cs, ct in zip(s, t):
        # Check s → t consistency
        if cs in s_to_t:
            if s_to_t[cs] != ct:
                return False
        else:
            s_to_t[cs] = ct

        # Check t → s consistency (ensures bijection, not just injection)
        if ct in t_to_s:
            if t_to_s[ct] != cs:
                return False
        else:
            t_to_s[ct] = cs

    return True


# Tests
assert is_isomorphic("egg", "add") is True
assert is_isomorphic("foo", "bar") is False
assert is_isomorphic("paper", "title") is True
assert is_isomorphic("ab", "aa") is False   # 'a','b' → 'a','a': b conflicts
assert is_isomorphic("ba", "aa") is False   # both 'b' and 'a' would map to 'a'
```

**Why both directions?** Without the reverse map, `is_isomorphic("ab", "aa")` returns True incorrectly — 'a'→'a' and 'b'→'a' is not a bijection.

---

## Exercise 3 — Word frequency from file lines

```python
"""
Count lowercase word frequencies across all lines, stripping punctuation.
O(total characters) time and space.
"""
import string
from collections import defaultdict

def word_frequency(lines: list[str]) -> dict[str, int]:
    freq: dict[str, int] = defaultdict(int)
    for line in lines:
        for word in line.split():
            cleaned = word.strip(string.punctuation).lower()
            if cleaned:                # skip empty strings from stripping
                freq[cleaned] += 1
    return dict(freq)


# Tests
lines = ["Hello, world!", "hello World.", "foo bar. foo"]
result = word_frequency(lines)
assert result["hello"] == 2
assert result["world"] == 2
assert result["foo"] == 2
assert result["bar"] == 1
```

**Note on `string.punctuation`:** Covers `!"#$%&'()*+,-./:;<=>?@[\]^_{|}~\`` — broader than the task's strip set but correct for general text. The `if cleaned` guard handles a token that is entirely punctuation (e.g., `"---"`).

---

## Exercise 4 — Longest consecutive sequence

```python
"""
Find length of longest run of consecutive integers.
Key insight: only start counting from the *beginning* of a run (num-1 not in set).
This ensures each element is visited at most twice → O(n) total.
"""
def longest_consecutive(nums: list[int]) -> int:
    num_set: set[int] = set(nums)
    best = 0

    for num in num_set:
        if num - 1 not in num_set:        # start of a new run
            length = 1
            current = num
            while current + 1 in num_set:
                current += 1
                length += 1
            best = max(best, length)

    return best


# Tests
assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4   # 1,2,3,4
assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
assert longest_consecutive([]) == 0
assert longest_consecutive([1]) == 1
```

**Why not sort?** Sorting costs O(n log n). Iterating the set and only starting a walk at sequence boundaries gives O(n): each integer is "walked through" at most once across all iterations.

---

## Exercise 5 — Subarray sum equals k

```python
"""
Count contiguous subarrays summing to k.
Prefix-sum insight: subarray(i,j) sums to k  iff  prefix[j] - prefix[i-1] == k
                    iff  prefix[i-1] == prefix[j] - k
Store prefix sum counts as we scan; for each new prefix check if (prefix - k) is in the map.
O(n) time, O(n) space.
"""
from collections import defaultdict

def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    count = 0
    prefix_sum = 0
    prefix_counts: dict[int, int] = defaultdict(int)
    prefix_counts[0] = 1          # empty prefix (sum = 0) exists once

    for num in nums:
        prefix_sum += num
        # How many previous prefixes make a valid subarray ending here?
        count += prefix_counts[prefix_sum - k]
        prefix_counts[prefix_sum] += 1

    return count


# Tests
assert subarray_sum_equals_k([1, 1, 1], 2) == 2          # [1,1] at 0..1 and 1..2
assert subarray_sum_equals_k([1, 2, 3], 3) == 2           # [3] and [1,2]
assert subarray_sum_equals_k([1, -1, 0], 0) == 3          # handles negatives
assert subarray_sum_equals_k([], 5) == 0
```

**Trace for `[1,1,1]`, k=2:**

| i | num | prefix_sum | prefix_counts before | count added |
|---|-----|------------|----------------------|-------------|
| - | -   | 0          | {0:1}                | -           |
| 0 | 1   | 1          | {0:1}                | counts[1-2=−1]=0 → +0 |
| 1 | 1   | 2          | {0:1,1:1}            | counts[2-2=0]=1 → +1 |
| 2 | 1   | 3          | {0:1,1:1,2:1}        | counts[3-2=1]=1 → +1 |

Total = 2. Correct.
