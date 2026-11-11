# Phase 2 · Week 01 · Day 5 — Prefix Sums & String Algorithms

> Objective: Use prefix sum arrays for O(1) range queries; implement core string algorithms (anagrams, palindromes, pattern matching).  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

### Prefix sums

A **prefix sum array** `P` where `P[i] = arr[0] + arr[1] + ... + arr[i-1]` (using a sentinel `P[0] = 0`) enables O(1) range sum queries:

```
sum(arr[l..r]) = P[r+1] - P[l]
```

Build once in O(n), answer unlimited range queries in O(1) each.

**2-D prefix sums** extend this to matrices: `sum(submatrix)` in O(1) after O(n·m) build.

### String algorithms

Three fundamental problems:

| Problem | Naive | Optimized |
|---|---|---|
| Anagram detection | O(n log n) sort | O(n) frequency count |
| Palindrome check | O(n) | O(n) two pointers |
| Pattern matching | O(n·m) brute force | O(n+m) KMP |

**Anagram:** two strings are anagrams iff they have identical character frequency maps.

**Palindrome:** string reads the same forwards and backwards. Center expansion or two pointers.

**KMP (Knuth-Morris-Pratt):** precompute a "failure function" (partial match table) on the pattern to avoid rescanning already-matched characters. O(n+m) total vs O(n·m) brute force.

---

## How it works internally

### Prefix sum derivation

```
arr = [3, -2, 5,  1, -4]
P   = [0,  3,  1,  6,  7,  3]
       ^sentinel

sum(arr[1..3]) = P[4] - P[1] = 7 - 3 = 4
verify: -2 + 5 + 1 = 4 ✓
```

The trick: every range query becomes a subtraction of two O(1) lookups.

### KMP failure function

The failure function `f[i]` = length of the longest proper prefix of `pattern[0..i]` that is also a suffix.

```
pattern  = "AABAAB"
indices:    0 1 2 3 4 5
f[]      =  0 1 0 1 2 3
```

During matching, when a mismatch occurs at `pattern[j]`, instead of restarting at `j=0` we jump to `j = f[j-1]` — reusing the already-matched prefix. This guarantees each character in text is processed at most twice.

### Hashing for anagram groups

`collections.Counter` is a frequency hash map. Two strings are anagrams iff their `Counter` objects are equal. For grouping: use the sorted-characters string as a canonical key — `"".join(sorted(word))`.

---

## Worked example

```python
"""
prefix_strings.py — prefix sums, anagrams, palindromes, KMP
Python 3.12+
"""
import logging
from collections import Counter, defaultdict

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Prefix Sums ───────────────────────────────────────────────────────────

def build_prefix(arr: list[int]) -> list[int]:
    """Build prefix sum array with sentinel P[0]=0. Time: O(n) Space: O(n)."""
    prefix = [0] * (len(arr) + 1)
    for i, val in enumerate(arr):
        prefix[i + 1] = prefix[i] + val
    return prefix


def range_sum(prefix: list[int], left: int, right: int) -> int:
    """Sum of arr[left..right] inclusive. Time: O(1)."""
    return prefix[right + 1] - prefix[left]


def subarray_sum_equals_k(arr: list[int], k: int) -> int:
    """
    Count subarrays with sum == k (elements may be negative).
    Uses prefix sum + hash map.
    Time: O(n)  Space: O(n)
    """
    count = 0
    prefix_sum = 0
    freq: dict[int, int] = defaultdict(int)
    freq[0] = 1                                   # empty prefix
    for val in arr:
        prefix_sum += val
        count += freq[prefix_sum - k]             # how many prior prefixes differ by k
        freq[prefix_sum] += 1
    return count


# ── String algorithms ─────────────────────────────────────────────────────

def are_anagrams(s: str, t: str) -> bool:
    """O(n) anagram check using character frequency."""
    return Counter(s) == Counter(t)


def group_anagrams(words: list[str]) -> list[list[str]]:
    """
    Group words by anagram class.
    Time: O(n · L log L)  Space: O(n·L)  where L = avg word length
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for word in words:
        key = "".join(sorted(word))               # canonical form
        groups[key].append(word)
    return list(groups.values())


def longest_palindromic_substring(s: str) -> str:
    """
    Find longest palindromic substring via center expansion.
    Time: O(n²)  Space: O(1)
    """
    def expand(l: int, r: int) -> str:
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1: r]

    best = ""
    for i in range(len(s)):
        odd  = expand(i, i)        # odd-length palindromes
        even = expand(i, i + 1)    # even-length palindromes
        if len(odd) > len(best):
            best = odd
        if len(even) > len(best):
            best = even
    return best


def kmp_search(text: str, pattern: str) -> list[int]:
    """
    KMP pattern matching — return all starting indices where pattern occurs in text.
    Time: O(n + m)  Space: O(m)
    """
    if not pattern:
        return list(range(len(text)))

    # Build failure function
    m = len(pattern)
    f = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = f[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        f[i] = j

    # Search
    matches: list[int] = []
    j = 0
    for i, ch in enumerate(text):
        while j > 0 and ch != pattern[j]:
            j = f[j - 1]
        if ch == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - m + 1)
            j = f[j - 1]
    return matches


if __name__ == "__main__":
    arr = [3, -2, 5, 1, -4]
    prefix = build_prefix(arr)
    log.info("prefix: %s", prefix)
    log.info("range_sum(1,3) = %d  (expected 4)", range_sum(prefix, 1, 3))
    log.info("range_sum(0,4) = %d  (expected 3)", range_sum(prefix, 0, 4))

    log.info("subarray_sum_k=3 in [1,1,1]: %d",
             subarray_sum_equals_k([1, 1, 1], 3))       # 1
    log.info("subarray_sum_k=3 in [1,2,3]: %d",
             subarray_sum_equals_k([1, 2, 3], 3))       # 2 ([1,2] and [3])

    log.info("are_anagrams('listen','silent'): %s", are_anagrams("listen", "silent"))
    log.info("group_anagrams: %s",
             group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))

    log.info("longest_palindrome in 'babad': %r",
             longest_palindromic_substring("babad"))
    log.info("longest_palindrome in 'cbbd': %r",
             longest_palindromic_substring("cbbd"))

    log.info("kmp_search 'AABAAB' in 'AABAACAABAAB': %s",
             kmp_search("AABAACAABAAB", "AABAAB"))
```

---

## Exercises

1. Implement `product_except_self(arr: list[int]) -> list[int]` — return array where `result[i]` = product of all elements except `arr[i]`, in O(n) time and O(1) extra space (the output array doesn't count). Hint: use prefix product and suffix product.

2. Given a 2-D matrix of integers, precompute a 2-D prefix sum and implement `submatrix_sum(P, r1, c1, r2, c2) -> int` returning the sum of elements in the rectangle `(r1,c1)` to `(r2,c2)` in O(1).

3. Implement `count_palindromic_substrings(s: str) -> int` — count all palindromic substrings (including single chars). Use center expansion.

4. Implement `find_all_anagrams_in_string(s: str, p: str) -> list[int]` — find all starting indices in `s` where a substring is an anagram of `p`. Use sliding window + character frequency. Time: O(n).

5. Walk through the KMP failure function construction for `pattern = "ABCABD"`. Write out `f[]` by hand, then verify with code.

6. The `group_anagrams` solution uses `sorted(word)` as a key, giving O(L log L) per word. Propose an alternative key that is O(L) to compute (hint: use a fixed-size frequency tuple of length 26 for lowercase ASCII).

*(Solutions: ../solutions/week-01/day-5.md)*

---

## Self-check

- [ ] I can build a prefix sum array and answer range sum queries in O(1).
- [ ] I can explain the `subarray_sum_equals_k` trick using prefix sums and a frequency map.
- [ ] I can implement `are_anagrams` and `group_anagrams` using Counter.
- [ ] I can trace the KMP failure function construction for a small pattern.

---

## Supplementary links

- [Prefix sums — cp-algorithms.com](https://cp-algorithms.com/algebra/prefix-sums.html)
- [KMP explained — Knuth-Morris-Pratt on YouTube by Back To Back SWE](https://www.youtube.com/watch?v=GTJr8OvyEVQ)
- [LeetCode Sliding Window + HashMap patterns](https://leetcode.com/tag/hash-table/)
