# Solutions — Week 01 Assessment

## Part A — Conceptual

**A1.**

- **O(f):** upper bound — f(n) grows *at most* as fast as g(n) for large n (up to a constant factor). "Algorithm runs in no more than c·g(n) steps."
- **Ω(f):** lower bound — f(n) grows *at least* as fast as g(n). "Algorithm always takes at least c·g(n) steps."
- **Θ(f):** tight bound — both O and Ω hold simultaneously. "Algorithm scales exactly like g(n)."

Example Θ(n log n): merge sort. It is not O(n) because no matter what constant you choose, for large enough n, merge sort performs more than c·n comparisons — the log n factor cannot be absorbed into a constant.

**A2.** Overall: **O(n²)** — use the "take the dominant term" rule. Sequential blocks add: O(n log n) + O(n²) → keep O(n²) since n² grows faster than n log n. For large n, the n² step completely dominates.

**A3.** A single `append` that triggers a reallocation copies all current n elements — O(n) for that call. However, reallocation doubles capacity, so the *next* reallocation is not until n more appends. Spreading the O(n) copy cost across those n appends gives O(n)/n = O(1) amortized per append. The amortized bound matters because algorithm analysis typically reasons about sequences of operations, not individual worst cases. A data structure with O(1) amortized operations is fine for loops — the total cost over n operations is O(n).

**A4.**

| Operation | Time complexity |
|---|---|
| `lst[i]` | O(1) |
| `lst.append(x)` | O(1) amortized |
| `lst.insert(0, x)` | O(n) |
| `x in lst` | O(n) |
| `lst.sort()` | O(n log n) |

**A5.**
- (a) Naive: 10,000 queries × O(n) each = O(10,000 × 1,000,000) = **O(10¹⁰) total**.
- (b) Prefix sum: O(n) build = O(1,000,000) + O(1) × 10,000 queries = **O(n + q) = O(1,000,000 + 10,000) ≈ O(n) total**.

Prefix sums are the clear winner for repeated range queries on static data.

**A6.** Wrong: `list.insert(0, x)` shifts all elements right to make room — it is O(len(lst)) per call, not O(1). Over n calls: 0 + 1 + 2 + … + (n-1) = n(n-1)/2 → **O(n²) total**. Fix: use `append` for O(n) total, or prepend to a `collections.deque` in O(1) per call.

---

## Part B — Debugging

**B1.**
- **Bug:** Correctness is fine (finds pair) but complexity is wrong — nested loops = O(n²), not O(n).
- **Fix:** Two-pointer on sorted array:

```python
def find_pair(arr: list[int], target: int) -> bool:
    """Time: O(n log n) sort + O(n) scan = O(n log n). Space: O(1)."""
    arr_s = sorted(arr)
    left, right = 0, len(arr_s) - 1
    while left < right:
        s = arr_s[left] + arr_s[right]
        if s == target:
            return True
        elif s < target:
            left += 1
        else:
            right -= 1
    return False
```

**B2.**
- **Bug:** Nested loop — outer O(n), inner O(n) in worst case → **O(n²) total**, not O(n).
- **Fix:** Sliding window with a frequency dict:

```python
from collections import defaultdict

def longest_two_distinct(s: str) -> int:
    """Time: O(n)  Space: O(1) — at most 3 keys in freq at any time."""
    freq: dict[str, int] = defaultdict(int)
    left = 0
    best = 0
    for right, ch in enumerate(s):
        freq[ch] += 1
        while len(freq) > 2:
            freq[s[left]] -= 1
            if freq[s[left]] == 0:
                del freq[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best
```

**B3.**
- **Bug:** `sorted(s) == sorted(t)` returns True if both strings have the same characters *and counts*, but the length check is implicit since if lengths differ, sorted lists differ. Re-testing: `is_anagram("rat", "tare")` → `sorted("rat")=['a','r','t']`, `sorted("tare")=['a','e','r','t']` → these are NOT equal → the function returns False correctly.
- **Actual bug:** The question is a trap. The code `sorted(s) == sorted(t)` IS correct. The comment "should be False" in the problem is intentionally testing whether you verify rather than trust the claim. `is_anagram("rat", "tare")` correctly returns **False** because the sorted lists have different lengths. The bug was in the problem statement's claim, not the code.
- However: a real improvement is the O(n) Counter version (avoid O(n log n) sort) — correct in both cases, faster.

---

## Part C — Coding Challenge

**C1.**

```python
def all_pairs_with_sum(arr: list[int], target: int) -> list[tuple[int, int]]:
    """
    Return all unique pairs (a, b) where a + b == target, a ≤ b.
    Time: O(n)  Space: O(n)
    """
    seen: set[int] = set()
    result: set[tuple[int, int]] = set()
    for val in arr:
        complement = target - val
        if complement in seen:
            pair = (min(val, complement), max(val, complement))
            result.add(pair)
        seen.add(val)
    return sorted(result)
```

**C2.**

```python
def count_subarrays_product_less_than_k(arr: list[int], k: int) -> int:
    """
    Count subarrays with product < k. All elements positive.
    Time: O(n)  Space: O(1)
    Key insight: for each right, count subarrays ending at right with product < k.
    That count = right - left + 1 (all starting positions from left to right).
    """
    if k <= 1:
        return 0
    left = 0
    product = 1
    count = 0
    for right in range(len(arr)):
        product *= arr[right]
        while product >= k and left <= right:
            product //= arr[left]
            left += 1
        count += right - left + 1   # subarrays [left..right], [left+1..right], ..., [right..right]
    return count
```

**C3.**

```python
def longest_palindrome_details(s: str) -> tuple[int, str]:
    """
    Return (length, substring) of longest palindromic substring.
    Time: O(n²)  Space: O(1)
    """
    if not s:
        return (0, "")
    best_l = best_r = 0

    def expand(l: int, r: int) -> None:
        nonlocal best_l, best_r
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l > best_r - best_l:
                best_l, best_r = l, r
            l -= 1; r += 1

    for i in range(len(s)):
        expand(i, i)
        expand(i, i + 1)
    substr = s[best_l: best_r + 1]
    return (len(substr), substr)
```

---

## Part D — Self-assessment notes

Checklist reviewed against Part C solutions above:
- Type hints: all present (`list[int]`, `tuple[int, int]`, `set[int]`, etc.)
- Docstrings with Time/Space: all present.
- Magic constants: `26` avoided (alphabet size tuples not used here; `k <= 1` is self-explanatory).
- No `print()`: all return values.
- Edge cases: empty input, k=0/k=1, single-char strings all handled.
- Meaningful names: `complement`, `left`, `right`, `product`, `best_l`, `best_r`.

Score: 6/6 on code quality checklist.
