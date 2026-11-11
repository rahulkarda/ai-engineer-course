# Solutions — Day 5: Prefix Sums & String Algorithms

## Exercise 1 — product_except_self

```python
def product_except_self(arr: list[int]) -> list[int]:
    """
    result[i] = product of all elements except arr[i].
    Time: O(n)  Space: O(1) extra (output not counted)
    """
    n = len(arr)
    result = [1] * n

    # Pass 1: result[i] = product of arr[0..i-1]
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= arr[i]

    # Pass 2: multiply by suffix product arr[i+1..n-1]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= arr[i]

    return result

assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
assert product_except_self([0, 0]) == [0, 0]
```

---

## Exercise 2 — 2-D prefix sum

```python
def build_2d_prefix(matrix: list[list[int]]) -> list[list[int]]:
    """
    Build 2-D prefix sum with 1-indexed sentinel row/col.
    P[i][j] = sum of all elements in rows [0..i-1], cols [0..j-1].
    Time: O(n·m)  Space: O(n·m)
    """
    n, m = len(matrix), len(matrix[0])
    P = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = (matrix[i-1][j-1]
                       + P[i-1][j] + P[i][j-1]
                       - P[i-1][j-1])
    return P


def submatrix_sum(P: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> int:
    """
    Sum of elements in rectangle (r1,c1) to (r2,c2) inclusive (0-indexed).
    Time: O(1)
    """
    return (P[r2+1][c2+1]
            - P[r1][c2+1]
            - P[r2+1][c1]
            + P[r1][c1])

mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
P = build_2d_prefix(mat)
assert submatrix_sum(P, 0, 0, 1, 1) == 12   # 1+2+4+5
assert submatrix_sum(P, 1, 1, 2, 2) == 28   # 5+6+8+9
```

---

## Exercise 3 — count_palindromic_substrings

```python
def count_palindromic_substrings(s: str) -> int:
    """
    Count all palindromic substrings (including single chars).
    Time: O(n²)  Space: O(1)
    """
    count = 0

    def expand(l: int, r: int) -> None:
        nonlocal count
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1

    for i in range(len(s)):
        expand(i, i)        # odd length
        expand(i, i + 1)    # even length
    return count

assert count_palindromic_substrings("abc") == 3    # a, b, c
assert count_palindromic_substrings("aaa") == 6    # a,a,a,aa,aa,aaa
```

---

## Exercise 4 — find_all_anagrams_in_string

```python
from collections import Counter

def find_all_anagrams_in_string(s: str, p: str) -> list[int]:
    """
    Find all starting indices in s where a substring is an anagram of p.
    Time: O(n)  Space: O(1) — alphabet size bounded
    """
    if len(p) > len(s):
        return []
    k = len(p)
    need = Counter(p)
    have = Counter(s[:k])
    result: list[int] = []

    if have == need:
        result.append(0)

    for i in range(k, len(s)):
        # add right char
        have[s[i]] += 1
        # remove left char
        left_ch = s[i - k]
        have[left_ch] -= 1
        if have[left_ch] == 0:
            del have[left_ch]
        if have == need:
            result.append(i - k + 1)
    return result

assert find_all_anagrams_in_string("cbaebabacd", "abc") == [0, 6]
assert find_all_anagrams_in_string("abab", "ab") == [0, 1, 2]
```

---

## Exercise 5 — KMP failure function for "ABCABD"

```
pattern = "ABCABD"
indices:   0 1 2 3 4 5

i=0: f[0]=0 (base case)
i=1: B≠A, j stays 0, f[1]=0
i=2: C≠A, j stays 0, f[2]=0
i=3: A==A, j becomes 1, f[3]=1  (prefix "A" is also suffix of "ABCA")
i=4: B==B, j becomes 2, f[4]=2  (prefix "AB" is also suffix of "ABCAB")
i=5: D≠C, fall back: j=f[1]=0; D≠A; f[5]=0
```

`f = [0, 0, 0, 1, 2, 0]`

---

## Exercise 6 — O(L) anagram key using frequency tuple

```python
def group_anagrams_o_n(words: list[str]) -> list[list[str]]:
    """
    Time: O(n·L)  Space: O(n·L)
    Key: tuple of 26 counts for lowercase ASCII letters.
    """
    from collections import defaultdict
    groups: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for word in words:
        freq = [0] * 26                          # 26 = lowercase ASCII alphabet
        for ch in word:
            freq[ord(ch) - ord('a')] += 1
        groups[tuple(freq)].append(word)
    return list(groups.values())
```

Building the 26-length counter is O(L) per word. Sorting `sorted(word)` is O(L log L). For large L and many words, the O(L) key gives a constant-factor speedup.
