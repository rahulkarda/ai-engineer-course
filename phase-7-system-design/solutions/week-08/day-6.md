# Solutions: Week 08 · Day 6 — 4-Hour Timed Problem Set

## P1 — Valid Palindrome II

```python
def valid_palindrome(s: str) -> bool:
    """At most one deletion.  Time: O(n)  Space: O(1)"""
    def is_palindrome(lo: int, hi: int) -> bool:
        while lo < hi:
            if s[lo] != s[hi]:
                return False
            lo += 1; hi -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)
        left += 1; right -= 1
    return True
```

**Pattern:** Two pointers.  
**Time/Space:** O(n) / O(1).  
**Key decision:** when mismatch found, try skipping left char OR right char — if either yields a palindrome, answer is True.

---

## P2 — Best Time to Buy and Sell Stock

```python
def max_profit(prices: list[int]) -> int:
    """One pass.  Time: O(n)  Space: O(1)"""
    min_price = float("inf")
    best = 0
    for price in prices:
        min_price = min(min_price, price)
        best = max(best, price - min_price)
    return best
```

**Pattern:** Greedy / sliding window (track running minimum).

---

## P3 — Find Peak Element

```python
def find_peak_element(nums: list[int]) -> int:
    """Binary search.  Time: O(log n)  Space: O(1)"""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid       # peak is at mid or left
        else:
            left = mid + 1    # peak is strictly right of mid
    return left
```

**Invariant:** if `nums[mid] < nums[mid+1]`, the function is still ascending → any peak must be to the right. Otherwise, `mid` is a potential peak → keep in range by setting `right = mid`.

---

## P4 — Insert Interval

```python
def insert(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:
    """Time: O(n)  Space: O(n)"""
    result: list[list[int]] = []
    i = 0
    n = len(intervals)
    # Phase 1: add all intervals ending before new_interval starts
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    # Phase 2: merge all overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)
    # Phase 3: add remaining
    while i < n:
        result.append(intervals[i])
        i += 1
    return result
```

**Edge case — empty intervals:** `n=0`, all while loops skip, `new_interval` appended directly. Correct.

---

## P5 — Validate Binary Search Tree

```python
from __future__ import annotations
from typing import Optional


class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None) -> None:
        self.val, self.left, self.right = val, left, right


def is_valid_bst(root: Optional[TreeNode]) -> bool:
    """Pass bounds through recursion.  Time: O(n)  Space: O(h)"""
    def validate(node: Optional[TreeNode], lo: float, hi: float) -> bool:
        if node is None:
            return True
        if not (lo < node.val < hi):
            return False
        return (validate(node.left, lo, node.val) and
                validate(node.right, node.val, hi))

    return validate(root, float("-inf"), float("inf"))
```

**Common mistake:** checking only `left < root < right` at each node, not the full ancestor bounds. A node in the right subtree of an ancestor must be greater than the ancestor — pass `lo` through all right children.

---

## P6 — Word Break

```python
def word_break(s: str, word_dict: list[str]) -> bool:
    """1-D DP.  Time: O(n²)  Space: O(n)"""
    words = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True   # empty prefix is always valid
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[n]
```

**Definition:** `dp[i]` = can the first i characters of s be segmented.  
**Optimisation:** use word lengths as the inner loop range to avoid O(n) string slices:

```python
for i in range(1, n + 1):
    for word in words:
        L = len(word)
        if i >= L and dp[i - L] and s[i - L:i] == word:
            dp[i] = True; break
```

---

## P7 — Kth Largest Element

```python
import heapq


def find_kth_largest(nums: list[int], k: int) -> int:
    """Min-heap of size k.  Time: O(n log k)  Space: O(k)"""
    heap: list[int] = []
    for n in nums:
        heapq.heappush(heap, n)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]
```

**Quickselect bonus (O(n) average):**

```python
import random

def find_kth_largest_qs(nums: list[int], k: int) -> int:
    target = len(nums) - k

    def partition(lo: int, hi: int) -> int:
        pivot_idx = random.randint(lo, hi)
        nums[pivot_idx], nums[hi] = nums[hi], nums[pivot_idx]
        pivot = nums[hi]
        store = lo
        for i in range(lo, hi):
            if nums[i] <= pivot:
                nums[i], nums[store] = nums[store], nums[i]
                store += 1
        nums[store], nums[hi] = nums[hi], nums[store]
        return store

    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        p = partition(lo, hi)
        if p == target:
            return nums[p]
        elif p < target:
            lo = p + 1
        else:
            hi = p - 1
    return -1
```

---

## P8 — Number of Islands II

```python
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = [-1] * n
        self.rank = [0] * n
        self.count = 0

    def add(self, x: int) -> None:
        self.parent[x] = x
        self.count += 1

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.count -= 1


def num_islands2(m: int, n: int, positions: list[list[int]]) -> list[int]:
    """Time: O(k α(mn))  Space: O(mn)"""
    uf = UnionFind(m * n)
    result: list[int] = []
    land: set[int] = set()
    for r, c in positions:
        idx = r * n + c
        if idx not in land:
            land.add(idx)
            uf.add(idx)
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                nidx = nr * n + nc
                if 0 <= nr < m and 0 <= nc < n and nidx in land:
                    uf.union(idx, nidx)
        result.append(uf.count)
    return result
```

---

## P9 — Alien Dictionary

```python
from collections import defaultdict, deque


def alien_order(words: list[str]) -> str:
    """Kahn's BFS topological sort.  Time: O(C)  Space: O(1) chars"""
    adj: dict[str, set[str]] = {ch: set() for word in words for ch in word}
    in_degree: dict[str, int] = {ch: 0 for ch in adj}

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        # Edge case: prefix rule — "abc" before "ab" is invalid
        if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    in_degree[w2[j]] += 1
                break

    queue: deque[str] = deque(ch for ch in in_degree if in_degree[ch] == 0)
    result: list[str] = []
    while queue:
        ch = queue.popleft()
        result.append(ch)
        for neighbor in adj[ch]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return "".join(result) if len(result) == len(in_degree) else ""
```

---

## P10 — Trapping Rain Water

```python
def trap(height: list[int]) -> int:
    """Two pointers.  Time: O(n)  Space: O(1)"""
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water
```

**Prefix-array approach (O(n) space):** compute `left_max[i]` = max height in `[0..i]` and `right_max[i]` = max in `[i..n-1]`; water at i = `min(left_max[i], right_max[i]) - height[i]`. Two pointers superior because it achieves the same in O(1) extra space by observing: when `height[left] < height[right]`, `right_max` is at least `height[right] > height[left]`, so `min(left_max, right_max) = left_max` — no need to compute the full array.

---

## Complexity summary

| Problem | Pattern | Time | Space |
|---|---|---|---|
| P1 Valid Palindrome II | Two pointers | O(n) | O(1) |
| P2 Best Time Buy/Sell | Greedy/1-pass | O(n) | O(1) |
| P3 Find Peak Element | Binary search | O(log n) | O(1) |
| P4 Insert Interval | Intervals sweep | O(n) | O(n) |
| P5 Validate BST | DFS with bounds | O(n) | O(h) |
| P6 Word Break | 1-D DP | O(n²) | O(n) |
| P7 Kth Largest | Heap | O(n log k) | O(k) |
| P8 Islands II | Union-Find | O(k α(mn)) | O(mn) |
| P9 Alien Dictionary | Topo sort (BFS) | O(C) | O(C) |
| P10 Trapping Rain Water | Two pointers | O(n) | O(1) |
