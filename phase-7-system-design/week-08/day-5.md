# Phase 7 · Week 08 · Day 5 — Heaps/Top-K + Tries + Union-Find + Greedy

> Objective: Apply heap, trie, union-find, and greedy patterns to their canonical problem classes.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Heaps / Top-K

Python's `heapq` is a **min-heap**. For max-heap, negate values.

**Top-K pattern:** maintain a size-K min-heap; push every element; pop when size exceeds K. Result: K largest elements in O(n log K).

```python
import heapq

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    freq = {}
    for n in nums:
        freq[n] = freq.get(n, 0) + 1
    # min-heap of (count, value); keep k largest by count
    heap: list[tuple[int, int]] = []
    for val, count in freq.items():
        heapq.heappush(heap, (count, val))
        if len(heap) > k:
            heapq.heappop(heap)
    return [val for _, val in heap]
```

**K-way merge:** push `(value, list_index, element_index)` → merge k sorted lists in O(n log k).

---

### Tries (Prefix Trees)

**When to use:** autocomplete, prefix search, word existence in dictionary.

```python
class TrieNode:
    __slots__ = ("children", "is_end")
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False

class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
```

---

### Union-Find (Disjoint Set Union)

**When to use:** connected components; detect cycle in undirected graph; Kruskal's MST.

```python
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False   # already connected
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

With path compression + union by rank: amortised **O(α(n)) ≈ O(1)** per operation.

---

### Greedy

**When to use:** interval scheduling; jump game; task scheduling; Huffman coding; minimum spanning tree.

**Greedy correctness requires an exchange argument:** show that swapping any non-greedy choice for the greedy choice never makes the solution worse.

**Common greedy patterns:**
- Sort by some criterion (end time, deadline, ratio)
- Always pick the locally optimal choice
- Prove no "repair" step can improve it

---

## How it works internally

**Heap invariant:** in a min-heap, parent ≤ both children. `heappush` sifts up (O(log n)); `heappop` removes root and sifts down (O(log n)). `heapify` on existing list is O(n) — not O(n log n).

**Trie memory:** each node stores a dict of up to 26 (or alphabet size) children. Worst-case O(n × L × alphabet) space where L = avg word length. Using arrays instead of dicts reduces constant but requires index mapping.

**Path compression proof:** after `find(x)` flattens all ancestors of x to point directly to root, subsequent `find` on any of them is O(1). Union by rank keeps tree height O(log n) even before compression.

---

## Worked example

### Problem 1: Merge K Sorted Lists

```python
# heaps_tries_uf.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import heapq
import logging

logger = logging.getLogger(__name__)


@dataclass
class ListNode:
    val: int
    next: Optional["ListNode"] = None
    def __lt__(self, other: "ListNode") -> bool:   # heapq comparison
        return self.val < other.val


def merge_k_lists(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    """K-way merge.  Time: O(n log k)  Space: O(k)"""
    heap: list[ListNode] = []
    for node in lists:
        if node:
            heapq.heappush(heap, node)
    dummy = ListNode(0)
    curr = dummy
    while heap:
        node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, node.next)
    return dummy.next
```

### Problem 2: Number of Connected Components (Union-Find)

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    """Count connected components.  Time: O(E α(n))"""
    uf = UnionFind(n)
    components = n
    for u, v in edges:
        if uf.union(u, v):
            components -= 1
    return components
```

### Problem 3: Jump Game II (Greedy — min jumps)

```python
def jump(nums: list[int]) -> int:
    """Minimum jumps to reach last index.  Time: O(n)  Space: O(1)"""
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:          # exhausted current jump range
            jumps += 1
            current_end = farthest    # commit to best reach
    return jumps
```

---

## Exercises

1. **Find Median from Data Stream** — design a class supporting `add_num(int)` and `find_median()`. Use two heaps (max-heap for lower half, min-heap for upper half). State the rebalancing invariant.

2. **Word Search II** — given a board of chars and a list of words, return all words that exist in the board (adjacent cells). Build a Trie from the word list; DFS with trie traversal for O(4^L) search per starting cell. Why is Trie better than checking each word separately?

3. **Redundant Connection** — given an undirected graph that started as a tree and has one extra edge, find and return that edge. Use Union-Find: the first `union` call that returns `False` (already connected) identifies the redundant edge.

4. **Task Scheduler** — given tasks with a cooling interval n between same-type tasks, find the minimum intervals to finish all tasks. Greedy: always schedule the most frequent remaining task. Use a max-heap and a queue for the cooldown period.

5. **Longest Word in Dictionary** — given a list of words, find the longest word that can be built one character at a time (each prefix also in the list). Use a Trie; BFS by word length.

(Solutions: ../solutions/week-08/day-5.md)

---

## Self-check

- [ ] I can implement a top-K heap pattern from memory
- [ ] I can implement Trie insert/search/starts_with
- [ ] I can implement Union-Find with path compression and union by rank
- [ ] I can articulate why greedy works for jump game (exchange argument sketch)
- [ ] I know O(α(n)) amortised cost of Union-Find operations

---

## Supplementary links

- NeetCode Heap / Priority Queue: https://neetcode.io/roadmap
- NeetCode Tries: https://neetcode.io/roadmap
- NeetCode Advanced Graphs (Union-Find): https://neetcode.io/roadmap
- NeetCode Greedy: https://neetcode.io/roadmap
