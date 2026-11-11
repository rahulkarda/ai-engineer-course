# Solutions: Week 08 · Day 5 — Heaps/Top-K + Tries + Union-Find + Greedy

## Exercise 1: Find Median from Data Stream

```python
import heapq


class MedianFinder:
    """Two-heap approach.  Time: O(log n) add, O(1) find_median.  Space: O(n)

    Invariant:
    - max_heap (lower half): all values ≤ median, stored negated for max semantics
    - min_heap (upper half): all values ≥ median
    - len(max_heap) == len(min_heap) OR len(max_heap) == len(min_heap) + 1
    """

    def __init__(self) -> None:
        self.max_heap: list[int] = []   # lower half, negated
        self.min_heap: list[int] = []   # upper half

    def add_num(self, num: int) -> None:
        heapq.heappush(self.max_heap, -num)
        # Balance: largest of lower half must be ≤ smallest of upper half
        if self.max_heap and self.min_heap and (-self.max_heap[0]) > self.min_heap[0]:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        # Rebalance sizes
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def find_median(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

**Rebalancing invariant:** after every insertion, `max_heap` may have at most one more element than `min_heap`, never fewer. `find_median` is O(1) — just peek at heap tops.

---

## Exercise 2: Word Search II

```python
class TrieNode:
    __slots__ = ("children", "word")
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.word: str | None = None   # non-None at word end


def find_words(board: list[list[str]], words: list[str]) -> list[str]:
    """Trie + DFS.  Time: O(m × n × 4^L)  Space: O(total chars in words)"""
    root = TrieNode()
    # Build trie
    for word in words:
        node = root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.word = word

    rows, cols = len(board), len(board[0])
    result: list[str] = []

    def dfs(r: int, c: int, node: TrieNode) -> None:
        ch = board[r][c]
        if ch not in node.children:
            return
        next_node = node.children[ch]
        if next_node.word:
            result.append(next_node.word)
            next_node.word = None     # avoid duplicates
        board[r][c] = "#"            # mark visited
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != "#":
                dfs(nr, nc, next_node)
        board[r][c] = ch             # restore

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)
    return result
```

**Why Trie beats per-word DFS:** with W words of length L on an m×n board, per-word DFS is O(W × m × n × 4^L). Trie shares prefixes — the DFS prunes entire branches when the current prefix matches no word, reducing cost to O(m × n × 4^L) with a constant proportional to unique prefixes.

---

## Exercise 3: Redundant Connection

```python
class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def find_redundant_connection(edges: list[list[int]]) -> list[int]:
    """First union() returning False identifies the redundant edge.
    Time: O(n α(n))  Space: O(n)
    """
    uf = UnionFind(len(edges))
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
    return []
```

---

## Exercise 4: Task Scheduler

```python
import heapq
from collections import Counter, deque


def least_interval(tasks: list[str], n: int) -> int:
    """Greedy: always schedule most frequent task.  Time: O(m log 26) = O(m)  Space: O(26)"""
    freq = Counter(tasks)
    max_heap = [-count for count in freq.values()]
    heapq.heapify(max_heap)
    time = 0
    cooldown: deque[tuple[int, int]] = deque()  # (available_at, -count)

    while max_heap or cooldown:
        time += 1
        if max_heap:
            count = 1 + heapq.heappop(max_heap)   # -(-count) - 1
            if count < 0:
                cooldown.append((time + n, count))
        if cooldown and cooldown[0][0] == time:
            heapq.heappush(max_heap, cooldown.popleft()[1])
    return time
```

**Alternative O(1) formula:** `max(len(tasks), (max_freq - 1) * (n + 1) + count_max_freq)` where `max_freq` is the highest task frequency. Useful for interviews as a one-liner proof after explaining the logic.

---

## Exercise 5: Longest Word in Dictionary

```python
class TrieNode:
    __slots__ = ("children", "is_end")
    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False


def longest_word(words: list[str]) -> str:
    """Trie + BFS by word length.  Time: O(total chars)  Space: O(total chars)"""
    root = TrieNode()
    for word in words:
        node = root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    result = ""
    # DFS through trie; only follow edges where node.is_end == True
    stack = [(root, "")]
    while stack:
        node, current = stack.pop()
        for ch, child in node.children.items():
            if child.is_end:
                candidate = current + ch
                if len(candidate) > len(result) or (
                    len(candidate) == len(result) and candidate < result
                ):
                    result = candidate
                stack.append((child, candidate))
    return result
```

---

## Test file

```python
# tests/test_day5.py
import pytest
from day5_solutions import MedianFinder, find_words, find_redundant_connection, least_interval


def test_median_finder() -> None:
    mf = MedianFinder()
    mf.add_num(1)
    mf.add_num(2)
    assert mf.find_median() == 1.5
    mf.add_num(3)
    assert mf.find_median() == 2.0


def test_find_words() -> None:
    board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    words = ["oath","pea","eat","rain"]
    assert sorted(find_words(board, words)) == ["eat", "oath"]


def test_find_redundant_connection() -> None:
    assert find_redundant_connection([[1,2],[1,3],[2,3]]) == [2, 3]
    assert find_redundant_connection([[1,2],[2,3],[3,4],[1,4],[1,5]]) == [1, 4]


def test_least_interval() -> None:
    assert least_interval(["A","A","A","B","B","B"], 2) == 8
    assert least_interval(["A","A","A","B","B","B"], 0) == 6
```
