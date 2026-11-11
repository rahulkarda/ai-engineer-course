# Phase 7 · Week 08 · Day 3 — Trees/Graphs: BFS & DFS + Backtracking

> Objective: Apply BFS/DFS templates to tree and graph problems; recognise backtracking structure.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### BFS — Breadth-First Search

**When to use:** shortest path in unweighted graph; level-order traversal; "minimum steps" questions.

```python
from collections import deque

def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    visited: set[int] = {start}
    queue: deque[int] = deque([start])
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
```

For **level-by-level** processing (e.g., binary tree level order):
```python
while queue:
    for _ in range(len(queue)):   # snapshot current level size
        node = queue.popleft()
        ...
```

---

### DFS — Depth-First Search

**Iterative (explicit stack):**
```python
def dfs_iterative(graph: dict[int, list[int]], start: int) -> list[int]:
    visited: set[int] = set()
    stack = [start]
    order: list[int] = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in reversed(graph[node]):
            stack.append(neighbor)
    return order
```

**Recursive:**
```python
def dfs(node, visited, graph):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, visited, graph)
```

**On trees:** no visited set needed (no cycles); recurse left/right directly.

---

### Backtracking

**When to use:** generate all combinations/permutations/subsets; constraint-satisfaction (N-queens, sudoku).

```python
def backtrack(state, choices, result):
    if is_complete(state):
        result.append(state[:])   # copy snapshot
        return
    for choice in choices:
        if is_valid(state, choice):
            state.append(choice)          # choose
            backtrack(state, ..., result) # explore
            state.pop()                   # un-choose
```

**Pruning** is critical for performance: skip branches that can never lead to a valid solution.

---

## How it works internally

**BFS queue invariant:** the deque always holds nodes at the current or next frontier level. Because FIFO order is preserved, the first time a node is dequeued is via the shortest path.

**DFS call stack vs. recursion stack:** recursive DFS uses the language call stack; for deep graphs (n > 1000) Python's default recursion limit (1000) can overflow — use iterative DFS or `sys.setrecursionlimit`.

**Backtracking complexity:** worst case is exponential (O(n!), O(2ⁿ)) but pruning reduces practical runtime dramatically. Always state the theoretical worst case, then explain your pruning strategy.

---

## Worked example

### Problem 1: Binary Tree Level-Order Traversal

```python
# bfs_dfs.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    """BFS level-by-level.  Time: O(n)  Space: O(n)"""
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level: list[int] = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### Problem 2: Number of Islands (DFS on grid)

```python
def num_islands(grid: list[list[str]]) -> int:
    """Count connected components of '1's.  Time: O(m*n)  Space: O(m*n)"""
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int) -> None:
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
            return
        grid[r][c] = "0"   # mark visited in-place
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                dfs(r, c)
                count += 1
    return count
```

### Problem 3: Combination Sum (backtracking)

```python
def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """All combos that sum to target; unlimited reuse.  Time: O(n^(t/m))"""
    result: list[list[int]] = []
    candidates.sort()

    def backtrack(start: int, current: list[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current[:])
            return
        for i in range(start, len(candidates)):
            c = candidates[i]
            if c > remaining:    # pruning: sorted, so rest are too big
                break
            current.append(c)
            backtrack(i, current, remaining - c)   # i not i+1: can reuse
            current.pop()

    backtrack(0, [], target)
    return result
```

---

## Exercises

1. **Maximum Depth of Binary Tree** — implement both recursive DFS and iterative BFS solutions. State which is simpler and why.

2. **Clone Graph** — given a reference to a node in a connected undirected graph, return a deep copy. Use BFS with a `{original: clone}` hash map.

3. **Word Ladder** — given `begin_word`, `end_word`, and a `word_list`, return the shortest transformation sequence length where each step changes one letter. Model as BFS; use a set for O(1) lookups. Handle the case where `end_word` is not in the list.

4. **Subsets II** — given integers (may contain duplicates), return all possible subsets with no duplicate subsets. Sort first; backtrack with duplicate-skip logic.

5. **Course Schedule** — detect if n courses with prerequisites form a cycle (topological sort via DFS). Return `True` if all courses can be finished.

(Solutions: ../solutions/week-08/day-3.md)

---

## Self-check

- [ ] I can write BFS level-order without reference
- [ ] I can implement DFS on a grid (islands) from memory
- [ ] I can implement the backtracking template (choose/explore/un-choose)
- [ ] I know when iterative vs. recursive DFS is safer in Python
- [ ] I can explain BFS shortest-path guarantee with a one-sentence invariant

---

## Supplementary links

- NeetCode Trees roadmap: https://neetcode.io/roadmap
- NeetCode Graphs section: https://neetcode.io/roadmap
- NeetCode Backtracking section: https://neetcode.io/roadmap
