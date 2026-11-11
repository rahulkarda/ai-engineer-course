# Solutions: Week 08 · Day 3 — Trees/Graphs BFS & DFS + Backtracking

## Exercise 1: Maximum Depth of Binary Tree

```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def max_depth_recursive(root: Optional[TreeNode]) -> int:
    """DFS recursive.  Time: O(n)  Space: O(h) call stack"""
    if root is None:
        return 0
    return 1 + max(max_depth_recursive(root.left), max_depth_recursive(root.right))


def max_depth_bfs(root: Optional[TreeNode]) -> int:
    """BFS iterative.  Time: O(n)  Space: O(w) max width"""
    if root is None:
        return 0
    depth = 0
    queue: deque[TreeNode] = deque([root])
    while queue:
        depth += 1
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return depth
```

**Simpler:** recursive DFS — one line of logic. BFS is more verbose but avoids call stack overflow on skewed trees. For balanced trees, both are O(log n) space; for skewed trees recursive is O(n) stack, BFS is O(1) effectively.

---

## Exercise 2: Clone Graph

```python
from dataclasses import dataclass, field
from collections import deque


@dataclass
class GraphNode:
    val: int
    neighbors: list["GraphNode"] = field(default_factory=list)


def clone_graph(node: Optional[GraphNode]) -> Optional[GraphNode]:
    """BFS with visited map.  Time: O(n + e)  Space: O(n)"""
    if node is None:
        return None
    visited: dict[GraphNode, GraphNode] = {}
    queue: deque[GraphNode] = deque([node])
    visited[node] = GraphNode(node.val)
    while queue:
        curr = queue.popleft()
        for neighbor in curr.neighbors:
            if neighbor not in visited:
                visited[neighbor] = GraphNode(neighbor.val)
                queue.append(neighbor)
            visited[curr].neighbors.append(visited[neighbor])
    return visited[node]
```

**Key:** create the clone node at first visit (before processing its neighbors). The `visited` map prevents re-cloning and resolves back-edges.

---

## Exercise 3: Word Ladder

```python
from collections import deque, defaultdict


def ladder_length(begin_word: str, end_word: str, word_list: list[str]) -> int:
    """BFS shortest transformation.  Time: O(m² × n)  Space: O(m² × n)
    where m = word length, n = word list size.
    """
    word_set = set(word_list)
    if end_word not in word_set:
        return 0

    # Build adjacency by wildcard pattern: "hot" → {"*ot": ["hot", ...], ...}
    pattern_map: dict[str, list[str]] = defaultdict(list)
    for word in word_set | {begin_word}:
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i + 1:]
            pattern_map[pattern].append(word)

    queue: deque[tuple[str, int]] = deque([(begin_word, 1)])
    visited: set[str] = {begin_word}
    while queue:
        word, length = queue.popleft()
        for i in range(len(word)):
            pattern = word[:i] + "*" + word[i + 1:]
            for neighbor in pattern_map[pattern]:
                if neighbor == end_word:
                    return length + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, length + 1))
    return 0
```

**Edge case:** `end_word not in word_list` → return 0 immediately. The pattern map is O(m × n) preprocessing that makes each BFS step O(m) instead of O(m × n).

---

## Exercise 4: Subsets II (with duplicates)

```python
def subsets_with_dup(nums: list[int]) -> list[list[int]]:
    """Backtracking with duplicate skip.  Time: O(n × 2^n)"""
    nums.sort()
    result: list[list[int]] = []

    def backtrack(start: int, current: list[int]) -> None:
        result.append(current[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue    # skip duplicate at same recursion level
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

**Duplicate skip logic:** `i > start` ensures we only skip duplicates at the *same level* of the recursion tree — not across levels. This is the critical condition; `i > 0` would be wrong.

---

## Exercise 5: Course Schedule (topological sort)

```python
from collections import defaultdict, deque


def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """Kahn's algorithm (BFS topological sort).  Time: O(V + E)  Space: O(V + E)"""
    in_degree = [0] * num_courses
    graph: dict[int, list[int]] = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    queue: deque[int] = deque(c for c in range(num_courses) if in_degree[c] == 0)
    completed = 0
    while queue:
        course = queue.popleft()
        completed += 1
        for neighbor in graph[course]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return completed == num_courses   # False if cycle exists
```

**Cycle detection:** if a cycle exists, nodes in the cycle never reach in-degree 0 and are never enqueued. `completed < num_courses` reveals the cycle.

---

## Test file

```python
# tests/test_day3.py
import pytest
from day3_solutions import (
    max_depth_recursive, max_depth_bfs, ladder_length, subsets_with_dup, can_finish, TreeNode
)


def make_tree(vals: list[int | None]) -> Optional[TreeNode]:
    """Build tree from level-order list."""
    if not vals or vals[0] is None:
        return None
    root = TreeNode(vals[0])
    queue = deque([root])
    i = 1
    while queue and i < len(vals):
        node = queue.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            queue.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            queue.append(node.right)
        i += 1
    return root


def test_max_depth_empty() -> None:
    assert max_depth_recursive(None) == 0
    assert max_depth_bfs(None) == 0


def test_max_depth() -> None:
    root = make_tree([3, 9, 20, None, None, 15, 7])
    assert max_depth_recursive(root) == 3
    assert max_depth_bfs(root) == 3


def test_ladder_length() -> None:
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0


def test_subsets_with_dup() -> None:
    result = sorted(map(tuple, subsets_with_dup([1, 2, 2])))
    expected = sorted(map(tuple, [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]))
    assert result == expected


def test_course_schedule() -> None:
    assert can_finish(2, [[1, 0]]) is True
    assert can_finish(2, [[1, 0], [0, 1]]) is False
    assert can_finish(1, []) is True
```
