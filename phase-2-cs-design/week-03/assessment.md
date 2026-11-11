# Week 03 Assessment — Trees, BSTs & Heaps

> Time: ~1 hr  ·  Pass gate: 75%

---

## Conceptual (interview-style) — 25 pts

Answer each question concisely (2–5 sentences or a diagram where appropriate).

1. What is the BST invariant? Give an example of a tree that looks like a valid BST locally (every node satisfies `left.val < node.val < right.val`) but is actually invalid. Explain why.

2. Compare inorder, preorder, and postorder traversal: what order do they visit nodes, and give one practical use case for each.

3. A BST is built by inserting the values `[8, 4, 12, 2, 6, 10, 14]` in that order. (a) Draw the resulting tree. (b) What is its height? (c) Is it height-balanced?

4. What is the heap property? Why is a heap stored as an array rather than as linked nodes? Give the index formulas for parent, left child, and right child in a 0-indexed array.

5. You need a data structure that supports `insert(val)` in O(log n) and `get_min()` in O(1). Which structure do you choose, and why? What if you also need O(log n) `delete_arbitrary(val)`?

6. Explain why the BST `delete` operation's case of "two children" uses the in-order successor (not just any node) and why the successor has at most one child.

7. What is the time complexity of building a heap via repeated `heappush` vs. `heapify`? Explain intuitively why the latter is faster.

---

## Coding challenge — 40 pts

Implement the following **in a single file** called `assessment_solution.py`. Include a `pytest`-compatible test block at the bottom.

### Part A — BST operations (20 pts)

Implement a `BST` class with:

```python
class BST:
    def insert(self, val: int) -> None: ...
    def search(self, val: int) -> bool: ...
    def delete(self, val: int) -> None: ...
    def inorder(self) -> list[int]: ...      # sorted list
    def is_valid(self) -> bool: ...           # BST invariant check
    def height(self) -> int: ...              # -1 for empty
```

Requirements:
- Delete must handle all three cases (leaf, one child, two children).
- `is_valid` must catch cross-subtree violations (not just parent–child).
- All methods must use type hints.

### Part B — MinHeap priority queue (15 pts)

Implement a `PriorityQueue` class backed by a min-heap (do NOT use `heapq` internally — implement sift-up/sift-down yourself):

```python
class PriorityQueue:
    def push(self, priority: int, item: str) -> None: ...
    def pop(self) -> tuple[int, str]: ...    # returns (priority, item)
    def peek(self) -> tuple[int, str]: ...
    def __len__(self) -> int: ...
```

Ties in priority: any order acceptable.

### Part C — Tree problem (5 pts)

Using your `BST` class, implement:
```python
def closest_pair_bst(root: TreeNode, target: float) -> int:
    """Return the BST value closest to target."""
    ...
```

---

## Debugging challenge — 20 pts

The following code has **three bugs**. Identify each, explain what it does wrong, and provide the corrected line.

```python
from __future__ import annotations
from typing import Optional

class TreeNode:
    def __init__(self, val: int, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_valid_bst(node: Optional[TreeNode]) -> bool:
    # Bug 1
    if node is None:
        return True
    if node.left and node.left.val >= node.val:
        return False
    if node.right and node.right.val <= node.val:
        return False
    return is_valid_bst(node.left) and is_valid_bst(node.right)


def delete_bst(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    if root is None:
        return None
    if val < root.val:
        root.left = delete_bst(root.left, val)
    elif val > root.val:
        root.right = delete_bst(root.right, val)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        # Bug 2: find in-order successor
        successor = root.right
        while successor.right:   # should be .left
            successor = successor.right
        root.val = successor.val
        root.right = delete_bst(root.right, successor.val)
    return root


def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    if root is None:
        return []
    result = []
    queue = [root]
    while queue:
        level = []
        for node in queue:        # Bug 3: iterating over queue while modifying it
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            queue.remove(node)
        result.append(level)
    return result
```

---

## Code quality — 15 pts

Your submission will be evaluated on:

| Criterion | Points |
|---|---|
| Type hints on all function signatures | 4 |
| `logging` used instead of `print` | 3 |
| Tests cover: insert, search, delete (all 3 cases), validate, heap push/pop, edge cases (empty, single node) | 5 |
| No bare `except`, no global mutable state | 3 |

---

## Rubric (pass ≥ 75 / 100)

| Dimension | Weight |
|---|---|
| Independent implementation (Part A + B + C) | 40% |
| Conceptual understanding (questions 1–7) | 25% |
| Debugging & problem-solving (debug challenge) | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-03/assessment.md
