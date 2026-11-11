# Phase 2 · Week 03 · Day 2 — Iterative Traversals & Level-Order (BFS)

> Objective: Implement all four traversals iteratively using an explicit stack or queue.  ·  Time: ~1 hr  ·  Prereqs: [[day-1 (binary trees & recursive traversals)]]

## Concept

Recursive traversals are elegant but limited by Python's call-stack depth (~1000 default). **Iterative traversals** replace the implicit call stack with an explicit one (a `list` used as a stack) or a **queue** (for level-order / BFS).

Four traversal orders recap:

| Order | Visit sequence | Use case |
|---|---|---|
| Preorder | Root → L → R | Serialize tree, copy tree |
| Inorder | L → Root → R | Sorted output from BST |
| Postorder | L → R → Root | Delete tree, eval expressions |
| Level-order | Row by row, L→R | Shortest-path, nearest-neighbour |

Level-order is also called **Breadth-First Search (BFS)** on a tree — it visits all nodes at depth `d` before any at depth `d+1`.

## How it works internally

### Stack-based (pre/in/post)

The call stack holds **activation records** — the function's local variables and where to return. An explicit stack stores exactly the same information but as plain objects you control.

For **preorder**: push root; loop — pop node, record value, push right then left (so left is processed first).

For **inorder**: we must go deep-left before recording; use a pointer that walks left, pushing nodes until null, then pop-and-record, then move to right.

For **postorder**: trickier — one approach is reverse-preorder: visit root-right-left and reverse the result.

### Queue-based (level-order)

A FIFO queue ensures we exhaust an entire level before descending. Python's `collections.deque` gives O(1) `appendleft`/`popleft`.

## Worked example

```python
"""
day2_iterative_traversals.py — All four traversals, iteratively.
Python 3.12+
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = field(default=None, repr=False)
    right: Optional["TreeNode"] = field(default=None, repr=False)


# ── Iterative Preorder ──────────────────────────────────────────────────────

def preorder_iter(root: Optional[TreeNode]) -> list[int]:
    """Root → L → R using explicit stack."""
    if root is None:
        return []
    result: list[int] = []
    stack: list[TreeNode] = [root]
    while stack:
        node = stack.pop()         # process current node
        result.append(node.val)
        if node.right:             # push right first so left is popped first
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result


# ── Iterative Inorder ───────────────────────────────────────────────────────

def inorder_iter(root: Optional[TreeNode]) -> list[int]:
    """L → Root → R: walk left chain, then record, then go right."""
    result: list[int] = []
    stack: list[TreeNode] = []
    current: Optional[TreeNode] = root
    while current or stack:
        # descend to leftmost node
        while current:
            stack.append(current)
            current = current.left
        # backtrack
        current = stack.pop()
        result.append(current.val)
        # move to right subtree
        current = current.right
    return result


# ── Iterative Postorder ─────────────────────────────────────────────────────

def postorder_iter(root: Optional[TreeNode]) -> list[int]:
    """L → R → Root via reverse of modified preorder."""
    if root is None:
        return []
    result: list[int] = []
    stack: list[TreeNode] = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)    # Root first
        if node.left:              # push left before right (reversed)
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    result.reverse()               # reverse gives L→R→Root
    return result


# ── Level-Order (BFS) ───────────────────────────────────────────────────────

def level_order(root: Optional[TreeNode]) -> list[list[int]]:
    """Returns list of levels; each level is a list of node values."""
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)        # snapshot current level width
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result


# ── Demo ────────────────────────────────────────────────────────────────────

#         4
#        / \
#       2   6
#      / \ / \
#     1  3 5  7

root = TreeNode(4,
    TreeNode(2, TreeNode(1), TreeNode(3)),
    TreeNode(6, TreeNode(5), TreeNode(7)),
)

log.info("Preorder  (iter): %s", preorder_iter(root))   # [4,2,1,3,6,5,7]
log.info("Inorder   (iter): %s", inorder_iter(root))    # [1,2,3,4,5,6,7]
log.info("Postorder (iter): %s", postorder_iter(root))  # [1,3,2,5,7,6,4]
log.info("Level-order:      %s", level_order(root))     # [[4],[2,6],[1,3,5,7]]
```

### Level-order trace

```
queue: [4]
  pop 4 → level=[4]; push 2,6 → queue:[2,6]
  level 0: [4]

queue: [2, 6]
  pop 2 → push 1,3;  pop 6 → push 5,7
  level 1: [2, 6]

queue: [1, 3, 5, 7]
  pop each, no children to push
  level 2: [1, 3, 5, 7]
```

## Exercises

1. **Zigzag level order** — modify `level_order` so even-indexed levels are reversed (right-to-left). Return the same `list[list[int]]` structure.

2. **Right-side view** — return the list of values visible when you look at the tree from the right side (last node in each level). Implement iteratively with BFS.

3. **Max width** — using BFS, find the maximum number of nodes at any single level. What is the max width of a complete binary tree with `n` nodes?

4. **Iterative inorder — trace** — trace `inorder_iter` step by step on the tree `[3, 1, 4]` (root=3, left=1, right=4). Write out the stack contents and `current` pointer at each iteration.

5. **Convert to BST array** — using level-order, produce a list where `result[i]` is the value at position `i` in a 0-indexed BFS array (use `-1` for absent nodes). This is the "array representation" of a binary tree.

(Solutions: ../solutions/week-03/day-2.md)

## Self-check

- [ ] I can implement iterative preorder without reference
- [ ] I can implement iterative inorder (pointer-walking pattern)
- [ ] I can implement BFS level-order with a deque
- [ ] I understand why level-order needs a queue while DFS needs a stack
- [ ] I can explain the `level_size` snapshot trick to separate levels

## Supplementary links

- [`collections.deque` docs](https://docs.python.org/3.12/library/collections.html#collections.deque)
- [LeetCode 102 — Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/)
- [LeetCode 94 — Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)
