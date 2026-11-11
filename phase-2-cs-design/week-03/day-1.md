# Phase 2 · Week 03 · Day 1 — Binary Trees & Recursive Traversals

> Objective: Understand binary tree structure and implement pre/in/post-order traversals recursively.  ·  Time: ~1 hr  ·  Prereqs: [[day-7 of week-02 (linked lists)]]

## Concept

A **binary tree** is a hierarchical data structure where each node holds a value and has at most two children: a **left** child and a **right** child. Trees model hierarchies — file systems, expression ASTs, HTML DOMs, decision trees.

Key vocabulary:
- **Root**: top node (no parent)
- **Leaf**: node with no children
- **Height**: longest root-to-leaf path (edges)
- **Depth**: distance from root to a node
- **Subtree**: a node plus all its descendants

A tree with `n` nodes has exactly `n − 1` edges.

### Why recursion fits trees

Every non-empty binary tree is either:
1. A leaf (base case), or
2. A root + left subtree + right subtree (recursive case)

This self-similar structure makes recursive algorithms natural and concise.

## How it works internally

Each node is a small object on the heap. Python stores object references, so `node.left` is just a pointer (8 bytes) to another node object. The tree itself has no contiguous memory block — it is a web of heap pointers.

**Call stack depth = tree height.** A balanced tree of `n` nodes has height ≈ `log₂(n)`, so recursion depth is safe. A degenerate (all-left) tree of `n` nodes has height `n` and can hit Python's default `sys.getrecursionlimit()` (~1000). Day 2 covers iterative alternatives.

## Worked example

```python
"""
day1_binary_tree.py — Binary tree node and recursive traversals.
Python 3.12+
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A single node in a binary tree."""
    val: int
    left: Optional["TreeNode"] = field(default=None, repr=False)
    right: Optional["TreeNode"] = field(default=None, repr=False)


def preorder(node: Optional[TreeNode]) -> list[int]:
    """Root → Left → Right."""
    if node is None:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)


def inorder(node: Optional[TreeNode]) -> list[int]:
    """Left → Root → Right.  Gives sorted order for a BST."""
    if node is None:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)


def postorder(node: Optional[TreeNode]) -> list[int]:
    """Left → Right → Root.  Useful for deletion / expression evaluation."""
    if node is None:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]


def height(node: Optional[TreeNode]) -> int:
    """Height = max depth of any leaf.  Empty tree = -1 by convention."""
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))


# Build the example tree:
#
#         4
#        / \
#       2   6
#      / \ / \
#     1  3 5  7
#
root = TreeNode(4,
    TreeNode(2, TreeNode(1), TreeNode(3)),
    TreeNode(6, TreeNode(5), TreeNode(7)),
)

log.info("Preorder  (root-left-right): %s", preorder(root))   # [4,2,1,3,6,5,7]
log.info("Inorder   (left-root-right): %s", inorder(root))    # [1,2,3,4,5,6,7]
log.info("Postorder (left-right-root): %s", postorder(root))  # [1,3,2,5,7,6,4]
log.info("Height: %d", height(root))                           # 2
```

Run it:
```
python day1_binary_tree.py
```

Expected output:
```
Preorder  (root-left-right): [4, 2, 1, 3, 6, 5, 7]
Inorder   (left-root-right): [1, 2, 3, 4, 5, 6, 7]
Postorder (left-right-root): [1, 3, 2, 5, 7, 6, 4]
Height: 2
```

### Trace — preorder on the left subtree of root

```
preorder(4)
  [4] + preorder(2) + preorder(6)
         [2] + preorder(1) + preorder(3)
                [1] + [] + []   →  [1]
                         [3] + [] + []  →  [3]
         → [2,1,3]
  → [4,2,1,3] + [6,5,7]
  → [4,2,1,3,6,5,7]
```

### Memory model

```
TreeNode(4)
├─ .left  → TreeNode(2)
│           ├─ .left  → TreeNode(1)  # leaf
│           └─ .right → TreeNode(3)  # leaf
└─ .right → TreeNode(6)
            ├─ .left  → TreeNode(5)  # leaf
            └─ .right → TreeNode(7)  # leaf
```

## Exercises

1. **Count nodes** — write `count_nodes(node) -> int` that returns the total number of nodes in the tree using recursion. What is the base case?

2. **Leaf count** — write `count_leaves(node) -> int` that counts only leaf nodes.

3. **Mirror** — write `mirror(node) -> TreeNode | None` that returns a new tree that is the mirror image (every left/right pair swapped) of the input. Do not mutate the original.

4. **Path sum** — given a target integer, write `has_path_sum(node, target) -> bool` that returns `True` if any root-to-leaf path has values summing to `target`.

5. **Serialize** — write `serialize(node) -> str` that produces a string representation of the tree using preorder traversal, using `"null"` for missing children (e.g., `"4,2,1,null,null,3,null,null,6,5,null,null,7,null,null"`). Why is preorder useful for serialization?

(Solutions: ../solutions/week-03/day-1.md)

## Self-check

- [ ] I can draw a binary tree and label root, leaf, height, depth from memory
- [ ] I can trace preorder, inorder, and postorder by hand on a 7-node tree
- [ ] I can implement all three traversals recursively without reference
- [ ] I understand why call-stack depth equals tree height
- [ ] I can explain why inorder on a BST yields sorted output

## Supplementary links

- [Visualgo — binary tree](https://visualgo.net/en/bst)
- [Python `dataclasses` docs](https://docs.python.org/3.12/library/dataclasses.html)
- [Sedgewick & Wayne — *Algorithms* Ch. 3.2 (Binary Search Trees)](https://algs4.cs.princeton.edu/32bst/)
