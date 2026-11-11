# Phase 2 · Week 03 · Day 3 — Binary Search Trees: Insert, Search, Delete

> Objective: Implement a BST with correct insert, search, and delete operations.  ·  Time: ~1 hr  ·  Prereqs: [[day-1 (binary trees)], [day-2 (traversals)]]

## Concept

A **Binary Search Tree (BST)** is a binary tree satisfying the **BST invariant**:

> For every node `n`: all values in `n.left` subtree < `n.val` < all values in `n.right` subtree.

This ordering turns random-access search into a O(h) operation where `h` is height — O(log n) for balanced trees, O(n) for degenerate ones.

### Core operations

| Operation | Average | Worst (skewed) |
|---|---|---|
| Search | O(log n) | O(n) |
| Insert | O(log n) | O(n) |
| Delete | O(log n) | O(n) |
| Inorder traversal | O(n) | O(n) |

### Delete — three cases

Deletion is the trickiest operation. Given the node to remove:

1. **Leaf** (no children): set parent's pointer to `None`.
2. **One child**: replace the node with its sole child.
3. **Two children**: find the **in-order successor** (smallest value in right subtree), copy its value into the current node, then delete the successor (which has at most one child — no left child by definition).

## How it works internally

Python's BST node is a heap object with three fields (`val`, `left`, `right`). Traversal follows pointers without copying data — only references move.

**In-order successor**: the leftmost node of the right subtree. It has no left child, so deleting it falls into case 1 or 2.

Why not use the **in-order predecessor** (max of left subtree)? You can — it's equally valid and produces the same BST invariant. Convention varies; this lesson uses the successor.

## Worked example

```python
"""
day3_bst.py — BST with insert, search, delete.
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
    val: int
    left: Optional["TreeNode"] = field(default=None, repr=False)
    right: Optional["TreeNode"] = field(default=None, repr=False)


# ── Insert ──────────────────────────────────────────────────────────────────

def insert(root: Optional[TreeNode], val: int) -> TreeNode:
    """Insert val; return the (possibly new) root. Duplicates ignored."""
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    # val == root.val: duplicate, no change
    return root


# ── Search ──────────────────────────────────────────────────────────────────

def search(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """Return the node with val, or None if not found."""
    if root is None or root.val == val:
        return root
    if val < root.val:
        return search(root.left, val)
    return search(root.right, val)


# ── Delete ──────────────────────────────────────────────────────────────────

def _min_node(node: TreeNode) -> TreeNode:
    """Return the leftmost (minimum) node in a non-empty subtree."""
    current = node
    while current.left:
        current = current.left
    return current


def delete(root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
    """Delete val from BST; return the (possibly new) root."""
    if root is None:
        return None  # val not found

    if val < root.val:
        root.left = delete(root.left, val)
    elif val > root.val:
        root.right = delete(root.right, val)
    else:
        # Found the node to delete
        if root.left is None:
            return root.right          # Case 1 or 2 (no left child)
        if root.right is None:
            return root.left           # Case 2 (no right child)

        # Case 3: two children — replace with in-order successor
        successor = _min_node(root.right)
        root.val = successor.val                  # overwrite value
        root.right = delete(root.right, successor.val)  # remove successor

    return root


# ── Inorder helper (to verify BST property) ─────────────────────────────────

def inorder(root: Optional[TreeNode]) -> list[int]:
    if root is None:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)


# ── Demo ────────────────────────────────────────────────────────────────────

values = [5, 3, 7, 1, 4, 6, 8]
root: Optional[TreeNode] = None
for v in values:
    root = insert(root, v)

log.info("Inorder after insert: %s", inorder(root))  # [1,3,4,5,6,7,8]

node = search(root, 4)
log.info("Search 4: found=%s val=%s", node is not None, node.val if node else None)

node = search(root, 9)
log.info("Search 9: found=%s", node is not None)  # False

# Delete leaf (1)
root = delete(root, 1)
log.info("After deleting 1: %s", inorder(root))  # [3,4,5,6,7,8]

# Delete node with one child
root = delete(root, 3)
log.info("After deleting 3: %s", inorder(root))  # [4,5,6,7,8]

# Delete node with two children
root = delete(root, 7)
log.info("After deleting 7: %s", inorder(root))  # [4,5,6,8]
```

### Deletion trace — removing 7 from `[5,3,7,1,4,6,8]`

```
delete(root=5, val=7)
  7 > 5 → root.right = delete(node=7, val=7)
    node.val == val, two children
    successor = _min_node(node.right=8) → 8 (no left child)
    node.val = 8
    node.right = delete(node.right=8, val=8)
      node.val == val, no children → return None
    node.right = None
    return node (val=8, left=6, right=None)
  root.right = node(8, left=6)
return root(5)
Inorder: [1,3,4,5,6,8]  ✓
```

## Exercises

1. **Iterative insert** — rewrite `insert` without recursion using a `while` loop. Which approach is clearer? Which is more stack-safe?

2. **Floor and ceiling** — implement `floor(root, val) -> int | None` (largest value ≤ val) and `ceil(root, val) -> int | None` (smallest value ≥ val).

3. **Rank** — implement `rank(root, val) -> int` that returns how many values in the BST are strictly less than `val`.

4. **Sorted array → balanced BST** — given a sorted list, build the most balanced BST possible. *Hint*: pick the middle element as root.

5. **Delete stress test** — insert integers 1..100 in random order, then delete them in another random order. After each deletion, assert `inorder()` is sorted and contains no duplicates. Write this as a `pytest` test using `random.seed(42)`.

(Solutions: ../solutions/week-03/day-3.md)

## Self-check

- [ ] I can state the BST invariant precisely
- [ ] I can implement insert recursively
- [ ] I can implement the three deletion cases without reference
- [ ] I can explain what in-order successor is and why it has at most one child
- [ ] I know BST operations are O(h) and understand when h = O(n)

## Supplementary links

- [Visualgo — BST operations](https://visualgo.net/en/bst)
- [LeetCode 701 — Insert into a BST](https://leetcode.com/problems/insert-into-a-binary-search-tree/)
- [LeetCode 450 — Delete Node in a BST](https://leetcode.com/problems/delete-node-in-a-bst/)
