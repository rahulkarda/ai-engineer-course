# Phase 2 · Week 03 · Day 4 — BST Validation & Balanced Trees (AVL/Red-Black Overview)

> Objective: Validate BST invariants, understand balance, and know when to reach for self-balancing trees.  ·  Time: ~1 hr  ·  Prereqs: [[day-3 (BST operations)]]

## Concept

### BST Validation

A common interview trap: checking only `node.left.val < node.val < node.right.val` is **wrong**. It misses cross-subtree violations.

```
    5
   / \
  1   4    ← 4 < 5 locally OK, but 4 is in the right subtree of 5 — INVALID
     / \
    3   6
```

The correct approach: propagate **min/max bounds** down the tree. At every node, its value must fall within `(lower, upper)` inherited from ancestors.

### Balance

A BST's performance degrades to O(n) when the tree becomes **skewed** — e.g., inserting already-sorted keys builds a linked list. **Balanced BSTs** maintain height O(log n) automatically via **rotations**.

**Height-balanced**: for every node, `|height(left) - height(right)| ≤ 1`.

Two production-grade self-balancing BSTs:

| Tree | Balance condition | Rotations | Use case |
|---|---|---|---|
| **AVL** | Every node height-balanced | 1–2 per insert/delete | Read-heavy workloads |
| **Red-Black** | No two consecutive red nodes; equal black height | ≤ 2 per insert/delete | General-purpose (`std::map`, Java `TreeMap`) |

This lesson covers **concepts and validation**; full AVL/RB implementation is beyond scope (Python's `sortedcontainers.SortedList` covers the practical need).

## How it works internally

### Validation via bounds propagation

```
validate(node, lo=-∞, hi=+∞):
  if node is None: return True
  if not lo < node.val < hi: return False
  return validate(node.left, lo, node.val)
     and validate(node.right, node.val, hi)
```

Each recursive call narrows the legal range. Left children inherit the current node's value as the new upper bound; right children inherit it as the new lower bound.

### AVL rotations (conceptual)

When inserting causes a node's balance factor (`height(left) - height(right)`) to exceed ±1, a **rotation** restores balance in O(1) by rearranging three nodes.

```
Right rotation (LL imbalance):
      z                y
     /      →        /   \
    y               x     z
   /
  x
```

Red-Black trees use a slightly weaker balance guarantee (height ≤ 2 log n) in exchange for fewer rotations on insert/delete — better for write-heavy workloads.

## Worked example

```python
"""
day4_bst_validate.py — Validate BST, compute balance factor, detect skew.
Python 3.12+
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = field(default=None, repr=False)
    right: Optional["TreeNode"] = field(default=None, repr=False)


# ── Correct BST validation ───────────────────────────────────────────────────

def is_valid_bst(
    node: Optional[TreeNode],
    lo: float = -math.inf,
    hi: float = math.inf,
) -> bool:
    """Validate BST by propagating (lo, hi) bounds."""
    if node is None:
        return True
    if not (lo < node.val < hi):
        return False
    return is_valid_bst(node.left, lo, node.val) and \
           is_valid_bst(node.right, node.val, hi)


# ── Height ───────────────────────────────────────────────────────────────────

def height(node: Optional[TreeNode]) -> int:
    if node is None:
        return -1
    return 1 + max(height(node.left), height(node.right))


# ── Balance factor ───────────────────────────────────────────────────────────

def balance_factor(node: Optional[TreeNode]) -> int:
    """height(left) - height(right).  AVL requires |bf| ≤ 1 everywhere."""
    if node is None:
        return 0
    return height(node.left) - height(node.right)


def is_height_balanced(node: Optional[TreeNode]) -> bool:
    """True iff every node has |balance_factor| ≤ 1 (AVL property)."""
    if node is None:
        return True
    if abs(balance_factor(node)) > 1:
        return False
    return is_height_balanced(node.left) and is_height_balanced(node.right)


# ── Demo ─────────────────────────────────────────────────────────────────────

# Valid BST
#     4
#    / \
#   2   6
#  / \
# 1   3
valid_root = TreeNode(4,
    TreeNode(2, TreeNode(1), TreeNode(3)),
    TreeNode(6),
)
log.info("Valid BST?    %s", is_valid_bst(valid_root))   # True
log.info("Balanced?     %s", is_height_balanced(valid_root))  # True

# Invalid BST (cross-subtree violation)
#     5
#    / \
#   1   4   ← 4 should not be in right subtree of 5
#      / \
#     3   6
invalid_root = TreeNode(5,
    TreeNode(1),
    TreeNode(4, TreeNode(3), TreeNode(6)),
)
log.info("Invalid BST?  %s", is_valid_bst(invalid_root))  # False

# Degenerate (right-skewed) BST — worst-case O(n) performance
skewed: Optional[TreeNode] = None
from dataclasses import replace as dc_replace  # noqa: E402
for v in [1, 2, 3, 4, 5]:
    # build manually (inserting sorted builds right-skewed tree)
    if skewed is None:
        skewed = TreeNode(v)
    else:
        # walk to rightmost leaf and attach
        cur = skewed
        while cur.right:
            cur = cur.right
        cur.right = TreeNode(v)

log.info("Skewed height=%d (n=5, optimal=%d)", height(skewed), 2)
log.info("Skewed balanced? %s", is_height_balanced(skewed))  # False


# ── Conceptual AVL right rotation ────────────────────────────────────────────

def rotate_right(z: TreeNode) -> TreeNode:
    """
    Right rotation: z is the unbalanced node (LL case).
         z              y
        /      →       / \\
       y              x   z
      /
     x
    Returns the new subtree root.
    """
    y = z.left
    assert y is not None, "Cannot right-rotate a node with no left child"
    t2 = y.right           # y's right subtree becomes z's left
    y.right = z
    z.left = t2
    return y               # y is now the subtree root


# Demo: LL imbalance at z=3 with left chain 3→2→1
z = TreeNode(3, TreeNode(2, TreeNode(1), None), None)
new_root = rotate_right(z)
log.info("After right rotation: root=%d left=%s right=%s",
         new_root.val,
         new_root.left.val if new_root.left else None,
         new_root.right.val if new_root.right else None)
# root=2, left=1, right=3
```

## Exercises

1. **Iterative validation** — rewrite `is_valid_bst` iteratively using an explicit stack that tracks `(node, lo, hi)` tuples.

2. **Inorder check** — implement a second validator: run inorder traversal and verify the resulting list is strictly increasing. Compare this to the bounds-propagation approach — which is more efficient? Why?

3. **Fix the broken tree** — given the invalid tree from the demo (root=5, left=1, right=4), reconstruct a valid BST containing the same values `{1, 3, 4, 5, 6}`. Show the resulting inorder.

4. **Closest value** — given a BST root and a float `target`, find the node whose value is numerically closest to `target`. Implement in O(h).

5. **Detect BST from preorder** — given a list that is the preorder traversal of *some* binary tree, determine whether it could be the preorder of a BST (without reconstructing the tree). *Hint*: use a stack-based O(n) algorithm.

(Solutions: ../solutions/week-03/day-4.md)

## Self-check

- [ ] I can explain why checking only parent–child relationships is insufficient for BST validation
- [ ] I can implement bounds-propagation validation without reference
- [ ] I can define height-balanced and state the AVL balance factor condition
- [ ] I can describe an AVL right rotation with a diagram
- [ ] I can explain the trade-off between AVL and Red-Black trees

## Supplementary links

- [LeetCode 98 — Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
- [AVL tree visualization — USFCA](https://www.cs.usfca.edu/~galles/visualization/AVLtree.html)
- [Red-Black tree — Wikipedia](https://en.wikipedia.org/wiki/Red%E2%80%93black_tree)
- [`sortedcontainers` — Python's practical sorted BST](https://grantjenks.com/docs/sortedcontainers/)
