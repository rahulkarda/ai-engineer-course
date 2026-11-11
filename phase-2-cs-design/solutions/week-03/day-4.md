# Solutions — Week 03 · Day 4

## Exercise 1 — Iterative validation

```python
import math
from typing import Optional

def is_valid_bst_iter(root: Optional[TreeNode]) -> bool:
    """Iterative BST validation using explicit stack with bounds."""
    if root is None:
        return True
    stack: list[tuple[Optional[TreeNode], float, float]] = [
        (root, -math.inf, math.inf)
    ]
    while stack:
        node, lo, hi = stack.pop()
        if node is None:
            continue
        if not (lo < node.val < hi):
            return False
        stack.append((node.left, lo, node.val))
        stack.append((node.right, node.val, hi))
    return True
```

Each stack entry carries the (lo, hi) bounds for that subtree — equivalent to what the recursion passes via parameters.

## Exercise 2 — Inorder check vs bounds propagation

```python
def is_valid_bst_inorder(root: Optional[TreeNode]) -> bool:
    """Validate by checking inorder is strictly increasing."""
    prev: list[Optional[int]] = [None]  # mutable container for nonlocal state

    def _inorder(node: Optional[TreeNode]) -> bool:
        if node is None:
            return True
        if not _inorder(node.left):
            return False
        if prev[0] is not None and node.val <= prev[0]:
            return False
        prev[0] = node.val
        return _inorder(node.right)

    return _inorder(root)
```

**Comparison**:
- Both are O(n) time and O(h) space.
- Inorder approach does the same total work.
- Bounds propagation **can short-circuit earlier** in practice: it detects the violation the moment it reaches the invalid node, without needing to visit left siblings first. Inorder must reach the violation in sorted order — which might require traversing more nodes.
- Bounds propagation is the canonical interview answer.

## Exercise 3 — Fix the broken tree

The invalid tree has root=5, right=TreeNode(4, left=3, right=6). Values: {1, 3, 4, 5, 6}.

A valid BST containing {1, 3, 4, 5, 6}:

```python
valid = TreeNode(4,
    TreeNode(3, TreeNode(1), None),
    TreeNode(5, None, TreeNode(6)),
)
# Inorder: [1, 3, 4, 5, 6]  ✓
```

Or sorted-array approach:
```python
valid2 = sorted_array_to_bst([1, 3, 4, 5, 6])
# Inorder: [1, 3, 4, 5, 6]  ✓
```

## Exercise 4 — Closest value

```python
def closest_value(root: Optional[TreeNode], target: float) -> Optional[int]:
    """Return BST value numerically closest to target. O(h)."""
    if root is None:
        return None
    closest: int = root.val
    node: Optional[TreeNode] = root
    while node:
        if abs(node.val - target) < abs(closest - target):
            closest = node.val
        if target < node.val:
            node = node.left
        elif target > node.val:
            node = node.right
        else:
            return node.val   # exact match
    return closest
```

At each step we descend toward `target` (using BST property), updating `closest` whenever we find a nearer value. Because we follow the BST search path, we visit O(h) nodes — we may miss nodes in the other subtree, but they are guaranteed to be farther from `target` than any node on the path.

## Exercise 5 — Detect BST from preorder

```python
def is_preorder_of_bst(preorder: list[int]) -> bool:
    """
    Stack-based O(n) check: in preorder of a BST, once we go right
    (value increases past a previous value), all subsequent values
    must be greater than the lower bound set by that point.
    """
    stack: list[int] = []
    lower_bound = float("-inf")
    for val in preorder:
        if val < lower_bound:
            return False
        # pop all stack values smaller than current (we're entering a right subtree)
        while stack and stack[-1] < val:
            lower_bound = stack.pop()
        stack.append(val)
    return True
```

**Intuition**: in a valid BST preorder, once you step into a right subtree (current value > parent), you can never go back below the parent's value. The stack tracks the "left spine" (ancestors we entered from the left). When `val > stack[-1]`, we popped into a right subtree and set `lower_bound`.
