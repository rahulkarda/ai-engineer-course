# Solutions — Week 03 · Day 7 (Warm-ups)

## Warm-up 1 — BSTIterator

```python
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = field(default=None, repr=False)
    right: Optional["TreeNode"] = field(default=None, repr=False)


class BSTIterator:
    """
    Iterates over a BST in inorder (sorted) order.
    Uses an explicit stack — amortised O(h) per __next__ call.
    """

    def __init__(self, root: Optional[TreeNode]) -> None:
        self._stack: list[TreeNode] = []
        self._push_left(root)

    def _push_left(self, node: Optional[TreeNode]) -> None:
        while node:
            self._stack.append(node)
            node = node.left

    def __iter__(self) -> "BSTIterator":
        return self

    def __next__(self) -> int:
        if not self._stack:
            raise StopIteration
        node = self._stack.pop()
        self._push_left(node.right)
        return node.val


# Usage:
# root = ...  (BST)
# for val in BSTIterator(root):
#     log.info(val)
```

**Complexity**: each node is pushed and popped exactly once across all `__next__` calls → O(n) total, O(h) amortised per call. Space: O(h) for the stack.

---

## Warm-up 2 — Heap merge

```python
from src.heap import MinHeap  # or inline the class

def merge_heaps(h1: MinHeap, h2: MinHeap) -> MinHeap:
    """Merge two MinHeaps into a new one. O(n + m)."""
    combined = h1._data + h2._data
    return MinHeap.heapify(combined)
```

**Complexity**: concatenation O(n+m), then heapify O(n+m). Total: O(n+m).
Cannot do better than O(n+m) with an array-backed heap because we must read all elements. Fibonacci heaps support O(1) amortised merge, but Python's standard heap is array-backed.

---

## Warm-up 3 — BST range sum

```python
def range_sum(root: Optional[TreeNode], lo: int, hi: int) -> int:
    """Sum of all values in [lo, hi]. Prune branches outside range."""
    if root is None:
        return 0
    total = 0
    if lo <= root.val <= hi:
        total += root.val
    # Only recurse left if there might be valid values there
    if root.val > lo:
        total += range_sum(root.left, lo, hi)
    # Only recurse right if there might be valid values there
    if root.val < hi:
        total += range_sum(root.right, lo, hi)
    return total
```

**Pruning**: if `root.val <= lo`, nothing in the left subtree can be ≥ lo (all smaller), so we skip it. If `root.val >= hi`, skip the right. This avoids O(n) traversal when the range is narrow.
