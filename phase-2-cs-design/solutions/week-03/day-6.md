# Solutions — Week 03 · Day 6 (Problem Set)

```python
"""
day6_solutions.py — Complete solutions for all 10 tree problems.
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
```

---

## Problem 1 — Diameter

```python
def diameter(root: Optional[TreeNode]) -> int:
    """O(n): compute height via DFS and track max diameter."""
    max_diam: list[int] = [0]   # mutable container (avoids nonlocal keyword)

    def _height(node: Optional[TreeNode]) -> int:
        if node is None:
            return -1
        lh = _height(node.left)
        rh = _height(node.right)
        # path through this node spans (lh+1) + (rh+1) edges
        max_diam[0] = max(max_diam[0], lh + rh + 2)
        return 1 + max(lh, rh)

    _height(root)
    return max_diam[0]
```

Key insight: diameter through a node = height(left) + height(right) + 2 (counting edges). Single DFS avoids recomputing heights.

---

## Problem 2 — LCA

```python
def lca_bst(root: Optional[TreeNode], p: int, q: int) -> Optional[TreeNode]:
    """Use BST property: if both < root go left; if both > root go right; else root."""
    node = root
    while node:
        if p < node.val and q < node.val:
            node = node.left
        elif p > node.val and q > node.val:
            node = node.right
        else:
            return node   # split point = LCA
    return None


def lca_binary_tree(root: Optional[TreeNode], p: int, q: int) -> Optional[TreeNode]:
    """General tree LCA via post-order DFS."""
    if root is None:
        return None
    if root.val == p or root.val == q:
        return root   # found one of the targets
    left_lca = lca_binary_tree(root.left, p, q)
    right_lca = lca_binary_tree(root.right, p, q)
    if left_lca and right_lca:
        return root   # p and q are in different subtrees → current root is LCA
    return left_lca or right_lca
```

---

## Problem 3 — Zigzag level order

```python
def zigzag_level_order(root: Optional[TreeNode]) -> list[list[int]]:
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    left_to_right = True
    while queue:
        level = [queue.popleft().val for _ in range(len(queue))]
        # re-enqueue children (done separately to avoid queue size change mid-loop)
        # Corrected version:
        result_level: list[int] = []
        for _ in range(len(queue) + len(result_level)):  # use snapshot
            pass

    # Clean implementation:
    result = []
    queue = deque([root])
    left_to_right = True
    while queue:
        sz = len(queue)
        level: list[int] = []
        for _ in range(sz):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return result
```

---

## Problem 4 — Maximum path sum

```python
def max_path_sum(root: Optional[TreeNode]) -> int:
    """O(n): at each node, max contribution = node.val + max(0, left_gain, right_gain)."""
    best: list[int] = [root.val if root else 0]

    def _gain(node: Optional[TreeNode]) -> int:
        """Max gain going downward from this node (can be negative → return 0)."""
        if node is None:
            return 0
        left_gain = max(0, _gain(node.left))
        right_gain = max(0, _gain(node.right))
        # path through this node (can't extend upward for both sides simultaneously)
        best[0] = max(best[0], node.val + left_gain + right_gain)
        return node.val + max(left_gain, right_gain)

    _gain(root)
    return best[0]
```

---

## Problem 5 — BST from preorder

```python
def bst_from_preorder(preorder: list[int]) -> Optional[TreeNode]:
    """O(n) using a stack with upper-bound tracking."""
    if not preorder:
        return None
    root = TreeNode(preorder[0])
    stack: list[TreeNode] = [root]
    for val in preorder[1:]:
        node = TreeNode(val)
        if val < stack[-1].val:
            stack[-1].left = node    # left child
        else:
            parent = stack[-1]
            while stack and stack[-1].val < val:
                parent = stack.pop()
            parent.right = node
        stack.append(node)
    return root
```

---

## Problem 6 — Kth smallest (early-stop iterative inorder)

```python
def kth_smallest(root: Optional[TreeNode], k: int) -> int:
    stack: list[TreeNode] = []
    current: Optional[TreeNode] = root
    count = 0
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        count += 1
        if count == k:
            return current.val
        current = current.right
    raise IndexError(f"k={k} exceeds tree size")
```

---

## Problem 7 — Symmetric tree

```python
def is_symmetric_recursive(root: Optional[TreeNode]) -> bool:
    def _mirror(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return a.val == b.val and _mirror(a.left, b.right) and _mirror(a.right, b.left)

    return root is None or _mirror(root.left, root.right)


def is_symmetric_iter(root: Optional[TreeNode]) -> bool:
    if root is None:
        return True
    queue: deque[tuple[Optional[TreeNode], Optional[TreeNode]]] = deque(
        [(root.left, root.right)]
    )
    while queue:
        a, b = queue.popleft()
        if a is None and b is None:
            continue
        if a is None or b is None or a.val != b.val:
            return False
        queue.append((a.left, b.right))
        queue.append((a.right, b.left))
    return True
```

---

## Problem 8 — Flatten binary tree

```python
def flatten(root: Optional[TreeNode]) -> None:
    """In-place: preorder, right-pointer linked list, left=None."""
    if root is None:
        return
    # Find the rightmost node of the left subtree
    if root.left:
        # Find rightmost of left subtree
        rightmost = root.left
        while rightmost.right:
            rightmost = rightmost.right
        # Attach current right subtree to rightmost
        rightmost.right = root.right
        # Move left subtree to right
        root.right = root.left
        root.left = None
    flatten(root.right)
```

---

## Problem 9 — Top K frequent

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """O(n log k): min-heap of size k over (freq, value) pairs."""
    freq = Counter(nums)
    heap: list[tuple[int, int]] = []
    for val, count in freq.items():
        heapq.heappush(heap, (count, val))
        if len(heap) > k:
            heapq.heappop(heap)  # remove least-frequent
    return [val for _, val in heap]
```

---

## Problem 10 — Serialize / Deserialize

```python
def serialize(root: Optional[TreeNode]) -> str:
    parts: list[str] = []
    def _pre(node: Optional[TreeNode]) -> None:
        if node is None:
            parts.append("#")
            return
        parts.append(str(node.val))
        _pre(node.left)
        _pre(node.right)
    _pre(root)
    return ",".join(parts)


def deserialize(data: str) -> Optional[TreeNode]:
    it = iter(data.split(","))
    def _build() -> Optional[TreeNode]:
        token = next(it)
        if token == "#":
            return None
        node = TreeNode(int(token))
        node.left = _build()
        node.right = _build()
        return node
    return _build()
```

**Round-trip test**:
```python
def test_serialize_roundtrip() -> None:
    root = TreeNode(4,
        TreeNode(2, TreeNode(1), TreeNode(3)),
        TreeNode(6, TreeNode(5), TreeNode(7)),
    )
    s = serialize(root)
    restored = deserialize(s)
    assert serialize(restored) == s
```
