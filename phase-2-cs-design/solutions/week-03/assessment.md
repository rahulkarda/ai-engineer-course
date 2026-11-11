# Solutions — Week 03 Assessment

## Conceptual answers

**1. BST invariant & invalid example**

The BST invariant: for every node `n`, every value in `n`'s left subtree is strictly less than `n.val`, and every value in `n`'s right subtree is strictly greater.

Invalid example:
```
    5
   / \
  1   4    ← 4 is in right subtree of 5 but 4 < 5 — INVALID
     / \
    3   6
```
Locally `4 > 1` (left child) and `3 < 4 < 6` — each parent-child pair looks fine. But the cross-subtree constraint is violated: 4 (and 3) are in the right subtree of 5, yet both are less than 5.

**2. Traversal use cases**

- **Inorder** (L→Root→R): yields sorted output for a BST; used for sorted iteration.
- **Preorder** (Root→L→R): serialize/copy a tree; the root is encoded first enabling reconstruction.
- **Postorder** (L→R→Root): evaluate expression trees (operands before operator); safe tree deletion (children before parent).

**3. BST from [8,4,12,2,6,10,14]**

```
        8
       / \
      4  12
     / \ / \
    2  6 10 14
```

Height = 2. It is height-balanced: every node's subtree heights differ by ≤ 1.

**4. Heap in array**

The heap property: parent ≤ both children (min-heap). Array storage avoids pointer overhead — a complete binary tree maps perfectly to a contiguous array with no wasted space, giving O(1) index arithmetic:
- parent(i) = (i−1) // 2
- left(i) = 2i + 1
- right(i) = 2i + 2

**5. insert O(log n) + get_min O(1)**

A **min-heap**. The root is always the minimum (O(1) peek); insert is O(log n) sift-up. For O(log n) `delete_arbitrary`: use a **heap + hash-map** (lazy deletion or indexed heap). Python's `heapq` does not support arbitrary deletion natively; use `sortedcontainers.SortedList` or a custom indexed heap.

**6. Two-children delete uses in-order successor**

The in-order successor (minimum of right subtree) is the smallest value that is still larger than the node to delete — replacing the deleted node with it preserves the BST invariant. The successor is the leftmost node of the right subtree, so it has no left child. Deleting it therefore falls into case 1 (leaf) or case 2 (one right child) — both simple.

**7. heapify vs. n×heappush**

n×heappush: each of n pushes costs O(log n) → total O(n log n).
heapify: works bottom-up. Leaves are already valid heaps. Internal nodes near the bottom have small subtrees (cheap sift-down); nodes near the root have large subtrees but there are few of them (exponentially fewer). The sum of all sift-down costs is a geometric series summing to O(n).

---

## Coding challenge solutions

### Part A — BST class

```python
"""assessment_solution.py"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class _Node:
    val: int
    left: Optional["_Node"] = field(default=None, repr=False)
    right: Optional["_Node"] = field(default=None, repr=False)


class BST:
    def __init__(self) -> None:
        self._root: Optional[_Node] = None

    # ── insert ─────────────────────────────────────────────────────────────
    def insert(self, val: int) -> None:
        log.debug("BST.insert(%d)", val)
        self._root = self._insert(self._root, val)

    def _insert(self, node: Optional[_Node], val: int) -> _Node:
        if node is None:
            return _Node(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        return node

    # ── search ─────────────────────────────────────────────────────────────
    def search(self, val: int) -> bool:
        node = self._root
        while node:
            if val == node.val:
                return True
            node = node.left if val < node.val else node.right
        return False

    # ── delete ─────────────────────────────────────────────────────────────
    def delete(self, val: int) -> None:
        log.debug("BST.delete(%d)", val)
        self._root = self._delete(self._root, val)

    def _delete(self, node: Optional[_Node], val: int) -> Optional[_Node]:
        if node is None:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = self._min_node(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
        return node

    def _min_node(self, node: _Node) -> _Node:
        while node.left:
            node = node.left
        return node

    # ── inorder ────────────────────────────────────────────────────────────
    def inorder(self) -> list[int]:
        result: list[int] = []
        def _traverse(n: Optional[_Node]) -> None:
            if n:
                _traverse(n.left)
                result.append(n.val)
                _traverse(n.right)
        _traverse(self._root)
        return result

    # ── is_valid ───────────────────────────────────────────────────────────
    def is_valid(self) -> bool:
        def _check(n: Optional[_Node], lo: float, hi: float) -> bool:
            if n is None:
                return True
            if not (lo < n.val < hi):
                return False
            return _check(n.left, lo, n.val) and _check(n.right, n.val, hi)
        return _check(self._root, -math.inf, math.inf)

    # ── height ─────────────────────────────────────────────────────────────
    def height(self) -> int:
        def _h(n: Optional[_Node]) -> int:
            if n is None:
                return -1
            return 1 + max(_h(n.left), _h(n.right))
        return _h(self._root)
```

### Part B — PriorityQueue

```python
class PriorityQueue:
    """Min-heap priority queue with FIFO tie-breaking. No heapq."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, str]] = []  # (priority, counter, item)
        self._counter = 0

    def push(self, priority: int, item: str) -> None:
        import heapq as _hq
        # Using our own MinHeap from day5; here inline for self-containedness:
        entry = (priority, self._counter, item)
        self._counter += 1
        self._heap.append(entry)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> tuple[int, str]:
        if not self._heap:
            raise IndexError("pop from empty PriorityQueue")
        self._heap[0], self._heap[-1] = self._heap[-1], self._heap[0]
        pri, _, item = self._heap.pop()
        if self._heap:
            self._sift_down(0)
        return pri, item

    def peek(self) -> tuple[int, str]:
        if not self._heap:
            raise IndexError("peek from empty PriorityQueue")
        pri, _, item = self._heap[0]
        return pri, item

    def __len__(self) -> int:
        return len(self._heap)

    def _sift_up(self, i: int) -> None:
        d = self._heap
        while i > 0:
            p = (i - 1) // 2
            if d[i] < d[p]:
                d[i], d[p] = d[p], d[i]
                i = p
            else:
                break

    def _sift_down(self, i: int) -> None:
        d = self._heap
        n = len(d)
        while True:
            smallest = i
            l, r = 2*i+1, 2*i+2
            if l < n and d[l] < d[smallest]:
                smallest = l
            if r < n and d[r] < d[smallest]:
                smallest = r
            if smallest == i:
                break
            d[i], d[smallest] = d[smallest], d[i]
            i = smallest
```

### Part C — Closest pair

```python
def closest_pair_bst(root: Optional[_Node], target: float) -> int:
    closest = root.val
    node: Optional[_Node] = root
    while node:
        if abs(node.val - target) < abs(closest - target):
            closest = node.val
        if target < node.val:
            node = node.left
        elif target > node.val:
            node = node.right
        else:
            return node.val
    return closest
```

---

## Debugging challenge — three bugs

**Bug 1** (in `is_valid_bst`): Only checks direct parent-child relationships. A node in the right subtree could be less than its grandparent. Fix: propagate bounds.

```python
# Buggy:
def is_valid_bst(node):
    if node.left and node.left.val >= node.val: return False
    if node.right and node.right.val <= node.val: return False
    return is_valid_bst(node.left) and is_valid_bst(node.right)

# Fixed:
def is_valid_bst(node, lo=-math.inf, hi=math.inf):
    if node is None: return True
    if not (lo < node.val < hi): return False
    return is_valid_bst(node.left, lo, node.val) and \
           is_valid_bst(node.right, node.val, hi)
```

**Bug 2** (in `delete_bst`): The in-order successor is the **leftmost** (minimum) of the right subtree, not the rightmost. `while successor.right` finds the maximum (wrong).

```python
# Buggy:
while successor.right:
    successor = successor.right

# Fixed:
while successor.left:
    successor = successor.left
```

**Bug 3** (in `level_order`): Iterating over `queue` with a `for` loop while calling `queue.append` and `queue.remove` inside the loop causes incorrect behaviour — the for-loop iterates over elements added in the current level, mixing levels. Also, `list.remove` is O(n). Fix: use `deque` with a level-size snapshot.

```python
# Fixed:
from collections import deque

def level_order(root):
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        level_size = len(queue)   # snapshot
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

---

## pytest tests (assessment_solution.py — bottom section)

```python
import pytest

def test_bst_insert_inorder():
    bst = BST()
    for v in [5, 3, 7, 1, 4, 6, 8]:
        bst.insert(v)
    assert bst.inorder() == [1, 3, 4, 5, 6, 7, 8]

def test_bst_delete_leaf():
    bst = BST()
    for v in [5, 3, 7]: bst.insert(v)
    bst.delete(3)
    assert bst.inorder() == [5, 7]

def test_bst_delete_two_children():
    bst = BST()
    for v in [5, 3, 7, 1, 4]: bst.insert(v)
    bst.delete(3)
    assert bst.inorder() == [1, 4, 5, 7]
    assert bst.is_valid()

def test_bst_is_valid_false():
    # Manually construct an invalid BST
    root = _Node(5, _Node(1), _Node(4, _Node(3), _Node(6)))
    bst = BST()
    bst._root = root
    assert not bst.is_valid()

def test_pq_order():
    pq = PriorityQueue()
    pq.push(10, "low")
    pq.push(1, "critical")
    pq.push(5, "medium")
    assert pq.pop() == (1, "critical")
    assert pq.pop() == (5, "medium")
    assert pq.pop() == (10, "low")

def test_pq_fifo_tiebreak():
    pq = PriorityQueue()
    pq.push(1, "first")
    pq.push(1, "second")
    assert pq.pop()[1] == "first"
    assert pq.pop()[1] == "second"

def test_closest():
    root = _Node(5, _Node(3, _Node(1), _Node(4)), _Node(7, _Node(6), _Node(8)))
    assert closest_pair_bst(root, 4.6) == 5
    assert closest_pair_bst(root, 3.4) == 3
```
