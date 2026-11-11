# Solutions — Week 03 Project

Complete reference implementation. Learner should attempt the project independently before consulting this.

---

## `src/bst.py`

```python
"""
bst.py — Full BST implementation.
Python 3.12+
"""

from __future__ import annotations
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = field(default=None, repr=False)
    right: Optional["TreeNode"] = field(default=None, repr=False)


class BST:
    def __init__(self) -> None:
        self._root: Optional[TreeNode] = None

    # ── Mutators ──────────────────────────────────────────────────────────

    def insert(self, val: int) -> None:
        log.debug("insert(%d)", val)
        self._root = self._insert(self._root, val)

    def _insert(self, node: Optional[TreeNode], val: int) -> TreeNode:
        if node is None:
            return TreeNode(val)
        if val < node.val:
            node.left = self._insert(node.left, val)
        elif val > node.val:
            node.right = self._insert(node.right, val)
        # duplicate: ignore
        return node

    def delete(self, val: int) -> None:
        log.debug("delete(%d)", val)
        self._root = self._delete(self._root, val)

    def _delete(self, node: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if node is None:
            return None
        if val < node.val:
            node.left = self._delete(node.left, val)
        elif val > node.val:
            node.right = self._delete(node.right, val)
        else:
            # Three cases
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = self._min_node(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
        return node

    def _min_node(self, node: TreeNode) -> TreeNode:
        while node.left:
            node = node.left
        return node

    def _max_node(self, node: TreeNode) -> TreeNode:
        while node.right:
            node = node.right
        return node

    # ── Queries ───────────────────────────────────────────────────────────

    def search(self, val: int) -> bool:
        node = self._root
        while node:
            if val == node.val:
                return True
            node = node.left if val < node.val else node.right
        return False

    def minimum(self) -> Optional[int]:
        if self._root is None:
            return None
        return self._min_node(self._root).val

    def maximum(self) -> Optional[int]:
        if self._root is None:
            return None
        return self._max_node(self._root).val

    def floor(self, val: int) -> Optional[int]:
        """Largest value ≤ val."""
        result: Optional[int] = None
        node = self._root
        while node:
            if node.val == val:
                return val
            if node.val < val:
                result = node.val
                node = node.right
            else:
                node = node.left
        return result

    def ceil(self, val: int) -> Optional[int]:
        """Smallest value ≥ val."""
        result: Optional[int] = None
        node = self._root
        while node:
            if node.val == val:
                return val
            if node.val > val:
                result = node.val
                node = node.left
            else:
                node = node.right
        return result

    def kth_smallest(self, k: int) -> int:
        """k-th smallest (1-indexed). Early-stop iterative inorder."""
        stack: list[TreeNode] = []
        current: Optional[TreeNode] = self._root
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

    # ── Structural ────────────────────────────────────────────────────────

    def height(self) -> int:
        def _h(n: Optional[TreeNode]) -> int:
            if n is None:
                return -1
            return 1 + max(_h(n.left), _h(n.right))
        return _h(self._root)

    def size(self) -> int:
        def _sz(n: Optional[TreeNode]) -> int:
            if n is None:
                return 0
            return 1 + _sz(n.left) + _sz(n.right)
        return _sz(self._root)

    def is_valid(self) -> bool:
        def _check(n: Optional[TreeNode], lo: float, hi: float) -> bool:
            if n is None:
                return True
            if not (lo < n.val < hi):
                return False
            return _check(n.left, lo, n.val) and _check(n.right, n.val, hi)
        return _check(self._root, -math.inf, math.inf)

    # ── Traversals ────────────────────────────────────────────────────────

    def inorder(self) -> list[int]:
        result: list[int] = []
        def _traverse(n: Optional[TreeNode]) -> None:
            if n:
                _traverse(n.left)
                result.append(n.val)
                _traverse(n.right)
        _traverse(self._root)
        return result

    def level_order(self) -> list[list[int]]:
        if self._root is None:
            return []
        result: list[list[int]] = []
        queue: deque[TreeNode] = deque([self._root])
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
            result.append(level)
        return result

    def __repr__(self) -> str:
        return f"BST(size={self.size()}, height={self.height()})"
```

---

## `src/heap.py`

```python
"""
heap.py — MinHeap and MinHeapPQ.
Python 3.12+  — does NOT use heapq internally.
"""

from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger(__name__)


class MinHeap:
    """Array-backed integer min-heap."""

    def __init__(self) -> None:
        self._data: list[int] = []

    def push(self, val: int) -> None:
        log.debug("MinHeap.push(%d)", val)
        self._data.append(val)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty MinHeap")
        self._data[0], self._data[-1] = self._data[-1], self._data[0]
        val = self._data.pop()
        log.debug("MinHeap.pop() -> %d", val)
        if self._data:
            self._sift_down(0)
        return val

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek from empty MinHeap")
        return self._data[0]

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    @classmethod
    def heapify(cls, values: list[int]) -> "MinHeap":
        h = cls()
        h._data = values[:]
        n = len(h._data)
        for i in range((n - 2) // 2, -1, -1):
            h._sift_down(i)
        return h

    def _sift_up(self, i: int) -> None:
        d = self._data
        while i > 0:
            parent = (i - 1) // 2
            if d[i] < d[parent]:
                d[i], d[parent] = d[parent], d[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        d = self._data
        n = len(d)
        while True:
            smallest = i
            l, r = 2 * i + 1, 2 * i + 2
            if l < n and d[l] < d[smallest]:
                smallest = l
            if r < n and d[r] < d[smallest]:
                smallest = r
            if smallest == i:
                break
            d[i], d[smallest] = d[smallest], d[i]
            i = smallest


class MinHeapPQ:
    """
    Priority queue backed by MinHeap.
    Tuple comparison: (priority, counter, item)
    Lower priority = higher urgency. FIFO tie-breaking via counter.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Any]] = []
        self._counter = 0
        self._heap_impl = _InternalHeap()

    def push(self, priority: int, item: Any) -> None:
        log.debug("PQ.push(priority=%d, item=%r)", priority, item)
        entry = (priority, self._counter, item)
        self._counter += 1
        self._heap.append(entry)
        self._sift_up(len(self._heap) - 1)

    def pop(self) -> tuple[int, Any]:
        if not self._heap:
            raise IndexError("pop from empty MinHeapPQ")
        self._heap[0], self._heap[-1] = self._heap[-1], self._heap[0]
        pri, _, item = self._heap.pop()
        log.debug("PQ.pop() -> (priority=%d, item=%r)", pri, item)
        if self._heap:
            self._sift_down(0)
        return pri, item

    def peek(self) -> tuple[int, Any]:
        if not self._heap:
            raise IndexError("peek from empty MinHeapPQ")
        pri, _, item = self._heap[0]
        return pri, item

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

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


class _InternalHeap:
    """Placeholder — MinHeapPQ manages its own heap directly above."""
    pass
```

---

## `tests/test_bst.py`

```python
"""tests/test_bst.py"""

from __future__ import annotations
import pytest
from src.bst import BST, TreeNode


class TestBSTInsertSearch:
    def test_empty_search(self):
        bst = BST()
        assert not bst.search(5)

    def test_insert_single(self):
        bst = BST()
        bst.insert(10)
        assert bst.search(10)
        assert not bst.search(5)

    def test_inorder_sorted(self):
        bst = BST()
        for v in [5, 3, 7, 1, 4, 6, 8]:
            bst.insert(v)
        assert bst.inorder() == [1, 3, 4, 5, 6, 7, 8]

    def test_duplicate_ignored(self):
        bst = BST()
        bst.insert(5)
        bst.insert(5)
        assert bst.size() == 1


class TestBSTDelete:
    def test_delete_leaf(self):
        bst = BST()
        for v in [5, 3, 7]: bst.insert(v)
        bst.delete(3)
        assert bst.inorder() == [5, 7]
        assert bst.is_valid()

    def test_delete_one_child_left(self):
        bst = BST()
        for v in [5, 3, 7, 2]: bst.insert(v)
        bst.delete(3)   # has only left child (2)
        assert bst.inorder() == [2, 5, 7]
        assert bst.is_valid()

    def test_delete_one_child_right(self):
        bst = BST()
        for v in [5, 3, 7, 4]: bst.insert(v)
        bst.delete(3)   # has only right child (4)
        assert bst.inorder() == [4, 5, 7]
        assert bst.is_valid()

    def test_delete_two_children(self):
        bst = BST()
        for v in [5, 3, 7, 1, 4, 6, 8]: bst.insert(v)
        bst.delete(3)
        assert bst.inorder() == [1, 4, 5, 6, 7, 8]
        assert bst.is_valid()

    def test_delete_root(self):
        bst = BST()
        for v in [5, 3, 7]: bst.insert(v)
        bst.delete(5)
        assert bst.is_valid()
        assert bst.search(3) and bst.search(7)

    def test_delete_nonexistent(self):
        bst = BST()
        bst.insert(5)
        bst.delete(99)   # no-op
        assert bst.inorder() == [5]


class TestBSTValidation:
    def test_valid_bst(self):
        bst = BST()
        for v in [5, 3, 7, 1, 4, 6, 8]: bst.insert(v)
        assert bst.is_valid()

    def test_invalid_cross_subtree(self):
        # Manually build invalid tree
        bst = BST()
        bst._root = TreeNode(5,
            TreeNode(1),
            TreeNode(4, TreeNode(3), TreeNode(6)),
        )
        assert not bst.is_valid()


class TestBSTQueries:
    def setup_method(self):
        self.bst = BST()
        for v in [5, 3, 7, 1, 4, 6, 8]:
            self.bst.insert(v)

    def test_min_max(self):
        assert self.bst.minimum() == 1
        assert self.bst.maximum() == 8

    def test_floor_ceil(self):
        assert self.bst.floor(5) == 5
        assert self.bst.floor(2) == 1
        assert self.bst.floor(0) is None
        assert self.bst.ceil(5) == 5
        assert self.bst.ceil(2) == 3
        assert self.bst.ceil(9) is None

    def test_kth_smallest(self):
        assert self.bst.kth_smallest(1) == 1
        assert self.bst.kth_smallest(4) == 5
        assert self.bst.kth_smallest(7) == 8

    def test_kth_out_of_range(self):
        with pytest.raises(IndexError):
            self.bst.kth_smallest(100)


class TestBSTStructure:
    def test_height_empty(self):
        assert BST().height() == -1

    def test_height_balanced(self):
        bst = BST()
        for v in [4, 2, 6, 1, 3, 5, 7]: bst.insert(v)
        assert bst.height() == 2

    def test_size(self):
        bst = BST()
        for v in range(10): bst.insert(v)
        assert bst.size() == 10

    def test_level_order(self):
        bst = BST()
        for v in [4, 2, 6, 1, 3, 5, 7]: bst.insert(v)
        assert bst.level_order() == [[4], [2, 6], [1, 3, 5, 7]]
```

---

## `tests/test_heap.py`

```python
"""tests/test_heap.py"""

from __future__ import annotations
import pytest
from src.heap import MinHeap, MinHeapPQ


def _is_min_heap(data: list[int]) -> bool:
    n = len(data)
    for i in range(n):
        l, r = 2*i+1, 2*i+2
        if l < n and data[l] < data[i]:
            return False
        if r < n and data[r] < data[i]:
            return False
    return True


class TestMinHeap:
    def test_push_pop_order(self):
        h = MinHeap()
        for v in [5, 3, 8, 1, 4]:
            h.push(v)
        result = []
        while h:
            result.append(h.pop())
        assert result == sorted([5, 3, 8, 1, 4])

    def test_peek_does_not_remove(self):
        h = MinHeap()
        h.push(3)
        h.push(1)
        assert h.peek() == 1
        assert len(h) == 2

    def test_empty_pop_raises(self):
        with pytest.raises(IndexError):
            MinHeap().pop()

    def test_empty_peek_raises(self):
        with pytest.raises(IndexError):
            MinHeap().peek()

    def test_heapify(self):
        h = MinHeap.heapify([5, 3, 8, 1, 4])
        assert h.peek() == 1
        assert _is_min_heap(h._data)

    def test_heap_property_after_each_push(self):
        h = MinHeap()
        for v in [9, 2, 7, 4, 1, 6, 3]:
            h.push(v)
            assert _is_min_heap(h._data), f"Heap property violated after push({v})"

    def test_single_element(self):
        h = MinHeap()
        h.push(42)
        assert h.pop() == 42
        assert len(h) == 0


class TestMinHeapPQ:
    def test_priority_order(self):
        pq = MinHeapPQ()
        pq.push(10, "low")
        pq.push(1, "critical")
        pq.push(5, "medium")
        assert pq.pop() == (1, "critical")
        assert pq.pop() == (5, "medium")
        assert pq.pop() == (10, "low")

    def test_fifo_tiebreaking(self):
        pq = MinHeapPQ()
        pq.push(1, "first")
        pq.push(1, "second")
        pq.push(1, "third")
        assert pq.pop()[1] == "first"
        assert pq.pop()[1] == "second"
        assert pq.pop()[1] == "third"

    def test_mixed_priorities(self):
        pq = MinHeapPQ()
        items = [(3, "c"), (1, "a"), (2, "b"), (1, "d")]
        for pri, item in items:
            pq.push(pri, item)
        pri1, item1 = pq.pop()
        assert pri1 == 1
        assert item1 == "a"   # FIFO: "a" inserted before "d"

    def test_empty_pop_raises(self):
        with pytest.raises(IndexError):
            MinHeapPQ().pop()

    def test_peek(self):
        pq = MinHeapPQ()
        pq.push(5, "x")
        pq.push(2, "y")
        assert pq.peek() == (2, "y")
        assert len(pq) == 2  # peek didn't remove
```
