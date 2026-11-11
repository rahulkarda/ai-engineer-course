# Week 03 Project — BST + MinHeap Priority Queue

> Time: ~3 hr  ·  Language: Python 3.12+

## Overview

Build two well-tested data structures from scratch:

1. **`BST`** — a Binary Search Tree class with full CRUD operations, validation, and traversal.
2. **`MinHeapPQ`** — a min-heap-backed priority queue.

Together these form the foundation for tree-based indexing and task scheduling — used in databases, OS schedulers, and event-driven systems.

---

## Repository layout

```
week03_project/
├── src/
│   ├── bst.py          # BST implementation
│   └── heap.py         # MinHeap + MinHeapPQ
├── tests/
│   ├── test_bst.py
│   └── test_heap.py
└── pyproject.toml      # (optional) or just run pytest from root
```

---

## Part 1 — `BST` class (`src/bst.py`)

### Specification

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import math

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
        """Insert val. Duplicates are silently ignored."""
        ...

    def delete(self, val: int) -> None:
        """Delete val. No-op if val not present."""
        ...

    # ── Queries ───────────────────────────────────────────────────────────

    def search(self, val: int) -> bool:
        """Return True iff val exists in the BST."""
        ...

    def minimum(self) -> Optional[int]:
        """Return the minimum value, or None if empty."""
        ...

    def maximum(self) -> Optional[int]:
        """Return the maximum value, or None if empty."""
        ...

    def floor(self, val: int) -> Optional[int]:
        """Largest value ≤ val, or None."""
        ...

    def ceil(self, val: int) -> Optional[int]:
        """Smallest value ≥ val, or None."""
        ...

    def kth_smallest(self, k: int) -> int:
        """Return the k-th smallest value (1-indexed). Raise IndexError if k out of range."""
        ...

    # ── Structural ────────────────────────────────────────────────────────

    def height(self) -> int:
        """Height of the tree. Empty tree = -1."""
        ...

    def size(self) -> int:
        """Total number of nodes."""
        ...

    def is_valid(self) -> bool:
        """True iff BST invariant holds throughout the tree."""
        ...

    # ── Traversals ────────────────────────────────────────────────────────

    def inorder(self) -> list[int]:
        """Inorder traversal → sorted list."""
        ...

    def level_order(self) -> list[list[int]]:
        """Level-order traversal → list of levels."""
        ...

    # ── Display ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"BST(size={self.size()}, height={self.height()})"
```

### Requirements

- `delete` must handle all three cases correctly.
- `is_valid` must propagate bounds (not just local checks).
- `kth_smallest` must stop early — do not collect all inorder values first.
- All methods: type hints, log debug messages for insert/delete.
- No global mutable state; BST state is encapsulated in `self._root`.

---

## Part 2 — `MinHeap` & `MinHeapPQ` (`src/heap.py`)

### Specification

```python
from __future__ import annotations
import logging
from typing import Any, Generic, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class MinHeap:
    """Integer min-heap. DO NOT use heapq internally."""

    def __init__(self) -> None:
        self._data: list[int] = []

    def push(self, val: int) -> None: ...
    def pop(self) -> int: ...
    def peek(self) -> int: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...

    @classmethod
    def heapify(cls, values: list[int]) -> "MinHeap": ...

    def _sift_up(self, i: int) -> None: ...
    def _sift_down(self, i: int) -> None: ...


class MinHeapPQ:
    """
    Priority queue backed by MinHeap.
    Lower priority number = higher urgency (min-priority semantics).
    Tie-breaking: FIFO (earlier insertions win).
    """

    def push(self, priority: int, item: Any) -> None: ...
    def pop(self) -> tuple[int, Any]: ...
    def peek(self) -> tuple[int, Any]: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...
```

### Requirements

- `MinHeap` must NOT use `heapq` internally — implement sift-up and sift-down yourself.
- `MinHeapPQ` tie-breaking must be FIFO (use an insertion counter).
- Raise `IndexError` on `pop`/`peek` of empty structures.
- Log push/pop at DEBUG level.

---

## Part 3 — Test suite

### `tests/test_bst.py` — required test cases

```python
import pytest
from src.bst import BST


class TestBSTInsertSearch:
    def test_empty_search(self): ...
    def test_insert_single(self): ...
    def test_inorder_sorted(self): ...
    def test_duplicate_ignored(self): ...


class TestBSTDelete:
    def test_delete_leaf(self): ...
    def test_delete_one_child(self): ...
    def test_delete_two_children(self): ...
    def test_delete_root(self): ...
    def test_delete_nonexistent(self): ...


class TestBSTValidation:
    def test_valid_bst(self): ...
    def test_invalid_cross_subtree(self): ...


class TestBSTQueries:
    def test_min_max(self): ...
    def test_floor_ceil(self): ...
    def test_kth_smallest(self): ...
    def test_kth_out_of_range(self): ...


class TestBSTStructure:
    def test_height_empty(self): ...
    def test_height_balanced(self): ...
    def test_size(self): ...
    def test_level_order(self): ...
```

### `tests/test_heap.py` — required test cases

```python
import pytest
from src.heap import MinHeap, MinHeapPQ


class TestMinHeap:
    def test_push_pop_order(self): ...
    def test_peek_does_not_remove(self): ...
    def test_empty_pop_raises(self): ...
    def test_heapify(self): ...
    def test_heap_property_after_each_push(self): ...


class TestMinHeapPQ:
    def test_priority_order(self): ...
    def test_fifo_tiebreaking(self): ...
    def test_mixed_priorities(self): ...
    def test_empty_pop_raises(self): ...
```

All test classes must have fully implemented test bodies (no `pass`).

---

## Stretch goals (optional)

1. Add a `BSTIterator` class with `__iter__` / `__next__` (iterative inorder, O(h) amortised per `next`).
2. Add `rank(val) -> int` to `BST`: number of values strictly less than `val`.
3. Add `merge(other: MinHeap) -> MinHeap` that combines two heaps in O(n).
4. Implement a `MaxHeap` by subclassing or composing `MinHeap` (negation strategy).

---

## Submission checklist

- [ ] `src/bst.py` — all methods implemented, type-hinted, logging
- [ ] `src/heap.py` — `MinHeap` (no `heapq`), `MinHeapPQ` with FIFO tie-breaking
- [ ] `tests/test_bst.py` — all required test classes with passing tests
- [ ] `tests/test_heap.py` — all required test classes with passing tests
- [ ] `pytest` runs with zero failures and zero warnings
- [ ] No hardcoded credentials or `print()` statements
- [ ] Each test is independent (no shared mutable state between tests)
