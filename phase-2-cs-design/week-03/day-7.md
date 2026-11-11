# Phase 2 · Week 03 · Day 7 — Week Project: BST + Min-Heap Priority Queue

> Objective: Build and test a production-quality BST class and a MinHeap priority queue.  ·  Time: ~3 hr (project) + 1 hr (see assessment.md)  ·  Prereqs: [[day-1..day-6]]

This is the project day. See `project.md` for the full specification and deliverables.

## Warm-up (30 min)

Before starting the project, complete these three exercises to cement the week's concepts:

### Warm-up 1 — Inorder iterator

Implement a `BSTIterator` class that behaves like Python's iterator protocol over a BST in sorted (inorder) order.

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]) -> None: ...
    def __iter__(self) -> "BSTIterator": ...
    def __next__(self) -> int: ...
```

Use an iterative inorder approach internally (explicit stack). `__next__` must be O(h) amortised.

### Warm-up 2 — Heap merge

Given two `MinHeap` instances (from day 5), implement `merge(h1: MinHeap, h2: MinHeap) -> MinHeap` that returns a new heap containing all elements. Complexity?

### Warm-up 3 — BST range sum

Implement `range_sum(root, lo, hi) -> int` that returns the sum of all BST values in `[lo, hi]` inclusive. Prune branches that cannot contribute — avoid visiting nodes outside the range.

---

## Project

See `project.md` for the full spec. Summary:

- Implement `BST` class with `insert`, `search`, `delete`, `validate`, `height`, and `inorder`
- Implement `MinHeapPQ` priority queue class wrapping `MinHeap`
- Full `pytest` test suite (`tests/test_bst.py`, `tests/test_heap.py`)
- All code: Python 3.12+, type hints, `logging`

---

## Assessment

Complete `assessment.md` after finishing the project.

---

## Reflection checklist

Before moving to week 4, confirm:

- [ ] BST insert/search/delete work correctly for all three deletion cases
- [ ] `is_valid_bst` correctly catches cross-subtree violations
- [ ] MinHeap correctly satisfies the heap property after every push/pop
- [ ] Test suite passes with no warnings
- [ ] I can explain time complexity of every implemented operation

## Supplementary links

- [Python `pytest` docs](https://docs.pytest.org/en/stable/)
- [Real Python — pytest tutorial](https://realpython.com/pytest-python-testing/)
- [Python `dataclasses` docs](https://docs.python.org/3.12/library/dataclasses.html)
