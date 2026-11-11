# Phase 2 · Week 02 · Day 1 — Singly and Doubly Linked Lists

> Objective: Implement insert, delete, and reverse on singly and doubly linked lists from scratch.  ·  Time: ~1 hr  ·  Prereqs: [[phase-2-cs-design/week-01/day-1]]

---

## Concept

A **linked list** is a sequence of nodes where each node stores a value and one or two pointers to adjacent nodes. Unlike arrays, nodes are not contiguous in memory — they are connected through references.

### Singly Linked List

Each node holds a value and a `next` pointer. Traversal is one-directional (head → tail).

```
[head] → [A|→] → [B|→] → [C|None]
```

### Doubly Linked List

Each node holds a value plus `prev` and `next` pointers. Traversal is bi-directional. Insertions and deletions at known nodes are O(1) with no prior traversal needed.

```
None ← [A|↔] ↔ [B|↔] ↔ [C|↔] → None
        head                 tail
```

### Complexity summary

| Operation | Singly | Doubly |
|---|---|---|
| Access by index | O(n) | O(n) |
| Insert at head | O(1) | O(1) |
| Insert at tail | O(n)* | O(1) with tail ptr |
| Delete at head | O(1) | O(1) |
| Delete at tail | O(n) | O(1) with tail ptr |
| Delete arbitrary node | O(n) search | O(1) if node given |
| Reverse | O(n) | O(n) |

*O(1) if you maintain a tail pointer.

---

## How it works internally

**Memory layout:** Each node is a Python object on the heap. The list is only the chain of references — GC frees a node when `next` (and `prev`) references drop to zero.

**Why not use Python lists?** Python's `list` is a dynamic array: O(1) amortized append, O(n) insert at arbitrary position due to shifting. Linked lists trade O(1) positional insert/delete for O(n) indexed access — the right tool when you frequently splice in the middle without caring about indices (e.g., LRU cache, undo stack).

**Sentinel/dummy nodes:** Many implementations add a dummy `head` and `tail` node that never hold real data. This eliminates edge-case branches for empty-list or single-node operations — every real node always has a `prev` and `next` (pointing to sentinels), so the same code runs for all cases.

**Reverse in-place (singly):** Walk the list, flip each `next` pointer to point backwards. Requires tracking `prev`, `current`, and `next` simultaneously because flipping `current.next` loses the forward reference.

```
Before:  None ← A → B → C → None
After:   None ← C → B → A → None
```

---

## Worked example

```python
"""
linked_lists.py — singly and doubly linked list implementations
Python 3.12+
"""
from __future__ import annotations

import logging
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singly Linked List
# ---------------------------------------------------------------------------

class SLLNode[T]:
    """Node for a singly linked list."""

    def __init__(self, value: T) -> None:
        self.value: T = value
        self.next: SLLNode[T] | None = None


class SinglyLinkedList[T]:
    """Singly linked list with O(1) head/tail insert and O(n) search."""

    def __init__(self) -> None:
        self._head: SLLNode[T] | None = None
        self._tail: SLLNode[T] | None = None
        self._size: int = 0

    # -- mutations ----------------------------------------------------------

    def prepend(self, value: T) -> None:
        """Insert at head. O(1)."""
        node = SLLNode(value)
        node.next = self._head
        self._head = node
        if self._tail is None:      # first node is also the tail
            self._tail = node
        self._size += 1

    def append(self, value: T) -> None:
        """Insert at tail. O(1) because we track tail."""
        node = SLLNode(value)
        if self._tail is None:
            self._head = self._tail = node
        else:
            self._tail.next = node
            self._tail = node
        self._size += 1

    def insert_after(self, target_value: T, new_value: T) -> bool:
        """Insert new_value after the first node with target_value. O(n)."""
        current = self._head
        while current is not None:
            if current.value == target_value:
                node = SLLNode(new_value)
                node.next = current.next
                current.next = node
                if current is self._tail:   # new node becomes tail
                    self._tail = node
                self._size += 1
                return True
            current = current.next
        return False  # target not found

    def delete(self, value: T) -> bool:
        """Remove first occurrence of value. O(n)."""
        if self._head is None:
            return False
        # Special case: deleting the head
        if self._head.value == value:
            self._head = self._head.next
            if self._head is None:          # list is now empty
                self._tail = None
            self._size -= 1
            return True
        # General case: find node before the target
        prev = self._head
        current = self._head.next
        while current is not None:
            if current.value == value:
                prev.next = current.next
                if current is self._tail:   # deleted the tail
                    self._tail = prev
                self._size -= 1
                return True
            prev = current
            current = current.next
        return False

    def reverse(self) -> None:
        """Reverse list in-place. O(n) time, O(1) space."""
        prev: SLLNode[T] | None = None
        current = self._head
        self._tail = self._head          # old head becomes new tail
        while current is not None:
            next_node = current.next     # save forward reference
            current.next = prev          # flip pointer
            prev = current
            current = next_node
        self._head = prev                # prev is now the new head

    # -- queries ------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        current = self._head
        while current is not None:
            yield current.value
            current = current.next

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return " -> ".join(str(v) for v in self) + " -> None"


# ---------------------------------------------------------------------------
# Doubly Linked List (with sentinel nodes)
# ---------------------------------------------------------------------------

class DLLNode[T]:
    """Node for a doubly linked list."""

    def __init__(self, value: T) -> None:
        self.value: T = value
        self.prev: DLLNode[T] | None = None
        self.next: DLLNode[T] | None = None


class DoublyLinkedList[T]:
    """
    Doubly linked list using sentinel head/tail nodes.
    All edge cases (empty list, single node) collapse to the same code path.
    """

    def __init__(self) -> None:
        # Sentinel nodes — never hold real data
        self._head: DLLNode[T] = DLLNode(None)  # type: ignore[arg-type]
        self._tail: DLLNode[T] = DLLNode(None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size: int = 0

    # -- low-level insert/remove --------------------------------------------

    def _insert_between(
        self,
        value: T,
        before: DLLNode[T],
        after: DLLNode[T],
    ) -> DLLNode[T]:
        """Core primitive: insert new node between two existing nodes. O(1)."""
        node = DLLNode(value)
        node.prev = before
        node.next = after
        before.next = node
        after.prev = node
        self._size += 1
        return node

    def _remove_node(self, node: DLLNode[T]) -> T:
        """Core primitive: unlink a node. O(1). Do not call on sentinels."""
        assert node.prev is not None and node.next is not None
        node.prev.next = node.next
        node.next.prev = node.prev
        self._size -= 1
        return node.value

    # -- public API ---------------------------------------------------------

    def prepend(self, value: T) -> DLLNode[T]:
        """Insert at logical head. O(1). Returns the new node."""
        return self._insert_between(value, self._head, self._head.next)  # type: ignore[arg-type]

    def append(self, value: T) -> DLLNode[T]:
        """Insert at logical tail. O(1). Returns the new node."""
        return self._insert_between(value, self._tail.prev, self._tail)  # type: ignore[arg-type]

    def delete_node(self, node: DLLNode[T]) -> T:
        """Delete a node in O(1) — no traversal needed."""
        return self._remove_node(node)

    def delete_first(self, value: T) -> bool:
        """Delete first occurrence by value. O(n) search."""
        current = self._head.next
        while current is not self._tail:
            if current.value == value:  # type: ignore[union-attr]
                self._remove_node(current)  # type: ignore[arg-type]
                return True
            current = current.next  # type: ignore[union-attr]
        return False

    def reverse(self) -> None:
        """Reverse in-place by swapping all prev/next pointers. O(n)."""
        current: DLLNode[T] | None = self._head
        while current is not None:
            # Swap prev and next for every node (including sentinels)
            current.prev, current.next = current.next, current.prev
            current = current.prev   # move forward (was next before swap)
        # Swap sentinels so head/tail roles are correct
        self._head, self._tail = self._tail, self._head

    # -- queries ------------------------------------------------------------

    def __iter__(self) -> Iterator[T]:
        current = self._head.next
        while current is not self._tail:
            yield current.value  # type: ignore[misc]
            current = current.next  # type: ignore[union-attr]

    def __reversed__(self) -> Iterator[T]:
        current = self._tail.prev
        while current is not self._head:
            yield current.value  # type: ignore[misc]
            current = current.prev  # type: ignore[union-attr]

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return "None <-> " + " <-> ".join(str(v) for v in self) + " <-> None"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=== Singly Linked List ===")
    sll: SinglyLinkedList[int] = SinglyLinkedList()
    for v in [1, 2, 3, 4]:
        sll.append(v)
    log.info("Initial:      %s", sll)
    sll.prepend(0)
    log.info("After prepend(0): %s", sll)
    sll.insert_after(2, 99)
    log.info("After insert_after(2, 99): %s", sll)
    sll.delete(99)
    log.info("After delete(99): %s", sll)
    sll.reverse()
    log.info("After reverse:    %s", sll)

    log.info("")
    log.info("=== Doubly Linked List ===")
    dll: DoublyLinkedList[str] = DoublyLinkedList()
    nodes = [dll.append(c) for c in ["a", "b", "c", "d"]]
    log.info("Initial:     %s", dll)
    dll.delete_node(nodes[1])   # delete 'b' in O(1)
    log.info("Deleted 'b': %s", dll)
    dll.reverse()
    log.info("Reversed:    %s", dll)
    log.info("Backwards:   %s", list(reversed(dll)))
```

**Key trace — `SinglyLinkedList.reverse()`:**

Starting state: `head → 1 → 2 → 3 → None`

| Iteration | prev | current | next_node | Action |
|---|---|---|---|---|
| 1 | None | 1 | 2 | `1.next = None` |
| 2 | 1 | 2 | 3 | `2.next = 1` |
| 3 | 2 | 3 | None | `3.next = 2` |
| end | — | None | — | `head = 3` |

Result: `head → 3 → 2 → 1 → None`

---

## Exercises

1. **Detect a cycle:** Implement `has_cycle(head: SLLNode | None) -> bool` using Floyd's tortoise-and-hare algorithm. Do not use extra memory (no set or dict).

2. **Find the kth node from the tail:** Implement `kth_from_tail(lst: SinglyLinkedList[int], k: int) -> int | None` in a single pass — traverse the list exactly once. What is the space complexity?

3. **Merge two sorted singly linked lists:** Given two `SinglyLinkedList[int]` objects whose values are in ascending order, return a new `SinglyLinkedList[int]` that merges them in sorted order. Achieve O(m + n) time.

4. **Implement an LRU cache** using only `DoublyLinkedList` and a plain `dict`. Support `get(key) -> int | None` and `put(key, value)` both in O(1). The capacity is fixed at construction. When capacity is exceeded, evict the least recently used entry.

5. **Palindrome check:** Implement `is_palindrome(lst: SinglyLinkedList[int]) -> bool` in O(n) time and O(1) extra space. Hint: reverse the second half in-place, compare, then restore.

*(Solutions: ../solutions/week-02/day-1.md)*

---

## Self-check

- [ ] I can explain why doubly linked lists allow O(1) deletion given a node reference, whereas singly linked lists cannot.
- [ ] I can implement `reverse` for a singly linked list without using extra memory, tracing the pointer states step by step.
- [ ] I can implement `append`, `prepend`, and `delete` for both list types, handling empty-list and single-node edge cases.
- [ ] I can describe the sentinel node pattern and explain what edge cases it eliminates.
- [ ] I can articulate when a linked list is preferable to a Python `list` and when it is not.

---

## Supplementary links

- [Visualgo — Linked List](https://visualgo.net/en/list)
- [MIT 6.006 Lecture 7 — Hashing](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/pages/lecture-videos/)
- [Python `collections.deque` source (CPython)](https://github.com/python/cpython/blob/main/Modules/_collectionsmodule.c)
