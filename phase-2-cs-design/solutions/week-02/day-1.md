# Solutions — Phase 2 · Week 02 · Day 1

> Singly and Doubly Linked Lists — insert, delete, reverse

---

## Exercise 1 — Detect a cycle (Floyd's algorithm)

**Key insight:** Use two pointers — `slow` advances one step, `fast` advances two. If there is a cycle they must eventually occupy the same node. If there is no cycle, `fast` reaches `None` first.

```python
"""
solution_day1_ex1.py — cycle detection with Floyd's tortoise-and-hare
Python 3.12+
"""
from __future__ import annotations

import logging
from linked_lists import SLLNode  # assumes worked-example file is importable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def has_cycle(head: SLLNode[int] | None) -> bool:
    """
    Detect cycle using two pointers. O(n) time, O(1) space.

    Proof of correctness:
    - If no cycle: fast reaches None, return False.
    - If cycle of length L: once both pointers enter the cycle,
      the gap between them shrinks by 1 each iteration → they meet in ≤ L steps.
    """
    slow = head
    fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next          # type: ignore[union-attr]
        fast = fast.next.next
        if slow is fast:
            return True
    return False


if __name__ == "__main__":
    # Build: 1 -> 2 -> 3 -> 4 -> (back to node 2)
    n1, n2, n3, n4 = SLLNode(1), SLLNode(2), SLLNode(3), SLLNode(4)
    n1.next, n2.next, n3.next, n4.next = n2, n3, n4, n2   # cycle at n2

    log.info("Cyclic list has_cycle: %s", has_cycle(n1))   # True

    # Linear list: 1 -> 2 -> 3 -> None
    a, b, c = SLLNode(1), SLLNode(2), SLLNode(3)
    a.next, b.next = b, c

    log.info("Linear list has_cycle: %s", has_cycle(a))    # False
```

---

## Exercise 2 — kth node from the tail (single pass)

**Key insight:** Use two pointers separated by k steps. Advance both together until `ahead` hits the tail — `behind` is then exactly k steps from the end.

```python
"""
solution_day1_ex2.py — kth from tail in one pass
Python 3.12+
"""
from __future__ import annotations

import logging
from linked_lists import SinglyLinkedList

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def kth_from_tail(lst: SinglyLinkedList[int], k: int) -> int | None:
    """
    Return the kth value from the tail (0-indexed: k=0 is the tail).
    Single pass. O(n) time, O(1) space.

    Strategy: advance 'ahead' k steps first, then move both pointers
    together. When 'ahead' reaches None, 'behind' is k steps from end.
    """
    if k < 0:
        return None

    # Access internal head for raw-node traversal
    ahead = lst._head  # noqa: SLF001
    behind = lst._head

    # Advance 'ahead' by k+1 steps
    for _ in range(k + 1):
        if ahead is None:
            return None   # k is out of range
        ahead = ahead.next

    # Move together until 'ahead' falls off the end
    while ahead is not None:
        ahead = ahead.next
        behind = behind.next  # type: ignore[union-attr]

    return behind.value if behind is not None else None  # type: ignore[return-value]


if __name__ == "__main__":
    sll: SinglyLinkedList[int] = SinglyLinkedList()
    for v in [10, 20, 30, 40, 50]:
        sll.append(v)
    # 0-indexed from tail: k=0→50, k=1→40, k=4→10
    for k in range(6):
        log.info("kth_from_tail(k=%d) = %s", k, kth_from_tail(sll, k))
```

**Space complexity:** O(1) — only two extra pointer variables regardless of list length.

---

## Exercise 3 — Merge two sorted singly linked lists

**Key insight:** Compare heads, attach the smaller one to the result, advance that list's pointer. Similar to the merge step in merge sort.

```python
"""
solution_day1_ex3.py — merge two sorted singly linked lists
Python 3.12+
"""
from __future__ import annotations

import logging
from linked_lists import SinglyLinkedList

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def merge_sorted(
    a: SinglyLinkedList[int],
    b: SinglyLinkedList[int],
) -> SinglyLinkedList[int]:
    """
    Merge two ascending-sorted lists into a new ascending-sorted list.
    O(m + n) time, O(m + n) space (new nodes).

    This implementation creates new nodes to avoid mutating the originals.
    """
    result: SinglyLinkedList[int] = SinglyLinkedList()
    iter_a = iter(a)
    iter_b = iter(b)

    val_a: int | None = next(iter_a, None)
    val_b: int | None = next(iter_b, None)

    while val_a is not None and val_b is not None:
        if val_a <= val_b:
            result.append(val_a)
            val_a = next(iter_a, None)
        else:
            result.append(val_b)
            val_b = next(iter_b, None)

    # Drain the non-empty iterator
    while val_a is not None:
        result.append(val_a)
        val_a = next(iter_a, None)
    while val_b is not None:
        result.append(val_b)
        val_b = next(iter_b, None)

    return result


if __name__ == "__main__":
    a: SinglyLinkedList[int] = SinglyLinkedList()
    b: SinglyLinkedList[int] = SinglyLinkedList()
    for v in [1, 3, 5, 7]:
        a.append(v)
    for v in [2, 4, 6, 8, 9]:
        b.append(v)
    merged = merge_sorted(a, b)
    log.info("Merged: %s", merged)  # 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> None
```

---

## Exercise 4 — LRU Cache using DoublyLinkedList + dict

**Key insight:**
- The doubly linked list represents access order (MRU at head, LRU at tail).
- The dict maps keys to DLL nodes for O(1) lookup.
- On `get` or `put`: move the accessed node to the front in O(1).
- On eviction: remove the tail node (LRU) in O(1).

```python
"""
solution_day1_ex4.py — O(1) LRU cache
Python 3.12+
"""
from __future__ import annotations

import logging
from linked_lists import DoublyLinkedList, DLLNode

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class LRUCache:
    """
    Least-Recently-Used cache with O(1) get and put.

    Internally: a DoublyLinkedList of (key, value) pairs + a dict[key → node].
    Most-recently-used end = list head.  Least-recently-used end = list tail.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._dll: DoublyLinkedList[tuple[int, int]] = DoublyLinkedList()
        self._map: dict[int, DLLNode[tuple[int, int]]] = {}

    def get(self, key: int) -> int | None:
        """Return value for key, or None if not present. O(1)."""
        node = self._map.get(key)
        if node is None:
            return None
        self._move_to_front(node)
        return node.value[1]

    def put(self, key: int, value: int) -> None:
        """Insert or update key. Evict LRU if over capacity. O(1)."""
        if key in self._map:
            node = self._map[key]
            self._dll.delete_node(node)  # remove old position
        elif len(self._dll) == self._capacity:
            # Evict LRU: it is the last real node (just before tail sentinel)
            lru_node = self._dll._tail.prev   # noqa: SLF001
            if lru_node is not self._dll._head:  # noqa: SLF001
                evicted_key = lru_node.value[0]  # type: ignore[union-attr]
                self._dll.delete_node(lru_node)  # type: ignore[arg-type]
                del self._map[evicted_key]

        new_node = self._dll.prepend((key, value))
        self._map[key] = new_node

    def _move_to_front(self, node: DLLNode[tuple[int, int]]) -> None:
        """Unlink node and re-insert at list head. O(1)."""
        val = self._dll.delete_node(node)
        new_node = self._dll.prepend(val)
        self._map[val[0]] = new_node


if __name__ == "__main__":
    cache = LRUCache(capacity=3)
    cache.put(1, 10)
    cache.put(2, 20)
    cache.put(3, 30)
    log.info("get(1)=%s", cache.get(1))   # 10, moves 1 to MRU
    cache.put(4, 40)                        # evicts key 2 (LRU)
    log.info("get(2)=%s", cache.get(2))   # None — evicted
    log.info("get(3)=%s", cache.get(3))   # 30
    log.info("get(4)=%s", cache.get(4))   # 40
```

---

## Exercise 5 — Palindrome check in O(n) time, O(1) space

**Key insight:** Find the midpoint (slow/fast pointers), reverse the second half in-place, compare front and reversed back half, then restore.

```python
"""
solution_day1_ex5.py — palindrome check O(n) time, O(1) space
Python 3.12+
"""
from __future__ import annotations

import logging
from linked_lists import SinglyLinkedList, SLLNode

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _reverse_from(node: SLLNode[int] | None) -> SLLNode[int] | None:
    """Reverse a singly linked sub-list starting at node. Return new head."""
    prev: SLLNode[int] | None = None
    current = node
    while current is not None:
        nxt = current.next
        current.next = prev
        prev = current
        current = nxt
    return prev


def is_palindrome(lst: SinglyLinkedList[int]) -> bool:
    """
    Check if the list reads the same forwards and backwards.
    O(n) time, O(1) extra space.

    Steps:
    1. Find the midpoint with slow/fast pointers.
    2. Reverse the second half in-place.
    3. Compare the two halves.
    4. Restore the second half (good practice).
    """
    if len(lst) <= 1:
        return True

    # Step 1: find midpoint
    slow: SLLNode[int] | None = lst._head   # noqa: SLF001
    fast: SLLNode[int] | None = lst._head   # noqa: SLF001
    while fast is not None and fast.next is not None:
        slow = slow.next          # type: ignore[union-attr]
        fast = fast.next.next

    # slow is now at the start of the second half
    second_head = _reverse_from(slow)

    # Step 3: compare
    first: SLLNode[int] | None = lst._head  # noqa: SLF001
    second: SLLNode[int] | None = second_head
    result = True
    while second is not None:
        if first is None or first.value != second.value:  # type: ignore[union-attr]
            result = False
            break
        first = first.next  # type: ignore[union-attr]
        second = second.next

    # Step 4: restore
    _reverse_from(second_head)

    return result


if __name__ == "__main__":
    def make_list(values: list[int]) -> SinglyLinkedList[int]:
        sll: SinglyLinkedList[int] = SinglyLinkedList()
        for v in values:
            sll.append(v)
        return sll

    cases = [
        ([1, 2, 3, 2, 1], True),
        ([1, 2, 2, 1], True),
        ([1], True),
        ([], True),
        ([1, 2, 3], False),
        ([1, 2, 3, 4, 2, 1], False),
    ]
    for values, expected in cases:
        lst = make_list(values)
        result = is_palindrome(lst)
        status = "PASS" if result == expected else "FAIL"
        log.info("[%s] %s → is_palindrome=%s (expected=%s)", status, values, result, expected)
```
