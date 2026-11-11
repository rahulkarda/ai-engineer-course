# Week 02 Assessment — Linked Lists, Hash Tables & Caching

> Time: ~90 min  ·  Open notes  ·  No internet  ·  Pass gate: ≥ 75%

---

## Conceptual (interview-style)

Answer each question as you would in a live technical interview — concise but precise.  
Aim for 3–5 sentences per question.

**1.** A singly linked list supports O(n) deletion when given only the key to delete. A doubly linked list supports O(1) deletion when given the *node* itself. Explain the asymmetry. Under what circumstances does the O(1) guarantee break down even with a doubly linked list?

**2.** Python's built-in `dict` offers O(1) average-case lookup but O(n) worst-case. What causes the worst case, and why is it rarely hit in practice? How does Python's dict implementation defend against deliberate worst-case attacks?

**3.** You are designing a cache for a web API that serves product detail pages. The product catalog has 2 million items; your cache can hold 50,000. Compare LRU, LFU, and a simple FIFO cache for this workload. Which policy would you choose, and why? What additional information about the access pattern would change your answer?

**4.** A classmate proposes implementing an LRU cache using only a Python `list` (no dict, no linked list). For `get(key)`, they scan the list linearly; for `put(key, value)`, they append and truncate. What is the Big-O complexity of each operation? At what cache size does this become impractical compared to the dict + DLL approach, and why?

**5.** Explain the role of sentinel (dummy) nodes in the doubly linked list used for LRU. What edge cases do they eliminate? Could you implement the same LRU cache without sentinels? What would change?

---

## Coding challenge

### Task: Implement `LRUCache`

Implement a class `LRUCache` that satisfies the interface below using a **dict and a doubly linked list** (not `collections.OrderedDict` or any other standard library container that hides the linked-list mechanics).

```python
class LRUCache:
    def __init__(self, capacity: int) -> None: ...
    def get(self, key: int) -> int: ...        # return -1 if not present
    def put(self, key: int, value: int) -> None: ...
```

**Constraints:**
- `capacity >= 1`
- Keys and values are integers.
- `get` returns `-1` (not `None`) for a missing key — this matches LeetCode 146 convention.
- Both `get` and `put` must run in **O(1)** average time.
- You must implement the doubly linked list node and pointer operations yourself. Internal use of `list`, `deque`, or `OrderedDict` is not permitted.
- Type hints on all functions and class attributes.
- Use `logging` not `print`.

**Acceptance criteria — all tests below must pass:**

```python
import pytest
from solution import LRUCache   # your file


def test_basic_get_put() -> None:
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1


def test_eviction_evicts_lru() -> None:
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    c.get(1)        # key 1 becomes most recent
    c.put(3, 3)     # evicts key 2
    assert c.get(2) == -1
    assert c.get(1) == 1
    assert c.get(3) == 3


def test_update_existing_key() -> None:
    c = LRUCache(2)
    c.put(2, 1)
    c.put(2, 2)     # update; size stays 1
    assert c.get(2) == 2


def test_capacity_one() -> None:
    c = LRUCache(1)
    c.put(1, 1)
    c.put(2, 2)     # evicts key 1
    assert c.get(1) == -1
    assert c.get(2) == 2


def test_get_moves_to_front() -> None:
    c = LRUCache(3)
    c.put(1, 10)
    c.put(2, 20)
    c.put(3, 30)
    c.get(1)        # 1 is now MRU; LRU is 2
    c.put(4, 40)    # evicts key 2
    assert c.get(2) == -1
    assert c.get(1) == 10
    assert c.get(3) == 30
    assert c.get(4) == 40


def test_put_then_get_sequential() -> None:
    c = LRUCache(3)
    for i in range(1, 4):
        c.put(i, i * 100)
    for i in range(1, 4):
        assert c.get(i) == i * 100
```

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Full marks criteria |
|---|---|---|
| Independent implementation | 40% | Correct dict + DLL; no banned containers; all acceptance tests pass |
| Conceptual understanding | 25% | All 5 conceptual questions answered accurately and concisely |
| Debugging & problem-solving | 20% | Edge cases handled (capacity 1, duplicate keys, get on empty cache) |
| Code quality & testing | 15% | Type hints present; logging used; code readable; no magic numbers |

**Scoring:**

| Score range | Result |
|---|---|
| 90–100% | Distinction |
| 75–89% | Pass |
| < 75% | Re-attempt |

Answer key: ../solutions/week-02/assessment.md
