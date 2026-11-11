# Phase 2 · Week 01 · Day 2 — Python List Internals & Amortized Analysis

> Objective: Understand CPython's list implementation and explain why append is O(1) amortized.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

### Python list is a dynamic array

A Python `list` is not a linked list — it is a **dynamic array** (resizable contiguous block of pointers). Each slot holds a reference (pointer) to a Python object.

Key invariants:
- Elements are stored in contiguous memory → O(1) random access by index.
- When the array runs out of capacity, it is **reallocated** to a larger block and all pointers are copied.
- Python over-allocates by roughly 12.5% beyond the needed capacity, so reallocation is rare.

### The growth factor

CPython (as of 3.11+) uses this growth formula (from `listobject.c`):

```
new_capacity = old_capacity + (old_capacity >> 3) + (3 if old_capacity < 9 else 6)
```

For large lists this approximates a ~1.125× factor. The classic textbook uses 2×; the amortized argument works for any factor > 1.

---

## How it works internally

### Memory layout

```
list object
┌─────────────┐
│ ob_refcnt   │  reference count
│ ob_type     │  pointer to list type
│ ob_size     │  current length (len)
│ allocated   │  current capacity
│ ob_item ────┼──► [ *obj0 | *obj1 | *obj2 | ... | (empty slots) ]
└─────────────┘    ↑ contiguous block of PyObject* pointers
```

`sys.getsizeof(lst)` returns the size of the list struct + the pointer array, NOT the size of the objects.

### Why `append` is O(1) amortized

**Amortized analysis** — "accounting" argument:

Assign each element a "credit" of `k` operations at insertion time.  
When a reallocation occurs copying `m` elements, the accumulated credits of those `m` elements pay for the copy.

With a growth factor of 2× starting from capacity 1:
- Capacities hit: 1, 2, 4, 8, 16, …
- Elements copied at each resize: 1, 2, 4, 8, … — a geometric series.
- Total copies after `n` appends ≤ 1 + 2 + 4 + … + n = 2n.
- Each of the `n` appends does at most 2n/n = 2 copies on average → **O(1) amortized**.

**Contrast with worst-case O(n):** a single `append` that triggers reallocation costs O(n) for that one call — the amortized claim only holds across a sequence of operations.

### Operation complexities for `list`

| Operation | Time | Notes |
|---|---|---|
| `lst[i]` | O(1) | pointer arithmetic |
| `lst.append(x)` | O(1) amortized | occasional O(n) resize |
| `lst.pop()` | O(1) amortized | may shrink |
| `lst.insert(i, x)` | O(n) | shift all elements right of i |
| `lst.pop(i)` / `del lst[i]` | O(n) | shift left |
| `x in lst` | O(n) | linear scan |
| `lst.sort()` | O(n log n) | Timsort |
| `len(lst)` | O(1) | stored in header |
| `lst + lst2` | O(n + m) | new allocation |
| slice `lst[a:b]` | O(b−a) | copies pointers |

---

## Worked example

```python
"""
list_internals.py — measuring Python list growth and amortized cost
Python 3.12+
"""
import logging
import sys
import timeit
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def show_growth() -> None:
    """Print list capacity at each append — observe over-allocation."""
    lst: list[Any] = []
    prev_size = sys.getsizeof(lst)
    log.info("len=%-4d  sizeof=%d  (baseline)", len(lst), prev_size)
    for i in range(33):
        lst.append(i)
        size = sys.getsizeof(lst)
        if size != prev_size:
            log.info("len=%-4d  sizeof=%d  ← reallocation", len(lst), size)
            prev_size = size


def count_copies_amortized(n: int) -> dict[str, float]:
    """
    Simulate append-with-copy using a fixed 2× growth factor.
    Return total copies and amortized copies per append.
    """
    capacity = 1
    length = 0
    total_copies = 0
    for _ in range(n):
        if length == capacity:
            total_copies += length        # copy all existing elements
            capacity *= 2                 # double capacity
        length += 1
    return {
        "n": n,
        "total_copies": total_copies,
        "amortized_per_append": total_copies / n,
    }


def benchmark_insert_vs_append(n: int = 10_000) -> None:
    """Compare O(n) insert(0, x) vs O(1) amortized append."""
    t_append = timeit.timeit(
        stmt="lst.append(0)",
        setup="lst = []",
        number=n,
    )
    t_insert = timeit.timeit(
        stmt="lst.insert(0, 0)",
        setup="lst = []",
        number=n,
    )
    log.info("append ×%d: %.4fs", n, t_append)
    log.info("insert(0) ×%d: %.4fs  (O(n) per call)", n, t_insert)
    log.info("ratio: %.1f×", t_insert / t_append)


if __name__ == "__main__":
    log.info("=== list growth pattern ===")
    show_growth()

    log.info("\n=== amortized copy analysis (2× growth) ===")
    for n in [10, 100, 1_000, 10_000]:
        result = count_copies_amortized(n)
        log.info("n=%6d  copies=%6d  amortized=%.2f per append",
                 result["n"], result["total_copies"], result["amortized_per_append"])

    log.info("\n=== append vs insert benchmark ===")
    benchmark_insert_vs_append()
```

**Expected output excerpt:**
```
=== list growth pattern ===
len=0     sizeof=56  (baseline)
len=1     sizeof=88  ← reallocation
len=5     sizeof=120 ← reallocation
len=9     sizeof=184 ← reallocation
...

=== amortized copy analysis (2× growth) ===
n=    10  copies=    9  amortized=0.90 per append
n=   100  copies=   99  amortized=0.99 per append
n=  1000  copies=  999  amortized=1.00 per append
```

The amortized copies per append converges to a constant (≤ 2) — confirming O(1) amortized.

---

## Exercises

1. Run `show_growth()` yourself. After how many elements does the list reallocate? Does it match `old + (old >> 3) + 6`?

2. Explain why `lst.insert(0, x)` in a loop is O(n²) total while `lst.append(x)` in a loop is O(n) total amortized.

3. Given the following code, classify the overall time complexity and explain the dominant term:
   ```python
   def build_prefix(lst: list[int]) -> list[int]:
       result: list[int] = []
       for x in lst:
           result.insert(0, sum(result) + x)   # insert at front
       return result
   ```

4. Why does `sys.getsizeof([1, 2, 3])` not equal `sys.getsizeof(1) * 3 + sys.getsizeof([])`?

5. A colleague claims "Python tuples are always better than lists for read-only data because tuples are O(1) access." Evaluate: is the O notation correct? What are the non-asymptotic reasons to prefer a tuple?

6. Implement `count_copies_amortized` using Python's actual `list.append` and `sys.getsizeof` to detect real reallocations (not a 2× simulation). Compare your results to the simulation.

*(Solutions: ../solutions/week-01/day-2.md)*

---

## Self-check

- [ ] I can draw the memory layout of a CPython list from memory.
- [ ] I can explain the amortized O(1) append argument in under 60 seconds.
- [ ] I know which list operations are O(1), O(n), and O(n log n).
- [ ] I can predict the sizeof growth sequence for small lists.

---

## Supplementary links

- [CPython listobject.c source](https://github.com/python/cpython/blob/main/Objects/listobject.c)
- [Python docs — list complexity](https://wiki.python.org/moin/TimeComplexity)
- [Amortized analysis — CLRS § 17](https://en.wikipedia.org/wiki/Amortized_analysis)
