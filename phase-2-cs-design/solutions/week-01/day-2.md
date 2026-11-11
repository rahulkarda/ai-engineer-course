# Solutions — Day 2: Python List Internals & Amortized Analysis

## Exercise 1 — Growth pattern observation

Running `show_growth()` produces reallocation at approximately: 1, 5, 9, 17, 25, 33 elements.

CPython formula: `new = old + (old >> 3) + (3 if old < 9 else 6)`

Verification for old=4: 4 + 0 + 3 = 7. Actual capacity jumps differ slightly due to initial capacities. The formula matches `listobject.c` — run and observe your Python version's exact thresholds.

---

## Exercise 2 — insert(0) loop is O(n²)

`lst.insert(0, x)` shifts all existing elements right by one — O(len(lst)) per call.  
Over n calls: 0 + 1 + 2 + … + (n-1) = n(n-1)/2 = **O(n²) total**.

`lst.append(x)` adds at end — O(1) amortized per call.  
Over n calls: at most 2n copies total (geometric series argument) = **O(n) total**.

---

## Exercise 3 — `build_prefix` complexity

```python
def build_prefix(lst: list[int]) -> list[int]:
    result: list[int] = []
    for x in lst:
        result.insert(0, sum(result) + x)   # insert at front
    return result
```

- `sum(result)` is O(k) where k is current length → O(1)+O(2)+…+O(n) = O(n²) for the sums.
- `result.insert(0, ...)` is O(k) per call → O(1)+O(2)+…+O(n) = O(n²) for the inserts.
- **Total: O(n²)** — two separate O(n²) contributions; both are dominant.

---

## Exercise 4 — `sys.getsizeof` discrepancy

`sys.getsizeof([1, 2, 3])` returns the size of the **list object struct + the pointer array** (3 pointers × 8 bytes = 24 bytes on 64-bit). It does NOT count the objects pointed to.

`sys.getsizeof(1)` returns the size of the int object itself (28 bytes on CPython).

The list stores *references* (pointers), not objects. Adding the object sizes would double-count if the same object is referenced from multiple lists.

---

## Exercise 5 — Tuples for read-only data

O notation claim: **correct** — tuple index access `t[i]` is O(1), same as list.  
Implementation: both `list` and `tuple` are backed by a contiguous pointer array.

Non-asymptotic reasons to prefer tuple:
- Tuples are **immutable** — signals intent to reader; prevents accidental mutation.
- Tuples are **hashable** (if elements are) — can be used as dict keys or set members.
- Tuples have **slightly smaller memory footprint** (no `allocated` field or over-allocation).
- `sys.getsizeof((1,2,3))` < `sys.getsizeof([1,2,3])` typically.

---

## Exercise 6 — Real reallocation detection

```python
import sys

def count_real_reallocations(n: int) -> dict[str, int]:
    """Detect actual CPython list reallocations via sizeof change."""
    lst: list[int] = []
    prev_sizeof = sys.getsizeof(lst)
    reallocations = 0
    total_copies_approx = 0     # we estimate: copies = old_len at each realloc

    for i in range(n):
        lst.append(i)
        new_sizeof = sys.getsizeof(lst)
        if new_sizeof != prev_sizeof:
            reallocations += 1
            # capacity before realloc was len-1 (we just appended)
            total_copies_approx += len(lst) - 1
            prev_sizeof = new_sizeof

    return {
        "n": n,
        "reallocations": reallocations,
        "estimated_copies": total_copies_approx,
        "amortized_per_append": total_copies_approx / n if n else 0,
    }
```

CPython uses a ~1.125× growth factor (not 2×), so the amortized constant is lower than the 2× simulation. For n=10000 you should see roughly 50-70 reallocations vs ~14 for 2×.
