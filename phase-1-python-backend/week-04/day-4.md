# Phase 1 · Week 04 · Day 4 — Memory Management, References & Garbage Collection

> Objective: Understand Python's memory model, reference counting, and the cyclic GC so you can reason about object lifetimes and diagnose memory issues.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]]

## Concept

Python's memory model differs from languages like C or Java in a fundamental way: **every value is an object on the heap, and variables are name bindings (references) to those objects, not storage boxes**.

When you write:

```python
x = 42
y = x
```

You are not copying `42` into `y`. Both `x` and `y` are names that point to the same integer object `42` on the heap. Rebinding `y = 100` makes `y` point to a different object; `x` is unaffected.

This has direct implications for:
- Mutability: reassigning a name never changes the object it pointed to; mutating via a method (e.g., `list.append`) changes the object in-place and all references see the change.
- Object lifetime: objects live as long as at least one reference to them exists.
- Performance: passing a large list to a function is cheap (one pointer copy), but in-place mutation inside the function affects the caller's object.

## How it works internally

### Reference counting

CPython tracks, inside every object's header, an integer called `ob_refcnt`. Every time a new reference to an object is created (assignment, passing as argument, appending to a container), the count increments. Every time a reference is removed (name goes out of scope, `del` statement, container item removed), it decrements. When `ob_refcnt` reaches zero, the object is immediately deallocated and its memory returned to the allocator.

`sys.getrefcount(obj)` reports the current refcount. Note: calling `getrefcount` itself creates a temporary reference, so the returned value is always at least 1 higher than you might expect.

What increments refcount:
- Assignment: `a = obj`
- Function call argument: `f(obj)` — inside `f`, the parameter holds a reference
- Storing in a container: `lst.append(obj)`, `d["k"] = obj`
- Augmented assignment that binds: `a = b = obj`
- Closures capturing a variable from an outer scope

What decrements refcount:
- `del name`
- Name goes out of scope (function returns)
- Item removed from container: `lst.pop()`, `del d["k"]`
- Variable rebound: `a = something_else` — the old object loses the `a` reference

### Cyclic garbage collector

Reference counting alone cannot reclaim cycles:

```python
a = []
b = [a]
a.append(b)  # a -> b -> a: neither refcount reaches zero when we del a, b
```

CPython includes a supplemental cyclic GC (`gc` module) that periodically scans for groups of objects whose references form cycles but are unreachable from any root (no live name in any frame points to them). It uses a mark-and-sweep variant.

**Generations**: Objects are sorted into three generations (0, 1, 2). Newly allocated objects start in generation 0. If they survive a collection of gen-0, they are promoted to gen-1; surviving gen-1 promotes to gen-2. Gen-0 is collected most frequently (threshold: ~700 allocations), gen-2 least frequently. Long-lived objects accumulate in gen-2 and are rarely re-scanned, which is efficient.

`gc.collect(generation)` triggers a collection manually. `gc.get_count()` returns `(gen0, gen1, gen2)` allocation counts since last collection. `gc.get_threshold()` returns the thresholds.

### id() and identity vs equality

`id(obj)` returns a unique integer for the object's identity — in CPython, this is its memory address. Two objects are *identical* if `a is b` (same id). They are *equal* if `a == b` (same value, per `__eq__`).

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b   # True  — same value
a is b   # False — different objects
```

Use `is` only for singletons: `None`, `True`, `False`. Never use `is` to compare integers or strings in general code — interning makes it work sometimes but not reliably.

### Interning

To save memory and speed equality checks, CPython reuses certain objects:

- **Small integers** in the range `[-5, 256]` are pre-allocated at interpreter startup; `id(1) == id(1)` is always true.
- **String interning**: string literals that look like identifiers are often interned automatically. `sys.intern(s)` forces interning of string `s`, returning the canonical instance. Useful when you store millions of identical strings (e.g., column names).

### Memory views and the buffer protocol

The **buffer protocol** allows objects to expose their raw memory to other objects without copying. `memoryview(b)` creates a view over a bytes-like object (`bytes`, `bytearray`, `array.array`, `numpy` arrays). Slicing a `memoryview` returns another view — no copy. This is critical for high-throughput I/O and numerical code.

```python
data = bytearray(b"hello world")
view = memoryview(data)
chunk = view[0:5]   # no copy
chunk[0] = ord("H")  # mutates original
```

### tracemalloc

`tracemalloc` is a stdlib module that hooks into CPython's allocator to record the Python call stack at each allocation. It lets you take two snapshots and compare them to find allocation hot-spots.

### Weak references

`weakref.ref(obj)` creates a reference that does not increment `ob_refcnt`. When the object is otherwise unreachable, it is collected and the weakref becomes dead (calling it returns `None`). Use `weakref.WeakValueDictionary` for caches where you do not want the cache to prevent GC of values.

## Worked example

```python
"""
day4_memory.py
Demonstrates: refcount observation, cycle creation/detection, tracemalloc.
Python 3.12+
"""

import gc
import logging
import sys
import tracemalloc
import weakref
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Reference counting
# ---------------------------------------------------------------------------

def demo_refcount() -> None:
    log.info("=== Reference counting ===")

    obj: list[int] = [1, 2, 3]
    # getrefcount adds 1 for the argument; expect 2 here (obj + arg)
    log.info("After creation: refcount=%d", sys.getrefcount(obj))

    alias = obj
    log.info("After alias=obj: refcount=%d", sys.getrefcount(obj))

    container: list[Any] = [obj]
    log.info("After list append: refcount=%d", sys.getrefcount(obj))

    del container
    log.info("After del container: refcount=%d", sys.getrefcount(obj))

    del alias
    log.info("After del alias: refcount=%d", sys.getrefcount(obj))


# ---------------------------------------------------------------------------
# 2. Identity vs equality
# ---------------------------------------------------------------------------

def demo_identity() -> None:
    log.info("=== Identity vs equality ===")

    a: list[int] = [10, 20]
    b: list[int] = [10, 20]
    log.info("a == b: %s | a is b: %s | id(a)=%d id(b)=%d", a == b, a is b, id(a), id(b))

    # Small integer interning
    x: int = 100
    y: int = 100
    log.info("x=100 is y=100: %s (interned)", x is y)

    # Integers outside [-5, 256] are NOT guaranteed to be interned
    p: int = 1000
    q: int = 1000
    log.info("p=1000 is q=1000: %s (may vary)", p is q)

    # String interning
    s1 = sys.intern("shared_column_name")
    s2 = sys.intern("shared_column_name")
    log.info("s1 is s2 (interned): %s", s1 is s2)


# ---------------------------------------------------------------------------
# 3. Reference cycles and the gc
# ---------------------------------------------------------------------------

class Node:
    """A simple node that can form a cycle."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.child: "Node | None" = None

    def __del__(self) -> None:
        # __del__ makes objects uncollectable in Python < 3.4; fine in 3.12
        log.info("Node(%s) finalised", self.name)


def demo_gc_cycle() -> None:
    log.info("=== Cyclic GC ===")

    # Disable automatic GC to observe manually
    gc.disable()
    gc.collect()  # clear any existing garbage

    before_counts = gc.get_count()
    log.info("GC counts before cycle creation: %s", before_counts)

    # Create a cycle: parent -> child -> parent
    parent = Node("parent")
    child = Node("child")
    parent.child = child
    child.child = parent  # cycle

    # Drop our references — refcount does not reach 0 due to cycle
    parent_id = id(parent)
    child_id = id(child)
    del parent, child

    mid_counts = gc.get_count()
    log.info("GC counts after del (cycle alive): %s", mid_counts)
    log.info("Cycle objects still alive — calling gc.collect()...")

    collected = gc.collect()
    log.info("gc.collect() reclaimed %d objects", collected)

    gc.enable()


# ---------------------------------------------------------------------------
# 4. Weak references
# ---------------------------------------------------------------------------

def demo_weakref() -> None:
    log.info("=== Weak references ===")

    class Resource:
        def __init__(self, label: str) -> None:
            self.label = label

    obj = Resource("expensive")
    weak: weakref.ref[Resource] = weakref.ref(obj)

    log.info("Weak ref alive: %s | object: %s", weak() is not None, weak())
    del obj
    # obj is now collected; weakref returns None
    log.info("After del obj — weak ref alive: %s", weak() is not None)


# ---------------------------------------------------------------------------
# 5. tracemalloc snapshot comparison
# ---------------------------------------------------------------------------

def demo_tracemalloc() -> None:
    log.info("=== tracemalloc ===")

    tracemalloc.start()

    snapshot1 = tracemalloc.take_snapshot()

    # Allocate some objects
    big_list: list[bytes] = [bytes(1024) for _ in range(500)]

    snapshot2 = tracemalloc.take_snapshot()

    stats = snapshot2.compare_to(snapshot1, "lineno")
    log.info("Top 3 allocation increases:")
    for stat in stats[:3]:
        log.info("  %s", stat)

    del big_list
    tracemalloc.stop()


# ---------------------------------------------------------------------------
# 6. Memory views — zero-copy slice
# ---------------------------------------------------------------------------

def demo_memoryview() -> None:
    log.info("=== memoryview (buffer protocol) ===")

    data = bytearray(b"Hello, World!")
    view = memoryview(data)

    # Slice is a view, not a copy
    chunk: memoryview = view[7:12]
    log.info("chunk via memoryview: %s", bytes(chunk))

    # Mutate through the view
    chunk[0] = ord("w")
    log.info("data after mutation through view: %s", data.decode())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_refcount()
    demo_identity()
    demo_gc_cycle()
    demo_weakref()
    demo_tracemalloc()
    demo_memoryview()
```

Run with:
```bash
python day4_memory.py
```

Expected output includes refcount changes, log messages from `Node.__del__` after `gc.collect()`, and tracemalloc stats showing the large `bytes` allocation.

## Exercises

1. Write a function `count_refs(obj: object) -> int` that returns the "real" refcount (subtracting the extra reference created by `getrefcount` itself). Verify it with a simple integer and a list.

2. Create two dictionaries that reference each other (a cycle). Disable the GC, delete your local references, then confirm the objects are still alive by checking `gc.get_objects()`. Re-enable GC and collect; confirm they are gone.

3. Implement a simple `WeakCache` class backed by `weakref.WeakValueDictionary`. It should have `get(key)` and `put(key, value)` methods. Write a test that confirms a cached object is evicted once all strong references to it are dropped.

4. Use `tracemalloc` to compare memory usage before and after loading a 1 MB string repeated 10 000 times into a list. Then repeat the experiment with `sys.intern()` applied to each string. Report the difference in allocated bytes.

5. Write a `Node` class with `left` and `right` children. Build a small binary tree and verify with `sys.getrefcount` how the refcount of the root changes as you add/remove children.

6. Demonstrate with `id()` that Python reuses the memory address of a short-lived object for a newly created object. (Hint: create an object, delete it, create another of the same type, compare ids.)

(Solutions: ../solutions/week-04/day-4.md)

## Self-check

- [ ] I can explain why `y = x` does not copy the value of `x` in Python.
- [ ] I can predict by how much `sys.getrefcount` changes after `alias = obj` and after `del alias`.
- [ ] I can describe what happens to a cyclic object graph when both local names are deleted and GC is disabled vs enabled.
- [ ] I know when `is` is appropriate (singletons) and when to use `==`.
- [ ] I can use `tracemalloc.take_snapshot()` and `compare_to()` to find allocation hot-spots.
- [ ] I can create a `weakref.ref` and check whether the referent is still alive.
- [ ] I understand that `memoryview` slicing avoids copying bytes.

## Supplementary links

- https://docs.python.org/3/c-api/refcounting.html — CPython reference counting API
- https://docs.python.org/3/library/gc.html — gc module documentation
- https://docs.python.org/3/library/tracemalloc.html — tracemalloc module documentation
- https://docs.python.org/3/library/weakref.html — weakref module documentation
- https://docs.python.org/3/library/stdtypes.html#memoryview — memoryview documentation
- https://realpython.com/python-memory-management/ — accessible deep-dive on CPython memory
- https://pythonspeed.com/articles/python-memory-models/ — practical mental model for Python memory
