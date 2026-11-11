# Week 4 Day 4 — Memory Management, References & GC: Solutions

## Exercise 1: sys.getrefcount() at Different Points

```python
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_refcount() -> None:
    """Show how reference count changes as references are created/dropped."""
    x: list[int] = [1, 2, 3]

    # sys.getrefcount adds 1 because the argument itself is a reference
    count_initial = sys.getrefcount(x)
    logger.info("After assignment to x: refcount = %d (includes getrefcount arg)", count_initial)
    # Expected: 2 (x + getrefcount temporary)

    y = x
    count_two_refs = sys.getrefcount(x)
    logger.info("After y = x: refcount = %d", count_two_refs)
    # Expected: 3

    ref_list: list[list[int]] = [x]
    count_in_list = sys.getrefcount(x)
    logger.info("After appending x to list: refcount = %d", count_in_list)
    # Expected: 4

    del y
    count_after_del = sys.getrefcount(x)
    logger.info("After del y: refcount = %d", count_after_del)
    # Expected: 3

    ref_list.pop()
    count_after_pop = sys.getrefcount(x)
    logger.info("After removing from list: refcount = %d", count_after_pop)
    # Expected: 2

    def inner() -> None:
        logger.info("Inside function: refcount = %d", sys.getrefcount(x))
        # Expected: ~4 (x, closure cell, getrefcount arg, frame reference)

    inner()


demonstrate_refcount()
```

**Key insight:** `sys.getrefcount()` always reads 1 higher than "real" count because passing the object to the function creates a temporary reference.

---

## Exercise 2: Reference Cycles and gc.collect()

```python
import gc
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class Node:
    """Node with a back-reference that creates a cycle."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.other: Optional["Node"] = None

    def __repr__(self) -> str:
        return f"Node({self.name!r})"

    def __del__(self) -> None:
        logger.info("__del__ called on %r", self)


def create_cycle() -> None:
    """Create two nodes that reference each other, then drop all external refs."""
    gc.disable()          # ensure the cyclic GC doesn't run automatically
    gc.collect()          # start from a clean slate
    gc.garbage.clear()

    a = Node("A")
    b = Node("B")
    a.other = b           # A → B
    b.other = a           # B → A  (cycle!)

    logger.info("Before deleting variables — gc.garbage: %s", gc.garbage)

    # Drop the only external references; refcount hits 0 for neither (cycle)
    del a, b

    logger.info("After del a, b — gc.garbage before collect: %s", gc.garbage)

    # Run the cyclic garbage collector
    unreachable_count = gc.collect()
    logger.info("gc.collect() found %d unreachable object(s)", unreachable_count)
    logger.info("gc.garbage after collect: %s", gc.garbage)
    # gc.garbage is normally empty unless objects define __del__ in CPython < 3.4;
    # since 3.4 objects with __del__ CAN be collected — you may still see __del__ fired.

    gc.enable()


create_cycle()
```

**Key insight:** CPython's reference counting alone cannot reclaim cycles. The cyclic GC (generation-based) detects them. Objects with `__del__` were historically moved to `gc.garbage` to avoid ordering issues; since Python 3.4 they are safely finalised.

---

## Exercise 3: Integer Interning

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_interning() -> None:
    """Show CPython's small-integer cache (-5..256) and its limits."""

    # --- Interned range ---
    for val in (-5, 0, 100, 256):
        a = val
        b = val
        logger.info(
            "val=%4d  id(a)==id(b): %-5s  a is b: %-5s",
            val,
            id(a) == id(b),
            a is b,
        )
    # All True — same object reused

    logger.info("---")

    # --- Outside the interned range ---
    # Use a function to guarantee separate compilation units (avoids compile-time
    # constant folding that the REPL / module level may perform).
    def make_int(n: int) -> int:
        return n + 0  # trivial expression to defeat constant folding

    for val in (257, 1000, -6):
        a = make_int(val)
        b = make_int(val)
        logger.info(
            "val=%4d  id(a)==id(b): %-5s  a is b: %-5s  a==b: %-5s",
            val,
            id(a) == id(b),
            a is b,
            a == b,
        )
    # `is` may be False (different objects); `==` always True


demonstrate_interning()
```

**Key insight:** Only compare integers with `==`, never `is`. The interning range is a CPython implementation detail, not guaranteed by the language spec.

---

## Exercise 4: tracemalloc — Top 5 Allocations

```python
import tracemalloc
import logging
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class AllocationStat(NamedTuple):
    filename: str
    lineno: int
    size_kb: float
    count: int


def memory_hungry_work() -> list[str]:
    """Creates many strings to show up in tracemalloc."""
    parts: list[str] = []
    for i in range(50_000):
        parts.append(f"item-{i:06d}")
    joined = "-".join(parts)          # one large string
    return [joined, *parts[:100]]     # return mix so nothing is optimised away


def profile_allocations() -> list[AllocationStat]:
    tracemalloc.start()

    _ = memory_hungry_work()

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot.statistics("lineno")[:5]

    results: list[AllocationStat] = []
    for stat in top_stats:
        frame = stat.traceback[0]
        info = AllocationStat(
            filename=frame.filename,
            lineno=frame.lineno,
            size_kb=stat.size / 1024,
            count=stat.count,
        )
        results.append(info)
        logger.info(
            "%s:%d  size=%.1f KiB  count=%d",
            info.filename,
            info.lineno,
            info.size_kb,
            info.count,
        )

    return results


stats = profile_allocations()
```

**Key insight:** `tracemalloc.take_snapshot()` captures per-line allocation data. `statistics("lineno")` ranks by total size; use `"traceback"` for full call-stack grouping.

---

## Exercise 5: Object Pool with weakref.WeakValueDictionary

```python
import weakref
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ExpensiveResource:
    """Simulates a resource that is costly to create."""

    _instance_count: int = 0

    def __init__(self, key: str) -> None:
        ExpensiveResource._instance_count += 1
        self.key = key
        self._serial = ExpensiveResource._instance_count
        logger.info("Created ExpensiveResource(%r) serial=%d", key, self._serial)

    def __repr__(self) -> str:
        return f"ExpensiveResource({self.key!r}, serial={self._serial})"

    def __del__(self) -> None:
        logger.info("Destroyed %r", self)


class ObjectPool:
    """
    Pool backed by WeakValueDictionary.
    Returns a cached object while any caller holds a strong reference;
    creates a new one once the previous is garbage-collected.
    """

    def __init__(self) -> None:
        self._cache: weakref.WeakValueDictionary[str, Any] = (
            weakref.WeakValueDictionary()
        )

    def get(self, key: str) -> ExpensiveResource:
        resource = self._cache.get(key)
        if resource is not None:
            logger.info("Cache HIT for key=%r -> %r", key, resource)
            return resource

        logger.info("Cache MISS for key=%r — creating new resource", key)
        resource = ExpensiveResource(key)
        self._cache[key] = resource
        return resource

    def __len__(self) -> int:
        return len(self._cache)


def demo_pool() -> None:
    pool = ObjectPool()

    # First access — miss
    r1 = pool.get("db-connection")
    logger.info("Pool size: %d", len(pool))

    # Second access while r1 alive — hit (same object)
    r2 = pool.get("db-connection")
    assert r1 is r2, "Should return the same object"

    # Drop all strong references → object is GC'd
    del r1, r2
    logger.info("Dropped references — pool size: %d (weak ref gone)", len(pool))

    # Next access — miss again, new object created
    r3 = pool.get("db-connection")
    logger.info("New resource: %r", r3)


demo_pool()
```

**Key insight:** `WeakValueDictionary` holds only weak references. The pool acts as a lightweight cache that doesn't prevent GC — ideal for expensive-to-create, reusable objects (connections, parsed schemas, compiled regexes).
