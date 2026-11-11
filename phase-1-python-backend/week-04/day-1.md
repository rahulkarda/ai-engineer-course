# Phase 1 · Week 04 · Day 1 — Iterables, Iterators & Generators

> Objective: Understand Python's iterator protocol, implement custom iterables, and harness generators for memory-efficient lazy sequences.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-5]] (functions, *args/**kwargs)

## Concept

### Iterable vs Iterator

An **iterable** is any object that can return an iterator — it implements `__iter__()`.
An **iterator** is an object that produces values one at a time — it implements both `__iter__()` and `__next__()`.

```
for x in collection:    # Python desugars this to:
    ...
# 1. calls iter(collection) -> gets an iterator object
# 2. repeatedly calls next(iterator) -> gets each value
# 3. catches StopIteration -> exits the loop
```

Every `for` loop, list comprehension, `zip()`, `map()`, `filter()`, and `*unpacking` uses this protocol under the hood.

### Generators

A **generator function** contains `yield`. Calling it returns a **generator object** (which is both an iterable and an iterator). Execution is **suspended** at each `yield` and **resumed** on the next `next()` call.

```python
def count_up(n: int):
    i = 0
    while i < n:
        yield i   # suspend here, hand value to caller
        i += 1    # resume here on next next() call
```

Generator **expressions** are the comprehension equivalent:
```python
squares = (x * x for x in range(1_000_000))  # lazy — no list built
```

## How it works internally

### The iterator protocol in CPython

When the interpreter sees `for x in obj:`, it calls `type(obj).__iter__(obj)`. The returned iterator is stored as a temporary variable. The interpreter then calls `type(iterator).__next__(iterator)` in a loop, catching `StopIteration` to break.

`StopIteration` is a signal, not an error — it propagates out of `__next__` to end iteration cleanly.

### Generator frame suspension

A generator object wraps a **code frame** (a `PyFrameObject`). When you call `next()`:
1. CPython resumes execution of that frame from the byte after the last `YIELD_VALUE` opcode.
2. Local variables and the value stack are preserved across `yield`.
3. When the function body exits (or hits `return`), `StopIteration` is raised automatically.

This means generators are **O(1) memory** for the running state, regardless of how many values they could produce. A list comprehension materialises all values; a generator expression materialises none until asked.

### `yield from`

`yield from iterable` delegates to a sub-generator: it forwards `next()`, `send()`, and `throw()` calls through, and collects the sub-generator's return value. Useful for composing pipelines without manual loops.

```python
def chain(*iterables):
    for it in iterables:
        yield from it
```

## Worked example

```python
"""
iterators_demo.py — custom iterable, generator pipeline, memory comparison.
Run: python iterators_demo.py
"""
from __future__ import annotations
import logging
import sys
from collections.abc import Iterator, Generator
from typing import TypeVar

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

T = TypeVar("T")


# ── 1. Custom iterable ────────────────────────────────────────────────────────

class NumberRange:
    """An iterable that yields integers in [start, stop)."""

    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop

    def __iter__(self) -> Iterator[int]:
        return _NumberRangeIterator(self.start, self.stop)

    def __repr__(self) -> str:
        return f"NumberRange({self.start}, {self.stop})"


class _NumberRangeIterator:
    """Separate iterator object — stateful, single-use."""

    def __init__(self, current: int, stop: int) -> None:
        self._current = current
        self._stop = stop

    def __iter__(self) -> "_NumberRangeIterator":
        return self   # iterators return themselves

    def __next__(self) -> int:
        if self._current >= self._stop:
            raise StopIteration
        value = self._current
        self._current += 1
        return value


# ── 2. Generator pipeline ─────────────────────────────────────────────────────

def integers_from(start: int = 0) -> Generator[int, None, None]:
    """Infinite generator — safe because it's lazy."""
    n = start
    while True:
        yield n
        n += 1


def take(n: int, source: Iterator[T]) -> Generator[T, None, None]:
    """Take the first n items from any iterator."""
    for i, value in enumerate(source):
        if i >= n:
            return
        yield value


def evens(source: Iterator[int]) -> Generator[int, None, None]:
    """Filter: keep only even numbers."""
    for v in source:
        if v % 2 == 0:
            yield v


# ── 3. Memory comparison ──────────────────────────────────────────────────────

def memory_demo(n: int = 1_000_000) -> None:
    # List materialises all values immediately
    list_bytes = sys.getsizeof(list(range(n)))
    # Generator stores only the frame — tiny
    gen_bytes = sys.getsizeof(x for x in range(n))
    log.info("List of %d ints: %d bytes", n, list_bytes)
    log.info("Generator for %d ints: %d bytes", n, gen_bytes)


# ── 4. yield from ─────────────────────────────────────────────────────────────

def flatten(nested: list) -> Generator:
    """Recursively flatten nested lists using yield from."""
    for item in nested:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


if __name__ == "__main__":
    # Custom iterable
    rng = NumberRange(1, 6)
    log.info("NumberRange: %s", list(rng))
    log.info("Can iterate again: %s", list(rng))  # iterable reusable; iterator is not

    # Generator pipeline
    first_10_evens = list(take(10, evens(integers_from(0))))
    log.info("First 10 even non-negative integers: %s", first_10_evens)

    # Memory
    memory_demo()

    # Flatten
    nested = [1, [2, [3, 4], 5], [6, 7]]
    log.info("Flattened: %s", list(flatten(nested)))
```

## Exercises

1. Implement a `Fibonacci` iterable class (not a generator) that yields the Fibonacci sequence up to a given maximum value. It must be re-iterable (each `for` loop starts fresh).

2. Write a generator function `running_average(source: Iterable[float]) -> Generator[float, None, None]` that yields the cumulative average after each new value.

3. Implement a `chunked(iterable, size)` generator that groups items into chunks of `size`. The last chunk may be smaller than `size`. `list(chunked(range(10), 3))` should return `[[0,1,2],[3,4,5],[6,7,8],[9]]`.

4. Using only generators (no intermediate lists), write a pipeline that reads lines from a file path, strips whitespace, filters blank lines, and yields each non-blank line upper-cased. Compose it as three separate generator functions and chain them.

5. Explain the difference between an iterable and an iterator. Why does a list qualify as an iterable but not an iterator? What happens if you call `next()` on a list?

(Solutions: ../solutions/week-04/day-1.md)

## Self-check
- [ ] I can explain the difference between an iterable and an iterator.
- [ ] I can implement `__iter__` and `__next__` on a custom class.
- [ ] I can write a generator function that yields values lazily.
- [ ] I can compose generator pipelines without building intermediate lists.
- [ ] I understand why generators are memory-efficient compared to lists.
- [ ] I can use `yield from` to delegate to a sub-generator.

## Supplementary links
- [Python Docs — Iterator types](https://docs.python.org/3/library/stdtypes.html#iterator-types)
- [PEP 255 — Simple Generators](https://peps.python.org/pep-0255/)
- [PEP 380 — yield from](https://peps.python.org/pep-0380/)
- [Real Python — Generators](https://realpython.com/introduction-to-python-generators/)
