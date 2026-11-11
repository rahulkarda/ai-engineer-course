# Week 4 Day 1 — Solutions: Iterables, Iterators & Generators

---

## Exercise 1: `Fibonacci` Iterable Class

An iterable must implement `__iter__`, returning an iterator object. An iterator implements both `__iter__` (returning `self`) and `__next__`. To make a class **re-iterable**, `__iter__` must return a *fresh* iterator each time — not `self`. The cleanest pattern is a separate inner iterator class.

```python
from __future__ import annotations
from typing import Iterator


class Fibonacci:
    """Re-iterable Fibonacci sequence up to a maximum value (inclusive)."""

    def __init__(self, max_value: int) -> None:
        self.max_value = max_value

    def __iter__(self) -> Iterator[int]:
        return _FibonacciIterator(self.max_value)


class _FibonacciIterator:
    def __init__(self, max_value: int) -> None:
        self._max = max_value
        self._a = 0
        self._b = 1

    def __iter__(self) -> _FibonacciIterator:
        return self

    def __next__(self) -> int:
        if self._a > self._max:
            raise StopIteration
        value = self._a
        self._a, self._b = self._b, self._a + self._b
        return value


# Usage
fib = Fibonacci(100)
print(list(fib))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
print(list(fib))  # Same result — re-iterable confirmed
```

**Key point:** If `__iter__` returned `self`, re-iteration would silently produce an empty sequence the second time because the internal state would be exhausted.

---

## Exercise 2: `running_average` Generator

A generator function suspends at each `yield`, preserving local state between calls. We accumulate the total and count without any intermediate list.

```python
from collections.abc import Generator, Iterable


def running_average(source: Iterable[float]) -> Generator[float, None, None]:
    """Yield the cumulative average after each new value from source."""
    total: float = 0.0
    count: int = 0
    for value in source:
        total += value
        count += 1
        yield total / count


# Usage
data = [10.0, 20.0, 30.0, 40.0]
for avg in running_average(data):
    print(avg)
# 10.0
# 15.0
# 20.0
# 25.0
```

**Space complexity:** O(1) — no list of values stored, only `total` and `count`.

---

## Exercise 3: `chunked` Generator

Split any iterable into chunks of at most `size` items. `itertools.islice` is perfect here: it lazily pulls exactly `size` items from an iterator, avoiding slicing (which requires a sequence).

```python
from collections.abc import Generator, Iterable
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def chunked(iterable: Iterable[T], size: int) -> Generator[list[T], None, None]:
    """Yield successive non-overlapping chunks of `size` from iterable."""
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


# Usage
print(list(chunked(range(10), 3)))
# [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

print(list(chunked("abcde", 2)))
# [['a', 'b'], ['c', 'd'], ['e']]
```

**Why `iter()` first?** `islice` needs an iterator, not just an iterable. Calling `iter()` once and passing the *same* iterator into repeated `islice` calls is what makes them advance through the sequence rather than restart.

---

## Exercise 4: Generator Pipeline (File Lines)

Three separate generator functions chained together. No intermediate lists anywhere in the pipeline — each stage pulls lazily from the previous one.

```python
from collections.abc import Generator, Iterator
from pathlib import Path


def read_lines(path: str | Path) -> Generator[str, None, None]:
    """Yield raw lines from a file, one at a time."""
    with open(path, encoding="utf-8") as f:
        yield from f


def strip_lines(lines: Iterator[str]) -> Generator[str, None, None]:
    """Yield each line with leading/trailing whitespace removed."""
    for line in lines:
        yield line.strip()


def filter_and_upper(lines: Iterator[str]) -> Generator[str, None, None]:
    """Skip blank lines; yield remaining lines upper-cased."""
    for line in lines:
        if line:
            yield line.upper()


def process_file(path: str | Path) -> Generator[str, None, None]:
    """Full pipeline: read → strip → filter blanks → upper-case."""
    return filter_and_upper(strip_lines(read_lines(path)))


# Usage
# for line in process_file("data.txt"):
#     print(line)
```

**Memory profile:** At any moment only one line lives in memory regardless of file size. The `with open(...)` context in `read_lines` stays open for the lifetime of the generator — `close()` is called when the generator is garbage-collected or explicitly closed.

---

## Exercise 5: Iterable vs Iterator — Conceptual

**Iterable**
Any object that implements `__iter__()` and returns an iterator. It can be iterated *multiple times* — each call to `__iter__()` hands back a fresh iterator with its own position. Examples: `list`, `tuple`, `str`, `dict`, custom classes with `__iter__`.

**Iterator**
An object that implements both `__iter__()` (returns `self`) and `__next__()` (returns the next value or raises `StopIteration`). It has *state* — it knows where it is in the sequence. Once exhausted, it is done; there is no reset.

**Why is a list an iterable but not an iterator?**

```python
lst = [1, 2, 3]

# list has __iter__ — it is iterable
it = iter(lst)         # calls lst.__iter__(), returns a list_iterator

# list does NOT have __next__
next(lst)              # TypeError: 'list' object is not an iterator

# The iterator does have __next__
print(next(it))        # 1
print(next(it))        # 2

# list is re-iterable: iter() always gives a fresh iterator
it2 = iter(lst)
print(next(it2))       # 1  (starts over)
print(next(it))        # 3  (it1 kept its position)
```

**`next()` on a list raises `TypeError: 'list' object is not an iterator`** because `list` has no `__next__` method.

**Rule of thumb:**
- Every iterator is an iterable (it has `__iter__` returning `self`).
- Not every iterable is an iterator (lists, sets, dicts are not).
- A generator object is always an iterator (and therefore also an iterable).
