# Phase 1 · Week 04 · Day 3 — Context Managers & Shallow vs Deep Copy

> Objective: Use the `with` statement to guarantee resource cleanup, build custom context managers, and understand the difference between shallow and deep object copies.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-2]] (decorators & closures)

## Concept

### The `with` statement and context managers

The `with` statement guarantees that setup and teardown code runs even when an exception occurs. It relies on the **context manager protocol**: any object that implements `__enter__` and `__exit__` can be used in a `with` block.

```python
with open("data.txt") as f:
    data = f.read()
# file is closed here regardless of whether read() raised an exception
```

Under the hood:
```
cm = open("data.txt")
f = cm.__enter__()          # setup; return value bound to 'f'
try:
    data = f.read()
except BaseException as exc:
    if not cm.__exit__(type(exc), exc, exc.__traceback__):
        raise               # re-raise unless __exit__ returns True
else:
    cm.__exit__(None, None, None)
```

`__exit__` receives three arguments: `exc_type`, `exc_val`, `exc_tb`. Returning a truthy value **suppresses** the exception; returning `None` / `False` lets it propagate.

### Implementing `__enter__` and `__exit__`

```python
class ManagedResource:
    def __enter__(self):
        # acquire the resource; return the value bound by 'as'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # release the resource
        # return True to suppress exceptions, False/None to propagate
        return False
```

### `contextlib.contextmanager`

For simpler cases, the `@contextlib.contextmanager` decorator lets you write a context manager as a generator function with exactly one `yield`. Code before `yield` runs on `__enter__`; code after `yield` runs on `__exit__`. Wrap the `yield` in `try/finally` to guarantee cleanup on exceptions.

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire()   # setup
    try:
        yield resource     # value bound by 'as'
    finally:
        release(resource)  # teardown — always runs
```

### Common use cases

- **File / network / DB handles**: guarantee close/disconnect on exit.
- **Locks**: `threading.Lock` is a context manager — `__enter__` acquires, `__exit__` releases.
- **Transactions**: commit on clean exit, rollback on exception.
- **Temporary state**: change a setting and restore it regardless of outcome (e.g., `cd` to a directory, mock a global).
- **Timing / profiling**: start a timer in `__enter__`, log elapsed in `__exit__`.

### Multiple context managers

Python allows multiple context managers in one `with` statement (evaluated left to right, exited right to left):

```python
with open("src.txt") as src, open("dst.txt", "w") as dst:
    dst.write(src.read())
```

---

### Object references, identity, and equality

Before covering copy, the underlying model matters.

In Python, variables hold **references** (pointers) to objects — they do not hold values directly.

```python
a = [1, 2, 3]
b = a           # b and a point to the same list object
b.append(4)
print(a)        # [1, 2, 3, 4] — a is affected
```

Two operators test object identity and equality differently:
- `is` — tests **identity**: are `a` and `b` the same object in memory? (`id(a) == id(b)`)
- `==` — tests **equality**: do `a` and `b` have the same value? (calls `__eq__`)

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b   # True  — same contents
a is b   # False — different objects
```

### Shallow copy (`copy.copy`)

A **shallow copy** creates a new container object, but populates it with **references to the same inner objects** as the original.

```python
import copy
original = [[1, 2], [3, 4]]
shallow  = copy.copy(original)

shallow is original        # False — different list objects
shallow[0] is original[0]  # True  — same inner list objects
shallow[0].append(99)
print(original[0])         # [1, 2, 99] — mutation visible in original
```

Shallow copy is created by:
- `copy.copy(obj)`
- List/dict/set literals with slice: `lst[:]`, `dict.copy()`, `set.copy()`
- `list(other)`, `dict(other)`, `set(other)`

### Deep copy (`copy.deepcopy`)

A **deep copy** recursively duplicates every object in the graph — new container and new inner objects. Mutations to the copy never affect the original.

```python
deep = copy.deepcopy(original)
deep[0].append(99)
print(original[0])   # [1, 2] — original unaffected
```

`copy.deepcopy` handles cycles in object graphs via a memo dict (`{id: copy}`) to avoid infinite recursion.

### When to use which

| Situation | Use |
|---|---|
| Container of immutable objects (ints, strings, tuples of primitives) | Shallow copy is sufficient — inner objects cannot be mutated |
| Container of mutable objects and you need full independence | Deep copy |
| Large object graph, performance critical | Prefer shallow + explicit mutation discipline |
| Custom class with resources (file handles, locks) | Override `__copy__` / `__deepcopy__` |

Overuse of `deepcopy` on large nested structures is a common performance mistake. Profile first; copy selectively.

## How it works internally

### `__exit__` exception suppression

When the `with` body raises, Python calls `__exit__(exc_type, exc_val, exc_tb)`. If it returns a truthy value, the interpreter swallows the exception and continues after the `with` block. This is intentional for cases like `contextlib.suppress`. Returning `None` or `False` lets the exception propagate — this is the default and usually what you want; silently swallowing exceptions is dangerous.

### `@contextmanager` mechanics

The decorator wraps the generator in a `_GeneratorContextManager` class. On `__enter__`, it calls `next()` to advance to the `yield`. On `__exit__`, it either calls `next()` again (clean exit) or calls `generator.throw(exc_type, exc_val, exc_tb)` (exception path). The `try/finally` inside your generator ensures the `finally` clause runs on `throw`. If the generator does not catch the thrown exception and re-raises it, `__exit__` propagates it; if the generator suppresses it and returns, `__exit__` returns `True`.

### Shallow copy internals

Most built-in containers implement `__copy__` by constructing a new container and filling it with `list(self)` (references only). For custom classes, `copy.copy` calls `cls.__copy__(self)` if present, otherwise falls back to `__reduce_ex__` / `__reduce__`, which is the pickle protocol.

### Deep copy memo dict

`copy.deepcopy(obj, memo)` passes the same `memo` dict through every recursive call. Before copying an object, it checks `memo[id(obj)]`; if found, it returns the already-copied version. This handles both cycles (an object that references itself) and DAGs (shared sub-objects that should map to the same copy). The memo dict also prevents re-copying large shared sub-structures unnecessarily.

## Worked example

```python
"""
context_copy_demo.py — custom context managers, contextlib.contextmanager,
                        temporary-state pattern, shallow vs deep copy.
Run: python context_copy_demo.py
"""
from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── 1. Class-based context manager: timer ─────────────────────────────────────

class Timer:
    """Measure elapsed wall-clock time for a block of code."""

    def __init__(self, label: str = "block") -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self   # bound to 'as' target

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.elapsed = time.perf_counter() - self._start
        log.info("[%s] elapsed: %.6f s", self.label, self.elapsed)
        return False  # do not suppress exceptions


# ── 2. contextmanager decorator: temporary directory ─────────────────────────

@contextmanager
def temp_directory() -> Generator[Path, None, None]:
    """Create a temporary directory, yield its Path, delete it on exit."""
    tmp = Path(tempfile.mkdtemp())
    log.info("Created temp dir: %s", tmp)
    try:
        yield tmp
    finally:
        # Remove all files then the dir — real cleanup would use shutil.rmtree
        for child in tmp.iterdir():
            child.unlink()
        tmp.rmdir()
        log.info("Removed temp dir: %s", tmp)


# ── 3. Temporary-state context manager: change working directory ──────────────

@contextmanager
def working_directory(path: Path) -> Generator[None, None, None]:
    """Temporarily change the process working directory."""
    original = Path.cwd()
    os.chdir(path)
    log.info("Changed cwd to: %s", path)
    try:
        yield
    finally:
        os.chdir(original)
        log.info("Restored cwd to: %s", original)


# ── 4. Exception-suppressing context manager ─────────────────────────────────

@contextmanager
def ignore_errors(*exc_types: type[Exception]) -> Generator[None, None, None]:
    """Suppress specified exception types — log and continue."""
    try:
        yield
    except exc_types as exc:
        log.warning("Ignored %s: %s", type(exc).__name__, exc)


# ── 5. Shallow vs deep copy ───────────────────────────────────────────────────

def demonstrate_copies() -> None:
    original: list[list[int]] = [[1, 2, 3], [4, 5, 6]]

    # Shallow copy
    shallow = copy.copy(original)
    assert shallow is not original, "different container objects"
    assert shallow[0] is original[0], "same inner list objects"

    shallow[0].append(99)  # mutate inner list through shallow copy
    log.info("After shallow mutation, original[0]: %s", original[0])
    # original[0] is now [1, 2, 3, 99] — shared reference!

    # Reset for deep copy demo
    original = [[1, 2, 3], [4, 5, 6]]
    deep = copy.deepcopy(original)
    assert deep is not original
    assert deep[0] is not original[0], "independent inner list objects"

    deep[0].append(99)     # mutate inner list through deep copy
    log.info("After deep mutation, original[0]: %s", original[0])
    # original[0] is still [1, 2, 3] — independent copy


# ── 6. Custom __deepcopy__ to exclude un-copyable resources ──────────────────

class Connection:
    """Simulates a resource that cannot be deep-copied."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._socket = object()  # pretend this is a live socket

    def __deepcopy__(self, memo: dict) -> "Connection":
        # Create a new Connection with the same DSN but a fresh socket
        new = Connection(self.dsn)
        memo[id(self)] = new
        log.info("Custom __deepcopy__: new Connection(%s)", self.dsn)
        return new


if __name__ == "__main__":
    # Timer
    with Timer("sleep demo") as t:
        time.sleep(0.05)
    log.info("Stored elapsed: %.6f s", t.elapsed)

    # Temp directory
    with temp_directory() as tmp:
        (tmp / "hello.txt").write_text("hi")
        log.info("Files in temp dir: %s", list(tmp.iterdir()))
    # directory is gone here

    # Temporary cwd
    with temp_directory() as tmp:
        with working_directory(tmp):
            log.info("Inside cwd: %s", Path.cwd())
        log.info("Back to:    %s", Path.cwd())

    # suppress from contextlib (stdlib equivalent of ignore_errors)
    with suppress(FileNotFoundError):
        Path("/nonexistent/path").read_text()
    log.info("Continued after suppressed FileNotFoundError")

    # Custom ignore_errors
    with ignore_errors(ZeroDivisionError):
        _ = 1 / 0

    # Multiple context managers in one with
    with temp_directory() as src_dir, temp_directory() as dst_dir:
        (src_dir / "data.txt").write_text("content")
        log.info("src: %s  dst: %s", src_dir, dst_dir)

    # Shallow vs deep copy
    demonstrate_copies()

    # Custom __deepcopy__
    conn = Connection("postgresql://localhost/mydb")
    conn_copy = copy.deepcopy(conn)
    log.info("conn is conn_copy: %s", conn is conn_copy)
    log.info("conn.dsn == conn_copy.dsn: %s", conn.dsn == conn_copy.dsn)
```

## Exercises

1. Implement a `DatabaseTransaction` context manager class. On `__enter__`, log "BEGIN". On clean exit, log "COMMIT". On any exception, log "ROLLBACK" and re-raise the exception. Test it with both a clean block and a block that raises `ValueError`.

2. Write a `@contextmanager` function `atomic_write(path: Path)` that writes to a temporary file and, on clean exit, atomically replaces the target path. On exception, delete the temporary file and do not touch the target. Hint: use `Path.replace()` for the atomic move.

3. Write a `set_env(**variables)` context manager that temporarily sets environment variables from keyword arguments and restores their original values (or deletes them if they did not exist) on exit. Use `os.environ`.

4. Given `a = {"x": [1, 2], "y": [3, 4]}`, write a short script that demonstrates the difference between `copy.copy(a)` and `copy.deepcopy(a)` by modifying a nested list in each copy and printing both the copy and the original after each mutation.

5. You have a class `Config` with a `plugins: list[Plugin]` attribute, where `Plugin` holds an open `socket.socket`. Explain why `copy.deepcopy(config)` would fail, and describe two strategies to handle this: implementing `__deepcopy__`, and using `__reduce__`.

(Solutions: ../solutions/week-04/day-3.md)

## Self-check
- [ ] I can explain what `__enter__` and `__exit__` do and when each is called.
- [ ] I can implement a context manager as a class and as a `@contextmanager` generator.
- [ ] I know when `__exit__` should return `True` vs `False` and why returning `True` carelessly is dangerous.
- [ ] I can explain the difference between `is` and `==`.
- [ ] I can predict which mutations affect the original after a shallow copy vs a deep copy.
- [ ] I can implement `__deepcopy__` on a class that holds un-copyable resources.

## Supplementary links
- [Python Docs — contextlib](https://docs.python.org/3/library/contextlib.html)
- [Python Docs — copy — Shallow and deep copy operations](https://docs.python.org/3/library/copy.html)
- [PEP 343 — The "with" Statement](https://peps.python.org/pep-0343/)
- [Real Python — Context Managers and Python's with Statement](https://realpython.com/python-with-statement/)
- [Real Python — Shallow vs Deep Copying of Python Objects](https://realpython.com/copying-python-objects/)
