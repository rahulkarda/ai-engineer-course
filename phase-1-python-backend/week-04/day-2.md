# Phase 1 · Week 04 · Day 2 — Decorators & Closures

> Objective: Master Python closures and the decorator pattern to add reusable behaviour to functions and classes without modifying their source.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-1]] (iterables, generators)

## Concept

### First-class functions

In Python, functions are objects. They can be assigned to variables, stored in data structures, passed as arguments, and returned from other functions. This is the foundation of decorators.

```python
def greet(name: str) -> str:
    return f"Hello, {name}"

say_hello = greet          # assign function to a variable
say_hello("world")         # identical to greet("world")

funcs = [greet, str.upper] # store in a list
```

### Closures

A **closure** is a function that remembers the variables from its enclosing scope even after that scope has finished executing. The captured variables are stored in **cell objects** accessible via `func.__closure__`.

```python
def make_adder(n: int):
    # n is a free variable — it lives in the enclosing scope
    def add(x: int) -> int:
        return x + n   # closes over n
    return add

add5 = make_adder(5)
add5(3)   # -> 8  (n=5 is remembered)
```

### LEGB scope resolution

Python resolves names in this order:
1. **L**ocal — inside the current function
2. **E**nclosing — in any enclosing function scopes (for closures)
3. **G**lobal — module-level names
4. **B**uilt-ins — `len`, `print`, etc.

The `nonlocal` keyword allows a nested function to rebind (not just read) a variable in the enclosing scope.

```python
def counter():
    count = 0
    def increment() -> int:
        nonlocal count   # rebind, not shadow
        count += 1
        return count
    return increment
```

### Decorators

A **decorator** is syntactic sugar for wrapping a function in another function. The `@` syntax is equivalent to reassigning the name:

```python
@decorator
def func(): ...
# identical to:
# func = decorator(func)
```

A decorator receives the original function, returns a replacement (the **wrapper**), and usually calls the original inside. Use `@functools.wraps(func)` on the wrapper to copy the original's `__name__`, `__doc__`, `__annotations__`, and `__module__` — without it, introspection and tooling break.

```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # do something before
        result = func(*args, **kwargs)
        # do something after
        return result
    return wrapper
```

### Parametrised decorators

When a decorator needs arguments, add one more level of nesting: a factory that returns the decorator:

```python
@repeat(times=3)
def greet(): ...
# desugars to:
# greet = repeat(times=3)(greet)
# repeat(times=3) must return a decorator
```

Structure: outer function (accepts decorator args) → middle function (accepts func) → inner function (wrapper).

### Class-based decorators

A class with `__call__` can act as a decorator. Useful when the decorator needs to maintain state across calls.

```python
class CountCalls:
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.func(*args, **kwargs)
```

### Stacking decorators

Multiple decorators are applied bottom-up (closest to the function first):

```python
@A
@B
@C
def func(): ...
# equivalent to: func = A(B(C(func)))
```

## How it works internally

### Cell objects and free variables

When CPython compiles a closure, it detects that `n` in `make_adder` is referenced by the inner function. It allocates a **cell object** — a heap-allocated box — shared between the enclosing frame and the closure. The enclosing frame stores `n` in the cell instead of the normal fast-local slot. The inner function's `__closure__` tuple holds references to these cells.

This is why mutations to the cell (via `nonlocal`) are visible to all functions that share it. The cell object is not the value; it is a pointer to the value.

### Wrapper call chain and `functools.wraps`

When `@functools.wraps(func)` is applied, it calls `functools.update_wrapper`, which copies `__module__`, `__name__`, `__qualname__`, `__annotations__`, `__doc__`, and `__dict__` from `func` onto the wrapper, and sets `wrapper.__wrapped__ = func`. This allows tools like `help()`, `inspect.signature()`, and pytest to see through the wrapper to the original function. Without it, all wrapped functions would appear as `wrapper` with no signature.

### Performance note

Each decorator call creates a new closure/function object. For hot paths called millions of times, the indirection adds a small but measurable overhead. Profile before abstracting hot loops with decorators.

## Worked example

```python
"""
decorators_demo.py — closures, function decorators, parametrised decorator,
                     class-based decorator, stacking.
Run: python decorators_demo.py
"""
from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ── 1. Closure: simple counter factory ────────────────────────────────────────

def make_counter(start: int = 0) -> Callable[[], int]:
    """Return a stateful counter closure."""
    count = start  # free variable captured by inner

    def increment() -> int:
        nonlocal count
        count += 1
        return count

    return increment


# ── 2. Simple decorator: logging ──────────────────────────────────────────────

def log_calls(func: F) -> F:
    """Log function name and arguments on every call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        log.info("Calling %s | args=%s kwargs=%s", func.__name__, args, kwargs)
        result = func(*args, **kwargs)
        log.info("  -> %s returned %r", func.__name__, result)
        return result

    return wrapper  # type: ignore[return-value]


# ── 3. Simple decorator: timing ───────────────────────────────────────────────

def timeit(func: F) -> F:
    """Measure and log wall-clock time for each call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log.info("%s took %.6f s", func.__name__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


# ── 4. Parametrised decorator: retry ─────────────────────────────────────────

def retry(
    times: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """Retry a function up to `times` times on the given exception types."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    log.warning(
                        "%s attempt %d/%d failed: %s",
                        func.__name__, attempt, times, exc,
                    )
            raise RuntimeError(
                f"{func.__name__} failed after {times} attempts"
            ) from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


# ── 5. Class-based decorator: call counter ────────────────────────────────────

class CountCalls:
    """Decorator that tracks how many times a function has been called."""

    def __init__(self, func: Callable[..., Any]) -> None:
        functools.update_wrapper(self, func)  # preserve metadata
        self.func = func
        self.call_count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        log.info("%s called %d time(s)", self.func.__name__, self.call_count)
        return self.func(*args, **kwargs)


# ── 6. Stacking decorators ────────────────────────────────────────────────────

# Applied bottom-up: log_calls wraps timeit wraps the function.
# Call order: log_calls wrapper -> timeit wrapper -> add -> timeit wrapper -> log_calls wrapper
@log_calls
@timeit
def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b


# ── 7. Demonstrate retry ──────────────────────────────────────────────────────

_attempt_tracker: dict[str, int] = {}

@retry(times=3, exceptions=(ValueError,))
def flaky_parse(value: str) -> int:
    """Succeeds on the third attempt only (demo)."""
    _attempt_tracker["calls"] = _attempt_tracker.get("calls", 0) + 1
    if _attempt_tracker["calls"] < 3:
        raise ValueError(f"Not ready yet (attempt {_attempt_tracker['calls']})")
    return int(value)


# ── 8. Class-based decorator usage ───────────────────────────────────────────

@CountCalls
def multiply(x: int, y: int) -> int:
    return x * y


if __name__ == "__main__":
    # Closure
    c1 = make_counter(0)
    c2 = make_counter(10)
    log.info("c1: %d %d %d", c1(), c1(), c1())   # 1 2 3
    log.info("c2: %d %d",    c2(), c2())           # 11 12  — independent state

    # Stacked decorators
    result = add(3, 4)
    log.info("add result: %d", result)

    # Metadata preserved despite decoration
    log.info("add.__name__  = %s", add.__name__)   # 'add', not 'wrapper'
    log.info("add.__doc__   = %s", add.__doc__)

    # Parametrised retry
    parsed = flaky_parse("42")
    log.info("Parsed: %d", parsed)

    # Class-based counter
    multiply(2, 3)
    multiply(4, 5)
    log.info("multiply.call_count = %d", multiply.call_count)
```

## Exercises

1. Write a `memoize` decorator that caches a function's return value keyed on its positional arguments. Call the underlying function only on a cache miss. Test it with a recursive Fibonacci function and verify it is faster than the un-memoized version.

2. Write a parametrised `validate_types` decorator that, given a mapping of parameter names to expected types, raises `TypeError` with a clear message if a caller passes a value of the wrong type. Hint: use `inspect.signature` to bind arguments to parameter names.

3. Implement a `rate_limit(calls: int, period: float)` decorator that raises `RuntimeError` if the decorated function is called more than `calls` times within any `period`-second window. Use `collections.deque` to track call timestamps.

4. Create a class-based `Singleton` decorator. When applied to a class, repeated calls to the class constructor should return the same instance. Verify that `MyClass() is MyClass()` returns `True`.

5. Explain what `functools.wraps` does and what breaks if you omit it. Write a short script that demonstrates the breakage by inspecting `__name__` and `__doc__` before and after adding `@functools.wraps`.

(Solutions: ../solutions/week-04/day-2.md)

## Self-check
- [ ] I can explain what a closure is and what a cell object stores.
- [ ] I can trace LEGB scope resolution for a given nested function.
- [ ] I can write a decorator that wraps a function and preserves its metadata.
- [ ] I can build a parametrised decorator (three-level nesting).
- [ ] I can implement a decorator as a class using `__call__`.
- [ ] I understand the order in which stacked decorators are applied.

## Supplementary links
- [Python Docs — functools.wraps](https://docs.python.org/3/library/functools.html#functools.wraps)
- [Python Docs — Closures (reference manual)](https://docs.python.org/3/reference/expressions.html#atom-identifiers)
- [PEP 318 — Decorators for Functions and Methods](https://peps.python.org/pep-0318/)
- [Real Python — Primer on Python Decorators](https://realpython.com/primer-on-python-decorators/)
- [Python Docs — inspect.signature](https://docs.python.org/3/library/inspect.html#inspect.signature)
