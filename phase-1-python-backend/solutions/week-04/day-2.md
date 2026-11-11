# Week 4 Day 2 — Solutions: Decorators & Closures

---

## Exercise 1: `memoize` Decorator

Cache results keyed on `args`. `@functools.wraps` copies the wrapped function's `__name__`, `__doc__`, and other metadata so the decorator is transparent to introspection tools.

```python
import functools
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def memoize(func: F) -> F:
    """Cache function return values keyed on positional args."""
    cache: dict[tuple[Any, ...], Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any) -> Any:
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    wrapper.cache = cache  # type: ignore[attr-defined]  # expose for testing
    return wrapper  # type: ignore[return-value]


# Usage
@memoize
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(35))   # fast — no exponential blowup
print(fibonacci.cache) # inspect cached entries
```

**Note:** This implementation only supports hashable positional arguments. For kwargs or unhashable types, use `functools.lru_cache` from the standard library.

---

## Exercise 2: `retry` Parametrised Decorator

A parametrised decorator is a callable that *returns* a decorator. The three-layer structure is: outer factory → decorator → wrapper.

```python
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(max_attempts: int = 3, delay_secs: float = 1.0) -> Callable[[F], F]:
    """Retry a function up to max_attempts times on any exception."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s",
                        attempt,
                        max_attempts,
                        func.__qualname__,
                        exc,
                    )
                    if attempt < max_attempts:
                        time.sleep(delay_secs)
            raise RuntimeError(
                f"{func.__qualname__} failed after {max_attempts} attempts"
            ) from last_exc

        return wrapper  # type: ignore[return-value]

    return decorator


# Usage
@retry(max_attempts=3, delay_secs=0.1)
def unstable_request(url: str) -> str:
    import random
    if random.random() < 0.7:
        raise ConnectionError("flaky network")
    return f"OK: {url}"
```

**`re-raise` design decision:** Wrapping in `RuntimeError ... from last_exc` preserves the original traceback via exception chaining while giving callers a single exception type to catch.

---

## Exercise 3: `make_counter` Closure

A closure captures variables from the enclosing scope into *cell objects*. The returned callable has a reference to the cell, not to the value — mutations are visible across calls.

```python
from collections.abc import Callable


def make_counter(start: int = 0, step: int = 1) -> Callable[[], int]:
    """Return a callable that increments by step on each call."""
    current = start

    def counter() -> int:
        nonlocal current
        value = current
        current += step
        return value

    return counter


# Usage
c = make_counter(start=10, step=5)
print(c())   # 10
print(c())   # 15
print(c())   # 20

c2 = make_counter()  # independent instance
print(c2())  # 0
print(c())   # 25  (c is unaffected by c2)
```

**`nonlocal`:** Without `nonlocal current`, the assignment `current += step` would create a new local variable, shadowing the cell — you'd get `UnboundLocalError`. `nonlocal` tells Python to mutate the existing cell in the enclosing frame.

---

## Exercise 4: `timer` Decorator

Use `logging` (not `print`) at DEBUG level so the timing output is suppressible in production without changing application code.

```python
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def timer(func: F) -> F:
    """Log execution time of the wrapped function at DEBUG level."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug("%s completed in %.4fs", func.__qualname__, elapsed)
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.debug("%s raised after %.4fs", func.__qualname__, elapsed)
            raise

    return wrapper  # type: ignore[return-value]


# Usage
logging.basicConfig(level=logging.DEBUG)

@timer
def slow_sum(n: int) -> int:
    return sum(range(n))

slow_sum(10_000_000)
# DEBUG: slow_sum completed in 0.3521s
```

**Why `perf_counter`?** `time.perf_counter()` has the highest available resolution for measuring short durations; `time.time()` is wall-clock and can jump on NTP adjustments.

---

## Exercise 5: `RateLimit` Class-Based Decorator

A class decorator implements `__call__`. State (the per-function call history) lives on the instance, making each decorated function independent.

```python
import functools
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class RateLimit:
    """Limit calls to at most `calls_per_second` per second using a sliding window."""

    def __init__(self, calls_per_second: float) -> None:
        if calls_per_second <= 0:
            raise ValueError("calls_per_second must be > 0")
        self._min_interval = 1.0 / calls_per_second
        self._window = 1.0
        self._calls: deque[float] = deque()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            now = time.monotonic()
            # Evict calls outside the rolling window
            while self._calls and self._calls[0] <= now - self._window:
                self._calls.popleft()

            max_calls = int(self._window / self._min_interval)
            if len(self._calls) >= max_calls:
                sleep_for = self._calls[0] + self._window - now
                if sleep_for > 0:
                    logger.debug(
                        "RateLimit: sleeping %.3fs before %s",
                        sleep_for,
                        func.__qualname__,
                    )
                    time.sleep(sleep_for)
                # Re-evict after sleep
                now = time.monotonic()
                while self._calls and self._calls[0] <= now - self._window:
                    self._calls.popleft()

            self._calls.append(time.monotonic())
            return func(*args, **kwargs)

        return wrapper


# Usage
@RateLimit(calls_per_second=2)
def fetch(url: str) -> str:
    return f"fetched {url}"

for i in range(5):
    fetch(f"https://example.com/{i}")
    # Will automatically throttle to 2 calls/second
```

---

## Exercise 6: LEGB Scope and Cell Objects

**LEGB** is Python's name lookup order:

| Letter | Scope | Description |
|--------|-------|-------------|
| **L** | Local | Names assigned inside the current function |
| **E** | Enclosing | Names in any enclosing `def` or `lambda` (from inner to outer) |
| **G** | Global | Names at module top level |
| **B** | Built-in | Pre-assigned names in the `builtins` module (`len`, `range`, …) |

```python
# Concrete LEGB example
x = "global"          # (G) global scope

def outer() -> None:
    x = "enclosing"   # (E) enclosing scope for inner()

    def inner() -> None:
        x = "local"   # (L) local scope
        print(x)      # → "local"  (L wins)

    inner()
    print(x)          # → "enclosing"  (L gone, E wins)

outer()
print(x)              # → "global"  (L and E gone)
```

**Cell object**

When an inner function references a variable from an enclosing function, CPython does not copy the value — it creates a *cell object* that both the enclosing frame and the closure share. The cell holds a pointer to the actual value.

```python
def make_adder(n: int):
    def add(x: int) -> int:
        return x + n          # n lives in a cell
    return add

adder = make_adder(10)

import dis
# adder.__code__.co_freevars → ('n',)   — names captured from enclosing scope
# adder.__closure__[0]       → <cell at 0x...>
# adder.__closure__[0].cell_contents → 10

print(adder.__closure__[0].cell_contents)  # 10
```

**Why cells matter for `nonlocal`:** When you write `nonlocal n`, Python modifies `cell.cell_contents` rather than creating a new local variable. Both the inner and outer function see the update because they share the same cell object.
