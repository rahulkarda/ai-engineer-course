# Week 4 Day 6 — Type Hints, Generics & TypedDict: Solutions

## Exercise 1: Generic Stack[T]

```python
from typing import Generic, TypeVar

T = TypeVar("T")


class Stack(Generic[T]):
    """A LIFO stack, fully typed."""

    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def __repr__(self) -> str:
        return f"Stack({self._items!r})"


# Usage
s: Stack[int] = Stack()
s.push(1)
s.push(2)
assert s.peek() == 2
assert s.pop() == 2
assert len(s) == 1
```

---

## Exercise 2: TypedDict with Required and NotRequired Fields

```python
from typing import Required, NotRequired, TypedDict


class DatabaseConfig(TypedDict, total=False):
    """
    Configuration for a database connection.

    total=False makes all fields NotRequired by default;
    use Required[] to make specific ones mandatory.
    """
    host: Required[str]          # always needed
    port: Required[int]          # always needed
    database: Required[str]      # always needed
    username: NotRequired[str]   # optional (default handled by driver)
    password: NotRequired[str]   # optional
    pool_size: NotRequired[int]  # optional, default 5
    ssl: NotRequired[bool]       # optional, default False


# Minimal valid config
minimal: DatabaseConfig = {"host": "localhost", "port": 5432, "database": "mydb"}

# Full config
full: DatabaseConfig = {
    "host": "prod-db.example.com",
    "port": 5432,
    "database": "transactions",
    "username": "app",
    "password": "s3cr3t",
    "pool_size": 10,
    "ssl": True,
}
```

**Alternative — mixed total approach (two classes):**

```python
class _DatabaseConfigRequired(TypedDict):
    host: str
    port: int
    database: str


class DatabaseConfig(_DatabaseConfigRequired, total=False):
    username: str
    password: str
    pool_size: int
    ssl: bool
```

Both approaches are equivalent; the `Required`/`NotRequired` form (PEP 655, Python 3.11+) is more concise.

---

## Exercise 3: Typed Decorator Using ParamSpec

```python
import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that logs execution time.
    Uses ParamSpec to preserve the wrapped function's full signature.
    """
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s took %.4f s", func.__qualname__, elapsed)
        return result

    return wrapper


@timed
def compute_sum(n: int, *, start: int = 0) -> int:
    return sum(range(start, n))


# Type checker knows: compute_sum(n: int, *, start: int = 0) -> int
result: int = compute_sum(1_000_000, start=1)
```

**Key insight:** `ParamSpec` captures both positional and keyword argument types. Without it, `*args: Any, **kwargs: Any` erases the signature and type checkers lose the ability to validate callers.

---

## Exercise 4: Protocol — Comparable Interface

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Comparable(Protocol):
    """Any object that supports < and == comparisons."""

    def __lt__(self, other: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...


def find_minimum(items: list[Comparable]) -> Comparable:
    """Return the smallest element from any list of Comparable items."""
    if not items:
        raise ValueError("Cannot find minimum of empty list")
    minimum = items[0]
    for item in items[1:]:
        if item < minimum:
            minimum = item
    return minimum


# Works with any Comparable — no inheritance required
assert find_minimum([3, 1, 4, 1, 5]) == 1
assert find_minimum(["banana", "apple", "cherry"]) == "apple"

# Runtime check
assert isinstance(42, Comparable)
assert isinstance("hello", Comparable)
```

**Key insight:** `Protocol` enables structural subtyping ("duck typing" with static checking). `@runtime_checkable` allows `isinstance()` checks at runtime, but only verifies the presence of methods, not their signatures.

---

## Exercise 5: Generic LRU Cache with TypeVar

```python
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """
    Fixed-capacity LRU cache.
    On capacity overflow the least-recently-used entry is evicted.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._store: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)   # mark as most-recently-used
        return self._store[key]

    def put(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._capacity:
            self._store.popitem(last=False)  # evict LRU (first item)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: object) -> bool:
        return key in self._store


# Usage
cache: LRUCache[str, int] = LRUCache(capacity=3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
cache.put("d", 4)   # evicts "a"

assert cache.get("a") is None   # evicted
assert cache.get("b") == 2
assert len(cache) == 3
```

---

## Exercise 6: Type Narrowing — isinstance, assert, TypeGuard

```python
from typing import TypeGuard


def process_value(value: int | str | None) -> str:
    # --- isinstance narrowing ---
    if isinstance(value, int):
        # type checker knows: value is int here
        return f"integer: {value * 2}"

    if isinstance(value, str):
        # type checker knows: value is str here
        return f"string: {value.upper()}"

    # At this point type checker knows: value is None
    return "null value"


# --- assert narrowing ---
def require_string(value: object) -> str:
    assert isinstance(value, str), f"Expected str, got {type(value).__name__}"
    # Narrowed to str after assert
    return value.strip()


# --- TypeGuard ---
def is_list_of_strings(val: list[object]) -> TypeGuard[list[str]]:
    """Return True only if every element is a string."""
    return all(isinstance(item, str) for item in val)


def join_strings(items: list[object]) -> str:
    if is_list_of_strings(items):
        # type checker knows: items is list[str] here
        return ", ".join(items)
    raise TypeError("All items must be strings")


assert process_value(42) == "integer: 84"
assert process_value("hello") == "string: HELLO"
assert process_value(None) == "null value"
assert join_strings(["a", "b", "c"]) == "a, b, c"
```

---

## Exercise 7: @typing.overload — Multiple Return Types

```python
from typing import overload


@overload
def parse_value(raw: str) -> str: ...
@overload
def parse_value(raw: int) -> int: ...
@overload
def parse_value(raw: bytes) -> str: ...


def parse_value(raw: str | int | bytes) -> str | int:
    """
    Return int unchanged; decode bytes to str; strip str.
    Overloads let callers get the specific return type without casts.
    """
    if isinstance(raw, int):
        return raw
    if isinstance(raw, bytes):
        return raw.decode("utf-8").strip()
    return raw.strip()


# Type checker infers correct return types:
s: str = parse_value("  hello  ")   # -> str
n: int = parse_value(42)            # -> int
b: str = parse_value(b"bytes")      # -> str

assert s == "hello"
assert n == 42
assert b == "bytes"
```

**Key insight:** `@overload` stubs are for the type checker only — they are never called at runtime. The actual implementation (no `@overload` decorator) handles all cases. The stubs must come before the implementation.

---

## Exercise 8: Annotated — Metadata / Validation Hints

```python
from typing import Annotated, get_type_hints, get_args, get_origin
from dataclasses import dataclass


# --- Constraint descriptors (library-agnostic) ---
class Gt:
    """Value must be greater than `value`."""
    def __init__(self, value: float) -> None:
        self.value = value

class MaxLen:
    """String/sequence length must be <= `length`."""
    def __init__(self, length: int) -> None:
        self.length = length

class Pattern:
    """String must match `regex`."""
    def __init__(self, regex: str) -> None:
        self.regex = regex


# --- Model using Annotated ---
@dataclass
class Payment:
    amount:   Annotated[float, Gt(0)]                  # > 0
    currency: Annotated[str, MaxLen(3), Pattern(r"^[A-Z]{3}$")]  # e.g. "USD"
    reference: Annotated[str, MaxLen(64)]


# --- Runtime introspection ---
def validate(obj: object) -> list[str]:
    """
    Toy validator that reads Annotated metadata to check constraints.
    (Pydantic does this automatically; this shows the underlying mechanism.)
    """
    errors: list[str] = []
    hints = get_type_hints(type(obj), include_extras=True)

    for field_name, hint in hints.items():
        if get_origin(hint) is not Annotated:
            continue
        value = getattr(obj, field_name)
        for constraint in get_args(hint)[1:]:          # skip the base type (index 0)
            if isinstance(constraint, Gt) and not (value > constraint.value):
                errors.append(f"{field_name}: must be > {constraint.value}")
            elif isinstance(constraint, MaxLen) and len(value) > constraint.length:
                errors.append(f"{field_name}: length must be <= {constraint.length}")
    return errors


p = Payment(amount=100.0, currency="USD", reference="REF-001")
assert validate(p) == []

bad = Payment(amount=-5.0, currency="US DOLLARS", reference="x" * 100)
errs = validate(bad)
assert any("amount" in e for e in errs)
assert any("currency" in e for e in errs)
print(errs)
```

**Key insight:** `Annotated[T, metadata...]` attaches arbitrary metadata to a type without affecting runtime type checking. Both Pydantic and `beartype` use this mechanism for validation constraints.
