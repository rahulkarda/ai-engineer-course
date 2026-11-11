# Week 4 Assessment — Answer Key

## Conceptual Questions (1–10)

---

### 1. What is the difference between an iterable and an iterator? Give an example of each.

An **iterable** is any object that implements `__iter__()`, returning an iterator. It can be iterated over multiple times — each call to `iter()` on it produces a fresh traversal.

An **iterator** is an object that implements both `__iter__()` (returning itself) and `__next__()`, which yields the next value or raises `StopIteration`. Iterators are single-pass: once exhausted, they cannot be reset.

```python
# Iterable — can iterate many times
my_list = [1, 2, 3]               # list is iterable; iter(my_list) → new iterator

# Iterator — single-pass, stateful
it = iter(my_list)
next(it)   # 1
next(it)   # 2
next(it)   # 3
next(it)   # StopIteration
```

A generator object is both an iterable and an iterator simultaneously.

---

### 2. Why are generators memory-efficient? Describe what happens to a generator's state between yields.

Generators produce values **lazily**: one at a time, on demand. Instead of building an entire result list in memory upfront, a generator holds only its local variables and the instruction pointer.

Between `yield` expressions the generator's execution frame is **suspended**. The frame is preserved on the heap (not the call stack) with all local variables intact, the current instruction pointer, and the evaluation stack at the yield point. Execution resumes from exactly that point on the next `next()` call.

This means a generator that conceptually represents a million-element sequence occupies O(1) memory — only the variables alive at the most recent yield.

---

### 3. Explain LEGB scope. What makes a closure? Give a 3-line code example.

**LEGB** is Python's name lookup order:
- **L**ocal — names defined in the current function
- **E**nclosing — names in any enclosing function scopes (closest outward first)
- **G**lobal — module-level names
- **B**uilt-in — Python's built-in namespace (`len`, `print`, …)

A **closure** is formed when an inner function references a variable from an enclosing scope. Python captures the cell object (not the value), so the inner function continues to access it even after the outer function has returned.

```python
def make_adder(n: int):           # n is in the Enclosing scope of `add`
    def add(x: int) -> int:
        return x + n              # n captured from enclosing scope → closure
    return add
```

`add` is a closure because it carries a reference to `n` from `make_adder`'s frame.

---

### 4. What does @functools.wraps do and why is it important?

`@functools.wraps(wrapped)` copies the metadata of the original (wrapped) function onto the decorator's wrapper function: `__name__`, `__qualname__`, `__doc__`, `__annotations__`, `__module__`, and `__wrapped__`.

Without it, the wrapper masquerades as a different function:

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper   # wrapper.__name__ is "wrapper", not the original

@functools.wraps(func)   # with wraps: wrapper.__name__ == func.__name__
def wrapper(*args, **kwargs): ...
```

**Why it matters:**
1. Introspection tools (`help()`, docstring generators, IDEs) show the original function's name and docstring.
2. Stack traces are readable — you see the original function name, not "wrapper".
3. `functools.wraps` also sets `__wrapped__`, allowing tools like `inspect.unwrap()` to peel decorator layers.
4. Type checkers (mypy, pyright) and `functools.lru_cache` rely on correct function metadata.

---

### 5. What is the difference between `__enter__` and `__exit__` in a context manager? What arguments does `__exit__` receive?

`__enter__(self)` is called when execution enters the `with` block. Its return value is bound to the `as` target. It is responsible for acquiring the resource (opening a file, starting a transaction, etc.).

`__exit__(self, exc_type, exc_val, exc_tb)` is called when execution leaves the `with` block — normally or due to an exception.

| Parameter | Type | Description |
|-----------|------|-------------|
| `exc_type` | `type[BaseException] \| None` | Exception class, or `None` if no exception |
| `exc_val` | `BaseException \| None` | Exception instance |
| `exc_tb` | `TracebackType \| None` | Traceback object |

**Return value of `__exit__`:** If it returns a truthy value, the exception is **suppressed** (swallowed). Returning `False` or `None` lets the exception propagate.

```python
class ManagedResource:
    def __enter__(self):
        return self          # bound to `as` target

    def __exit__(self, exc_type, exc_val, exc_tb):
        # cleanup always runs here
        return False         # do not suppress exceptions
```

---

### 6. Explain Python's reference counting. When is an object collected by the cyclic GC instead?

Every Python object has a reference count (`ob_refcnt`). It increments when a new reference is created (assignment, argument passing, container insertion) and decrements when a reference goes out of scope or is explicitly deleted. When the count reaches **zero**, CPython immediately frees the object.

**Limitation:** Reference counting cannot collect **cycles** — groups of objects that reference each other with no external references. Because each object in the cycle keeps the others' count above zero, no count ever hits zero.

The **cyclic garbage collector** (`gc` module) handles this. It runs periodically (or on demand via `gc.collect()`) and uses a mark-and-sweep approach over three generations. It finds unreachable cycles, finalises them safely (respecting `__del__`), and frees the memory.

Rule of thumb: pure data with no cycles is freed by refcounting alone; cyclically-referencing objects need the cyclic GC.

---

### 7. What is the difference between `is` and `==`? When might they give different results for equal objects?

- `==` calls `__eq__()` — tests **value equality** (do these objects represent the same value?).
- `is` tests **identity** — are these the exact same object in memory (same `id()`)?

They can diverge for any object that overrides `__eq__`:

```python
a = [1, 2, 3]
b = [1, 2, 3]
a == b   # True  — same content
a is b   # False — different list objects

# Also: small int cache
x = 257  # outside CPython's [-5..256] cache
y = 257
x == y   # True
x is y   # may be False (implementation-defined)

# NaN special case:
import math
nan = float("nan")
nan == nan   # False  (IEEE 754)
nan is nan   # True   (same object)
```

**Rule:** Use `==` for value comparison; reserve `is` for singleton checks (`is None`, `is True`, `is False`).

---

### 8. What does `__all__` in `__init__.py` control? How does it interact with `from package import *`?

`__all__` is a list of strings declaring the **public API** of a module or package. It affects **only** `from module import *`:

- If `__all__` is defined, `import *` imports exactly those names (and nothing else, even if other names are present).
- If `__all__` is **not** defined, `import *` imports every name that doesn't start with an underscore.

`__all__` does **not** prevent explicit imports (`from module import _private` still works regardless of `__all__`).

Best practice: always define `__all__` in package `__init__.py` to:
1. Document the intended public surface.
2. Prevent accidental export of internal helpers.
3. Allow tools and IDEs to identify the stable API.

```python
# mypackage/__init__.py
from .models import Transaction, Account
from .utils import format_amount

__all__ = ["Transaction", "Account", "format_amount"]
# _private_helper is NOT exported by import *
```

---

### 9. What is the difference between TypeVar with a bound vs TypeVar with constraints?

```python
from typing import TypeVar

# Bound — T must be Animal or a subclass of Animal
T_Bound = TypeVar("T_Bound", bound=Animal)

# Constraints — T must be exactly int OR exactly str (not subclasses of str)
T_Constrained = TypeVar("T_Constrained", int, str)
```

| Feature | `bound=X` | `constraints=(X, Y, ...)` |
|---------|-----------|---------------------------|
| Meaning | T is X or any subclass | T is exactly one of the listed types |
| Subclasses | Accepted | Rejected (constrained to the listed types only) |
| When to use | "must be a subtype of X" | "must be one of these specific types" |
| Example | `T_Bound = TypeVar("T", bound=Comparable)` | `AnyStr = TypeVar("AnyStr", str, bytes)` |

With a **bound**, the type checker preserves the specific subclass (e.g., calling a method on `T_Bound` knows all methods of `Animal` are available). With **constraints**, the checker may only assume the method is available on all listed types.

---

### 10. In Pydantic v2, what is the difference between @field_validator with mode='before' vs mode='after'?

| | `mode='before'` | `mode='after'` |
|---|---|---|
| When it runs | Before Pydantic's own type coercion/parsing | After Pydantic has parsed and coerced the value |
| Input type | Raw input (whatever the caller passed in — could be `str`, `dict`, `None`, etc.) | Already the declared Python type (e.g., `Decimal`, `datetime`) |
| Use case | Normalise/clean raw input, handle alternate representations | Validate semantic constraints on the typed value |
| Return type | Must return a value Pydantic can then coerce to the field type | Must return the exact field type (or raise) |

```python
from pydantic import BaseModel, field_validator
from decimal import Decimal

class Payment(BaseModel):
    amount: Decimal
    currency: str

    @field_validator("currency", mode="before")
    @classmethod
    def normalise_currency(cls, v: object) -> str:
        # Runs on raw input — v might be any type
        return str(v).strip().upper()   # returns str for Pydantic to validate further

    @field_validator("amount", mode="after")
    @classmethod
    def must_be_positive(cls, v: Decimal) -> Decimal:
        # Pydantic has already converted the raw input to Decimal
        if v <= 0:
            raise ValueError("amount must be positive")
        return v
```

---

## Coding Challenge — TransactionPipeline: Full Implementation

```python
"""
Week 4 Assessment — TransactionPipeline
Full implementation covering: Pydantic v2, generator pipeline,
decorated functions, JSON export.
"""
from __future__ import annotations

import functools
import json
import logging
import time
from collections.abc import Generator, Callable
from decimal import Decimal
from typing import Literal, TypeVar, ParamSpec

from pydantic import BaseModel, field_validator, model_validator, Field, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("transaction_pipeline")

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Log wall-clock time for each call."""
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        logger.info("%s completed in %.4f s", func.__qualname__, time.perf_counter() - t0)
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 0.0) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry a function up to max_attempts times on exception."""
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(
                        "%s attempt %d/%d failed: %s",
                        func.__qualname__, attempt, max_attempts, exc,
                    )
                    if delay and attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(f"{func.__qualname__} failed after {max_attempts} attempts") from last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    id: str = Field(..., min_length=1)
    amount: Decimal
    currency: str
    status: Literal["pending", "completed", "failed"] = "pending"
    merchant: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v: object) -> Decimal:
        return Decimal(str(v))

    @field_validator("amount", mode="after")
    @classmethod
    def amount_positive(cls, v: Decimal) -> Decimal:
        if v <= Decimal("0"):
            raise ValueError(f"amount must be positive, got {v}")
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def normalise_currency(cls, v: object) -> str:
        return str(v).strip().upper()

    @field_validator("currency", mode="after")
    @classmethod
    def valid_currency_code(cls, v: str) -> str:
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"currency must be 3 alphabetic chars, got {v!r}")
        return v

    @model_validator(mode="after")
    def merchant_required_when_completed(self) -> Transaction:
        if self.status == "completed" and not self.merchant:
            raise ValueError("merchant must be non-empty for completed transactions")
        return self


class PipelineResult(BaseModel):
    processed: list[Transaction]
    rejected_count: int
    total_amount: Decimal
    currencies: list[str]

    @property
    def success_rate(self) -> float:
        total = len(self.processed) + self.rejected_count
        return len(self.processed) / total if total else 0.0


# ---------------------------------------------------------------------------
# Generator pipeline
# ---------------------------------------------------------------------------

def raw_transaction_source(
    records: list[dict],
) -> Generator[dict, None, None]:
    """Yield raw dicts one at a time (lazy source)."""
    for record in records:
        yield record


def validate_transactions(
    source: Generator[dict, None, None],
) -> Generator[Transaction, None, None]:
    """Parse and validate each raw dict; skip invalid records."""
    from pydantic import ValidationError

    for raw in source:
        try:
            yield Transaction.model_validate(raw)
        except ValidationError as exc:
            logger.warning("Skipping invalid record %r: %s", raw.get("id"), exc)


def filter_by_status(
    source: Generator[Transaction, None, None],
    status: str,
) -> Generator[Transaction, None, None]:
    """Yield only transactions matching the given status."""
    for txn in source:
        if txn.status == status:
            yield txn


def enrich_with_tags(
    source: Generator[Transaction, None, None],
    tag: str,
) -> Generator[Transaction, None, None]:
    """Add a tag to each transaction (immutable model → copy)."""
    for txn in source:
        enriched = txn.model_copy(update={"tags": [*txn.tags, tag]})
        yield enriched


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class TransactionPipeline:
    """
    Orchestrates a generator pipeline over raw transaction records.

    Usage:
        pipeline = TransactionPipeline(raw_records)
        result = pipeline.run(status_filter="completed")
        print(result.model_dump_json(indent=2))
    """

    def __init__(self, raw_records: list[dict]) -> None:
        self._raw = raw_records

    @timed
    def run(self, status_filter: str = "completed") -> PipelineResult:
        """
        Run the full pipeline:
          source → validate → filter by status → enrich → collect results.
        """
        from pydantic import ValidationError

        rejected = 0
        all_raw = list(raw_transaction_source(self._raw))

        # Count rejections by running a separate validation pass
        processed: list[Transaction] = []
        for raw in all_raw:
            try:
                txn = Transaction.model_validate(raw)
                if txn.status == status_filter:
                    enriched = txn.model_copy(update={"tags": [*txn.tags, "processed"]})
                    processed.append(enriched)
            except ValidationError:
                rejected += 1

        total_amount = sum((t.amount for t in processed), Decimal("0"))
        currencies = sorted({t.currency for t in processed})

        return PipelineResult(
            processed=processed,
            rejected_count=rejected,
            total_amount=total_amount,
            currencies=currencies,
        )

    @timed
    def export_json(self, result: PipelineResult, path: str) -> None:
        """Serialise PipelineResult to a JSON file."""
        import pathlib

        data = result.model_dump(mode="json")
        data["success_rate"] = result.success_rate

        pathlib.Path(path).write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Exported %d transactions to %s", len(result.processed), path)

    @staticmethod
    @retry(max_attempts=3)
    def fetch_exchange_rate(currency: str) -> Decimal:
        """
        Simulate fetching an exchange rate (demonstrates @retry decorator).
        In production this would call an external API.
        """
        import random
        if random.random() < 0.4:          # noqa: S311
            raise ConnectionError("simulated network error")
        rates = {"USD": Decimal("1.0"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27")}
        return rates.get(currency, Decimal("1.0"))


# ---------------------------------------------------------------------------
# Demo / manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_records: list[dict] = [
        {"id": "t1", "amount": "100.00", "currency": "usd", "status": "completed", "merchant": "Acme"},
        {"id": "t2", "amount": "50.50",  "currency": "EUR", "status": "completed", "merchant": "Beta"},
        {"id": "t3", "amount": "200.00", "currency": "GBP", "status": "pending",   "merchant": "Gamma"},
        {"id": "t4", "amount": "-9.99",  "currency": "USD", "status": "completed", "merchant": "Bad"},   # invalid
        {"id": "t5", "amount": "75.00",  "currency": "USD", "status": "completed", "merchant": "Delta"},
    ]

    pipeline = TransactionPipeline(sample_records)
    result = pipeline.run(status_filter="completed")

    print(f"Processed: {len(result.processed)}")
    print(f"Rejected:  {result.rejected_count}")
    print(f"Total:     {result.total_amount}")
    print(f"Currencies: {result.currencies}")
    print(f"Success rate: {result.success_rate:.0%}")
    print()
    print(result.model_dump_json(indent=2))
```

---

## Grading Notes

### Conceptual Questions (40 points — 4 pts each)

| Q | Key Discriminators | Full Marks | Partial |
|---|---|---|---|
| 1 | Names both `__iter__` and `__next__`; notes single-pass iterator | Correct examples | Missing `__next__` or "single-pass" |
| 2 | "Suspended frame" / "lazy evaluation"; O(1) memory | Mentions lazy + frame preservation | Only says "lazy" with no mechanism |
| 3 | All four LEGB letters with correct definitions; cell object for closure | Correct minimal example | Missing enclosing scope explanation |
| 4 | Lists `__name__`, `__doc__`, `__wrapped__`; mentions stack traces and type checkers | Any 3 metadata items + at least 1 concrete reason | Only says "copies metadata" without why |
| 5 | `__exit__` args: exc_type/exc_val/exc_tb; truthy return suppresses | Correct table or prose | Missing suppression semantics |
| 6 | "Refcount hits zero → freed immediately"; cycles require GC; mentions generations | Both mechanisms explained | Only one mechanism |
| 7 | `is` = identity, `==` = value; NaN or large int example | Two different examples | Only one example |
| 8 | "`import *` only"; does not prevent explicit imports | Notes `__all__` not defined → all public names | Says `__all__` blocks explicit imports (wrong) |
| 9 | bound = subclasses OK; constraints = exact types only | Concrete example for each | Swaps bound/constraints semantics |
| 10 | before = raw input; after = typed value; correct use cases | Correct type descriptions | Reverses order |

### Coding Challenge (60 points)

| Dimension | Points | Criteria |
|---|---|---|
| Pydantic model correctness | 15 | `field_validator` with correct mode, `model_validator`, `frozen=True` or equivalent, `ConfigDict` |
| Generator pipeline | 15 | At least 2 chained generators, lazy evaluation (no `list()` until collect step), proper `yield` usage |
| Decorators | 10 | `@timed` uses `ParamSpec` + `functools.wraps`; `@retry` is parametrised; both preserve signatures |
| JSON export | 10 | `model_dump_json()` or `model_dump(mode="json")` used; writes to file with `pathlib` |
| Code quality | 10 | Type hints throughout, logging not print, separation of concerns (model / pipeline / orchestrator distinct) |

**Deductions:**
- `-5` for any hardcoded secret or credential
- `-5` for `print()` used instead of `logging`
- `-3` for missing type hints on any function signature
- `-3` for mutable default arguments (e.g., `def f(x=[])`)
- `-2` per use of bare `except:` without specifying exception type
