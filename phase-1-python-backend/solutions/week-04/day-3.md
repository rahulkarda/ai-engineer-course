# Week 4 Day 3 — Solutions: Context Managers & Copying

---

## Exercise 1: `TempDirectory` Context Manager Class

Implement `__enter__` and `__exit__`. The three-argument signature of `__exit__` receives exception info; returning `False` (or `None`) re-raises any exception — the cleanup still runs either way because we put `shutil.rmtree` in the `finally`-equivalent logic of `__exit__`.

```python
import logging
import shutil
import tempfile
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)


class TempDirectory:
    """Context manager that creates a temp directory and removes it on exit."""

    def __init__(self, prefix: str = "tmp_") -> None:
        self._prefix = prefix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        self._path = Path(tempfile.mkdtemp(prefix=self._prefix))
        logger.debug("TempDirectory created: %s", self._path)
        return self._path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._path is not None and self._path.exists():
            shutil.rmtree(self._path)
            logger.debug("TempDirectory removed: %s", self._path)
        return False  # do not suppress exceptions


# Usage
with TempDirectory(prefix="week4_") as tmp:
    (tmp / "data.txt").write_text("hello")
    print(list(tmp.iterdir()))  # [PosixPath('/tmp/week4_xxxx/data.txt')]

# tmp no longer exists after the block
print(tmp.exists())  # False
```

**`return False` in `__exit__`:** Returning a truthy value would *suppress* the exception. Almost always wrong — only suppress when you deliberately handle an error inside the context manager.

---

## Exercise 2: `managed_connection` with `@contextlib.contextmanager`

The `@contextlib.contextmanager` decorator turns a generator function with exactly one `yield` into a context manager. Code before `yield` is the `__enter__` body; code after (in the `finally` block) is the `__exit__` body.

```python
import logging
from contextlib import contextmanager
from collections.abc import Generator

logger = logging.getLogger(__name__)


class _FakeConnection:
    """Placeholder for a real connection object."""
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def close(self) -> None:
        pass  # real teardown would go here


@contextmanager
def managed_connection(host: str, port: int) -> Generator[_FakeConnection, None, None]:
    """Open a connection, yield it, and ensure it is closed on exit."""
    logger.info("Connecting to %s:%d", host, port)
    conn = _FakeConnection(host, port)
    try:
        yield conn
    except Exception:
        logger.error("Exception while connected to %s:%d", host, port, exc_info=True)
        raise
    finally:
        conn.close()
        logger.info("Disconnected from %s:%d", host, port)


# Usage
logging.basicConfig(level=logging.INFO)

with managed_connection("db.example.com", 5432) as conn:
    print(f"Using connection to {conn.host}:{conn.port}")
# INFO  Connecting to db.example.com:5432
# INFO  Disconnected from db.example.com:5432
```

**`finally` vs bare `except`:** Putting cleanup in `finally` ensures it runs whether or not an exception occurred. The `except` block is only needed when you want to log or transform the exception before re-raising.

---

## Exercise 3: Shallow vs Deep Copy — Concrete Demonstration

```python
import copy


original = [[1, 2, 3], [4, 5, 6]]

shallow = copy.copy(original)
deep    = copy.deepcopy(original)

# ── Shallow copy ──────────────────────────────────────────────────────────────
# The outer list is a new object...
assert shallow is not original          # different list object

# ...but the inner lists are SHARED references.
assert shallow[0] is original[0]       # same inner list!

original[0].append(99)
print(original)  # [[1, 2, 3, 99], [4, 5, 6]]
print(shallow)   # [[1, 2, 3, 99], [4, 5, 6]]  ← mutation visible in shallow copy!

# ── Deep copy ─────────────────────────────────────────────────────────────────
assert deep is not original
assert deep[0] is not original[0]      # independent inner list

original[1].append(88)
print(original)  # [[1, 2, 3, 99], [4, 5, 6, 88]]
print(deep)      # [[1, 2, 3, 99], [4, 5, 6]]     ← deep copy unaffected
```

**Summary:**

| | Outer container | Inner mutable objects |
|---|---|---|
| `copy.copy` | New object | Shared references |
| `copy.deepcopy` | New object | Fully independent copies |

---

## Exercise 4: `transaction_context` Context Manager

Accumulate operations in a list during the block. On clean exit, "commit" (keep the list as-is and signal success). On exception, "roll back" by clearing the list.

```python
import logging
from contextlib import contextmanager
from collections.abc import Generator
from types import TracebackType

logger = logging.getLogger(__name__)


class TransactionContext:
    """Collect operations; commit on success, rollback on exception."""

    def __init__(self) -> None:
        self.operations: list[str] = []

    def add(self, operation: str) -> None:
        self.operations.append(operation)
        logger.debug("Staged: %s", operation)

    def __enter__(self) -> "TransactionContext":
        logger.info("Transaction started")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            logger.info("Transaction committed: %s", self.operations)
        else:
            logger.warning(
                "Transaction rolled back due to %s: %s", exc_type.__name__, exc_val
            )
            self.operations.clear()
        return False  # never suppress exceptions


# Usage — successful transaction
logging.basicConfig(level=logging.INFO)

with TransactionContext() as tx:
    tx.add("INSERT users (alice)")
    tx.add("UPDATE balance SET amount=100")
print(tx.operations)  # ['INSERT users (alice)', 'UPDATE balance SET amount=100']

# Usage — failed transaction
with TransactionContext() as tx:
    tx.add("INSERT users (bob)")
    try:
        raise ValueError("constraint violation")
    except ValueError:
        pass  # caught inside block — context manager does NOT see this
print(tx.operations)  # ['INSERT users (bob)']  ← not rolled back because exception was caught inside

# To trigger rollback, let the exception propagate OUT of the with block:
try:
    with TransactionContext() as tx2:
        tx2.add("DELETE users")
        raise RuntimeError("disk full")
except RuntimeError:
    pass
print(tx2.operations)  # []  ← rolled back
```

---

## Exercise 5: Shallow Copy and Mutable Attribute Behaviour

```python
import copy
from dataclasses import dataclass, field


@dataclass
class Config:
    name: str
    tags: list[str] = field(default_factory=list)


obj1 = Config(name="original", tags=["a", "b"])
obj2 = copy.copy(obj1)

print(obj1 is obj2)         # False — different Config instances
print(obj1.tags is obj2.tags)  # True  — SAME list object!

# Appending to obj1's list is visible through obj2
obj1.tags.append("c")
print(obj1.tags)  # ['a', 'b', 'c']
print(obj2.tags)  # ['a', 'b', 'c']  ← surprising if you expected isolation

# Reassigning obj1.tags does NOT affect obj2
obj1.tags = ["x", "y"]
print(obj1.tags)  # ['x', 'y']
print(obj2.tags)  # ['a', 'b', 'c']  ← obj2 still holds the old shared list
```

**Explanation:**

`copy.copy` creates a new `Config` object and copies the attribute references. `obj2.tags` is the *same list* as `obj1.tags` — a mutation (`.append`) through either reference modifies the shared list, so the change is visible from both sides.

Reassignment (`obj1.tags = [...]`) only rebinds the name `obj1.tags` to a new list — it does not mutate the shared object, so `obj2.tags` still points to the original list.

**Fix:** Use `copy.deepcopy` if you need fully independent instances:

```python
obj3 = copy.deepcopy(obj1)
obj1.tags.append("z")
print(obj3.tags)  # ['x', 'y']  ← unaffected
```
