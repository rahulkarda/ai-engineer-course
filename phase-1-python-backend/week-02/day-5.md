# Phase 1 · Week 02 · Day 5 — File I/O, JSON Serialization, and Exception Handling

> Objective: Read and write files safely using context managers; serialize/deserialize JSON; handle and raise exceptions correctly.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### File I/O with context managers
The `open()` built-in returns a file object. Always use `with open(...) as f:` — the context manager guarantees `f.close()` even if an exception is raised. Never hold open file handles without a `with` block.

```python
from pathlib import Path

path = Path("data/transactions.txt")
path.parent.mkdir(parents=True, exist_ok=True)   # create dirs if needed

with open(path, mode="w", encoding="utf-8") as f:
    f.write("line 1\n")
```

Use `pathlib.Path` over `os.path` — it is object-oriented, composes paths with `/`, and is cross-platform.

Common modes: `"r"` (read, default), `"w"` (write/truncate), `"a"` (append), `"rb"`/`"wb"` (binary). Always specify `encoding="utf-8"` for text files.

### JSON
`json.dump(obj, f)` serializes Python objects to a file stream. `json.dumps(obj)` returns a string. `json.load(f)` deserializes from a stream; `json.loads(s)` from a string.

Python ↔ JSON type map:

| Python | JSON |
|--------|------|
| `dict` | object |
| `list`, `tuple` | array |
| `str` | string |
| `int`, `float` | number |
| `True`/`False` | `true`/`false` |
| `None` | `null` |

Non-JSON-serializable types (`datetime`, `Decimal`, custom classes) require a custom encoder or conversion before serialization.

### Exception handling
Python exceptions form a class hierarchy. Catch the most specific exception you can. The `try / except / else / finally` structure:

```python
try:
    result = risky()
except ValueError as e:
    log.error("Bad value: %s", e)
    raise               # re-raise same exception
except (TypeError, KeyError) as e:
    raise RuntimeError("Unexpected structure") from e   # chain
else:
    process(result)     # runs only if no exception
finally:
    cleanup()           # always runs
```

**Key rules**:
- Never `except Exception: pass` — it hides bugs.
- Use `raise ... from e` to preserve the original traceback when wrapping exceptions.
- Create custom exception classes for domain errors (`class TransactionError(ValueError): ...`).
- Log at the level where you handle, not where you raise.

## How it works internally

**File buffering**: `open()` defaults to system-buffered I/O. `write()` may not reach disk immediately. `f.flush()` flushes the buffer; `with` triggers flush+close automatically.

**JSON encoding speed**: `json` (CPython) has a C accelerator (`_json`). For very large payloads, `orjson` is ~5–10× faster and natively handles `datetime`, `UUID`, and numpy types.

**Exception objects** are normal Python objects. When you `raise ValueError("bad")`, Python creates an instance with a `__traceback__` attribute that records the call stack frames. `raise ... from e` chains `__cause__` onto the new exception.

```python
try:
    raise ValueError("inner")
except ValueError as e:
    raise RuntimeError("outer") from e
# RuntimeError.__cause__ is the ValueError
```

## Worked example

```python
# file_json_exceptions.py
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# --- Custom domain exceptions ---
class TransactionError(ValueError):
    """Raised when a transaction record is malformed."""

class FileProcessingError(OSError):
    """Raised when a transaction file cannot be read or parsed."""


# --- Write JSON ---
def save_transactions(transactions: list[dict], path: Path) -> None:
    """Serialize transaction list to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False)
    log.info("Saved %d transactions to %s", len(transactions), path)


# --- Read JSON with error handling ---
def load_transactions(path: Path) -> list[dict]:
    """
    Load transactions from a JSON file.
    Raises FileProcessingError on I/O or parse errors.
    """
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileProcessingError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise FileProcessingError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        raise FileProcessingError(f"Expected list at top level in {path}")
    return data


# --- Validate a single transaction ---
REQUIRED_FIELDS = {"id", "amount", "status"}

def validate_transaction(tx: Any) -> dict:
    """
    Validate structure and types of a transaction dict.
    Raises TransactionError for any violation.
    """
    if not isinstance(tx, dict):
        raise TransactionError(f"Transaction must be a dict, got {type(tx).__name__}")
    missing = REQUIRED_FIELDS - tx.keys()
    if missing:
        raise TransactionError(f"Missing fields: {missing}")
    if not isinstance(tx["amount"], (int, float)):
        raise TransactionError(f"'amount' must be numeric, got {tx['amount']!r}")
    if tx["amount"] < 0:
        raise TransactionError(f"'amount' cannot be negative: {tx['amount']}")
    return tx


# --- Process file: skip invalid, collect errors ---
def process_transaction_file(path: Path) -> dict[str, Any]:
    """Load, validate, and summarize transactions. Invalid rows are skipped."""
    raw = load_transactions(path)   # raises FileProcessingError on I/O failure
    valid: list[dict] = []
    errors: list[str] = []

    for i, item in enumerate(raw):
        try:
            valid.append(validate_transaction(item))
        except TransactionError as e:
            errors.append(f"Row {i}: {e}")
            log.warning("Skipping invalid row %d: %s", i, e)

    return {
        "total": len(raw),
        "valid": len(valid),
        "invalid": len(errors),
        "errors": errors,
        "transactions": valid,
    }


# --- Demo ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    sample_path = Path("/tmp/sample_transactions.json")

    sample_data = [
        {"id": "T1", "amount": 250.0, "status": "success"},
        {"id": "T2", "amount": -10.0, "status": "failed"},   # invalid: negative
        {"id": "T3", "status": "pending"},                    # invalid: missing amount
        {"id": "T4", "amount": 3200.0, "status": "pending"},
    ]

    save_transactions(sample_data, sample_path)
    result = process_transaction_file(sample_path)
    print(f"Valid: {result['valid']}, Invalid: {result['invalid']}")
    for err in result["errors"]:
        print(f"  ERROR: {err}")
```

## Exercises

1. **Append mode**: Write `append_transaction(tx: dict, path: Path) -> None` that appends a single transaction to a JSONL file (one JSON object per line). Handle the case where the file does not yet exist.

2. **Custom encoder**: `datetime` objects are not JSON-serializable. Write `DateTimeEncoder(json.JSONEncoder)` that handles `datetime` by converting it to ISO-8601 string. Test it with a transaction that has a `"created_at": datetime(...)` field.

3. **Exception hierarchy**: Design a three-level exception hierarchy for a transaction processor: a base `ProcessorError`, then `ValidationError` and `StorageError`, then `DuplicateTransactionError(ValidationError)`. Show how `except ProcessorError` catches all three.

4. **Context manager file safety**: Write `safe_write(path: Path, content: str) -> None` that writes to a temporary file first, then renames it into place atomically. Use `try/except/finally` to ensure the temp file is deleted on failure.

5. **Load with schema check**: Write `load_validated_transactions(path: Path) -> list[dict]` that calls `load_transactions` and then `validate_transaction` on each row, collecting all errors in a list. If more than 50% of rows are invalid, raise `FileProcessingError`.

(Solutions: ../solutions/week-02/day-5.md)

## Self-check
- [ ] I can explain what happens if I forget `with` when opening a file
- [ ] I can serialize/deserialize a list of dicts to/from a JSON file
- [ ] I can explain `raise ... from e` and why it matters
- [ ] I can design a domain exception hierarchy
- [ ] I can process a file that has a mix of valid and invalid rows

## Supplementary links
- [pathlib — Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)
- [json module](https://docs.python.org/3/library/json.html)
- [Built-in Exceptions hierarchy](https://docs.python.org/3/library/exceptions.html)
- [orjson — fast JSON library](https://github.com/ijl/orjson)
