# Solutions — Week 02 · Day 5

## Exercise 1 — Append mode (JSONL)

```python
import json
from pathlib import Path

def append_transaction(tx: dict, path: Path) -> None:
    """Append one JSON object as a line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(tx, ensure_ascii=False) + "\n")

# Reading back
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, mode="r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

from pathlib import Path
import tempfile, os

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "txns.jsonl"
    append_transaction({"id": "T1", "amount": 10.0}, p)
    append_transaction({"id": "T2", "amount": 20.0}, p)
    records = load_jsonl(p)
    assert len(records) == 2
    assert records[0]["id"] == "T1"
```

## Exercise 2 — Custom JSON encoder for datetime

```python
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)   # raises TypeError for unknown types

tx = {
    "id": "T1",
    "amount": 250.0,
    "created_at": datetime(2024, 3, 15, 10, 30, 0),
}

serialized = json.dumps(tx, cls=DateTimeEncoder)
parsed = json.loads(serialized)
assert parsed["created_at"] == "2024-03-15T10:30:00"

# Round-trip: string back to datetime
from datetime import datetime as dt
back = dt.fromisoformat(parsed["created_at"])
assert back == datetime(2024, 3, 15, 10, 30, 0)
```

## Exercise 3 — Exception hierarchy

```python
class ProcessorError(Exception):
    """Base for all processor errors."""

class ValidationError(ProcessorError):
    """Input data is structurally or semantically invalid."""

class StorageError(ProcessorError):
    """I/O or persistence failure."""

class DuplicateTransactionError(ValidationError):
    """A transaction ID was seen more than once."""


# Demonstration: except ProcessorError catches all subtypes
def process_all():
    try:
        raise DuplicateTransactionError("T1 appears twice")
    except ProcessorError as e:
        print(f"Caught as ProcessorError: {e}")   # ← catches it
        assert isinstance(e, ValidationError)
        assert isinstance(e, DuplicateTransactionError)

process_all()

# except ValidationError also catches DuplicateTransactionError
try:
    raise DuplicateTransactionError("T2 again")
except ValidationError as e:
    print(f"Caught as ValidationError: {e}")

# except StorageError does NOT catch DuplicateTransactionError
try:
    raise DuplicateTransactionError("T3")
except StorageError:
    assert False, "should not reach here"
except ValidationError:
    pass   # correct
```

## Exercise 4 — Atomic write with temp file

```python
import os
import tempfile
from pathlib import Path

def safe_write(path: Path, content: str) -> None:
    """Write content atomically: write to temp, rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        # Create temp file in same directory as target (same filesystem for rename)
        fd, tmp_str = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        tmp_path = Path(tmp_str)
        with os.fdopen(fd, mode="w", encoding="utf-8") as f:
            f.write(content)
        tmp_path.rename(path)   # atomic on POSIX (same filesystem)
        tmp_path = None          # rename succeeded — nothing to clean up
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise

import tempfile as _tmp
with _tmp.TemporaryDirectory() as d:
    p = Path(d) / "output.txt"
    safe_write(p, "hello world\n")
    assert p.read_text() == "hello world\n"
```

Note: `os.rename` (and `Path.rename`) is atomic on POSIX when source and destination are on the same filesystem. On Windows, use `pathlib.Path.replace` (Python 3.3+).

## Exercise 5 — load_validated_transactions

```python
import json
from pathlib import Path

class FileProcessingError(OSError):
    pass

class TransactionError(ValueError):
    pass

REQUIRED = {"id", "amount", "status"}

def validate_transaction(tx: object) -> dict:
    if not isinstance(tx, dict):
        raise TransactionError(f"Not a dict: {type(tx).__name__}")
    missing = REQUIRED - tx.keys()
    if missing:
        raise TransactionError(f"Missing: {missing}")
    if not isinstance(tx["amount"], (int, float)) or tx["amount"] <= 0:
        raise TransactionError(f"Invalid amount: {tx['amount']}")
    return tx

def load_transactions(path: Path) -> list[object]:
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileProcessingError(f"Not found: {path}") from e
    except json.JSONDecodeError as e:
        raise FileProcessingError(f"Bad JSON: {e}") from e

def load_validated_transactions(path: Path) -> list[dict]:
    raw = load_transactions(path)
    valid, errors = [], []
    for i, item in enumerate(raw):
        try:
            valid.append(validate_transaction(item))
        except TransactionError as e:
            errors.append(f"Row {i}: {e}")

    total = len(raw)
    if total > 0 and len(errors) / total > 0.5:
        raise FileProcessingError(
            f"More than 50% of rows invalid ({len(errors)}/{total})"
        )
    return valid
```
