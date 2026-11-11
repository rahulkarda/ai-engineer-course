# Solutions — Week 02 Project: Financial Transaction Processor

Complete reference implementation. Study only after your own attempt.

---

## pyproject.toml

```toml
[project]
name = "transaction-processor"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"
```

---

## src/transaction_processor/models.py

```python
from __future__ import annotations

Transaction = dict[str, object]

VALID_STATUSES: frozenset[str] = frozenset({"success", "failed", "pending", "refunded"})
VALID_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "GBP"})


class TransactionError(ValueError):
    """A transaction record is structurally or semantically invalid."""


class StorageError(OSError):
    """A file I/O or serialization failure."""
```

---

## src/transaction_processor/validator.py

```python
from __future__ import annotations
from typing import Any
from .models import Transaction, VALID_STATUSES, TransactionError


def validate_transaction(tx: Any) -> Transaction:
    if not isinstance(tx, dict):
        raise TransactionError(f"Expected dict, got {type(tx).__name__}")
    for field in ("id", "amount", "status"):
        if field not in tx:
            raise TransactionError(f"Missing required field: '{field}'")
    if not isinstance(tx["id"], str) or not tx["id"].strip():
        raise TransactionError("'id' must be a non-empty string")
    if not isinstance(tx["amount"], (int, float)) or tx["amount"] <= 0:
        raise TransactionError(f"'amount' must be positive, got {tx['amount']!r}")
    if tx["status"] not in VALID_STATUSES:
        raise TransactionError(
            f"'status' must be one of {sorted(VALID_STATUSES)}, got {tx['status']!r}"
        )
    return tx


def validate_all(
    raw: list[object],
) -> tuple[list[Transaction], list[tuple[int, str]]]:
    valid: list[Transaction] = []
    errors: list[tuple[int, str]] = []
    for i, item in enumerate(raw):
        try:
            valid.append(validate_transaction(item))
        except TransactionError as e:
            errors.append((i, str(e)))
    return valid, errors
```

---

## src/transaction_processor/aggregator.py

```python
from __future__ import annotations
from collections import defaultdict
from .models import Transaction


def total_successful(transactions: list[Transaction]) -> float:
    return round(
        sum(tx["amount"] for tx in transactions if tx["status"] == "success"),  # type: ignore[operator]
        2,
    )


def find_failed(transactions: list[Transaction]) -> list[Transaction]:
    return sorted(
        (tx for tx in transactions if tx["status"] == "failed"),
        key=lambda tx: tx["amount"],  # type: ignore[return-value]
        reverse=True,
    )


def group_by_status(transactions: list[Transaction]) -> dict[str, list[Transaction]]:
    groups: dict[str, list[Transaction]] = defaultdict(list)
    for tx in transactions:
        groups[str(tx["status"])].append(tx)
    return dict(groups)


def count_by_status(transactions: list[Transaction]) -> dict[str, int]:
    return {status: len(txns) for status, txns in group_by_status(transactions).items()}


def currency_totals(transactions: list[Transaction]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for tx in transactions:
        totals[str(tx["currency"])] += float(tx["amount"])  # type: ignore[arg-type]
    return {k: round(v, 2) for k, v in totals.items()}
```

---

## src/transaction_processor/deduplicator.py

```python
from __future__ import annotations
from collections import defaultdict
from .models import Transaction


def find_duplicate_ids(transactions: list[Transaction]) -> dict[str, list[Transaction]]:
    groups: dict[str, list[Transaction]] = defaultdict(list)
    for tx in transactions:
        groups[str(tx["id"])].append(tx)
    return {k: v for k, v in groups.items() if len(v) > 1}


def remove_duplicates(transactions: list[Transaction]) -> list[Transaction]:
    # Sort descending by amount so highest is first; stable sort preserves first-seen for ties.
    sorted_txns = sorted(
        transactions,
        key=lambda tx: tx["amount"],  # type: ignore[return-value]
        reverse=True,
    )
    seen: set[str] = set()
    result: list[Transaction] = []
    for tx in sorted_txns:
        tid = str(tx["id"])
        if tid not in seen:
            seen.add(tid)
            result.append(tx)
    return result
```

---

## src/transaction_processor/sorter.py

```python
from __future__ import annotations
from .models import Transaction
from .aggregator import group_by_status


def sort_by_amount(
    transactions: list[Transaction],
    *,
    descending: bool = True,
) -> list[Transaction]:
    return sorted(
        transactions,
        key=lambda tx: tx["amount"],  # type: ignore[return-value]
        reverse=descending,
    )


def sort_by_status_then_amount(
    transactions: list[Transaction],
    status_order: list[str] | None = None,
) -> list[Transaction]:
    if status_order is None:
        status_order = sorted({str(tx["status"]) for tx in transactions})
    rank = {s: i for i, s in enumerate(status_order)}
    return sorted(
        transactions,
        key=lambda tx: (rank.get(str(tx["status"]), 999), -float(tx["amount"])),  # type: ignore[arg-type]
    )


def top_n(transactions: list[Transaction], n: int) -> list[Transaction]:
    return sort_by_amount(transactions, descending=True)[:n]
```

---

## src/transaction_processor/io.py

```python
from __future__ import annotations
import json
import logging
from pathlib import Path
from .models import Transaction, StorageError

log = logging.getLogger(__name__)


def load_transactions(path: Path) -> list[object]:
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise StorageError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise StorageError(f"Invalid JSON in {path}: {e}") from e


def save_transactions(transactions: list[Transaction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(transactions, f, indent=2, ensure_ascii=False)
    except OSError as e:
        raise StorageError(f"Could not write to {path}: {e}") from e
    log.info("Saved %d transactions → %s", len(transactions), path)
```

---

## src/transaction_processor/pipeline.py

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aggregator import count_by_status, currency_totals, total_successful
from .deduplicator import find_duplicate_ids, remove_duplicates
from .io import load_transactions, save_transactions
from .models import Transaction
from .sorter import sort_by_amount
from .validator import validate_all

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    input_count: int
    invalid_rows: list[tuple[int, str]]
    duplicate_ids: list[str]
    output_count: int
    total_successful: float
    currency_totals: dict[str, float]
    status_breakdown: dict[str, int]
    output_path: Path


def run(input_path: Path, output_path: Path) -> PipelineResult:
    raw = load_transactions(input_path)   # raises StorageError on failure
    valid, errors = validate_all(raw)

    log.info(
        "Loaded %d rows: %d valid, %d invalid", len(raw), len(valid), len(errors)
    )
    for idx, msg in errors:
        log.warning("Skipped row %d: %s", idx, msg)

    dupes = find_duplicate_ids(valid)
    clean = remove_duplicates(valid)
    clean_sorted = sort_by_amount(clean, descending=True)

    save_transactions(clean_sorted, output_path)

    return PipelineResult(
        input_count=len(raw),
        invalid_rows=errors,
        duplicate_ids=sorted(dupes.keys()),
        output_count=len(clean_sorted),
        total_successful=total_successful(clean_sorted),
        currency_totals=currency_totals(clean_sorted),
        status_breakdown=count_by_status(clean_sorted),
        output_path=output_path,
    )
```

---

## tests/conftest.py

```python
import pytest

@pytest.fixture
def sample_transactions() -> list[dict]:
    return [
        {"id": "T1", "amount": 250.0, "status": "success", "category": "retail", "currency": "USD"},
        {"id": "T2", "amount": 1_000.0, "status": "failed",  "category": "wholesale", "currency": "EUR"},
        {"id": "T1", "amount": 500.0,   "status": "success", "category": "retail", "currency": "USD"},  # dupe
        {"id": "T3", "amount": 75.5,    "status": "pending", "category": "internal", "currency": "GBP"},
        {"id": "T4", "amount": 3_200.0, "status": "success", "category": "wholesale", "currency": "USD"},
    ]
```

---

## tests/test_pipeline.py (integration)

```python
import json
import pytest
from pathlib import Path
from src.transaction_processor.pipeline import run
from src.transaction_processor.io import save_transactions

def test_pipeline_happy_path(tmp_path, sample_transactions):
    inp = tmp_path / "input.json"
    out = tmp_path / "output.json"
    save_transactions(sample_transactions, inp)

    result = run(inp, out)

    assert result.output_count <= result.input_count
    assert result.output_count == 4   # T1 duplicate resolved → 4 unique IDs

    output_data = json.loads(out.read_text())
    ids = [tx["id"] for tx in output_data]
    assert len(ids) == len(set(ids))   # no duplicates in output

    # Amounts should be descending
    amounts = [tx["amount"] for tx in output_data]
    assert amounts == sorted(amounts, reverse=True)

    # T1 should have the higher-amount record (500 > 250)
    t1 = next(tx for tx in output_data if tx["id"] == "T1")
    assert t1["amount"] == 500.0

def test_pipeline_all_invalid(tmp_path):
    inp = tmp_path / "bad.json"
    out = tmp_path / "output.json"
    inp.write_text(json.dumps([{"broken": True}, 42, None]))

    result = run(inp, out)

    assert result.output_count == 0
    assert len(result.invalid_rows) == 3
    assert json.loads(out.read_text()) == []
```
