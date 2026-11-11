# Week 02 Project — Financial Transaction Processor

> Estimated time: 3 hours  ·  Language: Python 3.12+  ·  No third-party libraries except `pytest`

---

## Goal

Build a self-contained **Financial Transaction Processor** module that:

1. Reads transaction records from a JSON file
2. Validates each record and reports errors
3. Computes aggregates (totals, groupings, summaries)
4. Detects and resolves duplicate transaction IDs
5. Sorts transactions by multiple criteria
6. Writes clean output to a new JSON file

This is the first time you build a real processing pipeline from scratch. The code you write here will be refactored with OOP in Week 3.

---

## Project structure

```
transaction_processor/
├── src/
│   └── transaction_processor/
│       ├── __init__.py
│       ├── models.py          ← type aliases and constants
│       ├── validator.py       ← validation logic
│       ├── aggregator.py      ← aggregation functions
│       ├── deduplicator.py    ← duplicate detection/removal
│       ├── sorter.py          ← sorting functions
│       ├── io.py              ← file read/write
│       └── pipeline.py        ← top-level pipeline
├── tests/
│   ├── conftest.py            ← shared fixtures
│   ├── test_validator.py
│   ├── test_aggregator.py
│   ├── test_deduplicator.py
│   ├── test_sorter.py
│   ├── test_io.py
│   └── test_pipeline.py       ← integration test
├── data/
│   └── sample_transactions.json
└── pyproject.toml
```

---

## Module specifications

### `models.py`

Define:
- `Transaction = dict[str, object]` type alias
- `VALID_STATUSES: frozenset[str] = frozenset({"success", "failed", "pending", "refunded"})`
- `VALID_CURRENCIES: frozenset[str] = frozenset({"USD", "EUR", "GBP"})`
- Custom exception `TransactionError(ValueError)`
- Custom exception `StorageError(OSError)`

### `validator.py`

```python
def validate_transaction(tx: object) -> dict:
    """
    Validate a single transaction. Returns the dict unchanged if valid.
    Raises TransactionError with a descriptive message if:
      - tx is not a dict
      - any of id, amount, status is missing
      - id is not a non-empty string
      - amount is not a positive number
      - status is not in VALID_STATUSES
    """

def validate_all(
    raw: list[object],
) -> tuple[list[dict], list[tuple[int, str]]]:
    """
    Validate a list of raw objects.
    Returns (valid_list, [(row_index, error_message), ...]).
    Never raises — invalid rows are collected, not propagated.
    """
```

### `aggregator.py`

```python
def total_successful(transactions: list[dict]) -> float:
    """Sum of amount for status == 'success'. Returns 0.0 for empty input."""

def find_failed(transactions: list[dict]) -> list[dict]:
    """All failed transactions, sorted by amount descending."""

def group_by_status(transactions: list[dict]) -> dict[str, list[dict]]:
    """Map each status to its transactions. Keys present only if transactions exist."""

def count_by_status(transactions: list[dict]) -> dict[str, int]:
    """Count transactions per status."""

def currency_totals(transactions: list[dict]) -> dict[str, float]:
    """Sum of amounts per currency, rounded to 2 dp."""
```

### `deduplicator.py`

```python
def find_duplicate_ids(transactions: list[dict]) -> dict[str, list[dict]]:
    """
    Return {id: [all_transactions_with_that_id]} for IDs appearing more than once.
    """

def remove_duplicates(transactions: list[dict]) -> list[dict]:
    """
    Return a deduplicated list keeping the highest-amount record per ID.
    On tie, keep the record that appears first (sort-stable guarantee).
    """
```

### `sorter.py`

```python
def sort_by_amount(
    transactions: list[dict],
    *,
    descending: bool = True,
) -> list[dict]:
    """Return a new list sorted by amount. Does not mutate input."""

def sort_by_status_then_amount(
    transactions: list[dict],
    status_order: list[str] | None = None,
) -> list[dict]:
    """
    Sort by status (using status_order priority, defaulting to alphabetical),
    then by amount descending within each status group.
    """

def top_n(transactions: list[dict], n: int) -> list[dict]:
    """Return the n highest-amount transactions."""
```

### `io.py`

```python
def load_transactions(path: Path) -> list[object]:
    """
    Load raw JSON list from path.
    Raises StorageError (wrapping original) on FileNotFoundError or JSONDecodeError.
    """

def save_transactions(transactions: list[dict], path: Path) -> None:
    """
    Write transactions to path as a JSON array with indent=2.
    Creates parent directories if needed.
    Raises StorageError on write failure.
    """
```

### `pipeline.py`

```python
from dataclasses import dataclass, field
from pathlib import Path

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
    """
    Full pipeline:
    1. Load raw JSON (raises StorageError on failure)
    2. Validate all rows; collect invalid
    3. Remove duplicates (keep highest-amount per ID)
    4. Sort by amount descending
    5. Save clean data to output_path
    6. Return PipelineResult
    """
```

---

## Sample data

Create `data/sample_transactions.json` with at least 20 records including:
- At least 2 duplicate IDs
- At least 2 invalid records (one missing a field, one with a negative amount)
- At least one transaction per status
- At least two currencies

---

## Tests

Minimum test coverage:

| Module | Minimum tests |
|--------|--------------|
| `validator` | 6 (happy path, each error type) |
| `aggregator` | 5 (one per function, empty input) |
| `deduplicator` | 4 (no dupes, with dupes, tie-breaking, empty) |
| `sorter` | 4 (asc, desc, multi-key, top-n) |
| `io` | 4 (round-trip, file-not-found, bad JSON, write-then-read) |
| `pipeline` | 2 (end-to-end happy path, all-invalid file) |

Total minimum: **25 tests**. All must pass.

---

## Acceptance criteria

- [ ] `python -m transaction_processor data/sample_transactions.json` (or equivalent) runs without error
- [ ] Output JSON is valid and contains no duplicate IDs
- [ ] `pytest tests/` exits 0 with ≥ 25 passing tests
- [ ] No `print()` — use `logging`
- [ ] Every function has a type annotation and a one-line docstring
- [ ] No single file exceeds 80 lines of non-blank, non-comment code
- [ ] `.env` is not needed (no secrets)

---

## Submission checklist

- [ ] All six source modules complete
- [ ] `pyproject.toml` with `[project]` metadata and `[tool.pytest.ini_options]`
- [ ] `tests/conftest.py` with at least one shared fixture (e.g., a list of sample transactions)
- [ ] `data/sample_transactions.json` present
- [ ] `pytest` passes
- [ ] You can explain every line of `pipeline.py` without notes
