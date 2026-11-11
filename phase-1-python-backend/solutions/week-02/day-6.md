# Solutions — Week 02 · Day 6 (Saturday Lab)

Reference implementations for all five exercises. Read only after genuine attempt.

---

## Exercise 0 — generate_transactions

```python
# generate.py
from __future__ import annotations
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_transactions(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    statuses = ["success", "failed", "pending", "refunded"]
    categories = ["retail", "wholesale", "internal"]
    currencies = ["USD", "EUR", "GBP"]
    base_date = datetime(2024, 1, 1)

    ids: list[str] = [f"TXN-{i:06d}" for i in range(1, n + 1)]
    transactions: list[dict] = []

    for i, tid in enumerate(ids):
        # ~10% chance of reusing an earlier ID
        if i > 0 and rng.random() < 0.10:
            tid = ids[rng.randint(0, i - 1)]

        amount = round(rng.uniform(1.0, 10_000.0), 2)
        offset = rng.randint(0, 29)
        hour = rng.randint(0, 23)
        minute = rng.randint(0, 59)
        created_at = (base_date + timedelta(days=offset, hours=hour, minutes=minute)).isoformat()

        transactions.append({
            "id": tid,
            "amount": amount,
            "status": rng.choice(statuses),
            "category": rng.choice(categories),
            "currency": rng.choice(currencies),
            "created_at": created_at,
        })

    return transactions


if __name__ == "__main__":
    data = generate_transactions(200)
    out = Path("data/transactions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} transactions → {out}")
```

---

## Exercise 1 — load_and_validate

```python
# processor.py (partial)
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VALID_STATUSES = {"success", "failed", "pending", "refunded"}
VALID_CURRENCIES = {"USD", "EUR", "GBP"}


class FileProcessingError(OSError):
    pass


class TransactionError(ValueError):
    pass


def _validate_single(tx: Any) -> dict:
    if not isinstance(tx, dict):
        raise TransactionError(f"Expected dict, got {type(tx).__name__}")
    for field in ("id", "amount", "status", "category", "currency", "created_at"):
        if field not in tx:
            raise TransactionError(f"Missing field: '{field}'")
    if not isinstance(tx["id"], str) or not tx["id"]:
        raise TransactionError(f"'id' must be a non-empty string, got {tx['id']!r}")
    if not isinstance(tx["amount"], (int, float)) or tx["amount"] <= 0:
        raise TransactionError(f"'amount' must be positive, got {tx['amount']!r}")
    if tx["status"] not in VALID_STATUSES:
        raise TransactionError(f"'status' {tx['status']!r} not in {VALID_STATUSES}")
    if tx["currency"] not in VALID_CURRENCIES:
        raise TransactionError(f"'currency' {tx['currency']!r} not in {VALID_CURRENCIES}")
    # Validate ISO date string (cheap check — full parse would use datetime)
    try:
        datetime.fromisoformat(tx["created_at"])
    except (ValueError, TypeError) as e:
        raise TransactionError(f"'created_at' not valid ISO-8601: {e}") from e
    return tx


from datetime import datetime


def load_and_validate(path: Path) -> tuple[list[dict], list[str]]:
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as e:
        raise FileProcessingError(f"File not found: {path}") from e
    except json.JSONDecodeError as e:
        raise FileProcessingError(f"Invalid JSON: {e}") from e

    if not isinstance(raw, list):
        raise FileProcessingError("Top-level JSON must be a list")

    valid: list[dict] = []
    errors: list[str] = []
    for i, item in enumerate(raw):
        try:
            valid.append(_validate_single(item))
        except TransactionError as e:
            msg = f"Row {i}: {e}"
            errors.append(msg)
            log.warning(msg)
    return valid, errors
```

**Tests (test_validator.py excerpt)**:
```python
import json, pytest
from pathlib import Path
from processor import load_and_validate, FileProcessingError

def make_valid_tx(**overrides) -> dict:
    base = {
        "id": "T1", "amount": 100.0, "status": "success",
        "category": "retail", "currency": "USD",
        "created_at": "2024-01-01T00:00:00",
    }
    return base | overrides

def test_all_valid(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([make_valid_tx(id="T1"), make_valid_tx(id="T2")]))
    valid, errors = load_and_validate(p)
    assert len(valid) == 2
    assert errors == []

def test_invalid_rows(tmp_path):
    rows = [
        make_valid_tx(id="T1"),
        make_valid_tx(amount=-10.0),          # negative amount
        {"id": "T3"},                          # missing fields
    ]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(rows))
    valid, errors = load_and_validate(p)
    assert len(valid) == 1
    assert len(errors) == 2

def test_file_not_found():
    with pytest.raises(FileProcessingError):
        load_and_validate(Path("/nonexistent/path.json"))
```

---

## Exercise 2 — Aggregation functions

```python
from collections import defaultdict

def total_successful(transactions: list[dict]) -> float:
    return round(sum(tx["amount"] for tx in transactions if tx["status"] == "success"), 2)

def find_failed(transactions: list[dict]) -> list[dict]:
    return sorted(
        (tx for tx in transactions if tx["status"] == "failed"),
        key=lambda tx: tx["amount"],
        reverse=True,
    )

def group_by_status(transactions: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for tx in transactions:
        groups[tx["status"]].append(tx)
    return dict(groups)

def group_by_category(transactions: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for tx in transactions:
        groups[tx["category"]].append(tx)
    return dict(groups)

def currency_totals(transactions: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for tx in transactions:
        totals[tx["currency"]] += tx["amount"]
    return {k: round(v, 2) for k, v in totals.items()}
```

---

## Exercise 3 — Duplicate detection and resolution

```python
def find_duplicate_ids(transactions: list[dict]) -> dict[str, list[dict]]:
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for tx in transactions:
        groups[tx["id"]].append(tx)
    return {k: v for k, v in groups.items() if len(v) > 1}

def remove_duplicates(transactions: list[dict]) -> list[dict]:
    # Sort descending by amount so highest-amount comes first.
    # Stable sort preserves original order for equal amounts.
    sorted_txns = sorted(transactions, key=lambda tx: tx["amount"], reverse=True)
    seen: set[str] = set()
    result: list[dict] = []
    for tx in sorted_txns:
        if tx["id"] not in seen:
            seen.add(tx["id"])
            result.append(tx)
    # Re-sort by original order? For pipeline use, sort by amount after.
    # For deterministic output, sort by id ascending:
    return sorted(result, key=lambda tx: tx["id"])
```

---

## Exercise 4 — Sorting pipeline

```python
def sort_by_amount(
    transactions: list[dict],
    *,
    descending: bool = True,
    status_filter: str | None = None,
) -> list[dict]:
    txns = transactions if status_filter is None else [
        tx for tx in transactions if tx["status"] == status_filter
    ]
    return sorted(txns, key=lambda tx: tx["amount"], reverse=descending)

def top_n_per_status(transactions: list[dict], n: int) -> dict[str, list[dict]]:
    groups = group_by_status(transactions)
    return {
        status: sort_by_amount(txns, descending=True)[:n]
        for status, txns in groups.items()
    }
```

---

## Exercise 5 — Full pipeline

```python
def run_pipeline(input_path: Path, output_path: Path) -> dict:
    valid, errors = load_and_validate(input_path)
    input_count = len(valid) + len(errors)

    dupes = find_duplicate_ids(valid)
    clean = remove_duplicates(valid)
    clean_sorted = sort_by_amount(clean, descending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_sorted, f, indent=2)

    status_counts = {s: len(txns) for s, txns in group_by_status(clean_sorted).items()}

    return {
        "input_count": input_count,
        "invalid_count": len(errors),
        "duplicate_ids": sorted(dupes.keys()),
        "output_count": len(clean_sorted),
        "total_successful": total_successful(clean_sorted),
        "currency_totals": currency_totals(clean_sorted),
        "status_breakdown": status_counts,
    }
```

---

## Stretch goal — Performance comparison

```python
import timeit

transactions = [
    {"id": f"T{i}", "amount": float(i % 100), "status": "success" if i % 2 == 0 else "failed"}
    for i in range(50_000)
]

def method_for_loop(txns):
    total = 0.0
    for tx in txns:
        if tx["status"] == "success":
            total += tx["amount"]
    return total

def method_generator(txns):
    return sum(tx["amount"] for tx in txns if tx["status"] == "success")

def method_map_filter(txns):
    return sum(map(lambda tx: tx["amount"], filter(lambda tx: tx["status"] == "success", txns)))

for name, fn in [("for-loop", method_for_loop), ("generator", method_generator), ("map/filter", method_map_filter)]:
    t = timeit.timeit(lambda: fn(transactions), number=100)
    print(f"{name}: {t:.3f}s")

# Typical result: generator ≈ for-loop, both faster than map/filter due to lambda overhead.
# The generator expression avoids building an intermediate list.
# map+filter chain with lambdas adds per-element function call overhead.
```
