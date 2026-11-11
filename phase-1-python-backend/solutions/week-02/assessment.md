# Solutions — Week 02 Assessment

## Conceptual answers

**1. Time complexity — dict vs list membership**

`x in my_dict` is O(1) average because a dict is a hash map: Python computes `hash(x)`, looks up the slot directly. `x in my_list` is O(n) because Python scans from index 0, comparing each element until it finds a match or exhausts the list.

**2. LEGB rule**

Python resolves names in this order: Local → Enclosing (closures, inner-to-outer) → Global (module top level) → Built-in. When you assign to a variable inside a function that shares a name with a module-level variable, Python creates a new *local* binding — it never touches the outer variable. Without a `global` declaration, assignment inside a function is always local.

**3. Mutable default argument bug — fix**

```python
# Bug: [] is evaluated once at definition; all calls share the same list
def register(user: str, roles: list[str] = []) -> None:
    roles.append("viewer")   # accumulates across calls

# Fix: use None, create list inside
def register(user: str, roles: list[str] | None = None) -> None:
    if roles is None:
        roles = []
    roles.append("viewer")
    ...
```

**4. map vs comprehension**

Prefer `map` when passing a callable to another API (e.g., `functools.reduce(fn, map(transform, items))`), or when chaining functional pipelines and avoiding temporary lists. Prefer comprehensions for readability, conditional filtering in one expression, and when the transformation logic is more complex than a single expression.

**5. raise ... from e**

`raise ValueError("bad input") from e` sets `__cause__` on the new exception, preserving the original traceback. Without `from e`, the original exception is still attached as `__context__` but the chain is implicit. `from e` makes the relationship explicit and shows both tracebacks in the error message, making debugging much faster.

**6. set vs list for (transaction_id, amount) pairs**

Use a `set[tuple[str, float]]` — tuples are hashable (both `str` and `float` are hashable), so the pair can be a set member. Membership test is O(1). A `list` would be O(n) for deduplication checks. A `dict` would work but adds unnecessary values. What you lose: insertion order (irrelevant for deduplication) and the ability to store unhashable types as tuple elements.

**7. Sort stability**

`sorted()` and `list.sort()` guarantee that elements comparing as equal appear in their original relative order in the output. Practical use: sort by secondary key first, then by primary key. The primary sort preserves the secondary order within tied groups.

---

## Coding challenge — reference implementation

```python
# challenge.py
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

VALID_STATUSES: frozenset[str] = frozenset({"success", "failed", "pending", "refunded"})


class TransactionValidationError(ValueError):
    """Raised when a transaction record fails validation."""


def validate(tx: Any) -> dict:
    if not isinstance(tx, dict):
        raise TransactionValidationError(f"Expected dict, got {type(tx).__name__}")
    for field in ("id", "amount", "status"):
        if field not in tx:
            raise TransactionValidationError(f"Missing required field: '{field}'")
    if not isinstance(tx["id"], str) or not tx["id"].strip():
        raise TransactionValidationError(f"'id' must be a non-empty string")
    if not isinstance(tx["amount"], (int, float)) or tx["amount"] <= 0:
        raise TransactionValidationError(f"'amount' must be a positive number, got {tx['amount']!r}")
    if tx["status"] not in VALID_STATUSES:
        raise TransactionValidationError(
            f"'status' must be one of {sorted(VALID_STATUSES)}, got {tx['status']!r}"
        )
    return tx


def load_and_validate(path: Path) -> tuple[list[dict], list[str]]:
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    valid: list[dict] = []
    errors: list[str] = []
    for i, item in enumerate(data):
        try:
            valid.append(validate(item))
        except TransactionValidationError as e:
            errors.append(f"Row {i}: {e}")
            log.warning("Row %d invalid: %s", i, e)
    return valid, errors


def summarize(transactions: list[dict]) -> dict[str, Any]:
    if not transactions:
        return {
            "total_count": 0,
            "total_amount": 0.0,
            "successful_amount": 0.0,
            "by_status": {},
            "duplicate_ids": [],
            "top_3_by_amount": [],
        }

    total_amount = round(sum(tx["amount"] for tx in transactions), 2)
    successful_amount = round(
        sum(tx["amount"] for tx in transactions if tx["status"] == "success"), 2
    )

    by_status: dict[str, int] = {}
    for tx in transactions:
        by_status[tx["status"]] = by_status.get(tx["status"], 0) + 1

    id_counts: dict[str, int] = {}
    for tx in transactions:
        id_counts[tx["id"]] = id_counts.get(tx["id"], 0) + 1
    duplicate_ids = sorted(k for k, v in id_counts.items() if v > 1)

    top_3 = sorted(transactions, key=lambda tx: tx["amount"], reverse=True)[:3]

    return {
        "total_count": len(transactions),
        "total_amount": total_amount,
        "successful_amount": successful_amount,
        "by_status": by_status,
        "duplicate_ids": duplicate_ids,
        "top_3_by_amount": top_3,
    }


def save(transactions: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False)
    log.info("Saved %d transactions to %s", len(transactions), path)
```

**Tests:**
```python
# test_challenge.py
import json, pytest
from pathlib import Path
from challenge import validate, load_and_validate, summarize, save, TransactionValidationError

VALID = {"id": "T1", "amount": 100.0, "status": "success"}

def test_validate_happy():
    assert validate(dict(VALID)) == VALID

def test_validate_missing_field():
    with pytest.raises(TransactionValidationError, match="Missing"):
        validate({"id": "T1", "amount": 10.0})   # no status

def test_validate_wrong_type():
    with pytest.raises(TransactionValidationError):
        validate({"id": "T1", "amount": "not a number", "status": "success"})

def test_validate_negative_amount():
    with pytest.raises(TransactionValidationError):
        validate({"id": "T1", "amount": -5.0, "status": "success"})

def test_validate_bad_status():
    with pytest.raises(TransactionValidationError):
        validate({"id": "T1", "amount": 10.0, "status": "unknown"})

def test_load_and_validate(tmp_path):
    data = [VALID, {"id": "T2", "amount": -1.0, "status": "failed"}]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(data))
    valid, errors = load_and_validate(p)
    assert len(valid) == 1
    assert len(errors) == 1

def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_and_validate(Path("/no/such/file.json"))

def test_summarize_empty():
    s = summarize([])
    assert s["total_count"] == 0
    assert s["duplicate_ids"] == []
    assert s["top_3_by_amount"] == []

def test_summarize_duplicates():
    txns = [VALID, VALID, {"id": "T2", "amount": 50.0, "status": "failed"}]
    s = summarize(txns)
    assert "T1" in s["duplicate_ids"]

def test_save_round_trip(tmp_path):
    p = tmp_path / "out.json"
    save([VALID], p)
    loaded = json.loads(p.read_text())
    assert loaded == [VALID]
```

---

## Debugging challenge — four bugs identified

```python
import json

def process(file_path):
    f = open(file_path)          # BUG 1: no 'with' — file never closed on exception
    data = json.load(f)
    
    results = []
    seen = []                    # BUG 2: list for membership — O(n) per check; use set
    for tx in data:
        if tx["id"] in seen:
            continue
        seen.append(tx["id"])
        
        amount = tx["amount"]
        if amount > 0:
            results.append(tx)
    
    totals = {}
    for tx in results:
        status = tx.get("status")
        totals[status] = totals[status] + tx["amount"]   # BUG 3: KeyError on first occurrence
                                                           # (key not yet in totals)
    
    return sorted(results, key=tx["amount"])   # BUG 4: tx is the loop variable from the
                                                # PREVIOUS loop (last item); should be lambda
```

**Corrected version:**
```python
import json
from pathlib import Path

def process(file_path: str) -> list[dict]:
    # Fix 1: context manager
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    seen: set[str] = set()   # Fix 2: O(1) membership
    for tx in data:
        if tx["id"] in seen:
            continue
        seen.add(tx["id"])
        if tx["amount"] > 0:
            results.append(tx)

    totals: dict[str, float] = {}
    for tx in results:
        status = tx.get("status", "unknown")
        totals[status] = totals.get(status, 0.0) + tx["amount"]   # Fix 3: .get with default

    return sorted(results, key=lambda tx: tx["amount"])   # Fix 4: lambda, not bare variable
```

---

## Code quality — refactored version

```python
# Original (low quality):
def process_transactions(l):
    x = []
    for i in range(len(l)):
        item = l[i]
        if item['status'] == 'success' or item['status'] == 'pending':
            new_item = {}
            new_item['id'] = item['id']
            new_item['amount'] = item['amount']
            new_item['status'] = item['status']
            if item['amount'] > 1000:
                new_item['tier'] = 'high'
            else:
                new_item['tier'] = 'low'
            x.append(new_item)
    return x

# Refactored (9 non-blank lines):
ACTIVE_STATUSES = {"success", "pending"}

def enrich_active_transactions(transactions: list[dict]) -> list[dict]:
    """Return enriched records for active (success/pending) transactions."""
    def _tier(amount: float) -> str:
        return "high" if amount > 1000 else "low"

    return [
        {"id": tx["id"], "amount": tx["amount"], "status": tx["status"], "tier": _tier(tx["amount"])}
        for tx in transactions
        if tx["status"] in ACTIVE_STATUSES
    ]
```

**Changes made and why:**
- `process_transactions(l)` → descriptive name + type annotation
- `range(len(l))` → iterate directly over `transactions`
- `or` chain → membership test against a `set` (extensible, readable)
- Manual dict building → inline dict literal in comprehension
- `if/else` for tier → helper `_tier` function (single expression)
- `for` loop + `append` → list comprehension (idiomatic, faster)
