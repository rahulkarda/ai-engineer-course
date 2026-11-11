# Solutions — Week 02 · Day 3

## Exercise 1 — weighted_average

```python
def weighted_average(*values: float, weight: float = 1.0) -> float:
    if not values:
        raise ValueError("At least one value required")
    return round(sum(v * weight for v in values) / len(values), 6)

assert weighted_average(10.0, 20.0, 30.0) == 20.0
assert weighted_average(10.0, 20.0, weight=2.0) == 30.0  # (20+40)/2
assert weighted_average(100.0, weight=0.5) == 50.0
```

Note: `weight` is keyword-only (after `*values`). You cannot call `weighted_average(10, 20, 2.0)` — the third positional would be another value, not `weight`.

## Exercise 2 — build_record

```python
from datetime import datetime
from typing import Any

def build_record(record_id: str, record_type: str, **fields: Any) -> dict:
    return {
        "id": record_id,
        "type": record_type,
        "created_at": datetime.now().isoformat(),
        **fields,
    }

rec = build_record("T1", "transaction", amount=250.0, status="success")
assert rec["id"] == "T1"
assert rec["type"] == "transaction"
assert "created_at" in rec
assert rec["amount"] == 250.0
```

## Exercise 3 — make_threshold_filter (closure)

```python
from collections.abc import Callable

def make_threshold_filter(
    min_amount: float, max_amount: float
) -> Callable[[list[dict]], list[dict]]:
    def filter_fn(transactions: list[dict]) -> list[dict]:
        return [
            tx for tx in transactions
            if min_amount <= tx["amount"] <= max_amount
        ]
    return filter_fn

mid_range = make_threshold_filter(100.0, 500.0)

txns = [
    {"id": "T1", "amount": 50.0},
    {"id": "T2", "amount": 250.0},
    {"id": "T3", "amount": 600.0},
]
result = mid_range(txns)
assert len(result) == 1
assert result[0]["id"] == "T2"

# Closure captures min_amount and max_amount by reference (cell objects)
high_value = make_threshold_filter(1000.0, float("inf"))
assert high_value(txns) == []
```

## Exercise 4 — scope bug

```python
# Bug: count += 1 inside a function creates a local 'count' binding.
# Python sees the assignment and marks 'count' as local.
# But 'count' is read on the right-hand side before any local assignment,
# which raises UnboundLocalError.

count = 0

# WRONG
def increment_broken() -> None:
    count += 1   # UnboundLocalError: local variable 'count' referenced before assignment

# FIX 1: return a new value (preferred — no global state mutation)
def increment(current: int) -> int:
    return current + 1

count = increment(count)
assert count == 1

# FIX 2: use global (avoid if possible)
def increment_global() -> None:
    global count
    count += 1
```

## Exercise 5 — default argument trap fix

```python
# Original bug: the list [] is created ONCE at function definition time.
# All callers share the same list object. Appending to it accumulates across calls.

def attach_tags_buggy(tx: dict, tags: list[str] = []) -> dict:
    tags.append("processed")
    tx["tags"] = tags
    return tx

# Demonstrating the bug:
tx1 = attach_tags_buggy({"id": "T1"})
tx2 = attach_tags_buggy({"id": "T2"})
assert tx1["tags"] == ["processed", "processed"]   # BUG: two entries!

# Fix: use None as default, create the list inside
def attach_tags(tx: dict, tags: list[str] | None = None) -> dict:
    if tags is None:
        tags = []
    tags.append("processed")
    tx["tags"] = tags
    return tx

tx1 = attach_tags({"id": "T1"})
tx2 = attach_tags({"id": "T2"})
assert tx1["tags"] == ["processed"]
assert tx2["tags"] == ["processed"]
assert tx1["tags"] is not tx2["tags"]   # different list objects
```

## Exercise 6 — log_and_call

```python
import logging
from typing import Any
from collections.abc import Callable

log = logging.getLogger(__name__)

def log_and_call(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    log.debug("Calling %s with args=%s kwargs=%s", fn.__name__, args, kwargs)
    return fn(*args, **kwargs)

def add(a: int, b: int) -> int:
    return a + b

result = log_and_call(add, 3, b=4)
assert result == 7

# Works with any callable, preserving all argument forms
result2 = log_and_call(sorted, [3, 1, 2], reverse=True)
assert result2 == [3, 2, 1]
```
