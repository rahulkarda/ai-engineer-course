# Phase 1 · Week 02 · Day 1 — Lists, Tuples, and Dictionaries

> Objective: Master Python's three core ordered/mapped collection types and their runtime characteristics.  ·  Time: ~1 hr  ·  Prereqs: [[week-01/day-1]]

## Concept

Python ships three workhorse collection types you'll use every day:

| Type | Mutable | Ordered | Duplicate values | Key lookup |
|------|---------|---------|-----------------|------------|
| `list` | yes | yes | yes | O(n) |
| `tuple` | no | yes | yes | O(n) |
| `dict` | yes | insertion-ordered (3.7+) | keys unique, values dup | O(1) avg |

**Lists** are dynamic arrays — append is amortized O(1), insert/delete at arbitrary index is O(n) because elements shift.

**Tuples** are immutable sequences. Immutability makes them hashable (when all elements are hashable), so they can be dict keys or set members. Prefer tuples for data that should not change (coordinates, DB rows, function return pairs).

**Dicts** are hash maps. Lookup, insert, delete are O(1) average. Python 3.7+ preserves insertion order. Values are retrieved by key, not position.

## How it works internally

**List** allocates a contiguous block of pointers (not the objects themselves) with extra capacity. When the block fills, Python reallocates at ~1.125× the old size and copies pointers — that's why appending to a list is amortized O(1) but `insert(0, x)` is O(n).

**Tuple** is stored identically but the pointer block is fixed at creation time and cannot grow. The interpreter can intern small tuples and reuse their memory.

**Dict** (CPython 3.6+) uses a compact hash table with an index array and a separate entries array. Collision is resolved by open addressing with a probe sequence derived from `hash(key)`. Loading above ~67% triggers a resize.

```python
import sys

lst = [1, 2, 3]
tpl = (1, 2, 3)
print(sys.getsizeof(lst))   # larger — has extra capacity
print(sys.getsizeof(tpl))   # smaller — exact fit
```

## Worked example

```python
# financial_data.py
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

# --- List: mutable sequence of transaction amounts ---
amounts: list[float] = [250.0, 1_000.0, 75.50, 3_200.0, 42.0]

# Common list operations
amounts.append(500.0)                   # O(1) amortized
amounts.insert(0, 10.0)                 # O(n) — shifts all elements
total = sum(amounts)
largest = max(amounts)
amounts.sort()                          # in-place, Timsort O(n log n)

log.debug("Sorted amounts: %s", amounts)
print(f"Total: {total:.2f}, Largest: {largest:.2f}")


# --- Tuple: immutable record ---
Transaction = tuple[str, float, str]   # (id, amount, status)

tx: Transaction = ("TXN-001", 250.0, "success")
tx_id, tx_amount, tx_status = tx        # unpacking

# Tuples as dict keys (hashable)
coords: dict[tuple[int, int], str] = {(0, 0): "origin", (1, 2): "point_a"}


# --- Dict: string → mixed data ---
transaction: dict[str, object] = {
    "id": "TXN-001",
    "amount": 250.0,
    "currency": "USD",
    "status": "success",
    "tags": ["retail", "domestic"],
}

# Safe access patterns
status = transaction.get("status", "unknown")
currency = transaction.get("currency", "USD")

# Iterating
for key, value in transaction.items():
    log.debug("%s = %s", key, value)

# Updating
transaction["status"] = "processed"
transaction.update({"fee": 2.50, "processed": True})

# Merging dicts (Python 3.9+ | operator)
defaults: dict[str, object] = {"currency": "USD", "fee": 0.0}
merged = defaults | transaction          # transaction values win on conflict

print(merged)
```

## Exercises

1. **List operations**: Given `prices = [10.5, 3.2, 99.0, 45.0, 3.2, 72.1]`, write a function `price_stats(prices: list[float]) -> dict[str, float]` that returns `{"min": ..., "max": ..., "mean": ..., "count": ...}`. Remove duplicates before computing stats.

2. **Tuple records**: Write a function `make_transaction(tx_id: str, amount: float, status: str) -> tuple[str, float, str]` and a function `is_high_value(tx: tuple[str, float, str], threshold: float = 1000.0) -> bool`.

3. **Dict manipulation**: Given a list of transaction dicts (each with `"id"`, `"amount"`, `"status"`), write `index_by_id(transactions: list[dict]) -> dict[str, dict]` that returns a dict keyed by transaction ID.

4. **Dict merging**: Write `apply_defaults(tx: dict, defaults: dict) -> dict` that returns a new dict with `defaults` values for any key not present in `tx`. Do not mutate either input.

5. **Nested access**: Given `data = {"user": {"name": "Alice", "balance": 500.0}}`, write `safe_get(data: dict, *keys: str, default=None)` that traverses nested dicts and returns `default` if any key is missing.

(Solutions: ../solutions/week-02/day-1.md)

## Self-check
- [ ] I can explain why list `insert(0, x)` is O(n) and `append(x)` is O(1) amortized
- [ ] I can explain when to prefer a tuple over a list
- [ ] I can describe how a Python dict resolves hash collisions
- [ ] I can implement index-by-key lookups without a loop at call time
- [ ] I can safely access nested dict values without KeyError

## Supplementary links
- [Python Data Model — sequences](https://docs.python.org/3/reference/datamodel.html#emulating-container-types)
- [Time complexity of list/dict operations (Python wiki)](https://wiki.python.org/moin/TimeComplexity)
- [CPython dict internals (Raymond Hettinger)](https://code.activestate.com/recipes/578375/)
