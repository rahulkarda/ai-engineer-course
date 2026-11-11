# Solutions — Week 02 · Day 2

## Exercise 1 — unique_transactions

```python
def unique_transactions(txns: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result: list[dict] = []
    for tx in txns:
        if tx["id"] not in seen:
            seen.add(tx["id"])
            result.append(tx)
    return result

txns = [
    {"id": "T1", "amount": 10.0},
    {"id": "T2", "amount": 20.0},
    {"id": "T1", "amount": 99.0},   # duplicate — dropped
]
unique = unique_transactions(txns)
assert len(unique) == 2
assert unique[0]["id"] == "T1"
assert unique[0]["amount"] == 10.0   # first occurrence kept
```

## Exercise 2 — transaction_report

```python
def transaction_report(
    processed: list[str], billed: list[str]
) -> dict[str, set[str]]:
    p, b = set(processed), set(billed)
    return {
        "only_processed": p - b,
        "only_billed": b - p,
        "both": p & b,
    }

report = transaction_report(["T1", "T2", "T3"], ["T2", "T3", "T4"])
assert report["only_processed"] == {"T1"}
assert report["only_billed"] == {"T4"}
assert report["both"] == {"T2", "T3"}
```

## Exercise 3 — filter_by_status (comprehension)

```python
def filter_by_status(txns: list[dict], status: str) -> list[float]:
    return [tx["amount"] for tx in txns if tx["status"] == status]

txns = [
    {"amount": 100.0, "status": "success"},
    {"amount": 200.0, "status": "failed"},
    {"amount": 50.0, "status": "success"},
]
assert filter_by_status(txns, "success") == [100.0, 50.0]
assert filter_by_status(txns, "pending") == []
```

## Exercise 4 — build_lookup (with negative amount filter)

```python
def build_lookup(txns: list[dict]) -> dict[str, dict]:
    return {
        tx["id"]: tx
        for tx in txns
        if tx.get("amount", 0) > 0
    }

txns = [
    {"id": "T1", "amount": 100.0},
    {"id": "T2", "amount": 0.0},    # excluded
    {"id": "T3", "amount": -5.0},   # excluded
    {"id": "T4", "amount": 200.0},
]
lookup = build_lookup(txns)
assert set(lookup.keys()) == {"T1", "T4"}
```

## Exercise 5 — all_transaction_ids (nested comprehension)

```python
def all_transaction_ids(data: dict) -> list[str]:
    return [
        tx["id"]
        for region_txns in data["regions"].values()
        for tx in region_txns
    ]

data = {"regions": {"EU": [{"id": "T1"}, {"id": "T2"}], "US": [{"id": "T3"}]}}
ids = all_transaction_ids(data)
assert set(ids) == {"T1", "T2", "T3"}
assert len(ids) == 3
```

Note: dict `.values()` iteration order is insertion-order in Python 3.7+, so results are deterministic.

## Exercise 6 — clone_transaction (deep copy)

```python
import copy

def clone_transaction(tx: dict) -> dict:
    return copy.deepcopy(tx)

original = {"id": "T1", "amount": 100.0, "tags": ["retail", "domestic"]}
clone = clone_transaction(original)
clone["tags"].append("modified")

assert "modified" not in original["tags"]   # deep copy — isolated
assert clone["id"] == original["id"]         # same value, different object
```

**Why `copy.deepcopy` and not `dict.copy()`?**
`dict.copy()` creates a new dict but inner objects (like `tags`) are shared references. Modifying `clone["tags"]` would also modify `original["tags"]`. `deepcopy` recursively copies all nested objects.
