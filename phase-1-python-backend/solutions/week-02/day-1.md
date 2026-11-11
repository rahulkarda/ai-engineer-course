# Solutions — Week 02 · Day 1

## Exercise 1 — price_stats

```python
def price_stats(prices: list[float]) -> dict[str, float]:
    unique = list(set(prices))
    return {
        "min": min(unique),
        "max": max(unique),
        "mean": round(sum(unique) / len(unique), 4),
        "count": float(len(unique)),
    }

# Test
prices = [10.5, 3.2, 99.0, 45.0, 3.2, 72.1]
result = price_stats(prices)
assert result["count"] == 5.0          # one 3.2 removed
assert result["min"] == 3.2
assert result["max"] == 99.0
```

## Exercise 2 — make_transaction / is_high_value

```python
def make_transaction(tx_id: str, amount: float, status: str) -> tuple[str, float, str]:
    return (tx_id, amount, status)

def is_high_value(tx: tuple[str, float, str], threshold: float = 1000.0) -> bool:
    _, amount, _ = tx
    return amount > threshold

assert is_high_value(("T1", 1500.0, "success"))
assert not is_high_value(("T2", 500.0, "failed"))
assert is_high_value(("T3", 200.0, "success"), threshold=100.0)
```

## Exercise 3 — index_by_id

```python
def index_by_id(transactions: list[dict]) -> dict[str, dict]:
    return {tx["id"]: tx for tx in transactions}

txns = [{"id": "T1", "amount": 100.0}, {"id": "T2", "amount": 200.0}]
idx = index_by_id(txns)
assert idx["T1"]["amount"] == 100.0
assert "T2" in idx
```

Note: if duplicate IDs exist, later entries overwrite earlier ones. Exercise 3 from day 2 handles deduplication explicitly.

## Exercise 4 — apply_defaults

```python
def apply_defaults(tx: dict, defaults: dict) -> dict:
    # defaults | tx → tx values win on conflict (Python 3.9+)
    return defaults | tx

# Alternative (pre-3.9):
def apply_defaults_pre39(tx: dict, defaults: dict) -> dict:
    result = dict(defaults)
    result.update(tx)
    return result

defaults = {"currency": "USD", "fee": 0.0, "status": "pending"}
tx = {"id": "T1", "amount": 50.0, "status": "success"}
merged = apply_defaults(tx, defaults)
assert merged["currency"] == "USD"       # from defaults
assert merged["status"] == "success"     # tx wins
assert merged["id"] == "T1"
assert tx == {"id": "T1", "amount": 50.0, "status": "success"}   # not mutated
```

## Exercise 5 — safe_get

```python
def safe_get(data: dict, *keys: str, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current

data = {"user": {"name": "Alice", "balance": 500.0}}
assert safe_get(data, "user", "name") == "Alice"
assert safe_get(data, "user", "missing") is None
assert safe_get(data, "missing", "name") is None
assert safe_get(data, "user", "balance") == 500.0
assert safe_get(data, "user", "balance", default=0.0) == 500.0
```

**Key concepts reinforced**:
- Dict comprehension for inversion/indexing
- Tuple unpacking in signatures
- `|` merge operator (Python 3.9+)
- Safe traversal of arbitrary depth without repeated `.get` chaining
