# Solutions — Week 02 · Day 4

## Exercise 1 — Multi-key sort (lambda)

```python
products = [
    {"name": "B", "price": 3.5},
    {"name": "A", "price": 3.5},
    {"name": "C", "price": 1.0},
    {"name": "D", "price": 10.0},
]

# Descending price, ascending name on tie:
# -price for descending (negate), name for ascending
result = sorted(products, key=lambda p: (-p["price"], p["name"]))

assert result[0]["name"] == "D"           # highest price
assert result[1]["name"] == "A"           # 3.5, 'A' before 'B'
assert result[2]["name"] == "B"
assert result[3]["name"] == "C"           # lowest price
```

## Exercise 2 — map vs comprehension (with timeit)

```python
import timeit

transactions = [{"id": f"T{i}", "amount": float(i)} for i in range(10_000)]

def extract_with_map(txns: list[dict]) -> list[float]:
    return list(map(lambda tx: tx["amount"], txns))

def extract_with_comprehension(txns: list[dict]) -> list[float]:
    return [tx["amount"] for tx in txns]

# Benchmark
map_time = timeit.timeit(lambda: extract_with_map(transactions), number=1000)
comp_time = timeit.timeit(lambda: extract_with_comprehension(transactions), number=1000)

print(f"map: {map_time:.3f}s, comprehension: {comp_time:.3f}s")
# Comprehension is typically faster or comparable because:
# - No lambda call overhead per element
# - Specialized LIST_APPEND bytecode
# Actual results vary by Python version and machine.

# Verify same result
assert extract_with_map(transactions) == extract_with_comprehension(transactions)
```

## Exercise 3 — filter chain

```python
def high_value_successes(txns: list[dict], threshold: float) -> list[dict]:
    successful = filter(lambda tx: tx["status"] == "success", txns)
    return list(filter(lambda tx: tx["amount"] > threshold, successful))

txns = [
    {"id": "T1", "amount": 1500.0, "status": "success"},
    {"id": "T2", "amount": 500.0,  "status": "success"},
    {"id": "T3", "amount": 2000.0, "status": "failed"},
]
result = high_value_successes(txns, threshold=1000.0)
assert len(result) == 1
assert result[0]["id"] == "T1"
```

## Exercise 4 — ISO string sort (no datetime parse)

```python
transactions = [
    {"id": "T1", "created_at": "2024-03-15T10:30:00"},
    {"id": "T2", "created_at": "2024-01-01T09:00:00"},
    {"id": "T3", "created_at": "2024-03-15T08:00:00"},
    {"id": "T4", "created_at": "2023-12-31T23:59:59"},
]

# ISO-8601 strings compare lexicographically in the same order as datetime values
# because the format is YYYY-MM-DDTHH:MM:SS — most significant part leftmost.
chronological = sorted(transactions, key=lambda tx: tx["created_at"])

assert chronological[0]["id"] == "T4"   # 2023-12-31
assert chronological[1]["id"] == "T2"   # 2024-01-01
assert chronological[2]["id"] == "T3"   # 2024-03-15T08:00
assert chronological[3]["id"] == "T1"   # 2024-03-15T10:30
```

## Exercise 5 — Demonstrating sort stability

```python
# Start with transactions sorted by amount
base = [
    {"id": "T1", "amount": 100.0, "status": "failed"},
    {"id": "T2", "amount": 200.0, "status": "success"},
    {"id": "T3", "amount": 300.0, "status": "failed"},
    {"id": "T4", "amount": 400.0, "status": "success"},
]

# Sort by status. Within each group, original (amount-ascending) order must be preserved.
by_status = sorted(base, key=lambda tx: tx["status"])

# "failed" group: T1 (100), T3 (300) — original relative order
failed_group = [tx for tx in by_status if tx["status"] == "failed"]
assert [t["id"] for t in failed_group] == ["T1", "T3"]

# "success" group: T2 (200), T4 (400)
success_group = [tx for tx in by_status if tx["status"] == "success"]
assert [t["id"] for t in success_group] == ["T2", "T4"]
# Stability preserved: within each status, amount order (from initial sort) is kept.
```

## Exercise 6 — itemgetter limitations

```python
from operator import itemgetter

products = [
    {"name": "B", "price": 3.5},
    {"name": "A", "price": 3.5},
    {"name": "C", "price": 1.0},
]

# itemgetter: ascending by name, then ascending by price
result = sorted(products, key=itemgetter("name", "price"))
# Works fine for same-direction sorts.

# LIMITATION: itemgetter cannot negate a field for descending sort.
# For mixed directions (price desc, name asc), you must use a lambda:
result_mixed = sorted(products, key=lambda p: (-p["price"], p["name"]))
# itemgetter has no way to express -p["price"].

# itemgetter is faster than lambda for same-direction sorts because
# it is implemented in C and avoids Python frame overhead.
```
