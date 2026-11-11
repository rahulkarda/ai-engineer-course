# Phase 1 · Week 02 · Day 4 — Lambda, map/filter/sorted, and Custom Sorting

> Objective: Use lambda expressions and the functional trio (map/filter/sorted) with custom key functions.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### Lambda expressions
A `lambda` is an anonymous, single-expression function. Its scope rules and return behavior are identical to a normal `def` — the expression is the implicit return value.

```python
square = lambda x: x ** 2    # equivalent to:
def square(x): return x ** 2
```

**Use lambdas for short, throwaway key functions passed inline.** For anything with logic, conditionals, or a docstring, use `def`.

### map and filter
`map(fn, iterable)` applies `fn` to every element; `filter(fn, iterable)` keeps elements where `fn(x)` is truthy. Both return **lazy iterators** — nothing runs until you iterate or convert to a list.

In modern Python, comprehensions are usually preferred for readability. Reach for `map`/`filter` when chaining functional pipelines or when passing them to APIs that expect callables.

### sorted and sort
`sorted(iterable, key=fn, reverse=False)` returns a **new sorted list**. `list.sort()` sorts **in place** and returns `None`. Both use Timsort (O(n log n), stable).

**Stability matters**: equal-key elements preserve their relative input order. This lets you sort by multiple criteria by sorting multiple times, outermost sort last.

The `key` function is called exactly once per element, and its return value is cached. This is the **Schwartzian transform** — transform, sort, extract — which avoids recomputing the key on each comparison.

### operator module
`operator.attrgetter`, `operator.itemgetter`, and `operator.methodcaller` provide faster, more readable key functions than lambdas for common access patterns.

```python
from operator import itemgetter
records = [{"name": "Alice", "score": 88}, {"name": "Bob", "score": 95}]
records.sort(key=itemgetter("score"), reverse=True)
```

## How it works internally

**Timsort** is a hybrid merge/insertion sort. It detects already-sorted "runs" in the data and merges them — real-world data often has partial order, so Timsort beats pure merge sort in practice. The sort is **stable** by definition in CPython (guaranteed since Python 2.2).

**`key=` optimization**: Python calls `key(x)` once per element and stores `(key_result, x)` pairs. Comparisons happen on `key_result` only, so the key function's cost is paid exactly `n` times, not `O(n log n)` times.

**Lazy `map`/`filter`**: they return iterator objects that hold a reference to the source iterable and the function. Elements are computed on demand. Memory usage is O(1) regardless of input size.

```python
# Demonstrating laziness
m = map(lambda x: print(f"processing {x}") or x*2, range(5))
print("map created — nothing printed yet")
next(m)   # "processing 0" — computed on first next()
```

## Worked example

```python
# functional_sorting.py
from __future__ import annotations
import logging
from operator import itemgetter

log = logging.getLogger(__name__)

transactions: list[dict] = [
    {"id": "T1", "amount": 250.0,   "status": "success", "category": "retail"},
    {"id": "T2", "amount": 1_000.0, "status": "failed",  "category": "wholesale"},
    {"id": "T3", "amount": 75.5,    "status": "success", "category": "retail"},
    {"id": "T4", "amount": 3_200.0, "status": "pending", "category": "wholesale"},
    {"id": "T5", "amount": 75.5,    "status": "failed",  "category": "retail"},
]


# --- Lambda + sorted ---
by_amount = sorted(transactions, key=lambda tx: tx["amount"])
by_amount_desc = sorted(transactions, key=lambda tx: tx["amount"], reverse=True)

# --- itemgetter (faster, more readable) ---
by_amount_ig = sorted(transactions, key=itemgetter("amount"))


# --- Multi-key sort: primary status, secondary amount descending ---
STATUS_ORDER = {"success": 0, "pending": 1, "failed": 2}

by_status_then_amount = sorted(
    transactions,
    key=lambda tx: (STATUS_ORDER.get(tx["status"], 99), -tx["amount"]),
)
# Negative amount → higher amounts first within same status


# --- map: extract amounts ---
amounts = list(map(lambda tx: tx["amount"], transactions))
# Equivalent comprehension (prefer this):
amounts_comp = [tx["amount"] for tx in transactions]


# --- filter: keep successful only ---
successful = list(filter(lambda tx: tx["status"] == "success", transactions))
# Equivalent comprehension:
successful_comp = [tx for tx in transactions if tx["status"] == "success"]


# --- Chained pipeline ---
total_successful = sum(
    map(
        lambda tx: tx["amount"],
        filter(lambda tx: tx["status"] == "success", transactions),
    )
)
log.info("Total successful: %.2f", total_successful)


# --- sorted with a custom class key ---
from dataclasses import dataclass

@dataclass
class Transaction:
    id: str
    amount: float
    status: str

txn_objects = [
    Transaction("T1", 250.0, "success"),
    Transaction("T2", 1_000.0, "failed"),
    Transaction("T3", 75.5, "success"),
]

# Sort by amount ascending, then id ascending on tie
txn_sorted = sorted(txn_objects, key=lambda t: (t.amount, t.id))
log.debug("Sorted txns: %s", txn_sorted)
```

## Exercises

1. **Lambda sort**: Given `products = [{"name": "A", "price": 3.5}, ...]`, write one `sorted()` call to sort by price descending, then name ascending on tie.

2. **map pipeline**: Write `extract_amounts(txns: list[dict]) -> list[float]` using `map` (not a comprehension). Then write the equivalent comprehension version. Benchmark both with `timeit` on 10,000 transactions.

3. **filter chain**: Write `high_value_successes(txns: list[dict], threshold: float) -> list[dict]` using `filter` twice (once for status, once for amount).

4. **Custom key**: Given transactions with `"created_at"` as an ISO datetime string (`"2024-01-15T10:30:00"`), sort them chronologically using `key=` without parsing to a datetime object. Hint: ISO-8601 strings sort lexicographically.

5. **Stable sort**: Demonstrate that Python's sort is stable: start with transactions already sorted by amount; then sort by status. Show that within each status group, the amount order is preserved.

6. **operator.itemgetter**: Rewrite exercise 1 using `operator.itemgetter` instead of a lambda. What are the limitations of `itemgetter` for multi-field sort with mixed directions?

(Solutions: ../solutions/week-02/day-4.md)

## Self-check
- [ ] I can explain when to use `lambda` vs `def`
- [ ] I can explain that `map` and `filter` are lazy and what that means for memory
- [ ] I can write a multi-key sort using a tuple key
- [ ] I can explain sort stability and give a practical use case
- [ ] I can use `operator.itemgetter` and explain when it beats lambda

## Supplementary links
- [Sorting HOW TO (Python docs)](https://docs.python.org/3/howto/sorting.html)
- [Timsort — Wikipedia](https://en.wikipedia.org/wiki/Timsort)
- [operator module](https://docs.python.org/3/library/operator.html)
