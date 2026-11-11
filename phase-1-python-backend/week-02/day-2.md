# Phase 1 · Week 02 · Day 2 — Sets, Comprehensions, and Nested Collections

> Objective: Use sets for membership and deduplication; write list/dict/set comprehensions; navigate nested data structures.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Sets
A `set` is an unordered collection of unique, hashable objects backed by a hash table. Core guarantees:

- Membership test `x in s` — O(1) average
- `add`, `remove`, `discard` — O(1) average
- Set algebra: `|` union, `&` intersection, `-` difference, `^` symmetric difference

Use sets when you need uniqueness, fast membership, or set-algebra operations.

### Comprehensions
Comprehensions are concise, single-expression syntax for building collections from iterables. They are more Pythonic *and faster* than equivalent `for`-loop-with-append patterns because the bytecode specializes the loop.

| Form | Syntax | Result |
|------|--------|--------|
| List | `[expr for x in iter if cond]` | `list` |
| Dict | `{k: v for x in iter if cond}` | `dict` |
| Set | `{expr for x in iter if cond}` | `set` |

### Nested collections
Real-world data is rarely flat. JSON APIs, database rows, and config files routinely embed lists inside dicts inside lists. Python has no restriction on nesting depth — any object (including another list or dict) can be a value.

## How it works internally

**Sets** share the hash table implementation with `dict` but store only keys (no values). CPython uses the same collision resolution strategy (open addressing with a probe sequence). Insertion triggers resize when load factor exceeds ~67%.

**Comprehensions** compile to a specialized bytecode (`LIST_APPEND`, `MAP_ADD`, `SET_ADD`) that avoids the per-iteration overhead of a `list.append` lookup. A comprehension also creates its own implicit scope (since Python 3), so loop variables don't leak.

**Nested dicts/lists** are just normal objects referenced via pointers. Shallow copy (`dict.copy()`, `list[:]`) copies only the top-level container — inner objects are shared. Use `copy.deepcopy` or manual reconstruction to fully decouple nested structures.

```python
import copy

original = {"transactions": [{"id": "T1", "amount": 100}]}
shallow = original.copy()
deep = copy.deepcopy(original)

shallow["transactions"][0]["amount"] = 999   # mutates original too
deep["transactions"][0]["amount"] = 0         # isolated
print(original["transactions"][0]["amount"])  # 999 — shallow shared inner list
```

## Worked example

```python
# sets_comprehensions.py
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

transactions: list[dict] = [
    {"id": "T1", "amount": 250.0, "status": "success", "category": "retail"},
    {"id": "T2", "amount": 1_000.0, "status": "failed", "category": "wholesale"},
    {"id": "T1", "amount": 250.0, "status": "success", "category": "retail"},  # duplicate
    {"id": "T3", "amount": 75.5, "status": "success", "category": "retail"},
    {"id": "T4", "amount": 3_200.0, "status": "pending", "category": "wholesale"},
]

# --- Set: find duplicate IDs ---
seen: set[str] = set()
duplicates: set[str] = set()
for tx in transactions:
    tid = tx["id"]
    if tid in seen:
        duplicates.add(tid)
    seen.add(tid)
log.info("Duplicate IDs: %s", duplicates)   # {'T1'}

# --- Set algebra ---
success_ids = {tx["id"] for tx in transactions if tx["status"] == "success"}
failed_ids  = {tx["id"] for tx in transactions if tx["status"] == "failed"}
all_ids     = {tx["id"] for tx in transactions}

only_success = success_ids - failed_ids          # difference
in_both      = success_ids & failed_ids          # intersection (empty here)
all_status   = success_ids | failed_ids          # union

# --- List comprehension ---
amounts: list[float] = [tx["amount"] for tx in transactions]
high_value: list[dict] = [tx for tx in transactions if tx["amount"] > 500]

# --- Dict comprehension: id → amount index ---
amount_index: dict[str, float] = {
    tx["id"]: tx["amount"]
    for tx in transactions
}

# --- Nested comprehension: flatten categories ---
# Each tx could have multiple categories as a list
multi_tx: list[dict] = [
    {"id": "T1", "categories": ["retail", "domestic"]},
    {"id": "T2", "categories": ["wholesale", "international"]},
]
all_categories: list[str] = [
    cat
    for tx in multi_tx
    for cat in tx["categories"]   # inner loop second
]
# → ['retail', 'domestic', 'wholesale', 'international']

# --- Dict of lists: group by status ---
from collections import defaultdict

grouped: dict[str, list[dict]] = defaultdict(list)
for tx in transactions:
    grouped[tx["status"]].append(tx)

# Access nested result
log.info("Successful txns: %s", [t["id"] for t in grouped["success"]])
```

## Exercises

1. **Deduplication**: Write `unique_transactions(txns: list[dict]) -> list[dict]` that removes duplicates by `"id"` while preserving the first occurrence and original order.

2. **Set algebra**: Given two lists of transaction IDs, write `transaction_report(processed: list[str], billed: list[str]) -> dict[str, set[str]]` returning `{"only_processed": ..., "only_billed": ..., "both": ...}`.

3. **List comprehension**: Write `filter_by_status(txns: list[dict], status: str) -> list[float]` that returns a list of amounts for transactions matching `status`, using a single comprehension expression.

4. **Dict comprehension**: Write `build_lookup(txns: list[dict]) -> dict[str, dict]` that indexes transactions by `"id"`, but skip any transaction where `"amount"` is 0 or negative.

5. **Nested collection**: Given `data = {"regions": {"EU": [{"id": "T1"}, {"id": "T2"}], "US": [{"id": "T3"}]}}`, write `all_transaction_ids(data: dict) -> list[str]` using a nested comprehension.

6. **Shallow vs deep copy**: Write a function `clone_transaction(tx: dict) -> dict` that returns a fully independent copy (modifying the clone's nested lists must not affect the original). Add an `assert` to verify.

(Solutions: ../solutions/week-02/day-2.md)

## Self-check
- [ ] I can explain why `x in set` is O(1) but `x in list` is O(n)
- [ ] I can write list, dict, and set comprehensions from scratch
- [ ] I can use nested comprehensions to flatten one level of nesting
- [ ] I can explain the difference between shallow and deep copy for nested dicts
- [ ] I can use `defaultdict` to group items without a key-existence check

## Supplementary links
- [Python set operations](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset)
- [Comprehension expressions (PEP 202, 274)](https://peps.python.org/pep-0202/)
- [copy — Shallow and deep copy operations](https://docs.python.org/3/library/copy.html)
