# Week 02 Assessment — Python Data Structures and Functions

> Time: 60 minutes  ·  Pass gate: 75%

---

## Conceptual (interview-style) — 25 points

Answer each question concisely. One to three sentences per answer is sufficient.

1. What is the average-case time complexity of `x in my_dict` vs `x in my_list`? Explain why they differ.

2. Explain Python's LEGB rule. What happens when you assign to a variable inside a function that has the same name as a module-level variable?

3. What is wrong with this function signature, and how do you fix it?
   ```python
   def register(user: str, roles: list[str] = []) -> None:
       roles.append("viewer")
       ...
   ```

4. `map()` and a list comprehension can both transform a list. Name one situation where you would prefer `map`, and one where you would prefer a comprehension.

5. What does `raise ValueError("bad input") from e` accomplish that `raise ValueError("bad input")` alone does not?

6. You need to store a set of `(transaction_id, amount)` pairs for deduplication. Would you use a `list`, `set`, or `dict`? Justify your choice.

7. What does `sorted()` guarantee about the relative order of elements that compare as equal? What is the practical benefit?

---

## Coding challenge — 40 points

Implement the following module. You may not import third-party libraries (standard library only).

### Specification

```python
# challenge.py
from __future__ import annotations
from pathlib import Path
from typing import Any


class TransactionValidationError(ValueError):
    """Raised when a transaction record fails validation."""


def validate(tx: Any) -> dict:
    """
    Validate a single transaction dict.
    Required fields: id (str), amount (float > 0), status (str in VALID_STATUSES).
    Raises TransactionValidationError with a descriptive message on failure.
    """


VALID_STATUSES = {"success", "failed", "pending", "refunded"}


def load_and_validate(path: Path) -> tuple[list[dict], list[str]]:
    """
    Load a JSON file containing a list of transactions.
    Returns (valid_transactions, error_messages).
    Each error message must include the row index.
    File-level errors (missing file, bad JSON) raise FileNotFoundError or ValueError.
    """


def summarize(transactions: list[dict]) -> dict[str, Any]:
    """
    Return a summary dict with these exact keys:
      - total_count: int
      - total_amount: float (sum of all amounts, rounded to 2 dp)
      - successful_amount: float (sum for status == "success", rounded to 2 dp)
      - by_status: dict[str, int]  (count per status)
      - duplicate_ids: list[str]   (IDs appearing more than once, sorted)
      - top_3_by_amount: list[dict] (three highest-amount transactions, desc)
    Works correctly on an empty list.
    """


def save(transactions: list[dict], path: Path) -> None:
    """Write transactions to a JSON file with indent=2."""
```

### Acceptance criteria
- `validate` raises `TransactionValidationError` for missing fields, wrong types, and `amount <= 0`.
- `load_and_validate` returns all valid rows and one error string per invalid row; does not raise for row-level errors.
- `summarize` returns correct values for an empty list, a single transaction, and 100 generated transactions.
- `save` produces valid JSON that can be reloaded with `json.load`.
- All four functions are covered by at least one `pytest` test each.

---

## Debugging challenge — 20 points

The following code has **four bugs**. Identify each bug, explain what it does wrong, and write the corrected version.

```python
import json

def process(file_path):
    f = open(file_path)
    data = json.load(f)
    
    results = []
    seen = []
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
        totals[status] = totals[status] + tx["amount"]
    
    return sorted(results, key=tx["amount"])
```

---

## Code quality — 15 points

Refactor this working but low-quality function. Do not change its behavior.

```python
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
```

Your refactored version must:
- Have a descriptive function name and type annotations.
- Use a comprehension or `filter` instead of a manual index loop.
- Use a helper function or expression instead of the if/else for `tier`.
- Be no longer than 10 lines of non-blank, non-comment code.

---

## Rubric

| Dimension | Weight | Points |
|-----------|--------|--------|
| Independent implementation (coding challenge) | 40% | 40 |
| Conceptual understanding (interview questions) | 25% | 25 |
| Debugging & problem-solving | 20% | 20 |
| Code quality & refactoring | 15% | 15 |
| **Total** | 100% | **100** |

**Pass gate: 75 / 100**

Answer key: ../solutions/week-02/assessment.md
