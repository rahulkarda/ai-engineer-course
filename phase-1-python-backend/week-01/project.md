# Week 01 Project — CLI Expense Tracker

> Integrates: variables, types, operators, conditionals, loops, strings, I/O, logging, pathlib, JSON  ·  Target: ~3 hr build

Reference implementation: ../solutions/week-01/project.md

---

## Overview

Build a command-line expense tracker in pure Python. No frameworks, no database, no third-party packages. Data persists in a local JSON file. This is a full tool you could actually use day-to-day.

The point is not novelty — it is integration. Every topic from this week appears in this one file.

---

## Requirements

### Functional requirements

| Command | Behaviour |
|---|---|
| `add <amount> <category> <description>` | Create a new expense; persist to `expenses.json` |
| `list` | Display all expenses in a formatted table |
| `total` | Print the sum of all expense amounts |
| `filter <category>` | Display expenses matching the given category (case-insensitive) |
| `highest` | Print the single expense with the largest amount |
| `average` | Print the mean expense amount |

Persistence rules:
- Load `expenses.json` on every invocation (if it does not exist, start with an empty list)
- Write `expenses.json` after every mutation (`add`)
- The file must be valid JSON at all times

### Non-functional requirements

- **Type hints** on every function signature and variable where the type is not immediately obvious
- **`logging`** for all non-display output: `INFO` for user-facing operations, `DEBUG` for file I/O
- **`pathlib.Path`** for all file access — no `open()` with raw string paths
- **Error handling**: invalid amount, missing file, empty list, unknown command — all handled gracefully
- **Single file** `expense_tracker.py`, 150 lines or fewer
- **`ruff check expense_tracker.py`** passes with no errors

### Data model

```python
from typing import TypedDict

class Expense(TypedDict):
    amount: float
    category: str
    description: str
    date: str  # ISO format: YYYY-MM-DD
```

Use `datetime.date.today().isoformat()` to auto-fill `date` when an expense is added.

---

## CLI interface

```
python expense_tracker.py add 12.50 food "Lunch at cafe"
python expense_tracker.py add 3.00 transport "Bus fare"
python expense_tracker.py list
python expense_tracker.py total
python expense_tracker.py filter food
python expense_tracker.py highest
python expense_tracker.py average
```

`sys.argv` layout for `add`:
```
sys.argv[0]  = "expense_tracker.py"
sys.argv[1]  = "add"
sys.argv[2]  = "12.50"     # must convert to float; catch ValueError
sys.argv[3]  = "food"
sys.argv[4]  = "Lunch at cafe"
```

---

## Acceptance criteria

All of the following must pass before you consider the project done:

- [ ] `add` stores the expense and `expenses.json` is updated on disk
- [ ] `list` shows a formatted table with columns: `#`, `Date`, `Category`, `Amount`, `Description`
- [ ] `total` prints the correct sum formatted to 2 decimal places
- [ ] `filter food` shows only food expenses; `filter FOOD` also works (case-insensitive)
- [ ] `highest` prints the expense record with the maximum amount
- [ ] `average` prints the mean amount formatted to 2 decimal places
- [ ] Running any read command on an empty or missing `expenses.json` prints "No expenses recorded." and exits cleanly
- [ ] `python expense_tracker.py add abc food "test"` prints a friendly error and exits with code 1 — no traceback
- [ ] `python expense_tracker.py` (no arguments) prints usage and exits with code 1
- [ ] `python expense_tracker.py unknown` prints "Unknown command: 'unknown'" and exits with code 1
- [ ] `ruff check expense_tracker.py` reports no errors

---

## Getting started

### 1. Create the file

```
week-01/
    expense_tracker.py
```

### 2. Skeleton structure

```python
"""CLI Expense Tracker — Week 1 project."""

from __future__ import annotations

import json
import logging
import sys
import datetime
from pathlib import Path
from typing import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_FILE = Path("expenses.json")

class Expense(TypedDict):
    amount: float
    category: str
    description: str
    date: str

def load_expenses(path: Path) -> list[Expense]: ...
def save_expenses(path: Path, expenses: list[Expense]) -> None: ...
def add_expense(expenses: list[Expense], amount: float, category: str, description: str) -> list[Expense]: ...
def display_expenses(expenses: list[Expense]) -> None: ...
def show_total(expenses: list[Expense]) -> None: ...
def filter_by_category(expenses: list[Expense], category: str) -> None: ...
def show_highest(expenses: list[Expense]) -> None: ...
def show_average(expenses: list[Expense]) -> None: ...
def main() -> None: ...

if __name__ == "__main__":
    main()
```

### 3. Build order (recommended)

Build in this order so you can test each piece before adding the next:

1. `load_expenses` + `save_expenses` — verify JSON round-trip manually
2. `add_expense` + `main()` dispatch for `add` — test with `list` of real runs
3. `display_expenses` — verify table formatting
4. `show_total` — one-liner using `sum()`
5. `filter_by_category` — reuse `display_expenses` on a filtered list
6. `show_highest` — use `max()` with a key function
7. `show_average` — use `sum()` / `len()`
8. Error handling pass — wrap `float()` call, check `len(sys.argv)`, handle unknown command

### 4. Useful stdlib patterns

**Reading JSON:**
```python
data = json.loads(path.read_text(encoding="utf-8"))
```

**Writing JSON:**
```python
path.write_text(json.dumps(expenses, indent=2), encoding="utf-8")
```

**Finding max by key:**
```python
priciest = max(expenses, key=lambda e: e["amount"])
```

**Case-insensitive filter:**
```python
matched = [e for e in expenses if e["category"].lower() == category.lower()]
```

**F-string column alignment:**
```python
# left-align in 12-char field, right-align float in 8-char field to 2dp
f"{e['category']:<12} {e['amount']:>8.2f}"
```

**Safe float conversion:**
```python
try:
    amount = float(sys.argv[2])
except ValueError:
    print(f"Error: '{sys.argv[2]}' is not a valid amount.", file=sys.stderr)
    sys.exit(1)
```

---

## Extension challenges

Complete the core project first, then attempt these if you want more practice:

1. **`delete <index>`** — Remove the expense at 1-based index from `list` output. Handle out-of-range index.

2. **`export csv`** — Write all expenses to `expenses.csv` using the `csv` module. Include a header row.

3. **Monthly summary** — New command `summary` that groups expenses by `YYYY-MM` and prints total per category per month.

4. **Budget warnings** — New command `budget set <category> <limit>` stores a per-category limit in a separate `budget.json`. Warn at `add` time when the running category total exceeds the limit.

5. **`search <keyword>`** — Filter expenses where the description contains the keyword (case-insensitive).
