# Phase 1 · Week 01 · Day 7 — CLI Expense Tracker (Project Build Day)

> Objective: Build a complete CLI expense tracker from scratch, integrating every topic from Week 1.  ·  Time: ~3 hr build + 1 hr assessment  ·  Prereqs: [[day-6]]

---

## Project overview and learning goals

Today you build a real, working tool. No toy exercises — a command-line program you could actually use.

By the end you will have applied:
- Variables, types, type conversion (Day 1)
- Operators, mutability, dicts vs. immutable records (Day 2)
- Conditionals and loops (Day 3)
- Strings, f-strings, slicing (Day 4)
- Functions, I/O, debugging, `logging` (Days 5–6)

The tracker runs from the terminal like this:

```
python expense_tracker.py add 12.50 food "Lunch at cafe"
python expense_tracker.py list
python expense_tracker.py total
python expense_tracker.py filter food
python expense_tracker.py highest
python expense_tracker.py average
```

---

## Architecture walkthrough

### Data model decision

We need to store each expense as a record with four fields: `amount`, `category`, `description`, `date`. Python gives us several options:

| Option | Pros | Cons |
|---|---|---|
| Plain `dict` | Simple, JSON-serialisable | No field guarantees, typos silently create new keys |
| `TypedDict` | Type-checked by mypy/pyright, still a dict at runtime | Slightly more ceremony |
| `dataclass` | Clean attribute access, `@dataclass` handles `__init__` | Needs extra step to serialise to JSON |
| `namedtuple` | Immutable, readable | Immutability is a mismatch for records you might edit |

Best fit for this stage: **`TypedDict`**. It is a regular dict at runtime (easy JSON round-trip) and tells type checkers exactly what fields exist.

```python
from typing import TypedDict

class Expense(TypedDict):
    amount: float
    category: str
    description: str
    date: str  # ISO format YYYY-MM-DD
```

### Storage

All expenses live in a Python `list[Expense]` in memory. On every mutation (add) we write the whole list to `expenses.json`. On startup we read it back. This is "load-process-save" — the simplest persistence pattern.

```
expenses.json   <-->   list[Expense] in RAM   <-->   CLI commands
```

`pathlib.Path` for file access, `json` module for serialisation. No third-party packages needed.

### Program flow

```
main()
  |
  +-- load_expenses(path) -> list[Expense]
  |
  +-- dispatch on sys.argv[1]
  |     "add"     -> add_expense(...)
  |     "list"    -> display_expenses(...)
  |     "total"   -> show_total(...)
  |     "filter"  -> filter_by_category(...)
  |     "highest" -> show_highest(...)
  |     "average" -> show_average(...)
  |
  +-- save_expenses(path, expenses)  [only after mutations]
```

Each command is a separate function. `main()` owns the dispatch table and I/O plumbing. Commands receive the list and return nothing (they print their own output) or return a modified list.

### File layout

```
week-01/
    expense_tracker.py    # single file, <= 150 lines
    expenses.json         # created automatically on first run
```

---

## Starter scaffold

Copy this into `expense_tracker.py` and fill in every `TODO`.

```python
"""CLI Expense Tracker — Week 1 project."""

from __future__ import annotations

import json
import logging
import sys
import datetime
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
DATA_FILE = Path("expenses.json")


class Expense(TypedDict):
    amount: float
    category: str
    description: str
    date: str  # ISO format YYYY-MM-DD


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_expenses(path: Path) -> list[Expense]:
    """Load expenses from JSON file; return empty list if file missing."""
    if not path.exists():
        logger.debug("No data file found at %s — starting fresh.", path)
        return []
    # TODO: open the file, parse JSON, return list[Expense]
    # Hint: json.loads(path.read_text())
    raise NotImplementedError


def save_expenses(path: Path, expenses: list[Expense]) -> None:
    """Persist expense list to JSON file."""
    logger.debug("Saving %d expenses to %s", len(expenses), path)
    # TODO: serialise expenses to JSON and write to path
    # Hint: path.write_text(json.dumps(expenses, indent=2))
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def add_expense(
    expenses: list[Expense],
    amount: float,
    category: str,
    description: str,
) -> list[Expense]:
    """Create a new Expense and append it to the list."""
    today = datetime.date.today().isoformat()
    # TODO: build the Expense dict and append to expenses
    # TODO: log at INFO: "Added expense: ..."
    raise NotImplementedError


def display_expenses(expenses: list[Expense]) -> None:
    """Print a formatted table of all expenses."""
    if not expenses:
        print("No expenses recorded.")
        return
    # TODO: print a header row, then one row per expense
    # Suggested columns: #  Date        Category   Amount   Description
    # Hint: f"{i+1:<4} {e['date']:<12} {e['category']:<12} {e['amount']:>8.2f}  {e['description']}"
    raise NotImplementedError


def show_total(expenses: list[Expense]) -> None:
    """Print the sum of all expense amounts."""
    if not expenses:
        print("No expenses recorded.")
        return
    # TODO: compute total and print
    raise NotImplementedError


def filter_by_category(expenses: list[Expense], category: str) -> None:
    """Print expenses matching category (case-insensitive)."""
    # TODO: filter, then reuse display_expenses on the filtered list
    raise NotImplementedError


def show_highest(expenses: list[Expense]) -> None:
    """Print the single expense with the largest amount."""
    if not expenses:
        print("No expenses recorded.")
        return
    # TODO: find max by amount and print it
    # Hint: max(expenses, key=lambda e: e["amount"])
    raise NotImplementedError


def show_average(expenses: list[Expense]) -> None:
    """Print the mean expense amount."""
    if not expenses:
        print("No expenses recorded.")
        return
    # TODO: compute and print average
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    expenses = load_expenses(DATA_FILE)

    if len(sys.argv) < 2:
        print("Usage: python expense_tracker.py <command> [args]")
        print("Commands: add <amount> <category> <description>")
        print("          list | total | filter <category> | highest | average")
        sys.exit(1)

    command = sys.argv[1]

    if command == "add":
        if len(sys.argv) < 5:
            print("Usage: add <amount> <category> <description>")
            sys.exit(1)
        try:
            amount = float(sys.argv[2])
        except ValueError:
            print(f"Error: '{sys.argv[2]}' is not a valid amount.")
            sys.exit(1)
        category = sys.argv[3]
        description = sys.argv[4]
        expenses = add_expense(expenses, amount, category, description)
        save_expenses(DATA_FILE, expenses)

    elif command == "list":
        display_expenses(expenses)

    elif command == "total":
        show_total(expenses)

    elif command == "filter":
        if len(sys.argv) < 3:
            print("Usage: filter <category>")
            sys.exit(1)
        filter_by_category(expenses, sys.argv[2])

    elif command == "highest":
        show_highest(expenses)

    elif command == "average":
        show_average(expenses)

    else:
        print(f"Unknown command: '{command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Build steps

Work through each step in order. Each milestone produces a runnable program — test it before moving on.

### Step 1 — Data model + add expense (~30 min)

Implement `load_expenses`, `save_expenses`, and `add_expense`.

After this step:
```
python expense_tracker.py add 12.50 food "Lunch at cafe"
# INFO: Added expense: 12.50 food Lunch at cafe
python expense_tracker.py add 3.00 transport "Bus fare"
# INFO: Added expense: 3.00 transport Bus fare
```
`expenses.json` should now exist and contain two records.

Key things to verify:
- `expenses.json` is created on first run
- Running `add` twice accumulates (not overwrites) entries
- The `date` field is today in `YYYY-MM-DD` format

### Step 2 — Display + total (~30 min)

Implement `display_expenses` and `show_total`.

After this step:
```
python expense_tracker.py list
#   #  Date        Category     Amount  Description
#   1  2024-01-15  food          12.50  Lunch at cafe
#   2  2024-01-15  transport      3.00  Bus fare

python expense_tracker.py total
# Total: 15.50
```

Formatting tip: use f-string alignment specifiers to line up columns:
- `:<12` — left-align in 12-char field
- `:>8.2f` — right-align float with 2 decimal places

### Step 3 — Filter + highest + average (~30 min)

Implement `filter_by_category`, `show_highest`, `show_average`.

After this step:
```
python expense_tracker.py filter food
# shows only food expenses

python expense_tracker.py highest
# Highest: 12.50 — food — Lunch at cafe (2024-01-15)

python expense_tracker.py average
# Average: 7.75
```

For `filter_by_category`: compare `.lower()` on both sides so `Filter FOOD` matches `food`.

### Step 4 — Persistence (load/save JSON) (~30 min)

You likely already wired this in Step 1. Verify the full round-trip:

1. Close your terminal completely.
2. Reopen and run `python expense_tracker.py list`.
3. Expenses should still be there.

If they vanished: debug `load_expenses`. Add `logger.debug` calls to trace what is read back.

Edge cases to handle:
- `expenses.json` missing: return `[]`
- `expenses.json` contains `[]` (empty array): return `[]`, don't crash

### Step 5 — Polish (~30 min)

Run through this checklist:

- [ ] Every function has a type-annotated signature
- [ ] `logging.basicConfig` level can be switched to `DEBUG` to see file I/O messages
- [ ] Invalid amount (`python expense_tracker.py add abc food "test"`) shows a friendly error, not a traceback
- [ ] Running with no arguments prints usage, exits with code 1
- [ ] Unknown command shows a helpful message
- [ ] Column headers in `list` output are bold or clearly separated (use `"-" * 50` or similar)
- [ ] Run `ruff check expense_tracker.py` — fix any issues

---

## Testing your implementation

Run these commands in sequence and verify the expected outputs:

```bash
# Start fresh
rm -f expenses.json

# 1. Add expenses
python expense_tracker.py add 12.50 food "Lunch at cafe"
python expense_tracker.py add 3.00 transport "Bus fare"
python expense_tracker.py add 45.00 shopping "New shoes"
python expense_tracker.py add 8.75 food "Dinner takeaway"

# 2. List all
python expense_tracker.py list
# Expected: 4 rows, sorted by insertion order

# 3. Total
python expense_tracker.py total
# Expected: Total: 69.25

# 4. Filter
python expense_tracker.py filter food
# Expected: 2 rows (12.50 and 8.75)

# 5. Highest
python expense_tracker.py highest
# Expected: 45.00 shopping New shoes

# 6. Average
python expense_tracker.py average
# Expected: Average: 17.31 (or 17.3125 — your call on rounding)

# 7. Error handling
python expense_tracker.py add abc food "bad amount"
# Expected: error message, no crash, no file written

# 8. Empty state
rm -f expenses.json
python expense_tracker.py list
# Expected: "No expenses recorded."
python expense_tracker.py total
# Expected: "No expenses recorded."
```

---

## Self-check

- [ ] I can explain why `TypedDict` is a dict at runtime and how that helps with JSON serialisation
- [ ] I can implement `load_expenses` and `save_expenses` without reference
- [ ] I can use `max(iterable, key=...)` to find the highest expense
- [ ] I can format a table in the terminal using f-string alignment specifiers
- [ ] I understand why `sys.argv` is a `list[str]` and how to convert `argv[2]` to float safely
- [ ] My `expense_tracker.py` passes `ruff check` with no errors
- [ ] All 8 manual test cases above produce the expected output

---

After finishing the build, work through the **Week 1 Assessment** (`assessment.md`).

Reference implementation: ../solutions/week-01/day-7.md
