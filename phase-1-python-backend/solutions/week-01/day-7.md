# Solutions — Phase 1 · Week 01 · Day 7 — Expense Tracker Build Steps

> Guidance notes for each build step. Full reference implementation: `../week-01/project.md` (solutions copy: `project.md` in this folder).

---

## Step 1 — Data model + add + persistence

```python
"""expense_tracker.py — Step 1: data model, persistence, add."""
import datetime
import json
import logging
from pathlib import Path
from typing import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_FILE = Path("expenses.json")


class Expense(TypedDict):
    amount: float
    category: str
    description: str
    date: str


def load_expenses() -> list[Expense]:
    """Load expenses from disk; return empty list if file doesn't exist."""
    if not DATA_FILE.exists():
        logging.debug("No data file found at %s — starting fresh", DATA_FILE)
        return []
    data: list[Expense] = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    logging.debug("Loaded %d expenses from %s", len(data), DATA_FILE)
    return data


def save_expenses(expenses: list[Expense]) -> None:
    """Persist the expenses list to disk as JSON."""
    DATA_FILE.write_text(json.dumps(expenses, indent=2), encoding="utf-8")
    logging.debug("Saved %d expenses to %s", len(expenses), DATA_FILE)


def add_expense(
    expenses: list[Expense],
    amount: float,
    category: str,
    description: str,
) -> None:
    """Append a new expense and save."""
    expense: Expense = {
        "amount": amount,
        "category": category.lower(),   # normalise: "Food" == "food"
        "description": description,
        "date": datetime.date.today().isoformat(),  # ISO-8601: "2026-11-11"
    }
    expenses.append(expense)
    save_expenses(expenses)
    logging.info("Added: %.2f  [%s]  %s", amount, category, description)
```

**Design notes:**
- `TypedDict` gives static type-checking without runtime overhead — it is just a dict at runtime.
- Categories are normalised to lowercase on entry so `filter("Food")` and `filter("food")` behave identically.
- ISO-8601 date strings sort lexicographically, which makes future date-range queries trivial.

---

## Step 2 — Display and total

```python
def display_expenses(expenses: list[Expense]) -> None:
    """Pretty-print all expenses as a formatted table."""
    if not expenses:
        print("No expenses recorded.")
        return
    print(f"{'#':<4} {'Date':<12} {'Category':<14} {'Amount':>8}  Description")
    print("-" * 60)
    for i, e in enumerate(expenses, 1):
        print(
            f"{i:<4} {e['date']:<12} {e['category']:<14}"
            f" {e['amount']:>8.2f}  {e['description']}"
        )


def total_expenses(expenses: list[Expense]) -> float:
    """Return the sum of all expense amounts."""
    return sum(e["amount"] for e in expenses)
```

**Design notes:**
- `enumerate(expenses, 1)` starts the counter at 1 (user-visible row numbers). The internal list is still 0-indexed.
- `:<14` is left-aligned in a 14-character field; `:>8.2f` is right-aligned, fixed-point, 2 decimal places.
- `sum()` on a generator expression is efficient — it does not build an intermediate list.

---

## Step 3 — Filter, highest, average

```python
def filter_by_category(expenses: list[Expense], category: str) -> list[Expense]:
    """Return expenses whose category matches (case-insensitive)."""
    target: str = category.lower()
    return [e for e in expenses if e["category"] == target]


def highest_expense(expenses: list[Expense]) -> Expense | None:
    """Return the single most expensive expense, or None if the list is empty."""
    return max(expenses, key=lambda e: e["amount"]) if expenses else None


def average_expense(expenses: list[Expense]) -> float:
    """Return the mean expense amount, or 0.0 if the list is empty."""
    if not expenses:
        return 0.0
    return sum(e["amount"] for e in expenses) / len(expenses)
```

**Design notes:**
- `| None` return type (Python 3.10+ union syntax) signals that callers must handle the empty case.
- `max(..., key=lambda e: e["amount"])` is O(n) — one pass through the list.
- Returning `0.0` for empty average is a pragmatic choice for a CLI tool; a production API might raise `ValueError` instead.

---

## Step 4 — CLI dispatcher skeleton

```python
import sys

USAGE = """Usage:
  python expense_tracker.py add <amount> <category> "<description>"
  python expense_tracker.py list
  python expense_tracker.py total
  python expense_tracker.py filter <category>
  python expense_tracker.py highest
  python expense_tracker.py average
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    expenses: list[Expense] = load_expenses()
    command: str = sys.argv[1]

    match command:
        case "add":
            if len(sys.argv) < 5:
                logging.error("add requires: <amount> <category> \"<description>\"")
                sys.exit(1)
            try:
                amount: float = float(sys.argv[2])
            except ValueError:
                logging.error("Amount must be a number, got: %s", sys.argv[2])
                sys.exit(1)
            add_expense(expenses, amount, sys.argv[3], sys.argv[4])
        case "list":
            display_expenses(expenses)
        case "total":
            print(f"Total: {total_expenses(expenses):.2f}")
        case "filter":
            if len(sys.argv) < 3:
                logging.error("filter requires: <category>")
                sys.exit(1)
            matches = filter_by_category(expenses, sys.argv[2])
            display_expenses(matches)
        case "highest":
            top = highest_expense(expenses)
            if top:
                print(f"Highest: {top['amount']:.2f}  [{top['category']}]  {top['description']}")
            else:
                print("No expenses recorded.")
        case "average":
            print(f"Average: {average_expense(expenses):.2f}")
        case _:
            logging.error("Unknown command: %s", command)
            print(USAGE)
            sys.exit(1)


if __name__ == "__main__":
    main()
```

**Design notes:**
- `match`/`case` (Python 3.10+ structural pattern matching) is cleaner than a chain of `if/elif`.
- `sys.exit(1)` signals failure to the shell (exit code 0 = success, non-zero = error). Scripts in pipelines depend on this.
- All user-facing error messages go through `logging.error` so they appear on stderr, not stdout. This keeps `stdout` clean for piping.

---

## Step 5 — Wiring it all together

See the complete annotated reference in `project.md` (this solutions folder). The key integration points:
1. `load_expenses()` is called once at the top of `main()`.
2. `save_expenses()` is called inside `add_expense()` — the caller never needs to remember to save.
3. Every command branch validates its own argument count before touching the data.
4. The `__name__ == "__main__"` guard prevents `main()` from running on import (important when you add tests).

---

## Common mistakes and fixes

| Mistake | Fix |
|---------|-----|
| `float("abc")` crashes without `try/except` | Wrap in `try/except ValueError` |
| Category comparison is case-sensitive | Normalise with `.lower()` on input AND on lookup |
| File written but not readable by next run | Always use `encoding="utf-8"` in both `write_text` and `read_text` |
| `expenses` list mutated but not saved | `save_expenses()` called inside the mutating function, not the caller |
| `max()` on empty list raises `ValueError` | Guard with `if expenses else None` before calling `max()` |
