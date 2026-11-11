# Reference Implementation — CLI Expense Tracker

> Complete, production-quality solution for the Week 1 project. Study this after building your own version.

---

## expense_tracker.py

```python
"""CLI Expense Tracker — Week 1 reference implementation.

Usage:
  python expense_tracker.py add <amount> <category> "<description>"
  python expense_tracker.py list
  python expense_tracker.py total
  python expense_tracker.py filter <category>
  python expense_tracker.py highest
  python expense_tracker.py average
"""
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATA_FILE = Path("expenses.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Expense(TypedDict):
    """A single expense record.

    TypedDict gives static type-checking while remaining a plain dict at
    runtime — no serialisation overhead compared to a dataclass or Pydantic
    model, which is fine for a small CLI tool.
    """
    amount: float
    category: str
    description: str
    date: str          # ISO-8601, e.g. "2026-11-11"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_expenses() -> list[Expense]:
    """Load expenses from DATA_FILE; return [] if it does not exist."""
    if not DATA_FILE.exists():
        logging.debug("No data file at %s — starting with empty list", DATA_FILE)
        return []
    data: list[Expense] = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    logging.debug("Loaded %d expense(s) from %s", len(data), DATA_FILE)
    return data


def save_expenses(expenses: list[Expense]) -> None:
    """Persist the expense list to DATA_FILE as formatted JSON."""
    DATA_FILE.write_text(json.dumps(expenses, indent=2), encoding="utf-8")
    logging.debug("Saved %d expense(s) to %s", len(expenses), DATA_FILE)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def add_expense(
    expenses: list[Expense],
    amount: float,
    category: str,
    description: str,
) -> None:
    """Append a new expense and persist immediately."""
    expense: Expense = {
        "amount": amount,
        "category": category.lower(),           # normalise on write
        "description": description,
        "date": datetime.date.today().isoformat(),
    }
    expenses.append(expense)
    save_expenses(expenses)
    logging.info("Added: %.2f  [%s]  %s", amount, category, description)


def display_all(expenses: list[Expense]) -> None:
    """Print all expenses as a formatted table."""
    if not expenses:
        print("No expenses recorded.")
        return
    header = f"{'#':<4} {'Date':<12} {'Category':<14} {'Amount':>8}  Description"
    print(header)
    print("-" * len(header))
    for i, e in enumerate(expenses, 1):
        print(
            f"{i:<4} {e['date']:<12} {e['category']:<14}"
            f" {e['amount']:>8.2f}  {e['description']}"
        )


def show_total(expenses: list[Expense]) -> None:
    """Print the sum of all expense amounts."""
    total: float = sum(e["amount"] for e in expenses)
    print(f"Total: {total:.2f}  ({len(expenses)} expense(s))")


def filter_category(expenses: list[Expense], category: str) -> None:
    """Display expenses for a given category and their subtotal."""
    target: str = category.lower()
    matches: list[Expense] = [e for e in expenses if e["category"] == target]
    if not matches:
        print(f"No expenses in category '{category}'.")
        return
    display_all(matches)
    subtotal: float = sum(e["amount"] for e in matches)
    print(f"Category total [{category}]: {subtotal:.2f}")


def show_highest(expenses: list[Expense]) -> None:
    """Print the single most expensive entry."""
    if not expenses:
        print("No expenses recorded.")
        return
    top: Expense = max(expenses, key=lambda e: e["amount"])
    print(
        f"Highest: {top['amount']:.2f}  "
        f"[{top['category']}]  {top['description']}  ({top['date']})"
    )


def show_average(expenses: list[Expense]) -> None:
    """Print the mean expense amount."""
    if not expenses:
        print("No expenses recorded.")
        return
    avg: float = sum(e["amount"] for e in expenses) / len(expenses)
    print(f"Average: {avg:.2f}  (over {len(expenses)} expense(s))")


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

USAGE = """\
CLI Expense Tracker

Commands:
  add <amount> <category> "<description>"   Add a new expense
  list                                       Show all expenses
  total                                      Show grand total
  filter <category>                          Filter by category
  highest                                    Show most expensive entry
  average                                    Show mean expense amount

Examples:
  python expense_tracker.py add 12.50 food "lunch at cafe"
  python expense_tracker.py filter food
  python expense_tracker.py total
"""


def main() -> None:
    """Entry point — parse sys.argv and dispatch to the appropriate function."""
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    expenses: list[Expense] = load_expenses()
    command: str = sys.argv[1]

    match command:
        case "add":
            if len(sys.argv) < 5:
                logging.error(
                    "add requires: <amount> <category> \"<description>\""
                )
                sys.exit(1)
            try:
                amount: float = float(sys.argv[2])
            except ValueError:
                logging.error(
                    "Amount must be a number, got: %s", sys.argv[2]
                )
                sys.exit(1)
            if amount <= 0:
                logging.error("Amount must be positive, got: %.2f", amount)
                sys.exit(1)
            add_expense(expenses, amount, sys.argv[3], sys.argv[4])

        case "list":
            display_all(expenses)

        case "total":
            show_total(expenses)

        case "filter":
            if len(sys.argv) < 3:
                logging.error("filter requires: <category>")
                sys.exit(1)
            filter_category(expenses, sys.argv[2])

        case "highest":
            show_highest(expenses)

        case "average":
            show_average(expenses)

        case _:
            logging.error("Unknown command: '%s'", command)
            print(USAGE)
            sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## Sample session

```bash
$ python expense_tracker.py add 12.50 food "lunch at cafe"
INFO: Added: 12.50  [food]  lunch at cafe

$ python expense_tracker.py add 3.20 transport "bus fare"
INFO: Added: 3.20  [transport]  bus fare

$ python expense_tracker.py add 45.00 food "weekly groceries"
INFO: Added: 45.00  [food]  weekly groceries

$ python expense_tracker.py list
#    Date         Category        Amount  Description
----------------------------------------------------------------------
1    2026-11-11   food             12.50  lunch at cafe
2    2026-11-11   transport         3.20  bus fare
3    2026-11-11   food             45.00  weekly groceries

$ python expense_tracker.py total
Total: 60.70  (3 expense(s))

$ python expense_tracker.py filter food
#    Date         Category        Amount  Description
----------------------------------------------------------------------
1    2026-11-11   food             12.50  lunch at cafe
2    2026-11-11   food             45.00  weekly groceries
Category total [food]: 57.50

$ python expense_tracker.py highest
Highest: 45.00  [food]  weekly groceries  (2026-11-11)

$ python expense_tracker.py average
Average: 20.23  (over 3 expense(s))
```

---

## Design decisions explained

| Decision | Reason |
|----------|--------|
| `TypedDict` not `dataclass` | No serialisation boilerplate; JSON round-trips trivially |
| Categories lowercased on write | Queries don't need `.lower()` at read time |
| ISO-8601 date strings | Sorts lexicographically; no `datetime` parsing needed for display |
| `save_expenses` inside `add_expense` | Callers can't forget to save; data is never silently lost |
| `logging` for diagnostics | Errors go to stderr; stdout stays clean for piping |
| `match`/`case` dispatcher | Cleaner than long `if/elif`; requires Python 3.10+ |
| Exit code 1 on error | Standard POSIX convention — scripts in pipelines depend on this |

---

## Extension ideas

```python
# Delete by row number
def delete_expense(expenses: list[Expense], index: int) -> None:
    if not 1 <= index <= len(expenses):
        logging.error("Index %d out of range (1–%d)", index, len(expenses))
        return
    removed = expenses.pop(index - 1)
    save_expenses(expenses)
    logging.info("Deleted: %.2f [%s]", removed["amount"], removed["category"])

# Export to CSV
import csv
def export_csv(expenses: list[Expense], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "category", "amount", "description"])
        writer.writeheader()
        writer.writerows(expenses)
    logging.info("Exported %d expense(s) to %s", len(expenses), path)

# Monthly summary — group by YYYY-MM prefix
from itertools import groupby
def monthly_summary(expenses: list[Expense]) -> None:
    sorted_exp = sorted(expenses, key=lambda e: e["date"][:7])
    for month, group in groupby(sorted_exp, key=lambda e: e["date"][:7]):
        items = list(group)
        total = sum(e["amount"] for e in items)
        print(f"{month}: {total:.2f}  ({len(items)} expense(s))")
```
