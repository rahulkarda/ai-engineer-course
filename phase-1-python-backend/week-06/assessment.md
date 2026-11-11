# Week 06 Assessment — SQL, PostgreSQL, and Database Engineering

> Time: ~90 min  ·  Pass gate: 75%

## Conceptual (interview-style) — 25 points

Answer each question in 2–4 sentences. No code required unless the question asks for it.

1. Explain the logical execution order of a SQL SELECT statement (FROM through LIMIT). Why does this order matter when writing queries?

2. What is the difference between `INNER JOIN` and `LEFT JOIN`? Give a scenario in the financial app where you would use each.

3. Explain MVCC (Multi-Version Concurrency Control). Why does it allow readers and writers to not block each other?

4. What is a **covering index**? How does `INCLUDE (...)` in PostgreSQL differ from adding those columns to the key portion of the index?

5. Compare the four SQL isolation levels (READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE). Which anomalies does each prevent? What is the performance trade-off of SERIALIZABLE?

6. What is the difference between `session.flush()` and `session.commit()` in SQLAlchemy? When would you call `flush()` without immediately calling `commit()`?

7. Explain the three normal forms (1NF, 2NF, 3NF). Give an example of a table that violates 2NF and show how to fix it.

---

## Coding Challenge — 40 points

### Scenario

You are building the reporting layer of the Financial Transaction Management System. All tables from the week's schema are available (`customers`, `accounts`, `transactions`, `transaction_categories`, `audit_logs`).

### Task A — SQL (20 pts)

Write a single SQL query that produces a **monthly spending report** per customer for the last 6 calendar months. Requirements:

- Columns: `customer_id`, `full_name`, `month` (YYYY-MM), `total_debit`, `total_credit`, `net` (credit − debit), `rank_this_month` (rank by `total_debit` DESC within each month)
- Only include customers who have at least one transaction in the period
- Order by `month DESC`, then `rank_this_month ASC`
- Use at least one CTE and at least one window function

Acceptance criteria:
- Query runs without error on the seeded database
- `rank_this_month` resets correctly per month
- `net` can be negative (more debits than credits)

### Task B — Python / SQLAlchemy (20 pts)

Implement `TransactionService.get_account_summary` in `src/financial_app/service.py`:

```python
from dataclasses import dataclass
from decimal import Decimal
from datetime import date

from sqlalchemy.orm import Session


@dataclass
class AccountSummary:
    account_id: int
    account_number: str
    account_type: str
    current_balance: Decimal
    total_credits_30d: Decimal
    total_debits_30d: Decimal
    tx_count_30d: int
    last_transaction_date: date | None


class TransactionService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_account_summary(self, customer_id: int) -> list[AccountSummary]:
        """
        Return a summary for each account owned by customer_id.
        The 30-day totals cover the last 30 calendar days (inclusive of today).
        Use a single database round-trip (one query).
        """
        ...
```

Requirements:
- Single query (no N+1 queries)
- Use SQLAlchemy Core or ORM with `func.sum`, `func.count`, `func.max`
- Return an empty list if the customer has no accounts
- All monetary values as `Decimal`, not `float`

Write at least two pytest tests:
- One verifying correct 30-day totals
- One verifying empty list for a customer with no accounts

---

## Debugging Challenge — 20 points

The following query runs correctly but is extremely slow on a table with 10 million transactions:

```sql
SELECT c.full_name, SUM(t.amount) AS total
FROM transactions t
JOIN accounts a ON a.account_id = t.account_id
JOIN customers c ON c.customer_id = a.customer_id
WHERE EXTRACT(YEAR FROM t.created_at) = 2024
GROUP BY c.customer_id, c.full_name
ORDER BY total DESC
LIMIT 20;
```

1. Identify at least two performance problems with this query.
2. Rewrite it to be index-friendly.
3. Write the `CREATE INDEX` statement(s) that would most improve performance.
4. Paste the `EXPLAIN (ANALYZE)` output for both versions and explain what changed.

---

## Code Quality — 15 points

Evaluated on your Task B implementation:

- Type hints on all functions and class attributes
- No magic numbers/strings (use constants or Pydantic models)
- Proper logging (not `print`)
- Docstrings on public methods
- Tests use fixtures, not setup/teardown in test bodies
- No secrets or hardcoded connection strings

---

## Rubric

| Dimension | Weight | Full marks criteria |
|-----------|--------|---------------------|
| Independent implementation | 40% | Both Task A and B complete and correct |
| Conceptual understanding | 25% | 6/7 conceptual questions answered accurately |
| Debugging & problem-solving | 20% | Both problems identified; rewrite is faster |
| Code quality & testing | 15% | Type hints, logging, ≥2 passing tests |

**Pass gate: 75% (≥75 points out of 100)**

---

Answer key: ../../solutions/week-06/assessment.md
