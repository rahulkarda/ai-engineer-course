# Week 05 Assessment — Testing, Debugging, Git & Clean Code

> Time: ~1 hr  ·  Pass gate: ≥ 75%

Answer key: ../solutions/week-05/assessment.md

---

## Part 1 — Conceptual (interview-style)  (25%)

Answer each question in 2–5 sentences. Where code is clearer than prose, use code.

1. What is the difference between a unit test and an integration test? Give one example of something a unit test *cannot* catch that an integration test can.

2. Explain what `@pytest.mark.parametrize` does. When would you prefer it over writing three separate test functions?

3. You want to test a function that calls `requests.get(url)` to fetch data. Explain exactly how you would mock that call using `unittest.mock.patch`, and why you must patch it *in the module under test* rather than in the `requests` module itself.

4. Compare `git merge` and `git rebase`. In which scenario is each preferred? Why should you never rebase commits that have already been pushed to a shared branch?

5. What is the purpose of the fixture `scope` parameter in pytest? What side effect might you see if you mistakenly use `scope="session"` for a fixture that mutates state?

6. A junior developer hardcodes `API_KEY = "sk-prod-..."` in `config.py`. List three specific risks this creates and explain how pydantic-settings with a `.env` file solves each one.

7. Describe the Python logging hierarchy. If you create a logger named `src.ledger.accounts` and set its level to `WARNING`, which messages does it emit? Which levels does it suppress? What happens to those records at the root logger?

8. What does "separation of concerns" mean in a layered backend service? Draw or describe a four-layer diagram and identify which layer each of these belongs to: HTTP request parsing, SQL query, business rule validation, Pydantic model definition.

---

## Part 2 — Coding challenge  (40%)

### Specification

Implement a `TransactionLog` class that tracks deposits and withdrawals with the following requirements:

**Functional requirements:**
- `deposit(amount: Decimal, ref: str) -> str` — records a deposit, returns a unique transaction ID (format: `TXN-{n:04d}` where n increments from 1).
- `withdraw(amount: Decimal, ref: str) -> str` — records a withdrawal if funds are sufficient, returns transaction ID.
- `balance() -> Decimal` — current balance (deposits minus withdrawals).
- `history(type_filter: str | None = None) -> list[dict]` — returns transaction dicts. If `type_filter` is `"deposit"` or `"withdraw"`, returns only matching records. Each dict has keys: `id`, `type`, `amount`, `ref`.
- `summary() -> dict` — returns `{"deposits": Decimal, "withdrawals": Decimal, "net": Decimal}`.

**Error handling (use custom exceptions):**
- `deposit` or `withdraw` with `amount <= 0` → `InvalidAmountError`
- `withdraw` when `amount > balance` → `InsufficientFundsError`
- `history` with invalid `type_filter` value → `ValueError`

**Code quality:**
- Use `logging` for all state changes (INFO level).
- `balance()` and `history()` must not expose internal mutable state.
- All config (e.g. a configurable `max_single_transaction` limit) must come from a `Settings` object, not hardcoded.

### Acceptance criteria

Your implementation is complete when all of the following pass:

```python
from decimal import Decimal
from src.transaction_log import TransactionLog
from src.exceptions import InvalidAmountError, InsufficientFundsError

log_obj = TransactionLog()

# IDs increment
id1 = log_obj.deposit(Decimal("100"), ref="payroll")
id2 = log_obj.deposit(Decimal("50"), ref="bonus")
assert id1 == "TXN-0001"
assert id2 == "TXN-0002"

# Balance is correct
assert log_obj.balance() == Decimal("150")

# Withdraw
id3 = log_obj.withdraw(Decimal("30"), ref="rent")
assert log_obj.balance() == Decimal("120")

# History filter
deposits = log_obj.history("deposit")
assert len(deposits) == 2
assert all(d["type"] == "deposit" for d in deposits)

# Summary
s = log_obj.summary()
assert s["deposits"] == Decimal("150")
assert s["withdrawals"] == Decimal("30")
assert s["net"] == Decimal("120")

# Errors
import pytest
with pytest.raises(InvalidAmountError): log_obj.deposit(Decimal("0"), ref="bad")
with pytest.raises(InsufficientFundsError): log_obj.withdraw(Decimal("9999"), ref="big")
```

### Deliverables

- `src/transaction_log.py` — implementation
- `src/exceptions.py` — custom exception hierarchy
- `src/config.py` — settings
- `tests/unit/test_transaction_log.py` — full test suite (≥ 10 tests, parametrized where appropriate)
- `tests/unit/test_transaction_log_errors.py` — all error-path tests
- `.github/workflows/ci.yml` — CI pipeline that runs tests + ruff

---

## Part 3 — Debugging challenge  (20%)

The function below has **three bugs**. Identify each bug, explain what causes it, and write the corrected version.

```python
from decimal import Decimal

def batch_transfer(
    accounts: dict[str, Decimal],
    transfers: list[tuple[str, str, Decimal]],
) -> dict[str, Decimal]:
    """
    Apply a list of (from, to, amount) transfers to account balances.
    Returns the updated balances dict.
    Bug hunt: find all three bugs.
    """
    result = accounts         # Bug A
    for from_acc, to_acc, amount in transfers:
        if from_acc not in result or to_acc not in result:
            raise KeyError(f"unknown account")
        if result[from_acc] < 0:  # Bug B
            raise ValueError("insufficient funds")
        result[from_acc] -= amount
        result[to_acc] -= amount  # Bug C
    return result
```

For each bug:
- State the bug (one sentence).
- Explain the consequence.
- Write the fixed line.

Then write one pytest test per bug that would catch it in regression.

---

## Part 4 — Code quality  (15%)

Review the following function and list every code quality issue (naming, logging, exception handling, SoC, config, type hints). Then rewrite it cleanly.

```python
def do_stuff(x, y, key="abc123"):
    import requests
    try:
        r = requests.get(f"https://api.example.com/rate?from={x}&to={y}", headers={"X-Key": key})
        data = r.json()
        rate = data["rate"]
        print("Got rate:", rate)
        return float(x) * rate
    except:
        print("error")
        return 0
```

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Indicators |
|---|---|---|
| Independent implementation | 40% | `TransactionLog` passes all acceptance criteria; no skeleton code left |
| Conceptual understanding | 25% | Part 1 answers are accurate and specific, not vague |
| Debugging & problem-solving | 20% | All three bugs in Part 3 correctly identified with regression tests |
| Code quality & testing | 15% | ruff clean, type hints, logging, ≥ 10 tests, CI passes |
