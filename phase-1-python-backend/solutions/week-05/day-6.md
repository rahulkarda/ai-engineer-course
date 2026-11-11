# Solutions — Week 05 · Day 6 (Saturday Lab)

## Exercise 1 — Integration test with SQLite

```python
# tests/integration/test_persistence.py
"""Integration test: write and read transactions from a real SQLite DB."""
import pytest
import sqlite3
from decimal import Decimal
from pathlib import Path


@pytest.fixture
def sqlite_db(tmp_path: Path):
    """Create a fresh SQLite DB for the test, cleaned up after."""
    db_path = tmp_path / "test_ledger.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE transactions "
        "(id INTEGER PRIMARY KEY, owner TEXT, type TEXT, amount TEXT)"
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.mark.integration
def test_persist_and_reload_transactions(sqlite_db):
    sqlite_db.execute(
        "INSERT INTO transactions (owner, type, amount) VALUES (?, ?, ?)",
        ("alice", "deposit", "100.00"),
    )
    sqlite_db.commit()

    row = sqlite_db.execute(
        "SELECT owner, type, amount FROM transactions WHERE owner=?", ("alice",)
    ).fetchone()

    assert row[0] == "alice"
    assert row[1] == "deposit"
    assert Decimal(row[2]) == Decimal("100.00")
```

Run only integration tests: `pytest -m integration`
Run only unit tests: `pytest -m "not integration"`

## Exercise 2 — Verify CI fails below 80% coverage

```yaml
# In ci.yml:
- name: Run tests
  run: pytest tests/unit/ --cov=src --cov-fail-under=80
```

To verify: delete `tests/unit/test_invalid_inputs.py` temporarily, push, observe the Actions run fails with exit code 2 ("Coverage failure"). Restore the file and push again to confirm green.

## Exercise 3 — Fixture that skips if real API key present

```python
# tests/conftest.py
import pytest
from src.config import settings


@pytest.fixture(scope="session", autouse=True)
def guard_against_real_api_key():
    """Prevent tests from running if APP_API_KEY looks like a real key."""
    key = settings.app_api_key
    # A real key would be non-empty and longer than a placeholder
    if key and len(key) > 10:
        pytest.skip(
            f"APP_API_KEY appears to be a real credential (length={len(key)}). "
            "Unset it before running unit tests."
        )
```

This protects against accidentally running unit tests in a production environment where the env var is set to a real key.

## Exercise 4 — ruff bugbear violations and fixes

Common `B` violations found in typical transaction code:

```python
# B006 — mutable default argument
def add_fee(transactions: list = []):  # WRONG — list shared across calls
    ...

# Fixed:
def add_fee(transactions: list | None = None):
    if transactions is None:
        transactions = []
    ...

# B007 — unused loop variable
for i in range(len(transactions)):  # WRONG — i used only for indexing
    process(transactions[i])

# Fixed:
for txn in transactions:
    process(txn)

# B023 — function in loop captures loop variable by reference
handlers = []
for name in ["deposit", "withdraw"]:
    handlers.append(lambda: name)   # all lambdas capture same `name`

# Fixed: use default argument to capture current value
handlers = []
for name in ["deposit", "withdraw"]:
    handlers.append(lambda n=name: n)
```

## Exercise 5 — pytest-xdist parallel execution

```bash
pip install pytest-xdist
pytest tests/unit/ -n auto -v
```

If tests fail in parallel mode but pass sequentially, the cause is **shared mutable state**:
- Module-level variables mutated across tests.
- A `scope="session"` fixture that is not thread-safe.
- Tests that write to the same temp file without using `tmp_path` (which is per-test).

Fix: ensure every test creates its own isolated objects. Use `tmp_path` for all file I/O. Avoid module-level mutable state in `conftest.py` session fixtures.
