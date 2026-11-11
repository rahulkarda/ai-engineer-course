# Week 4 Day 5 — Modules, Packages & Imports: Solutions

## Exercise 1: src/ Layout Package with __init__.py, __all__, and Re-exports

### Directory structure

```
transaction_lib/
├── pyproject.toml
└── src/
    └── transaction_lib/
        ├── __init__.py          ← public API, re-exports, __all__
        ├── models.py            ← data models submodule
        └── utils.py             ← utility helpers submodule
```

### src/transaction_lib/models.py

```python
"""Core data models for transaction_lib."""
from dataclasses import dataclass
from decimal import Decimal


@dataclass
class Transaction:
    id: str
    amount: Decimal
    currency: str


@dataclass
class Account:
    id: str
    owner: str
    balance: Decimal
```

### src/transaction_lib/utils.py

```python
"""Utility helpers for transaction_lib."""
from decimal import Decimal


def format_amount(amount: Decimal, currency: str) -> str:
    """Return a human-readable money string, e.g. '€1,234.56'."""
    symbols: dict[str, str] = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency + " ")
    return f"{symbol}{amount:,.2f}"


def is_positive(amount: Decimal) -> bool:
    return amount > Decimal("0")
```

### src/transaction_lib/__init__.py

```python
"""
transaction_lib public API.

Consumers should import from this package, not from submodules directly:

    from transaction_lib import Transaction, Account, format_amount
"""
from transaction_lib.models import Account, Transaction
from transaction_lib.utils import format_amount, is_positive

# Controls what `from transaction_lib import *` exposes.
# Also documents the intentional public surface.
__all__ = [
    "Account",
    "Transaction",
    "format_amount",
    "is_positive",
]
```

**Key insight:** Putting everything in `__all__` and re-exporting from `__init__.py` gives consumers a stable, single import path. Internal refactoring of submodule layout does not break downstream code.

---

## Exercise 2: Absolute vs Relative Imports Inside a Package

### Absolute import (preferred)

```python
# src/transaction_lib/utils.py  — absolute style
from transaction_lib.models import Transaction   # full dotted path from sys.path root


def describe(txn: Transaction) -> str:
    return f"Txn {txn.id}: {txn.amount} {txn.currency}"
```

### Relative import (also valid inside a package)

```python
# src/transaction_lib/utils.py  — relative style
from .models import Transaction   # single dot = same package


def describe(txn: Transaction) -> str:
    return f"Txn {txn.id}: {txn.amount} {txn.currency}"
```

### Difference summary

| | Absolute | Relative |
|---|---|---|
| Syntax | `from transaction_lib.models import X` | `from .models import X` |
| Works when run as `__main__` | Yes | No (no package context) |
| Survives package rename | No | Yes |
| PEP 8 recommendation | Preferred | OK inside packages |

**Key insight:** Relative imports are convenient during internal refactoring but break if you try to run a module as a script directly (`python src/transaction_lib/utils.py`). Absolute imports always work and are easier to grep/trace.

---

## Exercise 3: Circular Import — Scenario and Fix

### Problem code (circular)

```
# problem/a.py
from problem.b import greet_b

def greet_a() -> str:
    return "Hello from A"

# problem/b.py
from problem.a import greet_a          # ← circular: a imports b, b imports a

def greet_b() -> str:
    return f"B says: {greet_a()}"
```

Running `from problem.a import greet_b` raises `ImportError: cannot import name 'greet_a' from partially initialized module`.

### Fix 1 — Import inside the function (deferred import)

```python
# fixed_deferred/b.py
def greet_b() -> str:
    from fixed_deferred.a import greet_a   # imported at call time, not module load
    return f"B says: {greet_a()}"

# fixed_deferred/a.py
from fixed_deferred.b import greet_b

def greet_a() -> str:
    return "Hello from A"
```

The import runs only when `greet_b()` is called, by which point both modules are fully initialised.

### Fix 2 — Restructuring (preferred for larger packages)

```python
# fixed_restructured/common.py  ← new module; no dependencies on a or b
def greet_a() -> str:
    return "Hello from A"

# fixed_restructured/a.py
from fixed_restructured.common import greet_a   # no cycle
from fixed_restructured.b import greet_b        # fine; b no longer imports a

# fixed_restructured/b.py
from fixed_restructured.common import greet_a   # depends only on common

def greet_b() -> str:
    return f"B says: {greet_a()}"
```

**Key insight:** Deferred imports are a quick fix for legacy code. Restructuring into a dependency-free `common`/`base` module is the clean long-term solution and makes the dependency graph explicit.

---

## Exercise 4: pyproject.toml for transaction_lib

```toml
[build-system]
requires = ["setuptools>=70", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "transaction_lib"
version = "0.1.0"
description = "Core transaction models and utilities"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
    "pydantic>=2.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.4",
]

[tool.setuptools.packages.find]
where = ["src"]

# ---------------------------------------------------------------------------
# Ruff — linter + formatter (replaces flake8, isort, pyupgrade, black)
# ---------------------------------------------------------------------------
[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "SIM", # flake8-simplify
    "ANN", # type annotations
]
ignore = ["ANN101", "ANN102"]   # skip 'self' / 'cls' annotation

[tool.ruff.lint.isort]
known-first-party = ["transaction_lib"]

# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -v --cov=src/transaction_lib --cov-report=term-missing"
markers = [
    "unit: fast, no I/O",
    "integration: may touch network or filesystem",
]
```

---

## Exercise 5: importlib.import_module() — Dynamic Import

```python
import importlib
import logging
from types import ModuleType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_module(dotted_name: str) -> ModuleType:
    """
    Dynamically import a module by its dotted string name.
    Raises ImportError if the module does not exist.
    """
    module = importlib.import_module(dotted_name)
    logger.info("Loaded module: %r (file: %s)", dotted_name, getattr(module, "__file__", "built-in"))
    return module


def load_attribute(dotted_name: str, attribute: str) -> object:
    """Import a module and return one of its attributes."""
    module = load_module(dotted_name)
    attr = getattr(module, attribute)
    logger.info("Fetched attribute %r from %r: %r", attribute, dotted_name, attr)
    return attr


# --- Demo ---
if __name__ == "__main__":
    # 1. Load a stdlib module by name
    json_mod = load_module("json")
    print(json_mod.dumps({"key": "value"}))   # {"key": "value"}

    # 2. Load a submodule
    os_path = load_module("os.path")
    print(os_path.join("/tmp", "demo.txt"))   # /tmp/demo.txt

    # 3. Fetch a specific attribute (useful for plugin systems)
    dumps_fn = load_attribute("json", "dumps")
    print(dumps_fn({"dynamic": True}))

    # 4. Plugin-loader pattern
    plugin_names: list[str] = ["json", "csv", "pathlib"]
    plugins: dict[str, ModuleType] = {}
    for name in plugin_names:
        try:
            plugins[name] = load_module(name)
        except ImportError:
            logger.warning("Plugin %r not available", name)

    logger.info("Loaded plugins: %s", list(plugins.keys()))
```

**Key insight:** `importlib.import_module()` is the correct way to do dynamic imports (vs `__import__()` which is lower-level). It's essential for plugin systems, CLI entry-point loaders, and test fixtures that need to import user-specified modules.
