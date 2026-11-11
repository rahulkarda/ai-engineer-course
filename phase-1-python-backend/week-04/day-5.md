# Phase 1 · Week 04 · Day 5 — Modules, Packages & Imports; __init__.py

> Objective: Understand Python's import system end-to-end — from finder/loader internals to the src/ layout — so you can structure real projects without import confusion.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

A **module** is any `.py` file. A **package** is a directory that contains an `__init__.py` file (regular package) or simply exists on `sys.path` (namespace package). When you write `import foo`, Python searches `sys.path` for `foo.py` or `foo/__init__.py`.

The import system underpins every multi-file project. Understanding it removes the mystery from `ImportError`, `ModuleNotFoundError`, circular import crashes, and the difference between running `python src/app/main.py` vs `python -m app.main`.

## How it works internally

### The import pipeline

When CPython executes `import foo`:

1. **sys.modules cache**: First checks `sys.modules["foo"]`. If found, returns the cached module object immediately — no disk access.
2. **Finders** (`sys.meta_path`): Iterates meta-path finders in order. Each finder's `find_spec(fullname, path, target)` is called. The default finders are:
   - `BuiltinImporter` — handles built-in C modules (`sys`, `builtins`).
   - `FrozenImporter` — handles frozen (compiled-in) modules.
   - `PathFinder` — searches `sys.path` entries for files.
3. **Loader**: The finder returns a `ModuleSpec` that names the loader. The loader's `exec_module(module)` executes the module's source in the module's namespace.
4. **sys.modules registration**: The module object is stored in `sys.modules[fullname]` before execution completes (important for circular imports — see below).

### sys.path

`sys.path` is a list of strings searched by `PathFinder`. It is initialised from:
- The directory of the script being run (or `""` for the interactive interpreter, meaning CWD).
- `PYTHONPATH` environment variable entries.
- Installation-specific defaults (site-packages).

Modifying `sys.path` at runtime works but is a code smell in production code. The right approach is to use `pyproject.toml` and install the package.

### Absolute vs relative imports

**Absolute**: `from mypackage.utils import helper` — always resolved from `sys.path` roots. Unambiguous. Preferred.

**Relative**: `from . import helper` (same package), `from ..models import User` (parent package). Only valid inside a package; fails if the module is run directly (not as part of a package). Relative imports communicate "this is an internal dependency" and are acceptable inside packages, but absolute imports are easier to follow.

### __init__.py

`__init__.py` serves three purposes:

1. **Marks a directory as a regular package** (distinguishes from namespace packages).
2. **Runs package initialisation code** when the package is first imported.
3. **Controls the public API** via `__all__` and re-exports.

What to put in `__init__.py`:
- Package-level `__version__` string.
- Re-exports of the most important names from sub-modules so callers can do `from mypackage import MyModel` instead of the full path.
- `__all__`: list of names exported by `from mypackage import *`. Even if you do not use star imports, `__all__` documents intent and is used by tools like `mypy` and `pdoc`.

What **not** to put in `__init__.py`:
- Heavy logic or imports that slow startup (import time matters for CLI tools).
- Circular dependency chains.

### The src/ layout

Without a `src/` layout, placing your project root on `sys.path` means Python might accidentally import your source directory directly rather than an installed version — subtle bugs during testing. The `src/` layout separates source from project files:

```
my-project/
    src/
        mypackage/
            __init__.py
            models.py
    tests/
        test_models.py
    pyproject.toml
```

With this layout, `tests/` cannot accidentally import from `src/mypackage` unless the package is installed (or `src/` is explicitly on `sys.path` via `pip install -e .`). This catches missing `__init__.py` files, wrong directory references, and packaging mistakes early.

`pyproject.toml` with `setuptools`:
```toml
[tool.setuptools.packages.find]
where = ["src"]
```

This tells setuptools to look for packages inside `src/`.

### Circular imports

Circular imports occur when module A imports from module B, and module B imports from module A. Because `sys.modules` stores the module object before it finishes executing, a circular import may succeed partially — but the imported name may not yet exist in the partially-executed module.

How to break them:
- **Move shared code** into a third module that neither A nor B imports.
- **Defer the import** inside a function body so it runs only when needed (not at module load time).
- **Import the module, not the name**: `import mypackage.models` instead of `from mypackage.models import MyModel` — you access the name later when it is definitely defined.
- **TYPE_CHECKING guard**: wrap import-only-for-type-hints in `if TYPE_CHECKING:` so it only runs for static analysis, not at runtime.

### __name__ == "__main__"

When Python runs a file directly (`python myscript.py`), `__name__` is set to `"__main__"`. When imported as a module, `__name__` is the module's dotted name. The guard `if __name__ == "__main__":` prevents entry-point code from running on import.

For packages, `python -m mypackage` runs `mypackage/__main__.py`.

### importlib for dynamic imports

`importlib.import_module("mypackage.submodule")` imports a module by name string at runtime. Useful for plugin architectures where module names are discovered from config files.

```python
import importlib

plugin_name = "myproject.plugins.auth"
plugin = importlib.import_module(plugin_name)
plugin.setup()
```

### pyproject.toml structure

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["pydantic>=2.0", "fastapi>=0.110"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

The `pythonpath = ["src"]` line in `[tool.pytest.ini_options]` is the modern way to add `src/` to `sys.path` for pytest without installing the package.

## Worked example

The example below defines a small `src/` package with sub-packages and demonstrates all import patterns. The structure created:

```
src/
    analytics/
        __init__.py
        models.py
        utils/
            __init__.py
            math_ops.py
            string_ops.py
main.py
```

```python
# src/analytics/__init__.py
"""
analytics package — top-level re-exports and version.
Python 3.12+
"""

__version__ = "0.1.0"

# Re-export the most-used names so callers can write:
#   from analytics import DataPoint
# instead of:
#   from analytics.models import DataPoint
from analytics.models import DataPoint, Summary

__all__ = ["DataPoint", "Summary", "__version__"]
```

```python
# src/analytics/models.py
"""
Core data models for the analytics package.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class DataPoint:
    label: str
    value: float
    tags: list[str] = field(default_factory=list)


@dataclass
class Summary:
    mean: float
    std_dev: float
    count: int
```

```python
# src/analytics/utils/__init__.py
"""
analytics.utils sub-package.
"""

from analytics.utils.math_ops import mean, std_dev
from analytics.utils.string_ops import slugify

__all__ = ["mean", "std_dev", "slugify"]
```

```python
# src/analytics/utils/math_ops.py
"""
Mathematical utilities.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty sequence."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def std_dev(values: list[float]) -> float:
    """Return the population standard deviation."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
```

```python
# src/analytics/utils/string_ops.py
"""
String utilities.
"""
from __future__ import annotations

import re


def slugify(text: str) -> str:
    """Convert a string to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)
```

```python
# main.py
"""
Entry point demonstrating absolute imports, dynamic imports, and
the __name__ guard.
Python 3.12+
"""
from __future__ import annotations

import importlib
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def demonstrate_imports() -> None:
    # --- Absolute imports ---
    from analytics import DataPoint, Summary, __version__  # re-exported by __init__
    from analytics.utils import mean, std_dev, slugify      # from sub-package __init__
    from analytics.utils.math_ops import mean as math_mean  # direct absolute import

    log.info("analytics version: %s", __version__)

    points: list[DataPoint] = [
        DataPoint("cpu", 82.5, tags=["host:web01"]),
        DataPoint("cpu", 77.1, tags=["host:web02"]),
        DataPoint("cpu", 91.3, tags=["host:web03"]),
    ]

    values = [p.value for p in points]
    summary = Summary(
        mean=mean(values),
        std_dev=std_dev(values),
        count=len(values),
    )
    log.info("Summary: %s", summary)

    slug = slugify("Hello, World! This is a Test.")
    log.info("Slug: %s", slug)

    # --- Dynamic import via importlib ---
    module_name = "analytics.utils.string_ops"
    string_ops = importlib.import_module(module_name)
    log.info("Dynamically imported: %s", string_ops)
    log.info("Dynamic slugify: %s", string_ops.slugify("Dynamic Import Example"))

    # --- Inspect sys.modules cache ---
    cached = [k for k in sys.modules if k.startswith("analytics")]
    log.info("Cached analytics modules: %s", cached)


if __name__ == "__main__":
    demonstrate_imports()
```

To run:
```bash
# From the project root, with src/ on PYTHONPATH:
PYTHONPATH=src python main.py

# Or install in editable mode first:
pip install -e .
python main.py
```

## Exercises

1. Add a `analytics/pipeline.py` module that imports from both `analytics.models` and `analytics.utils`. Verify there are no circular imports. Then introduce a deliberate circular import between `models.py` and `pipeline.py`, observe the `ImportError`, and fix it using one of the strategies described above.

2. In `analytics/__init__.py`, add `__all__` with only `DataPoint` excluded. Confirm with a `from analytics import *` that `DataPoint` is not imported but `Summary` is.

3. Write a `analytics/plugins/` sub-package with a `registry.py` that uses `importlib.import_module` to load plugin modules by name from a list of strings. Test it by adding a `analytics/plugins/csv_exporter.py` with a single `export()` function.

4. Change `main.py` to use relative imports instead of absolute imports for the analytics sub-package. Observe what happens when you run `python main.py` directly vs `python -m main`. Explain the difference.

5. Add a `src/analytics/__main__.py` so that `python -m analytics` prints the package version and summary stats for a hardcoded list of values.

6. Instrument the import system: before calling `demonstrate_imports()`, record `len(sys.modules)`. After, record it again and log how many new modules were loaded.

(Solutions: ../solutions/week-04/day-5.md)

## Self-check

- [ ] I can trace what happens step-by-step when Python executes `import analytics.utils.math_ops`.
- [ ] I know what `sys.modules` is and why modifying `sys.path` at runtime is a code smell.
- [ ] I can explain why the `src/` layout prevents accidental direct imports.
- [ ] I can describe three strategies to break a circular import.
- [ ] I know what `__all__` controls and why it matters even without star imports.
- [ ] I can write `pyproject.toml` with `[tool.setuptools.packages.find]` pointing to `src/`.
- [ ] I can use `importlib.import_module` to load a module by name string.

## Supplementary links

- https://docs.python.org/3/reference/import.html — The import system (official reference)
- https://docs.python.org/3/library/importlib.html — importlib documentation
- https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/ — src layout rationale
- https://setuptools.pypa.io/en/latest/userguide/package_discovery.html — package discovery with setuptools
- https://peps.python.org/pep-0328/ — PEP 328: relative imports
- https://realpython.com/python-import/ — comprehensive import system walkthrough
