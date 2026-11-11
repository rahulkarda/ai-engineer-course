# Week 01 Project — Array/String Algorithm Library with Complexity Annotations

> Time: ~3 hr  ·  Prereqs: all day-1 through day-6 files

---

## Goal

Build a small, production-quality Python library of array and string algorithms, each annotated with its complexity. Include a pytest benchmark harness that measures actual runtime across different input sizes and compares it against predicted asymptotic growth.

---

## Project structure

```
week-01-project/
├── src/
│   └── algo_lib/
│       ├── __init__.py           # re-export all public functions
│       ├── complexity.py         # complexity annotations helper
│       ├── sorting.py            # merge_sort, quick_sort (for comparison)
│       ├── search.py             # binary_search, linear_search
│       ├── arrays.py             # two_sum, prefix_sum, sliding_window utils
│       └── strings.py            # anagram, palindrome, kmp
├── tests/
│   ├── unit/
│   │   ├── test_sorting.py
│   │   ├── test_search.py
│   │   ├── test_arrays.py
│   │   └── test_strings.py
│   └── bench/
│       └── benchmark_algo_lib.py
├── pyproject.toml
└── .gitignore
```

---

## Requirements

### R1 — `complexity.py` — annotation decorator

Implement a `@complexity(time="O(...)", space="O(...)")` decorator that:
- Stores complexity metadata on the wrapped function as attributes.
- Does NOT add runtime overhead (annotation only).
- Exposes a `complexity_report(module)` function that prints a table of all annotated functions in a module.

```python
# Example usage
@complexity(time="O(n log n)", space="O(n)")
def merge_sort(arr: list[int]) -> list[int]:
    ...

# After import:
assert merge_sort.time_complexity == "O(n log n)"
assert merge_sort.space_complexity == "O(n)"
```

### R2 — `sorting.py`

Implement:
- `merge_sort(arr: list[int]) -> list[int]` — stable, O(n log n) time, O(n) space
- `insertion_sort(arr: list[int]) -> list[int]` — O(n²) worst, O(n) best (nearly sorted), O(1) space
- `counting_sort(arr: list[int], max_val: int) -> list[int]` — O(n + k) where k=max_val, O(k) space

Each must be decorated with `@complexity`.

### R3 — `search.py`

Implement:
- `linear_search(arr: list[int], target: int) -> int` — O(n), O(1)
- `binary_search(arr: list[int], target: int) -> int` — O(log n), O(1), requires sorted input

### R4 — `arrays.py`

Implement:
- `two_sum_sorted(arr: list[int], target: int) -> tuple[int, int] | None` — two pointers, O(n)
- `three_sum_zero(arr: list[int]) -> list[tuple[int, int, int]]` — O(n²), returns all unique triplets
- `build_prefix_sum(arr: list[int]) -> list[int]` — O(n) build
- `range_sum_query(prefix: list[int], l: int, r: int) -> int` — O(1) query
- `max_sum_subarray_k(arr: list[int], k: int) -> int` — fixed window, O(n)
- `longest_subarray_sum_le(arr: list[int], k: int) -> int` — variable window, O(n), positive elements

### R5 — `strings.py`

Implement:
- `are_anagrams(s: str, t: str) -> bool` — O(n)
- `group_anagrams(words: list[str]) -> list[list[str]]` — O(n·L)
- `longest_palindrome(s: str) -> str` — center expansion, O(n²)
- `kmp_search(text: str, pattern: str) -> list[int]` — O(n+m)

### R6 — Unit tests

Each module has a corresponding test file with:
- At least 3 test cases per function (normal, edge, boundary).
- pytest parametrize for functions with multiple input patterns.
- All tests passing with `pytest tests/unit/`.

### R7 — Benchmark harness

`tests/bench/benchmark_algo_lib.py` must:
- Use `pytest-benchmark` to measure each sort and search function.
- Test on input sizes: `[100, 1_000, 10_000, 100_000]` integers (random, reversed, sorted).
- Log results as a table to confirm empirical complexity matches theoretical: if `n` doubles, does O(n log n) time roughly double (×2.something), while O(n²) roughly quadruples?

---

## Starter code: `complexity.py`

```python
"""
complexity.py — function complexity annotation decorator
Python 3.12+
"""
import logging
import types
from collections.abc import Callable
from typing import Any, TypeVar

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def complexity(time: str, space: str) -> Callable[[F], F]:
    """
    Decorator: attach complexity metadata to a function without runtime overhead.

    Usage:
        @complexity(time="O(n log n)", space="O(n)")
        def my_sort(arr: list[int]) -> list[int]: ...
    """
    def decorator(func: F) -> F:
        func.time_complexity = time    # type: ignore[attr-defined]
        func.space_complexity = space  # type: ignore[attr-defined]
        return func
    return decorator


def complexity_report(module: types.ModuleType) -> None:
    """Print a table of all @complexity-annotated functions in a module."""
    rows = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "time_complexity"):
            rows.append((name, obj.time_complexity, obj.space_complexity))
    if not rows:
        log.info("No complexity-annotated functions found in %s", module.__name__)
        return
    header = f"{'Function':<35} {'Time':>14} {'Space':>14}"
    log.info(header)
    log.info("-" * len(header))
    for name, t, s in sorted(rows):
        log.info("%-35s %14s %14s", name, t, s)
```

---

## Starter code: `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "algo-lib"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-benchmark>=4", "ruff>=0.4"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

---

## Deliverables checklist

- [ ] All R1–R7 functions implemented and type-hinted.
- [ ] `complexity_report(algo_lib.arrays)` prints a clean table.
- [ ] `pytest tests/unit/` → all green.
- [ ] `pytest tests/bench/ --benchmark-only` → runs and produces timing output.
- [ ] `ruff check src/ tests/` → no errors.
- [ ] `git log --oneline` shows at least 3 commits (not one giant commit).

---

## Stretch goals (optional)

1. Add a `@complexity` enforcement test: for `merge_sort`, measure actual runtime at n=1000 and n=2000, assert the ratio is between 2.0 and 2.5 (expected for n log n).
2. Implement Rabin-Karp rolling hash pattern matching as an alternative to KMP. Annotate and test it.
3. Add a 2-D prefix sum matrix and `submatrix_sum` query.
4. Implement Manacher's algorithm for O(n) longest palindrome (instead of O(n²) center expansion).

*(Solutions: ../solutions/week-01/project.md)*
