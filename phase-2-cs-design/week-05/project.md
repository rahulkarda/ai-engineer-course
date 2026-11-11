# Week 05 Project — DP Problem-Set Library

> Time: ~3 hr  ·  Deliverable: runnable Python package with tests

## Overview

Build a self-contained Python library `dp_library/` that implements five classic algorithms in **both memoized and tabulated forms**, plus a `complexity.py` module that measures and compares their runtime empirically.

The project is the synthesis of Days 1–6. You are the implementor and tester — structure it cleanly, write tests, and document your complexity analysis.

---

## Project structure

```
week-05-project/
├── dp_library/
│   ├── __init__.py
│   ├── fibonacci.py          # memoized + tabulated, space-optimized
│   ├── knapsack.py           # 0/1 knapsack: memoized + tabulated + space-opt
│   ├── lcs.py                # LCS: memoized + tabulated, reconstruction
│   ├── edit_distance.py      # edit distance: memoized + tabulated + op list
│   ├── lis.py                # LIS: O(n²) DP + O(n log n) patience sort
├── tests/
│   ├── __init__.py
│   ├── test_fibonacci.py
│   ├── test_knapsack.py
│   ├── test_lcs.py
│   ├── test_edit_distance.py
│   ├── test_lis.py
├── complexity.py             # measure and log empirical timings
├── pyproject.toml
└── README.md
```

---

## Module specifications

### `fibonacci.py`

```python
def fib_memo(n: int) -> int: ...        # @functools.cache, O(n) time O(n) space
def fib_tab(n: int) -> int: ...         # O(n) time O(1) space
```

Both must agree: `fib_memo(n) == fib_tab(n)` for all `0 ≤ n ≤ 100`.

### `knapsack.py`

```python
def knapsack_memo(weights: tuple[int, ...], values: tuple[int, ...], capacity: int) -> int: ...
def knapsack_tab(weights: list[int], values: list[int], capacity: int) -> int: ...
def knapsack_space_opt(weights: list[int], values: list[int], capacity: int) -> int: ...
```

All three must agree. Include an `items_taken(weights, values, capacity) -> list[int]` function that returns the indices of chosen items.

### `lcs.py`

```python
def lcs_memo(s: str, t: str) -> int: ...
def lcs_tab(s: str, t: str) -> int: ...
def lcs_reconstruct(s: str, t: str) -> str: ...  # return the actual LCS string
```

### `edit_distance.py`

```python
def edit_distance_memo(s: str, t: str) -> int: ...
def edit_distance_tab(s: str, t: str) -> int: ...
def edit_ops(s: str, t: str) -> list[str]: ...  # human-readable list of operations
```

### `lis.py`

```python
def lis_dp(nums: list[int]) -> int: ...       # O(n²)
def lis_nlogn(nums: list[int]) -> int: ...    # O(n log n)
def lis_sequence(nums: list[int]) -> list[int]: ...  # reconstruct actual subsequence
```

---

## `complexity.py` — empirical analysis

Implement a script that:

1. For each algorithm pair, generates inputs of increasing size (e.g., n = 10, 100, 1000, 5000).
2. Times each implementation with `time.perf_counter`.
3. Logs a comparison table like:

```
[INFO] Algorithm   | n      | memo (ms) | tab (ms)  | ratio
[INFO] ------------|--------|-----------|-----------|------
[INFO] Fibonacci   | 10     | 0.002     | 0.001     | 2.0x
[INFO] Fibonacci   | 1000   | 0.04      | 0.02      | 2.0x
...
```

4. Notes where memoization has higher overhead (recursion stack) vs tabulation.

---

## Test requirements

Each test module must include:

- At least **4 test functions** per algorithm.
- Tests for: typical case, edge cases (empty input, n=0, single element), large `n`.
- Use `pytest.mark.parametrize` for data-driven cases.
- Mock no external dependencies (pure functions — no mocking needed).
- All tests run with `pytest tests/ -v` without failures.

Example test structure:

```python
# tests/test_knapsack.py
import pytest
from dp_library.knapsack import knapsack_memo, knapsack_tab, knapsack_space_opt

@pytest.mark.parametrize("weights,values,cap,expected", [
    ([2, 3, 4, 5], [3, 4, 5, 6], 5, 7),
    ([1], [10], 1, 10),
    ([], [], 10, 0),
    ([5], [10], 4, 0),         # item doesn't fit
    ([1, 2, 3], [6, 10, 12], 5, 22),
])
def test_knapsack_memo(weights, values, cap, expected):
    assert knapsack_memo(tuple(weights), tuple(values), cap) == expected

def test_memo_tab_agree():
    weights = [2, 3, 1, 5, 4]
    values  = [3, 4, 1, 7, 5]
    cap = 8
    assert knapsack_memo(tuple(weights), tuple(values), cap) == \
           knapsack_tab(weights, values, cap) == \
           knapsack_space_opt(weights, values, cap)
```

---

## pyproject.toml

```toml
[project]
name = "dp-library"
version = "0.1.0"
requires-python = ">=3.12"

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

---

## Deliverable checklist

- [ ] All five modules implemented with both solution strategies.
- [ ] `complexity.py` runs end-to-end and prints a comparison table.
- [ ] `pytest tests/ -v` all green.
- [ ] Type hints on every public function.
- [ ] `logging` used throughout (no `print`).
- [ ] No hardcoded secrets or external API calls.
- [ ] `README.md` with: purpose, how to install, how to run tests, how to run complexity analysis.

---

## Extension challenges (optional)

1. Add a **greedy** module with: interval scheduling, jump game, fractional knapsack — with tests.
2. Implement **merge sort with inversion count** and time it against `sorted()`.
3. Add type-safe Pydantic models for knapsack input validation (`KnapsackInput`, `KnapsackResult`).
4. Profile with `cProfile` and identify the bottleneck in the O(n²) LIS for `n=5000`.

(Solutions / reference implementation: ../solutions/week-05/project.md)
