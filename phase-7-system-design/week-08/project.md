# Week 08 Project — Timed Problem-Set Solution Suite

> Deliverable: A tested Python package containing clean solutions to 10 interview problems, each with documented complexity analysis.  ·  Time: ~3 hr

## Goal

Produce production-quality interview solutions — not just correct, but readable, typed, and tested. This is the artefact you would show in a portfolio or engineering interview.

---

## Structure

```
week_08_project/
├── solutions/
│   ├── __init__.py
│   ├── two_pointers.py
│   ├── binary_search.py
│   ├── intervals.py
│   ├── trees_graphs.py
│   ├── dynamic_programming.py
│   ├── heaps_tries_uf.py
│   └── greedy.py
├── tests/
│   ├── __init__.py
│   ├── test_two_pointers.py
│   ├── test_binary_search.py
│   ├── test_intervals.py
│   ├── test_trees_graphs.py
│   ├── test_dynamic_programming.py
│   ├── test_heaps_tries_uf.py
│   └── test_greedy.py
├── pyproject.toml
└── complexity_analysis.md
```

---

## Required implementations

Implement each function with:
- Full type hints
- Docstring: one-line summary + `Time:` + `Space:` lines
- `logging.debug` at key decision points
- At least one assert-based invariant check in debug builds (optional but encouraged)

### Module: `two_pointers.py`

```python
def container_with_most_water(height: list[int]) -> int: ...
def three_sum(nums: list[int]) -> list[list[int]]: ...
```

### Module: `binary_search.py`

```python
def search_rotated(nums: list[int], target: int) -> int: ...
def min_eating_speed(piles: list[int], h: int) -> int: ...
```

### Module: `intervals.py`

```python
def merge_intervals(intervals: list[list[int]]) -> list[list[int]]: ...
def non_overlapping_intervals(intervals: list[list[int]]) -> int: ...
```

### Module: `trees_graphs.py`

```python
def max_depth(root: Optional[TreeNode]) -> int: ...
def course_schedule(num_courses: int, prerequisites: list[list[int]]) -> bool: ...
```

### Module: `dynamic_programming.py`

```python
def coin_change(coins: list[int], amount: int) -> int: ...
def longest_common_subsequence(text1: str, text2: str) -> int: ...
```

### Module: `heaps_tries_uf.py`

```python
def find_median_stream() -> tuple[Callable[[int], None], Callable[[], float]]: ...
def count_components(n: int, edges: list[list[int]]) -> int: ...
```

### Module: `greedy.py`

```python
def jump_game_ii(nums: list[int]) -> int: ...
def task_scheduler(tasks: list[str], n: int) -> int: ...
```

Total: **14 functions** across 7 modules covering all week patterns.

---

## Tests

Each test module must include:

- **Normal case** — standard, expected input
- **Edge cases:**
  - Empty or single-element input
  - All identical elements
  - Already-sorted input (where applicable)
  - Maximum constraints approximation (use small stand-in: n=1000)
- **Parametrised tests** using `@pytest.mark.parametrize` for at least one function per module

Example skeleton:

```python
# tests/test_two_pointers.py
import pytest
from solutions.two_pointers import container_with_most_water, three_sum


@pytest.mark.parametrize("height, expected", [
    ([1, 8, 6, 2, 5, 4, 8, 3, 7], 49),
    ([1, 1], 1),
    ([4, 3, 2, 1, 4], 16),
])
def test_container_normal(height: list[int], expected: int) -> None:
    assert container_with_most_water(height) == expected


def test_container_single_element() -> None:
    # Need at least 2 lines to form a container
    assert container_with_most_water([5]) == 0


@pytest.mark.parametrize("nums, expected", [
    ([-1, 0, 1, 2, -1, -4], [[-1, -1, 2], [-1, 0, 1]]),
    ([0, 0, 0], [[0, 0, 0]]),
    ([1, 2, 3], []),
])
def test_three_sum(nums: list[int], expected: list[list[int]]) -> None:
    result = sorted(three_sum(nums))
    assert result == sorted(expected)
```

---

## complexity_analysis.md

Write a table:

| Function | Time | Space | Pattern | Notes |
|---|---|---|---|---|
| container_with_most_water | O(n) | O(1) | Two pointers | exchange argument: always move shorter line |
| ... | | | | |

---

## Deliverable checklist

- [ ] All 14 functions implemented with docstrings and type hints
- [ ] All 7 test modules pass `pytest` with zero failures
- [ ] `pytest --tb=short -q` output is clean
- [ ] `complexity_analysis.md` filled in for all 14 functions
- [ ] No `print()` calls — use `logging`
- [ ] `pyproject.toml` lists `pytest` and any stdlib-only dependencies

---

## Assessment hook

This project feeds directly into the Week 08 assessment coding challenge. The `interval_pattern_solver.py` required for the assessment should be consistent with the style established here.

(Solutions reference: ../solutions/week-08/project.md)
