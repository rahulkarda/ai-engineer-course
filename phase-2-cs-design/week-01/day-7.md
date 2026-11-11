# Phase 2 · Week 01 · Day 7 — Week Project + Assessment Day

> Objective: Complete the week project (algorithm library with complexity annotations + pytest benchmark harness) and sit the formal assessment.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-1]] – [[day-6]]

---

## Day structure

| Time | Activity |
|---|---|
| 0:00 – 2:30 | Build the week project (see project.md) |
| 2:30 – 3:00 | Self-review against rubric |
| 3:00 – 4:00 | Formal assessment (assessment.md) |

---

## Project review checklist (before starting the assessment)

Work through `project.md` and verify these before moving on:

- [ ] `src/algo_lib/__init__.py` exports all public functions.
- [ ] Every function has a docstring with **Time:** and **Space:** annotations.
- [ ] `tests/unit/` has at least one test per function, all passing.
- [ ] `tests/bench/benchmark_algo_lib.py` runs via `pytest --benchmark-only` and produces output.
- [ ] `ruff check src/` and `ruff check tests/` both return no errors.
- [ ] No hardcoded secrets; any configuration uses `os.getenv`.

---

## Mini-lecture: tying it all together

### Complexity annotation as documentation

Complexity annotations are first-class documentation. They answer the question "will this code still work at 10× the data?" before it's asked in a production postmortem.

Convention used in this course:
```python
def my_function(arr: list[int], k: int) -> int:
    """
    Brief description.

    Time:  O(n)      — single pass
    Space: O(1)      — constant extra
    Requires: arr must be sorted
    """
```

### When to use each technique

```
Problem shape                      → Technique
─────────────────────────────────────────────────────
"range sum query (multiple)"       → prefix sums
"longest/shortest subarray where…" → sliding window
"pair/triplet summing to target"   → two pointers (sorted)
"remove/partition in-place"        → two pointers (fast/slow)
"are these words the same letters" → frequency map / Counter
"pattern in text"                  → KMP or Rabin-Karp
"is this string a palindrome"      → two pointers or center expand
```

### Benchmark harness pattern

```python
# pytest-benchmark: pip install pytest-benchmark
def test_merge_sort_bench(benchmark):
    data = list(range(10_000, 0, -1))
    result = benchmark(merge_sort, data)
    assert result == sorted(data)
```

`benchmark()` wraps the call, runs it multiple times, reports min/mean/stddev. Use `--benchmark-only` flag to run only benchmark tests.

---

## Worked review: catching a complexity bug

```python
# Bug: claimed O(n), actually O(n²)
def build_result_buggy(arr: list[int]) -> list[int]:
    """
    Claimed: O(n) time.
    Actual: O(n²) — why?
    """
    result: list[int] = []
    for x in arr:
        result = result + [x]   # list concatenation creates NEW list each time
    return result

# Fix: use append (O(1) amortized)
def build_result_fixed(arr: list[int]) -> list[int]:
    """Time: O(n) amortized."""
    result: list[int] = []
    for x in arr:
        result.append(x)        # in-place, amortized O(1)
    return result
```

The bug: `result = result + [x]` allocates a new list of growing size each iteration. Total allocations: 1 + 2 + 3 + … + n = n(n+1)/2 → O(n²).

This is a classic "hidden O(n²)" pattern. The fix is always to use `append` or pre-allocate.

---

## Reflection prompts

Before starting the assessment, answer briefly in your notes:

1. Pick one algorithm from this week. Explain its complexity to a non-programmer using an analogy.
2. Name one case from a real codebase (or hypothetical) where choosing O(n²) over O(n) could cause a production incident.
3. What is the relationship between "amortized O(1) append" and "sliding window O(n) total"?

---

## Proceed to assessment

Open `assessment.md` in this directory and work through it under timed conditions (~60 min).

*(Solutions: ../solutions/week-01/assessment.md and ../solutions/week-01/project.md)*
