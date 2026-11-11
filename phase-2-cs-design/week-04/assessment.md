# Week 04 Assessment — Graphs

## Conceptual (interview-style)

1. **BFS vs DFS: when would you choose each, and what does each guarantee?**

2. **Compare adjacency list and adjacency matrix representations. Give the time and space complexity for each, and name a scenario where each is preferred.**

3. **What is topological sort, and when does it fail? Describe two algorithms that compute it.**

4. **Explain Dijkstra's algorithm. What assumption about edge weights must hold, and why does violating it break correctness?**

5. **What is Union-Find (Disjoint Set Union)? Describe the path compression and union-by-rank optimizations and the amortized complexity they achieve.**

6. **How do you detect a cycle in a directed graph using DFS? Why does the undirected case require a different approach?**

---

## Coding challenge

### Problem: Dependency Resolver

You are given a list of `(package, dependency)` pairs. Each pair means "package requires dependency to be installed first." Implement a function `install_order` that returns a valid installation order, or raises a `CycleError` if a circular dependency exists.

```python
from typing import Any

def install_order(deps: list[tuple[str, str]]) -> list[str]:
    """
    Given dependency pairs (package, depends_on), return a valid install order.

    The first element of the result must be installable with no prior installs.
    All packages and their dependencies must appear in the result.
    If any circular dependency exists, raise CycleError.

    Args:
        deps: list of (package, depends_on) pairs.

    Returns:
        Ordered list of package names, safe to install left-to-right.

    Raises:
        CycleError: if a circular dependency is detected.
    """
```

### Example

```python
deps = [
    ("django", "python"),
    ("celery", "django"),
    ("celery", "redis"),
    ("redis", "python"),
]
result = install_order(deps)
# One valid result: ["python", "redis", "django", "celery"]
# "python" before "django"; "redis" before "celery"; "django" before "celery"
# "python" also satisfies redis's dependency
```

### Acceptance criteria

| # | Criterion |
|---|---|
| 1 | Returns a list where for every `(pkg, dep)` pair in input, `dep` appears before `pkg` in the result |
| 2 | Every package that appears in any pair (as package or dependency) is present in the result exactly once |
| 3 | Raises `CycleError` (not returns `None`, not returns `[]`) on circular input |
| 4 | Handles a single-package input (one pair where package != dependency) correctly |
| 5 | Handles a disconnected dependency graph (two unrelated packages) correctly |
| 6 | Handles a package with multiple dependencies (fan-in) correctly |
| 7 | Implementation uses Kahn's algorithm or DFS-based topo-sort (not brute-force sorting) |
| 8 | No use of `networkx` or other graph libraries |
| 9 | Type hints present on all functions |
| 10 | At least one test case using `pytest.raises(CycleError)` |

### Circular dependency input (must raise)

```python
deps = [
    ("a", "b"),
    ("b", "c"),
    ("c", "a"),   # cycle: a -> b -> c -> a
]
install_order(deps)  # must raise CycleError
```

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

**Independent implementation (40%):** Solution written without looking up algorithm pseudocode during the session. Correct topological sort. Correct cycle detection. Correct handling of all acceptance criteria.

**Conceptual understanding (25%):** Conceptual answers are complete and accurate. Candidate can explain why their algorithm works, not just that it works. Complexity analysis correct.

**Debugging & problem-solving (20%):** When given a failing test case, candidate isolates the bug within a few minutes. Able to trace through their algorithm by hand on a small example.

**Code quality & testing (15%):** Type hints on all functions. At least one `pytest` test per code path. No `print` statements. Variable names are meaningful.

Answer key: ../solutions/week-04/assessment.md
