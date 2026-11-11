# Phase 2 · Week 04 · Day 7 — Graph Library: Dependency Resolver (Project Day)

> Objective: Build a complete graph library with BFS/DFS/topo-sort/Dijkstra and a dependency-resolver use case  ·  Time: ~3 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Project Overview

This project consolidates everything from the week: you will build `graph_lib`, a reusable Python package that implements the core graph algorithms studied in Days 1–5, then use it to solve a real-world problem — resolving package installation order in the presence of dependency cycles.

The result is a library you can cite in interviews and use in future projects. It demonstrates:
- clean API design (a general `Graph` class that works for many problem shapes)
- practical algorithm implementation without leaning on `networkx`
- property-based thinking (if a cycle exists, topo-sort must raise, not silently produce garbage)
- test-driven construction (write the test, then the implementation)

The dependency resolver mirrors what `pip`, `npm`, and `cargo` do internally: given a list of `(package, depends_on)` pairs, determine a valid install order, surface circular dependencies clearly, and find all transitive dependencies of any package.

---

## Specifications

### `graph_lib` package

**`Graph` class** — directed or undirected, weighted or unweighted, adjacency-list representation.

Required methods:

| Method | Signature | Description |
|---|---|---|
| `add_node` | `(node: str) -> None` | Add isolated node if absent |
| `add_edge` | `(u: str, v: str, weight: float = 1.0) -> None` | Add edge; auto-creates nodes |
| `neighbors` | `(node: str) -> list[str]` | Neighbors in adjacency-list order |
| `has_cycle` | `() -> bool` | True if graph contains a cycle |
| `bfs` | `(start: str) -> list[str]` | BFS traversal order from start |
| `dfs` | `(start: str) -> list[str]` | DFS traversal order from start (iterative) |
| `topological_sort` | `() -> list[str]` | Kahn's algorithm; raises `CycleError` if cycle |
| `dijkstra` | `(start: str) -> dict[str, float]` | Shortest distances from start |
| `connected_components` | `() -> list[set[str]]` | List of component node-sets |

**`DependencyResolver` class** — wraps a `Graph` to model package dependencies.

Required methods:

| Method | Signature | Description |
|---|---|---|
| `add_dependency` | `(package: str, depends_on: str) -> None` | Record one dependency edge |
| `install_order` | `() -> list[str]` | Topological order; raises `CycleError` on cycle |
| `transitive_deps` | `(package: str) -> set[str]` | All packages that must be installed before `package` |
| `has_circular_dep` | `() -> bool` | True if any circular dependency exists |
| `explain_cycle` | `() -> list[str]` | Return one cycle as an ordered node list; empty list if none |

**Custom exception:**

```python
class CycleError(Exception):
    """Raised when a topological operation is requested on a cyclic graph."""
```

---

## Implementation Guide

### Step 1 — scaffold the package (15 min)

Create the directory structure:

```
graph_lib/
    __init__.py          # re-export Graph, DependencyResolver, CycleError
    graph.py             # Graph class
    resolver.py          # DependencyResolver class
    exceptions.py        # CycleError
tests/
    __init__.py
    test_graph.py
    test_resolver.py
pyproject.toml
```

In `pyproject.toml`, declare the package and a `[tool.pytest.ini_options]` section so `pytest` finds `graph_lib` without install.

Hint: use `pythonpath = ["."]` in pytest config to avoid editable installs for now.

### Step 2 — define the data model (20 min)

In `graph.py`, use `dataclasses`. The core state is:

```python
@dataclass
class Graph:
    directed: bool = False
    _adj: dict[str, dict[str, float]] = field(default_factory=dict)
```

Using `dict[str, float]` for each adjacency list (neighbor -> weight) handles both weighted and unweighted edges cleanly. Unweighted edges default to `weight=1.0`.

Add `add_node` and `add_edge` first. Write tests for them before moving on.

### Step 3 — BFS and DFS (30 min)

BFS: use `collections.deque`. Return visited nodes in traversal order.

DFS: use an explicit stack (`list` with `.pop()`). Do not use recursion — Python's recursion limit is 1000, which would break on large dependency trees.

Write tests with a small fixed graph (5–6 nodes) before moving on.

### Step 4 — cycle detection (20 min)

For directed graphs, use DFS with three-color marking:
- WHITE = not visited
- GRAY = in current DFS path (back-edge = cycle)
- BLACK = fully processed

For undirected graphs, use simple DFS tracking parent — a back-edge that is not the parent's edge means a cycle.

### Step 5 — topological sort with Kahn's algorithm (30 min)

Kahn's algorithm:
1. Compute in-degree for every node.
2. Enqueue all nodes with in-degree 0.
3. Pop a node, append to result, decrement in-degree of its successors; enqueue any that hit 0.
4. If result length < number of nodes, a cycle exists — raise `CycleError`.

Kahn's is preferred over DFS topo-sort here because it naturally detects cycles without separate cycle-detection code.

### Step 6 — Dijkstra (30 min)

Use `heapq.heappush` / `heapq.heappop`. Maintain a `dist` dict initialized to `math.inf` for all nodes. Only process a node when it is first popped from the heap (lazy deletion — skip if already finalized).

Return `dict[str, float]` mapping every reachable node to its shortest distance. Unreachable nodes are absent from the result (do not include `inf` entries).

### Step 7 — connected components (15 min)

Works only meaningfully on undirected graphs. Iterate over all nodes; for each unvisited node, run BFS/DFS and collect the reachable set as one component. Return `list[set[str]]`.

For directed graphs, return each node in its own component (or raise `ValueError` — either is acceptable; document the choice).

### Step 8 — DependencyResolver (30 min)

`DependencyResolver` owns a `Graph(directed=True)`. `add_dependency(pkg, dep)` calls `graph.add_edge(dep, pkg)` — the edge points from dependency to dependent (dep must be installed before pkg).

`install_order` delegates to `graph.topological_sort`. The first element of the result is a package with no dependencies; the last is the final package.

`transitive_deps(pkg)`: run BFS/DFS on the *reversed* graph from `pkg` — all nodes reachable in the reversed graph are packages that `pkg` transitively depends on. Alternatively, maintain a separate reversed graph.

`explain_cycle`: run DFS with the three-color scheme and record the back-edge; walk the GRAY path to extract the cycle.

### Step 9 — polish and edge cases (30 min)

- Empty graph: `topological_sort([])` should return `[]`.
- Single node with no edges: `bfs("a")` returns `["a"]`; `topological_sort` returns `["a"]`.
- Disconnected graph: `bfs` from one component does not visit the other; `connected_components` returns both.
- Self-loop: must be detected as a cycle.
- Duplicate `add_edge` calls: decide whether to allow (multi-graph) or deduplicate; document the choice.

---

## Testing Requirements

Write tests before implementation (TDD). Use `pytest`. All tests live in `tests/`.

**Minimum test coverage:**

| Area | Tests to write |
|---|---|
| `add_node` / `add_edge` | nodes present after add; undirected edge appears in both directions |
| `neighbors` | correct list; missing node raises `KeyError` |
| `bfs` | traversal order on a known graph; disconnected start |
| `dfs` | traversal visits all reachable; does not revisit |
| `has_cycle` | DAG returns False; graph with back-edge returns True; self-loop returns True |
| `topological_sort` | valid order on DAG; raises `CycleError` on cycle; empty graph |
| `dijkstra` | known shortest paths; handles disconnected nodes |
| `connected_components` | 1-component graph; 2-component graph |
| `DependencyResolver` | install_order on acyclic deps; raises `CycleError` on circular dep; transitive_deps correct |
| `explain_cycle` | returned list forms a valid cycle in the graph |

Aim for at least 20 test cases total. Use `pytest.mark.parametrize` where multiple inputs test the same property.

---

## Deliverables checklist

- [ ] `graph_lib/exceptions.py` — `CycleError` defined
- [ ] `graph_lib/graph.py` — `Graph` class with all 9 methods implemented
- [ ] `graph_lib/resolver.py` — `DependencyResolver` class with all 5 methods implemented
- [ ] `graph_lib/__init__.py` — re-exports `Graph`, `DependencyResolver`, `CycleError`
- [ ] `tests/test_graph.py` — at least 12 test cases covering all `Graph` methods
- [ ] `tests/test_resolver.py` — at least 8 test cases covering all `DependencyResolver` methods
- [ ] `pyproject.toml` — project config with pytest settings
- [ ] All code passes `ruff check .` with no errors
- [ ] All tests pass with `pytest -v`
- [ ] No `print` calls — use `logging` throughout
- [ ] All public methods have type hints and docstrings

---

## Supplementary links

- [Python `collections.deque` docs](https://docs.python.org/3/library/collections.html#collections.deque)
- [Python `heapq` docs](https://docs.python.org/3/library/heapq.html)
- [Python `dataclasses` docs](https://docs.python.org/3/library/dataclasses.html)
- [Kahn's algorithm — Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
- [Dijkstra's algorithm — Wikipedia](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Three-color DFS cycle detection — CLRS Section 22.3](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/)
