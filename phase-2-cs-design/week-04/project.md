# Week 04 Project — Graph Library with Dependency Resolver

## Project Goals

By the end of this project you will have:

1. Built a general-purpose `Graph` class from scratch using only the Python standard library.
2. Implemented BFS, DFS, topological sort (Kahn's algorithm), Dijkstra, cycle detection, and connected components inside that class.
3. Applied the library to a realistic use case: a `DependencyResolver` that models package dependency graphs and answers practical questions about install order and circular dependencies.
4. Written a `pytest` test suite with at least 20 cases that prove correctness at the boundary.

### Learning outcomes

After completing this project you can:
- Explain the adjacency-list representation and implement it in Python using `dict[str, dict[str, float]]`.
- Implement Kahn's topological sort, articulate why it detects cycles, and contrast it with DFS-based topo-sort.
- Implement Dijkstra's algorithm with a min-heap and explain lazy deletion.
- Use three-color DFS to detect and explain cycles in directed graphs.
- Design a library API (separation between data structure and use case) and test it.

---

## Directory Structure Scaffold

```
graph_lib_project/
    graph_lib/
        __init__.py          # public API re-exports
        graph.py             # Graph + CycleError
        resolver.py          # DependencyResolver
        exceptions.py        # CycleError (imported by graph.py and resolver.py)
    tests/
        __init__.py
        test_graph.py        # Graph class tests
        test_resolver.py     # DependencyResolver tests
    pyproject.toml
```

`pyproject.toml` minimum content:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "graph_lib"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

---

## Detailed API Specification

### `exceptions.py`

```python
class CycleError(Exception):
    """Raised when a graph operation requires acyclicity but a cycle exists."""
    def __init__(self, message: str = "Graph contains a cycle", cycle: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle: list[str] = cycle or []
```

### `graph.py` — `Graph` class

```python
from __future__ import annotations
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Graph:
    """Directed or undirected weighted graph using adjacency-list representation."""

    directed: bool = False
    # _adj: node -> {neighbor: weight}
    _adj: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def add_node(self, node: str) -> None:
        """Add an isolated node. No-op if already present."""

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """Add edge u -> v (and v -> u if undirected). Auto-creates nodes."""

    def neighbors(self, node: str) -> list[str]:
        """Return list of neighbor node names. Raises KeyError if node absent."""

    def has_cycle(self) -> bool:
        """Return True if graph contains at least one cycle."""

    def bfs(self, start: str) -> list[str]:
        """BFS traversal order from start. Raises KeyError if start absent."""

    def dfs(self, start: str) -> list[str]:
        """Iterative DFS traversal order from start. Raises KeyError if start absent."""

    def topological_sort(self) -> list[str]:
        """
        Kahn's algorithm topological sort.
        Raises CycleError if graph contains a cycle.
        Returns empty list for empty graph.
        Only meaningful for directed graphs.
        """

    def dijkstra(self, start: str) -> dict[str, float]:
        """
        Shortest distances from start to all reachable nodes.
        Returns dict of {node: distance}. Unreachable nodes are absent.
        Raises KeyError if start absent.
        Raises ValueError if any edge weight is negative.
        """

    def connected_components(self) -> list[set[str]]:
        """
        Find all connected components.
        For directed graphs, treats edges as undirected (weak components).
        Returns list of sets, one set per component.
        """
```

### `resolver.py` — `DependencyResolver` class

```python
from __future__ import annotations
import logging
from graph_lib.graph import Graph
from graph_lib.exceptions import CycleError

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Models a package dependency graph and answers install-order questions.

    Internally uses a directed Graph where an edge dep -> pkg means
    "dep must be installed before pkg".
    """

    def __init__(self) -> None:
        self._graph: Graph = Graph(directed=True)

    def add_dependency(self, package: str, depends_on: str) -> None:
        """
        Record that package requires depends_on.
        Edge direction: depends_on -> package.
        """

    def install_order(self) -> list[str]:
        """
        Return a valid installation order (topological sort).
        The first element has no dependencies; the last is the final package.
        Raises CycleError if circular dependency detected.
        """

    def transitive_deps(self, package: str) -> set[str]:
        """
        Return all packages that must be installed before package,
        including indirect (transitive) dependencies.
        Does not include package itself.
        Raises KeyError if package is unknown.
        """

    def has_circular_dep(self) -> bool:
        """Return True if any circular dependency exists."""

    def explain_cycle(self) -> list[str]:
        """
        Return one cycle as an ordered list of package names.
        Empty list if no cycle exists.
        Example: ["a", "b", "c", "a"] means a -> b -> c -> a.
        """
```

---

## Acceptance Criteria per Component

### `Graph.add_node` / `Graph.add_edge`

- [ ] `add_node("x")` followed by `"x" in graph._adj` returns `True`
- [ ] `add_edge("a", "b")` on undirected graph creates `"b"` in `graph._adj["a"]` and `"a"` in `graph._adj["b"]`
- [ ] `add_edge("a", "b")` on directed graph does NOT add reverse edge
- [ ] `add_edge` auto-creates both nodes if absent
- [ ] Re-adding existing node is a no-op (does not duplicate adjacency list)

### `Graph.bfs` / `Graph.dfs`

- [ ] Returns nodes in correct traversal order on a known small graph
- [ ] Only returns nodes reachable from start (not disconnected components)
- [ ] Each node appears at most once in the result
- [ ] `bfs` visits nodes in non-decreasing hop-count order (level by level)

### `Graph.has_cycle`

- [ ] DAG returns `False`
- [ ] Graph with a back-edge returns `True`
- [ ] Self-loop `(a, a)` returns `True`
- [ ] Undirected triangle returns `True`
- [ ] Linear chain (no back-edge) returns `False`

### `Graph.topological_sort`

- [ ] Result satisfies: for every edge `(u, v)`, `u` appears before `v`
- [ ] Raises `CycleError` on any graph with a cycle
- [ ] Returns `[]` for empty graph
- [ ] Works correctly for graph with multiple valid orderings (any valid ordering accepted)

### `Graph.dijkstra`

- [ ] Correct distances on a known weighted graph
- [ ] Source node has distance 0
- [ ] Unreachable nodes are absent from result dict
- [ ] Raises `ValueError` on negative weights
- [ ] Works on unweighted graph (all weights = 1.0)

### `Graph.connected_components`

- [ ] Single fully-connected graph returns one component
- [ ] Two disjoint subgraphs return two components
- [ ] Each node appears in exactly one component
- [ ] Result is correct for a graph with isolated nodes

### `DependencyResolver`

- [ ] `install_order` returns correct order for a chain of deps
- [ ] `install_order` raises `CycleError` on circular dep
- [ ] `transitive_deps("pkg")` includes all indirect deps, not just direct ones
- [ ] `transitive_deps("pkg")` does not include `pkg` itself
- [ ] `has_circular_dep` returns `True` iff a cycle exists
- [ ] `explain_cycle` returns a non-empty list forming a valid cycle when one exists

---

## Test Plan

### `tests/test_graph.py` — required test cases

```python
# Fixtures
# - linear_graph: a -> b -> c -> d (directed)
# - triangle_graph: a -> b -> c -> a (directed, has cycle)
# - undirected_square: a-b-c-d-a (undirected, connected)
# - weighted_graph: a->(b,1), a->(c,4), b->(c,2), b->(d,5), c->(d,1)
# - disconnected_graph: two disjoint components {a,b} and {c,d}

def test_add_node_creates_entry(): ...
def test_add_node_is_idempotent(): ...
def test_add_edge_directed_no_reverse(): ...
def test_add_edge_undirected_bidirectional(): ...
def test_add_edge_auto_creates_nodes(): ...
def test_neighbors_returns_correct_list(): ...
def test_neighbors_missing_node_raises(): ...
def test_bfs_order_linear_graph(): ...
def test_bfs_does_not_cross_components(): ...
def test_dfs_visits_all_reachable(): ...
def test_dfs_no_revisit(): ...
def test_has_cycle_dag_false(): ...
def test_has_cycle_back_edge_true(): ...
def test_has_cycle_self_loop_true(): ...
def test_topological_sort_valid_order(): ...
def test_topological_sort_raises_on_cycle(): ...
def test_topological_sort_empty_graph(): ...
def test_dijkstra_known_distances(): ...
def test_dijkstra_source_distance_zero(): ...
def test_dijkstra_unreachable_absent(): ...
def test_connected_components_count(): ...
def test_connected_components_isolated_node(): ...
```

### `tests/test_resolver.py` — required test cases

```python
def test_install_order_simple_chain(): ...
def test_install_order_diamond_shape(): ...
def test_install_order_raises_on_cycle(): ...
def test_install_order_empty_resolver(): ...
def test_transitive_deps_direct_only(): ...
def test_transitive_deps_multi_hop(): ...
def test_transitive_deps_excludes_self(): ...
def test_has_circular_dep_false(): ...
def test_has_circular_dep_true(): ...
def test_explain_cycle_valid_cycle_list(): ...
def test_explain_cycle_empty_when_no_cycle(): ...
```

---

## Stretch Goals

1. **Weighted dependency resolver**: add an optional `cost: float` to `add_dependency` and expose a `critical_path(package)` method that returns the costliest path to install a package using a longest-path algorithm on the DAG.

2. **Bellman-Ford**: implement `bellman_ford(start)` on `Graph` — handles negative weights, detects negative cycles. Compare its output to `dijkstra` on positive-weight graphs.

3. **Tarjan's SCC**: implement `strongly_connected_components()` on directed graphs using Tarjan's single-pass DFS with low-link values.

4. **Serialization**: add `Graph.to_dict() -> dict` and `Graph.from_dict(data: dict) -> Graph` for JSON round-tripping. Use `pathlib.Path` for file I/O in a `save` / `load` convenience pair.

5. **CLI**: add a `__main__.py` to `graph_lib` that reads a JSON file of `{deps: [[pkg, dep], ...]}` and prints the install order, or reports the cycle. Invoke with `python -m graph_lib deps.json`.

---

## Supplementary links

- [Python `dataclasses` — official docs](https://docs.python.org/3/library/dataclasses.html)
- [Python `heapq` — official docs](https://docs.python.org/3/library/heapq.html)
- [Python `logging` — official docs](https://docs.python.org/3/library/logging.html)
- [Kahn's topological sort — Wikipedia](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
- [Dijkstra's algorithm — Wikipedia](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [Three-color DFS — MIT OpenCourseWare 6.006 Lecture 14](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/)
- [Union-Find with path compression — CP-Algorithms](https://cp-algorithms.com/data_structures/disjoint_set_union.html)
- [pytest fixtures and parametrize — official docs](https://docs.pytest.org/en/stable/reference/fixtures.html)
