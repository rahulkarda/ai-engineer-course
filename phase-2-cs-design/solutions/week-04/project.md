# Week 04 Project — Reference Implementation

## Complete `graph_lib` package

All code is Python 3.12+, type hints throughout, `logging` not `print`, `dataclasses`, `pathlib` where relevant.

---

### `graph_lib/exceptions.py`

```python
"""Custom exceptions for graph_lib."""

from __future__ import annotations


class CycleError(Exception):
    """Raised when a graph operation requires acyclicity but a cycle exists."""

    def __init__(
        self,
        message: str = "Graph contains a cycle",
        cycle: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.cycle: list[str] = cycle or []

    def __str__(self) -> str:
        base = super().__str__()
        if self.cycle:
            return f"{base}: {' -> '.join(self.cycle)}"
        return base
```

---

### `graph_lib/graph.py`

```python
"""
Core Graph class: directed/undirected, weighted/unweighted, adjacency-list.
Python 3.12+, type hints, logging, dataclasses.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto

from graph_lib.exceptions import CycleError

logger = logging.getLogger(__name__)


class _Color(Enum):
    WHITE = auto()  # not yet visited
    GRAY = auto()   # on current DFS path
    BLACK = auto()  # fully processed


@dataclass
class Graph:
    """
    Directed or undirected weighted graph using adjacency-list representation.

    Internal storage: _adj maps each node to a dict of {neighbor: weight}.
    Unweighted edges are stored with weight=1.0.
    """

    directed: bool = False
    _adj: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: str) -> None:
        """Add an isolated node. No-op if already present."""
        if node not in self._adj:
            self._adj[node] = {}
            logger.debug("add_node: %s", node)

    def add_edge(self, u: str, v: str, weight: float = 1.0) -> None:
        """
        Add edge u -> v with optional weight.
        Auto-creates both nodes if absent.
        If undirected, also adds v -> u.
        Overwrites weight if edge already exists.
        """
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = weight
        logger.debug("add_edge: %s -> %s (weight=%.2f, directed=%s)", u, v, weight, self.directed)
        if not self.directed:
            self._adj[v][u] = weight

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[str]:
        """All node names in insertion order."""
        return list(self._adj.keys())

    def neighbors(self, node: str) -> list[str]:
        """
        Return neighbor node names in adjacency-list order.

        Raises:
            KeyError: if node is not in the graph.
        """
        if node not in self._adj:
            raise KeyError(f"Node not found: {node!r}")
        return list(self._adj[node].keys())

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def bfs(self, start: str) -> list[str]:
        """
        BFS traversal from start. Returns nodes in visit order.

        Raises:
            KeyError: if start is not in the graph.
        """
        if start not in self._adj:
            raise KeyError(f"Node not found: {start!r}")

        visited: set[str] = set()
        order: list[str] = []
        queue: deque[str] = deque([start])
        visited.add(start)

        while queue:
            node = queue.popleft()
            order.append(node)
            logger.debug("bfs visit: %s", node)
            for neighbor in self._adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return order

    def dfs(self, start: str) -> list[str]:
        """
        Iterative DFS traversal from start. Returns nodes in visit order.
        Uses explicit stack to avoid Python recursion limit.

        Raises:
            KeyError: if start is not in the graph.
        """
        if start not in self._adj:
            raise KeyError(f"Node not found: {start!r}")

        visited: set[str] = set()
        order: list[str] = []
        stack: list[str] = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            logger.debug("dfs visit: %s", node)
            # Push in reverse order so left-most neighbor is processed first
            for neighbor in reversed(list(self._adj[node])):
                if neighbor not in visited:
                    stack.append(neighbor)

        return order

    # ------------------------------------------------------------------
    # Cycle detection
    # ------------------------------------------------------------------

    def has_cycle(self) -> bool:
        """
        Return True if the graph contains at least one cycle.

        Directed: three-color DFS (back-edge detection).
        Undirected: DFS with parent tracking.
        """
        if self.directed:
            return self._has_cycle_directed()
        return self._has_cycle_undirected()

    def _has_cycle_directed(self) -> bool:
        color: dict[str, _Color] = {n: _Color.WHITE for n in self._adj}

        def dfs_visit(node: str) -> bool:
            color[node] = _Color.GRAY
            for neighbor in self._adj[node]:
                if color[neighbor] == _Color.GRAY:
                    logger.debug("Cycle detected: back-edge %s -> %s", node, neighbor)
                    return True
                if color[neighbor] == _Color.WHITE and dfs_visit(neighbor):
                    return True
            color[node] = _Color.BLACK
            return False

        return any(
            dfs_visit(node)
            for node in self._adj
            if color[node] == _Color.WHITE
        )

    def _has_cycle_undirected(self) -> bool:
        visited: set[str] = set()

        def dfs_visit(node: str, parent: str | None) -> bool:
            visited.add(node)
            for neighbor in self._adj[node]:
                if neighbor not in visited:
                    if dfs_visit(neighbor, node):
                        return True
                elif neighbor != parent:
                    logger.debug("Cycle detected: back-edge %s -> %s", node, neighbor)
                    return True
            return False

        return any(
            dfs_visit(node, None)
            for node in self._adj
            if node not in visited
        )

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """
        Kahn's algorithm topological sort for directed graphs.
        Returns nodes in an order where all prerequisites come first.

        Returns:
            Ordered list of node names.

        Raises:
            CycleError: if the graph contains a cycle.
        """
        if not self._adj:
            return []

        # Compute in-degrees
        in_degree: dict[str, int] = {n: 0 for n in self._adj}
        for node in self._adj:
            for neighbor in self._adj[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

        queue: deque[str] = deque(
            node for node in self._adj if in_degree[node] == 0
        )
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for successor in self._adj[node]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(self._adj):
            remaining = set(self._adj) - set(result)
            raise CycleError(
                f"Topological sort failed: cycle among {sorted(remaining)}",
                cycle=sorted(remaining),
            )

        logger.debug("topological_sort: %s", result)
        return result

    # ------------------------------------------------------------------
    # Dijkstra's shortest paths
    # ------------------------------------------------------------------

    def dijkstra(self, start: str) -> dict[str, float]:
        """
        Single-source shortest paths using Dijkstra's algorithm (min-heap).

        Args:
            start: source node.

        Returns:
            dict mapping reachable node -> shortest distance from start.
            Source node maps to 0.0. Unreachable nodes are absent.

        Raises:
            KeyError: if start is not in the graph.
            ValueError: if any edge weight is negative.
        """
        if start not in self._adj:
            raise KeyError(f"Node not found: {start!r}")

        # Check for negative weights upfront
        for node in self._adj:
            for neighbor, weight in self._adj[node].items():
                if weight < 0:
                    raise ValueError(
                        f"Negative weight on edge {node!r} -> {neighbor!r}: {weight}"
                    )

        import heapq

        dist: dict[str, float] = {start: 0.0}
        heap: list[tuple[float, str]] = [(0.0, start)]

        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, math.inf):
                continue  # stale entry — lazy deletion
            for neighbor, weight in self._adj[node].items():
                new_dist = d + weight
                if new_dist < dist.get(neighbor, math.inf):
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
                    logger.debug("dijkstra: relax %s -> %s, dist=%.2f", node, neighbor, new_dist)

        return dist

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------

    def connected_components(self) -> list[set[str]]:
        """
        Find all weakly connected components.
        For directed graphs, edges are treated as undirected.

        Returns:
            List of sets, one set per component.
        """
        visited: set[str] = set()
        components: list[set[str]] = []

        # Build undirected adjacency for traversal
        undirected: dict[str, set[str]] = defaultdict(set)
        for node in self._adj:
            undirected[node]  # ensure isolated nodes appear
            for neighbor in self._adj[node]:
                undirected[node].add(neighbor)
                undirected[neighbor].add(node)

        def bfs_component(start: str) -> set[str]:
            component: set[str] = set()
            queue: deque[str] = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                component.add(node)
                for neighbor in undirected[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return component

        for node in self._adj:
            if node not in visited:
                components.append(bfs_component(node))

        logger.debug("connected_components: %d components", len(components))
        return components
```

---

### `graph_lib/resolver.py`

```python
"""
DependencyResolver: uses Graph to model and resolve package dependencies.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum, auto

from graph_lib.exceptions import CycleError
from graph_lib.graph import Graph

logger = logging.getLogger(__name__)


class _Color(Enum):
    WHITE = auto()
    GRAY = auto()
    BLACK = auto()


class DependencyResolver:
    """
    Models a package dependency graph and answers install-order questions.

    Edge direction: dep -> package (dep must be installed before package).
    """

    def __init__(self) -> None:
        self._graph: Graph = Graph(directed=True)

    def add_dependency(self, package: str, depends_on: str) -> None:
        """
        Record that package requires depends_on.
        Creates edge: depends_on -> package.
        """
        self._graph.add_edge(depends_on, package)
        logger.debug("add_dependency: %s requires %s", package, depends_on)

    def install_order(self) -> list[str]:
        """
        Return a valid installation order (topological sort).
        First element: a package with no dependencies.
        Last element: the final package to install.

        Raises:
            CycleError: if a circular dependency is detected.
        """
        try:
            order = self._graph.topological_sort()
        except CycleError:
            logger.error("Circular dependency detected")
            raise
        logger.info("install_order: %s", order)
        return order

    def transitive_deps(self, package: str) -> set[str]:
        """
        Return all packages that must be installed before package,
        including indirect (transitive) dependencies.
        Does not include package itself.

        Strategy: BFS on the reversed graph from package.
        In the reversed graph (dep <- package), reachable nodes are
        packages that package transitively depends on.

        Raises:
            KeyError: if package is not in the graph.
        """
        if package not in self._graph._adj:
            raise KeyError(f"Unknown package: {package!r}")

        # Build reversed adjacency on the fly
        reversed_adj: dict[str, list[str]] = {n: [] for n in self._graph.nodes}
        for node in self._graph.nodes:
            for successor in self._graph.neighbors(node):
                reversed_adj[successor].append(node)

        # BFS from package in reversed graph
        visited: set[str] = set()
        queue: deque[str] = deque([package])
        visited.add(package)

        while queue:
            node = queue.popleft()
            for dep in reversed_adj.get(node, []):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)

        visited.discard(package)
        logger.debug("transitive_deps(%s): %s", package, sorted(visited))
        return visited

    def has_circular_dep(self) -> bool:
        """Return True if any circular dependency exists."""
        return self._graph.has_cycle()

    def explain_cycle(self) -> list[str]:
        """
        Return one cycle as an ordered list of package names.
        The last element equals the first (closed cycle).
        Returns empty list if no cycle exists.

        Example: ["a", "b", "c", "a"] means a -> b -> c -> a.
        """
        adj = self._graph._adj
        color: dict[str, _Color] = {n: _Color.WHITE for n in adj}
        parent: dict[str, str | None] = {n: None for n in adj}
        cycle_back: tuple[str, str] | None = None  # (from_node, to_node)

        def dfs(node: str) -> bool:
            nonlocal cycle_back
            color[node] = _Color.GRAY
            for neighbor in adj[node]:
                if color[neighbor] == _Color.GRAY:
                    cycle_back = (node, neighbor)
                    return True
                if color[neighbor] == _Color.WHITE:
                    parent[neighbor] = node
                    if dfs(neighbor):
                        return True
            color[node] = _Color.BLACK
            return False

        for node in adj:
            if color[node] == _Color.WHITE:
                if dfs(node):
                    break

        if cycle_back is None:
            return []

        # Reconstruct cycle by walking parent pointers
        tail, head = cycle_back  # tail -> head is the back-edge; head is in cycle
        path: list[str] = [head]
        current = tail
        while current != head:
            path.append(current)
            current = parent[current]  # type: ignore[assignment]
        path.append(head)
        path.reverse()
        logger.debug("explain_cycle: %s", path)
        return path
```

---

### `graph_lib/__init__.py`

```python
"""graph_lib: graph algorithms and dependency resolution."""

from graph_lib.exceptions import CycleError
from graph_lib.graph import Graph
from graph_lib.resolver import DependencyResolver

__all__ = ["Graph", "DependencyResolver", "CycleError"]
```

---

## Full pytest test suite

### `tests/test_graph.py`

```python
"""
Tests for Graph class.
Run: pytest tests/test_graph.py -v
"""

from __future__ import annotations

import pytest

from graph_lib import CycleError, Graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def directed_chain() -> Graph:
    """a -> b -> c -> d"""
    g = Graph(directed=True)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d")]:
        g.add_edge(u, v)
    return g


@pytest.fixture
def directed_cycle() -> Graph:
    """a -> b -> c -> a"""
    g = Graph(directed=True)
    for u, v in [("a", "b"), ("b", "c"), ("c", "a")]:
        g.add_edge(u, v)
    return g


@pytest.fixture
def undirected_square() -> Graph:
    """a - b - c - d - a (square, connected, undirected)"""
    g = Graph(directed=False)
    for u, v in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]:
        g.add_edge(u, v)
    return g


@pytest.fixture
def weighted_graph() -> Graph:
    """
    Weighted directed graph:
    a -1-> b, a -4-> c, b -2-> c, b -5-> d, c -1-> d
    Shortest: a->b=1, a->c=3, a->d=4
    """
    g = Graph(directed=True)
    g.add_edge("a", "b", 1.0)
    g.add_edge("a", "c", 4.0)
    g.add_edge("b", "c", 2.0)
    g.add_edge("b", "d", 5.0)
    g.add_edge("c", "d", 1.0)
    return g


@pytest.fixture
def disconnected_graph() -> Graph:
    """Two components: {a, b} and {c, d}"""
    g = Graph(directed=False)
    g.add_edge("a", "b")
    g.add_edge("c", "d")
    return g


# ---------------------------------------------------------------------------
# add_node / add_edge
# ---------------------------------------------------------------------------


def test_add_node_creates_entry() -> None:
    g = Graph()
    g.add_node("x")
    assert "x" in g._adj


def test_add_node_is_idempotent() -> None:
    g = Graph()
    g.add_node("x")
    g.add_node("x")
    assert len(g._adj) == 1


def test_add_edge_directed_no_reverse() -> None:
    g = Graph(directed=True)
    g.add_edge("a", "b")
    assert "b" in g._adj["a"]
    assert "a" not in g._adj["b"]


def test_add_edge_undirected_bidirectional() -> None:
    g = Graph(directed=False)
    g.add_edge("a", "b")
    assert "b" in g._adj["a"]
    assert "a" in g._adj["b"]


def test_add_edge_auto_creates_nodes() -> None:
    g = Graph()
    g.add_edge("x", "y")
    assert "x" in g._adj
    assert "y" in g._adj


def test_add_edge_stores_weight() -> None:
    g = Graph(directed=True)
    g.add_edge("a", "b", 3.5)
    assert g._adj["a"]["b"] == 3.5


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------


def test_neighbors_returns_correct_list(directed_chain: Graph) -> None:
    assert "b" in directed_chain.neighbors("a")
    assert "c" in directed_chain.neighbors("b")


def test_neighbors_missing_node_raises() -> None:
    g = Graph()
    with pytest.raises(KeyError):
        g.neighbors("ghost")


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def test_bfs_order_linear_graph(directed_chain: Graph) -> None:
    result = directed_chain.bfs("a")
    assert result == ["a", "b", "c", "d"]


def test_bfs_does_not_cross_components(disconnected_graph: Graph) -> None:
    result = disconnected_graph.bfs("a")
    assert set(result) == {"a", "b"}
    assert "c" not in result
    assert "d" not in result


def test_bfs_missing_start_raises() -> None:
    g = Graph()
    with pytest.raises(KeyError):
        g.bfs("missing")


def test_bfs_single_node() -> None:
    g = Graph()
    g.add_node("solo")
    assert g.bfs("solo") == ["solo"]


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------


def test_dfs_visits_all_reachable(directed_chain: Graph) -> None:
    result = directed_chain.dfs("a")
    assert set(result) == {"a", "b", "c", "d"}


def test_dfs_no_revisit(directed_chain: Graph) -> None:
    result = directed_chain.dfs("a")
    assert len(result) == len(set(result))


def test_dfs_does_not_cross_components(disconnected_graph: Graph) -> None:
    result = disconnected_graph.dfs("a")
    assert set(result) == {"a", "b"}


# ---------------------------------------------------------------------------
# has_cycle
# ---------------------------------------------------------------------------


def test_has_cycle_dag_false(directed_chain: Graph) -> None:
    assert directed_chain.has_cycle() is False


def test_has_cycle_back_edge_true(directed_cycle: Graph) -> None:
    assert directed_cycle.has_cycle() is True


def test_has_cycle_self_loop_true() -> None:
    g = Graph(directed=True)
    g.add_edge("a", "a")
    assert g.has_cycle() is True


def test_has_cycle_undirected_triangle_true() -> None:
    g = Graph(directed=False)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    assert g.has_cycle() is True


def test_has_cycle_undirected_tree_false() -> None:
    g = Graph(directed=False)
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert g.has_cycle() is False


# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------


def test_topological_sort_valid_order(directed_chain: Graph) -> None:
    result = directed_chain.topological_sort()
    assert result.index("a") < result.index("b")
    assert result.index("b") < result.index("c")
    assert result.index("c") < result.index("d")


def test_topological_sort_raises_on_cycle(directed_cycle: Graph) -> None:
    with pytest.raises(CycleError):
        directed_cycle.topological_sort()


def test_topological_sort_empty_graph() -> None:
    g = Graph(directed=True)
    assert g.topological_sort() == []


def test_topological_sort_diamond() -> None:
    g = Graph(directed=True)
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    result = g.topological_sort()
    assert result.index("a") < result.index("b")
    assert result.index("a") < result.index("c")
    assert result.index("b") < result.index("d")
    assert result.index("c") < result.index("d")


# ---------------------------------------------------------------------------
# dijkstra
# ---------------------------------------------------------------------------


def test_dijkstra_known_distances(weighted_graph: Graph) -> None:
    dist = weighted_graph.dijkstra("a")
    assert dist["a"] == 0.0
    assert dist["b"] == 1.0
    assert dist["c"] == 3.0  # a->b->c = 1+2
    assert dist["d"] == 4.0  # a->b->c->d = 1+2+1


def test_dijkstra_source_distance_zero(weighted_graph: Graph) -> None:
    dist = weighted_graph.dijkstra("a")
    assert dist["a"] == 0.0


def test_dijkstra_unreachable_absent() -> None:
    g = Graph(directed=True)
    g.add_edge("a", "b")
    g.add_node("z")  # isolated
    dist = g.dijkstra("a")
    assert "z" not in dist


def test_dijkstra_negative_weight_raises() -> None:
    g = Graph(directed=True)
    g.add_edge("a", "b", -1.0)
    with pytest.raises(ValueError):
        g.dijkstra("a")


def test_dijkstra_missing_start_raises() -> None:
    g = Graph()
    with pytest.raises(KeyError):
        g.dijkstra("missing")


# ---------------------------------------------------------------------------
# connected_components
# ---------------------------------------------------------------------------


def test_connected_components_single(undirected_square: Graph) -> None:
    components = undirected_square.connected_components()
    assert len(components) == 1
    assert components[0] == {"a", "b", "c", "d"}


def test_connected_components_two(disconnected_graph: Graph) -> None:
    components = disconnected_graph.connected_components()
    assert len(components) == 2
    node_sets = [frozenset(c) for c in components]
    assert frozenset({"a", "b"}) in node_sets
    assert frozenset({"c", "d"}) in node_sets


def test_connected_components_isolated_node() -> None:
    g = Graph(directed=False)
    g.add_edge("a", "b")
    g.add_node("lone")
    components = g.connected_components()
    assert len(components) == 2
    sizes = sorted(len(c) for c in components)
    assert sizes == [1, 2]


def test_connected_components_each_node_once(disconnected_graph: Graph) -> None:
    components = disconnected_graph.connected_components()
    all_nodes: list[str] = []
    for c in components:
        all_nodes.extend(c)
    assert len(all_nodes) == len(set(all_nodes))
```

---

### `tests/test_resolver.py`

```python
"""
Tests for DependencyResolver.
Run: pytest tests/test_resolver.py -v
"""

from __future__ import annotations

import pytest

from graph_lib import CycleError, DependencyResolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_resolver() -> DependencyResolver:
    """python -> redis -> celery; python -> django -> celery"""
    r = DependencyResolver()
    r.add_dependency("redis", "python")
    r.add_dependency("django", "python")
    r.add_dependency("celery", "redis")
    r.add_dependency("celery", "django")
    return r


@pytest.fixture
def cyclic_resolver() -> DependencyResolver:
    """a -> b -> c -> a"""
    r = DependencyResolver()
    r.add_dependency("b", "a")
    r.add_dependency("c", "b")
    r.add_dependency("a", "c")
    return r


# ---------------------------------------------------------------------------
# install_order
# ---------------------------------------------------------------------------


def test_install_order_simple_chain(simple_resolver: DependencyResolver) -> None:
    order = simple_resolver.install_order()
    assert order.index("python") < order.index("redis")
    assert order.index("python") < order.index("django")
    assert order.index("redis") < order.index("celery")
    assert order.index("django") < order.index("celery")


def test_install_order_all_packages_present(simple_resolver: DependencyResolver) -> None:
    order = simple_resolver.install_order()
    assert set(order) == {"python", "redis", "django", "celery"}


def test_install_order_no_duplicates(simple_resolver: DependencyResolver) -> None:
    order = simple_resolver.install_order()
    assert len(order) == len(set(order))


def test_install_order_raises_on_cycle(cyclic_resolver: DependencyResolver) -> None:
    with pytest.raises(CycleError):
        cyclic_resolver.install_order()


def test_install_order_empty_resolver() -> None:
    r = DependencyResolver()
    assert r.install_order() == []


def test_install_order_single_dependency() -> None:
    r = DependencyResolver()
    r.add_dependency("app", "lib")
    order = r.install_order()
    assert order.index("lib") < order.index("app")


# ---------------------------------------------------------------------------
# transitive_deps
# ---------------------------------------------------------------------------


def test_transitive_deps_direct_only() -> None:
    r = DependencyResolver()
    r.add_dependency("app", "libA")
    deps = r.transitive_deps("app")
    assert "libA" in deps


def test_transitive_deps_multi_hop(simple_resolver: DependencyResolver) -> None:
    deps = simple_resolver.transitive_deps("celery")
    assert "redis" in deps
    assert "django" in deps
    assert "python" in deps  # transitive via both paths


def test_transitive_deps_excludes_self(simple_resolver: DependencyResolver) -> None:
    deps = simple_resolver.transitive_deps("celery")
    assert "celery" not in deps


def test_transitive_deps_no_deps() -> None:
    r = DependencyResolver()
    r.add_dependency("app", "lib")
    # lib has no dependencies
    deps = r.transitive_deps("lib")
    assert deps == set()


def test_transitive_deps_unknown_package_raises() -> None:
    r = DependencyResolver()
    with pytest.raises(KeyError):
        r.transitive_deps("ghost")


# ---------------------------------------------------------------------------
# has_circular_dep / explain_cycle
# ---------------------------------------------------------------------------


def test_has_circular_dep_false(simple_resolver: DependencyResolver) -> None:
    assert simple_resolver.has_circular_dep() is False


def test_has_circular_dep_true(cyclic_resolver: DependencyResolver) -> None:
    assert cyclic_resolver.has_circular_dep() is True


def test_explain_cycle_empty_when_no_cycle(simple_resolver: DependencyResolver) -> None:
    assert simple_resolver.explain_cycle() == []


def test_explain_cycle_valid_cycle_list(cyclic_resolver: DependencyResolver) -> None:
    cycle = cyclic_resolver.explain_cycle()
    assert len(cycle) >= 3
    # First and last elements should be the same (closed cycle)
    assert cycle[0] == cycle[-1]
    # All intermediate nodes should be in the cycle set
    cycle_nodes = {"a", "b", "c"}
    assert set(cycle[:-1]).issubset(cycle_nodes)


def test_explain_cycle_self_loop() -> None:
    r = DependencyResolver()
    r.add_dependency("a", "a")  # a depends on itself
    cycle = r.explain_cycle()
    assert len(cycle) > 0
```

---

## Notes on the implementation

**Graph storage**: `dict[str, dict[str, float]]` gives O(1) average edge insertion and lookup, O(degree(v)) neighbor iteration. Storing weight alongside neighbors avoids a separate `weights` dict.

**Cycle detection in `has_cycle`**: uses Python-level recursion for the three-color DFS. For very large graphs (depth > 1000), switch to an iterative three-color DFS using an explicit stack with a state enum.

**`connected_components` treats directed graphs as undirected (weak components)**: this is the correct behavior for dependency analysis — whether a package can reach another along any edge matters. Strongly connected components (Kosaraju/Tarjan) are left as a stretch goal.

**`DependencyResolver.transitive_deps`**: builds a reversed adjacency on the fly per call. For performance-critical use, maintain a separate `_reversed` graph updated in `add_dependency`.

**Determinism**: `topological_sort` does not guarantee a unique order when multiple valid orderings exist. Tests use `index(a) < index(b)` checks rather than full-list equality to avoid brittle assertions.
