# Week 04 Assessment — Answer Key

## Conceptual answers

### 1. BFS vs DFS: when would you choose each, and what does each guarantee?

BFS explores level by level using a FIFO queue. It guarantees the **shortest path in an unweighted graph** because it visits nodes in non-decreasing hop-count order — the first time a node is reached, it is reached via the minimum number of edges. Use BFS when you need shortest paths, when the target is likely near the source, or when the graph is wide and shallow.

DFS dives along a single path as far as possible before backtracking, using a stack (implicit via recursion or explicit). It guarantees visiting all reachable nodes but gives no path-length guarantee. DFS produces a spanning forest, classifies edges (tree/back/forward/cross), and is the foundation for cycle detection, topological sort, and SCC algorithms. Use DFS when you need to detect cycles, compute topological order, or explore all paths.

Both run in O(V + E) time and O(V) space.

### 2. Adjacency list vs adjacency matrix: complexity and preferred scenarios

**Adjacency list** (`dict[node, list[neighbor]]` or `dict[node, dict[neighbor, weight]]`):
- Space: O(V + E)
- Edge lookup: O(degree(v)) average
- Iterate all neighbors of v: O(degree(v))
- Best for: sparse graphs (E << V^2), real-world networks (social graphs, road networks)

**Adjacency matrix** (V x V array, `matrix[i][j] = weight`):
- Space: O(V^2)
- Edge lookup: O(1)
- Iterate all neighbors of v: O(V)
- Best for: dense graphs, algorithms requiring fast arbitrary edge lookup (Floyd-Warshall), or when V is small and constant

Prefer adjacency list for almost all interview and production scenarios unless the graph is dense or you need O(1) edge existence checks.

### 3. Topological sort: definition, failure case, two algorithms

A topological ordering of a directed graph is a linear sequence of all nodes such that for every directed edge `(u, v)`, `u` appears before `v`. It represents a valid execution order of tasks with prerequisites.

Topological sort only exists for Directed Acyclic Graphs (DAGs). If the graph contains a cycle, no valid ordering exists — at least one node in the cycle would need to come before itself.

**Kahn's algorithm (BFS-based):** Compute in-degrees; enqueue all zero-in-degree nodes; iteratively pop a node, add to result, decrement successor in-degrees, enqueue any that hit zero. If result length < V, a cycle exists.

**DFS-based topo-sort:** Run DFS; when a node finishes (all descendants processed), push it onto a stack. Reverse the stack for the final order. Cycle detection requires the three-color scheme — a back-edge (GRAY to GRAY) indicates a cycle. Both algorithms run in O(V + E).

### 4. Dijkstra's algorithm: mechanism, weight assumption, and failure case

Dijkstra's algorithm finds shortest paths from a single source in a weighted graph. It maintains a priority queue (min-heap) of `(distance, node)` pairs and a `dist` dict initialized to infinity for all nodes. At each step it extracts the unprocessed node with the smallest known distance, then relaxes its outgoing edges: if `dist[u] + weight(u,v) < dist[v]`, update `dist[v]` and push the new estimate to the heap.

The critical assumption: **all edge weights must be non-negative**. If a negative weight exists, the algorithm may finalize a node's distance before discovering a cheaper path through a later-relaxed edge. Specifically, once a node is popped from the heap it is considered settled — but a negative edge arriving later could lower its cost. Bellman-Ford handles negative weights by relaxing all edges V-1 times without the settled-node assumption.

With a binary min-heap, Dijkstra runs in O((V + E) log V). With a Fibonacci heap, O(E + V log V).

### 5. Union-Find: structure, optimizations, and complexity

Union-Find (Disjoint Set Union, DSU) is a data structure that tracks a partition of elements into disjoint sets. It supports two operations: `find(x)` — return the representative (root) of x's set — and `union(x, y)` — merge the sets containing x and y.

Naive implementation is slow because trees can degenerate into linked lists. Two optimizations fix this:

**Path compression** (`find`): while walking from x to its root, make every visited node point directly to the root. Subsequent `find` calls on those nodes cost O(1).

**Union by rank** (`union`): always attach the shorter tree under the taller one (tracking height as "rank"). Keeps trees shallow.

Together, they give an amortized time complexity of O(alpha(n)) per operation, where alpha is the inverse Ackermann function — effectively constant (never exceeds 4 for any practical n). Union-Find is used for Kruskal's MST algorithm and for cycle detection in undirected graphs.

### 6. Cycle detection in directed vs undirected graphs via DFS

**Directed graph — three-color DFS:** Each node is colored WHITE (unvisited), GRAY (in current DFS path / recursion stack), or BLACK (fully processed). A cycle exists if and only if a DFS traversal encounters a back-edge — an edge from a GRAY node to another GRAY node. Finding an edge to a BLACK node is fine (it is a cross or forward edge, not a cycle). Track the GRAY set as a set of nodes currently on the recursion stack.

**Undirected graph — parent tracking:** In an undirected graph, every edge appears in both directions, so the "reverse" of a tree-edge would always look like a back-edge to the parent. The fix: track the node you came from (parent). An edge to a visited node that is not the parent is a genuine back-edge and indicates a cycle. This simpler approach works because undirected graphs have no forward or cross edges in their DFS trees — only tree-edges and back-edges.

The directed case is strictly harder because the edge direction matters and a path to a previously-seen node is only a cycle if that node is still on the active stack.

---

## Coding challenge solution

```python
"""
Week 04 Assessment — coding challenge solution.
Implements install_order using Kahn's topological sort.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class CycleError(Exception):
    """Raised when a circular dependency is detected."""

    def __init__(self, message: str = "Circular dependency detected", cycle: list[str] | None = None) -> None:
        super().__init__(message)
        self.cycle: list[str] = cycle or []


def install_order(deps: list[tuple[str, str]]) -> list[str]:
    """
    Given (package, depends_on) pairs, return a valid installation order.

    Uses Kahn's algorithm (BFS-based topological sort).
    Edge direction in internal graph: depends_on -> package
    (depends_on must be processed before package).

    Args:
        deps: list of (package, depends_on) pairs.

    Returns:
        Ordered list of all package names, safe to install left-to-right.

    Raises:
        CycleError: if a circular dependency is detected.
    """
    if not deps:
        return []

    # Build adjacency list and collect all nodes
    # Edge: dep -> pkg (dep before pkg)
    graph: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = defaultdict(int)
    all_nodes: set[str] = set()

    for package, depends_on in deps:
        all_nodes.add(package)
        all_nodes.add(depends_on)

    # Initialize in_degree for all nodes
    for node in all_nodes:
        if node not in in_degree:
            in_degree[node] = 0

    for package, depends_on in deps:
        graph[depends_on].append(package)
        in_degree[package] += 1

    logger.debug("Graph nodes: %s", sorted(all_nodes))
    logger.debug("In-degrees: %s", dict(in_degree))

    # Kahn's algorithm: enqueue nodes with in-degree 0
    queue: deque[str] = deque(
        sorted(node for node in all_nodes if in_degree[node] == 0)
    )
    # sorted() for deterministic output; not required for correctness

    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        logger.debug("Processing: %s", node)

        for successor in sorted(graph[node]):  # sorted for determinism
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(result) != len(all_nodes):
        # Some nodes were never enqueued — they are part of a cycle
        remaining = all_nodes - set(result)
        logger.error("Cycle detected involving nodes: %s", remaining)
        raise CycleError(
            f"Circular dependency detected among: {sorted(remaining)}",
            cycle=sorted(remaining),
        )

    logger.info("Install order: %s", result)
    return result


# ---------------------------------------------------------------------------
# Example usage (not imported by tests)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    example_deps = [
        ("django", "python"),
        ("celery", "django"),
        ("celery", "redis"),
        ("redis", "python"),
    ]
    order = install_order(example_deps)
    logger.info("Result: %s", order)
    # Expected: ["python", "redis", "django", "celery"] or another valid order

    circular_deps = [
        ("a", "b"),
        ("b", "c"),
        ("c", "a"),
    ]
    try:
        install_order(circular_deps)
    except CycleError as exc:
        logger.error("Caught expected error: %s", exc)
```

### Test suite for the coding challenge

```python
"""
Tests for install_order coding challenge solution.
Run with: pytest tests/test_assessment.py -v
"""

from __future__ import annotations

import pytest

from assessment_solution import CycleError, install_order


class TestInstallOrderCorrectness:
    def test_simple_chain(self) -> None:
        deps = [("b", "a"), ("c", "b")]
        result = install_order(deps)
        assert result.index("a") < result.index("b")
        assert result.index("b") < result.index("c")

    def test_diamond_shape(self) -> None:
        # a -> b, a -> c, b -> d, c -> d
        deps = [("b", "a"), ("c", "a"), ("d", "b"), ("d", "c")]
        result = install_order(deps)
        assert result.index("a") < result.index("b")
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("d")

    def test_all_nodes_present(self) -> None:
        deps = [("django", "python"), ("celery", "django"), ("redis", "python")]
        result = install_order(deps)
        assert set(result) == {"python", "django", "celery", "redis"}

    def test_each_node_appears_once(self) -> None:
        deps = [("b", "a"), ("c", "a"), ("c", "b")]
        result = install_order(deps)
        assert len(result) == len(set(result))

    def test_single_pair(self) -> None:
        result = install_order([("pkg", "dep")])
        assert result.index("dep") < result.index("pkg")

    def test_empty_input(self) -> None:
        assert install_order([]) == []

    def test_disconnected_packages(self) -> None:
        deps = [("b", "a"), ("d", "c")]
        result = install_order(deps)
        assert result.index("a") < result.index("b")
        assert result.index("c") < result.index("d")

    def test_multiple_dependencies_fan_in(self) -> None:
        deps = [("app", "libA"), ("app", "libB"), ("app", "libC")]
        result = install_order(deps)
        for lib in ["libA", "libB", "libC"]:
            assert result.index(lib) < result.index("app")


class TestInstallOrderCycleDetection:
    def test_simple_cycle_raises(self) -> None:
        deps = [("a", "b"), ("b", "c"), ("c", "a")]
        with pytest.raises(CycleError):
            install_order(deps)

    def test_self_loop_raises(self) -> None:
        deps = [("a", "a")]
        with pytest.raises(CycleError):
            install_order(deps)

    def test_two_node_cycle_raises(self) -> None:
        deps = [("a", "b"), ("b", "a")]
        with pytest.raises(CycleError):
            install_order(deps)

    def test_cycle_error_type(self) -> None:
        deps = [("a", "b"), ("b", "a")]
        with pytest.raises(CycleError) as exc_info:
            install_order(deps)
        assert isinstance(exc_info.value, CycleError)
```

---

## Grading notes per rubric dimension

### Independent implementation (40%)

Full marks: candidate built the graph (adjacency list with in-degree tracking) and Kahn's loop without referencing pseudocode during the session. The key insight — edge direction is `dep -> package`, not the reverse — should come from the candidate's own reasoning about what "depends on" means.

Partial marks (20%): used DFS-based topo-sort instead of Kahn's but implemented it correctly. Missing the determinism (sorted output) is acceptable; it is not in the spec.

No marks: used `networkx.topological_sort` or equivalent library call, or submitted a brute-force sort that does not actually check dependency order.

### Conceptual understanding (25%)

Full marks: all six answers cover the requested topics with correct complexity analysis. Candidate can articulate *why* Dijkstra requires non-negative weights, not just that it does. Candidate distinguishes directed/undirected cycle detection correctly.

Partial marks: four or five answers correct; complexity analysis has minor errors (e.g., O(E log E) instead of O(E log V) for Dijkstra).

### Debugging & problem-solving (20%)

Full marks: when presented with a failing test (e.g., a graph where all nodes end up in the cycle set incorrectly), candidate traces through Kahn's by hand, identifies that a duplicate edge inflated an in-degree, and fixes it within five minutes.

Partial marks: candidate identifies the wrong output but needs significant prompting to locate the source.

### Code quality & testing (15%)

Full marks: type hints on all function signatures, at least one test per code path (happy path, empty input, cycle), no `print` calls, meaningful variable names. `pytest.raises(CycleError)` used — not bare `try/except`.

Partial marks: type hints present but incomplete; tests present but only cover happy path.
