# Solutions — Phase 2 · Week 04 · Day 2 — DFS + Connected Components

Solutions to the four exercises in `week-04/day-2.md`.

---

## Exercise 1 — Cycle detection

**Problem:** Write `has_cycle(graph: Graph) -> bool` for an undirected graph.

**Why it works:**
In DFS on an undirected graph, a back edge goes from a node to an already-visited node.
Because the graph is undirected, every edge appears in both directions, so you must ignore
the edge back to the direct parent. If you reach a visited node that is *not* the parent,
you have found a cycle.

The iterative version tracks `(node, parent)` pairs on the stack.

```python
"""
solutions/week-04/day-2-ex1.py — Cycle detection via DFS
"""

from __future__ import annotations

import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

Graph = dict[int, list[int]]


def build_graph(edges: list[tuple[int, int]], n_nodes: int) -> Graph:
    graph: Graph = defaultdict(list)
    for node in range(n_nodes):
        graph[node]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return dict(graph)


def has_cycle(graph: Graph) -> bool:
    """
    Return True if the undirected graph contains at least one cycle.
    Uses iterative DFS. Stack stores (current_node, parent_node).
    """
    visited: set[int] = set()

    for start in graph:
        if start in visited:
            continue

        # Stack items: (node, parent)
        stack: list[tuple[int, int]] = [(start, -1)]

        while stack:
            node, parent = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            for neighbor in graph.get(node, []):
                if neighbor == parent:
                    # Undirected edge back to parent — not a cycle
                    continue
                if neighbor in visited:
                    log.info("cycle detected via edge %d -> %d", node, neighbor)
                    return True
                stack.append((neighbor, node))

    return False


def main() -> None:
    # Graph WITHOUT a cycle: simple path 0-1-2-3
    no_cycle_edges: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 3)]
    g_no_cycle = build_graph(no_cycle_edges, 4)
    log.info("no-cycle graph has cycle: %s", has_cycle(g_no_cycle))   # False

    # Graph WITH a cycle: triangle 0-1-2-0
    cycle_edges: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 0)]
    g_cycle = build_graph(cycle_edges, 3)
    log.info("triangle graph has cycle: %s", has_cycle(g_cycle))       # True

    # Mixed: path 0-1-2 plus cycle 3-4-5-3, disconnected
    mixed_edges: list[tuple[int, int]] = [
        (0, 1), (1, 2),
        (3, 4), (4, 5), (5, 3),
    ]
    g_mixed = build_graph(mixed_edges, 6)
    log.info("mixed graph has cycle: %s", has_cycle(g_mixed))          # True


if __name__ == "__main__":
    main()
```

**Key insight:** the `parent` parameter prevents treating the edge we arrived on as a
back edge. Without it, every edge in an undirected graph would look like a cycle.

---

## Exercise 2 — Bipartite check

**Problem:** Write `is_bipartite(graph: Graph) -> bool` using iterative DFS or BFS.

**Why it works:**
A graph is bipartite if and only if it contains no odd-length cycle. 2-coloring detects
this: assign color 0 to the start; each neighbor gets `1 - color`. If a neighbor already
carries the *same* color as the current node, an odd cycle exists and 2-coloring fails.
Run from every unvisited node to handle disconnected graphs.

```python
"""
solutions/week-04/day-2-ex2.py — Bipartite check via iterative DFS
"""

from __future__ import annotations

import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

Graph = dict[int, list[int]]


def build_graph(edges: list[tuple[int, int]], n_nodes: int) -> Graph:
    graph: Graph = defaultdict(list)
    for node in range(n_nodes):
        graph[node]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return dict(graph)


def is_bipartite(graph: Graph) -> bool:
    """
    Return True if the graph can be 2-colored (no odd cycles).
    Uses iterative DFS; stack stores (node, color).
    color dict: node -> 0 or 1.
    """
    color: dict[int, int] = {}

    for start in graph:
        if start in color:
            continue

        # Stack: (node, assigned_color)
        stack: list[tuple[int, int]] = [(start, 0)]

        while stack:
            node, c = stack.pop()

            if node in color:
                # Already colored — verify consistency
                if color[node] != c:
                    log.info("conflict at node %d (color %d vs %d)", node, color[node], c)
                    return False
                continue

            color[node] = c
            log.debug("colored node %d with %d", node, c)

            for neighbor in graph.get(node, []):
                if neighbor not in color:
                    stack.append((neighbor, 1 - c))
                elif color[neighbor] == c:
                    log.info("conflict: neighbor %d has same color %d as node %d", neighbor, c, node)
                    return False

    return True


def main() -> None:
    # Bipartite: even cycle (square 0-1-2-3-0)
    square_edges: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 3), (3, 0)]
    g_square = build_graph(square_edges, 4)
    log.info("square (even cycle) is bipartite: %s", is_bipartite(g_square))   # True

    # Not bipartite: odd cycle (triangle 0-1-2-0)
    triangle_edges: list[tuple[int, int]] = [(0, 1), (1, 2), (2, 0)]
    g_triangle = build_graph(triangle_edges, 3)
    log.info("triangle (odd cycle) is bipartite: %s", is_bipartite(g_triangle))  # False

    # Bipartite: complete bipartite K2,3
    # Set A: {0,1}, Set B: {2,3,4}
    k23_edges: list[tuple[int, int]] = [
        (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
    ]
    g_k23 = build_graph(k23_edges, 5)
    log.info("K2,3 is bipartite: %s", is_bipartite(g_k23))   # True

    # Disconnected: one bipartite component + one triangle
    mixed_edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # square — bipartite
        (4, 5), (5, 6), (6, 4),            # triangle — not bipartite
    ]
    g_mixed = build_graph(mixed_edges, 7)
    log.info("mixed graph is bipartite: %s", is_bipartite(g_mixed))  # False


if __name__ == "__main__":
    main()
```

**Key insight:** the `1 - c` flip alternates colors at every edge. Push the desired color
along with the node so you can check consistency when the node is popped.

---

## Exercise 3 — Component sizes and largest component

**Problem:** Return `dict[int, int]` of component id to size; then `largest_component`.

**Why it works:**
Instead of recording node lists, maintain a running counter per component. Each DFS
invocation increments a shared counter. At the end, `max(sizes.values())` gives the
largest component. O(V + E) — same as before.

```python
"""
solutions/week-04/day-2-ex3.py — Component sizes + largest component
"""

from __future__ import annotations

import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

Graph = dict[int, list[int]]


def build_graph(edges: list[tuple[int, int]], n_nodes: int) -> Graph:
    graph: Graph = defaultdict(list)
    for node in range(n_nodes):
        graph[node]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return dict(graph)


def _dfs_count(graph: Graph, start: int, visited: set[int]) -> int:
    """Run iterative DFS from `start`; return number of nodes discovered."""
    stack: list[int] = [start]
    count = 0

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        count += 1
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                stack.append(neighbor)

    return count


def component_sizes(graph: Graph) -> dict[int, int]:
    """
    Return mapping of component_id (0-indexed) -> number of nodes in that component.
    """
    visited: set[int] = set()
    sizes: dict[int, int] = {}
    component_id = 0

    for node in sorted(graph.keys()):
        if node not in visited:
            size = _dfs_count(graph, node, visited)
            sizes[component_id] = size
            log.info("component %d: size %d", component_id, size)
            component_id += 1

    return sizes


def largest_component(graph: Graph) -> int:
    """Return the size of the largest connected component."""
    sizes = component_sizes(graph)
    if not sizes:
        return 0
    result = max(sizes.values())
    log.info("largest component size: %d", result)
    return result


def main() -> None:
    # 3 components: sizes 5, 1, 2
    edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (0, 3), (3, 4),  # 5 nodes
        # 5 isolated
        (6, 7),                            # 2 nodes
    ]
    graph = build_graph(edges, 8)

    sizes = component_sizes(graph)
    log.info("all sizes: %s", sizes)           # {0: 5, 1: 1, 2: 2}

    lc = largest_component(graph)
    log.info("largest: %d", lc)                # 5

    # Edge case: fully disconnected (all isolated)
    iso_graph = build_graph([], 4)
    log.info("all isolated sizes: %s", component_sizes(iso_graph))  # {0:1, 1:1, 2:1, 3:1}

    # Edge case: single fully connected component
    chain_edges: list[tuple[int, int]] = [(i, i + 1) for i in range(9)]
    chain_graph = build_graph(chain_edges, 10)
    log.info("chain largest: %d", largest_component(chain_graph))   # 10


if __name__ == "__main__":
    main()
```

**Key insight:** extracting `_dfs_count` as a helper keeps the public functions
single-purpose and makes them easy to unit test independently.

---

## Exercise 4 — Recursive DFS depth limit + safe fallback

**Problem:** Demonstrate `RecursionError` on a 2000-node path graph; implement
`dfs_safe` that falls back to iterative DFS for large graphs.

**Why it works:**
Python's default recursion limit is 1000. A path graph (0-1-2-...-1999) has DFS depth
equal to the number of nodes. At depth ~1000 the interpreter raises `RecursionError`.
The safe wrapper inspects graph size before choosing the strategy.

```python
"""
solutions/week-04/day-2-ex4.py — RecursionError demo + safe DFS fallback
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

Graph = dict[int, list[int]]

# Threshold at which we switch to iterative to avoid stack overflow
RECURSION_SAFE_THRESHOLD = 900


def build_path_graph(n: int) -> Graph:
    """Build a path graph: 0-1-2-...-n-1."""
    graph: Graph = defaultdict(list)
    for i in range(n):
        graph[i]  # ensure isolated last node exists
    for i in range(n - 1):
        graph[i].append(i + 1)
        graph[i + 1].append(i)
    return dict(graph)


# ---- Recursive DFS (unsafe on deep graphs) ----

def _dfs_recursive_inner(
    graph: Graph,
    node: int,
    visited: set[int],
    result: list[int],
) -> None:
    visited.add(node)
    result.append(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            _dfs_recursive_inner(graph, neighbor, visited, result)


def dfs_recursive(graph: Graph, start: int) -> list[int]:
    visited: set[int] = set()
    result: list[int] = []
    _dfs_recursive_inner(graph, start, visited, result)
    return result


# ---- Iterative DFS (safe for large graphs) ----

def dfs_iterative(graph: Graph, start: int) -> list[int]:
    visited: set[int] = set()
    stack: list[int] = [start]
    result: list[int] = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        result.append(node)
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)

    return result


# ---- Safe wrapper ----

def dfs_safe(graph: Graph, start: int) -> list[int]:
    """
    Use recursive DFS for small graphs (< RECURSION_SAFE_THRESHOLD nodes).
    Fall back to iterative DFS for larger graphs to avoid RecursionError.
    """
    n_nodes = len(graph)
    if n_nodes >= RECURSION_SAFE_THRESHOLD:
        log.info(
            "graph has %d nodes (>= threshold %d): using iterative DFS",
            n_nodes,
            RECURSION_SAFE_THRESHOLD,
        )
        return dfs_iterative(graph, start)
    else:
        log.info(
            "graph has %d nodes (< threshold %d): using recursive DFS",
            n_nodes,
            RECURSION_SAFE_THRESHOLD,
        )
        return dfs_recursive(graph, start)


def main() -> None:
    log.info("Python recursion limit: %d", sys.getrecursionlimit())

    # --- Demonstrate RecursionError ---
    n_large = 2000
    large_graph = build_path_graph(n_large)
    log.info("attempting recursive DFS on path graph with %d nodes...", n_large)
    try:
        result = dfs_recursive(large_graph, 0)
        log.info("recursive DFS completed — visited %d nodes", len(result))
    except RecursionError:
        log.warning(
            "RecursionError raised as expected on %d-node path graph", n_large
        )

    # --- Safe fallback works ---
    log.info("now using dfs_safe on %d-node path graph...", n_large)
    safe_result = dfs_safe(large_graph, 0)
    log.info(
        "dfs_safe completed — visited %d nodes (expected %d): %s",
        len(safe_result),
        n_large,
        len(safe_result) == n_large,
    )

    # --- Small graph still uses recursive ---
    small_graph = build_path_graph(10)
    log.info("dfs_safe on 10-node path graph:")
    small_result = dfs_safe(small_graph, 0)
    log.info("visited: %s", small_result)


if __name__ == "__main__":
    main()
```

**Expected output (key lines):**
```
INFO | Python recursion limit: 1000
INFO | attempting recursive DFS on path graph with 2000 nodes...
WARNING | RecursionError raised as expected on 2000-node path graph
INFO | graph has 2000 nodes (>= threshold 900): using iterative DFS
INFO | dfs_safe completed — visited 2000 nodes (expected 2000): True
INFO | graph has 10 nodes (< threshold 900): using recursive DFS
INFO | visited: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Key insight:** the threshold is conservative (900, not 999) because the real stack
depth also includes frames from `main` and `dfs_safe` itself. In production code, prefer
always using iterative DFS for graph algorithms — the recursive form is useful for
understanding but fragile at scale.
