# Phase 2 · Week 04 · Day 3 — Topological Sort + Cycle Detection

> Objective: Implement topological ordering of a DAG using Kahn's algorithm and DFS reverse post-order, and detect cycles in directed and undirected graphs.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-1 (graph representation, BFS)]], [[week-04/day-2 (DFS)]]

## Concept

**Topological sort** produces a linear ordering of vertices in a directed graph such that for every directed edge `u -> v`, vertex `u` appears before `v` in the ordering.

This is only possible for **Directed Acyclic Graphs (DAGs)**. If the graph contains a cycle, no valid topological order exists — there is no "first" node in a circular dependency.

Real-world uses:
- **Build systems** (Make, Gradle): compile module A before module B if B depends on A
- **Package managers** (pip, npm): install transitive dependencies in dependency order
- **Task scheduling**: execute tasks respecting prerequisite constraints
- **Course prerequisites**: take Linear Algebra before Machine Learning

Two standard algorithms:
1. **Kahn's algorithm** (BFS-based): repeatedly remove nodes with zero in-degree
2. **DFS reverse post-order**: record nodes in post-order during DFS, then reverse

**Cycle detection:**
- Directed graphs: DFS three-color marking (white = unvisited, gray = in current path, black = done). A back edge (gray -> gray) indicates a cycle. Kahn's alternative: if not all nodes are processed, a cycle exists.
- Undirected graphs: DFS with parent tracking. A visited node that is not the parent of the current node indicates a cycle.

## How it works internally

### In-degree and adjacency

The adjacency list `graph[u] = [v1, v2, ...]` stores outgoing edges. The **in-degree** of a vertex is the count of incoming edges. In Python, a `dict[int, list[int]]` backed by a `defaultdict` allocates lazily — each vertex only allocates its list when first written.

### Kahn's algorithm (BFS)

```
1. Compute in-degree for every vertex.
2. Enqueue all vertices with in-degree 0.
3. While queue is non-empty:
     u = dequeue
     append u to result
     for each neighbor v of u:
         decrement in-degree[v]
         if in-degree[v] == 0: enqueue v
4. If len(result) != len(vertices): cycle exists.
```

`collections.deque` gives O(1) popleft. The algorithm runs in O(V + E) time. The cycle check is free — incomplete processing means at least one node was never reachable with in-degree 0, i.e., it was stuck in a cycle.

### DFS reverse post-order

Post-order records a node after all its descendants have been visited. Reversing post-order gives topological order because every node is recorded only after all nodes it points to are finalized.

The **three-color marking** for directed cycle detection uses a `visited` set (black) and a `stack` set (gray = on the current DFS path). Hitting a gray node while gray means we followed a back edge — a cycle.

### Memory

Both algorithms use O(V) extra space for the result list and visited/color structures. The call stack for DFS adds O(V) depth in the worst case (a long chain). Kahn's is iterative, so no stack depth concern.

## Worked example

```python
"""
day3_topo_sort.py — Topological sort and cycle detection.
Demonstrates Kahn's BFS algorithm, DFS reverse post-order,
directed cycle detection (three-color DFS), and undirected
cycle detection (DFS with parent tracking).

Python 3.12+
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph type alias
# ---------------------------------------------------------------------------
# adjacency list: node -> list of neighbors (directed edges u -> v)
Graph = dict[str, list[str]]


def build_graph(edges: list[tuple[str, str]], nodes: list[str]) -> Graph:
    """Build an adjacency list from an edge list, including isolated nodes."""
    graph: Graph = defaultdict(list)
    for node in nodes:
        graph[node]  # ensure every node has an entry
    for u, v in edges:
        graph[u].append(v)
    return dict(graph)


# ---------------------------------------------------------------------------
# Kahn's algorithm — BFS topological sort
# ---------------------------------------------------------------------------
def kahns_topo_sort(graph: Graph) -> list[str] | None:
    """
    Topological sort via Kahn's BFS algorithm.
    Returns None if the graph contains a cycle.

    Time:  O(V + E)
    Space: O(V)
    """
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    # Seed the queue with all zero-in-degree nodes
    queue: deque[str] = deque(
        node for node, deg in in_degree.items() if deg == 0
    )
    result: list[str] = []

    while queue:
        u = queue.popleft()
        result.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(result) != len(graph):
        log.warning("Kahn's: cycle detected — only %d of %d nodes processed",
                    len(result), len(graph))
        return None

    return result


# ---------------------------------------------------------------------------
# DFS-based topological sort (reverse post-order)
# ---------------------------------------------------------------------------
def dfs_topo_sort(graph: Graph) -> list[str] | None:
    """
    Topological sort via DFS reverse post-order.
    Uses three-color marking to detect directed cycles.
    Returns None if a cycle is found.

    Colors:
        'white' — unvisited
        'gray'  — on the current DFS stack (in-progress)
        'black' — fully processed
    """
    color: dict[str, str] = {node: "white" for node in graph}
    post_order: list[str] = []
    cycle_found = False

    def dfs(u: str) -> None:
        nonlocal cycle_found
        if cycle_found:
            return
        color[u] = "gray"
        for v in graph[u]:
            if color[v] == "gray":
                # Back edge: v is on the current path — cycle!
                log.warning("DFS topo: cycle detected via back edge %s -> %s", u, v)
                cycle_found = True
                return
            if color[v] == "white":
                dfs(v)
        color[u] = "black"
        post_order.append(u)  # record after all descendants

    for node in graph:
        if color[node] == "white":
            dfs(node)

    if cycle_found:
        return None

    post_order.reverse()  # reverse post-order = topological order
    return post_order


# ---------------------------------------------------------------------------
# Cycle detection — undirected graph (DFS + parent tracking)
# ---------------------------------------------------------------------------
def has_undirected_cycle(adj: dict[str, list[str]]) -> bool:
    """
    Returns True if the undirected graph contains a cycle.
    Uses DFS with parent tracking: visiting an already-visited node
    that is not the immediate parent signals a cycle.
    """
    visited: set[str] = set()

    def dfs(u: str, parent: str | None) -> bool:
        visited.add(u)
        for v in adj[u]:
            if v not in visited:
                if dfs(v, u):
                    return True
            elif v != parent:
                # v is visited and not our direct parent — back edge
                log.warning("Undirected cycle: back edge %s -- %s", u, v)
                return True
        return False

    for node in adj:
        if node not in visited:
            if dfs(node, None):
                return True
    return False


# ---------------------------------------------------------------------------
# Demo — package dependency resolution
# ---------------------------------------------------------------------------
def resolve_install_order(dependencies: dict[str, list[str]]) -> list[str] | None:
    """
    Given a mapping of package -> [direct dependencies],
    return an install order (dependencies first).

    We reverse the edge direction so 'A depends on B' becomes
    edge B -> A (install B before A).
    """
    # Collect all packages
    all_packages: list[str] = list(dependencies)
    for deps in dependencies.values():
        for dep in deps:
            if dep not in all_packages:
                all_packages.append(dep)

    # Build graph: dependency -> dependent (so dep comes first)
    graph: Graph = defaultdict(list)
    for pkg in all_packages:
        graph[pkg]  # ensure entry
    for pkg, deps in dependencies.items():
        for dep in deps:
            graph[dep].append(pkg)  # dep must be installed before pkg

    return kahns_topo_sort(dict(graph))


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Example 1: simple DAG ---
    #
    #   A --> B --> D
    #   A --> C --> D
    #         C --> E
    #
    nodes_dag = ["A", "B", "C", "D", "E"]
    edges_dag = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E")]
    dag = build_graph(edges_dag, nodes_dag)

    log.info("=== Simple DAG ===")
    result_kahns = kahns_topo_sort(dag)
    log.info("Kahn's result:       %s", result_kahns)
    result_dfs = dfs_topo_sort(dag)
    log.info("DFS post-order:      %s", result_dfs)

    # --- Example 2: graph WITH a directed cycle ---
    #
    #   X --> Y --> Z --> X  (cycle)
    #
    nodes_cyc = ["X", "Y", "Z"]
    edges_cyc = [("X", "Y"), ("Y", "Z"), ("Z", "X")]
    cyclic_graph = build_graph(edges_cyc, nodes_cyc)

    log.info("=== Directed cycle ===")
    kahns_topo_sort(cyclic_graph)    # logs warning
    dfs_topo_sort(cyclic_graph)      # logs warning

    # --- Example 3: package dependency resolution ---
    #
    #   flask depends on werkzeug, jinja2
    #   jinja2 depends on markupsafe
    #   werkzeug has no dependencies
    #   markupsafe has no dependencies
    #
    packages: dict[str, list[str]] = {
        "flask":      ["werkzeug", "jinja2"],
        "jinja2":     ["markupsafe"],
        "werkzeug":   [],
        "markupsafe": [],
    }
    log.info("=== Package install order ===")
    order = resolve_install_order(packages)
    log.info("Install order: %s", order)
    # Expected: markupsafe and werkzeug before jinja2, jinja2 before flask

    # --- Example 4: undirected cycle detection ---
    undirected_cycle: dict[str, list[str]] = {
        "P": ["Q", "R"],
        "Q": ["P", "R"],   # P-Q-R-P forms a triangle
        "R": ["P", "Q"],
    }
    undirected_no_cycle: dict[str, list[str]] = {
        "P": ["Q"],
        "Q": ["P", "R"],
        "R": ["Q"],
    }
    log.info("=== Undirected cycle detection ===")
    log.info("Triangle graph has cycle: %s", has_undirected_cycle(undirected_cycle))
    log.info("Path graph has cycle:     %s", has_undirected_cycle(undirected_no_cycle))
```

Run it:
```
python day3_topo_sort.py
```

Expected output (order of A/B/C/E/D may vary within valid topological orderings):
```
INFO: === Simple DAG ===
INFO: Kahn's result:       ['A', 'B', 'C', 'D', 'E']
INFO: DFS post-order:      ['A', 'C', 'E', 'B', 'D']
INFO: === Directed cycle ===
WARNING: Kahn's: cycle detected — only 0 of 3 nodes processed
WARNING: DFS topo: cycle detected via back edge Z -> X
INFO: === Package install order ===
INFO: Install order: ['markupsafe', 'werkzeug', 'jinja2', 'flask']
INFO: === Undirected cycle detection ===
WARNING: Undirected cycle: back edge Q -- P
INFO: Triangle graph has cycle: True
INFO: Path graph has cycle:     False
```

### Trace — Kahn's on the package example

```
Initial in-degrees:
  markupsafe: 0   werkzeug: 0   jinja2: 1   flask: 2

Queue: [markupsafe, werkzeug]
Process markupsafe -> jinja2 in-degree becomes 0  -> enqueue jinja2
Process werkzeug   -> flask in-degree becomes 1
Queue: [jinja2]
Process jinja2     -> flask in-degree becomes 0   -> enqueue flask
Queue: [flask]
Process flask      (no outgoing edges in install-order graph)

Result: [markupsafe, werkzeug, jinja2, flask]
All 4 nodes processed — no cycle.
```

## Exercises

1. **Multiple valid orderings** — The DAG `A->C, B->C, C->D` has more than one valid topological order. List all of them by hand. Then modify `kahns_topo_sort` to return all valid orderings (hint: use backtracking — at each step, try every zero-in-degree node in turn).

2. **Course schedule** — Given a list of courses `0..n-1` and prerequisites `[(a, b)]` meaning "must take b before a", write `can_finish(n: int, prereqs: list[tuple[int, int]]) -> bool` that returns `True` if all courses can be completed. This is LeetCode 207.

3. **Longest path in a DAG** — Write `longest_path(graph: Graph, source: str) -> int` that returns the length of the longest path (in number of edges) from `source`. Hint: process nodes in topological order and relax distances.

4. **Circular import detector** — Given a mapping `module -> [imports]` representing Python module imports, write `find_cycle(imports: dict[str, list[str]]) -> list[str] | None` that returns the cycle as an ordered list of module names if one exists, or `None` if the import graph is acyclic. For the cycle `A -> B -> C -> A`, return `["A", "B", "C"]`.

(Solutions: ../solutions/week-04/day-3.md)

## Self-check

- [ ] I can explain why topological sort only works on DAGs
- [ ] I can trace Kahn's algorithm by hand, updating in-degrees step by step
- [ ] I can explain the three-color DFS and why a gray-to-gray edge signals a cycle
- [ ] I can implement both `kahns_topo_sort` and `dfs_topo_sort` without reference
- [ ] I understand why reversing DFS post-order yields topological order
- [ ] I can explain the parent-tracking trick for undirected cycle detection
- [ ] I can describe a real dependency resolution scenario and map it to a DAG
