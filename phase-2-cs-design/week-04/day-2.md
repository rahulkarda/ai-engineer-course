# Phase 2 · Week 04 · Day 2 — DFS + Connected Components

> Objective: Implement recursive and iterative DFS; find connected components in undirected graphs.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

Depth-First Search (DFS) answers one question: which nodes are reachable from a starting node?
Strategy: commit to a path and go as deep as possible. Only backtrack when you hit a dead end (visited node or no unvisited neighbors). Then try the next unvisited neighbor.

Contrast with BFS, which fans out level by level. DFS dives.

Two immediate applications:
- **Reachability** — can you get from A to B?
- **Connected components** — how many disconnected subgraphs exist?

A **connected component** is a maximal set of nodes where every pair is connected by some path. An undirected graph with N nodes and zero edges has N components. A fully connected graph has 1.

## How it works internally

### Recursive DFS — call stack IS the stack

```
dfs(node):
    mark node visited
    for each neighbor:
        if not visited:
            dfs(neighbor)   # push frame; return = pop frame
```

The OS call stack holds the "to-do" state implicitly. Each recursive call is equivalent to pushing a node onto an explicit stack. Returning from a call is popping it.

**Risk:** Python's default recursion limit is 1000 (`sys.getrecursionlimit()`). A path graph with 10,000 nodes causes `RecursionError`. Mitigations: `sys.setrecursionlimit()` (fragile) or switch to iterative DFS.

**When recursive is fine:** sparse graphs, small N (< 500 depth), tree structures where the recursion models the problem naturally.

### Iterative DFS — explicit stack

Replace the call stack with a `list` used as a LIFO stack:

```
stack = [start]
while stack:
    node = stack.pop()          # LIFO — goes deep first
    if node in visited: continue
    mark visited
    push unvisited neighbors
```

Subtlety: neighbors are pushed in reverse order if you want left-to-right traversal order — popping reverses them. For connected-component counting, order doesn't matter.

**Memory:** both approaches are O(V + E) time, O(V) space (visited set + stack depth).

### Connected components

Run DFS from every unvisited node. Each fresh DFS invocation discovers one full component. Count the invocations.

```
components = 0
for each node:
    if not visited:
        dfs(node)       # marks all reachable nodes visited
        components += 1
```

### Strongly Connected Components (directed graphs)

In a **directed** graph, "reachable from A" and "reachable from B" are not symmetric. A Strongly Connected Component (SCC) is a maximal set where every node can reach every other node *following edge directions*.

**Kosaraju's algorithm** (two-pass DFS):
1. Run DFS on the original graph; record finish-time order (a stack).
2. Transpose the graph (reverse all edges).
3. Run DFS on the transposed graph in reverse finish-time order.
Each DFS invocation in pass 3 discovers one SCC.

Kosaraju is O(V + E). Tarjan's algorithm achieves the same in a single DFS pass using a low-link value. Both are beyond scope for today — come back after studying stacks and recursion deeply.

## Worked example

```python
"""
week-04/day-2-dfs.py

Demonstrates:
  - Recursive DFS
  - Iterative DFS
  - Connected component counting (undirected graph)

Graph used:
  0 - 1 - 2
  |
  3 - 4

  5   (isolated)

  6 - 7
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Generator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph representation: adjacency list (undirected)
# ---------------------------------------------------------------------------

Graph = dict[int, list[int]]


def build_graph(edges: list[tuple[int, int]], n_nodes: int) -> Graph:
    """Build an undirected adjacency list for nodes 0 .. n_nodes-1."""
    graph: Graph = defaultdict(list)
    # Ensure every node exists even if it has no edges (isolated nodes)
    for node in range(n_nodes):
        graph[node]  # touch key so defaultdict creates entry
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return dict(graph)


# ---------------------------------------------------------------------------
# Recursive DFS
# ---------------------------------------------------------------------------

def dfs_recursive(
    graph: Graph,
    node: int,
    visited: set[int],
    component: list[int],
) -> None:
    """
    Explore all nodes reachable from `node` via DFS, recording them in
    `component`. Modifies `visited` in-place.
    """
    visited.add(node)
    component.append(node)
    log.debug("recursive: visiting %d", node)

    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, component)


# ---------------------------------------------------------------------------
# Iterative DFS
# ---------------------------------------------------------------------------

def dfs_iterative(
    graph: Graph,
    start: int,
    visited: set[int],
) -> list[int]:
    """
    Explore all nodes reachable from `start` using an explicit stack.
    Returns list of nodes in DFS visit order.
    """
    stack: list[int] = [start]
    component: list[int] = []

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        component.append(node)
        log.debug("iterative: visiting %d", node)

        # Push neighbors in reverse so smallest index is processed first
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append(neighbor)

    return component


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def connected_components_recursive(graph: Graph) -> list[list[int]]:
    """Return a list of components; each component is a sorted list of nodes."""
    visited: set[int] = set()
    components: list[list[int]] = []

    for node in sorted(graph.keys()):
        if node not in visited:
            component: list[int] = []
            dfs_recursive(graph, node, visited, component)
            components.append(sorted(component))
            log.info("found component (recursive): %s", sorted(component))

    return components


def connected_components_iterative(graph: Graph) -> list[list[int]]:
    """Same as above but uses iterative DFS internally."""
    visited: set[int] = set()
    components: list[list[int]] = []

    for node in sorted(graph.keys()):
        if node not in visited:
            component = dfs_iterative(graph, node, visited)
            components.append(sorted(component))
            log.info("found component (iterative): %s", sorted(component))

    return components


# ---------------------------------------------------------------------------
# DFS path finder (bonus utility — recursive)
# ---------------------------------------------------------------------------

def find_path(
    graph: Graph,
    start: int,
    end: int,
    visited: set[int] | None = None,
) -> list[int] | None:
    """
    Return any path from `start` to `end`, or None if unreachable.
    Uses recursive DFS; returns the first path found (not necessarily shortest).
    """
    if visited is None:
        visited = set()

    visited.add(start)

    if start == end:
        return [start]

    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            path = find_path(graph, neighbor, end, visited)
            if path is not None:
                return [start] + path

    return None


# ---------------------------------------------------------------------------
# Main — tie it together
# ---------------------------------------------------------------------------

def main() -> None:
    edges: list[tuple[int, int]] = [
        (0, 1), (1, 2), (0, 3), (3, 4),  # component 1
        # node 5 is isolated                # component 2
        (6, 7),                             # component 3
    ]
    n_nodes = 8  # nodes 0-7
    graph = build_graph(edges, n_nodes)

    log.info("=== Graph adjacency list ===")
    for node, neighbors in sorted(graph.items()):
        log.info("  %d -> %s", node, neighbors)

    log.info("\n=== Recursive DFS from node 0 ===")
    visited_r: set[int] = set()
    comp: list[int] = []
    dfs_recursive(graph, 0, visited_r, comp)
    log.info("nodes reached from 0: %s", sorted(comp))

    log.info("\n=== Iterative DFS from node 0 ===")
    visited_i: set[int] = set()
    reached = dfs_iterative(graph, 0, visited_i)
    log.info("nodes reached from 0: %s", sorted(reached))

    log.info("\n=== Connected components (recursive) ===")
    rc = connected_components_recursive(graph)
    log.info("component count: %d", len(rc))
    for i, c in enumerate(rc):
        log.info("  component %d: %s", i, c)

    log.info("\n=== Connected components (iterative) ===")
    ic = connected_components_iterative(graph)
    log.info("component count: %d", len(ic))
    for i, c in enumerate(ic):
        log.info("  component %d: %s", i, c)

    log.info("\n=== Path finding: 4 -> 2 ===")
    path = find_path(graph, 4, 2)
    log.info("path: %s", path)

    log.info("\n=== Path finding: 4 -> 6 (different component) ===")
    path2 = find_path(graph, 4, 6)
    log.info("path: %s", path2)


if __name__ == "__main__":
    main()
```

Expected output (DEBUG lines omitted for brevity):
```
INFO | found component (recursive): [0, 1, 2, 3, 4]
INFO | found component (recursive): [5]
INFO | found component (recursive): [6, 7]
INFO | component count: 3
INFO | path: [4, 3, 0, 1, 2]
INFO | path: None
```

## Exercises

1. **Cycle detection.** Write `has_cycle(graph: Graph) -> bool` for an undirected graph.
   DFS-based: a cycle exists if you reach a node that is already visited *and* is not the
   direct parent you came from. Test on the example graph (it has a cycle: 0-1-2 does not,
   but 0-3-4 and back does not either — construct one that does and one that does not).

2. **Bipartite check.** A graph is bipartite if you can 2-color its nodes so no two adjacent
   nodes share a color. Write `is_bipartite(graph: Graph) -> bool` using iterative DFS (or
   BFS). Assign color 0 to the start node; each neighbor gets color 1 - current_color. If a
   neighbor already has the same color as the current node, it is not bipartite.

3. **Component sizes.** Modify `connected_components_iterative` to return
   `dict[int, int]` mapping component id (0-indexed) to its size, instead of the full node
   lists. Then write `largest_component(graph: Graph) -> int` that returns the size of the
   largest component.

4. **Recursive DFS depth limit.** Python's recursion limit is ~1000. Create a path graph
   (a line: 0-1-2-...-N) with N = 2000. Show that recursive DFS raises `RecursionError`.
   Then implement `dfs_safe(graph: Graph, start: int) -> list[int]` that falls back to
   iterative DFS automatically when the graph has more than 900 nodes, verified by running
   it on the 2000-node path graph without error.

(Solutions: ../solutions/week-04/day-2.md)

## Self-check

- [ ] I can explain why recursive DFS uses the call stack and what limits its depth
- [ ] I can trace iterative DFS step-by-step on a small graph using a written stack
- [ ] I can implement both recursive and iterative DFS from memory without reference
- [ ] I can explain why connected-component counting works by repeatedly starting fresh DFS from unvisited nodes
- [ ] I can describe in one sentence what Kosaraju's algorithm does differently for directed graphs
