# Phase 2 · Week 04 · Day 1 — Graph Representations + BFS

> Objective: Understand graph theory fundamentals, implement adjacency list and matrix representations, and traverse graphs with BFS to find shortest paths.  ·  Time: ~1 hr  ·  Prereqs: [[day-3-week-03]]

## Concept

A **graph** `G = (V, E)` consists of a set of **vertices** (nodes) `V` and a set of **edges** `E` connecting pairs of vertices.

**Key distinctions:**

- **Directed** (digraph): edges have direction; `(u, v)` means u points to v, but not necessarily v to u. Example: web links, dependency graphs.
- **Undirected**: edges are bidirectional; `(u, v)` and `(v, u)` are the same edge. Example: road networks, social friendships.
- **Weighted**: each edge carries a numeric cost/distance. Example: map distances, network latency.
- **Unweighted**: all edges treated as equal cost (or cost = 1).

**Degree:**
- Undirected graph: degree of a vertex = number of edges incident to it.
- Directed graph: **in-degree** = edges arriving; **out-degree** = edges leaving.

**Two standard representations:**

| | Adjacency List | Adjacency Matrix |
|---|---|---|
| Space | O(V + E) | O(V^2) |
| Edge lookup | O(degree(v)) | O(1) |
| Iterate neighbors | O(degree(v)) | O(V) |
| Best for | Sparse graphs | Dense graphs, fast lookup |

Adjacency list uses a dict mapping each node to its list of neighbors. Adjacency matrix uses a V x V grid where `matrix[i][j] = 1` (or weight) if edge exists.

## How it works internally

**Adjacency list — memory layout:**
Each key in the dict holds a reference to a list on the heap. Total pointers = 2E (undirected) or E (directed). For a sparse graph with millions of nodes but few edges, this avoids allocating V^2 entries.

**Adjacency matrix — memory layout:**
A contiguous 2D array. Row `i` stores all possible destinations from vertex `i`. Index math is O(1): `matrix[i][j]` is a single memory access. Cache-friendly for dense traversal, but wastes space if E << V^2.

**BFS — why it finds shortest paths:**
BFS explores level by level via a FIFO queue. Level 0 = source. Level 1 = all direct neighbors. Level k = all nodes reachable in exactly k hops. Because each node is first visited at its minimum hop distance, the path recorded at first visit is guaranteed shortest (in unweighted graphs). This guarantee breaks with weights — use Dijkstra instead.

**BFS complexity:** O(V + E) — each vertex enqueued once, each edge examined once.

## Worked example

```python
"""
Graph representations (adjacency list + matrix) and BFS shortest path.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Graph:
    """Undirected, unweighted graph with dual representation."""

    directed: bool = False
    # adjacency list: node -> list of neighbors
    adj: dict[str, list[str]] = field(default_factory=dict)

    def add_vertex(self, v: str) -> None:
        if v not in self.adj:
            self.adj[v] = []
            logger.debug("Added vertex %s", v)

    def add_edge(self, u: str, v: str) -> None:
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)
        logger.debug("Added edge %s -> %s (directed=%s)", u, v, self.directed)

    # ------------------------------------------------------------------ #
    # Adjacency matrix (derived from adj list for illustration)
    # ------------------------------------------------------------------ #
    def to_matrix(self) -> tuple[list[str], list[list[int]]]:
        """Return (ordered node list, V x V matrix) derived from adj list."""
        nodes: list[str] = sorted(self.adj.keys())
        index: dict[str, int] = {n: i for i, n in enumerate(nodes)}
        size = len(nodes)
        matrix: list[list[int]] = [[0] * size for _ in range(size)]
        for u, neighbors in self.adj.items():
            for v in neighbors:
                matrix[index[u]][index[v]] = 1
        return nodes, matrix

    # ------------------------------------------------------------------ #
    # BFS — returns shortest path (by hop count) or None if unreachable
    # ------------------------------------------------------------------ #
    def bfs_shortest_path(self, source: str, target: str) -> list[str] | None:
        """BFS from source to target; returns node list or None."""
        if source not in self.adj or target not in self.adj:
            logger.warning("Source or target not in graph")
            return None

        if source == target:
            return [source]

        visited: set[str] = {source}
        # Queue stores (current_node, path_so_far)
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            logger.debug("Visiting %s | path so far: %s", current, path)

            for neighbor in self.adj[current]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == target:
                        logger.info("Path found: %s", new_path)
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        logger.info("No path from %s to %s", source, target)
        return None


# ------------------------------------------------------------------ #
# Demo
# ------------------------------------------------------------------ #
def main() -> None:
    g = Graph(directed=False)
    # Build a simple undirected graph:
    #   A - B - D
    #   |   |
    #   C - E - F
    edges: list[tuple[str, str]] = [
        ("A", "B"), ("A", "C"),
        ("B", "D"), ("B", "E"),
        ("C", "E"), ("E", "F"),
    ]
    for u, v in edges:
        g.add_edge(u, v)

    # Adjacency list
    logger.info("Adjacency list: %s", dict(g.adj))

    # Adjacency matrix
    nodes, matrix = g.to_matrix()
    logger.info("Node order: %s", nodes)
    for i, row in enumerate(matrix):
        logger.info("  %s: %s", nodes[i], row)

    # BFS shortest path
    result = g.bfs_shortest_path("A", "F")
    logger.info("Shortest path A -> F: %s", result)   # ['A', 'B', 'E', 'F'] or ['A', 'C', 'E', 'F']

    result2 = g.bfs_shortest_path("D", "C")
    logger.info("Shortest path D -> C: %s", result2)  # ['D', 'B', 'A', 'C'] or ['D', 'B', 'E', 'C']


if __name__ == "__main__":
    main()
```

**Expected output (paths are level-optimal):**
- `A -> F`: length 3, e.g. `['A', 'B', 'E', 'F']`
- `D -> C`: length 3, e.g. `['D', 'B', 'A', 'C']`

## Exercises

1. **Add weighted edges.** Extend the `Graph` dataclass so each edge stores an integer weight. Update `add_edge(u, v, weight=1)` and `to_matrix()` to store weights instead of `1`. Keep the unweighted BFS unchanged — explain in a comment why BFS hop-count shortest path no longer applies once weights differ.

2. **Directed graph test.** Instantiate `Graph(directed=True)`. Add edges `A->B`, `B->C`, `C->A` (a cycle). Verify that `bfs_shortest_path("A", "C")` returns `['A', 'B', 'C']` and `bfs_shortest_path("C", "B")` returns `['C', 'A', 'B']`. Then add edge `A->C` and confirm the path shortens to `['A', 'C']`.

3. **BFS level map.** Write a function `bfs_levels(graph: Graph, source: str) -> dict[str, int]` that returns a dict mapping each reachable vertex to its hop distance from `source`. Return `{source: 0}` as the base case. Do not use the path-tracking approach from the worked example — track levels separately.

4. **Connected components.** Write a function `connected_components(graph: Graph) -> list[set[str]]` that returns a list of sets, each set being one connected component. Use BFS (or DFS — your choice) as the inner traversal. Test it on a graph with two disconnected subgraphs.

(Solutions: ../solutions/week-04/day-1.md)

## Self-check

- [ ] I can explain the difference between directed and undirected graphs and give a real-world example of each.
- [ ] I can describe when to prefer an adjacency list over an adjacency matrix and justify the space complexity of each.
- [ ] I can trace BFS by hand on a small graph and identify which nodes are at each level.
- [ ] I can implement BFS that returns a shortest path without reference.
- [ ] I can explain why BFS guarantees shortest hop-count path on an unweighted graph but not on a weighted one.
