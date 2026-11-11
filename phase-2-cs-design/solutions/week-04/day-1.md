# Solutions — Phase 2 · Week 04 · Day 1 — Graph Representations + BFS

## Exercise 1 — Add weighted edges

**Goal:** Store integer weights on edges; explain why BFS no longer gives shortest path.

```python
"""
Weighted graph: adjacency list stores (neighbor, weight) tuples.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WeightedGraph:
    directed: bool = False
    # node -> list of (neighbor, weight)
    adj: dict[str, list[tuple[str, int]]] = field(default_factory=dict)

    def add_vertex(self, v: str) -> None:
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u: str, v: str, weight: int = 1) -> None:
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def to_matrix(self) -> tuple[list[str], list[list[int]]]:
        """V x V matrix; cell = weight, 0 = no edge."""
        nodes: list[str] = sorted(self.adj.keys())
        index: dict[str, int] = {n: i for i, n in enumerate(nodes)}
        size = len(nodes)
        matrix: list[list[int]] = [[0] * size for _ in range(size)]
        for u, neighbors in self.adj.items():
            for v, w in neighbors:
                matrix[index[u]][index[v]] = w
        return nodes, matrix


def main() -> None:
    g = WeightedGraph(directed=False)
    g.add_edge("A", "B", weight=4)
    g.add_edge("A", "C", weight=1)
    g.add_edge("C", "B", weight=2)  # cheaper path A->C->B costs 3

    nodes, matrix = g.to_matrix()
    logger.info("Nodes: %s", nodes)
    for i, row in enumerate(matrix):
        logger.info("  %s: %s", nodes[i], row)

    # BFS from A to B would return ['A', 'B'] (1 hop).
    # But the minimum-COST path is A->C->B (cost 3 vs 4).
    # BFS counts hops, not weights, so it picks the wrong path here.
    # You need Dijkstra (greedy on cumulative cost) for weighted shortest paths.
    logger.info(
        "BFS hop-shortest A->B: ['A','B'] (1 hop, cost 4). "
        "True cheapest: ['A','C','B'] (2 hops, cost 3). "
        "BFS is wrong when edge weights differ."
    )


if __name__ == "__main__":
    main()
```

**Why it works:** The adjacency list now stores `(neighbor, weight)` pairs. `to_matrix()` writes weights instead of `1`. BFS still finds the fewest-hop path, but once weights are unequal, fewest hops != minimum cost — a 2-hop path at weight 1+2=3 beats a 1-hop path at weight 4. Dijkstra's algorithm uses a min-heap keyed on cumulative cost to fix this.

---

## Exercise 2 — Directed graph test

**Goal:** Confirm BFS path direction is respected; show path shortens when direct edge added.

```python
"""
Directed graph BFS verification.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Graph:
    directed: bool = False
    adj: dict[str, list[str]] = field(default_factory=dict)

    def add_vertex(self, v: str) -> None:
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u: str, v: str) -> None:
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)

    def bfs_shortest_path(self, source: str, target: str) -> list[str] | None:
        if source not in self.adj or target not in self.adj:
            return None
        if source == target:
            return [source]
        visited: set[str] = {source}
        queue: deque[tuple[str, list[str]]] = deque([(source, [source])])
        while queue:
            current, path = queue.popleft()
            for neighbor in self.adj[current]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == target:
                        return new_path
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        return None


def main() -> None:
    g = Graph(directed=True)
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")  # cycle back

    path_ac = g.bfs_shortest_path("A", "C")
    logger.info("A->C (no direct edge): %s", path_ac)
    # Expected: ['A', 'B', 'C']

    path_cb = g.bfs_shortest_path("C", "B")
    logger.info("C->B (must go C->A->B): %s", path_cb)
    # Expected: ['C', 'A', 'B']

    # Add direct A->C edge; BFS will now find it at level 1
    g.add_edge("A", "C")
    path_ac_short = g.bfs_shortest_path("A", "C")
    logger.info("A->C (with direct edge): %s", path_ac_short)
    # Expected: ['A', 'C']

    assert path_ac == ["A", "B", "C"], f"Unexpected: {path_ac}"
    assert path_cb == ["C", "A", "B"], f"Unexpected: {path_cb}"
    assert path_ac_short == ["A", "C"], f"Unexpected: {path_ac_short}"
    logger.info("All assertions passed.")


if __name__ == "__main__":
    main()
```

**Why it works:** In a directed graph, `add_edge("A", "B")` only inserts B into A's list, not the reverse. BFS follows outgoing edges only, so direction is enforced. After adding `A->C` directly, BFS visits C at level 1 (one hop) before it can reach C at level 2 via B, so the shorter path wins.

---

## Exercise 3 — BFS level map

**Goal:** Return a dict mapping every reachable vertex to its hop distance from source.

```python
"""
BFS level map: hop distance from a source vertex.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Graph:
    directed: bool = False
    adj: dict[str, list[str]] = field(default_factory=dict)

    def add_vertex(self, v: str) -> None:
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u: str, v: str) -> None:
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)


def bfs_levels(graph: Graph, source: str) -> dict[str, int]:
    """Return {vertex: hop_distance} for all vertices reachable from source."""
    if source not in graph.adj:
        logger.warning("Source %s not in graph", source)
        return {}

    levels: dict[str, int] = {source: 0}
    queue: deque[str] = deque([source])

    while queue:
        current = queue.popleft()
        current_level = levels[current]
        for neighbor in graph.adj[current]:
            if neighbor not in levels:
                levels[neighbor] = current_level + 1
                queue.append(neighbor)

    logger.info("Levels from %s: %s", source, levels)
    return levels


def main() -> None:
    g = Graph(directed=False)
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "E"), ("E", "F")]:
        g.add_edge(u, v)

    levels = bfs_levels(g, "A")
    # A=0, B=1, C=1, D=2, E=2, F=3
    assert levels["A"] == 0
    assert levels["B"] == 1
    assert levels["C"] == 1
    assert levels["D"] == 2
    assert levels["E"] == 2
    assert levels["F"] == 3
    logger.info("All level assertions passed.")


if __name__ == "__main__":
    main()
```

**Why it works:** The queue holds vertices to process. Each neighbor first encountered is assigned `parent_level + 1`. Because BFS processes nodes strictly in level order (FIFO), the first time a vertex is assigned a level it is always its minimum hop distance — no vertex is re-enqueued or re-assigned. Unreachable vertices simply never appear in the dict.

---

## Exercise 4 — Connected components

**Goal:** Partition all vertices of an undirected graph into connected components using BFS.

```python
"""
Connected components via BFS.
Python 3.12+, type hints, logging.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Graph:
    directed: bool = False
    adj: dict[str, list[str]] = field(default_factory=dict)

    def add_vertex(self, v: str) -> None:
        if v not in self.adj:
            self.adj[v] = []

    def add_edge(self, u: str, v: str) -> None:
        self.add_vertex(u)
        self.add_vertex(v)
        self.adj[u].append(v)
        if not self.directed:
            self.adj[v].append(u)


def connected_components(graph: Graph) -> list[set[str]]:
    """Return list of vertex sets; each set is one connected component."""
    visited: set[str] = set()
    components: list[set[str]] = []

    for start in graph.adj:
        if start in visited:
            continue
        # BFS from this unvisited vertex
        component: set[str] = set()
        queue: deque[str] = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in graph.adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
        logger.info("Found component: %s", component)

    return components


def main() -> None:
    g = Graph(directed=False)
    # Component 1: A-B-C
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    # Component 2: D-E
    g.add_edge("D", "E")
    # Isolated vertex
    g.add_vertex("F")

    comps = connected_components(g)
    logger.info("Total components: %d", len(comps))

    assert len(comps) == 3
    # Verify each expected set is in the result
    assert {"A", "B", "C"} in comps
    assert {"D", "E"} in comps
    assert {"F"} in comps
    logger.info("All component assertions passed.")


if __name__ == "__main__":
    main()
```

**Why it works:** The outer loop iterates every vertex exactly once as a potential starting point. If the vertex is already in `visited`, it belongs to a previously discovered component — skip it. Otherwise, run BFS from it, marking all reachable vertices visited and collecting them into one component set. Each BFS call exhausts exactly one connected component, so no vertex is counted twice and no component is split. Isolated vertices (no edges) produce singleton sets because their BFS loop body never enqueues anything.
