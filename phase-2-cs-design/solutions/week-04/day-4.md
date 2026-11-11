# Solutions — Phase 2 · Week 04 · Day 4 — Dijkstra's Shortest Path

> Reference: [[week-04/day-4]]

---

## Exercise 1 — Directed graph

**Task**: extend `build_graph` to accept `directed: bool`. When `True`, add only the forward edge.

```python
"""
sol_ex1_directed.py — Directed vs undirected graph with Dijkstra.
Python 3.12+
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[tuple[str, float]]]


def build_graph(
    edges: list[tuple[str, str, float]],
    directed: bool = False,
) -> Graph:
    """
    Build a weighted adjacency list.

    Args:
        edges:    List of (u, v, weight) triples.
        directed: If True add only u->v.  If False add both directions.
    """
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        if not directed:
            graph[v].append((u, w))
    # Ensure every mentioned vertex is a key even if it has no outgoing edges
    for u, v, _ in edges:
        graph.setdefault(v, [])
    return dict(graph)


def dijkstra(graph: Graph, source: str) -> dict[str, float]:
    """Return shortest distances from source to all reachable vertices."""
    dist: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get(u, []):
            if v in visited:
                continue
            new_dist = cost + weight
            if new_dist < dist.get(v, math.inf):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist


CITY_EDGES: list[tuple[str, str, float]] = [
    ("London",    "Amsterdam", 120.0),
    ("London",    "Paris",     450.0),
    ("Amsterdam", "Frankfurt", 210.0),
    ("Paris",     "Frankfurt", 340.0),
    ("Paris",     "Madrid",    740.0),
    ("Frankfurt", "Munich",    290.0),
    ("Munich",    "Vienna",    180.0),
]


if __name__ == "__main__":
    directed_graph = build_graph(CITY_EDGES, directed=True)
    dist = dijkstra(directed_graph, source="London")

    all_cities = {c for edge in CITY_EDGES for c in (edge[0], edge[1])}
    log.info("Directed graph from London:")
    for city in sorted(all_cities):
        d = dist.get(city, math.inf)
        status = f"{d:.0f} km" if d < math.inf else "UNREACHABLE"
        log.info("  %-20s  %s", city, status)
```

**Expected output**: `Madrid` is reachable (London->Paris->Madrid). All cities remain reachable because the directed edges all point *away* from London toward the listed destinations. If the edges were reversed (destination first), several cities would become unreachable from London.

**Key insight**: in a directed graph `build_graph` only adds `u -> v`. A city reachable undirectedly as `Vienna -> Munich -> ... -> London` would be unreachable directed if no return path exists.

---

## Exercise 2 — Disconnected graph

**Task**: handle isolated vertices; return all vertices in `distances` with `default_unreachable` for unreachable ones.

```python
"""
sol_ex2_disconnected.py — Dijkstra with disconnected vertices.
Python 3.12+
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[tuple[str, float]]]


def build_graph(edges: list[tuple[str, str, float]]) -> Graph:
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return dict(graph)


def dijkstra(
    graph: Graph,
    source: str,
    default_unreachable: float = math.inf,
) -> dict[str, float]:
    """
    Shortest distances from source to all vertices in the graph.

    Vertices unreachable from source are included in the result with
    value `default_unreachable` (default: math.inf).
    """
    dist: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get(u, []):
            if v in visited:
                continue
            new_dist = cost + weight
            if new_dist < dist.get(v, math.inf):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    # Fill in every vertex present in the graph
    all_vertices = set(graph.keys())
    for neighbours in graph.values():
        for v, _ in neighbours:
            all_vertices.add(v)

    for vertex in all_vertices:
        if vertex not in dist:
            dist[vertex] = default_unreachable

    return dist


CITY_EDGES: list[tuple[str, str, float]] = [
    ("London",    "Amsterdam", 120.0),
    ("London",    "Paris",     450.0),
    ("Amsterdam", "Frankfurt", 210.0),
    ("Paris",     "Frankfurt", 340.0),
    ("Paris",     "Madrid",    740.0),
    ("Frankfurt", "Munich",    290.0),
    ("Munich",    "Vienna",    180.0),
]


if __name__ == "__main__":
    graph = build_graph(CITY_EDGES)
    # Add isolated vertex (no edges)
    graph["Lisbon"] = []

    dist = dijkstra(graph, source="London", default_unreachable=math.inf)

    log.info("Results including isolated 'Lisbon':")
    for city, d in sorted(dist.items()):
        status = f"{d:.0f} km" if d < math.inf else "UNREACHABLE"
        log.info("  %-20s  %s", city, status)
```

**Expected output**: Lisbon shows as UNREACHABLE. All other cities have finite distances as before.

**Why no crash**: the algorithm only iterates over vertices it can actually reach via the heap. Isolated vertices are simply never pushed; the loop over `all_vertices` at the end sets their distance to the default.

---

## Exercise 3 — Top-K closest cities

**Task**: write `k_closest(graph, source, k)` returning the `k` nearest reachable cities (excluding source), sorted by distance.

```python
"""
sol_ex3_k_closest.py — K closest cities using Dijkstra.
Python 3.12+
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[tuple[str, float]]]


def build_graph(edges: list[tuple[str, str, float]]) -> Graph:
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return dict(graph)


def dijkstra(graph: Graph, source: str) -> dict[str, float]:
    dist: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, source)]
    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get(u, []):
            if v not in visited:
                new_dist = cost + weight
                if new_dist < dist.get(v, math.inf):
                    dist[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))
    return dist


def k_closest(graph: Graph, source: str, k: int) -> list[tuple[str, float]]:
    """
    Return the k nearest reachable vertices to source, excluding source itself.

    Args:
        graph:  Weighted adjacency list.
        source: Origin vertex.
        k:      Number of closest vertices to return.

    Returns:
        List of (vertex, distance) tuples sorted ascending by distance.
        If fewer than k reachable vertices exist, returns all reachable ones.
    """
    if k <= 0:
        return []

    dist = dijkstra(graph, source)
    # Exclude source itself; exclude unreachable (inf)
    reachable = [
        (vertex, d)
        for vertex, d in dist.items()
        if vertex != source and d < math.inf
    ]
    # Sort by distance, then alphabetically for stable output
    reachable.sort(key=lambda item: (item[1], item[0]))
    return reachable[:k]


CITY_EDGES: list[tuple[str, str, float]] = [
    ("London",    "Amsterdam", 120.0),
    ("London",    "Paris",     450.0),
    ("Amsterdam", "Frankfurt", 210.0),
    ("Paris",     "Frankfurt", 340.0),
    ("Paris",     "Madrid",    740.0),
    ("Frankfurt", "Munich",    290.0),
    ("Munich",    "Vienna",    180.0),
]


if __name__ == "__main__":
    graph = build_graph(CITY_EDGES)

    for k in (3, 10):
        result = k_closest(graph, source="London", k=k)
        log.info("Top-%d closest to London:", k)
        for rank, (city, d) in enumerate(result, start=1):
            log.info("  %d. %-20s  %.0f km", rank, city, d)
```

**Expected output (k=3)**:
```
Top-3 closest to London:
  1. Amsterdam             120 km
  2. Frankfurt             330 km
  3. Paris                 450 km
```

**Design note**: `k_closest` calls `dijkstra` once and slices the sorted result. This is correct and efficient — we need all distances anyway to guarantee the true top-k. An early-exit optimisation is possible (stop the heap loop once k vertices are settled) but adds complexity not warranted here.

---

## Exercise 4 — Negative-weight detection

**Task**: add a pre-flight check that raises `ValueError` for negative edge weights, with a pytest test.

```python
"""
sol_ex4_negative_check.py — Dijkstra with negative-weight guard + pytest tests.
Python 3.12+

Run tests:
    pytest sol_ex4_negative_check.py -v
"""

from __future__ import annotations

import heapq
import logging
import math
from collections import defaultdict

import pytest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[tuple[str, float]]]


def build_graph(edges: list[tuple[str, str, float]]) -> Graph:
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return dict(graph)


def dijkstra(graph: Graph, source: str) -> dict[str, float]:
    """
    Shortest distances from source using Dijkstra's algorithm.

    Raises:
        ValueError: if any edge weight in the graph is negative.
    """
    # --- Pre-flight: validate all edge weights ---
    for u, neighbours in graph.items():
        for v, weight in neighbours:
            if weight < 0:
                raise ValueError(
                    f"Negative edge weight detected: {u} -> {v} (weight={weight}). "
                    "Dijkstra requires non-negative weights. "
                    "Use Bellman-Ford for graphs with negative weights."
                )

    dist: dict[str, float] = {source: 0.0}
    visited: set[str] = set()
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        cost, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get(u, []):
            if v not in visited:
                new_dist = cost + weight
                if new_dist < dist.get(v, math.inf):
                    dist[v] = new_dist
                    heapq.heappush(heap, (new_dist, v))

    return dist


# ---------------------------------------------------------------------------
# pytest tests
# ---------------------------------------------------------------------------

def test_negative_edge_raises_value_error() -> None:
    """Dijkstra must raise ValueError when any edge has a negative weight."""
    bad_graph = build_graph([("A", "B", 5.0), ("B", "C", -3.0)])

    with pytest.raises(ValueError) as exc_info:
        dijkstra(bad_graph, source="A")

    msg = str(exc_info.value)
    assert "Negative edge weight" in msg, f"Unexpected message: {msg}"
    assert "Bellman-Ford" in msg, "Should mention Bellman-Ford as the alternative"


def test_negative_edge_message_contains_edge_info() -> None:
    """Error message should name the offending edge and its weight."""
    bad_graph = build_graph([("X", "Y", -7.5)])

    with pytest.raises(ValueError, match=r"weight=-7\.5"):
        dijkstra(bad_graph, source="X")


def test_zero_weight_edge_is_allowed() -> None:
    """Zero-weight edges are non-negative and must not raise."""
    graph = build_graph([("A", "B", 0.0), ("B", "C", 1.0)])
    dist = dijkstra(graph, source="A")
    assert dist["B"] == 0.0
    assert dist["C"] == 1.0


def test_positive_graph_works_correctly() -> None:
    """Sanity-check that the algorithm still produces correct results."""
    edges = [
        ("London",    "Amsterdam", 120.0),
        ("London",    "Paris",     450.0),
        ("Amsterdam", "Frankfurt", 210.0),
    ]
    graph = build_graph(edges)
    dist = dijkstra(graph, source="London")
    assert dist["Amsterdam"] == 120.0
    assert dist["Frankfurt"] == 330.0   # London -> Amsterdam -> Frankfurt


if __name__ == "__main__":
    # Quick smoke-test without pytest
    try:
        bad = build_graph([("A", "B", -1.0)])
        dijkstra(bad, "A")
    except ValueError as exc:
        log.info("Caught expected error: %s", exc)
```

Run tests:
```
pytest sol_ex4_negative_check.py -v
```

Expected:
```
PASSED test_negative_edge_raises_value_error
PASSED test_negative_edge_message_contains_edge_info
PASSED test_zero_weight_edge_is_allowed
PASSED test_positive_graph_works_correctly
```

**Design note**: the pre-flight check scans every edge in O(V + E) before the main O((V+E) log V) loop, so it does not change the asymptotic bound. Placing the check at the entry point makes the contract explicit and produces a helpful error rather than a silent wrong answer.

---

## Common mistakes to avoid

| Mistake | Effect | Fix |
|---------|--------|-----|
| Forgetting the `visited` check on pop | Processes stale heap entries as valid updates; wrong distances | Always skip if `u in visited` |
| Using `dist[u]` instead of `cost` after pop | Subtle: same value normally, but not if you modify `dist` later | Use `cost` from the heap tuple |
| Building the predecessor map from BFS | BFS ignores weights; paths will be wrong | Build `prev` inside Dijkstra during relaxation |
| Applying Dijkstra to a graph with negative edges | Greedy invariant breaks; incorrect shortest paths | Validate weights or use Bellman-Ford |
