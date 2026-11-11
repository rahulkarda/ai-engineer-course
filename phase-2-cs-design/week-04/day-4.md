# Phase 2 · Week 04 · Day 4 — Dijkstra's Shortest Path

> Objective: Implement Dijkstra's algorithm to find shortest paths in a weighted graph using a min-heap.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-1 (graphs & BFS)]]

## Concept

**Single-source shortest path (SSSP)**: given a weighted graph and a source vertex `s`, find the minimum-cost path from `s` to every other reachable vertex.

BFS finds shortest paths in *unweighted* graphs by counting hops. In a weighted graph, the fewest-hops path is rarely the cheapest. Consider:

```
A --1--> B --1--> C
A ------100------> C
```

BFS (treating both edges equal) reports `A→C` cost 1. With weights the correct answer is `A→B→C` (cost 2 vs 100). BFS has no concept of edge cost; it just counts levels.

**Dijkstra's algorithm** solves SSSP for graphs with *non-negative* edge weights. It is a **greedy** algorithm: at each step it finalises the globally cheapest not-yet-settled vertex, then relaxes its outgoing edges.

### Core invariant

When a vertex `u` is popped from the min-heap, its recorded distance `dist[u]` is guaranteed to be the true shortest distance from the source. This holds because all edge weights are >= 0, so no future path through an unvisited vertex can be cheaper.

### Time complexity

| Heap type | Time |
|-----------|------|
| Binary heap (Python `heapq`) | O((V + E) log V) |
| Fibonacci heap | O(E + V log V) (theoretical) |

`heapq` is the practical choice in Python; Fibonacci heaps have large constants.

### Limitations

Dijkstra **requires non-negative weights**. A negative-weight edge can make an already-settled vertex reachable at a lower cost, breaking the invariant. For graphs with negative weights use **Bellman-Ford** (O(VE)) or, for all-pairs, **Floyd-Warshall** (O(V^3)).

## How it works internally

### Min-heap mechanics

Python's `heapq` is a **binary min-heap** over a flat list. `heappush` and `heappop` run in O(log n). Each entry is a tuple `(distance, vertex)`; tuple comparison uses the first element first, so the vertex with the smallest tentative distance is always at index 0.

### Lazy deletion

`heapq` has no O(log n) decrease-key. Instead, when a better distance is found for vertex `v`, a *new* tuple `(better_dist, v)` is pushed onto the heap — the stale tuple stays. When the stale tuple is eventually popped, `dist[v]` will already be smaller, so we skip it. This "lazy" approach keeps the heap simple at the cost of extra entries (at most one per edge), which does not change the asymptotic bound.

### Memory layout

- `dist: dict[str, float]` — current best distance from source to each vertex; initialised to `math.inf`.
- `prev: dict[str, str | None]` — predecessor map for path reconstruction; walking backwards from a destination gives the shortest path.
- `heap: list[tuple[float, str]]` — entries `(cost, vertex)`.
- `visited: set[str]` — settled vertices; a vertex is settled once it is popped and not stale.

## Worked example

```python
"""
day4_dijkstra.py — Dijkstra's shortest-path algorithm.
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


# ---------------------------------------------------------------------------
# Graph type: adjacency list keyed by city name.
# Each neighbour is (neighbour_name, edge_weight).
# ---------------------------------------------------------------------------
Graph = dict[str, list[tuple[str, float]]]


class ShortestPaths(NamedTuple):
    """Result container returned by dijkstra()."""
    distances: dict[str, float]           # source -> vertex shortest distance
    paths: dict[str, list[str]]           # source -> vertex full path as list


def build_graph(edges: list[tuple[str, str, float]]) -> Graph:
    """Build an undirected weighted adjacency list from a list of (u, v, w) triples."""
    graph: Graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    return dict(graph)


def dijkstra(graph: Graph, source: str) -> ShortestPaths:
    """
    Compute shortest distances and paths from *source* to all reachable vertices.

    Args:
        graph:  Weighted adjacency list.  All weights must be >= 0.
        source: Starting vertex.

    Returns:
        ShortestPaths with distances dict and paths dict.
        Unreachable vertices are absent from both dicts.
    """
    # --- initialise ---
    dist: dict[str, float] = {source: 0.0}
    prev: dict[str, str | None] = {source: None}
    visited: set[str] = set()

    # Min-heap entries: (tentative_distance, vertex)
    heap: list[tuple[float, str]] = [(0.0, source)]

    log.debug("Starting Dijkstra from '%s'", source)

    while heap:
        cost, u = heapq.heappop(heap)

        # Lazy deletion: skip stale entries
        if u in visited:
            log.debug("  Skip stale entry (%s, %s)", cost, u)
            continue

        visited.add(u)
        log.debug("  Settle '%s' at cost %.1f", u, cost)

        for v, weight in graph.get(u, []):
            if v in visited:
                continue
            new_dist = dist[u] + weight
            if new_dist < dist.get(v, math.inf):
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
                log.debug("    Relax edge %s->%s: new dist %.1f", u, v, new_dist)

    # --- reconstruct paths ---
    paths: dict[str, list[str]] = {}
    for vertex in dist:
        path: list[str] = []
        node: str | None = vertex
        while node is not None:
            path.append(node)
            node = prev.get(node)
        paths[vertex] = list(reversed(path))

    return ShortestPaths(distances=dist, paths=paths)


def print_results(result: ShortestPaths, source: str) -> None:
    """Log all shortest paths and distances in a readable table."""
    log.info("Shortest paths from '%s':", source)
    log.info("  %-20s  %-8s  %s", "Destination", "Distance", "Path")
    log.info("  %s", "-" * 55)
    for dest, distance in sorted(result.distances.items()):
        path_str = " -> ".join(result.paths[dest])
        log.info("  %-20s  %-8.1f  %s", dest, distance, path_str)


# ---------------------------------------------------------------------------
# Example network: European city distances (km, simplified)
#
#  London --120-- Amsterdam
#    |               |
#   450             210
#    |               |
#  Paris  --340-- Frankfurt
#    |               |
#   740            290
#    |               |
#  Madrid         Munich --180-- Vienna
#
# ---------------------------------------------------------------------------
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
    result = dijkstra(graph, source="London")
    print_results(result, source="London")

    # Spot-check one path
    dest = "Vienna"
    log.info(
        "\nLondon -> Vienna: %.0f km via %s",
        result.distances[dest],
        " -> ".join(result.paths[dest]),
    )
```

Run it:
```
python day4_dijkstra.py
```

Expected output (distances in km):
```
INFO: Shortest paths from 'London':
INFO:   Destination           Distance  Path
INFO:   -------------------------------------------------------
INFO:   Amsterdam             120.0     London -> Amsterdam
INFO:   Frankfurt             330.0     London -> Amsterdam -> Frankfurt
INFO:   London                0.0       London
INFO:   Madrid                1190.0    London -> Paris -> Madrid
INFO:   Munich                620.0     London -> Amsterdam -> Frankfurt -> Munich
INFO:   Paris                 450.0     London -> Paris
INFO:   Vienna                800.0     London -> Amsterdam -> Frankfurt -> Munich -> Vienna

INFO: London -> Vienna: 800 km via London -> Amsterdam -> Frankfurt -> Munich -> Vienna
```

### Trace: first three heap pops

```
Initial heap: [(0.0, 'London')]

Pop (0.0, 'London')  -> settle London
  Relax London->Amsterdam: dist[Amsterdam] = 120  push (120, Amsterdam)
  Relax London->Paris:     dist[Paris]     = 450  push (450, Paris)
  Heap: [(120, Amsterdam), (450, Paris)]

Pop (120, 'Amsterdam')  -> settle Amsterdam
  London already visited; skip
  Relax Amsterdam->Frankfurt: dist[Frankfurt] = 120+210 = 330  push (330, Frankfurt)
  Heap: [(330, Frankfurt), (450, Paris)]

Pop (330, 'Frankfurt')  -> settle Frankfurt
  Amsterdam already visited; skip
  Relax Frankfurt->Paris: 330+340 = 670 > 450 (current); no update
  Relax Frankfurt->Munich: dist[Munich] = 330+290 = 620  push (620, Munich)
  Heap: [(450, Paris), (620, Munich)]
```

## Exercises

1. **Directed graph** — modify `build_graph` to accept a `directed: bool` parameter. When `True`, add only the forward edge. Re-run the city example as a directed graph (all edges one-way from the first city to the second). Which destinations become unreachable from London?

2. **Disconnected graph** — add an isolated vertex `"Lisbon"` to the graph (no edges). Confirm that `dijkstra` returns no entry for `"Lisbon"` in `distances` without crashing. Then extend the function to accept an optional `default_unreachable: float = math.inf` and include *all* vertices from the graph in the returned `distances` dict, using the default for unreachable ones.

3. **Top-K closest cities** — given a source and integer `k`, write `k_closest(graph, source, k) -> list[tuple[str, float]]` that returns the `k` nearest reachable cities (excluding the source) sorted by distance, using `dijkstra` internally.

4. **Negative-weight detection** — add a pre-flight check to `dijkstra` that raises `ValueError` with a descriptive message if any edge weight in the graph is negative. Write a pytest test that confirms the exception is raised with the correct message for a graph containing a negative edge.

(Solutions: ../solutions/week-04/day-4.md)

## Self-check

- [ ] I can explain why BFS fails for weighted shortest paths
- [ ] I can trace the first three iterations of Dijkstra by hand on a small graph
- [ ] I can implement `dijkstra` from scratch using `heapq` without reference
- [ ] I understand why lazy deletion is used instead of decrease-key
- [ ] I can reconstruct the actual path (not just the distance) using a predecessor map
- [ ] I can state the time complexity O((V + E) log V) and justify it

## Supplementary links

- [Visualgo — Dijkstra](https://visualgo.net/en/sssp)
- [Python `heapq` docs](https://docs.python.org/3.12/library/heapq.html)
- [Wikipedia — Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [CP-Algorithms — Dijkstra](https://cp-algorithms.com/graph/dijkstra.html)
