# Phase 2 · Week 04 · Day 5 — Union-Find / Disjoint Set

> Objective: Implement Union-Find with path compression and union by rank to track connected components in O(alpha(n)) time.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

Union-Find (Disjoint Set Union, DSU) answers one question efficiently: **are two elements in the same set?** It supports two operations:

- `find(x)` — return the *representative* (root) of the set containing `x`.
- `union(x, y)` — merge the sets containing `x` and `y`.

A set is represented as a rooted tree. All nodes point toward their root. Two nodes belong to the same component iff they share the same root.

Core insight: you never need the full tree structure. You only need to know "who is the boss of this element?" That minimal pointer is enough to merge groups and detect membership.

## How it works internally

### Naive array of parent pointers

Each element stores one integer: `parent[i]`. Initially `parent[i] = i` (every element is its own root).

`find(x)` walks up the chain: `x -> parent[x] -> parent[parent[x]] -> ...` until `parent[root] == root`.

Without optimizations this degrades to O(n) per operation on a chain-shaped tree.

### Quick-find vs Quick-union

| Variant | `find` | `union` |
|---|---|---|
| Quick-find | O(1) — store component id directly | O(n) — must update all members |
| Quick-union | O(n) worst case — walk to root | O(n) worst case |

Neither alone is satisfactory for large graphs.

### Path compression

During `find(x)`, after reaching the root, **point every node on the path directly to the root**. Future `find` calls on those nodes cost O(1).

Two styles:
- *Full path compression* (two-pass): walk up once to find root, walk again to rewire.
- *Path halving* (one-pass): set `parent[x] = parent[parent[x]]` at each step — achieves same amortized bound with one pass.

### Union by rank (or size)

When merging two trees, attach the *smaller* tree under the root of the *larger* tree. "Rank" is an upper bound on tree height; "size" tracks node count. Either keeps the tree shallow — O(log n) depth without compression.

### Combined complexity

Path compression + union by rank gives amortized **O(alpha(n))** per operation, where alpha is the inverse Ackermann function. For all practical n, alpha(n) <= 5. Treated as O(1) in practice.

### Applications

- **Connected components** (dynamic): add edges one at a time, query connectivity.
- **Cycle detection** in undirected graphs: if `find(u) == find(v)` before `union(u, v)`, edge (u, v) forms a cycle.
- **Kruskal's MST**: process edges by weight; skip edge if endpoints already connected (same component).
- **Duplicate connection detection**: if two nodes are already in the same set when a new edge arrives, the edge is redundant.

## Worked example

```python
"""
Union-Find with path compression + union by rank.

Use case: detect duplicate (redundant) connections in a network graph.
A connection is redundant if both endpoints are already in the same component
when the edge is processed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class UnionFind:
    """
    Disjoint Set Union with path compression and union by rank.

    Supports n elements indexed 0 .. n-1.
    """

    n: int
    parent: list[int] = field(init=False)
    rank: list[int] = field(init=False)
    _components: int = field(init=False)

    def __post_init__(self) -> None:
        self.parent = list(range(self.n))   # each node is its own root
        self.rank = [0] * self.n            # all trees have height 0
        self._components = self.n
        log.debug("UnionFind initialized with %d elements", self.n)

    # ------------------------------------------------------------------
    # find — path-halving variant (single pass, same amortized bound)
    # ------------------------------------------------------------------
    def find(self, x: int) -> int:
        """Return the representative of x's component."""
        while self.parent[x] != x:
            # path halving: jump x two levels up, rewire in one step
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    # ------------------------------------------------------------------
    # union — attach smaller-rank tree under larger-rank root
    # ------------------------------------------------------------------
    def union(self, x: int, y: int) -> bool:
        """
        Merge components of x and y.

        Returns True if they were in different components (merge happened),
        False if already connected (redundant edge / cycle detected).
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            log.debug("union(%d, %d): already connected (root=%d)", x, y, rx)
            return False  # redundant edge

        # attach smaller tree under larger
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx          # ensure rx has >= rank
        self.parent[ry] = rx         # ry's root now points to rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1       # only increment when ranks were equal
        self._components -= 1
        log.debug("union(%d, %d): merged, components=%d", x, y, self._components)
        return True

    def connected(self, x: int, y: int) -> bool:
        """True if x and y share the same component."""
        return self.find(x) == self.find(y)

    @property
    def components(self) -> int:
        """Current number of disjoint components."""
        return self._components


# ------------------------------------------------------------------
# Application: redundant connection detector
# ------------------------------------------------------------------
def find_redundant_connections(
    n_nodes: int,
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Given a graph with n_nodes (0-indexed) and a list of edges added in order,
    return all edges that form a cycle (redundant connections).

    Each redundant edge is one where both endpoints were already in the same
    component before the edge was processed.
    """
    uf = UnionFind(n_nodes)
    redundant: list[tuple[int, int]] = []

    for u, v in edges:
        merged = uf.union(u, v)
        if not merged:
            log.info("Redundant connection detected: (%d, %d)", u, v)
            redundant.append((u, v))

    return redundant


def main() -> None:
    # Network with 6 nodes, edges added one at a time
    # Nodes: 0=LA, 1=SF, 2=Seattle, 3=Denver, 4=Chicago, 5=NY
    edges: list[tuple[int, int]] = [
        (0, 1),  # LA -- SF
        (1, 2),  # SF -- Seattle
        (0, 3),  # LA -- Denver
        (3, 4),  # Denver -- Chicago
        (4, 5),  # Chicago -- NY
        (1, 3),  # SF -- Denver  (creates cycle: SF-LA-Denver)
        (2, 0),  # Seattle -- LA (creates cycle: Seattle-SF-LA)
        (3, 5),  # Denver -- NY  (creates cycle: Denver-Chicago-NY)
    ]

    log.info("Processing %d edges for 6-node network", len(edges))
    redundant = find_redundant_connections(6, edges)

    print("\n--- Results ---")
    print(f"Redundant connections ({len(redundant)}):")
    for u, v in redundant:
        print(f"  ({u}, {v})")

    # Verify final component count
    uf = UnionFind(6)
    for u, v in edges:
        uf.union(u, v)
    print(f"Final component count: {uf.components}")  # expect 1


if __name__ == "__main__":
    main()
```

Expected output (log lines at DEBUG level, then):
```
--- Results ---
Redundant connections (3):
  (1, 3)
  (2, 0)
  (3, 5)
Final component count: 1
```

## Exercises

1. **Weighted quick-union by size**: Rewrite `UnionFind` to use a `size` array instead of `rank`. In `union`, always attach the tree with fewer nodes under the one with more. Confirm the resulting tree depth stays O(log n) by printing `find` path lengths on a 1000-node chain.

2. **Dynamic connected components**: Given a stream of `(operation, u, v)` tuples where operation is either `"add"` or `"query"`, process each in order. For `"add"` call `union(u, v)`. For `"query"` print `"yes"` or `"no"` based on `connected(u, v)`. Use the `UnionFind` class from the worked example. Test with at least 10 operations including both types.

3. **Cycle detection**: Write a function `has_cycle(n_nodes: int, edges: list[tuple[int, int]]) -> bool` that returns `True` if any edge in `edges` creates a cycle in an undirected graph. It must stop at the first detected cycle without processing remaining edges. Add a unit test using `pytest` that covers: a tree (no cycle), a single cycle, and a graph with multiple components where one component has a cycle.

4. **Kruskal's MST sketch**: Given a weighted undirected graph as `(weight, u, v)` triples, implement Kruskal's algorithm using `UnionFind`. Sort edges by weight, then greedily add edges that do not form a cycle. Return the list of edges in the MST and the total weight. Test on a 5-node graph where you can verify the answer by hand.

(Solutions: ../solutions/week-04/day-5.md)

## Self-check

- [ ] I can explain why naive quick-union degrades to O(n) without optimizations.
- [ ] I can explain path compression and implement the path-halving variant from memory.
- [ ] I can explain union by rank and why it keeps tree depth at O(log n).
- [ ] I can implement the combined `UnionFind` class (with path compression + union by rank) without reference.
- [ ] I can use `UnionFind` to detect cycles in an undirected graph in O(E * alpha(n)) time.
- [ ] I can explain where `UnionFind` fits in Kruskal's MST algorithm.
