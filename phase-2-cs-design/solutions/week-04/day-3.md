# Phase 2 · Week 04 · Day 3 — Solutions

> Solutions for [[week-04/day-3]] exercises: Topological Sort + Cycle Detection

---

## Exercise 1 — Multiple valid orderings

**By hand** for `A->C, B->C, C->D`:

Constraints: A before C, B before C, C before D. A and B have no ordering constraint between them.

Valid orderings:
1. `[A, B, C, D]`
2. `[B, A, C, D]`

**Code — all valid topological orderings via backtracking:**

```python
"""
ex1_all_topo_orders.py — All valid topological orderings of a DAG.
Python 3.12+
"""

from __future__ import annotations
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[str]]


def all_topo_orders(graph: Graph) -> list[list[str]]:
    """
    Return every valid topological ordering of the DAG.
    Uses backtracking: at each step, try every zero-in-degree node,
    remove it (decrement neighbors' in-degrees), recurse, then restore.

    Time: O(V! * (V + E)) worst case — only tractable for small graphs.
    """
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1

    results: list[list[str]] = []
    current: list[str] = []
    visited: set[str] = set()

    def backtrack() -> None:
        if len(current) == len(graph):
            results.append(list(current))
            return
        for node in graph:
            if node not in visited and in_degree[node] == 0:
                # Choose
                visited.add(node)
                current.append(node)
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                # Explore
                backtrack()
                # Undo
                for neighbor in graph[node]:
                    in_degree[neighbor] += 1
                current.pop()
                visited.remove(node)

    backtrack()
    return results


if __name__ == "__main__":
    graph: Graph = {
        "A": ["C"],
        "B": ["C"],
        "C": ["D"],
        "D": [],
    }
    orders = all_topo_orders(graph)
    log.info("All valid orderings (%d total):", len(orders))
    for order in orders:
        log.info("  %s", order)
    # Expected:
    #   ['A', 'B', 'C', 'D']
    #   ['B', 'A', 'C', 'D']
```

Key insight: the backtracking loop iterates over all nodes and only picks those with in-degree 0 that have not yet been placed. After placing a node, its outgoing edges are "removed" by decrementing neighbor in-degrees. Undoing restores those counts for the next branch.

---

## Exercise 2 — Course schedule (`can_finish`)

**Approach:** Build a directed graph where `prereq[1] -> prereq[0]` (must take b before a, so edge b -> a). Run Kahn's algorithm. If all courses are processed, no cycle exists and all courses can be finished.

```python
"""
ex2_course_schedule.py — Can all courses be completed given prerequisites?
LeetCode 207 equivalent.
Python 3.12+
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def can_finish(n: int, prereqs: list[tuple[int, int]]) -> bool:
    """
    Returns True if all n courses (0..n-1) can be completed.

    prereqs[i] = (a, b) means course b must be taken before course a.
    Equivalent to asking: is the prerequisite graph a DAG?

    Uses Kahn's algorithm: if a cycle exists, not all nodes will be
    processed (some will remain stuck with non-zero in-degree).

    Time:  O(V + E)
    Space: O(V + E)
    """
    graph: dict[int, list[int]] = defaultdict(list)
    in_degree: dict[int, int] = {i: 0 for i in range(n)}

    for course, prereq in prereqs:
        graph[prereq].append(course)  # prereq -> course
        in_degree[course] += 1

    queue: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)
    processed = 0

    while queue:
        u = queue.popleft()
        processed += 1
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return processed == n


if __name__ == "__main__":
    # No cycle — can finish
    log.info("2 courses, no cycle: %s", can_finish(2, [(1, 0)]))         # True
    # Cycle 0 -> 1 -> 0 — cannot finish
    log.info("2 courses, cycle:    %s", can_finish(2, [(1, 0), (0, 1)])) # False
    # 4 courses, valid prerequisite chain
    log.info("4 courses, chain:    %s",
             can_finish(4, [(1, 0), (2, 1), (3, 2)]))                    # True
    # 3 courses, partial cycle
    log.info("3 courses, partial cycle: %s",
             can_finish(3, [(0, 1), (1, 2), (2, 0)]))                    # False
```

Why Kahn's works cleanly here: any node permanently stuck with `in_degree > 0` is part of a cycle. We do not need to identify which cycle — just detect that one exists.

---

## Exercise 3 — Longest path in a DAG

**Approach:** Process nodes in topological order (Kahn's). Maintain a `dist[v]` = longest path length from `source` to `v`. When processing node `u`, for each neighbor `v`: `dist[v] = max(dist[v], dist[u] + 1)`.

This works because topological order guarantees we fully relax `u` before processing any node reachable from `u`.

```python
"""
ex3_longest_path.py — Longest path (edge count) from a source in a DAG.
Python 3.12+
"""

from __future__ import annotations
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[str]]


def longest_path(graph: Graph, source: str) -> int:
    """
    Returns the number of edges on the longest path from source
    in the given DAG.

    Unreachable nodes contribute 0 to the answer (dist stays -inf / 0).

    Time:  O(V + E)
    Space: O(V)
    """
    # Compute in-degrees (full graph, for topo sort)
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for node in graph:
        for nb in graph[node]:
            in_degree[nb] += 1

    # Kahn's topo sort order
    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    topo_order: list[str] = []
    in_deg_copy = dict(in_degree)  # work on a copy so original is preserved

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)

    # Relax distances in topological order
    dist: dict[str, int] = {node: 0 for node in graph}
    dist[source] = 0

    for u in topo_order:
        for v in graph[u]:
            if dist[u] + 1 > dist[v]:
                dist[v] = dist[u] + 1

    return max(dist.values())


if __name__ == "__main__":
    #   A -> B -> C -> E
    #   A -> D -> E
    # Longest from A: A->B->C->E = 3 edges
    g: Graph = {
        "A": ["B", "D"],
        "B": ["C"],
        "C": ["E"],
        "D": ["E"],
        "E": [],
    }
    result = longest_path(g, "A")
    log.info("Longest path from A: %d edges", result)  # 3

    # Linear chain: A->B->C->D
    chain: Graph = {"A": ["B"], "B": ["C"], "C": ["D"], "D": []}
    log.info("Longest path in chain: %d edges", longest_path(chain, "A"))  # 3
```

Edge case: if `source` has no outgoing path at all, the function returns 0 (the trivial path of zero edges). The algorithm naturally handles disconnected nodes — they retain `dist = 0` and do not affect the max for nodes reachable from source.

---

## Exercise 4 — Circular import detector

**Approach:** Run DFS with three-color marking, tracking the current DFS path in an explicit list. When a back edge (gray -> gray) is found, reconstruct the cycle from the path list by slicing from the first occurrence of the target node.

```python
"""
ex4_circular_import.py — Detect and return circular import chains.
Python 3.12+
"""

from __future__ import annotations
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def find_cycle(imports: dict[str, list[str]]) -> list[str] | None:
    """
    Returns the cycle as an ordered list of module names if the import
    graph contains one, otherwise returns None.

    For cycle A -> B -> C -> A, returns ['A', 'B', 'C'].

    Uses three-color DFS:
        white  — unvisited
        gray   — on the current DFS path (in-progress)
        black  — fully explored (no cycle through this node)

    When a gray node is reached again, the cycle is reconstructed
    by slicing the current path from the repeated node's first occurrence.
    """
    color: dict[str, str] = {mod: "white" for mod in imports}
    path: list[str] = []        # current DFS stack (ordered)
    cycle: list[str] | None = None

    def dfs(u: str) -> bool:
        """Returns True if a cycle was found rooted here."""
        nonlocal cycle
        color[u] = "gray"
        path.append(u)

        for v in imports.get(u, []):
            if color[v] == "gray":
                # Back edge found — reconstruct cycle
                cycle_start = path.index(v)
                cycle = path[cycle_start:]   # slice from first occurrence of v
                return True
            if color[v] == "white":
                if dfs(v):
                    return True

        color[u] = "black"
        path.pop()
        return False

    for mod in imports:
        if color[mod] == "white":
            if dfs(mod):
                return cycle

    return None


if __name__ == "__main__":
    # Simple 3-node cycle
    imports_cyclic: dict[str, list[str]] = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
        "D": ["A"],   # D imports A but is not in the cycle
    }
    result = find_cycle(imports_cyclic)
    log.info("Cycle in A->B->C->A graph: %s", result)   # ['A', 'B', 'C']

    # Acyclic import graph
    imports_clean: dict[str, list[str]] = {
        "app":    ["utils", "models"],
        "models": ["utils"],
        "utils":  [],
    }
    result2 = find_cycle(imports_clean)
    log.info("Cycle in clean graph: %s", result2)   # None

    # Self-import (trivial cycle)
    imports_self: dict[str, list[str]] = {
        "bad_module": ["bad_module"],
    }
    result3 = find_cycle(imports_self)
    log.info("Self-import cycle: %s", result3)   # ['bad_module']

    # Longer cycle buried in a larger graph
    imports_buried: dict[str, list[str]] = {
        "main":    ["auth"],
        "auth":    ["db"],
        "db":      ["cache"],
        "cache":   ["auth"],   # cache -> auth -> db -> cache cycle
        "logging": [],
    }
    result4 = find_cycle(imports_buried)
    log.info("Buried cycle: %s", result4)   # ['auth', 'db', 'cache']
```

### Why `path.index(v)` is correct

When the back edge `u -> v` is detected, `v` is already on the current DFS path (color is gray) because it was pushed when we first visited it and has not been popped yet. `path.index(v)` finds that earliest position. Slicing `path[cycle_start:]` extracts exactly the nodes from v to the current node u, which together form the cycle: v -> ... -> u -> v.

### Edge cases addressed

- **Self-import** (`"bad_module": ["bad_module"]`): `v == u` is pushed to path, then immediately found as gray when we iterate its neighbors.
- **Cycle not including the DFS start node**: handled because the path accumulates all visited nodes regardless of where DFS started — only the slice matters.
- **Multiple components**: the outer loop in `find_cycle` restarts DFS from any unvisited node, so isolated subgraphs are all checked.
