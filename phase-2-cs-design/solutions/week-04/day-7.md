# Week 04 · Day 7 — Solution (Project Day)

Day 7 is the project build day — there are no standalone day-level exercises. The complete, annotated reference implementation for the `graph_lib` package and the `DependencyResolver` use case is in:

**[../week-04/project.md](../../week-04/project.md)** (spec) → **[project.md](project.md)** (full reference implementation + 41 pytest cases)

## What the reference solution covers

- `graph_lib/graph.py` — `Graph` supporting directed/undirected, BFS, DFS (recursive + iterative), topological sort (Kahn's), Dijkstra with path reconstruction, connected components, cycle detection
- `graph_lib/resolver.py` — `DependencyResolver` producing a valid install order and raising `CycleError` with the offending cycle on circular dependencies
- `tests/` — 41 pytest cases across graph algorithms and the resolver

## Self-check against the reference

- [ ] My topological sort raises `CycleError` when `result length < node count`
- [ ] My Dijkstra reconstructs the actual path, not just the distance
- [ ] My resolver reports *which* nodes form the cycle, not just that one exists
- [ ] All public methods are type-annotated and covered by at least one test

See [project.md](project.md) for the line-by-line reference and common-mistake notes.
