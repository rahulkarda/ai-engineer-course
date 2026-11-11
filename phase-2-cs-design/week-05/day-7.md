# Phase 2 · Week 05 · Day 7 (Sunday) — Week Project + Assessment

> Objective: Integrate the week's learning into a packaged DP problem-set library with both solution strategies and performance comparison.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-1]] through [[day-6]]

## Project brief (see [[project.md]] for full spec)

Today: complete the week project (if not already done), then sit the assessment.

**Time split**: 3 hr project work → 1 hr assessment (timed, no reference).

## Project review checklist

Before the assessment, verify your project satisfies:

- [ ] All five algorithms implemented in both memoized and tabulated forms.
- [ ] `complexity.py` table populated with measured timings and theoretical bounds.
- [ ] Tests passing: `pytest tests/ -v` shows all green.
- [ ] Type hints on all public functions.
- [ ] `logging` used (no bare `print` statements).
- [ ] `README.md` documents how to run.

## Assessment

See [[assessment.md]] for the full graded assessment.

## Week review — key concepts

**Recursion**: base case + recursive case. Backtracking = recursion + undo. Call stack depth is the main constraint.

**Memoization (top-down)**: cache pure function results. O(unique sub-problems) time after caching. Works when function arguments are hashable.

**Tabulation (bottom-up)**: fill a table from base cases up. No stack risk. Space optimization possible by keeping only needed rows/columns.

**Core DP archetypes**:
- Knapsack: take/skip decision at each item
- String alignment: LCS, edit distance — 2-D table on (i,j) prefixes
- Sequence: LIS — 1-D table, depends on all previous entries

**Greedy**: locally optimal = globally optimal only with proof (exchange argument + optimal substructure + greedy-choice property). Works for fractional knapsack, interval scheduling; fails for 0/1 knapsack.

**Divide and conquer**: split into *independent* sub-problems; combine results. Master theorem for complexity.

## Complexity reference

| Algorithm | Time | Space |
|---|---|---|
| Fibonacci (memoized/tabulated) | O(n) | O(n) / O(1) |
| 0/1 Knapsack | O(n × W) | O(n × W) → O(W) |
| LCS | O(m × n) | O(m × n) → O(min(m,n)) |
| Edit distance | O(m × n) | O(m × n) → O(min(m,n)) |
| LIS (DP) | O(n²) | O(n) |
| LIS (patience sort) | O(n log n) | O(n) |
| Merge sort | O(n log n) | O(n) |
| Interval scheduling (greedy) | O(n log n) | O(n) |

## Supplementary links

- "Dynamic Programming is not Black Magic" — Jonathan Paulson: https://codeforces.com/blog/entry/43256
- DP patterns cheat sheet: https://leetcode.com/discuss/study-guide/458695/
- CLRS Chapter 15 (Dynamic Programming): available at most university libraries
