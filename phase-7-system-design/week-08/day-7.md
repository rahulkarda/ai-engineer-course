# Phase 7 · Week 08 · Day 7 — Week Project + Reflection

> Objective: Build and test a clean timed problem-set solution suite with complexity analysis; consolidate the week.  ·  Time: ~3 hr project + 1 hr reflection  ·  Prereqs: [[day-6]]

## Project overview

See `project.md` for the full specification. This file is the reflection and wrap-up companion.

---

## Reflection protocol

Work through these after submitting your project:

### 1. Pattern inventory

For each pattern, rate your confidence: 1 (need review) → 5 (can teach it).

| Pattern | Confidence | Notes |
|---|---|---|
| Two pointers | | |
| Sliding window | | |
| Binary search variants | | |
| Intervals | | |
| Fast/slow pointers | | |
| BFS | | |
| DFS / backtracking | | |
| 1-D DP | | |
| 2-D DP / knapsack | | |
| Heaps / top-K | | |
| Tries | | |
| Union-Find | | |
| Greedy | | |

Patterns rated ≤ 3 go into next week's warm-up queue.

---

### 2. Time discipline review

From your Day 6 session log:
- Average time per easy problem:
- Average time per medium problem:
- Number of problems where you exceeded time limit:
- Primary bottleneck (setup? coding? testing?):

Target: easy ≤ 15 min, medium ≤ 25 min, hard ≤ 35 min.

---

### 3. Common mistakes log

Fill in any mistakes from the week:

| Problem | Mistake | Root cause | Fix |
|---|---|---|---|
| | off-by-one in binary search | wrong loop invariant | use `left < right`, exclusive `right` |
| | mutating input in DFS | side effects | copy or use visited set |
| | DP state not fully defined | ambiguous definition | write English definition before code |

---

### 4. What good solutions look like

Checklist for every interview solution you submit:

```
[ ] Restated the problem in my own words
[ ] Walked through at least one example manually
[ ] Named the brute-force approach and its complexity
[ ] Justified the chosen pattern
[ ] Coded with clean variable names and type hints
[ ] Stated O(time) and O(space) at the end
[ ] Walked through at least one edge case
```

---

## Concept consolidation

**Pattern recognition flowchart (mental model):**

```
Input type?
├── Array/string
│   ├── Contiguous subarray → Sliding window
│   ├── Sorted, pair sum → Two pointers
│   ├── Find value in sorted/rotated → Binary search
│   └── Max/min under constraint → Binary search on answer
├── Linked list
│   ├── Cycle / middle → Fast/slow pointers
│   └── Merge sorted → Heap (k-way) or two pointers
├── Tree / graph
│   ├── Shortest path / level order → BFS
│   ├── Path / connectivity / all solutions → DFS
│   ├── Cycle detection / components → Union-Find
│   └── Prefix queries → Trie
├── Optimal substructure → DP
│   ├── One sequence → 1-D DP
│   ├── Two sequences / grid → 2-D DP
│   └── Bag / items → Knapsack
└── Locally optimal = globally optimal → Greedy
    └── Scheduling / intervals → Sort + greedy sweep
```

---

## Looking ahead

Week 9 continues Phase 7 with system design depth: distributed caching, consistent hashing, and rate limiting design. The DSA patterns from this week apply in "design the algorithm component" portions of system design interviews.

Patterns to keep warm-up drilling next week: any rated ≤ 3 from your inventory above.

---

## Self-check

- [ ] I completed the project.md deliverables
- [ ] I filled in the pattern confidence table honestly
- [ ] I have a list of patterns to revisit
- [ ] I can draw the pattern-recognition flowchart from memory
- [ ] I am ready to cold-solve any medium problem in ≤ 25 min
