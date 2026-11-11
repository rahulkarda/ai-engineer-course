# Solutions: Week 08 · Day 7 — Reflection Companion

Day 7 is a reflection and self-assessment day. There are no exercise solutions to provide here — the deliverables are the learner's completed pattern confidence table and time-discipline review.

## Reflection guidance

### Worked example: filling in the pattern inventory

Use the Day 6 session log. If you could write a correct solution within the time limit without significant stalling, rate 4 or 5. If you stalled more than 5 minutes on the setup or made a major logic error, rate 2 or 3.

**Common weak-pattern indicators:**
- Binary search: confusion over `<=` vs `<`, off-by-one on `mid` update
- DP: difficulty defining `dp[i]` in English before coding
- Backtracking: forgetting to `pop()` after recursion (un-choose step)
- Union-Find: not applying path compression; rank vs. size confusion

### Common mistake table — reference entries

| Mistake | Root cause | Fix |
|---|---|---|
| `while left <= right` in boundary search | wrong invariant | use `left < right` with exclusive `right` for boundary search |
| In-place grid mutation not restored in DFS | forgotten backtrack step | restore `board[r][c] = ch` after recursion |
| `dp[i] = dp[i-1]` without base case for i=0 | skipping boundary | always set `dp[0]` explicitly before the loop |
| Three-sum duplicates included | not skipping after finding triplet | advance left past consecutive equal values after recording |
| Trie `is_end` not set | forgot terminal marker | `node.is_end = True` at loop end, not inside loop |
| Heap on custom objects without `__lt__` | heapq comparison failure | define `__lt__` or use `(key, tie_breaker, obj)` tuples |

### Pattern-recognition flowchart — reference

```
Contiguous subarray / substring → Sliding Window
Sorted array, pair → Two Pointers
Find in sorted (or on answer space) → Binary Search
Cycle / middle / duplicate → Fast/Slow Pointers
Intervals → Sort by start + sweep (+ heap for rooms)
Shortest path / levels → BFS
All paths / connected components / cycle detection → DFS
Components / connected / MST → Union-Find
Prefix / word existence → Trie
Opt. substructure + overlapping subproblems → DP
Locally optimal = globally optimal → Greedy
```

---

No automated test for this file — reflection is self-assessed against the rubric in `assessment.md`.
