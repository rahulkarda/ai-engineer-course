# Phase 7 · Week 08 · Day 6 — 4-Hour Timed Problem Set

> Objective: Solve 10 problems across all patterns under real interview time limits.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## How to run this session

This is deliberate practice, not study. Simulate real interview conditions:

- **No IDE autocomplete.** Write in a plain editor or on paper.
- **No Googling** during the 25-minute window for each problem.
- **Timer is mandatory** — start it before reading the problem.
- **Follow the 5-step protocol** from Day 1 for every problem, even if it costs time.
- After each problem, record: time taken, pattern used, what slowed you down.

After all 10, compare your solutions with `../solutions/week-08/day-6.md`. Analyse misses, not just correctness.

---

## Time budget

| Slot | Problems | Time limit |
|---|---|---|
| Warm-up | P1–P2 | 15 min each |
| Core | P3–P8 | 25 min each |
| Hard | P9–P10 | 35 min each |
| **Total** | | **~4 hr** |

---

## Problem Set

### P1 — Valid Palindrome II (Two Pointers) [15 min]

**Prompt:** Given a string `s`, return `True` if it can become a palindrome by removing **at most one** character.

**Constraints:** `1 ≤ len(s) ≤ 10⁵`, lowercase letters only.

**Examples:**
```
s = "aba"  → True   (already palindrome)
s = "abca" → True   (remove 'c')
s = "abc"  → False
```

**Acceptance criteria:** O(n) time, O(1) extra space.

---

### P2 — Best Time to Buy and Sell Stock (Sliding Window / Greedy) [15 min]

**Prompt:** Given prices array, find max profit from one buy and one sell (buy before sell). Return 0 if no profit.

**Constraints:** `1 ≤ len(prices) ≤ 10⁵`.

**Examples:**
```
[7,1,5,3,6,4] → 5   (buy at 1, sell at 6)
[7,6,4,3,1]   → 0   (no profit possible)
```

**Acceptance criteria:** O(n) time, O(1) space.

---

### P3 — Find Peak Element (Binary Search) [25 min]

**Prompt:** A peak element is strictly greater than its neighbours. Given array where `nums[-1] = nums[n] = -∞`, find **any** peak index. Array may have multiple peaks.

**Constraints:** O(log n) required.

**Examples:**
```
[1,2,3,1]     → 2
[1,2,1,3,5,6,4] → 1 or 5
```

---

### P4 — Insert Interval (Intervals) [25 min]

**Prompt:** Given a sorted list of non-overlapping intervals and a new interval, insert it (merging if necessary) and return the result.

**Examples:**
```
intervals=[[1,3],[6,9]], new=[2,5] → [[1,5],[6,9]]
intervals=[[1,2],[3,5],[6,7],[8,10],[12,16]], new=[4,8] → [[1,2],[3,10],[12,16]]
```

**Acceptance criteria:** O(n) time; handle empty `intervals`.

---

### P5 — Validate Binary Search Tree (DFS) [25 min]

**Prompt:** Given a binary tree, determine if it is a valid BST. A BST node satisfies: all left descendants < node < all right descendants.

**Examples:**
```
    2
   / \
  1   3   → True

    5
   / \
  1   4
     / \
    3   6   → False (4 < 5)
```

**Acceptance criteria:** O(n) time, O(h) space where h = height. Pass min/max bounds through recursion.

---

### P6 — Word Break (DP) [25 min]

**Prompt:** Given string `s` and list `word_dict`, return `True` if `s` can be segmented into space-separated words all in `word_dict`.

**Constraints:** words can be reused.

**Examples:**
```
s="leetcode", word_dict=["leet","code"]         → True
s="applepenapple", word_dict=["apple","pen"]    → True
s="catsandog", word_dict=["cats","dog","sand","cat","an"] → False
```

**Acceptance criteria:** O(n²) or better; use `set` for O(1) lookups.

---

### P7 — Kth Largest Element in Array (Heap) [25 min]

**Prompt:** Find the kth largest element in an unsorted array. Not the kth distinct element.

**Constraints:** O(n log k) required; O(n) (quickselect) is bonus.

**Examples:**
```
[3,2,1,5,6,4], k=2 → 5
[3,2,3,1,2,4,5,5,6], k=4 → 4
```

---

### P8 — Number of Islands II (Union-Find) [25 min]

**Prompt:** You have an m×n grid of water. Given a list of positions where land is added one at a time, return the number of islands after each addition.

**Constraints:** positions may contain duplicates (adding land to existing land cell).

**Examples:**
```
m=3, n=3, positions=[[0,0],[0,1],[1,2],[2,1]]
→ [1, 1, 2, 3]
```

**Acceptance criteria:** O(k α(mn)) where k = number of positions; use Union-Find with grid index mapping `r * n + c`.

---

### P9 — Alien Dictionary (Topological Sort / BFS) [35 min]

**Prompt:** Given a list of words from an alien language, sorted lexicographically in that language, return the alien alphabet order. Return `""` if it's invalid (cycle).

**Examples:**
```
words=["wrt","wrf","er","ett","rftt"]
→ "wertf"

words=["z","x"]
→ "zx"

words=["z","x","z"]
→ ""   (cycle: z → x → z)
```

**Acceptance criteria:** Kahn's algorithm (BFS topological sort) or DFS cycle detection. Handle edge case: prefix word appearing after longer word (invalid).

---

### P10 — Trapping Rain Water (Two Pointers) [35 min]

**Prompt:** Given an elevation map (non-negative integers), compute how much water it can trap after raining.

**Examples:**
```
[0,1,0,2,1,0,1,3,2,1,2,1] → 6
[4,2,0,3,2,5]              → 9
```

**Acceptance criteria:** O(n) time, O(1) space (two-pointer approach). Also describe the O(n) space prefix-array approach and explain why two-pointer is superior.

---

## Self-assessment after the session

After completing (or attempting) all problems, answer these honestly:

- Which pattern felt most natural? Which caused the most hesitation?
- On which problems did you exceed the time limit? Why?
- How often did you skip the 5-step protocol under time pressure?
- Did any problem require a pattern you hadn't identified correctly at the start?

Record your answers — they drive next week's revision focus.

(Solutions: ../solutions/week-08/day-6.md)

---

## Self-check

- [ ] I attempted all 10 problems within the time budget
- [ ] I followed the 5-step interview protocol for every problem
- [ ] I stated time/space complexity for each solution
- [ ] I have identified 2–3 patterns needing further revision
- [ ] I reviewed the solution file and understood every approach I missed
