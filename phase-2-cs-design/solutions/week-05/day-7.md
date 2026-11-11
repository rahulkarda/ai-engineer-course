# Solutions — Week 05 Day 7 (Sunday Review)

Day 7 is a project completion + assessment day. There are no new exercises.

Refer to:
- `solutions/week-05/assessment.md` for assessment answer key
- `solutions/week-05/project.md` for reference project implementation

## Complexity reference — worked answers

**Fibonacci tabulated O(1) space**: keep only `prev` and `curr`. At each step `curr = prev_prev + prev`, then shift.

**Knapsack space optimization**: right-to-left traversal ensures `dp[c - weight]` is the value *before* including item `i`, which is the "0/1" property. Left-to-right would allow an item to be counted multiple times.

**LCS rolling rows**: `dp[i][j]` depends only on `dp[i-1][j]`, `dp[i][j-1]`, `dp[i-1][j-1]` — all in the previous row or current row, never beyond. Two arrays suffice.

**Edit distance swap**: same as LCS rolling rows. `prev[j]` = `dp[i-1][j]`, `curr[j-1]` = `dp[i][j-1]`. Space O(min(m,n)) by always iterating over the shorter string.
