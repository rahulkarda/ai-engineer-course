# Week 08 Assessment — DSA Patterns & Interview Technique

> Time: ~1 hr  ·  Pass gate: ≥ 75%

## Conceptual (interview-style)

Answer these as you would in a verbal interview — concise, precise, one to three sentences each.

1. You are given an unsorted array and asked for the subarray of length k with the maximum sum. What pattern applies? Why can't two pointers solve this directly?

2. Explain why binary search requires O(log n) iterations. What property of the input makes this possible, and what breaks when that property is absent?

3. A BFS solution guarantees the shortest path in an unweighted graph. Does DFS offer the same guarantee? Justify your answer.

4. What is the difference between 0/1 knapsack and unbounded knapsack? How does the inner-loop iteration direction encode this difference in the space-optimised version?

5. Union-Find with path compression has amortised O(α(n)) per operation. What does α represent, and what is its practical value for n ≤ 10⁹?

6. You need to find all subsets of an array of n distinct integers. What is the time and space complexity of backtracking? How would you add pruning if the problem required subsets summing to at most T?

7. When should you use a Trie over a hash set for prefix queries? Give one example where Trie is clearly superior.

8. Describe the exchange argument for the greedy solution to the interval scheduling maximisation problem (select max non-overlapping intervals by finishing time).

---

## Coding Challenge

### Specification: `IntervalPatternSolver`

Implement a module `interval_pattern_solver.py` with the following three functions. Each must be fully typed, use `logging` (not `print`), and include a module-level docstring.

**Function 1: `meeting_rooms_required(intervals: list[tuple[int, int]]) -> int`**
- Given a list of meeting `(start, end)` intervals, return the minimum number of rooms required so all meetings can be held simultaneously.
- Time: O(n log n), Space: O(n).
- Edge case: empty list → 0.

**Function 2: `longest_increasing_subsequence(nums: list[int]) -> int`**
- Return the length of the longest strictly increasing subsequence.
- O(n log n) required (binary search + patience sorting). O(n²) earns partial credit.
- Edge case: empty list → 0, single element → 1.

**Function 3: `word_search(board: list[list[str]], word: str) -> bool`**
- Given a 2-D character board, return `True` if `word` exists as a connected path of adjacent cells (up/down/left/right). Each cell may be used at most once per path.
- Use DFS + backtracking.
- Time: O(m × n × 4^L) where L = word length.

**Acceptance criteria:**
- All three functions pass the provided test cases.
- Type hints on all function signatures and return values.
- Complexity stated in each function's docstring.
- `logging.debug` calls showing key decision points.
- Tests in `tests/test_interval_pattern_solver.py` covering: normal case, edge case (empty input), and at least one edge case specific to each function.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | Correct logic, no copy-paste; handles edge cases |
| Conceptual understanding | 25% | Conceptual answers precise and complete |
| Debugging & problem-solving | 20% | Wrong first attempt diagnosed and fixed; complexity stated accurately |
| Code quality & testing | 15% | Type hints, logging, docstrings, pytest tests with edge cases |

Answer key: ../solutions/week-08/assessment.md
