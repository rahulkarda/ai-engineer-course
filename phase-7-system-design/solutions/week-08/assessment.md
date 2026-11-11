# Solutions: Week 08 Assessment — Answer Key

## Conceptual answers

**1. Fixed-size sliding window vs. two pointers**

Sliding window applies. Two pointers require sorted input or the ability to move both pointers based on a sum comparison. For max sum of a fixed-size window, neither pointer movement criterion applies — instead, advance the window as a unit, adding the new right element and subtracting the outgoing left element. Two pointers are for variable-target searches on sorted data, not fixed-length range aggregation.

**2. Why O(log n) iterations**

Each iteration halves the search space: `right - left` decreases by ≥ 50% per step. After k iterations, the range has ≤ n / 2^k elements; we stop when this is ≤ 1, giving k = log₂(n). This requires the input to be **monotone** (sorted or monotonically satisfying a predicate) — binary search is inapplicable to unsorted data because halving doesn't guarantee the target is in the remaining half.

**3. BFS shortest path vs. DFS**

DFS does not guarantee shortest path. BFS processes nodes level by level, so the first time a node is dequeued, it is via the shortest path (BFS explores all distance-k nodes before any distance-k+1 node). DFS follows one path to its end before backtracking — it may reach a node via a longer path first.

**4. 0/1 vs. unbounded knapsack**

0/1: each item usable at most once. Unbounded: items reusable. In the space-optimised 1-D DP:
- 0/1: iterate capacity **right-to-left** — each item computed from a "previous item" state, preventing reuse of the same item in one pass.
- Unbounded: iterate **left-to-right** — when computing `dp[j]`, `dp[j - weight]` already reflects the current item, allowing reuse.

**5. α(n) — inverse Ackermann**

α is the inverse Ackermann function. For all practical n (including n = 10^80), α(n) ≤ 5. It represents how many times you can apply `log*` (iterated log) to n before reaching 1. In practice, Union-Find with path compression and union by rank runs in near-constant time per operation.

**6. Subset complexity**

Time: O(n × 2^n) — 2^n subsets, each taking up to O(n) to copy. Space: O(n) recursion stack (not counting output).

Pruning for sum ≤ T: after sorting, track the running sum; if `current_sum + remaining_min_elements` can never reach T (or always exceeds T), prune that branch. Specifically: if `current_sum > T`, return immediately.

**7. Trie vs. hash set for prefix queries**

Hash set gives O(L) exact-match lookup but cannot answer prefix queries without iterating all keys. Trie gives O(L) prefix search and can enumerate all words sharing a prefix in O(L + output) time. Clearly superior when: autocomplete (return all words with prefix), word break (check if any dictionary word starts from position i), or longest common prefix across many strings.

**8. Interval scheduling exchange argument**

Claim: always selecting the interval with the earliest finish time maximises the count.

Proof sketch (exchange): suppose an optimal solution skips interval A (earliest finish) in favour of B (later finish) for the same slot. Replace B with A. Because A finishes no later than B, every interval that was compatible with B is still compatible with A (it starts ≥ B's finish ≥ A's finish). The number of selected intervals doesn't decrease. Therefore the greedy choice is at least as good as any other — optimality follows by induction.

---

## Coding challenge — reference implementation

```python
# interval_pattern_solver.py
"""Solutions for the Week 08 assessment coding challenge."""
from __future__ import annotations
import heapq
import bisect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def meeting_rooms_required(intervals: list[tuple[int, int]]) -> int:
    """Minimum conference rooms for non-overlapping meetings.

    Time: O(n log n)  Space: O(n)
    """
    if not intervals:
        logger.debug("Empty intervals: 0 rooms needed")
        return 0
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    heap: list[int] = []   # min-heap of end times
    for start, end in sorted_intervals:
        logger.debug("Processing interval (%d, %d); heap size=%d", start, end, len(heap))
        if heap and heap[0] <= start:
            heapq.heapreplace(heap, end)
        else:
            heapq.heappush(heap, end)
    return len(heap)


def longest_increasing_subsequence(nums: list[int]) -> int:
    """LIS length using patience sorting.

    Time: O(n log n)  Space: O(n)
    """
    if not nums:
        return 0
    tails: list[int] = []   # tails[i] = smallest tail of all LIS of length i+1
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        logger.debug("num=%d → bisect pos=%d in tails=%s", num, pos, tails)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


def word_search(board: list[list[str]], word: str) -> bool:
    """Find word in board using DFS + backtracking.

    Time: O(m × n × 4^L)  Space: O(L) recursion stack, L = len(word)
    """
    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int, idx: int) -> bool:
        if idx == len(word):
            logger.debug("Word found")
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[idx]:
            return False
        tmp, board[r][c] = board[r][c], "#"
        found = (dfs(r + 1, c, idx + 1) or dfs(r - 1, c, idx + 1) or
                 dfs(r, c + 1, idx + 1) or dfs(r, c - 1, idx + 1))
        board[r][c] = tmp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False
```

```python
# tests/test_interval_pattern_solver.py
import pytest
from interval_pattern_solver import (
    meeting_rooms_required,
    longest_increasing_subsequence,
    word_search,
)


@pytest.mark.parametrize("intervals, expected", [
    ([(0, 30), (5, 10), (15, 20)], 2),
    ([(7, 10), (2, 4)], 1),
    ([], 0),
    ([(1, 5)], 1),
])
def test_meeting_rooms(intervals: list[tuple[int, int]], expected: int) -> None:
    assert meeting_rooms_required(intervals) == expected


@pytest.mark.parametrize("nums, expected", [
    ([10, 9, 2, 5, 3, 7, 101, 18], 4),
    ([0, 1, 0, 3, 2, 3], 4),
    ([7, 7, 7, 7], 1),
    ([], 0),
    ([5], 1),
])
def test_lis(nums: list[int], expected: int) -> None:
    assert longest_increasing_subsequence(nums) == expected


def test_word_search_found() -> None:
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert word_search(board, "ABCCED") is True


def test_word_search_not_found() -> None:
    board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
    assert word_search(board, "ABCB") is False


def test_word_search_single_cell() -> None:
    assert word_search([["a"]], "a") is True
    assert word_search([["a"]], "b") is False
```

---

## Rubric — grading guide

**Independent implementation (40%)**
- Full marks: all three functions correct including edge cases; no copy-paste from template
- Partial (20%): 2/3 functions correct, or correct but missing edge cases
- Minimal (10%): 1/3 function correct

**Conceptual (25%)**
- Full marks: all 8 answers precise (see reference above)
- Partial: 5–7 precise answers
- Minimal: 3–4 answers

**Debugging (20%)**
- Full marks: incorrect first attempt corrected with stated diagnosis; complexity stated accurately for all three functions
- Partial: corrected but no explanation; complexity partially stated
- Minimal: submitted incorrect code

**Code quality (15%)**
- Full marks: type hints on all signatures, `logging.debug` calls, docstrings with complexity lines, parametrised tests with edge cases
- Partial: type hints or tests missing but not both
- Minimal: neither type hints nor tests
