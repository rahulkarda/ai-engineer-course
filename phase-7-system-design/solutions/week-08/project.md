# Solutions: Week 08 Project — Reference Implementation

## Module implementations

All files below form the reference for `week_08_project/solutions/`. Each implementation is complete and runnable.

---

### `solutions/two_pointers.py`

```python
"""Two-pointer pattern solutions."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def container_with_most_water(height: list[int]) -> int:
    """Maximum water container formed by two lines.

    Time: O(n)  Space: O(1)
    """
    if len(height) < 2:
        return 0
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        area = (right - left) * min(height[left], height[right])
        logger.debug("left=%d right=%d area=%d", left, right, area)
        best = max(best, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best


def three_sum(nums: list[int]) -> list[list[int]]:
    """All unique triplets summing to 0.

    Time: O(n²)  Space: O(1) extra (output excluded)
    """
    nums.sort()
    result: list[list[int]] = []
    for i, a in enumerate(nums):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = a + nums[left] + nums[right]
            if total == 0:
                result.append([a, nums[left], nums[right]])
                left += 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
```

---

### `solutions/binary_search.py`

```python
"""Binary search pattern solutions."""
from __future__ import annotations
import math
import logging

logger = logging.getLogger(__name__)


def search_rotated(nums: list[int], target: int) -> int:
    """Binary search in rotated sorted array.

    Time: O(log n)  Space: O(1)
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def min_eating_speed(piles: list[int], h: int) -> int:
    """Minimum bananas/hour to finish in h hours (search on answer).

    Time: O(n log max(piles))  Space: O(1)
    """
    def feasible(speed: int) -> bool:
        return sum(math.ceil(p / speed) for p in piles) <= h

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        logger.debug("lo=%d hi=%d mid=%d feasible=%s", lo, hi, mid, feasible(mid))
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

---

### `solutions/intervals.py`

```python
"""Interval pattern solutions."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """Merge all overlapping intervals.

    Time: O(n log n)  Space: O(n)
    """
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0][:]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
            logger.debug("Merged to %s", merged[-1])
        else:
            merged.append([start, end])
    return merged


def non_overlapping_intervals(intervals: list[list[int]]) -> int:
    """Minimum intervals to remove for non-overlapping result (interval scheduling).

    Time: O(n log n)  Space: O(1)
    Greedy: keep the interval with earliest end time.
    """
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    end = intervals[0][1]
    keep = 1
    for start, e in intervals[1:]:
        if start >= end:
            keep += 1
            end = e
    return len(intervals) - keep
```

---

### `solutions/trees_graphs.py`

```python
"""Tree and graph pattern solutions."""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    val: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def max_depth(root: Optional[TreeNode]) -> int:
    """Maximum depth of binary tree via recursive DFS.

    Time: O(n)  Space: O(h)
    """
    if root is None:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))


def course_schedule(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """Detect cycle in prerequisites graph using Kahn's algorithm.

    Time: O(V + E)  Space: O(V + E)
    """
    in_degree = [0] * num_courses
    graph: dict[int, list[int]] = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    queue: deque[int] = deque(i for i in range(num_courses) if in_degree[i] == 0)
    completed = 0
    while queue:
        node = queue.popleft()
        completed += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    can_finish = completed == num_courses
    logger.debug("completed=%d / %d, can_finish=%s", completed, num_courses, can_finish)
    return can_finish
```

---

### `solutions/dynamic_programming.py`

```python
"""Dynamic programming pattern solutions."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def coin_change(coins: list[int], amount: int) -> int:
    """Fewest coins to make amount (unbounded knapsack).

    Time: O(amount × len(coins))  Space: O(amount)
    """
    INF = float("inf")
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)
    result = -1 if dp[amount] == INF else int(dp[amount])
    logger.debug("coin_change(%s, %d) = %d", coins, amount, result)
    return result


def longest_common_subsequence(text1: str, text2: str) -> int:
    """LCS length (2-D DP).

    Time: O(m × n)  Space: O(m × n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

---

### `solutions/heaps_tries_uf.py`

```python
"""Heap, Trie, and Union-Find pattern solutions."""
from __future__ import annotations
import heapq
import logging
from collections import Counter
from typing import Callable

logger = logging.getLogger(__name__)


def find_median_stream() -> tuple[Callable[[int], None], Callable[[], float]]:
    """Factory returning (add_num, find_median) using two heaps.

    add_num: Time O(log n)  find_median: Time O(1)  Space: O(n)
    """
    max_heap: list[int] = []   # lower half, negated
    min_heap: list[int] = []   # upper half

    def add_num(num: int) -> None:
        heapq.heappush(max_heap, -num)
        if max_heap and min_heap and (-max_heap[0]) > min_heap[0]:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
        logger.debug("Added %d; median=%.1f", num, find_median())

    def find_median() -> float:
        if len(max_heap) > len(min_heap):
            return float(-max_heap[0])
        return (-max_heap[0] + min_heap[0]) / 2.0

    return add_num, find_median


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def count_components(n: int, edges: list[list[int]]) -> int:
    """Connected components count.

    Time: O(E α(n))  Space: O(n)
    """
    uf = _UnionFind(n)
    components = n
    for u, v in edges:
        if uf.union(u, v):
            components -= 1
    logger.debug("count_components: %d components", components)
    return components
```

---

### `solutions/greedy.py`

```python
"""Greedy pattern solutions."""
from __future__ import annotations
import heapq
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def jump_game_ii(nums: list[int]) -> int:
    """Minimum jumps to reach last index.

    Time: O(n)  Space: O(1)
    """
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
            logger.debug("Jump %d: reach up to index %d", jumps, current_end)
    return jumps


def task_scheduler(tasks: list[str], n: int) -> int:
    """Minimum intervals to execute all tasks with cooldown n.

    Time: O(m log 26) ≈ O(m)  Space: O(26) = O(1)
    """
    freq = Counter(tasks)
    max_freq = max(freq.values())
    max_count = sum(1 for v in freq.values() if v == max_freq)
    # Formula: max(task count, idle slots + tasks)
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
```

---

## `complexity_analysis.md` — reference table

| Function | Time | Space | Pattern | Key insight |
|---|---|---|---|---|
| container_with_most_water | O(n) | O(1) | Two pointers | move shorter line; exchange argument |
| three_sum | O(n²) | O(1) | Two pointers + sort | fix outer; two-pointer inner; skip dups |
| search_rotated | O(log n) | O(1) | Binary search | identify sorted half each step |
| min_eating_speed | O(n log M) | O(1) | Binary search on answer | monotone feasibility function |
| merge_intervals | O(n log n) | O(n) | Intervals | sort by start; merge last |
| non_overlapping_intervals | O(n log n) | O(1) | Greedy intervals | keep earliest-end; exchange argument |
| max_depth | O(n) | O(h) | DFS recursive | 1 + max(left, right) |
| course_schedule | O(V+E) | O(V+E) | BFS topo sort | cycle → completed < n |
| coin_change | O(amount×C) | O(amount) | 1-D DP unbounded | left-to-right: allows reuse |
| longest_common_subsequence | O(m×n) | O(m×n) | 2-D DP | match → diagonal; else max(left, up) |
| find_median_stream | O(log n) add | O(n) | Two heaps | max_heap lower, min_heap upper |
| count_components | O(E α(n)) | O(n) | Union-Find | union returns False = already connected |
| jump_game_ii | O(n) | O(1) | Greedy | track farthest per jump range |
| task_scheduler | O(m) | O(1) | Greedy formula | (max_freq-1)×(n+1)+count_max_freq |

---

## Tests reference (pytest run command)

```bash
cd week_08_project
python -m pytest tests/ -v --tb=short
```

All tests should pass with zero failures against the reference implementations above.
