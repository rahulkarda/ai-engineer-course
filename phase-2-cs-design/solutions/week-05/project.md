# Reference Implementation — Week 05 Project

> Full working implementation for instructor reference. Learner should attempt before reading.

## Project layout

```
week-05-project/
├── dp_library/
│   ├── __init__.py
│   ├── fibonacci.py
│   ├── knapsack.py
│   ├── lcs.py
│   ├── edit_distance.py
│   └── lis.py
├── tests/
│   ├── __init__.py
│   ├── test_fibonacci.py
│   ├── test_knapsack.py
│   ├── test_lcs.py
│   ├── test_edit_distance.py
│   └── test_lis.py
├── complexity.py
├── pyproject.toml
└── README.md
```

---

## `dp_library/fibonacci.py`

```python
"""fibonacci.py — memoized and space-optimized tabulated Fibonacci."""
from __future__ import annotations
import logging
from functools import cache

logger = logging.getLogger(__name__)


@cache
def fib_memo(n: int) -> int:
    """O(n) time, O(n) space (cache). Raises ValueError for n < 0."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)


def fib_tab(n: int) -> int:
    """O(n) time, O(1) space — rolling two variables."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

---

## `dp_library/knapsack.py`

```python
"""knapsack.py — 0/1 knapsack: memoized, tabulated, space-optimized, item reconstruction."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def knapsack_memo(
    weights: tuple[int, ...],
    values: tuple[int, ...],
    capacity: int,
) -> int:
    memo: dict[tuple[int, int], int] = {}

    def dp(i: int, cap: int) -> int:
        if i == len(weights) or cap == 0:
            return 0
        if (i, cap) in memo:
            return memo[(i, cap)]
        skip = dp(i + 1, cap)
        take = 0
        if weights[i] <= cap:
            take = values[i] + dp(i + 1, cap - weights[i])
        memo[(i, cap)] = max(skip, take)
        return memo[(i, cap)]

    return dp(0, capacity)


def knapsack_tab(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> int:
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if weights[i - 1] <= c:
                dp[i][c] = max(dp[i][c], values[i - 1] + dp[i - 1][c - weights[i - 1]])
    return dp[n][capacity]


def knapsack_space_opt(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> int:
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for c in range(capacity, weights[i] - 1, -1):
            dp[c] = max(dp[c], values[i] + dp[c - weights[i]])
    return dp[capacity]


def items_taken(
    weights: list[int],
    values: list[int],
    capacity: int,
) -> list[int]:
    """Return indices of items chosen in the optimal knapsack solution."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if weights[i - 1] <= c:
                dp[i][c] = max(dp[i][c], values[i - 1] + dp[i - 1][c - weights[i - 1]])
    # Backtrack
    chosen: list[int] = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return list(reversed(chosen))
```

---

## `dp_library/lcs.py`

```python
"""lcs.py — Longest Common Subsequence."""
from __future__ import annotations
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def lcs_memo(s: str, t: str) -> int:
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> int:
        if i == len(s) or j == len(t):
            return 0
        if s[i] == t[j]:
            return 1 + dp(i + 1, j + 1)
        return max(dp(i + 1, j), dp(i, j + 1))
    return dp(0, 0)


def lcs_tab(s: str, t: str) -> int:
    m, n = len(s), len(t)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = 1 + prev[j - 1]
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def lcs_reconstruct(s: str, t: str) -> str:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    chars: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if s[i - 1] == t[j - 1]:
            chars.append(s[i - 1]); i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(chars))
```

---

## `dp_library/edit_distance.py`

```python
"""edit_distance.py — Levenshtein distance: memoized, tabulated, op list."""
from __future__ import annotations
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def edit_distance_memo(s: str, t: str) -> int:
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i
        if s[i - 1] == t[j - 1]:
            return dp(i - 1, j - 1)
        return 1 + min(dp(i - 1, j), dp(i, j - 1), dp(i - 1, j - 1))
    return dp(len(s), len(t))


def edit_distance_tab(s: str, t: str) -> int:
    m, n = len(s), len(t)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def edit_ops(s: str, t: str) -> list[str]:
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    ops: list[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s[i - 1] == t[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(f"replace s[{i-1}]='{s[i-1]}' -> '{t[j-1]}'"); i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(f"insert '{t[j-1]}' at pos {i}"); j -= 1
        else:
            ops.append(f"delete s[{i-1}]='{s[i-1]}'"); i -= 1
    return list(reversed(ops))
```

---

## `dp_library/lis.py`

```python
"""lis.py — Longest Increasing Subsequence: O(n²) DP, O(n log n) patience sort, reconstruction."""
from __future__ import annotations
import bisect
import logging

logger = logging.getLogger(__name__)


def lis_dp(nums: list[int]) -> int:
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


def lis_nlogn(nums: list[int]) -> int:
    tails: list[int] = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)


def lis_sequence(nums: list[int]) -> list[int]:
    if not nums:
        return []
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    best = max(range(n), key=lambda i: dp[i])
    seq: list[int] = []
    k = best
    while k != -1:
        seq.append(nums[k]); k = parent[k]
    return list(reversed(seq))
```

---

## `complexity.py`

```python
"""complexity.py — empirical timing comparison of memoized vs tabulated DP."""
from __future__ import annotations
import logging
import random
import time

from dp_library.fibonacci import fib_memo, fib_tab
from dp_library.knapsack import knapsack_memo, knapsack_tab
from dp_library.lcs import lcs_memo, lcs_tab
from dp_library.edit_distance import edit_distance_memo, edit_distance_tab
from dp_library.lis import lis_dp, lis_nlogn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def time_fn(fn, *args, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return min(times) * 1000  # ms


def compare(label: str, memo_fn, tab_fn, memo_args, tab_args, ns: list[int]) -> None:
    logger.info("%-20s | %-6s | %-10s | %-10s", "Algorithm", "n", "memo(ms)", "tab(ms)")
    logger.info("-" * 55)
    for n in ns:
        m_args = memo_args(n)
        t_args = tab_args(n)
        m_time = time_fn(memo_fn, *m_args)
        t_time = time_fn(tab_fn, *t_args)
        logger.info("%-20s | %-6d | %-10.4f | %-10.4f", label, n, m_time, t_time)


if __name__ == "__main__":
    # Fibonacci
    compare(
        "Fibonacci",
        fib_memo, fib_tab,
        lambda n: (n,), lambda n: (n,),
        [10, 100, 500],
    )
    # LIS
    compare(
        "LIS",
        lis_dp, lis_nlogn,
        lambda n: ([random.randint(0, n) for _ in range(n)],),
        lambda n: ([random.randint(0, n) for _ in range(n)],),
        [50, 200, 1000],
    )
```

---

## Sample test — `tests/test_fibonacci.py`

```python
import pytest
from dp_library.fibonacci import fib_memo, fib_tab

@pytest.mark.parametrize("n,expected", [
    (0, 0), (1, 1), (2, 1), (5, 5), (10, 55), (20, 6765),
])
def test_fib_memo(n: int, expected: int) -> None:
    assert fib_memo(n) == expected

@pytest.mark.parametrize("n,expected", [
    (0, 0), (1, 1), (2, 1), (5, 5), (10, 55), (20, 6765),
])
def test_fib_tab(n: int, expected: int) -> None:
    assert fib_tab(n) == expected

def test_fib_memo_tab_agree() -> None:
    for n in range(50):
        assert fib_memo(n) == fib_tab(n)

def test_fib_negative_raises() -> None:
    with pytest.raises(ValueError):
        fib_memo(-1)
    with pytest.raises(ValueError):
        fib_tab(-1)
```
