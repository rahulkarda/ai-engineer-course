# Solutions — Phase 2 · Week 02 · Day 2

> Stacks, Queues, and `collections.deque`

---

## Exercise 1 — Valid parentheses

**Key insight:** Push opening brackets onto the stack. When a closing bracket is encountered, check that the top of the stack is its matching opener.

```python
"""
solution_day2_ex1.py — valid parentheses using a stack
Python 3.12+
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MATCHING: dict[str, str] = {")": "(", "]": "[", "}": "{"}
OPENERS: frozenset[str] = frozenset(MATCHING.values())


def is_valid_parens(s: str) -> bool:
    """
    Return True if brackets are correctly nested and closed.
    O(n) time, O(n) space (stack).
    """
    stack: list[str] = []
    for ch in s:
        if ch in OPENERS:
            stack.append(ch)
        elif ch in MATCHING:
            if not stack or stack[-1] != MATCHING[ch]:
                return False
            stack.pop()
        # Ignore non-bracket characters
    return len(stack) == 0


if __name__ == "__main__":
    cases = [
        ("([{}])", True),
        ("([)]", False),
        ("{[]}", True),
        ("", True),
        ("(", False),
        (")(", False),
        ("hello(world)", True),
    ]
    for s, expected in cases:
        result = is_valid_parens(s)
        status = "PASS" if result == expected else "FAIL"
        log.info("[%s] %r → %s (expected %s)", status, s, result, expected)
```

---

## Exercise 2 — Min-stack in O(1)

**Key insight:** Maintain a second stack (`min_stack`) that mirrors the main stack. Each entry in `min_stack` records the current minimum at that point. Push the `min(current_value, min_stack.peek())` when the main stack grows; pop both stacks together.

```python
"""
solution_day2_ex2.py — MinStack with O(1) get_min
Python 3.12+
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


class MinStack:
    """
    Stack with O(1) push, pop, peek, and get_min.

    Maintains a parallel min_stack. Each position i in min_stack holds
    the minimum of all values in the main stack from index 0 through i.
    Therefore min_stack[-1] is always the overall minimum.
    """

    def __init__(self) -> None:
        self._stack: list[int] = []
        self._min_stack: list[int] = []

    def push(self, value: int) -> None:
        self._stack.append(value)
        current_min = value if not self._min_stack else min(value, self._min_stack[-1])
        self._min_stack.append(current_min)

    def pop(self) -> int:
        if not self._stack:
            raise IndexError("pop from empty MinStack")
        self._min_stack.pop()
        return self._stack.pop()

    def peek(self) -> int:
        if not self._stack:
            raise IndexError("peek at empty MinStack")
        return self._stack[-1]

    def get_min(self) -> int:
        if not self._min_stack:
            raise IndexError("get_min on empty MinStack")
        return self._min_stack[-1]

    def __len__(self) -> int:
        return len(self._stack)


if __name__ == "__main__":
    ms = MinStack()
    for v in [5, 3, 7, 2, 4]:
        ms.push(v)
        log.info("push(%d) → min=%d", v, ms.get_min())
    log.info("pop()=%d → min=%d", ms.pop(), ms.get_min())   # pop 4, min=2
    log.info("pop()=%d → min=%d", ms.pop(), ms.get_min())   # pop 2, min=3
```

**Why a second stack and not a single variable?** Because after popping the current minimum, we need to know the previous minimum — a single variable loses that history. The parallel stack restores it in O(1).

---

## Exercise 3 — Evaluate Reverse Polish Notation

**Key insight:** Operands push onto the stack. Operators pop two operands, compute, and push the result. Python's `int()` on division handles the "truncate toward zero" requirement correctly using `int(a / b)` (not `a // b`, which floors toward negative infinity).

```python
"""
solution_day2_ex3.py — evaluate RPN expression
Python 3.12+
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/"})


def eval_rpn(tokens: list[str]) -> int:
    """
    Evaluate a Reverse Polish Notation expression.
    O(n) time, O(n) space.

    Division truncates toward zero: 7/2=3, -7/2=-3.
    """
    stack: list[int] = []
    for token in tokens:
        if token in OPERATORS:
            if len(stack) < 2:
                raise ValueError(f"Not enough operands for operator {token!r}")
            b = stack.pop()   # second operand (right)
            a = stack.pop()   # first operand (left)
            match token:
                case "+": stack.append(a + b)
                case "-": stack.append(a - b)
                case "*": stack.append(a * b)
                case "/": stack.append(int(a / b))  # truncate toward zero
        else:
            stack.append(int(token))
    if len(stack) != 1:
        raise ValueError(f"Invalid RPN expression: leftover stack {stack}")
    return stack[0]


if __name__ == "__main__":
    cases = [
        (["2", "1", "+", "3", "*"], 9),           # (2+1)*3
        (["4", "13", "5", "/", "+"], 6),           # 4+(13/5)=6
        (["10", "6", "9", "3", "+", "-11", "*",
          "/", "*", "17", "+", "5", "+"], 22),
        (["-7", "2", "/"], -3),                    # truncate toward zero
    ]
    for tokens, expected in cases:
        result = eval_rpn(tokens)
        status = "PASS" if result == expected else "FAIL"
        log.info("[%s] %s → %d (expected %d)", status, tokens, result, expected)
```

---

## Exercise 4 — BFS shortest path

**Key insight:** BFS explores nodes level by level, so the first time it reaches the destination is via the shortest path. Use a queue to process nodes in FIFO order. Track visited nodes and parent pointers to reconstruct the path.

```python
"""
solution_day2_ex4.py — BFS shortest path using a queue
Python 3.12+
"""
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

Graph = dict[str, list[str]]


def bfs_shortest_path(graph: Graph, start: str, end: str) -> list[str] | None:
    """
    Return the shortest path from start to end as a list of nodes.
    Returns None if no path exists.
    O(V + E) time, O(V) space.
    """
    if start not in graph or end not in graph:
        return None
    if start == end:
        return [start]

    visited: set[str] = {start}
    parent: dict[str, str | None] = {start: None}
    queue: deque[str] = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                if neighbor == end:
                    return _reconstruct_path(parent, start, end)
                queue.append(neighbor)

    return None  # no path found


def _reconstruct_path(
    parent: dict[str, str | None],
    start: str,
    end: str,
) -> list[str]:
    """Walk parent pointers from end back to start, then reverse."""
    path: list[str] = []
    current: str | None = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


if __name__ == "__main__":
    graph: Graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"],
    }
    log.info("A → F: %s", bfs_shortest_path(graph, "A", "F"))   # A->C->F (len 3)
    log.info("A → E: %s", bfs_shortest_path(graph, "A", "E"))   # A->B->E (len 3)
    log.info("D → F: %s", bfs_shortest_path(graph, "D", "F"))   # D->B->E->F (len 4)
    log.info("A → A: %s", bfs_shortest_path(graph, "A", "A"))   # [A]

    disconnected: Graph = {"X": [], "Y": []}
    log.info("X → Y (disconnected): %s", bfs_shortest_path(disconnected, "X", "Y"))  # None
```

---

## Exercise 5 — Sliding window minimum

**Key insight:** The only change from the maximum version is flipping the comparison from `<` to `>` in the monotonic deque maintenance step. The deque now maintains an increasing (instead of decreasing) order, so its front always holds the minimum index.

```python
"""
solution_day2_ex5.py — sliding window minimum
Python 3.12+
"""
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def sliding_window_min(nums: list[int], k: int) -> list[int]:
    """
    Return the minimum in each window of size k.
    O(n) time, O(k) space.

    Difference from sliding_window_max:
    - Line 'while dq and nums[dq[-1]] < val' changes to '> val'
    - Deque now maintains increasing order (front = index of minimum).
    """
    if not nums or k == 0:
        return []

    dq: deque[int] = deque()
    result: list[int] = []

    for i, val in enumerate(nums):
        # Remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # Maintain increasing order: remove larger elements from back
        while dq and nums[dq[-1]] > val:   # NOTE: '>' instead of '<'
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


if __name__ == "__main__":
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    log.info("Input (k=%d): %s", k, nums)
    log.info("Window min:  %s", sliding_window_min(nums, k))
    # Expected: [-1, -3, -3, -3, 3, 3]

    # Verify against naive O(n*k) solution
    naive = [min(nums[i:i+k]) for i in range(len(nums) - k + 1)]
    log.info("Naive check: %s", naive)
    assert sliding_window_min(nums, k) == naive, "Mismatch!"
    log.info("Assertion passed.")
```

**The single change:** `nums[dq[-1]] > val` (instead of `< val`) ensures the deque head always points to the minimum, not the maximum.
