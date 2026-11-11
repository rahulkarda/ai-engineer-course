# Solutions — Week 03 · Day 2

## Exercise 1 — Zigzag level order

```python
from collections import deque
from typing import Optional

def zigzag_level_order(root: Optional[TreeNode]) -> list[list[int]]:
    if root is None:
        return []
    result: list[list[int]] = []
    queue: deque[TreeNode] = deque([root])
    left_to_right = True
    while queue:
        level_size = len(queue)
        level: list[int] = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level if left_to_right else level[::-1])
        left_to_right = not left_to_right
    return result
```

Reverse the level list on odd-indexed levels. `level[::-1]` is O(w) per level; total O(n).

## Exercise 2 — Right-side view

```python
def right_side_view(root: Optional[TreeNode]) -> list[int]:
    """Last node in each BFS level — what you'd see from the right."""
    if root is None:
        return []
    result: list[int] = []
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            if i == level_size - 1:
                result.append(node.val)   # rightmost node of this level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return result
```

Time: O(n). Space: O(w) where w is max width of the tree.

## Exercise 3 — Max width

```python
def max_width(root: Optional[TreeNode]) -> int:
    if root is None:
        return 0
    max_w = 0
    queue: deque[TreeNode] = deque([root])
    while queue:
        level_size = len(queue)
        max_w = max(max_w, level_size)
        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    return max_w
```

**Max width of a complete binary tree with n nodes**: a complete binary tree of height `h` has `2^h` nodes in the last level (which may not be full). The maximum width is at most `ceil(n/2)` (the last level for a perfect tree), which equals `(n+1)//2`.

## Exercise 4 — Inorder iterative trace on `[3, 1, 4]`

Tree: root=3, left=1, right=4.

```
current=3, stack=[]

Iteration 1:
  inner while: push 3 (stack=[3]), go left → current=1
  inner while: push 1 (stack=[3,1]), go left → current=None
  pop 1, record 1, current=1.right=None
  stack=[3]

Iteration 2:
  inner while: current=None, skip
  pop 3, record 3, current=3.right=4
  stack=[]

Iteration 3:
  inner while: push 4 (stack=[4]), go left → current=None
  pop 4, record 4, current=4.right=None
  stack=[]

Result: [1, 3, 4]  ✓
```

## Exercise 5 — Convert to BST array

```python
def to_array(root: Optional[TreeNode]) -> list[int]:
    """BFS array representation; -1 for absent nodes."""
    if root is None:
        return []
    result: list[int] = []
    queue: deque[Optional[TreeNode]] = deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            result.append(-1)
        else:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
    # strip trailing -1s (absent right-side nodes at the fringe)
    while result and result[-1] == -1:
        result.pop()
    return result
```

For the tree `[4, 2, 6, 1, 3, 5, 7]` (root=4):
```
result = [4, 2, 6, 1, 3, 5, 7]   (all present — no -1)
```
