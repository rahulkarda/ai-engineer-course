# Solutions — Week 03 · Day 3

## Exercise 1 — Iterative insert

```python
def insert_iter(root: Optional[TreeNode], val: int) -> TreeNode:
    """Iterative BST insert."""
    new_node = TreeNode(val)
    if root is None:
        return new_node
    current = root
    while True:
        if val < current.val:
            if current.left is None:
                current.left = new_node
                break
            current = current.left
        elif val > current.val:
            if current.right is None:
                current.right = new_node
                break
            current = current.right
        else:
            break  # duplicate
    return root
```

**Comparison**: recursive is cleaner and matches the mental model. Iterative is stack-safe for degenerate trees (depth > ~1000). For production with potentially large trees, prefer iterative.

## Exercise 2 — Floor and ceiling

```python
def floor(root: Optional[TreeNode], val: int) -> Optional[int]:
    """Largest value ≤ val."""
    result: Optional[int] = None
    node = root
    while node:
        if node.val == val:
            return val
        elif node.val < val:
            result = node.val   # candidate; try going right for something larger
            node = node.right
        else:
            node = node.left    # current too large; go left
    return result


def ceil(root: Optional[TreeNode], val: int) -> Optional[int]:
    """Smallest value ≥ val."""
    result: Optional[int] = None
    node = root
    while node:
        if node.val == val:
            return val
        elif node.val > val:
            result = node.val   # candidate; try going left for something smaller
            node = node.left
        else:
            node = node.right   # current too small; go right
    return result
```

Both run in O(h).

## Exercise 3 — Rank

```python
def rank(root: Optional[TreeNode], val: int) -> int:
    """Number of values strictly less than val in the BST."""
    count = 0
    node = root
    while node:
        if val <= node.val:
            node = node.left        # val is ≤ current; nothing in right subtree counts
        else:
            count += 1              # current node is < val
            node = node.right       # look for more in right subtree
    return count
```

Note: this iterative version counts only when we go right (meaning current value < val). Does not count nodes in left subtrees that we bypassed — those are guaranteed < current which is < val, but we only add 1 per node visited on the right path. This gives the correct count because left-subtree nodes would all be < current.val < val; however we don't count them here.

**Correct O(h) version using subtree size (augmented BST)**: without augmentation we need O(n) inorder. The above O(h) approach works only if rank is defined as "values encountered on the descent path that are < val". For exact rank on a standard BST, use inorder count O(n) or augment nodes with left-subtree sizes.

```python
def rank_exact(root: Optional[TreeNode], val: int) -> int:
    """O(n) exact rank — count of values < val via inorder."""
    count = 0
    def _inorder(node: Optional[TreeNode]) -> None:
        nonlocal count
        if node is None:
            return
        _inorder(node.left)
        if node.val < val:
            count += 1
        _inorder(node.right)
    _inorder(root)
    return count
```

## Exercise 4 — Sorted array to balanced BST

```python
def sorted_array_to_bst(nums: list[int]) -> Optional[TreeNode]:
    """Pick midpoint as root recursively → balanced BST."""
    if not nums:
        return None
    mid = len(nums) // 2
    node = TreeNode(nums[mid])
    node.left = sorted_array_to_bst(nums[:mid])
    node.right = sorted_array_to_bst(nums[mid + 1:])
    return node
```

For `[1,2,3,4,5,6,7]`, root=4, left subtree from `[1,2,3]` (root=2), right from `[5,6,7]` (root=6).
Inorder of result: `[1,2,3,4,5,6,7]`. Height = 2 (optimal for 7 nodes).

## Exercise 5 — Delete stress test (pytest)

```python
import random
import pytest
from src.bst import BST   # or inline the BST class


def test_delete_stress() -> None:
    rng = random.Random(42)
    values = list(range(1, 101))

    insert_order = values[:]
    rng.shuffle(insert_order)

    bst = BST()
    for v in insert_order:
        bst.insert(v)

    delete_order = values[:]
    rng.shuffle(delete_order)

    remaining = set(values)
    for v in delete_order:
        bst.delete(v)
        remaining.remove(v)

        result = bst.inorder()
        expected = sorted(remaining)

        assert result == expected, f"After deleting {v}: got {result}, expected {expected}"
        # also verify no duplicates
        assert len(result) == len(set(result)), "Duplicates found after deletion"
```
