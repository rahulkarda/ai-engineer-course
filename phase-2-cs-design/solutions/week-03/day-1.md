# Solutions — Week 03 · Day 1

## Exercise 1 — Count nodes

```python
def count_nodes(node: Optional[TreeNode]) -> int:
    if node is None:
        return 0
    return 1 + count_nodes(node.left) + count_nodes(node.right)
```

Base case: empty tree has 0 nodes. Recursive case: 1 (current) + count of both subtrees.

## Exercise 2 — Leaf count

```python
def count_leaves(node: Optional[TreeNode]) -> int:
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1  # this node is a leaf
    return count_leaves(node.left) + count_leaves(node.right)
```

A node is a leaf when both children are `None`. We return 1 for it and do not recurse further.

## Exercise 3 — Mirror

```python
def mirror(node: Optional[TreeNode]) -> Optional[TreeNode]:
    """Return a new mirrored tree; do not mutate the original."""
    if node is None:
        return None
    return TreeNode(
        val=node.val,
        left=mirror(node.right),   # swap: original right becomes new left
        right=mirror(node.left),   # swap: original left becomes new right
    )
```

We create new `TreeNode` objects — the original tree is untouched. Swapping left/right at every level produces the mirror.

## Exercise 4 — Path sum

```python
def has_path_sum(node: Optional[TreeNode], target: int) -> bool:
    if node is None:
        return False
    remaining = target - node.val
    # reached a leaf
    if node.left is None and node.right is None:
        return remaining == 0
    return has_path_sum(node.left, remaining) or \
           has_path_sum(node.right, remaining)
```

At each step subtract the current value. At a leaf, check whether the remainder is zero.

## Exercise 5 — Serialize

```python
def serialize(node: Optional[TreeNode]) -> str:
    """Preorder serialization with 'null' for missing nodes."""
    parts: list[str] = []

    def _helper(n: Optional[TreeNode]) -> None:
        if n is None:
            parts.append("null")
            return
        parts.append(str(n.val))
        _helper(n.left)
        _helper(n.right)

    _helper(node)
    return ",".join(parts)
```

**Why preorder?** The root is encoded first. During deserialization, reading the root first lets us immediately reconstruct the node before recursing into children — the same order as construction. Inorder would require knowing subtree boundaries first, making deserialization ambiguous without extra information.

```python
# Corresponding deserialization
def deserialize(data: str) -> Optional[TreeNode]:
    it = iter(data.split(","))

    def _helper() -> Optional[TreeNode]:
        token = next(it)
        if token == "null":
            return None
        node = TreeNode(int(token))
        node.left = _helper()
        node.right = _helper()
        return node

    return _helper()
```
