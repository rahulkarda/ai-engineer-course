# Phase 2 · Week 03 · Day 6 — Tree Problem Set (4-hour deep practice)

> Objective: Consolidate all tree concepts by solving 10 problems spanning traversals, BSTs, heaps, and combined patterns.  ·  Time: ~4 hr  ·  Prereqs: [[day-1..day-5]]

## How to use this session

Work each problem in order. For each:
1. Read problem + constraints.
2. Identify which tree concepts apply.
3. Write a solution with type hints and a test.
4. Verify output matches expected.
5. Analyse time and space complexity.

Aim for correctness first, then optimise. Check solutions only after a genuine attempt.

---

## Problem 1 — Tree Height & Diameter

**Diameter** of a binary tree: the length of the longest path between any two nodes (measured in edges). The path may or may not pass through the root.

```python
# Example:
#         1
#        / \
#       2   3
#      / \
#     4   5
# Diameter = 3  (path: 4 → 2 → 1 → 3,  or  5 → 2 → 1 → 3)
```

**Task**: Implement `diameter(root) -> int`. Constraint: O(n) time — no redundant height calls.

*Hint*: Compute height in a single DFS pass; track max diameter via a closure or mutable container.

---

## Problem 2 — Lowest Common Ancestor (LCA)

Given a BST and two node values `p` and `q` (both guaranteed to exist), find their **Lowest Common Ancestor** — the deepest node that is an ancestor of both.

```python
# BST:      6
#          / \
#         2   8
#        / \ / \
#       0  4 7  9
#         / \
#        3   5
# LCA(2, 8) = 6
# LCA(2, 4) = 2
# LCA(0, 5) = 2
```

**Task**: Implement `lca_bst(root, p, q) -> TreeNode`. Use the BST property — no traversal of the whole tree needed.

Also implement `lca_binary_tree(root, p, q) -> TreeNode` for a **general** binary tree (no BST property).

---

## Problem 3 — Level-Order Zigzag

Return the level-order traversal of a binary tree where even levels (0-indexed) are left-to-right and odd levels are right-to-left.

```python
# Input:
#     3
#    / \
#   9  20
#     /  \
#    15   7
# Output: [[3], [20, 9], [15, 7]]
```

**Task**: Implement `zigzag_level_order(root) -> list[list[int]]`.

---

## Problem 4 — Binary Tree Maximum Path Sum

A **path** in a binary tree is a sequence of nodes where each pair is connected by an edge. The path must contain at least one node and does not need to pass through the root. Return the maximum sum of any path.

```python
# Input:
#     -10
#    /   \
#   9    20
#       /  \
#      15   7
# Output: 42  (path: 15 → 20 → 7)
```

**Task**: Implement `max_path_sum(root) -> int`. Nodes may have negative values — a path can be a single node.

---

## Problem 5 — Construct BST from Preorder

Given a list that is the preorder traversal of a BST, reconstruct the BST.

```python
# Input: [8, 5, 1, 7, 10, 12]
# Output BST inorder: [1, 5, 7, 8, 10, 12]
```

**Task**: Implement `bst_from_preorder(preorder: list[int]) -> TreeNode`. Aim for O(n) using a stack or bounds.

---

## Problem 6 — Kth Smallest in BST

Return the `k`-th smallest value (1-indexed) in a BST without converting the whole tree to a list first.

```python
# BST inorder: [1,2,3,4,5,6,7]
# kth_smallest(root, 3) → 3
```

**Task**: Implement `kth_smallest(root, k) -> int`. Use an iterative inorder traversal that stops early.

---

## Problem 7 — Symmetric Tree

Determine whether a binary tree is a mirror of itself (symmetric around its centre).

```python
#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3   → True
```

**Task**: Implement both a recursive `is_symmetric_recursive(root)` and an iterative `is_symmetric_iter(root)` using a queue.

---

## Problem 8 — Flatten Binary Tree to Linked List

Flatten a binary tree in place into a "linked list" using the right child pointers (preorder order), setting all left pointers to `None`.

```python
# Input:    1          Output: 1→2→3→4→5→6 (all via .right)
#          / \
#         2   5
#        / \   \
#       3   4   6
```

**Task**: Implement `flatten(root) -> None` (in-place mutation).

---

## Problem 9 — Top K Frequent Elements (heap)

Given a list of integers, return the `k` most frequent elements. If multiple elements have the same frequency, any order among them is acceptable.

```python
# nums=[1,1,1,2,2,3], k=2 → [1, 2]
```

**Task**: Implement `top_k_frequent(nums: list[int], k: int) -> list[int]`. Use `heapq`. Target O(n log k) — maintain a min-heap of size `k` to avoid sorting all n elements.

---

## Problem 10 — Serialize and Deserialize a Binary Tree

Design an algorithm to serialize a binary tree to a string and deserialize it back.

```python
# serialize(root) → "4,2,1,#,#,3,#,#,6,5,#,#,7,#,#"
# deserialize(s)  → original tree
```

**Task**: Implement `serialize(root) -> str` and `deserialize(s: str) -> TreeNode | None`. Use preorder + `#` sentinels. Both must be inverse functions.

---

## Session structure (suggested time allocation)

| Problems | Minutes | Focus |
|---|---|---|
| 1–3 | 60 | Traversal-based |
| 4–6 | 60 | BST-specific |
| 7–8 | 45 | Tree mutation / symmetry |
| 9 | 30 | Heap application |
| 10 | 45 | Design + round-trip test |

Write a `pytest` test for each problem verifying your worked examples before moving on.

---

## Complexity reference

| Problem | Expected time | Expected space |
|---|---|---|
| 1 — Diameter | O(n) | O(h) |
| 2 — LCA BST | O(h) | O(h) |
| 2 — LCA general | O(n) | O(h) |
| 3 — Zigzag | O(n) | O(n) |
| 4 — Max path sum | O(n) | O(h) |
| 5 — BST from preorder | O(n) | O(n) |
| 6 — Kth smallest | O(h + k) | O(h) |
| 7 — Symmetric | O(n) | O(h) |
| 8 — Flatten | O(n) | O(h) |
| 9 — Top K frequent | O(n log k) | O(n) |
| 10 — Serialize | O(n) | O(n) |

(Solutions: ../solutions/week-03/day-6.md)

## Self-check

- [ ] Solved all 10 problems independently (solutions checked only after attempt)
- [ ] Each solution has at least one passing pytest test
- [ ] Correctly analyzed time/space complexity for each
- [ ] Identified which problems need the BST property and which work on general trees
- [ ] Implemented both recursive and iterative variants where asked

## Supplementary links

- [LeetCode 543 — Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
- [LeetCode 235 — LCA of BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
- [LeetCode 124 — Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
- [LeetCode 297 — Serialize and Deserialize Binary Tree](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)
