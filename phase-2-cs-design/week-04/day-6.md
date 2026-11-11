# Phase 2 · Week 04 · Day 6 — Graph Problem Set (Saturday Deep Practice)

> Objective: Consolidate the week's graph algorithms by solving six progressively harder problems spanning BFS, DFS, cycle detection, shortest path, and Eulerian path.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Overview

Six problems spanning the full graph algorithm toolkit introduced this week. Work through them in order — each one exercises a distinct technique. Problems 1-2 are warm-ups (20-30 min each), problems 3-4 are medium difficulty (40-50 min each), and problems 5-6 are hard (50-70 min each). Aim to code each solution from scratch before checking the solutions file.

Patterns covered: grid BFS/DFS for connected components, BFS graph cloning with a hash map, topological sort / cycle detection via DFS coloring, BFS shortest path on a word graph, Dijkstra single-source shortest path with a min-heap, Hierholzer's algorithm for an Eulerian path in a directed graph.

---

## Problem Set

### Problem 1: Number of Islands

You are given an `m x n` grid of `"1"` (land) and `"0"` (water) characters. An island is a group of adjacent land cells connected horizontally or vertically (not diagonally). Return the number of islands.

**Input:** `grid: list[list[str]]` — an `m x n` grid containing only `"1"` and `"0"`.

**Output:** `int` — total number of distinct islands.

**Examples:**

```
Input:
[["1","1","0","0","0"],
 ["1","1","0","0","0"],
 ["0","0","1","0","0"],
 ["0","0","0","1","1"]]
Output: 3

Input:
[["1","1","1","1","0"],
 ["1","1","0","1","0"],
 ["1","1","0","0","0"],
 ["0","0","0","0","0"]]
Output: 1
```

**Constraints:**
- `1 <= m, n <= 300`
- `grid[i][j]` is either `"0"` or `"1"`.
- You may modify the grid in place (mark visited cells) or keep a separate `visited` set.

**Expected time complexity:** O(m * n) — each cell visited at most once.
**Expected space complexity:** O(min(m, n)) for BFS queue in the worst case; O(m * n) for DFS recursion stack if the grid is all land.

**Hints:**
1. Iterate every cell. When you find a `"1"` that has not been visited, increment a counter and launch a BFS/DFS that marks all connected land cells as visited.
2. BFS: push the starting cell onto a queue, then while the queue is non-empty, pop a cell, and for each of its four neighbors that is `"1"` and unvisited, mark it visited and push it. This floods the entire island before you move on.

---

### Problem 2: Clone Graph

Given a reference to a node in a connected undirected graph, return a deep copy (clone) of the graph. Each node has an integer value and a list of neighbors.

```python
class Node:
    def __init__(self, val: int = 0, neighbors: list["Node"] | None = None) -> None:
        self.val = val
        self.neighbors: list[Node] = neighbors or []
```

**Input:** `node: Node | None` — reference to any node in the graph (or `None` for an empty graph).

**Output:** `Node | None` — reference to the corresponding node in the cloned graph.

**Examples:**

```
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
(node 1 connects to 2 and 4; node 2 connects to 1 and 3; etc.)
Output: a deep copy where no original Node objects appear

Input: adjList = [[]]  (single node, no neighbors)
Output: Node(val=1, neighbors=[])

Input: adjList = []  (empty graph, node is None)
Output: None
```

**Constraints:**
- Number of nodes: `0 <= n <= 100`
- `1 <= Node.val <= 100`; values are unique.
- No self-loops or repeated edges.

**Expected time complexity:** O(V + E) — visit each node and edge once.
**Expected space complexity:** O(V) for the hash map mapping original nodes to their clones.

**Hints:**
1. Use a dict `visited: dict[Node, Node]` mapping each original node to its clone. Before creating a new clone, check the dict — if the original is already there, return the existing clone. This handles cycles.
2. BFS: start from the given node. When dequeuing a node, iterate its neighbors. For each neighbor not yet in `visited`, create a clone and enqueue the original. Always append `visited[neighbor]` (the clone) to the current clone's neighbor list.

---

### Problem 3: Course Schedule

There are `num_courses` courses numbered `0` to `num_courses - 1`. You are given a list of prerequisite pairs `[a, b]` meaning you must take course `b` before course `a`. Return `True` if it is possible to finish all courses (i.e., there is no cycle in the dependency graph), `False` otherwise.

**Input:**
- `num_courses: int` — total number of courses.
- `prerequisites: list[list[int]]` — list of `[course, prereq]` pairs.

**Output:** `bool`

**Examples:**

```
Input: num_courses=2, prerequisites=[[1,0]]
Output: True
(Take 0 first, then 1.)

Input: num_courses=2, prerequisites=[[1,0],[0,1]]
Output: False
(Course 0 requires 1 and course 1 requires 0 — cycle.)

Input: num_courses=4, prerequisites=[[1,0],[2,0],[3,1],[3,2]]
Output: True
```

**Constraints:**
- `1 <= num_courses <= 2000`
- `0 <= prerequisites.length <= 5000`
- Each `prerequisites[i]` has length 2 and values in `[0, num_courses - 1]`.
- No duplicate prerequisite pairs.

**Expected time complexity:** O(V + E).
**Expected space complexity:** O(V + E) for the adjacency list and color array.

**Hints:**
1. Build a directed graph where edge `u -> v` means "u requires v". A valid finish order exists if and only if this graph is a directed acyclic graph (DAG). Cycle detection on a directed graph requires three node states: unvisited, currently in the DFS stack (gray), and fully processed (black). If DFS reaches a gray node, a cycle exists.
2. Alternatively, use Kahn's algorithm (BFS topological sort): track in-degrees, start with all zero-in-degree nodes, reduce neighbors' in-degrees as you process, and check whether all nodes were processed. If `processed_count < num_courses`, a cycle prevented some nodes from being reached.

---

### Problem 4: Word Ladder

Given a `begin_word`, an `end_word`, and a `word_list`, return the length of the shortest transformation sequence from `begin_word` to `end_word` such that each step changes exactly one letter and each intermediate word exists in `word_list`. Return `0` if no such sequence exists.

**Input:**
- `begin_word: str`
- `end_word: str`
- `word_list: list[str]`

**Output:** `int` — length of shortest sequence (counting both endpoints), or `0`.

**Examples:**

```
Input: begin_word="hit", end_word="cog",
       word_list=["hot","dot","dog","lot","log","cog"]
Output: 5
(hit -> hot -> dot -> dog -> cog)

Input: begin_word="hit", end_word="cog",
       word_list=["hot","dot","dog","lot","log"]
Output: 0
(end_word "cog" not in word_list)

Input: begin_word="a", end_word="c",
       word_list=["a","b","c"]
Output: 2
```

**Constraints:**
- `1 <= begin_word.length == end_word.length <= 10`
- `1 <= word_list.length <= 5000`
- All words consist of lowercase English letters.
- `begin_word != end_word`.

**Expected time complexity:** O(M^2 * N) where M = word length, N = number of words in the list.
**Expected space complexity:** O(M^2 * N) for the intermediate word pattern buckets (optimized BFS) or O(N) for the visited set (naive approach).

**Hints:**
1. Model the problem as a graph: each word is a node, and two words are connected if they differ by exactly one letter. BFS from `begin_word` finds the shortest path to `end_word` by hop count. Return the level (hop count + 1 to count words, not edges) when `end_word` is first dequeued.
2. Generating neighbors efficiently: for each word, try replacing each character position with every letter `a-z`. If the resulting word is in `word_set`, it is a neighbor. This avoids comparing all pairs of words.

---

### Problem 5: Network Delay Time

You have a network of `n` nodes labeled `1` to `n`. You are given a list of directed, weighted edges `times` where `times[i] = [u, v, w]` means it takes `w` milliseconds to send a signal from node `u` to node `v`. A signal is sent from node `k`. Return the minimum time it takes for all `n` nodes to receive the signal. Return `-1` if it is impossible for all nodes to receive the signal.

**Input:**
- `times: list[list[int]]` — list of `[source, destination, weight]` triples.
- `n: int` — number of nodes.
- `k: int` — source node.

**Output:** `int` — max of shortest distances to all nodes from `k`, or `-1` if any node is unreachable.

**Examples:**

```
Input: times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2
Output: 2

Input: times=[[1,2,1]], n=2, k=2
Output: -1
(Node 1 is unreachable from node 2.)

Input: times=[[1,2,1],[2,3,2],[1,3,4]], n=3, k=1
Output: 3
(1->2->3 costs 3, cheaper than 1->3 directly at 4.)
```

**Constraints:**
- `1 <= k <= n <= 100`
- `1 <= times.length <= 6000`
- `times[i]` values: `1 <= u, v <= n`, `u != v`, `1 <= w <= 100`.
- All `(u, v)` pairs are unique (no duplicate directed edges).

**Expected time complexity:** O((V + E) log V) with a binary min-heap.
**Expected space complexity:** O(V + E) for the adjacency list and dist array.

**Hints:**
1. Classic Dijkstra: maintain a `dist` dict initialized to infinity for all nodes except `k` (set to 0). Use `heapq` as a min-heap of `(cost, node)` tuples. Pop the lowest-cost node, skip if already finalized, then relax each outgoing edge. After Dijkstra, return `max(dist.values())` — the signal reaches the last node at that time. If any value is still infinity, return `-1`.
2. Remember to build the adjacency list from the `times` input before running Dijkstra. Nodes not appearing as sources still need entries in the adjacency list (even if empty) so you can detect unreachable nodes.

---

### Problem 6: Reconstruct Itinerary

You are given a list of airline tickets `tickets` where `tickets[i] = [from_i, to_i]`. Reconstruct the itinerary starting from `"JFK"`. All tickets must be used exactly once. If multiple valid itineraries exist, return the one with the smallest lexicographical order.

**Input:** `tickets: list[list[str]]` — list of `[departure, arrival]` airport code pairs.

**Output:** `list[str]` — the reconstructed itinerary starting with `"JFK"`.

**Examples:**

```
Input: [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]

Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
(not ["JFK","SFO","ATL","JFK","ATL","SFO"] — lexicographically larger)

Input: [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]
Output: ["JFK","NRT","JFK","KUL"]
```

**Constraints:**
- `1 <= tickets.length <= 300`
- `tickets[i].length == 2`
- `from_i != to_i`
- All airport codes are uppercase 3-letter strings.
- The input is guaranteed to have at least one valid itinerary.

**Expected time complexity:** O(E log E) — sorting adjacency lists dominates.
**Expected space complexity:** O(V + E) for the adjacency lists and result stack.

**Hints:**
1. This is an Eulerian path problem: find a path that uses every edge exactly once in a directed graph. Hierholzer's algorithm solves it in O(V + E). Sort each node's destination list in reverse lexicographical order so that `list.pop()` always yields the lexicographically smallest next destination.
2. DFS post-order: use an explicit stack (or recursion). When a node has no remaining outgoing edges, append it to the result. Reverse the result at the end. The key insight is that dead-end nodes (where you get "stuck") are always the last nodes in the final itinerary — post-order naturally places them at the back.

---

## Self-check

- [ ] I can implement grid BFS/DFS to count connected components without looking up the four-directional neighbor pattern.
- [ ] I can explain why a hash map is needed to clone a graph and how it handles cycles.
- [ ] I can perform DFS cycle detection on a directed graph using three-color marking (unvisited, in-stack, done).
- [ ] I can implement Kahn's algorithm (BFS topological sort) as an alternative cycle-detection approach.
- [ ] I can implement BFS on a word graph where neighbors are generated character-by-character, not from an explicit edge list.
- [ ] I can implement Dijkstra with `heapq` from memory and correctly handle the "already finalized" skip.
- [ ] I can explain why the maximum of all shortest distances gives the network delay time.
- [ ] I can describe Hierholzer's algorithm at a high level and explain why post-order DFS produces the Eulerian path.
- [ ] I solved all six problems within the 4-hour window (or identified and noted which ones need more review).
