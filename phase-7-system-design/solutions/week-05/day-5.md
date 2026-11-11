# Solutions — Week 05 · Day 5 (Geo + Typeahead Exercises)

---

## Exercise 1 — Geohash cell boundary diagram

**Scenario**: user at lat/lon exactly on the boundary between cells `9q8yy` and `9q8yz`. A restaurant is 50 m away, just inside cell `9q8yz`.

```
┌──────────────┬──────────────┐
│              │  Restaurant  │
│  9q8yy       │  9q8yz       │
│      ×  USER │              │
│     (boundary)              │
└──────────────┴──────────────┘
```

Without the 9-cell search, querying only `9q8yy` misses the restaurant entirely — it's in `9q8yz`.

**9-cell search**: query all cells adjacent to the user's cell:

```
9q8yw  9q8yx  9q8yy
9q8wu  9q8wv  9q8ww   ← example neighbors (actual values depend on geohash encoding)
9q8wt  9q8ws  9q8wr
```

The restaurant is in one of the 8 neighbors → found. After fetching all candidates, a haversine distance filter removes false positives (cells that overlap the search radius but not the exact circle).

**When 9-cell still fails**: at geohash precision 6 (cell ~1.2 km), if the search radius is 100 m, the target could be in a non-adjacent cell at a diagonal and still be within 100 m. Solution: use precision 7 or 8 for small radii so the cell size is smaller than the search radius.

---

## Exercise 2 — Multi-word typeahead

**Problem**: query "new york pizza" — the prefix is a phrase, not a single word.

**Strategy change**:

1. **Store full phrases as keys**: treat the entire query string as the indexed entity. `typeahead:new` contains "new york", "new jersey pizza", etc. This already works with the prefix → sorted set approach.

2. **Tokenized prefix**: index each word in the phrase separately. "new york pizza" inserts into `typeahead:new`, `typeahead:york`, `typeahead:pizza` with the full phrase as the member. Then a search for "york" would surface "new york pizza".

3. **Most common approach**: index by the full phrase prefix only (approach 1). Users who type "new y" expect "new york pizza" — they are typing the beginning of the phrase. Tokenized indexing is for full-text search, which is better served by Elasticsearch.

**Implementation**: no change to the Redis structure; change the prefix extraction:
```python
def prefixes(query: str) -> list[str]:
    """All prefixes of the full query string."""
    return [query[:i] for i in range(1, len(query) + 1)]
```

---

## Exercise 3 — Redis memory estimate for top 1 M queries

**Given**: top 1 M queries, avg length 8 characters.

**Keys per query**: one sorted set entry per prefix → avg 8 keys per query.

**Total sorted set entries**: 1 M queries × 8 prefixes = 8 M sorted set entries (across 8 M keys).

**Redis sorted set memory per entry**:
- String member (avg 8 bytes query text) + sorted set overhead ≈ 50 bytes/entry.
- But we have 8 M separate sorted set keys, each with multiple members.

**More accurate model**:
- Unique prefixes: for top 1 M queries of avg length 8, there are ~8 M unique prefix keys.
- Each key holds a sorted set of the queries sharing that prefix — on average, many share a given 1–2 char prefix and few share a 7–8 char prefix.
- Average entries per sorted set: ~10 (long-tail distribution).
- Memory per sorted set with 10 string entries: ~500 bytes.
- Total: 8 M keys × 500 bytes = 4 GB.

**Verdict**: 4 GB is feasible in a Redis instance. A Redis Cluster with 3 nodes × 8 GB RAM easily fits this with headroom. For 1 B queries, scale to ~4 TB — not feasible in RAM; need sharding + disk-backed sorted set (use Redis on Flash or Elasticsearch).

---

## Exercise 4 — Trie delete with top-K eviction

```python
# src/typeahead/trie.py — delete method addition

def delete(self, word: str) -> bool:
    """
    Remove a word from the trie.
    Returns True if word existed; False otherwise.
    Updates top-K at all ancestor nodes.
    """
    # First, verify the word exists
    node = self.root
    path: list[tuple[TrieNode, str]] = []  # (node, char) pairs
    for char in word:
        if char not in node.children:
            return False  # word not in trie
        path.append((node, char))
        node = node.children[char]

    if node.frequency == 0:
        return False  # word was never inserted as a complete word

    # Remove the word's frequency
    node.frequency = 0

    # Update top-K at all ancestor nodes (including the terminal node)
    all_nodes = [self.root] + [n for n, _ in path] + [node]
    for n in all_nodes:
        n.top_k = [(neg_f, w) for neg_f, w in n.top_k if w != word]
        heapq.heapify(n.top_k)

    # Clean up leaf nodes that are no longer needed
    # Walk back and prune childless, non-terminal nodes
    for i in range(len(path) - 1, -1, -1):
        parent, char = path[i]
        child = parent.children[char]
        if not child.children and child.frequency == 0:
            del parent.children[char]
        else:
            break  # stop pruning — still needed as prefix

    return True
```

**Test cases**:
```python
def test_delete_removes_from_search_results():
    t = Trie()
    t.insert("apple", 10)
    t.insert("apply", 5)
    assert "apple" in t.search("appl")
    t.delete("apple")
    assert "apple" not in t.search("appl")

def test_delete_nonexistent_returns_false():
    t = Trie()
    assert t.delete("banana") is False

def test_delete_prefix_word_keeps_longer_word():
    t = Trie()
    t.insert("app", 5)
    t.insert("apple", 10)
    t.delete("app")
    assert "apple" in t.search("app")

def test_delete_updates_top_k():
    t = Trie()
    for word, freq in [("apple", 10), ("apply", 8), ("apt", 6)]:
        t.insert(word, freq)
    t.delete("apple")
    results = t.search("ap")
    assert results[0] == "apply"  # highest remaining
    assert "apple" not in results
```

---

## Exercise 5 — 50 km radius with geohash precision 4

**Geohash precision 4**: cells ≈ 39 km × 20 km (latitude × longitude).

**9-cell search covers**: 3 × 3 = 9 cells × 39 km width ≈ 117 km × 60 km area.

A 50 km radius circle has diameter 100 km. The 9-cell grid at precision 4 covers a ~117 × 60 km rectangle around the user — this just barely covers a 50 km radius in the east-west direction but may miss corners.

**Problem**: precision 4 cells are so large that the 9-cell search may return businesses 100+ km away that are within the adjacent cells. The haversine filter then has to discard a large fraction of results — potentially thousands of false positives per query.

**Better approach for 50 km radius**: use precision 5 (cells ≈ 5 km × 5 km). Then the 50 km radius requires querying a 10 × 10 grid of cells = 100 cells. Still manageable with a Redis pipeline of 100 ZRANGE commands.

**Rule of thumb**: choose precision such that `cell_size ≈ radius / 2`. For 50 km: precision 5 (5 km cells) or precision 4 with a 25-cell search (5 × 5 grid) for full coverage.
