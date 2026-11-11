# Phase 7 · Week 05 · Day 5 — Geo/Nearby + Typeahead/Autocomplete

> Objective: Design a nearby-places search system and a typeahead autocomplete service.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-4]]

---

## Part A — Geo / Nearby Places

## Concept

"Find restaurants within 1 km of me" sounds like a simple proximity query. At scale — hundreds of millions of places, billions of queries per day — it requires specialized spatial indexing. Naive SQL `WHERE distance < X` is O(N) per query.

**The key insight**: encode 2D coordinates into 1D strings (geohash or quadtree cells) so that proximity becomes prefix matching — a problem that any B-tree index or key-value store can handle efficiently.

---

### A1 — Requirements

**Functional**
- Return businesses within a given radius of a lat/lon.
- Filter by category (restaurant, hotel, etc.).
- Results sorted by distance.
- Business CRUD for operators.
- Support area up to 50 km radius.

**Non-functional**
- 500 M places in the index.
- Read QPS: 1 M/s at peak.
- Write QPS: 100/s (business data changes infrequently).
- Freshness: new/updated businesses appear in search within 60 s.
- Query latency P99 < 50 ms.

---

### A2 — Spatial Indexing

#### Geohash

Divide the Earth into a grid. Each cell is encoded as a base32 string. Nearby cells share a common prefix.

```
Precision:  6 chars → ~1.2 km × 0.6 km cell
            7 chars → ~152 m × 152 m cell
            8 chars → ~38 m × 19 m cell

San Francisco: 9q8yy (6-char geohash)
Nearby places share prefix 9q8yy or 9q8yx (adjacent cells)
```

**Algorithm**:
1. Encode user's position to geohash of precision matching the desired radius.
2. Compute the 8 neighboring cells (user's cell + 8 adjacent).
3. Query index for all places with geohash starting with any of those 9 prefixes.
4. Filter by exact distance (haversine formula) to remove edge cases.

**Problem with geohashes**: precision boundaries. A place just across a cell boundary shares no prefix with you even if it's 1 m away. The 9-cell search solves most cases.

#### Quadtree

Recursively divide the map into 4 quadrants. Keep splitting until each leaf node has ≤ K businesses. Store tree in a DB as a flat table.

Good for: non-uniform density (city centers vs. deserts). Harder to query than geohash.

#### Google S2 Library

Hierarchical cells on a sphere (Hilbert curve-based). Used by Google Maps, Uber, etc. Better than geohash for edge cases (poles, meridian).

**Recommendation**: Use geohash for interviews (easy to explain + implement). Mention S2 as the production choice.

---

### A3 — Architecture

```
Client → API Gateway → [Nearby Search Service]
                              │
                         [Geohash Index]           [Business DB]
                         Redis ZSET per cell        Postgres
                         Key: geo:{geohash6}         (source of truth)
                         Members: business_ids
                         Score: geohash7 (for sort)
                              │
                         [Business Cache]           [Search Index]
                         Redis: biz:{id}            Elasticsearch
                         (name, lat, lon, category)  (text + geo queries)
```

**Write path**:
1. Operator creates/updates business via Business Service.
2. Writes to Postgres.
3. Publishes event to Kafka.
4. Geo-index worker: computes geohash, upserts into Redis ZSET.
5. Search-index worker: updates Elasticsearch.

**Read path**:
1. User sends `GET /nearby?lat=37.7&lon=-122.4&radius=1000&category=restaurant`.
2. Search Service computes geohash + 8 neighbors.
3. Redis `ZRANGEBYLEX` on each of the 9 keys → list of business IDs.
4. Fetch details for each ID from Redis (or Postgres on miss).
5. Haversine filter → sort by distance → return top 20.

---

### A4 — Trade-offs

| Approach | Latency | Complexity | Scales to |
|---|---|---|---|
| Postgres PostGIS | 10–50 ms | Low | ~10 M places |
| Geohash + Redis | < 1 ms | Medium | 500 M+ places |
| Elasticsearch geo | 5–20 ms | Medium | 100 M places |
| S2 + custom index | < 1 ms | High | 1 B+ places |

---

## Part B — Typeahead / Autocomplete

## Concept

Typeahead predicts what a user is typing and suggests completions. Google Search, YouTube, Amazon search — all use it. The challenge: return the top N completions (by popularity) for any prefix, in < 100 ms, across billions of possible strings.

---

### B1 — Requirements

**Functional**
- Return top 5 most popular completions for a prefix.
- Completions are search queries or product names.
- Update frequency scores as users interact.
- Support multi-language (unicode-safe).

**Non-functional**
- 10 B queries indexed; 1 M QPS reads; 10 K QPS writes (new queries).
- Response < 100 ms.
- Eventual consistency for frequency updates (a 1-minute lag is fine).
- Prefix queries up to 20 characters.

---

### B2 — Data Structures

#### Trie (prefix tree)

```
                    ('')
                   / | \
                  a  b  c
                 /|   \
                p p     e
                |  \     \
                p   l     ar
                |    \
               le   le

→ "apple", "apply", "belle", "bear"
```

- Each node: character + children map + is_terminal + top_K_suggestions (cached up the tree).
- Query: traverse path matching prefix → return `top_k` at that node.
- Update: increment frequency at leaf → bubble up top-k changes.

**Problem at scale**: a trie for 10 B strings doesn't fit in one machine. Need distributed trie or a different approach.

#### Distributed Approach: Prefix → Redis Sorted Set

Instead of a trie, store per-prefix data directly:

```
Key:    typeahead:{prefix}           (e.g. "typeahead:appl")
Type:   Sorted Set
Member: completion string
Score:  frequency (higher = better)

ZREVRANGE typeahead:appl 0 4 WITHSCORES → top 5 for "appl"
```

**Update**: when a query "apple" is submitted:
1. For each prefix of "apple" ("a", "ap", "app", "appl", "apple"):
   `ZINCRBY typeahead:{prefix} 1 "apple"`

**Storage**: a 20-character query generates 20 prefix keys. At 10 B distinct queries × 20 prefixes = 200 B keys. At 50 bytes/key = 10 TB — too much for RAM. Need sharding and tiering.

**Optimization**: only store prefixes up to length 10 (most users type < 10 chars before selecting). Cache top-100 completions per prefix.

#### Hybrid: Offline Trie + Online Redis Cache

Build the trie offline (daily/hourly batch job) from query logs. Serialize to a compact binary format. Load into memory per service shard. Use Redis only for real-time frequency deltas. Merge at query time.

---

### B3 — Architecture

```
User types "appl"
    │
    ▼
[Typeahead API Service]  (stateless, horizontal scale)
    │
    ├─ L1: in-process LRU cache (last 1 K popular prefixes)
    │
    ├─ L2: Redis Cluster (prefix → sorted set)
    │       Key: typeahead:{prefix}
    │
    └─ L3: Trie Service (gRPC, sharded by prefix range)
            Shards: a-f → shard 1, g-m → shard 2, n-z → shard 3

Write path (async):
User submits query → Kafka topic queries.submitted
Aggregator worker: batch 10 s windows → bulk ZINCRBY to Redis
Trie rebuild job: runs hourly → reads query logs → rebuilds trie shards
```

---

### B4 — Trie Implementation (Python reference)

```python
# src/typeahead/trie.py
"""
In-memory trie with top-K caching at each node.
Used by the Trie Service shard.
"""
import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

TOP_K = 5


@dataclass
class TrieNode:
    children: dict[str, "TrieNode"] = field(default_factory=dict)
    frequency: int = 0
    top_k: list[tuple[int, str]] = field(default_factory=list)  # (freq, word) max-heap


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str, frequency: int = 1) -> None:
        """Insert word with given frequency; update top-K along the path."""
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.frequency += frequency
        self._update_top_k(word, node.frequency)
        logger.debug("Inserted %r freq=%d", word, node.frequency)

    def _update_top_k(self, word: str, freq: int) -> None:
        """Walk back up from root, updating top-K at each node."""
        node = self.root
        path_nodes = [node]
        for char in word:
            node = node.children[char]
            path_nodes.append(node)
        for n in path_nodes:
            # Replace entry if word already in top-k, else push
            existing = {w for _, w in n.top_k}
            if word in existing:
                n.top_k = [(f, w) if w != word else (-freq, word) for f, w in n.top_k]
            else:
                heapq.heappush(n.top_k, (-freq, word))
            if len(n.top_k) > TOP_K:
                heapq.heappop(n.top_k)

    def search(self, prefix: str) -> list[str]:
        """Return top-K completions for the given prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        results = sorted(node.top_k)  # sort by neg-freq → best first
        return [word for _, word in results]
```

---

## Exercises

1. A user is at the exact boundary between two geohash cells. Show with a diagram how the 9-cell search ensures the nearby business is found.
2. The typeahead service must support multi-word queries (e.g. "new york pizza"). How does the prefix indexing strategy change?
3. Estimate the Redis memory usage for storing all prefixes of the top 1 M queries (avg length 8 chars). Is it feasible to keep this in RAM?
4. Implement a `delete(word)` method for the `Trie` class and handle top-K eviction correctly.
5. The geo search system must return results up to 50 km radius. Geohash precision 4 ≈ 39 × 20 km. How many cells do you need to query, and what is the risk of returning too many false positives?

(Solutions: ../solutions/week-05/day-5.md)

---

## Self-check
- [ ] I can explain how geohashing converts proximity search to prefix matching.
- [ ] I can describe why the 9-cell search is needed and when it still fails.
- [ ] I can implement a trie with cached top-K at each node.
- [ ] I can sketch the distributed typeahead architecture with Redis prefix keys.

---

## Supplementary
- Geohash encoding/decoding — geohash.org
- Google S2 geometry library — s2geometry.io
- "Rebuilding Our Search Engine" — LinkedIn Engineering Blog
