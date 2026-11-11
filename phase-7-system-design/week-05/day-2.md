# Phase 7 · Week 05 · Day 2 — News Feed / Timeline (Fanout Models)

> Objective: Design a scalable news feed system and understand push vs. pull fanout trade-offs.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-1]]

---

## Concept

A news feed aggregates posts from accounts you follow and surfaces them in reverse-chronological (or ranked) order. Instagram, Twitter/X, LinkedIn, and Facebook all have this feature. The core challenge is **fanout**: when user A posts, that post must appear in the feeds of all A's followers — potentially millions of people — in near real-time.

**Two fundamental models**:
- **Fanout-on-write (push model)**: when a post is created, immediately write it to every follower's feed cache.
- **Fanout-on-read (pull model)**: when a user opens the app, fetch recent posts from everyone they follow on the fly.

Neither model wins in all cases — the right answer is a hybrid.

---

## Step 1 — Requirements

### Functional
- Follow/unfollow users.
- Create a post (text, image, video).
- Retrieve your home feed (paginated, ~20 items).
- Like, comment, share a post.
- Feed is roughly reverse-chronological; ranking is optional scope.

### Non-functional
- 500 M daily active users (DAU).
- Feed load P99 < 200 ms.
- Posts visible to followers within ~2 s of creation.
- High availability; eventual consistency acceptable (a follower seeing a post 5 s late is fine).
- No data loss for posts.

---

## Step 2 — Estimation

```
DAU            : 500 M
Avg follows    : 200 per user
Posts/day      : 50 M (1 post per 10 DAU)
Fanout write   : 50 M × 200 avg followers = 10 B feed-row writes/day → ~115 000 writes/s (peak 3×)
Feed reads     : 500 M × 5 opens/day = 2.5 B reads/day → ~29 000 reads/s
Storage (posts): 50 M × 1 KB = 50 GB/day → 18 TB/year (media stored in object store)
Feed cache     : 500 M users × 20 post IDs × 8 B ≈ 80 GB (fits in Redis cluster)
```

The fanout write rate (115 K/s) is the hardest constraint — it drives the hybrid model decision.

---

## Step 3 — API Design

```
POST /api/v1/posts
Body: { "text": "...", "media_ids": ["..."], "visibility": "PUBLIC" }
→ 201 { "post_id": "p_abc123", "created_at": "..." }

GET /api/v1/feed?limit=20&cursor=<opaque>
→ 200 { "posts": [...], "next_cursor": "..." }

POST /api/v1/users/{user_id}/follow
DELETE /api/v1/users/{user_id}/follow

POST /api/v1/posts/{post_id}/like
GET  /api/v1/posts/{post_id}/comments?limit=20&cursor=<opaque>
```

Cursor-based pagination (not offset) — offset pagination is O(offset) on large tables.

---

## Step 4 — Data Model

```sql
-- Users
CREATE TABLE users (
    user_id     BIGINT PRIMARY KEY,
    username    VARCHAR(30) UNIQUE NOT NULL,
    follower_count INT NOT NULL DEFAULT 0,
    following_count INT NOT NULL DEFAULT 0
);

-- Social graph
CREATE TABLE follows (
    follower_id BIGINT NOT NULL,
    followee_id BIGINT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (follower_id, followee_id)
);
CREATE INDEX idx_follows_followee ON follows(followee_id);

-- Posts
CREATE TABLE posts (
    post_id     BIGINT PRIMARY KEY,      -- Snowflake ID (time-sortable)
    author_id   BIGINT NOT NULL,
    body        TEXT,
    media_urls  TEXT[],
    created_at  TIMESTAMPTZ NOT NULL,
    like_count  INT NOT NULL DEFAULT 0
);
CREATE INDEX idx_posts_author_time ON posts(author_id, created_at DESC);

-- Feed cache is Redis, not a SQL table
-- Key: feed:{user_id}  Value: sorted set of post_ids by score=timestamp
```

---

## Step 5 — High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Client                                                      │
│    ↓ POST /posts          ↓ GET /feed                        │
│  [Post Service]        [Feed Service]                        │
│       ↓                       ↓                             │
│  [Postgres: posts]     [Redis: feed:{uid}]  ← prebuilt feed  │
│       ↓                       ↓ miss                        │
│  [Fanout Service]      [Feed Builder] ← on-demand (celebs)  │
│  (async via Kafka)            ↓                             │
│       ↓                [Postgres: posts + follows]           │
│  [Redis: feed:{uid}]                                         │
│  (for regular users)                                         │
│                                                              │
│  Object Store (S3): media uploads                            │
│  CDN: media delivery                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 6 — Deep Dives

### 6a. Fanout-on-Write (Push Model)

When a post is created:
1. Write post to Postgres.
2. Publish post event to Kafka topic `post.created`.
3. **Fanout Workers** consume the event, look up the author's followers, and for each follower write `(post_id, timestamp)` into `feed:{follower_id}` in Redis (sorted set by timestamp).

**Pros**: feed reads are O(1) Redis sorted set range scans — extremely fast.

**Cons**: a user with 10 M followers creates 10 M Redis writes per post. At peak a celebrity post can saturate workers for seconds.

### 6b. Fanout-on-Read (Pull Model)

When a user opens their feed:
1. Fetch the list of accounts the user follows.
2. For each, fetch their latest N posts from Postgres (or a per-user post list in Redis).
3. Merge-sort the results.

**Pros**: no fanout cost at write time; always fresh.

**Cons**: N follow lookups + N post-list fetches = high read fanout. For users following 2 000 accounts this is unacceptably slow.

### 6c. Hybrid Model (the right answer)

```
Regular users (< 1 M followers) → fanout-on-write
  Post creation → Kafka → Fanout Worker → write to followers' Redis feeds

Celebrities (≥ 1 M followers) → fanout-on-read
  Their posts are NOT pushed to follower feeds.
  At feed-read time, fetch celebrity posts on-demand and merge with pre-built feed.
```

**Celebrity detection**: maintain a `celebrity` flag on the `users` table. Fanout workers skip celebrity posts.

```
Feed read = Redis_feed_for_user  UNION  recent_posts_from_followed_celebrities
                                (merge by timestamp, return top 20)
```

### 6d. Feed Ranking

Raw reverse-chronological feed is easy. Ranked feed requires:
- Score = f(recency, engagement_rate, relationship_strength, content_type).
- Pre-score at fanout time (store score instead of timestamp as sorted-set score).
- Or re-score at read time using a lightweight ML model (latency budget: ~10 ms).

### 6e. Pagination

Use cursor = last seen `post_id` (Snowflake — time-embedded). Next page: `WHERE post_id < :cursor ORDER BY post_id DESC LIMIT 20`. Stable across new posts being inserted.

---

## Step 7 — Bottlenecks

| Bottleneck | Impact | Mitigation |
|---|---|---|
| Celebrity post creates 10 M writes | Latency spike, worker queue backup | Hybrid model; rate-limit celebrity fanout to 500 K/s spread |
| Postgres `posts` table grows unboundedly | Slow range scans | Shard by `author_id`; archive posts > 90 days to cold storage |
| Redis feed cache eviction | User sees stale/empty feed | TTL = 7 days; if miss, fall back to pull model for that user |
| Follow graph read (celebrity) | N+1 query per feed load | Denormalize followers list into Redis set per user |
| Like counter hot row | High contention on `like_count` | Counter in Redis, batch-flush to Postgres every 30 s |

---

## Step 8 — Trade-offs

| Decision | Gain | Cost |
|---|---|---|
| Hybrid fanout | Handles both regular and celebrity cases | Added complexity in feed assembly |
| Redis sorted sets for feed | Sub-ms feed reads | Memory cost; feed state can diverge from DB |
| Kafka for fanout | Decouples write path; retryable | Operational overhead; ~200 ms additional fanout latency |
| Snowflake IDs | Time-sortable, no DB round-trip for ordering | Requires ID generation service |
| Eventual consistency on feed | High write throughput | User may not see new post for 1–2 s |

---

## Worked Example — Feed Assembly Logic (Python)

```python
# src/feed/assembly.py
"""
Assemble a user's feed by merging their pre-built feed cache
with on-demand celebrity posts.
"""
import heapq
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(order=True)
class FeedPost:
    score: int  # negative timestamp for max-heap behavior with heapq (min-heap)
    post_id: int
    author_id: int
    body: str


def merge_feeds(
    cached_feed: list[FeedPost],
    celebrity_posts: list[FeedPost],
    limit: int = 20,
) -> list[FeedPost]:
    """
    Merge cached regular-user feed with celebrity posts.
    Returns top `limit` posts sorted by score descending.
    Both input lists must be sorted by score descending.
    """
    result: list[FeedPost] = []
    # Use a heap merge (k-way merge); both inputs are sorted
    merged = heapq.merge(cached_feed, celebrity_posts)
    for post in merged:
        result.append(post)
        if len(result) == limit:
            break
    logger.debug("Assembled feed: %d posts (limit=%d)", len(result), limit)
    return result
```

---

## Exercises

1. A user follows 5 000 accounts. 2 000 of them are celebrities. How does your feed assembly strategy handle this efficiently?
2. Explain cursor-based vs. offset-based pagination. Why does offset degrade on large tables?
3. The engagement team wants to show posts ranked by a score (not purely chronological). What changes in the fanout pipeline?
4. A post goes viral: 1 M likes in 10 minutes. How do you avoid a hot row on `like_count` in Postgres?
5. Design the schema changes and migration plan for adding "story" posts (24 h TTL) to the existing system.

(Solutions: ../solutions/week-05/day-2.md)

---

## Self-check
- [ ] I can explain fanout-on-write vs. fanout-on-read and when each fails.
- [ ] I can describe the hybrid model and the celebrity threshold decision.
- [ ] I can sketch the Kafka-driven fanout pipeline from post creation to feed update.
- [ ] I can identify the like-counter hot-row problem and the Redis-batching solution.

---

## Supplementary
- Instagram Engineering: "What Powers Instagram" — engineering.fb.com
- Twitter Timelines at Scale — QCon talk (2013, still relevant)
- Redis Sorted Sets documentation — redis.io
