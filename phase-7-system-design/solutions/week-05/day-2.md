# Solutions — Week 05 · Day 2 (News Feed Exercises)

---

## Exercise 1 — User follows 5 000 accounts, 2 000 are celebrities

**Problem**: pure fanout-on-write for all 5 000 accounts would mean a celebrity's post triggers millions of writes (not 5 000 — each celebrity might have 5 M followers). But from the feed-reader's perspective, the question is: what does their feed-read do?

**Answer**:
With the hybrid model:
- 3 000 regular accounts → their posts were pushed to this user's Redis sorted set feed at write time. O(1) to read.
- 2 000 celebrity accounts → their posts were NOT pushed. At read time, the Feed Service fetches the last N posts for each of the 2 000 celebrity accounts.

**Optimization**: don't fetch all 2 000 individually. Pre-maintain a Redis sorted set per user listing only their celebrity followees: `celebrity_follows:{user_id}`. At read time, pipeline-fetch the latest 1 post from each celebrity's own post list (Redis key: `user_posts:{celebrity_id}`). Merge with the pre-built feed. A Redis pipeline of 2 000 GETs takes ~2–5 ms, which is within the 200 ms P99 budget.

**Further optimization**: if all 2 000 celebrities post rarely, cache the merged result for 30 s per user.

---

## Exercise 2 — Cursor-based vs. offset-based pagination

**Offset-based**: `SELECT ... LIMIT 20 OFFSET 1000`. The DB must scan and discard 1 000 rows before returning 20. At offset = 100 000, this is a full scan of 100 000 rows per page request. On a 10 B row table, this is catastrophically slow.

**Cursor-based**: `SELECT ... WHERE post_id < :cursor ORDER BY post_id DESC LIMIT 20`. The DB uses the index on `post_id` to seek directly to the cursor position. O(log N) seek + O(20) scan regardless of how deep in the pagination.

**Stability**: offset pagination breaks when new posts are inserted — items shift, causing duplicates or gaps across pages. Cursor pagination is stable: a cursor points to a specific row; new insertions above it don't affect it.

---

## Exercise 3 — Ranked feed

**What changes in the fanout pipeline**:

1. The score stored in the Redis sorted set changes from `timestamp` to `relevance_score`.
2. Relevance score is computed at fanout time (when writing to feed cache):
   ```
   score = w_recency × f(age) + w_engagement × engagement_velocity + w_relationship × closeness
   ```
3. The Fanout Worker now calls a lightweight Scoring Service (or inline ML model) per post.
4. Score staleness: if engagement spikes after fanout (post goes viral), the score in existing feeds is outdated.

**Mitigation**: store the timestamp as the score for ranking by recency, but also store engagement metadata in the feed entry. Re-score at read time using a fast in-process model. This adds ~5–10 ms at read but keeps scores fresh.

---

## Exercise 4 — Viral post: 1 M likes in 10 minutes

**Hot row problem**: `UPDATE posts SET like_count = like_count + 1 WHERE post_id = X` with 1 700 concurrent writers/s creates lock contention on a single Postgres row. This is a well-known pattern called a "hot row" or "counter problem."

**Solution — Redis counter with batch flush**:
1. Route all like increments to Redis: `INCR likes:{post_id}`.
2. A background job runs every 30 s: reads Redis counter, does a single bulk `UPDATE posts SET like_count = like_count + delta WHERE post_id = X`, resets Redis delta.

**Why this works**: Redis INCR is O(1) and non-blocking, handles millions of ops/s. Postgres write load drops from 1 700 writes/s to 1 write per 30 s for that row.

**Edge case**: if Redis fails, we lose the in-flight delta. Acceptable for like counts (eventual consistency). For critical counters (billing), use a durable queue instead.

---

## Exercise 5 — Adding "story" posts (24 h TTL)

**Schema changes**:

```sql
-- Add post_type column
ALTER TABLE posts ADD COLUMN post_type VARCHAR(10) NOT NULL DEFAULT 'post';
-- Values: 'post' | 'story'

-- Add expiry
ALTER TABLE posts ADD COLUMN expires_at TIMESTAMPTZ;
CREATE INDEX idx_posts_expires ON posts(expires_at) WHERE expires_at IS NOT NULL;
```

**Migration plan**:
1. Add nullable `post_type` and `expires_at` columns (backward-compatible; existing rows get defaults).
2. Deploy application code that reads both columns but only writes `story` type via new API endpoint.
3. Deploy a background expiry job: `DELETE FROM posts WHERE expires_at < now()` — run every 5 min.
4. Update feed assembly: filter stories to `expires_at > now()`. Add stories as a separate horizontal scroll in the UI (separate feed lane, not mixed with regular posts).

**Feed impact**: stories should NOT be pushed into the main feed sorted set. Store in a separate `stories:{user_id}` sorted set in Redis with `ZRANGEBYSCORE` using timestamp + 24 h window. Stories evict naturally when TTL expires.
