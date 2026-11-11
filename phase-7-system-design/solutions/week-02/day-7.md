# Solutions — Day 7: Project Review & Scaling Reads vs Writes

## Exercise 1 — CQRS sketch for trending products

```
Write path:
  [Product View Event] → POST /events/product-view
       │
       ▼
  [Event Handler] → writes to event_log table (append-only)
       │
       ▼
  [Async worker / cron, runs every 60s]
       │
       ▼
  SELECT category, product_id, COUNT(*) as views
  FROM event_log
  WHERE ts > NOW() - INTERVAL 1 HOUR
  GROUP BY category, product_id
  ORDER BY views DESC
  LIMIT 10
       │
       ▼
  [Write to read model: trending_products table]
  trending_products(category, rank, product_id, view_count, updated_at)

Read path:
  [API: GET /trending?category=electronics]
       │
       ▼
  [Query Handler] → SELECT * FROM trending_products WHERE category=?
       │
       ▼  (optionally cached in Redis, TTL=60s)
  [Response: top-10 products for category]
```

Trade-offs:
- Read model is stale by up to 60s (the worker interval) — acceptable for trending.
- Write model (event_log) is normalised and append-only — no contention.
- Read model is pre-aggregated — single-table query, no GROUP BY at read time.
- The two models can be scaled independently (event_log on write-heavy DB; trending_products on read-heavy replica).

---

## Exercise 2 — Shard key selection for a messaging app

**Proposed shard key: `user_id` (sender or conversation participant)**

Query patterns:
- "Fetch messages by user" → `WHERE sender_id = ?` or `WHERE thread_id IN (threads_for_user)`. Sharding by user_id keeps all of a user's sent messages on one shard → no cross-shard scatter for the primary query.
- "Fetch thread" → threads are identified by a pair of user_ids. Store threads on the shard of `min(user_id, counterpart_id)`. Thread fetches may require a cross-shard join for edge cases (rare — threads belong to one shard by definition).

Hot-key risk: celebrity users with millions of messages → one shard gets disproportionate load. Mitigation: consistent hashing (ring + virtual nodes) so adding shards distributes load more evenly; or tenant-aware shard routing for known large accounts.

Re-sharding: hash sharding (`shard = hash(user_id) % N`) makes re-sharding disruptive — prefer consistent hashing or a directory (lookup table) mapping user ranges to shards. Directory sharding allows moving individual users without rehashing the full keyspace.

---

## Exercise 3 — URL shortener design

**Scale targets**: 10K writes/sec, 1M reads/sec → read:write = 100:1

**Data model**:
```sql
short_urls (
    short_code VARCHAR(8) PRIMARY KEY,  -- e.g. "a3F9bZ"
    long_url    TEXT NOT NULL,
    created_at  TIMESTAMP,
    expires_at  TIMESTAMP,
    hit_count   BIGINT DEFAULT 0
)
```

**Write path** (10K/sec):
1. API accepts `POST /shorten {long_url}`.
2. Generate 8-char base62 code (random or sequential via distributed counter).
3. Write to primary DB + invalidate CDN/cache for that short_code (usually N/A — new code).
4. Optional: write-through to Redis simultaneously.

**Read path** (1M/sec):
1. `GET /{code}` hits CDN edge first.
2. CDN caches: `Cache-Control: max-age=300, stale-while-revalidate=60` for non-expiring URLs.
3. CDN miss → hits app server (load balanced via L7 ALB with least-conn).
4. App server checks Redis (cache-aside, TTL=600s) → if miss, queries DB.
5. 301 or 302 redirect to long URL.
6. ~95% of redirects served from CDN or Redis → DB sees ~50K req/sec at most.

**LB algorithm**: consistent hashing on `short_code` — routes requests for the same code to the same app server, increasing per-process in-memory cache hit rate.

**CDN**: enabled for all GET redirects. Cache key = short_code only (no query string). Purge on URL expiry via Cloudflare API.

**Architecture diagram**:
```
User → [CDN PoP] ──cache hit──► 302 Redirect
              │ cache miss
              ▼
          [ALB: consistent hash on short_code]
              │
    ┌─────────┴─────────┐
    ▼                   ▼
[App-A: Redis cache] [App-B: Redis cache]
         │ miss
         ▼
    [Read replica DB]    ← 99% of reads
    [Primary DB]         ← writes only
```

---

## Exercise 4 — Week reflection prompts

**`caching_layer.py` production improvements**:
- Add distributed tracing: generate/propagate a `trace_id` through `get()` and `invalidate()` calls; log it at each step.
- Add Prometheus metrics: replace `CacheMetrics` dataclass counters with `prometheus_client.Counter` and `Gauge`; expose `/metrics` endpoint.
- Add circuit breaker: if the DB backend fails N times consecutively, trip the circuit and return `None` instead of queueing more failing DB calls.

**`consistent_hash_lb.py` production improvements**:
- Add graceful drain: before `remove_backend`, call `drain(backend, timeout=30s)` that stops routing new requests to that backend but lets in-flight requests complete.
- Add replication awareness: support `get_n_backends(key, n)` to return the N responsible backends (for replication without a separate coordinator).
- Persist ring state: write ring snapshot to Redis/etcd so all service instances share the same ring view (avoiding split-brain during scale events).
