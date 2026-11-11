# Solutions — Week 01 Assessment

---

## Part 1: Conceptual

**1. Most dangerous mistake at step 3:**
Jumping directly to the component diagram without completing the API and data model steps. Arriving at "let me just draw the boxes" before defining the interface means the diagram is ungrounded — you can't explain why each component exists. Also dangerous: spending 15 minutes still on requirements with 30 minutes left. Signal to move: "I think I have enough to proceed — let me design the API."

**2. SLO tighter than SLA:**
SLA = customer contract: "we guarantee 99.9% availability." SLO = internal target: "we target 99.95% availability." The gap is deliberate headroom. If the system drifts toward 99.95% without triggering an incident, it still won't breach 99.9%. This gives engineering time to detect and fix issues before legal/financial consequences trigger.

**3. Why p50 is insufficient:**
p50 = 40 ms means half the users are faster. It says nothing about the other half. If p99 = 800 ms, 1 in 100 requests (possibly thousands per second) violates the SLO. The relevant metric is the SLO percentile — likely p95 or p99. Users who experience the tail are real users; optimising only the median can mask systematic problems under load.

**4. Cassandra vs Postgres — CAP analysis:**
Choose Postgres (CP). The NFR says "data must never be lost" — this rules out AP systems where a partition can allow divergent writes that later conflict. Cassandra's tunable consistency can be set to QUORUM to approximate CP, but the primary driver is durability, not raw throughput. At 100 k reads/sec, a Postgres read replica cluster handles this. At 10 k writes/sec, a single Postgres primary handles it (write ceiling is ~10–50 k simple inserts/sec on modern NVMe hardware). The stale-reads-ok constraint would allow replicas, which is consistent with Postgres + synchronous standby.

**5. Two-tier fan-out:**
Single-tier: the message service directly calls the notification API for each subscriber. At 10 M subscribers, a single call takes minutes; the message service is blocked; other messages queue up. Two-tier: Tier 1 publishes one message per topic partition to Kafka (fast). Tier 2: multiple consumer groups (fan-out workers), each responsible for a shard of subscribers, read from Kafka and call APNs/FCM in parallel. This converts a sequential operation into a parallel one bounded only by the number of workers and external API rate limits.

**6. 0.08% budget spent, 10 days left:**
Remaining budget: 0.1% - 0.08% = 0.02% of 43,200 min = 8.64 min. Engineering decisions forced:
- Freeze non-critical deployments for the rest of the month.
- Any deploy requires a staged rollout (canary) with automatic rollback on error rate spike.
- On-call escalation threshold lowered — any incident risks consuming the remaining 8 min.
- Post-mortem the 0.08% incident to prevent recurrence.

**7. Push vs. Pull:**
- Push: server sends updates to clients when available. Example: WebSocket-based Twitter/X real-time feed. Trade-off: server must maintain persistent connections and track all active clients; harder to implement backpressure.
- Pull: clients poll periodically. Example: RSS feed readers, legacy email clients. Trade-off: latency is bounded by poll interval (e.g., 30 s); simpler server, wastes bandwidth on empty polls.

**8. Cascading latency — 10 vs 3 hops:**
At p99 = 50 ms per service:
- 3 hops: ~150 ms chain p99.
- 10 hops: ~500 ms chain p99.
With network overhead (0.5 ms intra-DC each way), 10 hops add 10 ms network alone. But the real damage is statistical: with 10 independent services, the probability that at least one is in its tail is much higher. P(all 10 within p99) = 0.99^10 ≈ 90%, meaning 10% of requests hit a tail event somewhere in the chain vs. 3% for 3 hops. With 500 k requests/sec, that's 50 k requests/sec hitting a slow path.

---

## Part 2: Design Challenge — Pastebin (Reference Design)

### 1. Functional Requirements
- Users can create a paste (text, up to 10 MB) and receive a short URL.
- Anyone with the short URL can view the paste.
- Pastes expire according to user-specified retention (1 hr / 1 day / 1 week / 1 month / forever).
- Users can delete their own pastes before expiry.

**Out of scope:** syntax highlighting, user accounts/auth (assume anonymous), version history, comments, private/password-protected pastes.

### 2. Non-Functional Requirements
- Availability: 99.9% (43.2 min/month downtime budget).
- Read latency: p99 < 200 ms for pastes ≤ 1 KB; p99 < 2 s for pastes up to 10 MB.
- Scale: 5 M DAU; 1 M writes/day; 100 M reads/day.
- Durability: no data loss (RPO = 0).
- Consistency: strong (a created paste must be readable immediately).

### 3. Capacity Estimation
```
Write QPS: 1M / 86,400 ≈ 12 writes/sec. Peak (3×): 36/sec.
Read  QPS: 100M / 86,400 ≈ 1,157 reads/sec. Peak (3×): 3,500/sec.

Storage per paste:
  Metadata: paste_id (8B) + user_id (8B) + created_at (8B) + expires_at (8B) = 32 B
  Content:  avg 10 KB (text pastes skew small; 90% < 10 KB, 10% up to 10 MB)
  Effective avg: ~10 KB

Daily storage: 12/sec × 86,400 × 10 KB = ~10 GB/day
5-year: 10 × 365 × 5 ≈ 18 TB

Bandwidth:
  Write: 36 × 10 KB = 360 KB/s — trivial.
  Read:  3,500 × 10 KB = 35 MB/s — manageable. 
         Large pastes (10 MB): if 1% of reads are large: 35 × 10 MB = 350 MB/s peak. CDN needed.

Cache sizing:
  Hot pastes (20% of content, 80% of traffic): 0.2 × 18 TB → too large to cache entirely.
  Cache last 7 days: 7 × 10 GB = 70 GB. Use Redis cluster (128 GB). LRU eviction.
```

### 4. API Design
```
POST /pastes
  Request:  { "content": "...", "expires_in": 86400, "language": "python" }
  Response: 201 { "paste_id": "aB3xYz9", "url": "https://paste.io/aB3xYz9",
                  "expires_at": "2026-11-12T10:00:00Z" }
  Errors:   400 (content empty), 413 (content > 10 MB)

GET /pastes/{id}
  Response: 200 { "content": "...", "language": "python", "created_at": "...",
                  "expires_at": "...", "view_count": 1234 }
  Errors:   404, 410 (expired)
```

### 5. Data Model

**Choice: Postgres for metadata; object storage (S3) for paste content.**

```sql
CREATE TABLE pastes (
    id          CHAR(7)      NOT NULL PRIMARY KEY,
    storage_key TEXT         NOT NULL,  -- S3 object key
    user_id     BIGINT,
    language    VARCHAR(32),
    size_bytes  INT          NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    view_count  BIGINT       NOT NULL DEFAULT 0
);
CREATE INDEX idx_pastes_expires_at ON pastes(expires_at) WHERE expires_at IS NOT NULL;
```

Paste content stored in object store (not DB) because 10 MB BLOB per row would bloat the Postgres table and make backups expensive.

### 6. High-Level Design

```
Client
  │ HTTPS
  ▼
[ CDN ]  ← caches GET /pastes/{id} responses (Cache-Control: max-age=300)
  │ (miss)
  ▼
[ Load Balancer ]
  │
[ API Service ] ── Redis GET/SET ── { Redis Cache }
  │                                        (miss)
  ├── metadata query ─────────────── ( Postgres )
  └── content read/write ──────────── ( S3 Object Store )

Expiry Worker (cron, every 5 min):
  DELETE FROM pastes WHERE expires_at < now();
  S3 lifecycle rule: delete objects older than 5 years.
```

### 7. Deep Dive: Paste Storage Strategy for Large Pastes

For pastes ≤ 1 KB: store inline in Redis and Postgres `content` column (no S3 hop needed).
For pastes > 1 KB: store content in S3; Postgres stores the S3 key only.

Read path for large paste:
1. API checks Redis (key = `paste:{id}:content`). Cache hit: return immediately.
2. Cache miss: fetch metadata from Postgres → get `storage_key`.
3. Fetch content from S3 using pre-signed URL or direct SDK call.
4. Write to Redis with TTL proportional to recency (recent pastes: 1 h; old pastes: 5 min).

This two-tier approach means p99 < 200 ms for small pastes (Redis hit) and p99 < 2 s for large pastes (S3 retrieval, CDN cached on second access).

### 8. Bottlenecks

1. **S3 cold reads for large pastes:** First request after cache eviction fetches from S3 (~100–500 ms). Mitigation: aggressive CDN caching (Cache-Control: max-age=3600) for public pastes.
2. **Postgres `view_count` increment:** 1,157 UPDATE/sec on the `pastes` table can create hot rows. Mitigation: batch increment using a Redis counter; flush to Postgres every 60 s with an UPDATE ... + delta.

### 9. Trade-offs

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| Content storage | S3 + Postgres metadata | Postgres BYTEA | Postgres BYTEA at 10 MB/row would make table backups 18 TB/5yr; S3 is purpose-built for large objects |
| Caching | Redis + CDN | In-process only | Multi-instance consistency; CDN absorbs repeated reads |
| Expiry | DB cron + S3 lifecycle | Lazy delete on read | Proactive deletion frees storage; lazy delete leaves orphaned S3 objects |
| Read consistency | Strong | Eventual | A paste must be immediately readable after creation; eventual would confuse users |

---

## Part 3: Debugging & Problem-Solving

**1. Hypothesis:**
Cache-miss race condition: new short codes are written to Postgres but the Redis cache write fails (network error, timeout, or Redis eviction under memory pressure). Reads that hit a cold cache after 10 minutes fall back to Postgres — but only if the DB read path is correctly implemented. The 10-minute window matches the TTL of a newly populated cache entry expiring or never being written.

More specifically: if the system uses **write-around caching** (writes go directly to DB without populating cache), then all reads for new codes require a DB lookup. If the DB replica used for reads has replication lag > the request timing, reads to the replica return no row.

**2. Exact sequence (replica lag scenario):**
1. Client calls `POST /urls` → new URL written to Postgres primary.
2. Replication to replica is async; replica lag = 30 s.
3. Client calls `GET /aB3xYz9` within 30 s of creation.
4. API checks Redis → cache miss (new code, not yet cached).
5. API reads from Postgres **replica** → row not yet replicated → returns 404.
6. After 30+ minutes (codes are old): replica has caught up → reads succeed.

**3. Fix:**
Option A (preferred): Read new short codes from the primary until they are confirmed replicated. Use a **read-your-writes** pattern: after a write, pin the session to the primary for the next N seconds (e.g., 30 s), then allow replica reads.
Option B: On redirect, always read from Redis. On Redis cache miss, read from **primary** (not replica). Write the result to Redis with a 1-hour TTL. This adds slight primary read load but ensures correctness.
Option C: Populate Redis synchronously on URL creation (write-through). Then redirect reads only need Redis + primary fallback.

**4. Testing without production bug:**
- Write an integration test that creates a URL and immediately GETs it, using a test DB configuration with simulated replication lag (delay the replica WAL replay by 30 s using `pg_ctl` or a proxy like Toxiproxy).
- Assert that the GET returns 302, not 404.
- Test should fail before the fix, pass after.

---

## Part 4: Code Quality

**1. Correctness bug:**
Race condition between `INCR` and `EXPIRE`. If two requests arrive simultaneously when the key doesn't exist, both see `count == 1` and both try to set the TTL. This is benign. The real bug: if `INCR` succeeds but `EXPIRE` fails (e.g., Redis crashes between the two commands), the key never expires. After the window, the counter grows forever and the user is permanently rate-limited.

Fix: use a Lua script or `SET key 1 EX window NX` pattern to make the creation atomic.

**2. Operational/security concern:**
The Redis connection is a module-level global created at import time. This means:
- Connection parameters are hardcoded (`localhost`, port `6379`), not configurable via environment variables.
- No connection pool — every call to `check_rate_limit` may reuse or create connections non-deterministically.
- No authentication or TLS — connection is plaintext and unauthenticated.

Fix: use `redis.Redis.from_url(os.getenv("REDIS_URL"))` and inject the client.

**3. Type hints:**

```python
from __future__ import annotations
from pydantic import BaseModel


class RateLimitResult(BaseModel):
    allowed: bool
    current_count: int
    limit: int
    window_seconds: int


def check_rate_limit(
    client: redis.Redis,
    user_id: str,
    limit: int = 100,
    window: int = 60,
) -> RateLimitResult:
    ...
```

**4. Injectable Redis client:**

```python
from typing import Protocol

class RedisClient(Protocol):
    def incr(self, key: str) -> int: ...
    def expire(self, key: str, seconds: int) -> bool: ...

def check_rate_limit(
    client: RedisClient,
    user_id: str,
    limit: int = 100,
    window: int = 60,
) -> bool:
    ...
```

In tests: pass a `FakeRedis` or `fakeredis.FakeRedis()` instance. No real Redis server needed.

**5. Logging:**

```python
import logging

logger = logging.getLogger(__name__)

# After increment:
logger.info(
    "rate_limit_check",
    extra={"user_id": user_id, "count": count, "limit": limit, "window": window},
)

# When limit exceeded:
if count > limit:
    logger.warning(
        "rate_limit_exceeded",
        extra={"user_id": user_id, "count": count, "limit": limit},
    )
    return False
```

Fields to include: `user_id` (for incident investigation), `count` (current value), `limit` (configured threshold), `window` (so log entries are self-explanatory). Never log tokens or full request bodies.
