# Phase 7 · Week 05 · Day 1 — URL Shortener (Full Design)

> Objective: Walk through a complete end-to-end system design for a URL shortener.  ·  Time: ~1 hr  ·  Prereqs: [[week-01/day-1]] (8-step framework)

---

## Concept

A URL shortener converts a long URL into a short alias (e.g. `https://bit.ly/3xQpR5`). Click the alias → redirect to the original. Deceptively simple on the surface, this problem exercises every layer: hashing, storage, caching, CDN, analytics, and abuse prevention.

**Why interviewers love it**: it is small enough to finish in 45 minutes yet has genuine scaling subtleties — especially around key-generation correctness and read-heavy traffic patterns.

---

## Step 1 — Requirements

### Functional
- `POST /shorten` → given a long URL, return a short URL.
- `GET /{code}` → redirect to original URL (HTTP 301 or 302).
- Optional: custom aliases, expiry dates, click analytics.

### Non-functional (constrain scope fast)
- Read-heavy: ~100:1 read/write ratio.
- Availability > consistency — a stale redirect is tolerable; an outage is not.
- Redirect latency P99 < 50 ms globally.
- Codes must be globally unique; no collision.
- Codes must be hard to enumerate (no sequential integers).

---

## Step 2 — Estimation

```
Write QPS  : 100 million URLs shortened / day  → ~1 150 writes/s
Read  QPS  : 10 billion redirects / day         → ~115 000 reads/s
Code length: 62^7 = 3.5 trillion codes (7-char base62 is plenty for decades)
Storage    : 100 M URLs/day × 365 days × 5 yr × ~500 B/row ≈ 90 TB
Cache      : 20% of daily reads hit 80% of URLs → cache ~20 M URLs
```

---

## Step 3 — API Design

```
POST /api/v1/shorten
Body: { "long_url": "https://...", "alias": "optional", "ttl_days": 30 }
→ 201 { "short_url": "https://sho.rt/aB3kQ9x", "code": "aB3kQ9x" }

GET /{code}
→ 301 Location: <original_url>      (301 = permanent, CDN-cacheable)
  302 Location: <original_url>      (302 = no CDN cache; use when tracking each click)

DELETE /api/v1/{code}               (owner only, JWT auth)
GET    /api/v1/{code}/stats         (click count, geo breakdown)
```

**301 vs 302 trade-off**: 301 is cached by browsers and CDNs — lower latency, but you lose per-click analytics. 302 hits your servers every time — accurate analytics, higher load. Most shorteners use 302.

---

## Step 4 — Data Model

```sql
-- Core table (write once, read many)
CREATE TABLE urls (
    code        CHAR(7)      PRIMARY KEY,   -- base62 short code
    long_url    TEXT         NOT NULL,
    owner_id    BIGINT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    click_count BIGINT       NOT NULL DEFAULT 0
);
CREATE INDEX idx_urls_owner ON urls(owner_id);
CREATE INDEX idx_urls_expires ON urls(expires_at) WHERE expires_at IS NOT NULL;

-- Click events (append-only; can be a separate analytics DB)
CREATE TABLE clicks (
    id          BIGSERIAL    PRIMARY KEY,
    code        CHAR(7)      NOT NULL,
    clicked_at  TIMESTAMPTZ  NOT NULL,
    ip_hash     BYTEA,
    user_agent  TEXT,
    country     CHAR(2)
);
```

**Storage choices**:
- `urls` table: Postgres (strong consistency for writes) + Redis cache for hot reads.
- `clicks` table: ClickHouse or Cassandra (append-only, analytical queries, time-series).

---

## Step 5 — High-Level Architecture

```
Client
  │
  ├─ POST /shorten ──► [API Service]
  │                         │
  │                    [Key Generator] ← pre-generates codes
  │                         │
  │                    [Postgres: urls]
  │
  └─ GET /{code} ───► [CDN / Edge Cache]
                           │ cache miss
                      [Redirect Service]
                           │
                      [Redis: code→url] ← hot path, TTL=24h
                           │ cache miss
                      [Postgres: urls]
```

```
┌─────────────────────────────────────────────────────────────┐
│  DNS → CDN (CloudFront / Cloudflare)                        │
│    ↓ 301 hit → serve from edge (no origin touch)            │
│    ↓ miss                                                   │
│  Load Balancer (round-robin)                                │
│    ├── Redirect Service  (stateless, horizontal scale)      │
│    │       ↓                                                │
│    │   Redis Cluster (read replicas)                        │
│    │       ↓ miss                                           │
│    │   Postgres (read replica, primary for writes)          │
│    │                                                        │
│    └── API Service  (write path)                            │
│            ↓                                                │
│        Key Generator Service                                │
│            ↓                                                │
│        Postgres primary + Redis invalidation                │
│                                                             │
│  Analytics: click events → Kafka → ClickHouse              │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 6 — Deep Dives

### 6a. Key Generation — the hard part

**Option A — Hash approach**
`MD5(long_url)` → take first 7 chars. Problem: collisions. If two URLs hash to the same prefix, one overwrites the other. Fix with collision-retry loop — but now write latency spikes under load.

**Option B — Counter + base62 encode**
Maintain a global counter (e.g. auto-increment in Postgres). Encode as base62. Problem: enumerable (attacker can scrape all URLs by iterating the counter).

**Option C — Key Generation Service (KGS)**
Pre-generate a pool of unique random 7-char base62 codes offline. Store in a `keys_available` table. API service atomically moves a key to `keys_used` at write time.

```
┌─────────────────────────────────────────┐
│  KGS process (runs periodically)        │
│    generates random 7-char codes        │
│    bulk-inserts into keys_available     │
└─────────────────────────────────────────┘
          ↓ (atomic row delete + insert)
API Service: SELECT FOR UPDATE one key from keys_available,
             DELETE it, INSERT into keys_used on write
```

Benefit: O(1) key assignment, no collision risk, not enumerable. Downside: need to manage key pool size.

**Option D — Snowflake-style distributed ID**
64-bit ID (timestamp + machine ID + sequence), encode to base62. No central coordinator needed. Fine at scale; slightly longer codes.

**Recommendation**: KGS for small teams, Snowflake-style for large distributed deployments.

### 6b. Cache strategy

- Cache key: short code.  Cache value: long URL.
- Cache on read (lazy); also warm on write.
- Eviction: LRU.  TTL: 24 h (or expiry date if set).
- On DELETE: explicit cache invalidation.
- Redis Cluster with read replicas handles 115K reads/s at ~0.1 ms latency.

### 6c. Redirect latency

- Edge cache (CDN with 302): ~5 ms for cache hit.
- Redis cache hit: ~1 ms round-trip.
- Postgres read replica: ~5–10 ms.
- Full stack P99 < 50 ms is achievable with regional Redis + CDN.

---

## Step 7 — Bottlenecks

| Bottleneck | Impact | Mitigation |
|---|---|---|
| Single Postgres primary | Write SPOF | Postgres streaming replication + automatic failover (Patroni) |
| Redis node failure | Cache misses flood DB | Redis Sentinel / Cluster; graceful degradation to DB |
| KGS key exhaustion | New writes fail | Alert on pool < 1 M keys; background refill |
| Hot URL (viral link) | Cache stampede | Request coalescing; probabilistic early expiry refresh |
| Click table grows unboundedly | Slow analytics | Partition by month; TTL older partitions to cold storage |

---

## Step 8 — Trade-offs

| Decision | What you gain | What you give up |
|---|---|---|
| 302 redirect | Accurate analytics | Higher origin load |
| KGS pre-generation | Zero-collision guarantees | Operational complexity of key service |
| Redis in front of Postgres | 0.1 ms read path | Eventual consistency (write then cache miss window) |
| Separate ClickHouse for analytics | Cheap analytical queries | Cross-system joins are harder |
| CDN caching (if using 301) | Global low latency | Cannot invalidate redirects quickly |

---

## Worked Example — Collision-Free Key Assignment (Python)

```python
# src/url_shortener/key_service.py
"""
Minimal Key Generation Service.
Generates base62 codes and stores them in a pool.
"""
import logging
import os
import random
import string
from pathlib import Path

logger = logging.getLogger(__name__)

BASE62 = string.ascii_letters + string.digits  # 62 chars

def _random_code(length: int = 7) -> str:
    """Generate a cryptographically adequate random base62 code."""
    return "".join(random.SystemRandom().choices(BASE62, k=length))


def generate_key_batch(batch_size: int = 10_000) -> list[str]:
    """Return a batch of unique random codes."""
    codes: set[str] = set()
    while len(codes) < batch_size:
        codes.add(_random_code())
    keys = list(codes)
    logger.info("Generated %d keys", len(keys))
    return keys


def encode_base62(n: int) -> str:
    """Encode a positive integer as a base62 string (Snowflake-style alternative)."""
    if n == 0:
        return BASE62[0]
    result = []
    while n:
        n, rem = divmod(n, 62)
        result.append(BASE62[rem])
    return "".join(reversed(result))


def decode_base62(s: str) -> int:
    """Decode a base62 string back to an integer."""
    return sum(BASE62.index(c) * (62 ** i) for i, c in enumerate(reversed(s)))
```

---

## Exercises

1. The team proposes using MD5(long_url)[0:7] as the short code. Identify two failure modes and propose a fix for each.
2. Sketch the cache-invalidation flow when a user deletes a shortened URL. What race condition can occur, and how do you prevent it?
3. A viral tweet causes 500 K requests/s for a single short code. How does your architecture handle this? What changes, if any, are needed?
4. Design the `GET /api/v1/{code}/stats` endpoint. What aggregations do you pre-compute? How do you avoid the `clicks` table becoming a query bottleneck?
5. Implement `encode_base62` and `decode_base62` from scratch and verify round-trip correctness with pytest for values 0, 1, 61, 62, 3_521_614_606_207 (62^7 − 1).

(Solutions: ../solutions/week-05/day-1.md)

---

## Self-check
- [ ] I can explain why 302 is preferred over 301 for analytics-driven shorteners.
- [ ] I can describe three key-generation strategies and their trade-offs.
- [ ] I can draw the full redirect path from DNS to database from memory.
- [ ] I can identify the top two bottlenecks under viral traffic and propose mitigations.

---

## Supplementary
- Designing TinyURL — System Design Primer (GitHub)
- Snowflake ID generation — Twitter Engineering Blog
- Redis Cluster architecture — redis.io/docs
