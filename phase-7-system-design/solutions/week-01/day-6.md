# Solutions — Day 6: Full System-Design Document (URL Shortener — Reference Design)

This is the reference solution for the Day 6 Saturday practice session. Compare against your own document after completing it independently.

---

# System Design: URL Shortener (Reference)

## 1. Requirements

### 1.1 Functional
- Users can submit a long URL and receive a unique short code (7-character base62).
- Any user with the short URL can be redirected to the original URL.
- URLs can optionally have an expiry time.
- The system tracks how many times each short URL is clicked.
- Users can deactivate their own short URLs.

### 1.2 Non-Functional
- 99.99% availability (≤ 4.38 min downtime/month).
- Redirect p99 latency < 10 ms globally.
- 100 M new URLs/day; 10 B redirects/day.
- 5-year data retention; no data loss (RPO = 0).
- Analytics: eventually consistent, updated within 60 s.

### 1.3 Out of Scope
- Custom vanity aliases (e.g., `/my-brand`).
- Real-time analytics dashboard.
- A/B testing / link-in-bio pages.
- OAuth / SSO; assume bearer token authentication.

---

## 2. Capacity Estimation

### 2.1 QPS
```
Write: 100M / 86,400 ≈ 1,157 writes/sec. Peak (3×): 3,500/sec.
Read:  10B / 86,400  ≈ 115,700 reads/sec. Peak (3×): 347,000/sec.
Read:write ratio ≈ 100:1.
```

### 2.2 Storage
```
Per record:
  short_code   7 B (base62)
  long_url     100 B (average)
  user_id      8 B
  created_at   8 B
  expires_at   8 B
  ─────────────────
  Total        131 B ≈ 150 B (with overhead)

Daily new records: 1,157/sec × 86,400 ≈ 100 M/day × 150 B = 15 GB/day
5-year storage: 15 GB × 365 × 5 = 27.4 TB

Conclusion: fits on a single large server (100 TB SSD possible);
sharding not required for storage volume alone, but write throughput
(3,500/sec) warrants a primary + replica setup.
```

### 2.3 Bandwidth
```
Inbound: 3,500 × 150 B = 525 KB/s — trivial.
Outbound redirects: 347,000 × 300 B (redirect response) = 100 MB/s ≈ 800 Mbps.
CDN required: 800 Mbps sustained outbound exceeds a single origin server;
              redirect responses are short and highly cacheable (301).
```

### 2.4 Cache Sizing
```
Hot set heuristic: 20% of URLs receive 80% of clicks.
Recent 7 days: 100 M/day × 7 = 700 M records × 150 B = 105 GB.
20% of that: 21 GB.
Cache a single Redis node (64 GB) covers the hot set with room to spare.
Eviction: LRU. TTL = 24 h (refreshed on access).
```

---

## 3. API Design

```
POST /urls
  Request:  { "url": "https://example.com/very-long-path", "expires_in": 86400 }
  Response: 201 { "short_code": "aB3xYz9", "short_url": "https://sho.rt/aB3xYz9",
                  "expires_at": "2026-11-12T10:00:00Z" }
  Errors:   400 (invalid URL), 401 (unauthenticated), 422 (URL exceeds max length)

GET /{short_code}
  Response: 302 Location: <original_url>
  Errors:   404 (code not found), 410 (expired/deactivated)

DELETE /urls/{short_code}
  Response: 204 No Content
  Errors:   401, 403 (not owner), 404

GET /urls/{short_code}/stats
  Response: 200 { "short_code": "aB3xYz9", "total_clicks": 1427383,
                  "clicks_last_24h": 10234, "created_at": "..." }
  Errors:   404
```

---

## 4. Data Model

**Choice: PostgreSQL (SQL).** Data is relational (URLs belong to users); write volume (1,157/sec) is well within a single Postgres primary's capability; ACID guarantees simplify expiry logic.

```sql
-- urls
CREATE TABLE urls (
    short_code  CHAR(7)      NOT NULL PRIMARY KEY,
    long_url    TEXT         NOT NULL,
    user_id     BIGINT       NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    is_active   BOOLEAN      NOT NULL DEFAULT true
);
CREATE INDEX idx_urls_user_id ON urls(user_id);
CREATE INDEX idx_urls_expires_at ON urls(expires_at) WHERE expires_at IS NOT NULL;

-- analytics (write-heavy; separate table to avoid locking urls)
CREATE TABLE clicks (
    id          BIGSERIAL    PRIMARY KEY,
    short_code  CHAR(7)      NOT NULL REFERENCES urls(short_code),
    clicked_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    referrer    TEXT
);
-- Partition by day for efficient pruning
-- For high-volume analytics, pipe to a time-series store (ClickHouse)
```

---

## 5. High-Level Design

```
                     ┌─────────────────┐
                     │   CDN Edge      │  cached 301 redirects (TTL 1 h)
                     └────────┬────────┘
                              │ HTTPS
                              ▼
                     [ Load Balancer ]
                              │ HTTP
                 ┌────────────┴────────────┐
                 ▼                         ▼
         [ API Service ]           [ API Service ]  (stateless, N instances)
                 │
         ┌───────┼──────────────────┐
         │       │                  │
         ▼       ▼                  ▼
    { Redis }  ( Postgres     ( Postgres      [ Analytics Worker ]
     Cache      Primary )      Replica )       │
                  │               │             │ pub/sub
                  └── SQL repl ───┘         < Kafka >
                                                 │
                                          [ Click Aggregator ]
                                                 │ batch write
                                          ( ClickHouse / Postgres clicks )
```

**Component responsibilities:**
- **CDN Edge:** caches redirect responses. Cache-Control: `max-age=3600, public`.
- **Load Balancer:** health checks, SSL termination, round-robin to API.
- **API Service:** stateless; handles create/redirect/delete; reads from Redis first.
- **Redis:** LRU cache of `short_code → long_url`. Single write-through on create.
- **Postgres Primary:** authoritative URL store. Synchronous replication to one replica.
- **Postgres Replica:** serves read queries (stats, admin) to offload primary.
- **Kafka:** decouples click events from the critical redirect path.
- **Click Aggregator:** consumes Kafka topic; batches and upserts into analytics store every 60 s.

---

## 6. Deep Dives

### 6.1 Short-Code Generation Strategy

**Approach chosen: Auto-increment ID + base62 encode.**

```
ID space: 62^7 = 3,521,614,606,208 ≈ 3.5 trillion codes.
At 100 M/day: 3.5T / 100M = 35,000 days ≈ 95 years of capacity.
```

| Approach | Collision risk | Throughput | Enumerability | Chosen? |
|---|---|---|---|---|
| Hash + truncate | Yes (need retry logic) | High | Low | No |
| Pre-generated key pool | None | High (pre-computed) | Low | No (operational overhead) |
| Auto-increment + base62 | None | High | Medium (mitigated) | Yes |

Enumeration mitigation: XOR the integer ID with a secret salt before encoding. This makes codes non-sequential without adding computation.

```python
import os

_SALT = int.from_bytes(os.environ["CODE_SALT"].encode()[:8], "big")

def encode(n: int) -> str:
    chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    n = n ^ _SALT
    result: list[str] = []
    while n > 0:
        result.append(chars[n % 62])
        n //= 62
    return "".join(reversed(result)).zfill(7)
```

### 6.2 Redirect Performance

Target: p99 < 10 ms globally.

Request path (cache hit — > 99% of traffic):
```
1. Client → CDN edge (cached 301): ~1–5 ms (global PoP)
2. CDN miss → API service: add ~10 ms network (within SLO for intra-region)
3. API → Redis GET: ~0.2 ms
4. Return 302 + Location header
Total (CDN hit): < 5 ms. Total (Redis hit): < 15 ms.
```

To achieve p99 < 10 ms on cache hit:
- CDN must cache aggressively. Short URL redirects are immutable once created.
- Redis must be co-located with API (same AZ). Cross-AZ Redis adds ~1–2 ms.
- API code must avoid any synchronous I/O beyond Redis on the redirect path.
- Analytics write is fire-and-forget: publish to Kafka with `acks=0` (best-effort, tolerable given eventually consistent requirement).

---

## 7. Bottlenecks & Failure Modes

| Component | Bottleneck | Scale threshold | Mitigation |
|---|---|---|---|
| Redis single node | Memory | ~64 GB hot set | Redis Cluster (hash slots); shard by short_code prefix |
| Postgres primary | Write throughput | ~5,000 writes/sec | Read replica offloads reads; Postgres handles 3,500 writes/sec comfortably |
| CDN cache invalidation | Stale redirects after deactivation | Any deactivation | Purge CDN on `DELETE`; set short TTL (1 h) so stale window is bounded |
| Kafka consumer lag | Analytics staleness grows | High click burst | Scale consumer group; partition by short_code hash |

---

## 8. Trade-offs

| Decision | Chosen | Alternative | Reason |
|---|---|---|---|
| DB | Postgres (SQL) | DynamoDB (NoSQL) | Data is relational; write volume modest; ACID simplifies expiry |
| Code generation | Base62 auto-increment | Hash-truncate | No collision risk; trivial to implement |
| Cache | Redis (shared) | Local in-process | Consistent across all API instances; survives restart |
| Analytics path | Async (Kafka → aggregator) | Synchronous DB write | Keeps redirect p99 < 10 ms; analytics can be 60 s stale per NFRs |
| Redirect type | 302 (Found) | 301 (Moved Permanently) | 301 is cached by browsers forever — prevents future deactivation from working. 302 allows server-side control. Note: 301 would save more server load. |
| Global distribution | CDN + origin | Multi-region active-active | Simpler operationally; CDN covers 90%+ of traffic at edge |
