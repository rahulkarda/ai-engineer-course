# Solutions — Week 01 Project: URL Shortener Reference Design

This is the complete reference design document. Compare against your submission after completing it independently.

---

# System Design: URL Shortener — Reference Solution

## 1. Requirements

### 1.1 Functional
1. A user can submit any valid URL and receive a unique 7-character short code.
2. Anyone who visits the short URL is redirected to the original URL.
3. URLs can have an optional expiry time (1 hr / 1 day / 1 week / 1 month / no expiry).
4. Authenticated users can deactivate their own short URLs.
5. The system tracks how many times each short URL is visited (view count, eventually consistent).

### 1.2 Non-Functional
| NFR | Target | Source |
|---|---|---|
| Availability | 99.99% | Constraint |
| Redirect latency | p99 < 10 ms (global) | Constraint |
| New URLs/day | 100 M | Constraint |
| Redirects/day | 10 B | Constraint |
| Data retention | 5 years | Constraint |
| Analytics lag | ≤ 60 s | Constraint |
| Data durability | No loss (RPO = 0) | Implied by "5-year retention" |
| Geography | Global (US, EU, APAC) | Constraint |

### 1.3 Out of Scope
- Custom vanity aliases (e.g., `/my-brand`).
- Real-time analytics dashboard with time-series breakdowns.
- A/B testing variants per short URL.
- OAuth / SSO (authentication assumed via bearer token).
- Link previews / social metadata (Open Graph).

---

## 2. Capacity Estimation

### 2.1 QPS
```
Write QPS: 100 M / 86,400 ≈ 1,157 writes/sec
Peak write: 1,157 × 3 = 3,471 writes/sec (3× multiplier for business hours peak)

Read QPS:  10 B / 86,400 ≈ 115,741 reads/sec
Peak read: 115,741 × 3 = 347,222 reads/sec

Read:write ratio = 100:1. System is overwhelmingly read-heavy.
```

### 2.2 Storage
```
Per record:
  short_code   7 B (CHAR)
  long_url     100 B (average; some URLs are longer, some shorter)
  user_id      8 B (BIGINT)
  created_at   8 B (TIMESTAMPTZ)
  expires_at   8 B (TIMESTAMPTZ, nullable)
  is_active    1 B (BOOLEAN)
  ─────────────────────────────
  Row total    132 B ≈ 150 B (with index and row header overhead)

Daily new records: 1,157/sec × 86,400 sec = ~100 M records/day
Daily storage added: 100 M × 150 B = 15 GB/day
5-year total: 15 GB × 365 × 5 = 27.4 TB

Sharding assessment:
  27.4 TB fits on a single large server (modern NVMe arrays: 100+ TB).
  However, 3,471 writes/sec at 99.99% availability requires:
    - Primary + synchronous replica (no data loss)
    - Read replicas to serve 347 k reads/sec
  Horizontal sharding of the DB not required for this scale.
```

### 2.3 Bandwidth
```
Inbound (URL creation):
  3,471 writes/sec × ~500 B per request body = 1.7 MB/s — trivial

Outbound (redirects):
  347,222 reads/sec × ~300 B per redirect response = 100 MB/s = 800 Mbps sustained
  Peak: 300 MB/s = 2.4 Gbps

CDN required: origin serving 2.4 Gbps peak is feasible with a multi-server cluster,
but redirect responses (301/302 + Location header) are highly cacheable.
A CDN handles 90%+ of traffic at the edge with < 5 ms latency globally.
```

### 2.4 Cache Sizing
```
Hot set heuristic (Pareto): 20% of URLs receive 80% of clicks.
Active URLs in last 7 days: 100 M/day × 7 = 700 M × 150 B = 105 GB

20% hot set: 21 GB. A single 64 GB Redis node covers this.

Eviction policy: LRU (allkeys-lru).
TTL strategy: set TTL = 24 h on first access; refresh TTL on each cache hit.
Cold starts: use lazy population (cache on first read, not on write).

Base62 code capacity check:
  62^7 = 3,521,614,606,208 ≈ 3.5 trillion unique codes
  At 100 M/day: 3.5 T / 100 M = 35,000 days = 95+ years before exhaustion
  7-character code is sufficient.
```

---

## 3. API Design

```
POST /urls
  Auth: Bearer token (optional — allows unauthenticated creation)
  Request:
    {
      "url": "https://example.com/very/long/path?param=value",
      "expires_in": 86400   // seconds; omit for no expiry
    }
  Response 201:
    {
      "short_code": "aB3xYz9",
      "short_url": "https://sho.rt/aB3xYz9",
      "original_url": "https://example.com/very/long/path?param=value",
      "expires_at": "2026-11-12T10:00:00Z"  // null if no expiry
    }
  Errors:
    400 Bad Request    — invalid URL format
    413 Payload Too Large — URL exceeds 2048 chars
    422 Unprocessable  — URL scheme not http/https

GET /{short_code}
  Auth: None
  Response 302:
    Location: <original_url>
    Cache-Control: public, max-age=3600  // 1 h CDN caching
  Errors:
    404 Not Found    — code does not exist
    410 Gone         — code has expired or been deactivated

DELETE /urls/{short_code}
  Auth: Bearer token (must be owner)
  Response: 204 No Content
  Errors:
    401 Unauthorized
    403 Forbidden     — not the owner
    404 Not Found

GET /urls/{short_code}/stats
  Auth: Bearer token (must be owner or public if system is configured so)
  Response 200:
    {
      "short_code": "aB3xYz9",
      "total_clicks": 14273830,
      "clicks_last_24h": 102340,
      "clicks_last_7d": 875000,
      "created_at": "2026-11-11T08:00:00Z",
      "expires_at": null,
      "is_active": true
    }
  Errors: 401, 403, 404
```

---

## 4. Data Model

**Choice: PostgreSQL.**
- Data is relational (users → URLs, analytics aggregated per URL).
- Write volume (3,471/sec) is within a Postgres primary's capability.
- ACID semantics simplify expiry and deactivation logic.
- Schema is fixed and well-known; NoSQL flexibility not needed.

```sql
-- Core URL store
CREATE TABLE urls (
    short_code  CHAR(7)      NOT NULL,
    long_url    TEXT         NOT NULL,
    user_id     BIGINT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    is_active   BOOLEAN      NOT NULL DEFAULT true,
    CONSTRAINT pk_urls PRIMARY KEY (short_code)
);

-- Index: fast lookup of a user's own URLs
CREATE INDEX idx_urls_user_id
    ON urls (user_id)
    WHERE user_id IS NOT NULL;

-- Index: efficient expiry sweep (partial; only rows that have an expiry)
CREATE INDEX idx_urls_expires_at
    ON urls (expires_at)
    WHERE expires_at IS NOT NULL AND is_active = true;

-- Analytics: high-write table, separate from urls to avoid row contention
-- Partitioned by day for efficient pruning
CREATE TABLE clicks (
    id          BIGSERIAL    NOT NULL,
    short_code  CHAR(7)      NOT NULL,
    clicked_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    referrer    TEXT,
    country     CHAR(2)
) PARTITION BY RANGE (clicked_at);

CREATE TABLE clicks_2026_11 PARTITION OF clicks
    FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');
-- (automation via pg_partman or cron)

-- Aggregated stats (updated by analytics worker every 60 s)
CREATE TABLE url_stats (
    short_code     CHAR(7)  NOT NULL PRIMARY KEY REFERENCES urls(short_code),
    total_clicks   BIGINT   NOT NULL DEFAULT 0,
    last_updated   TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 5. High-Level Design

```
                   ┌──────────────────────┐
  Client ──HTTPS──►│    CDN Edge (Global)  │ 301 cached, TTL=1h
                   └──────────┬───────────┘
                              │ HTTPS (cache miss)
                              ▼
                   ┌──────────────────────┐
                   │    Load Balancer      │ SSL termination, health checks
                   └──────────┬───────────┘
                              │ HTTP
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        [API Service]   [API Service]   [API Service]   (stateless, auto-scaled)
              │
   ┌──────────┼────────────────────────────────┐
   │          │                                │
   ▼          ▼                                ▼
{Redis}  (Postgres Primary)──repl──►(Postgres Replica ×2)
 Cache        │ WAL
              ▼
        (WAL-E / pgBackup → S3)  ← continuous backup

Analytics path (async):
  [API Service] ──publish──► <Kafka: clicks topic (32 partitions)>
                                      │ consume
                               [Click Aggregator Workers]
                                      │ batch upsert every 60s
                              (Postgres url_stats / ClickHouse)

Expiry path (async):
  [Expiry Cron Worker] ─► SELECT expired codes ─► UPDATE is_active=false
                        ─► CDN purge API call per expired code
```

**Component responsibilities:**
- **CDN Edge:** serves 90%+ of redirects without hitting origin. Caches 302 responses.
- **Load Balancer:** round-robin + health checks; terminates TLS.
- **API Service:** stateless; all persistent state in Redis and Postgres.
- **Redis (LRU, 64 GB):** hot short_code → long_url mapping. Write-through on URL creation.
- **Postgres Primary:** single-writer; synchronous replication to one standby (RPO=0).
- **Postgres Replicas:** serve read traffic (stats queries, admin); asynchronous replication.
- **Kafka:** decouples click event ingestion from redirect latency.
- **Click Aggregator:** batches Kafka events; upserts into `url_stats` every 60 s.
- **Expiry Worker:** sweeps `urls` table every 5 min; invalidates CDN cache for expired codes.

---

## 6. Deep Dives

### 6.1 Short-Code Generation Strategy

Three approaches compared:

| Approach | Collision risk | Throughput | Enumerable | Verdict |
|---|---|---|---|---|
| MD5 hash + truncate to 7 chars | Yes — retry needed | High | Low | Rejected (retry complexity) |
| Pre-generated key pool service | None | High (pre-computed) | Low | Rejected (operational overhead) |
| DB auto-increment + base62 encode | None | High (single SERIAL) | Yes (mitigated) | **Chosen** |

**Chosen: Auto-increment ID + base62 encode with XOR scrambling.**

```python
import os
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_BASE = len(_ALPHABET)


@lru_cache(maxsize=1)
def _salt() -> int:
    raw = os.getenv("SHORT_CODE_SALT", "")
    if not raw:
        raise RuntimeError("SHORT_CODE_SALT must be set")
    return int.from_bytes(raw.encode()[:8], "big")


def encode_id(n: int, length: int = 7) -> str:
    """Encode a positive integer to a base62 short code of fixed length."""
    n = n ^ _salt()
    result: list[str] = []
    while n > 0:
        result.append(_ALPHABET[n % _BASE])
        n //= _BASE
    code = "".join(reversed(result)).zfill(length)
    logger.debug("encode_id", extra={"input_id": n, "code": code})
    return code
```

Capacity: 62^7 ≈ 3.5 trillion codes; 95 years at 100 M/day.
Enumeration: XOR with `_SALT` makes sequential IDs non-guessable without breaking uniqueness.

### 6.2 Redirect Performance

Target: p99 < 10 ms globally.

**Cache hit path (> 99% of requests):**
```
1. Client → CDN PoP nearest to user (< 5 ms for major cities)
   CDN check: does this short_code have a cached entry?
   Cache hit → CDN returns 302 directly. Total: 1–5 ms. ✓

2. CDN miss → API service (< 1 ms intra-DC network)
   API → Redis GET short_code (< 0.5 ms)
   Redis hit → API returns 302 + Location. Total: 3–8 ms. ✓
   CDN caches result for 1 h.
```

**Cache miss path (new codes or evicted entries):**
```
3. Redis miss → Postgres replica query (< 5 ms on SSD)
   API returns 302; writes to Redis (async, non-blocking).
   Total: ~10–15 ms (near the SLO boundary)

Mitigation: write-through on URL creation ensures Redis is populated
before any redirect can be requested. Cache miss only occurs if:
  a) Code was created > 24 h ago and was LRU-evicted (old, infrequently accessed codes)
  b) Redis restart (cold start)
In both cases, DB fallback at 15 ms is acceptable.
```

**Analytics on the redirect path:**
The click event must NOT block the redirect response. Approach: fire-and-forget Kafka publish with `acks=0` (no wait for broker confirmation). A missed click event is tolerable per NFRs; a blocked redirect is not.

```python
# Non-blocking analytics — publish and immediately return
producer.send("clicks", key=short_code.encode(), value=payload)
# Do not await delivery confirmation
return RedirectResponse(url=original_url, status_code=302)
```

---

## 7. Bottlenecks

| Component | Failure mode | Threshold | Mitigation |
|---|---|---|---|
| Redis single node | Memory exhaustion → all reads fall to DB | > 64 GB hot set | Redis Cluster (hash slots by short_code prefix); add nodes |
| Postgres primary | Write throughput ceiling | ~10 k writes/sec | Current load (3.5 k/sec) is well under limit; monitor write latency; if needed, shard by short_code range |
| CDN cache invalidation | Expired/deactivated URL still served from CDN | Any deactivation | Call CDN purge API synchronously on DELETE; set TTL = 1 h as backstop so stale window is bounded |
| Kafka consumer lag | Analytics lag > 60 s SLO | Click burst (e.g., viral URL) | Monitor consumer lag; auto-scale consumer group; partition by short_code hash (32 partitions → 32 consumers max) |
| Expiry worker missing codes | Expired URLs still accessible | Expiry cron fails | Add a secondary "lazy expiry" check in the API redirect path: if `expires_at < now()`, return 410 even if `is_active` hasn't been flipped yet |

---

## 8. Trade-offs

| Decision | Chosen | Alternative | Reason chosen |
|---|---|---|---|
| DB | Postgres (SQL) | DynamoDB / Cassandra | Relational data; modest write volume; ACID simplifies expiry and deactivation; team familiarity |
| Code generation | Auto-increment + base62 | Hash-truncate; key pool | Zero collision risk; no extra service; XOR mitigates enumeration |
| Cache | Shared Redis | In-process per-instance cache | Consistent view across all API instances; survives restarts; network cost (0.2 ms) within SLO |
| Redirect response | 302 (Found) | 301 (Moved Permanently) | 301 is cached permanently by browsers — if URL is later deactivated, browser ignores server response. 302 keeps server in control. Trade-off: 302 adds more origin hits (CDN still caches it) |
| Analytics write | Async Kafka + aggregator | Synchronous DB write | Synchronous write adds 5 ms+ to redirect path, risking p99 SLO breach. Async acceptable because NFR allows 60 s lag |
| Global distribution | CDN + single origin region | Multi-region active-active | Active-active requires global data synchronisation (complex, expensive). CDN covers 90%+ of requests at the edge. Remaining 10% (cache miss) tolerate 10–15 ms. Simpler to operate. |
| Expiry mechanism | DB sweep + CDN purge | Lazy delete only | Lazy delete leaves orphaned records forever; sweep reclaims storage and prevents zombie redirects |
