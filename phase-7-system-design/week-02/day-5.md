# Phase 7 · Week 02 · Day 5 — CDN, Reverse Proxy & Edge Rate Limiting

> Objective: Understand CDN architecture, configure a reverse proxy, and implement rate limiting at the edge.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Content Delivery Network (CDN)

A CDN is a globally distributed network of edge servers that cache and serve content from locations geographically close to end users.

```
User (Tokyo) ──────► CDN PoP (Tokyo)  ◄── cache hit → sub-20ms response
                          │ cache miss
                          ▼
                   Origin server (US-East)  ← only on misses
```

#### How CDN caching works

1. User requests `https://cdn.example.com/assets/app.js`.
2. DNS resolves to the nearest CDN **Point of Presence (PoP)**.
3. PoP checks its edge cache. **Hit** → serves immediately. **Miss** → fetches from origin, caches with the `Cache-Control` TTL from the origin response, then serves.
4. Subsequent users near that PoP get the cached copy.

#### What CDNs cache

| Content type | Cache-Control example | Notes |
|---|---|---|
| Static assets (JS, CSS, images) | `max-age=31536000, immutable` | Long TTL; bust with content-hash filename |
| HTML pages | `no-cache` or `max-age=60` | Short TTL or validate with ETag |
| API responses | `max-age=10, stale-while-revalidate=30` | Use carefully; user-specific data must not be cached |
| User-specific data | `Cache-Control: private, no-store` | Must never be cached at CDN |

#### Cache-Control directives cheat-sheet

- `max-age=N` — cache for N seconds.
- `s-maxage=N` — CDN-specific max-age (overrides max-age for shared caches).
- `no-cache` — must revalidate with origin before serving (uses ETag/Last-Modified).
- `no-store` — never cache anywhere.
- `immutable` — tell browser the file will never change (no revalidation for max-age period).
- `stale-while-revalidate=N` — serve stale while refreshing in background.

#### CDN cache invalidation

CDN TTLs can be minutes to days. Invalidating early requires a **purge API**:

```bash
# Cloudflare example
curl -X POST "https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" \
  -H "Authorization: Bearer $CF_TOKEN" \
  -d '{"files":["https://example.com/assets/app.js"]}'
```

For long-lived static assets, use **cache-busting**: include a content hash in the filename (`app.a3f8b2.js`). No invalidation needed — the URL itself changes.

---

### Reverse Proxy

A **reverse proxy** sits in front of application servers and acts as a single entry point:

```
Internet → [Reverse Proxy: NGINX/Caddy/Envoy] → [App servers]
```

Responsibilities:
- TLS termination (HTTPS → HTTP internally)
- Load balancing (proxy_pass to upstream pool)
- Request/response header rewriting
- Compression (gzip, Brotli)
- Static file serving
- Access logging
- Rate limiting
- Circuit breaking

#### NGINX as reverse proxy (conceptual config)

```nginx
# nginx.conf excerpt — illustrative, not runnable without NGINX installed
upstream app_servers {
    least_conn;
    server 127.0.0.1:8001 weight=2;
    server 127.0.0.1:8002 weight=1;
    server 127.0.0.1:8003 backup;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate     /etc/ssl/certs/example.crt;
    ssl_certificate_key /etc/ssl/private/example.key;

    location /api/ {
        proxy_pass http://app_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";          # keepalive
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 5s;
        proxy_read_timeout 30s;
    }

    location /static/ {
        root /var/www;
        add_header Cache-Control "max-age=31536000, immutable";
    }
}
```

---

### Rate Limiting at the Edge

Rate limiting caps how many requests a client can make in a time window. Doing it at the edge (reverse proxy or CDN) is far cheaper than application-layer limiting: requests are dropped before reaching app servers.

#### Algorithms

**Fixed window counter**  
Count requests in a fixed time window (e.g., 0–59s, 60–119s).  
Problem: a burst spanning the window boundary doubles the effective rate.

**Sliding window log**  
Store a timestamp log per client. Count timestamps in the last N seconds.  
Accurate, but memory-intensive at high volume.

**Sliding window counter (approximate)**  
Interpolate between the previous and current fixed windows:
```
count = prev_window_count × (window_remaining / window_size) + current_window_count
```
Memory-efficient and accurate enough for most uses.

**Token bucket**  
A bucket holds up to `capacity` tokens, refilled at `rate` tokens/second. Each request consumes one token. Allows controlled bursting.

**Leaky bucket**  
Requests enter a queue (the bucket) that drains at a fixed rate. Excess overflows (dropped). Enforces a strict output rate.

#### Redis-based sliding window counter

```python
# See worked example below
```

## How it works internally

### CDN PoP selection (anycast)

CDNs use **anycast routing**: multiple PoPs announce the same IP via BGP. The Internet's routing tables naturally direct traffic to the topologically closest PoP. No DNS-based geo-routing needed for UDP/TCP.

### Token bucket in Redis

```
KEYS[1] = bucket key (e.g., "ratelimit:user:42")
ARGV[1] = capacity
ARGV[2] = refill_rate (tokens/sec)
ARGV[3] = now (unix timestamp float)
ARGV[4] = cost (tokens this request consumes)
```

A Lua script atomically reads the bucket state, computes how many tokens have refilled since the last request, caps at capacity, and either consumes a token (allow) or returns 0 (deny). Lua in Redis executes atomically — no race conditions.

## Worked example

```python
"""
day5_rate_limiter.py
Token-bucket rate limiter implemented in Redis via a Lua script.
"""

import asyncio
import logging
import os
import time
from typing import NamedTuple

import redis.asyncio as aioredis

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Lua script: atomic token-bucket check-and-consume
# Returns [allowed (1/0), tokens_remaining (float), retry_after_ms (int)]
_TOKEN_BUCKET_LUA = """
local key        = KEYS[1]
local capacity   = tonumber(ARGV[1])
local rate       = tonumber(ARGV[2])   -- tokens per second
local now        = tonumber(ARGV[3])
local cost       = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "last_refill")
local tokens     = tonumber(data[1]) or capacity
local last       = tonumber(data[2]) or now

-- Refill tokens since last request
local elapsed    = math.max(0, now - last)
tokens           = math.min(capacity, tokens + elapsed * rate)

local allowed    = 0
local retry_ms   = 0

if tokens >= cost then
    tokens       = tokens - cost
    allowed      = 1
else
    retry_ms     = math.ceil((cost - tokens) / rate * 1000)
end

redis.call("HMSET", key, "tokens", tokens, "last_refill", now)
redis.call("PEXPIRE", key, math.ceil(capacity / rate * 1000) + 1000)
return {allowed, string.format("%.4f", tokens), retry_ms}
"""


class RateLimitResult(NamedTuple):
    allowed: bool
    tokens_remaining: float
    retry_after_ms: int


class TokenBucketRateLimiter:
    def __init__(
        self,
        redis: aioredis.Redis,
        capacity: float = 10.0,
        rate: float = 2.0,  # tokens/second
    ) -> None:
        self._redis = redis
        self._capacity = capacity
        self._rate = rate
        self._script: aioredis.client.Script | None = None

    async def _get_script(self) -> aioredis.client.Script:
        if self._script is None:
            self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)
        return self._script

    def _key(self, client_id: str) -> str:
        return f"ratelimit:token_bucket:{client_id}"

    async def check(self, client_id: str, cost: float = 1.0) -> RateLimitResult:
        script = await self._get_script()
        now = time.time()
        result = await script(
            keys=[self._key(client_id)],
            args=[self._capacity, self._rate, now, cost],
        )
        allowed = bool(int(result[0]))
        tokens_remaining = float(result[1])
        retry_after_ms = int(result[2])
        return RateLimitResult(allowed, tokens_remaining, retry_after_ms)


# ---------------------------------------------------------------------------
# Fixed-window counter (simpler, for comparison)
# ---------------------------------------------------------------------------

class FixedWindowRateLimiter:
    def __init__(self, redis: aioredis.Redis, limit: int = 5, window_seconds: int = 10) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_seconds

    def _key(self, client_id: str) -> str:
        bucket = int(time.time()) // self._window
        return f"ratelimit:fixed:{client_id}:{bucket}"

    async def check(self, client_id: str) -> RateLimitResult:
        key = self._key(client_id)
        count = await self._redis.incr(key)
        if count == 1:
            await self._redis.expire(key, self._window)
        allowed = count <= self._limit
        remaining = max(0, self._limit - count)
        return RateLimitResult(allowed, float(remaining), 0)


async def demo_token_bucket(redis: aioredis.Redis) -> None:
    limiter = TokenBucketRateLimiter(redis, capacity=5.0, rate=1.0)
    log.info("=== Token Bucket: capacity=5, rate=1/s ===")
    client = "user:alice"

    # Burst: consume 5 tokens rapidly
    for i in range(7):
        result = await limiter.check(client)
        status = "ALLOW" if result.allowed else f"DENY (retry in {result.retry_after_ms}ms)"
        log.info("req-%d: %s | tokens_remaining=%.2f", i + 1, status, result.tokens_remaining)
        await asyncio.sleep(0.1)

    # Wait for refill
    log.info("Waiting 3 seconds for refill...")
    await asyncio.sleep(3)
    result = await limiter.check(client)
    log.info("After refill: %s | tokens=%.2f", "ALLOW" if result.allowed else "DENY", result.tokens_remaining)


async def main() -> None:
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await demo_token_bucket(redis)
    await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

## Exercises

1. **Sliding window**: implement a `SlidingWindowRateLimiter` using the interpolation formula `prev × (remaining/size) + current`. Write a test that verifies no more than `limit` requests are allowed in any sliding window of size `window_seconds`, even across a window boundary.

2. **HTTP headers**: extend `TokenBucketRateLimiter` to return a dict of headers conforming to the draft IETF rate-limit standard: `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset`. Write a FastAPI middleware that applies the limiter and attaches these headers.

3. **Cache-Control audit**: for each of the following resources, write the correct `Cache-Control` header and explain the choice: (a) user avatar image, (b) homepage HTML, (c) `/api/v1/me` (user profile), (d) versioned JS bundle `main.a3f8b2.js`.

4. **CDN purge on deploy**: design a CI/CD pipeline step (pseudo-code or shell script) that, after deploying new static assets to origin, purges only the changed files from a Cloudflare CDN zone using the API. Assume the deploy emits a JSON file listing changed paths.

(Solutions: ../solutions/week-02/day-5.md)

## Self-check
- [ ] I can explain how CDN anycast routing works
- [ ] I understand `Cache-Control` directives and when to use each
- [ ] I can implement token-bucket rate limiting using a Redis Lua script
- [ ] I can configure a reverse proxy to terminate TLS, load balance, and serve static files

## Supplementary links
- [Cloudflare: How CDNs work](https://www.cloudflare.com/learning/cdn/what-is-a-cdn/)
- [MDN: Cache-Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control)
- [IETF: RateLimit header fields](https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/)
- [Redis: Lua scripting](https://redis.io/docs/manual/programmability/lua-api/)
