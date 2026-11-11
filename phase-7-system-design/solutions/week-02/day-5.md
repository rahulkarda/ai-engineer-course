# Solutions — Day 5: CDN, Reverse Proxy & Edge Rate Limiting

## Exercise 1 — Sliding window rate limiter

```python
"""
sliding_window_limiter.py
"""

import asyncio
import math
import time
import pytest
from caching_layer import InMemoryCacheBackend
import json


class SlidingWindowRateLimiter:
    """
    Approximate sliding window using interpolation between two fixed windows.
    count_approx = prev_count × (window_remaining / window_size) + current_count
    """

    def __init__(self, backend: InMemoryCacheBackend, limit: int, window_seconds: float) -> None:
        self._backend = backend
        self._limit = limit
        self._window = window_seconds

    def _keys(self, client_id: str, now: float) -> tuple[str, str]:
        current_bucket = int(now // self._window)
        prev_bucket = current_bucket - 1
        return (
            f"sw:{client_id}:{current_bucket}",
            f"sw:{client_id}:{prev_bucket}",
        )

    async def check(self, client_id: str) -> tuple[bool, float]:
        now = time.monotonic()
        current_key, prev_key = self._keys(client_id, now)

        current_raw = await self._backend.get(current_key)
        prev_raw = await self._backend.get(prev_key)

        current_count = int(json.loads(current_raw)) if current_raw else 0
        prev_count = int(json.loads(prev_raw)) if prev_raw else 0

        window_elapsed = now % self._window
        window_remaining = self._window - window_elapsed
        approx_count = prev_count * (window_remaining / self._window) + current_count

        if approx_count >= self._limit:
            return False, approx_count

        # Increment current window
        new_count = current_count + 1
        ttl = math.ceil(self._window * 2)
        await self._backend.set(current_key, json.dumps(new_count), ttl)
        return True, approx_count + 1


@pytest.mark.asyncio
async def test_sliding_window_no_more_than_limit():
    backend = InMemoryCacheBackend()
    limiter = SlidingWindowRateLimiter(backend, limit=5, window_seconds=1.0)
    client = "test-client"

    allowed_count = 0
    for _ in range(10):
        ok, _ = await limiter.check(client)
        if ok:
            allowed_count += 1

    # Should allow exactly 5 in the first window
    assert allowed_count <= 5
```

---

## Exercise 2 — HTTP rate-limit headers + FastAPI middleware

```python
"""
rate_limit_middleware.py
"""

import time
from fastapi import FastAPI, Request, Response
from day5_rate_limiter import TokenBucketRateLimiter
import redis.asyncio as aioredis
import os

app = FastAPI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_limiter: TokenBucketRateLimiter | None = None


@app.on_event("startup")
async def startup():
    global _limiter
    redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    _limiter = TokenBucketRateLimiter(redis, capacity=100.0, rate=10.0)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_id = request.client.host if request.client else "unknown"
    assert _limiter is not None

    result = await _limiter.check(client_id)
    reset_ts = int(time.time()) + int(result.retry_after_ms / 1000) + 1

    if not result.allowed:
        return Response(
            content="Rate limit exceeded",
            status_code=429,
            headers={
                "RateLimit-Limit": "100",
                "RateLimit-Remaining": "0",
                "RateLimit-Reset": str(reset_ts),
                "Retry-After": str(result.retry_after_ms // 1000 + 1),
            },
        )

    response = await call_next(request)
    response.headers["RateLimit-Limit"] = "100"
    response.headers["RateLimit-Remaining"] = str(int(result.tokens_remaining))
    response.headers["RateLimit-Reset"] = str(int(time.time()) + 60)
    return response
```

---

## Exercise 3 — Cache-Control audit

| Resource | Header | Rationale |
|---|---|---|
| User avatar `GET /avatars/user-42.jpg` | `Cache-Control: public, max-age=86400, stale-while-revalidate=3600` | Avatars change rarely; public CDN-cacheable; stale-while-revalidate reduces perceived latency when avatar updates |
| Homepage HTML `GET /` | `Cache-Control: no-cache` | HTML may contain user-specific rendering or A/B flags; validate freshness with ETag on every request, but serve from cache when ETag matches (304) |
| `/api/v1/me` (user profile) | `Cache-Control: private, no-store` | Contains PII — must never be stored at a CDN or shared proxy |
| Versioned JS `main.a3f8b2.js` | `Cache-Control: public, max-age=31536000, immutable` | Hash in filename guarantees content never changes for this URL; `immutable` tells browsers not to revalidate even when user presses refresh |

---

## Exercise 4 — CDN purge on deploy

```bash
#!/usr/bin/env bash
# cdn_purge.sh  — run after deploy; requires CF_TOKEN and CF_ZONE_ID env vars
# Usage: ./cdn_purge.sh changed_files.json

set -euo pipefail

CHANGED_FILES_JSON="${1:?Usage: $0 changed_files.json}"
BASE_URL="${CDN_BASE_URL:?Set CDN_BASE_URL e.g. https://assets.example.com}"
CF_API="https://api.cloudflare.com/client/v4/zones/${CF_ZONE_ID:?}/purge_cache"

# Build the list of full URLs from the changed-files manifest
URLS=$(python3 -c "
import json, sys
with open('${CHANGED_FILES_JSON}') as f:
    paths = json.load(f)['changed']
urls = ['${BASE_URL}' + p for p in paths]
print(json.dumps({'files': urls}))
")

echo "Purging ${CHANGED_FILES_JSON}..."
echo "${URLS}" | python3 -c "import sys; print(sys.stdin.read())"  # log what we're purging

curl -s -X POST "${CF_API}" \
  -H "Authorization: Bearer ${CF_TOKEN:?}" \
  -H "Content-Type: application/json" \
  -d "${URLS}" | python3 -m json.tool

echo "CDN purge complete."
```

`changed_files.json` format (emitted by deploy step):
```json
{
  "changed": ["/assets/main.js", "/assets/vendor.js", "/index.html"]
}
```

This avoids purging the entire cache (expensive, disrupts performance) and only invalidates files that actually changed.
