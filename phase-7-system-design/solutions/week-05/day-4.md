# Solutions — Week 05 · Day 4 (Rate Limiter + Notifications Exercises)

---

## Exercise 1 — Sliding window counter in Python

```python
# src/rate_limiter/sliding_window.py
"""
Redis-backed sliding window counter rate limiter.
Uses a Lua script for atomic operations.
"""
import logging
import time

import redis

logger = logging.getLogger(__name__)

_LUA_SCRIPT = """
local curr_key  = KEYS[1]
local prev_key  = KEYS[2]
local limit     = tonumber(ARGV[1])
local now       = tonumber(ARGV[2])
local window    = tonumber(ARGV[3])   -- window size in seconds

local window_start = math.floor(now / window) * window
local elapsed_frac = (now - window_start) / window

local curr_count = tonumber(redis.call("GET", curr_key)) or 0
local prev_count = tonumber(redis.call("GET", prev_key)) or 0

local estimated = curr_count + prev_count * (1 - elapsed_frac)

if estimated >= limit then
    return 0  -- denied
end

-- Increment current window counter
local new_count = redis.call("INCR", curr_key)
if new_count == 1 then
    redis.call("EXPIRE", curr_key, window * 2)
end
return 1  -- allowed
"""


class SlidingWindowLimiter:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._script = redis_client.register_script(_LUA_SCRIPT)

    def allow(self, key: str, limit: int, window_secs: int = 60) -> bool:
        """Return True if request is allowed under the sliding window limit."""
        now = time.time()
        window_start = int(now // window_secs) * window_secs
        prev_window_start = window_start - window_secs

        curr_key = f"rl:sw:{key}:{window_start}"
        prev_key = f"rl:sw:{key}:{prev_window_start}"

        try:
            result = self._script(
                keys=[curr_key, prev_key],
                args=[limit, now, window_secs],
            )
            return bool(result)
        except redis.exceptions.ConnectionError:
            logger.warning("Redis unavailable; failing open for key=%s", key)
            return True
```

---

## Exercise 2 — Fixed window boundary burst example

**Setup**: limit = 100 req/min, fixed window aligned to clock minutes.

- 00:59:00 — 100 requests sent → window [00:00, 01:00) counter = 100. All allowed.
- 01:00:05 — 100 requests sent → window [01:00, 02:00) counter = 100. All allowed.

**Result**: 200 requests in 65 seconds — double the intended rate. A user who knows the window boundary can reliably double throughput.

**What eliminates this**: sliding window counter. At 01:00:05 (elapsed_frac = 5/60 ≈ 0.083):
```
estimated = 0 + 100 × (1 − 0.083) = 91.7
```
Only 8 more requests allowed (100 − 91.7 ≈ 8), not 100. Boundary burst is eliminated.

---

## Exercise 3 — Frequency cap for notifications

**Best algorithm**: sliding window counter (not token bucket).

**Reason**: the requirement is "≤ 3 push per hour" — this is a rate limit with a fixed window that should not allow bursting. Token bucket allows bursting (up to `capacity`), which could deliver 3 notifications in 1 second, then nothing for the rest of the hour — still uncomfortable for users. The sliding window counter ensures a smooth 3-per-hour regardless of when in the hour they arrive.

**Redis keys**:
```
rl:notif:{user_id}:push:{window_start_hour}   (int counter, TTL 2 hours)
```

Check before sending each push notification: `estimated < 3` → send; else skip.

---

## Exercise 4 — Digest mode architecture

**Problem**: instead of 10 separate email notifications, batch into one hourly digest.

**Architecture change**:

1. Notification Router, instead of emitting directly to `notifications.email`, emits to `notifications.digest.{user_id}` (or a shared topic filtered by user).
2. A **Digest Aggregator** service:
   - Consumes from the digest topic.
   - Accumulates notifications per user in a Redis list: `RPUSH digest_pending:{user_id} {notification_json}`.
   - Sets a Redis key `digest_scheduled:{user_id}` with TTL = time until next hour boundary.
3. A **Digest Scheduler** (cron every 5 min):
   - Finds users where `digest_pending:{user_id}` is non-empty AND `digest_scheduled:{user_id}` is expired.
   - Reads the list: `LRANGE digest_pending:{user_id} 0 -1`.
   - Renders a single template with all items.
   - Emits one consolidated email to `notifications.email`.
   - Deletes `digest_pending:{user_id}`.

**User preference**: add a `digest_mode: bool` column to `notification_preferences`. The Router checks this and routes accordingly.

---

## Exercise 5 — APNs 410 error (device token no longer valid)

**What the notification worker should do**:

1. On receiving a 410 response from APNs for device token `T`:
   - Delete `T` from the `device_tokens` table for that user.
   - Do NOT retry sending to this token — it is permanently invalid.
2. Log the event: `device_token_invalidated user_id={X} token={T[:8]}...` (truncated, no full token in logs).
3. If the user has other registered tokens (e.g. multiple devices), retry on those.
4. If no valid tokens remain, write to `pending_delivery` for in-app notification on next app open.

**Why this matters**: sending to invalid tokens wastes APNs quota, can result in APNs rate-limiting your app, and in extreme cases gets your push certificate flagged. Stale device tokens accumulate quickly on apps with active uninstalls. Cleaning them proactively keeps delivery rates high and operational costs low.
