# Phase 7 · Week 05 · Day 4 — Distributed Rate Limiter + Notification System

> Objective: Design a distributed rate limiter and a multi-channel notification system.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-3]]

---

## Part A — Distributed Rate Limiter

## Concept

A rate limiter restricts how many requests a client can make in a time window. Without one, a single noisy client can starve every other client on your platform (or cost you a fortune in downstream API fees).

**Why it's hard in distributed systems**: with 20 API pods behind a load balancer, a simple in-process counter per pod underestimates usage — each pod only sees 1/20th of the traffic.

---

### A1 — Requirements

**Functional**
- Limit requests per user/IP/API key.
- Configurable limits: 100 req/min, 1 000 req/day, per endpoint.
- Return `429 Too Many Requests` with `Retry-After` header when exceeded.
- Allow burst (e.g. 10 req/s sustained, 20 req/s for 2 s).

**Non-functional**
- Decision latency < 5 ms P99 (inline on the request path).
- Accuracy: over-count or under-count by ≤ 5%.
- High availability: if the rate-limiter store goes down, fail open (allow traffic), not closed.
- Must work across N stateless API pods.

---

### A2 — Algorithms

#### Fixed Window Counter

```
Window: [00:00, 01:00)  Counter: 87/100
Window: [01:00, 02:00)  Counter: 0/100  ← resets at boundary
```

**Problem**: boundary burst. User sends 100 req at 00:59 and 100 req at 01:01 → 200 req in 2 seconds, both within their respective windows.

#### Sliding Window Log

Store a timestamp log per user. Count entries in the last 60 s.

**Problem**: O(N) storage per user (stores every timestamp).

#### Sliding Window Counter (approximate)

```
count = current_window_count + previous_window_count × (overlap fraction)
```

Example: current window is 40% through. Count = `current(30) + previous(70) × 0.60 = 30 + 42 = 72`.

Good approximation, O(1) space. **This is the recommended algorithm.**

#### Token Bucket

A bucket holds up to `capacity` tokens. Tokens refill at `rate` tokens/s. Each request consumes one token. If bucket is empty → reject.

- Allows bursting up to `capacity` requests.
- Natural for APIs that need burst tolerance.
- Redis implementation: store `(tokens, last_refill_time)`.

#### Leaky Bucket

Requests enter a FIFO queue (the bucket); they leak out at a fixed rate. Smooths traffic — no bursting. Used in network QoS, less common for API rate limiting.

---

### A3 — Redis-Based Sliding Window Implementation

```
Key:   rl:{user_id}:{window_start_minute}
Value: integer counter
TTL:   2 minutes

Algorithm per request:
  now_minute = floor(now / 60)
  prev_minute = now_minute - 1
  elapsed_fraction = (now % 60) / 60.0

  prev_count = GET rl:{uid}:{prev_minute}   (or 0)
  curr_count = INCR rl:{uid}:{now_minute}
  EXPIRE rl:{uid}:{now_minute} 120

  estimated = curr_count + prev_count * (1 - elapsed_fraction)
  if estimated > limit: return 429
  return 200
```

Atomic INCR + EXPIRE can be wrapped in a Lua script for atomicity.

---

### A4 — Architecture

```
Client → API Gateway
             │
             ├─ [Rate Limiter Middleware] (inline, per request)
             │       │
             │   Redis Cluster (sliding window counters)
             │       │
             │   if Redis unavailable → fail open (allow)
             │
             └─ [Upstream Service]

Rules Storage: separate config service or Redis hash
  Key: rl_rules:{user_id}  or  rl_rules:endpoint:{path}
  Value: { "limit": 100, "window_secs": 60 }
```

**Multi-tier limiting**:
- Tier 1: per-IP (block abusive IPs) at CDN/edge.
- Tier 2: per-API-key (quota enforcement) at API gateway.
- Tier 3: per-endpoint per-user (fine-grained) at service level.

---

### A5 — Trade-offs

| Algorithm | Space | Time | Burst | Accuracy |
|---|---|---|---|---|
| Fixed window | O(1) | O(1) | Yes (boundary issue) | ~±50% at boundary |
| Sliding log | O(N) | O(N) | No | Exact |
| Sliding counter | O(1) | O(1) | Approximate | ~±5% |
| Token bucket | O(1) | O(1) | Yes (controlled) | Exact |

**Recommendation**: Token Bucket for burst-tolerant APIs; Sliding Window Counter for strict per-minute quotas.

---

## Part B — Notification System

## Concept

A notification system routes events to users across multiple channels: push (APNs/FCM), email, SMS, in-app. The challenges: channel diversity, delivery guarantees, user preferences, and thundering-herd at scale.

---

### B1 — Requirements

**Functional**
- Send notifications via: mobile push, email, SMS, in-app.
- Users can configure preferences (opt out of SMS, quiet hours, etc.).
- Support templated notifications with dynamic data.
- Delivery tracking: sent / delivered / clicked.

**Non-functional**
- 10 M notifications/day across all channels.
- Delivery latency < 1 min for push/in-app; < 5 min for email/SMS.
- At-least-once delivery; idempotent consumers.
- No notification spam: respect user frequency caps.

---

### B2 — Architecture

```
Event Producers (services)
        │ publish to Kafka topic: notifications.raw
        ▼
[Notification Router]
  - reads user preferences from Preferences DB
  - decides: which channels? quiet hours? frequency cap?
  - emits to channel-specific Kafka topics:
      notifications.push
      notifications.email
      notifications.sms
      notifications.inapp
        │
  ┌─────┼──────────────┐
  ▼     ▼              ▼
[Push  [Email       [SMS
Worker] Worker]     Worker]
  │       │            │
APNs/  SendGrid/    Twilio/
FCM    SES          SNS

[In-App Worker] → write to notifications table → WebSocket push
```

### B3 — User Preference Engine

```python
# preferences schema (Postgres)
CREATE TABLE notification_preferences (
    user_id         BIGINT NOT NULL,
    channel         VARCHAR(10) NOT NULL,  -- push|email|sms|inapp
    event_type      VARCHAR(50) NOT NULL,  -- message|like|follow|promo
    enabled         BOOLEAN NOT NULL DEFAULT true,
    quiet_hours_start TIME,                -- e.g. 22:00
    quiet_hours_end   TIME,                -- e.g. 08:00
    PRIMARY KEY (user_id, channel, event_type)
);
```

### B4 — Rate / Frequency Cap

Even if a user hasn't opted out, send at most:
- 3 push notifications / hour per app.
- 1 marketing email / day.

Implement with same Redis sliding-window rate limiter from Part A:
```
Key: notif_cap:{user_id}:{channel}:{window}
```

### B5 — Retry and Idempotency

- Each notification gets a `notification_id` (UUID).
- Workers are idempotent: check if `notification_id` already delivered before sending.
- Exponential backoff on transient failures (APNs/FCM 5xx).
- Dead-letter queue after 3 retries → alert ops.

---

## Worked Example — Token Bucket Rate Limiter (Python)

```python
# src/rate_limiter/token_bucket.py
"""
Redis-backed token bucket rate limiter.
Uses a Lua script for atomic check-and-consume.
"""
import logging
import time
from dataclasses import dataclass

import redis

logger = logging.getLogger(__name__)

_LUA_SCRIPT = """
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate     = tonumber(ARGV[2])   -- tokens per second
local now      = tonumber(ARGV[3])   -- current time (float seconds)
local cost     = tonumber(ARGV[4])   -- tokens to consume

local data = redis.call("HMGET", key, "tokens", "last_refill")
local tokens     = tonumber(data[1]) or capacity
local last_refill = tonumber(data[2]) or now

-- Refill tokens since last_refill
local elapsed = now - last_refill
tokens = math.min(capacity, tokens + elapsed * rate)

if tokens >= cost then
    tokens = tokens - cost
    redis.call("HMSET", key, "tokens", tokens, "last_refill", now)
    redis.call("EXPIRE", key, 3600)
    return 1   -- allowed
else
    redis.call("HMSET", key, "tokens", tokens, "last_refill", now)
    redis.call("EXPIRE", key, 3600)
    return 0   -- denied
end
"""


@dataclass
class RateLimiterConfig:
    capacity: float  # max tokens (burst size)
    rate: float       # tokens refilled per second


class TokenBucketLimiter:
    def __init__(self, redis_client: redis.Redis, config: RateLimiterConfig) -> None:
        self._redis = redis_client
        self._config = config
        self._script = redis_client.register_script(_LUA_SCRIPT)

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """Return True if the request is allowed; False if rate-limited."""
        now = time.time()
        result = self._script(
            keys=[f"rl:{key}"],
            args=[self._config.capacity, self._config.rate, now, cost],
        )
        allowed = bool(result)
        if not allowed:
            logger.info("Rate limit exceeded for key=%s", key)
        return allowed
```

---

## Exercises

1. Implement the sliding window counter algorithm in Python using Redis. Include the Lua script for atomicity.
2. A rate limiter uses fixed window counters. Show with a concrete example how a user can double their effective rate at window boundaries.
3. Design the frequency cap system for notifications (≤ 3 push/hour). Which rate-limiting algorithm fits best, and why?
4. The notification system needs to support a "digest" mode: instead of sending 10 separate email notifications, batch them into one hourly email. Sketch the architecture change.
5. A push notification to APNs fails with error code 410 (device token no longer valid). What should the notification worker do, and why is this important?

(Solutions: ../solutions/week-05/day-4.md)

---

## Self-check
- [ ] I can explain four rate-limiting algorithms and their boundary conditions.
- [ ] I can implement a Redis token bucket using a Lua script.
- [ ] I can sketch a multi-channel notification pipeline with Kafka workers.
- [ ] I can describe how user preference checking and frequency caps fit into the notification flow.

---

## Supplementary
- Redis rate limiting patterns — redis.io/learn/rate-limiting
- APNs HTTP/2 provider API — developer.apple.com
- "Scaling Push Notifications" — Airbnb Engineering Blog
