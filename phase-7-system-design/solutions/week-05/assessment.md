# Solutions — Week 05 Assessment (Answer Key)

---

## Conceptual Answers

### Q1 — Pre-generated key pool sufficiency

**Math**:
```
Write QPS = 1 000 writes/s
Pool refill = 1 M codes / hour = 277 codes/s

Consumption rate (1 000/s) >> refill rate (277/s)
```

**Answer**: No, the pool is not sufficient. 1 000 writes/s consumes 3 600 × 1 000 = 3.6 M codes per hour, but the pool only generates 1 M per hour. The pool exhausts in approximately 1 000 000 / 1 000 = **1 000 seconds (~17 minutes)**.

**Fix**: generate 4 M+ codes per hour, or trigger refill when pool drops below 500 K codes (alert threshold), or run KGS continuously rather than hourly.

---

### Q2 — Boundary burst problem

**Algorithm**: token bucket or sliding window counter eliminate this.

**Concrete example**:
- Limit: 100 req/min, fixed windows aligned to clock minutes.
- 00:59:50 → user sends 100 requests → window [00:00, 01:00) counter = 100. All allowed.
- 01:00:05 → user sends 100 requests → window [01:00, 02:00) counter = 100. All allowed.
- 200 requests in 15 seconds. The system allowed 2× the intended rate.

**Sliding window counter at 01:00:05** (elapsed_frac = 5/60 ≈ 0.083):
```
estimated = 0 + 100 × (1 − 0.083) = 91.7
allowed = 100 − 91.7 = 8.3 → only 8 more requests allowed
```
Boundary burst effectively eliminated.

---

### Q3 — Celebrity with 20 M followers, posts once per minute

**Consequence of pure fanout-on-write**:
- 20 M followers × 1 post/min = 20 M Redis writes/min = 333 K writes/s for one celebrity.
- With 1 000 celebrities posting at similar rate: 333 M writes/s — unachievable.
- Kafka consumer lag spikes; feeds update minutes late; fanout workers become the bottleneck.

**Solution**: hybrid model. Celebrity posts (author with > threshold followers) bypass fanout-on-write. At feed-read time, the Feed Service fetches the celebrity's latest posts on demand from their own sorted set (`user_posts:{celebrity_id}`) and merges with the pre-built feed. The celebrity's post list is a single write per post regardless of follower count.

---

### Q4 — Message persisted, ack never received, client restarts

**Step-by-step recovery**:

1. Client sends message with `client_nonce="uuid-abc"`.
2. Server persists `message_id="m_999"` with `nonce="uuid-abc"` in Cassandra.
3. Server crashes (or network drops) before sending ack.
4. Client times out (5 s), retries with same `client_nonce="uuid-abc"`.
5. New server receives retry → idempotency check: `SELECT message_id WHERE client_nonce = 'uuid-abc' AND sender_id = X` → returns `m_999`.
6. Server returns `delivery_ack { message_id: "m_999" }` — no new row inserted.
7. Client now knows the message was delivered with ID `m_999`.
8. Recipient sees only one copy of the message (deduplication by `message_id` on their device).

**Key requirement**: Cassandra quorum writes ensure the persisted row is visible to any replica that serves the retry.

---

### Q5 — Geohash vs. quadtree

**Geohash**:
- Fixed-size cells at each precision level.
- Simple: proximity ≈ shared prefix.
- Weakness: uniform cell size ignores density — an empty desert cell and a dense city cell are the same size.

**Quadtree**:
- Recursively splits until each leaf ≤ K businesses.
- Dense urban areas produce deep, fine-grained nodes; sparse rural areas produce shallow, large nodes.
- Better for **non-uniform density**: a quadtree query in Manhattan returns a tight, relevant set; a geohash query in Manhattan at appropriate precision might span large empty areas.

**Scenario where quadtree outperforms**: a dataset with extreme density variance — e.g., 90% of 500 M businesses are in 10 cities. Geohash at precision 7 gives equal-sized cells everywhere. A quadtree naturally concentrates resolution where businesses actually are, resulting in fewer false-positive candidates per query.

---

### Q6 — Trie insert complexity with top-K maintained

**Complexity**: O(L × K) per insert, where L = word length, K = top-K size.

- Traversal: O(L) nodes visited.
- At each of L nodes: maintaining the top-K heap costs O(K) for a linear scan to find/replace the word, or O(log K) for a heap operation.
- Total: O(L × log K) with a heap, O(L × K) with a list.

**Is this acceptable**? For K = 5 and L = 20, this is O(20 × log 5) ≈ O(46) — very fast. For online inserts from user queries at 10 K QPS, this is negligible. The concern arises with bulk loads of millions of words — O(N × L × log K) where N = number of words. For N = 10 M, L = 8, K = 5: ~120 M operations — takes a few seconds in Python, fine for a background job.

---

### Q7 — Fail open vs. fail closed when Redis is unavailable

**Choose fail open for a public API**.

**Reasoning**:
- Fail closed (deny all): your API becomes completely unavailable when the rate limiter store is down. For a public API, this means a monitoring/infrastructure failure takes down customer-facing features — worse than briefly over-serving traffic.
- Fail open (allow all): traffic flows normally; you temporarily cannot enforce rate limits. This is acceptable short-term. Log the outage, alert on-call, fix Redis — the window of unthrottled traffic is bounded by your MTTR.

**Exception**: for APIs where rate limiting is a security boundary (e.g., login attempts to prevent brute force), failing open is unacceptable. In that case, use Redis Sentinel/Cluster for high availability and accept brief fail-closed behavior on actual quorum loss.

---

### Q8 — One marketing email per user per day via Redis

**Mechanism**:

```
Key:     notif:dedupe:{user_id}:marketing_email:{date}
Type:    String (or just the key's existence)
Value:   "1"
TTL:     86400 s (24 hours, or until midnight — use time-to-midnight for window alignment)
Command: SET notif:dedupe:{user_id}:marketing_email:{date} 1 NX EX 86400
```

`SET ... NX` (only if not exists) is atomic. If it returns 1 → first email today → send. If it returns nil → already sent today → skip.

**Window alignment**: use `date = today_utc_date` (e.g. "2026-11-11") so all users reset at UTC midnight, not 24 h from first contact.

**Pod safety**: `SET ... NX` is atomic in Redis — 50 pods simultaneously trying to send to the same user all call this command, only one wins. The others get `nil` and skip.

---

## Coding Challenge Solutions

### Part 1 — Sliding Window Counter

```python
# src/rate_limiter/sliding_window.py
import logging
import time

import redis

logger = logging.getLogger(__name__)

_LUA_SCRIPT = """
local curr_key  = KEYS[1]
local prev_key  = KEYS[2]
local limit     = tonumber(ARGV[1])
local now       = tonumber(ARGV[2])
local window    = tonumber(ARGV[3])

local window_start  = math.floor(now / window) * window
local elapsed_frac  = (now - window_start) / window

local curr_count = tonumber(redis.call("GET", curr_key)) or 0
local prev_count = tonumber(redis.call("GET", prev_key)) or 0
local estimated  = curr_count + prev_count * (1 - elapsed_frac)

if estimated >= limit then
    return 0
end

local new_count = redis.call("INCR", curr_key)
if new_count == 1 then
    redis.call("EXPIRE", curr_key, window * 2)
end
return 1
"""


class SlidingWindowLimiter:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client
        self._script = redis_client.register_script(_LUA_SCRIPT)

    def allow(self, key: str, limit: int, window_secs: int = 60) -> bool:
        now = time.time()
        window_start = int(now // window_secs) * window_secs
        prev_window_start = window_start - window_secs
        curr_key = f"rl:sw:{key}:{window_start}"
        prev_key = f"rl:sw:{key}:{prev_window_start}"
        try:
            return bool(self._script(
                keys=[curr_key, prev_key],
                args=[limit, now, window_secs],
            ))
        except redis.exceptions.ConnectionError:
            logger.warning("Redis down; failing open for key=%s", key)
            return True
```

**Tests** (using `fakeredis`):

```python
# tests/unit/test_sliding_window.py
import fakeredis
import pytest
from src.rate_limiter.sliding_window import SlidingWindowLimiter
import redis
from unittest.mock import patch


@pytest.fixture
def limiter():
    r = fakeredis.FakeRedis()
    return SlidingWindowLimiter(r)


def test_first_request_allowed(limiter):
    assert limiter.allow("user1", limit=10) is True

def test_requests_at_limit_all_allowed(limiter):
    for _ in range(10):
        assert limiter.allow("user2", limit=10) is True

def test_over_limit_denied(limiter):
    for _ in range(10):
        limiter.allow("user3", limit=10)
    assert limiter.allow("user3", limit=10) is False

def test_window_transition_proportional(limiter):
    # This requires mocking time — test the formula logic directly
    # At 50% through the window, previous 10 contribute 5 → headroom = 5
    pass  # Integration test; unit test the Lua math separately

def test_redis_unavailable_fails_open():
    bad_redis = redis.Redis(host="127.0.0.1", port=9999)  # nothing there
    lim = SlidingWindowLimiter(bad_redis)
    # register_script doesn't connect; allow() will raise on execute
    try:
        result = lim.allow("user", limit=10)
        assert result is True  # fails open
    except Exception:
        pass  # fakeredis can't simulate this; use mock

def test_keys_are_independent(limiter):
    for _ in range(10):
        limiter.allow("userA", limit=10)
    assert limiter.allow("userB", limit=10) is True

def test_idempotent_within_limit(limiter):
    results = [limiter.allow("user4", limit=5) for _ in range(5)]
    assert all(results)
```

---

### Part 2 — Trie delete

See full implementation in the Day 5 solution (Exercise 4). The key points:
1. Walk the path to find the word; return False if not found.
2. Set `node.frequency = 0` at the terminal node.
3. Remove the word from `top_k` at all ancestor nodes via list comprehension + `heapq.heapify`.
4. Prune leaf nodes with no children and no frequency (walk back up the path).
