# Solutions — Week 05 Project (Answer Key + Reference Implementations)

---

## Design Document A — Chat + Messaging System (Model Answer)

### 1. Requirements

**Functional**:
- 1:1 text chat between users.
- Send and receive messages in real time (WebSocket).
- Message delivery receipts (delivered, read).
- Message history (last 30 days, paginated).
- Push notifications for offline users.

**Non-functional**:
- 100 M DAU; 50 M concurrent WebSocket connections.
- Delivery latency P99 < 100 ms (online recipient).
- At-least-once delivery; client-side dedup by `message_id`.
- 5 nines availability for message storage (Cassandra replication factor 3).
- Eventual consistency for presence (lag ≤ 30 s acceptable).

### 2. Estimation

```
Messages/day   : 100 M users × 20 msg/day = 2 B msgs/day
Write QPS      : 2 B / 86 400 ≈ 23 000 writes/s (peak 3×)
Storage/day    : 23 000 × 200 B avg message = ~4.6 MB/s = ~400 GB/day
Annual storage : ~140 TB (Cassandra, 3× replication = 420 TB raw)
Cache (hot)    : last 100 msgs × 50 M active convos × 200 B = 1 TB → shard across Redis Cluster
```

### 3. API Design

```
WebSocket: wss://chat.example.com/ws  (auth token on connect)

Client → Server:
  { "type": "send_message",   "conversation_id": "c1", "body": "hi", "client_nonce": "uuid1" }
  { "type": "ack_received",   "message_id": "m_123" }
  { "type": "sync",           "last_seen_ids": { "c1": "m_120", "c2": "m_90" } }

Server → Client:
  { "type": "new_message",    "message_id": "m_123", "sender_id": "u1", "body": "hi", "created_at": "..." }
  { "type": "delivery_ack",   "client_nonce": "uuid1", "message_id": "m_123" }

REST:
  GET /api/v1/conversations/{id}/messages?before=<msg_id>&limit=50
```

### 4. Data Model

```sql
-- Cassandra (messages)
CREATE TABLE messages (
    conversation_id BIGINT,
    message_id      BIGINT,      -- Snowflake; partition key + cluster key
    sender_id       BIGINT,
    body            TEXT,
    client_nonce    UUID,
    PRIMARY KEY (conversation_id, message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);

-- Postgres (metadata)
CREATE TABLE conversations (
    conversation_id BIGINT PRIMARY KEY,
    member_a BIGINT, member_b BIGINT,
    created_at TIMESTAMPTZ
);

-- Redis: ws_pod:{user_id} → pod_id (TTL = connection lifetime)
-- Redis: presence:{user_id} → "online" (TTL = 30 s)
-- Redis: msg_cache:{conversation_id} → list of last 100 messages
```

### 5. Architecture

```
Client ──WebSocket──► Gateway Pod (1 of N)
                            │ gRPC
                      Message Service
                            │
                     ┌──────┴──────┐
                  Kafka          Cassandra
                     │
               Fanout Worker
                     │
              ┌──────┴──────┐
           Gateway Pod    Push Notification
           (recipient)    Service (APNs/FCM)
```

### 6. Deep Dives

**A — At-least-once delivery + dedup**: see Day 3, Section 6a. `client_nonce` as idempotency key. Client retries on no ack within 5 s. Server checks nonce before inserting.

**B — Offline sync**: client sends `sync` frame on reconnect with last-seen message IDs per conversation. Server queries Cassandra for `message_id > last_seen` for each conversation. Streams batches of 50.

### 7. Bottlenecks

| Bottleneck | Threshold | Mitigation |
|---|---|---|
| WebSocket connection memory | > 500 K conns/pod | Auto-scale pods; 10 KB/conn × 500 K = 5 GB RAM |
| Cassandra hot partition (busy conversation) | > 100 K msg/s per conversation | Partition splitting not needed for chat; limit group size to 500 |
| Fanout worker queue lag | > 1 M events/s | Scale Kafka partitions + fanout worker replicas |

### 8. Trade-offs

| Decision | Gain | Cost |
|---|---|---|
| WebSocket over polling | True push, < 100 ms | Stateful servers, harder LB |
| Cassandra for messages | Write scalability | No JOIN, eventual consistency |
| At-least-once delivery | No message loss | Client must dedup |
| Separate push service | Clean separation | APNs/FCM latency adds 200–500 ms |

**Real-world reference**: Discord uses a similar Cassandra partition model, documented in their "How Discord Stores Billions of Messages" blog post (2017, updated 2023).

---

## Design Document B — Distributed Rate Limiter (Model Answer)

### 1. Requirements

**Functional**: per-user (100 req/min), per-IP (1 000 req/min), per-endpoint overrides.

**Non-functional**: decision < 2 ms; ±5% accuracy; fail open on Redis failure.

### 2. Estimation

```
30 pods × 5 000 req/s/pod = 150 000 req/s total
Redis operations: 150 000 req/s × 1 Lua call/req = 150 000 ops/s
Redis Cluster (3 nodes) handles 1 M ops/s → comfortable headroom
Memory: 100 M users × 2 keys (curr + prev window) × 50 B = 10 GB → fine
```

### 3. Interface

```python
# Library call (not a sidecar — avoids network hop for < 2 ms budget)
limiter.allow(user_key="user_123", endpoint="/api/v1/search") -> bool
```

429 response: `{ "detail": "rate limit exceeded" }` with `Retry-After: 60` header.

### 4. Data Model (Redis)

```
Key:   rl:user:{user_id}:{window_start_minute}   Type: Integer  TTL: 120 s
Key:   rl:ip:{ip_hash}:{window_start_minute}     Type: Integer  TTL: 120 s
Key:   rl:rules:{endpoint}                       Type: Hash     TTL: none
```

### 5. Architecture

```
API Pod
  └── RateLimitMiddleware (in-process)
        └── TokenBucketLimiter.allow(key)
              └── Redis Lua script (atomic read-modify-write)
                    └── Redis Cluster (3 shards, 3 replicas each)

Rules config: stored in Redis hashes; hot-reloaded every 60 s
```

### 6. Deep Dives

**A — Lua atomicity**: without Lua, `GET tokens; compute; SET tokens` has a TOCTOU gap — two concurrent requests both read the same token count and both decrement, causing double-spend. Lua executes atomically in Redis: no other command runs between the read and write.

**B — Fail open**: Redis down → `ConnectionError` → catch and return `True` (allow). Log warning + increment `rate_limiter_redis_down_count` metric → alert on-call. Brief unthrottled traffic is preferable to a full API outage. Exception: auth endpoints — those should fail closed.

### 7. Bottlenecks

| Bottleneck | Threshold | Mitigation |
|---|---|---|
| Redis single node | > 100 K ops/s | Redis Cluster with consistent hashing |
| Rule hot reload (stale rules for 60 s) | Config changes take effect slowly | Pub/sub channel for immediate rule invalidation |
| Memory growth (100 M users) | > Redis RAM | TTL cleanup; only create keys for active users |

### 8. Trade-offs

| Decision | Gain | Cost |
|---|---|---|
| Token bucket over sliding window | Burst tolerance; more natural for APIs | Slightly more complex Lua; harder to reason about exact per-minute limits |
| In-process library over sidecar | < 1 ms overhead, no network hop | Must ship library update to change algorithm |
| Fail open | API stays up during Redis outage | Brief quota violations possible |

---

## Reference Implementation — Rate Limiter Tests

```python
# tests/unit/test_token_bucket.py
import threading
import fakeredis
import pytest
from unittest.mock import patch
import redis

from src.rate_limiter.token_bucket import TokenBucketLimiter, RateLimiterConfig


@pytest.fixture
def limiter():
    r = fakeredis.FakeRedis()
    config = RateLimiterConfig(capacity=10.0, rate=1.0)
    return TokenBucketLimiter(r, config)


def test_first_request_allowed(limiter):
    assert limiter.allow("user1") is True


def test_capacity_requests_all_allowed(limiter):
    for _ in range(10):
        assert limiter.allow("user_cap") is True


def test_over_capacity_denied(limiter):
    for _ in range(10):
        limiter.allow("user_over")
    assert limiter.allow("user_over") is False


def test_refill_after_wait(limiter):
    # Exhaust bucket
    for _ in range(10):
        limiter.allow("user_refill")
    # Mock time to advance 1 second
    import time
    with patch("src.rate_limiter.token_bucket.time.time") as mock_time:
        mock_time.return_value = time.time() + 1.1
        assert limiter.allow("user_refill") is True


def test_fractional_cost(limiter):
    # cost=0.5 → should allow 20 calls (10 tokens / 0.5)
    results = [limiter.allow("user_frac", cost=0.5) for _ in range(20)]
    assert all(results)
    assert limiter.allow("user_frac", cost=0.5) is False


def test_keys_are_isolated(limiter):
    for _ in range(10):
        limiter.allow("user_a")
    assert limiter.allow("user_b") is True


def test_concurrent_no_double_spend():
    r = fakeredis.FakeRedis()
    config = RateLimiterConfig(capacity=5.0, rate=0.0)  # no refill
    lim = TokenBucketLimiter(r, config)
    results = []

    def call():
        results.append(lim.allow("concurrent_user"))

    threads = [threading.Thread(target=call) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    allowed = sum(1 for r in results if r)
    assert allowed == 5  # exactly 5 allowed, no double-spend


def test_redis_down_fails_open():
    bad = redis.Redis(host="127.0.0.1", port=19999, socket_connect_timeout=0.01)
    config = RateLimiterConfig(capacity=10.0, rate=1.0)
    lim = TokenBucketLimiter(bad, config)
    # register_script is lazy; will fail on execute
    result = lim.allow("user_down")
    assert result is True  # fails open
```

---

## Reference Implementation — Trie Tests

```python
# tests/unit/test_trie.py
import pytest
from src.typeahead.trie import Trie


@pytest.fixture
def trie():
    return Trie()


def test_empty_trie_returns_empty(trie):
    assert trie.search("any") == []


def test_single_word_found(trie):
    trie.insert("apple", 1)
    assert "apple" in trie.search("apple")


def test_prefix_returns_multiple(trie):
    trie.insert("apple", 5)
    trie.insert("apply", 3)
    results = trie.search("appl")
    assert "apple" in results
    assert "apply" in results


def test_top_k_respected(trie):
    for i in range(20):
        trie.insert(f"word{i}", i)
    assert len(trie.search("word")) <= 5


def test_frequency_ordering(trie):
    trie.insert("banana", 1)
    trie.insert("band", 100)
    results = trie.search("ban")
    assert results[0] == "band"


def test_duplicate_insert_accumulates(trie):
    trie.insert("cat", 5)
    trie.insert("cat", 3)
    trie.insert("car", 10)
    results = trie.search("ca")
    # cat has freq=8, car has freq=10 → car first
    assert results[0] == "car"


def test_prefix_longer_than_word(trie):
    trie.insert("hi", 1)
    assert trie.search("hiya") == []


def test_case_normalized():
    from src.typeahead.service import TypeaheadService
    svc = TypeaheadService()
    svc.record_query("Apple", 10)
    results = svc.suggest("apple")
    assert "apple" in results


def test_delete_existing_word(trie):
    trie.insert("apple", 10)
    result = trie.delete("apple")
    assert result is True
    assert "apple" not in trie.search("appl")


def test_delete_updates_top_k(trie):
    trie.insert("apple", 10)
    trie.insert("apply", 5)
    trie.delete("apple")
    results = trie.search("app")
    assert "apple" not in results
    assert "apply" in results
```
