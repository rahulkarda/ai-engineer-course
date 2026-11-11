# Week 05 Project — Classic System Design: Two Full Designs + Reference Implementations

> Time: ~4 hr (Day 6 Saturday)  ·  Deliverable: two polished design docs + two working implementations with tests

---

## Overview

You will produce four artifacts that together constitute a mini system-design portfolio entry:

1. **Design Document A**: Chat + Messaging System (full 8-step design).
2. **Design Document B**: Distributed Rate Limiter (full 8-step design).
3. **Reference Implementation A**: Distributed rate limiter (token bucket, Redis-backed, FastAPI middleware).
4. **Reference Implementation B**: Typeahead autocomplete service (in-memory trie, FastAPI HTTP API).

Each design should stand alone — a reader should be able to understand the system entirely from the document. Each implementation should be runnable with `uvicorn` and testable with `pytest`.

---

## Deliverable Spec

### Repo structure

```
week-05-project/
├── designs/
│   ├── chat-system.md
│   └── rate-limiter.md
├── src/
│   ├── rate_limiter/
│   │   ├── __init__.py
│   │   ├── token_bucket.py
│   │   ├── config.py
│   │   └── middleware.py
│   └── typeahead/
│       ├── __init__.py
│       ├── trie.py
│       ├── service.py
│       └── api.py
├── tests/
│   ├── unit/
│   │   ├── test_token_bucket.py    (≥ 8 tests)
│   │   └── test_trie.py            (≥ 10 tests)
│   └── integration/
│       └── test_rate_limiter_redis.py
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Design Document A — Chat + Messaging System

### Required sections (8-step framework)

**1. Requirements**
- Functional: list the 5 most important features. Pick a scope: 1:1 only, or 1:1 + groups ≤ 100.
- Non-functional: latency, availability, consistency, scale (DAU, messages/day).

**2. Estimation**
- DAU → concurrent connections.
- Messages/day → write QPS → storage/day → annual storage.
- Cache size for hot messages.

**3. API Design**
- At minimum: `send_message`, `get_history`, WebSocket frames for push.
- Include payload examples with field types.

**4. Data Model**
- Tables or document schemas for: users, conversations, messages.
- State where each lives (Postgres vs. Cassandra vs. Redis) and justify.

**5. High-Level Architecture**
- ASCII or Mermaid diagram covering: client, gateway, message service, storage, push notification.
- Label every arrow with the protocol (WebSocket, gRPC, Kafka, REST).

**6. Deep Dives** (choose 2 of 3)
- A. Message delivery guarantee (at-least-once + client dedup)
- B. Presence system (heartbeat, Redis TTL, fan-out throttling)
- C. Offline sync (reconnect flow, cursor-based replay)

**7. Bottlenecks**
- Table: bottleneck | scale at which it hits | mitigation.
- Minimum 3 rows.

**8. Trade-offs**
- Table: decision | what you gain | what you give up.
- Minimum 4 rows.

**Real-world reference**: cite Discord, Slack, WhatsApp, or Telegram at least once to ground your choices.

---

## Design Document B — Distributed Rate Limiter

### Required sections

**1. Requirements**
- Functional: per-user limit, per-IP limit, per-endpoint limit (different rates).
- Non-functional: decision latency, accuracy target (±%), availability.

**2. Estimation**
- Total QPS across 30 pods → Redis ops/s needed.
- Redis memory for sliding window counters.

**3. API / Interface**
- How does the limiter expose its decision? (Middleware function signature, sidecar HTTP endpoint, or library call)
- Response format for 429 (include `Retry-After` calculation).

**4. Data Model**
- Redis key schema for: per-user counter, per-IP counter, per-endpoint override rules.

**5. Architecture**
- Diagram: API pod → rate limiter middleware → Redis Cluster.
- Show where rules are stored (config DB or Redis hash).

**6. Deep Dives** (both required)
- A. Lua script atomicity: show the exact race condition that non-atomic GET+INCR creates, and the Lua fix.
- B. Fail-open vs. fail-closed: when Redis is unavailable, what happens? Which mode, and why?

**7. Bottlenecks**
- Minimum 3 rows.

**8. Trade-offs**
- Sliding window counter vs. token bucket: one row each, minimum 3 trade-off rows total.

---

## Reference Implementation A — Rate Limiter

Implement the token bucket limiter from Day 4 in a production-quality form.

### Requirements

- `TokenBucketLimiter.allow(key: str, cost: float = 1.0) -> bool`
- `RateLimiterSettings` via pydantic-settings (reads from env).
- `RateLimitMiddleware` for Starlette/FastAPI.
- Lua script for atomic check-and-consume.
- Fails open on `redis.exceptions.ConnectionError`.

### Test requirements (test_token_bucket.py)

All tests must use `fakeredis` or `unittest.mock` — no live Redis required for unit tests.

```python
# Required test names (implement bodies)
def test_first_request_allowed(): ...
def test_capacity_requests_all_allowed(): ...
def test_over_capacity_denied(): ...
def test_refill_after_wait(): ...
def test_fractional_cost(): ...
def test_keys_are_isolated(): ...
def test_concurrent_requests_no_double_spend(): ...
def test_redis_down_fails_open(): ...
```

---

## Reference Implementation B — Typeahead Service

Implement the trie-based typeahead service from Day 5.

### Requirements

- `Trie.insert(word, frequency)`, `Trie.search(prefix) -> list[str]`, `Trie.delete(word) -> bool`.
- `TypeaheadService.suggest(prefix) -> list[str]`, `.record_query(query)`, `.bulk_load(data)`.
- `GET /suggest?q=<prefix>` → JSON response.
- All strings normalized to lowercase in the service layer.

### Test requirements (test_trie.py)

```python
# Required test names
def test_empty_trie_returns_empty(): ...
def test_single_word_found(): ...
def test_prefix_returns_multiple_matches(): ...
def test_top_k_respected(): ...
def test_frequency_ordering(): ...
def test_duplicate_insert_accumulates(): ...
def test_prefix_longer_than_word_empty(): ...
def test_case_normalized(): ...
def test_delete_existing_word(): ...
def test_delete_updates_top_k(): ...
```

---

## Grading

| Component | Weight | Pass threshold |
|---|---|---|
| Design A: all 8 sections present and substantive | 20% | All sections ≥ 1 paragraph |
| Design B: algorithm comparison + Lua deep dive | 20% | Lua race condition explained correctly |
| Rate limiter implementation: tests pass | 30% | ≥ 6/8 tests green |
| Typeahead implementation: tests pass | 30% | ≥ 8/10 tests green |
| **Total** | **100%** | **≥ 75%** |

**Bonus** (+5% each, max +10%):
- Integration test for rate limiter against a live Redis (Docker Compose setup provided).
- Typeahead API deployed locally with `uvicorn` and screenshot/curl output in README.
