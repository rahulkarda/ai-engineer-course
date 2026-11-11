# Phase 7 · Week 05 · Day 3 — Chat / Messaging System (Presence, Delivery)

> Objective: Design a scalable real-time chat system covering presence, message delivery guarantees, and offline sync.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-2]]

---

## Concept

A chat system is deceptively hard because it requires **bidirectional, low-latency communication** at scale. Unlike HTTP's request-response model, chat needs the server to push messages to clients the moment they arrive. The three hard problems are:

1. **Message delivery**: send exactly once, even when clients are offline.
2. **Presence**: show who is online/offline in real time without polling.
3. **Read receipts and ordering**: messages must appear in the same order for all participants.

---

## Step 1 — Requirements

### Functional
- 1:1 and group chats (≤ 500 members).
- Send text, images, files.
- Online/offline presence indicator.
- Delivered and read receipts.
- Message history (last 30 days hot, older archived).
- Push notifications for offline users.

### Non-functional
- 100 M DAU; peak 50 M concurrent connections.
- Message delivery latency P99 < 100 ms (online recipient).
- At-least-once delivery; dedup on client.
- 5 nines availability for message storage; eventual consistency for presence.

---

## Step 2 — Estimation

```
DAU             : 100 M
Concurrent conn : 50 M (WebSocket connections held open)
Msgs/day        : 60 B (avg 600 msgs/DAU)
Write QPS       : 60 B / 86 400 s ≈ 700 000 msgs/s
Storage/day     : 700 K msgs/s × 100 B avg = 70 MB/s = ~6 TB/day
Group chat msgs : 20% of messages go to groups of avg 10 members → additional 2× fanout
```

---

## Step 3 — API Design

```
WebSocket: wss://chat.example.com/ws
  Authenticate via token on connect.

  Client → Server frames:
    { "type": "send_message",   "conversation_id": "...", "body": "...", "client_nonce": "uuid" }
    { "type": "ack_received",   "message_id": "..." }
    { "type": "ack_read",       "message_id": "..." }
    { "type": "typing_start",   "conversation_id": "..." }

  Server → Client frames:
    { "type": "new_message",    "message_id": "...", "sender_id": "...", "body": "..." }
    { "type": "delivery_ack",   "client_nonce": "..." }
    { "type": "presence_update","user_id": "...", "status": "online|offline" }
    { "type": "read_receipt",   "message_id": "...", "reader_id": "..." }

REST (for history & metadata):
  GET  /api/v1/conversations/{id}/messages?before=<msg_id>&limit=50
  POST /api/v1/conversations                         (create group)
  GET  /api/v1/conversations                         (list user's chats)
```

---

## Step 4 — Data Model

```sql
-- Conversations (1:1 and groups)
CREATE TABLE conversations (
    conversation_id  BIGINT PRIMARY KEY,
    type             CHAR(5) NOT NULL,   -- 'one1' | 'group'
    name             TEXT,
    created_at       TIMESTAMPTZ NOT NULL
);

-- Membership
CREATE TABLE conversation_members (
    conversation_id  BIGINT NOT NULL,
    user_id          BIGINT NOT NULL,
    joined_at        TIMESTAMPTZ NOT NULL,
    last_read_msg_id BIGINT,            -- for read receipt tracking
    PRIMARY KEY (conversation_id, user_id)
);

-- Messages  (Cassandra/ScyllaDB is preferred here — see deep dive)
CREATE TABLE messages (
    conversation_id  BIGINT NOT NULL,
    message_id       BIGINT NOT NULL,   -- Snowflake; time-sortable
    sender_id        BIGINT NOT NULL,
    body             TEXT,
    media_url        TEXT,
    client_nonce     UUID NOT NULL,     -- idempotency key
    created_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (conversation_id, message_id)  -- partition by conv, cluster by msg
) WITH CLUSTERING ORDER BY (message_id DESC);

-- Presence  (Redis, not SQL)
-- Key: presence:{user_id}  Value: "online" | "offline"  TTL: 30 s
```

---

## Step 5 — High-Level Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  Mobile / Web Clients                                         │
│        │ WebSocket (persistent)                               │
│  ┌─────▼─────────────────────────────────────────────────┐   │
│  │  Chat Gateway (stateful, one per region)               │   │
│  │  - holds 50 M WebSocket connections across N pods      │   │
│  │  - routes messages to Message Service via gRPC         │   │
│  └─────┬─────────────────────────────────────────────────┘   │
│        │                                                      │
│  ┌─────▼─────────────┐    ┌──────────────────────────────┐   │
│  │  Message Service  │    │  Presence Service             │   │
│  │  - write to Kafka │    │  - heartbeats from clients    │   │
│  │  - fanout workers │    │  - Redis TTL per user         │   │
│  └─────┬─────────────┘    └──────────────────────────────┘   │
│        │                                                      │
│  ┌─────▼──────────┐   ┌────────────────────────────────┐     │
│  │  Cassandra     │   │  Notification Service           │     │
│  │  (messages)    │   │  APNs / FCM for offline users   │     │
│  └────────────────┘   └────────────────────────────────┘     │
│                                                               │
│  Kafka: message events, fanout, delivery acks                 │
│  Redis: presence, user→gateway mapping, typing indicators     │
└───────────────────────────────────────────────────────────────┘
```

---

## Step 6 — Deep Dives

### 6a. Message Delivery Guarantee

**The 3-state model** (every messaging system uses a variant of this):

```
SENT      → client sends; server assigns message_id and persists
DELIVERED → target client's Gateway confirms receipt via ack_received
READ      → target user views the message; client sends ack_read
```

**At-least-once delivery flow**:
1. Client sends `send_message` with `client_nonce`.
2. Message Service checks if `client_nonce` exists (idempotency) — if yes, return stored `message_id`.
3. Persist to Cassandra.
4. Publish to Kafka topic `messages.{conversation_id}`.
5. Fanout worker: look up online members → push via their Gateway → wait for `ack_received` → mark DELIVERED.
6. If no ack within 10 s (client offline): store in `pending_delivery` queue → send push notification via APNs/FCM.
7. When client reconnects: replay all undelivered messages (fetch by `last_seen_message_id`).

### 6b. Presence System

```
Client → heartbeat every 20 s → Presence Service
Presence Service: SETEX presence:{user_id} 30 "online"

If no heartbeat for 30 s → Redis TTL expires → user goes "offline"
Presence update → publish to Kafka → friends' Gateways → push presence_update frame

At reconnect: client sends "I'm back" → server replays missed messages
```

**Scaling presence**: at 100 M DAU, even 20 s heartbeats = 5 M heartbeats/s. Presence Service shards by `user_id % N`. Redis Cluster handles this at < 1 ms per SETEX.

**Fan-out problem for presence**: user A has 500 friends online → their presence change triggers 500 WebSocket pushes. Throttle: only push presence to friends who currently have a conversation open with A.

### 6c. Message Storage — Why Cassandra

Cassandra's partition-per-conversation + clustering by `message_id` means:
- `SELECT WHERE conversation_id = X AND message_id < cursor LIMIT 50` → single partition scan.
- Writes are O(1) regardless of conversation history length.
- Natural horizontal scaling by adding nodes.

Postgres would require range queries across a massive table — not suitable at 6 TB/day write rate.

### 6d. Gateway Routing

Each pod of the Chat Gateway holds a set of WebSocket connections. When Gateway A receives an incoming message destined for user B, it must find which pod holds B's connection.

**Solution**: store user → gateway_pod mapping in Redis.

```
On connect: SET ws_pod:{user_id} gateway_pod_id  (TTL = connection lifetime)
On message send: HGET ws_pod:{recipient_id} → forward to that pod via internal gRPC
```

### 6e. Group Chat Fanout

For a group of 500 members:
1. Store one copy of the message.
2. Fanout: look up 500 member IDs → look up each member's gateway pod → push.
3. For large groups (> 500), use Kafka consumer group per conversation — members subscribe, each pod consumes only messages for its connected users.

---

## Step 7 — Bottlenecks

| Bottleneck | Impact | Mitigation |
|---|---|---|
| 50 M WebSocket connections | Memory per connection ~10 KB → 500 GB RAM | Spread across hundreds of Gateway pods; each pod handles ~500 K connections |
| Cassandra hot partition (popular group) | Write amplification | Limit group size to 500; large-group communities use broadcast architecture instead |
| Presence fanout for users with many friends | O(N) pushes per status change | Fan out only to friends with open chat window |
| Push notification storms | APNs / FCM rate limits | Batch notifications; exponential backoff on failures |
| Ordering in distributed system | Messages can arrive out of order | Snowflake message_id (server-assigned) is the canonical order; clients sort on receive |

---

## Step 8 — Trade-offs

| Decision | Gain | Cost |
|---|---|---|
| WebSocket over long-polling | True push, lower latency | Stateful servers; harder load balancing |
| Cassandra for messages | Write scalability, time-range queries | No JOINs; eventual consistency |
| Redis for presence | Sub-ms reads | Data loss on Redis restart (acceptable for presence) |
| At-least-once delivery | No message loss | Client must deduplicate on `message_id` |
| Kafka for fanout | Durable, retryable, decoupled | Added latency ~200 ms; operational complexity |

---

## Worked Example — Presence Heartbeat Logic (Python)

```python
# src/chat/presence.py
"""
Presence service: update and query user online status via Redis.
"""
import logging
import os
import time
from typing import Literal

import redis

logger = logging.getLogger(__name__)

PresenceStatus = Literal["online", "offline"]

HEARTBEAT_TTL_SECONDS = 30


class PresenceService:
    def __init__(self, redis_client: redis.Redis) -> None:
        self._redis = redis_client

    def heartbeat(self, user_id: int) -> None:
        """Called every ~20 s by client; extends TTL."""
        key = f"presence:{user_id}"
        self._redis.setex(key, HEARTBEAT_TTL_SECONDS, "online")
        logger.debug("Heartbeat from user %d", user_id)

    def get_status(self, user_id: int) -> PresenceStatus:
        """Return current status for a user."""
        key = f"presence:{user_id}"
        value = self._redis.get(key)
        return "online" if value else "offline"

    def get_bulk_status(self, user_ids: list[int]) -> dict[int, PresenceStatus]:
        """Batch lookup for multiple users (e.g. friend list)."""
        pipe = self._redis.pipeline()
        for uid in user_ids:
            pipe.get(f"presence:{uid}")
        results = pipe.execute()
        return {
            uid: ("online" if val else "offline")
            for uid, val in zip(user_ids, results)
        }
```

---

## Exercises

1. Draw the message flow for sending a message to an offline user, including the push notification path.
2. Why is Cassandra preferred over Postgres for message storage at scale? Name three specific reasons related to the access pattern.
3. A user has 10 000 friends. They come online. How many Redis operations does the naive presence fanout generate? How do you reduce this?
4. Explain the `client_nonce` idempotency pattern. What happens if the network drops after the server persists but before sending the ack?
5. Design the reconnection flow: a mobile client was offline for 6 hours. What does it send on reconnect, and what does the server return?

(Solutions: ../solutions/week-05/day-3.md)

---

## Self-check
- [ ] I can explain the 3-state message delivery model (sent/delivered/read).
- [ ] I can describe how presence is implemented using Redis TTL.
- [ ] I can sketch the Gateway routing system for WebSocket connections.
- [ ] I can explain why Cassandra is chosen and the partition key design.

---

## Supplementary
- "How Discord Stores Billions of Messages" — discord.com/blog
- Building Real-Time Systems with Kafka — Confluent docs
- Cassandra data modeling guide — cassandra.apache.org
