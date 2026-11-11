# Solutions — Week 05 · Day 3 (Chat System Exercises)

---

## Exercise 1 — Message flow for an offline user

**Full flow with push notification**:

```
Sender Client
  │
  ├─ WebSocket: { type: "send_message", conversation_id: "c1", body: "hey", client_nonce: "uuid1" }
  │
Gateway (Sender's pod)
  │
  ├─ gRPC → Message Service
  │           │
  │     Idempotency check: nonce "uuid1" not seen
  │     Assign message_id: m_12345 (Snowflake)
  │     Persist to Cassandra (conversation_id=c1, message_id=m_12345)
  │     Publish to Kafka: messages.c1
  │           │
  │     ack to sender: { type: "delivery_ack", client_nonce: "uuid1", message_id: "m_12345" }
  │
Fanout Worker (consumes Kafka)
  │
  ├─ Look up members of c1: [sender_id, recipient_id]
  ├─ Check Redis: ws_pod:{recipient_id} → NOT FOUND (offline)
  ├─ Write to pending_delivery: { message_id: m_12345, recipient_id: ... }
  │
Push Notification Service
  ├─ Fetch device token for recipient from DB
  ├─ Send APNs/FCM notification: "New message from Alice"
  ├─ APNs returns 200 → mark push_sent in pending_delivery
  │
  (Recipient phone wakes up, user taps notification)
  │
Recipient Client reconnects
  ├─ WebSocket connect → authenticate
  ├─ Send: { type: "sync", last_seen_message_id: "m_11000" }
  │
Gateway (Recipient's pod)
  ├─ Store: ws_pod:{recipient_id} = this_pod
  ├─ Fetch from Cassandra: messages WHERE conversation_id IN (c1, ...) AND message_id > m_11000
  ├─ Push each missed message to client
  ├─ Client acks each: { type: "ack_received", message_id: "m_12345" }
  ├─ Fanout worker marks DELIVERED, clears pending_delivery
```

---

## Exercise 2 — Why Cassandra over Postgres for messages

**Three specific reasons**:

1. **Write path at scale**: at 700 K writes/s, Postgres's MVCC + WAL generates enormous I/O. Cassandra's LSM-tree storage is optimized for append-heavy write workloads and scales horizontally by adding nodes — no primary bottleneck.

2. **Access pattern alignment**: messages are always read by `(conversation_id, message_id)` — fetch the last N messages for a conversation. Cassandra's partition key `conversation_id` and clustering key `message_id` make this a single-partition sequential scan. Postgres requires a composite index and still does random I/O across rows.

3. **Time-to-live and data lifecycle**: Cassandra has native TTL per row (`INSERT ... USING TTL 2592000`). Old messages expire automatically without needing a separate cleanup job. Postgres requires manual `DELETE` jobs with associated vacuum overhead.

---

## Exercise 3 — Presence fanout for user with 10 000 friends

**Naive operation count**: user comes online → presence event → publish to Kafka → fanout worker looks up all 10 000 friends' ws_pod → sends 10 000 presence_update WebSocket frames. That's 10 000 Redis lookups + 10 000 gRPC sends per status change.

**Reduction strategies**:

1. **Only fan out to friends with an open conversation**: maintain a Redis set `active_chat_pairs:{user_id}` — users who have a conversation tab open with this user. Only push presence to those. Most of the 10 000 friends are not actively chatting with this user.

2. **Client-side polling for friend list**: for the friend list / contact list screen (which shows all 10 000 statuses), don't push — let the client poll `GET /api/v1/presence?user_ids=...` every 30 s. This trades push complexity for simple polling.

3. **Presence subscription model**: a client explicitly subscribes to presence for users it has open conversations with. The server only pushes to subscribed clients.

**Net result**: instead of 10 000 pushes per status change, typically 5–50 pushes (open conversations count).

---

## Exercise 4 — client_nonce idempotency + network drop

**What happens if server persists but crashes before ack**:

1. Server persists message with `client_nonce="uuid1"` and `message_id="m_12345"` to Cassandra.
2. Server crashes before sending `delivery_ack` to client.
3. Client never receives ack — times out after 5 s.
4. Client retries: sends same message with same `client_nonce="uuid1"`.
5. New server instance receives the retry.
6. Idempotency check: `SELECT message_id FROM messages WHERE client_nonce = 'uuid1' AND sender_id = X` → finds `m_12345`.
7. Returns `delivery_ack` with the original `message_id` — no duplicate stored.

**Key requirement**: the idempotency lookup must be on a table that is replicated consistently (Cassandra with quorum writes/reads) so any server instance sees the prior insert. Without this, two different pods could both "win" the idempotency check and insert two rows.

---

## Exercise 5 — Reconnection flow after 6 hours offline

**Client sends on reconnect**:
```json
{
  "type": "sync",
  "conversations": [
    { "conversation_id": "c1", "last_seen_message_id": "m_11000" },
    { "conversation_id": "c2", "last_seen_message_id": "m_9500" }
  ]
}
```

**Server response**:
1. For each conversation: `SELECT * FROM messages WHERE conversation_id = ? AND message_id > ? ORDER BY message_id ASC LIMIT 200`.
2. Stream batches to client: paginate if > 200 messages per conversation.
3. Clear `pending_delivery` entries for this user.
4. Send any pending "delivered" receipts for messages the user sent that were delivered while they were offline.
5. Send presence updates for friends who changed status.

**Pagination on reconnect**: after 6 hours the client may have thousands of missed messages across many conversations. Cap at 200 per conversation per sync request; client fetches older history separately if needed (lazy loading).
