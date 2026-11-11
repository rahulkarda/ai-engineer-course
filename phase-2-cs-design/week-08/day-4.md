# Phase 2 · Week 08 · Day 4 — Inter-Service Communication

> Objective: Understand synchronous (REST, gRPC) and asynchronous (queues, pub/sub) communication patterns, and how an API gateway ties them together.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]]

---

## Concept

Distributed systems decompose work across services. Services must talk to each other. Two fundamental modes:

**Synchronous** — caller blocks waiting for response. Tight temporal coupling. Simpler reasoning, harder fault isolation.

**Asynchronous** — caller publishes work and moves on. Temporal decoupling. Better resilience, harder tracing.

Neither is universally better. Choice depends on latency requirements, coupling tolerance, and consistency needs.

---

## How it works internally

### REST conventions (HTTP/1.1 + JSON)

REST is an architectural style, not a protocol. Key constraints that make REST "RESTful":

- **Uniform interface**: resources identified by URIs (`/orders/42`), operations via HTTP verbs.
- **Stateless**: each request carries all context — no server-side session.
- **Cacheable**: GET responses carry cache directives; safe operations (GET/HEAD/OPTIONS) are idempotent.

**Verb semantics matter:**

| Verb   | Idempotent | Safe | Typical use               |
|--------|-----------|------|---------------------------|
| GET    | yes       | yes  | read resource             |
| POST   | no        | no   | create / trigger action   |
| PUT    | yes       | no   | replace resource fully    |
| PATCH  | no (ideally yes) | no | partial update      |
| DELETE | yes       | no   | remove resource           |

**Idempotency key pattern**: POST is not inherently idempotent (retrying creates duplicates). Clients send a unique `Idempotency-Key` header; servers store the result keyed to it and replay the same response on duplicate calls.

**Versioning strategies:**
- URI path: `/v1/orders` — most visible, easy to route
- Header: `Accept: application/vnd.myapp.v1+json` — clean URIs, harder to test in browser
- Query param: `?version=1` — simplest to add, muddies query semantics

### gRPC (HTTP/2 + Protocol Buffers)

gRPC uses HTTP/2 multiplexing + binary serialization (protobuf). Key advantages over REST/JSON:

- **Protobuf encoding** — structs compiled to dense binary; 3–10x smaller payloads vs JSON, ~5x faster serialization.
- **Strongly typed contracts** — `.proto` files are the source of truth; client/server stubs auto-generated in any supported language.
- **Streaming modes** — unary, server-streaming, client-streaming, **bidirectional streaming** (both sides send independent streams over one HTTP/2 connection).
- **Multiplexing** — multiple in-flight RPCs over one TCP connection; eliminates head-of-line blocking.

**How protobuf works internally:**

Each field is encoded as a (field_number, wire_type, value) triplet. Field names are NOT in the wire format — only numbers. This means:
1. Renaming a field without changing its number is backward-compatible.
2. Changing a field number is breaking.
3. Missing fields take default values (no null by default in proto3).

**Bidirectional streaming**: client opens a stream, both sides write messages asynchronously. Useful for: live telemetry, chat, real-time ML inference pipelines where the model streams partial tokens back.

### Async messaging

**Queue (point-to-point):**
- Each message consumed by exactly one consumer.
- Consumer groups in Kafka = partition ownership; each partition owned by one consumer per group.
- Good for: work queues, task distribution, load leveling.

**Pub/Sub (broadcast):**
- Message published to a topic, delivered to all subscribers.
- Good for: event notification, fan-out (e.g., "order placed" → billing + fulfillment + analytics all receive it).

**Delivery guarantees:**

| Guarantee      | Mechanism                              | Risk                          |
|----------------|----------------------------------------|-------------------------------|
| At-most-once   | fire-and-forget, no ack                | message loss on crash         |
| At-least-once  | ack after processing; retry on timeout | duplicate delivery            |
| Exactly-once   | idempotent consumer + transactional producer | highest cost        |

**At-least-once + idempotent consumer = effective exactly-once** without the heavy transaction overhead. Store a `message_id` in a dedup table; skip if already seen.

**Consumer groups (Kafka model):**
- Topic has N partitions.
- Consumer group has M consumers.
- Each partition assigned to exactly one consumer in the group.
- Scale consumers up to N max (extra consumers idle).
- Multiple groups = independent read offsets = pub/sub semantics on top of a queue.

### API Gateway

A gateway is a reverse proxy that sits at the edge and enforces cross-cutting concerns so individual services don't have to:

- **Routing**: path/host-based routing to backend services (`/v1/orders` → order-service).
- **Auth**: validate JWT/OAuth tokens centrally; forward identity via trusted header.
- **Rate limiting**: per-client token bucket or sliding window; reject with 429 before hitting backends.
- **Aggregation / BFF**: combine responses from multiple services into one payload (Backend For Frontend pattern).
- **TLS termination**: decrypt once at the edge; internal traffic can be plain HTTP on a trusted network.
- **Observability**: inject correlation IDs, emit access logs, forward traces.

**Request flow:**

```
Client → Gateway → Auth middleware → Rate limiter → Router → Service A
                                                            → Service B (aggregation)
```

---

## Worked example

```python
"""
week-08/day-4 examples
- REST client with idempotency-key retry logic
- gRPC concept: protobuf schema + stub usage pattern (no real server needed)
- Async message producer/consumer pattern (in-process queue simulation)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. REST client with idempotency-key support
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    item_id: str
    quantity: int
    customer_id: str


class OrderResponse(BaseModel):
    order_id: str
    status: str
    created_at: float = Field(default_factory=time.time)


class IdempotentRESTClient:
    """
    HTTP client that attaches an Idempotency-Key header and retries safely.
    The key is derived from the request body so the same logical request
    always maps to the same key — clients do NOT need to store keys.
    """

    def __init__(self, base_url: str, max_retries: int = 3) -> None:
        self._base_url = base_url
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(base_url=base_url, timeout=5.0)

    @staticmethod
    def _derive_key(payload: dict[str, Any]) -> str:
        """Deterministic key from payload — same input = same key."""
        import json
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:32]

    async def create_order(self, request: OrderRequest) -> OrderResponse:
        payload = request.model_dump()
        idempotency_key = self._derive_key(payload)
        headers = {"Idempotency-Key": idempotency_key, "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await self._client.post(
                    "/v1/orders", json=payload, headers=headers
                )
                if response.status_code in {200, 201}:
                    return OrderResponse.model_validate(response.json())
                if response.status_code == 409:
                    # 409 = duplicate detected by server, treat as success
                    logger.info("Duplicate request detected; replaying cached response")
                    return OrderResponse.model_validate(response.json())
                response.raise_for_status()
            except httpx.TransportError as exc:
                last_error = exc
                wait = 2**attempt
                logger.warning("Attempt %d failed, retrying in %ds: %s", attempt + 1, wait, exc)
                await asyncio.sleep(wait)

        raise RuntimeError(f"All retries exhausted: {last_error}") from last_error

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# 2. gRPC protobuf concept — showing what stubs look like post-codegen
#    (actual gRPC requires grpcio; this shows the shape without the dep)
# ---------------------------------------------------------------------------

# The .proto definition this mirrors:
#
# syntax = "proto3";
# package inference;
#
# message InferRequest {
#   string model_id = 1;
#   repeated float features = 2;
# }
#
# message InferResponse {
#   repeated float logits = 1;
#   float latency_ms = 2;
# }
#
# service InferenceService {
#   rpc Predict(InferRequest) returns (InferResponse);
#   rpc PredictStream(InferRequest) returns (stream InferResponse);
# }


@dataclass
class InferRequest:
    model_id: str
    features: list[float]


@dataclass
class InferResponse:
    logits: list[float]
    latency_ms: float


async def mock_grpc_predict_stream(
    request: InferRequest,
) -> "AsyncGenerator[InferResponse, None]":
    """
    Simulates a gRPC server-streaming RPC.
    In real gRPC: stub.PredictStream(request) returns an async iterator.
    Bidirectional streaming: both sides call write() and read() concurrently
    on the same stream object.
    """

    async def _generator() -> "AsyncGenerator[InferResponse, None]":
        # Simulate partial token streaming (e.g., LLM inference)
        for chunk_idx in range(3):
            await asyncio.sleep(0.01)  # simulate network RTT
            yield InferResponse(
                logits=[float(chunk_idx) * 0.1],
                latency_ms=chunk_idx * 5.0,
            )

    return _generator()


# ---------------------------------------------------------------------------
# 3. Async messaging — in-process simulation of queue vs pub/sub
# ---------------------------------------------------------------------------


@dataclass
class Message:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class InMemoryBroker:
    """
    Minimal broker simulation:
    - Queue mode: one consumer per message (round-robin across registered consumers)
    - PubSub mode: all subscribers on a topic receive every message
    Demonstrates the conceptual difference; production uses Kafka/RabbitMQ/SQS.
    """

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Message]] = defaultdict(asyncio.Queue)
        self._pubsub: dict[str, list[asyncio.Queue[Message]]] = defaultdict(list)
        self._dedup: set[str] = set()  # idempotency store

    # --- Queue (point-to-point) ---

    def subscribe_queue(self, topic: str) -> asyncio.Queue[Message]:
        """Each call returns the SAME queue — single consumer per topic partition."""
        return self._queues[topic]

    async def publish_queue(self, msg: Message) -> None:
        if msg.message_id in self._dedup:
            logger.info("Duplicate message_id=%s skipped (dedup)", msg.message_id)
            return
        self._dedup.add(msg.message_id)
        await self._queues[msg.topic].put(msg)
        logger.info("Queue enqueued topic=%s id=%s", msg.topic, msg.message_id)

    # --- Pub/Sub (fan-out) ---

    def subscribe_pubsub(self, topic: str) -> asyncio.Queue[Message]:
        """Each subscriber gets its own queue — independent delivery."""
        subscriber_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._pubsub[topic].append(subscriber_queue)
        return subscriber_queue

    async def publish_pubsub(self, msg: Message) -> None:
        subscribers = self._pubsub.get(msg.topic, [])
        logger.info(
            "PubSub broadcast topic=%s to %d subscribers", msg.topic, len(subscribers)
        )
        for q in subscribers:
            await q.put(msg)


# ---------------------------------------------------------------------------
# 4. Demo — wire everything together
# ---------------------------------------------------------------------------


async def demo_messaging() -> None:
    broker = InMemoryBroker()

    # --- Queue demo ---
    order_queue = broker.subscribe_queue("orders")

    msg = Message(topic="orders", payload={"item": "widget", "qty": 2})
    await broker.publish_queue(msg)
    # Simulate retry — same message_id → should be deduped
    await broker.publish_queue(msg)

    received = await order_queue.get()
    logger.info("Queue consumer got: %s", received.payload)

    # --- Pub/Sub demo ---
    billing_sub = broker.subscribe_pubsub("order_placed")
    fulfillment_sub = broker.subscribe_pubsub("order_placed")

    event = Message(topic="order_placed", payload={"order_id": "ord-001"})
    await broker.publish_pubsub(event)

    billing_msg = await billing_sub.get()
    fulfillment_msg = await fulfillment_sub.get()
    logger.info("Billing received: %s", billing_msg.payload)
    logger.info("Fulfillment received: %s", fulfillment_msg.payload)
    assert billing_msg.message_id == fulfillment_msg.message_id  # same event, two copies


async def demo_grpc_stream() -> None:
    request = InferRequest(model_id="classifier-v2", features=[0.1, 0.2, 0.9])
    gen = await mock_grpc_predict_stream(request)
    async for chunk in gen:
        logger.info("Stream chunk: logits=%s latency=%.1fms", chunk.logits, chunk.latency_ms)


if __name__ == "__main__":
    asyncio.run(demo_messaging())
    asyncio.run(demo_grpc_stream())
```

---

## Exercises

1. **REST idempotency**: Implement a FastAPI endpoint `POST /v1/payments` that accepts an `Idempotency-Key` header. Store results in an in-memory `dict`. On duplicate key, return the stored response with HTTP 200 (not 201). Write a test that calls the endpoint twice with the same key and asserts the `payment_id` is identical in both responses.

2. **gRPC proto design**: Write a `.proto` schema for a `UserService` with two RPCs: `GetUser(GetUserRequest) returns (User)` and `ListUsers(ListUsersRequest) returns (stream User)`. Explain in a comment why server-streaming is preferable to returning `repeated User` for large result sets.

3. **Delivery guarantees**: Extend `InMemoryBroker` to simulate at-least-once delivery: if a consumer does not call `ack(message_id)` within 5 seconds, the broker re-enqueues the message. Use `asyncio` timeouts and a background task.

4. **API gateway routing**: Sketch (pseudocode or real code) an `httpx`-based gateway that reads a routing table from a dict `{"path_prefix": "http://backend-host"}` and proxies requests. Add a middleware layer that validates a `Bearer` JWT (use `python-jose` or `PyJWT`) before forwarding.

(Solutions: ../solutions/week-08/day-4.md)

---

## Self-check

- [ ] I can explain why POST is not idempotent and how an `Idempotency-Key` header fixes it.
- [ ] I can describe how protobuf encodes fields (field numbers, wire types) and why field renaming is safe but number changes are breaking.
- [ ] I can explain the difference between a queue and pub/sub, and when to choose each.
- [ ] I can explain at-least-once + idempotent consumer as a practical substitute for exactly-once.
- [ ] I can list three cross-cutting concerns an API gateway handles that individual services should not.

---

## Supplementary reading

- [HTTP Semantics — RFC 9110 §9 (Methods)](https://www.rfc-editor.org/rfc/rfc9110#section-9)
- [Protocol Buffers encoding guide](https://protobuf.dev/programming-guides/encoding/)
- [Kafka consumer groups explained (Confluent docs)](https://developer.confluent.io/courses/apache-kafka/consumer-group-protocol/)
