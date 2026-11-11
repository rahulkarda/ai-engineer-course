# Week 02 Assessment

## Conceptual (interview-style)

1. **When would you choose async over sync communication between services?**
   Choose async when the caller does not need an immediate response, when you want to decouple availability (producer continues even if consumer is down), when the workload is bursty and you need to absorb spikes, or when multiple downstream services need to react to the same event independently. Use sync (HTTP/gRPC) when you need an immediate result — e.g. a payment authorisation that must succeed before the user sees a confirmation.

2. **What is the difference between a queue and pub/sub? Give a concrete example.**
   Queue: one message, one consumer — work is load-balanced across a pool of workers. Example: order-processing workers pulling from a `payment_jobs` queue. Pub/sub: one message, many independent subscribers each get their own copy. Example: an `order_placed` event broadcast to an `inventory` service, a `notification` service, and an `analytics` service simultaneously. Kafka supports both patterns: consumer groups implement queue semantics; independent groups on the same topic implement pub/sub.

3. **Explain at-least-once delivery semantics and how you make a consumer idempotent.**
   At-least-once: the broker guarantees the message is delivered at least once but may redeliver on network failure or consumer crash before offset commit. A consumer may therefore process the same message more than once. To make it idempotent: use a natural deduplication key (e.g. `event_id` or `order_id`), store processed keys in a fast lookup (Redis SET, DB unique constraint, or an in-memory set for testing), and check before processing. The operation must be safe to repeat — e.g. `INSERT … ON CONFLICT DO NOTHING` or a state-machine guard (`if order.status != "placed": return`).

4. **What is a Kafka partition, and why does partition count matter?**
   A partition is an ordered, append-only log that is the unit of parallelism and storage in Kafka. Each topic is split into N partitions; each partition is owned by exactly one consumer in a consumer group at a time. Partition count sets the maximum parallelism for a consumer group — 10 partitions means at most 10 consumers process in parallel. Too few partitions becomes a throughput ceiling; too many adds broker overhead (file handles, replication traffic, rebalance cost). Choose based on target throughput ÷ single-partition throughput.

5. **What is a consumer group, and how does it enable horizontal scaling?**
   A consumer group is a named set of consumers that jointly consume a topic. Kafka assigns each partition to exactly one member of the group; if a member joins or leaves, Kafka rebalances. This means work is automatically sharded — adding consumers (up to the partition count) linearly increases throughput. Multiple independent groups on the same topic each get all messages, enabling both work-queue and fan-out patterns simultaneously.

6. **How does Kafka maintain ordering within a topic?**
   Ordering is guaranteed only within a single partition. Messages with the same partition key (e.g. `customer_id`) are always routed to the same partition, preserving per-key order. Across partitions there is no global order. For strict global ordering you must use a single partition, which sacrifices parallelism — trade-off must be deliberate. The Kafka producer's `enable.idempotence=true` plus `max.in.flight.requests.per.connection=5` preserves order under retries.

7. **When would you choose RabbitMQ over Kafka?**
   Choose RabbitMQ when: you need complex routing logic (topic/fanout/headers exchanges), per-message TTL, priority queues, or request/reply RPC patterns. RabbitMQ is operationally simpler for small-scale workloads, has mature dead-letter exchange support, and deletes messages after ACK (lower storage). Choose Kafka when you need high-throughput event streaming, long retention/replay, event sourcing, or ordered per-key processing at scale. Rule of thumb: RabbitMQ for task queues with rich routing; Kafka for event logs and data pipelines.

8. **What triggers a message to be routed to the DLQ?**
   A message is routed to the dead-letter queue (DLQ) when it cannot be processed successfully after a configured maximum number of retries. Common triggers: consumer throws an unhandled exception on every attempt, message fails schema validation (poison pill), deserialization error, or a downstream dependency is permanently unavailable. The DLQ preserves the original payload plus metadata (retry count, last error, original topic) for manual inspection, replay, or alerting. Without a DLQ, poison pills can block the queue indefinitely.

---

## Coding challenge

Build a **mini event-driven notification service** in Python 3.12+ that demonstrates async messaging, pub/sub fan-out, idempotency, and DLQ routing.

### Requirements

**Models (Pydantic v2)**
- `UserSignupEvent`: fields `event_id: UUID`, `user_id: str`, `email: str`, `timestamp: datetime`

**Broker abstraction**
- `AbstractBroker` (ABC): methods `publish(topic: str, event: BaseModel) -> None` and `subscribe(topic: str, handler: Callable) -> None`
- `InMemoryBroker(AbstractBroker)`: synchronous, stores subscribers per topic, delivers to all on publish

**Producers & consumers**
- `SignupProducer`: publishes `UserSignupEvent` to topic `"user.signup"`
- `WelcomeEmailConsumer`: subscribes to `"user.signup"`, logs a welcome email, **idempotent** — deduplicates by `user_id` (process each user only once)
- `AuditConsumer`: subscribes independently to `"user.signup"`, records every event to an audit log list

**DLQ**
- Any consumer handler that raises an exception has its message retried up to 3 times by the broker
- After 3 failures the event is published to `"dlq.<original_topic>"` with the original payload preserved
- `InMemoryBroker` must implement this retry + DLQ logic

**Tests (pytest)**
- Test that both consumers receive the same event (fan-out)
- Test idempotency: publishing the same `user_id` twice results in exactly one welcome email
- Test DLQ routing: a consumer that always raises is retried 3× then event appears on DLQ topic

### Acceptance criteria

- [ ] `UserSignupEvent` is a valid Pydantic v2 model with all required fields
- [ ] `AbstractBroker` defines the interface; `InMemoryBroker` is the concrete implementation
- [ ] Publishing one event triggers both `WelcomeEmailConsumer` and `AuditConsumer`
- [ ] `WelcomeEmailConsumer` processes a duplicate `user_id` only once
- [ ] A failing handler is retried exactly 3 times before DLQ routing
- [ ] Failed event appears on `"dlq.user.signup"` with original payload intact
- [ ] All secrets (if any) via `os.getenv`; `logging` used throughout — no `print`
- [ ] All tests pass with `pytest -q`; no external dependencies beyond `pydantic` and `pytest`

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-02/assessment.md
