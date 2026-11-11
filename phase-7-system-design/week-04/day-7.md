# Phase 7 · Week 04 · Day 7 — System Design Synthesis & Interview Preparation

> Objective: Apply the week's concepts to realistic system design scenarios, practice capacity estimation, and identify common failure modes in event-driven architectures.  ·  Time: ~1 hr  ·  Prereqs: [[day-6]]

---

## Concept

### Putting the week together

This week covered four inter-related layers:

```
LAYER                   PATTERN               DAY
-----------             ---------             ---
Delivery guarantees     At-least-once + dedup  Day 1-2
Coordination            Choreography / saga    Day 5
State management        Event sourcing          Day 6
Stream analytics        Windowed processing     Day 6
```

A production event-driven system stacks all of these. Here is where each appears:

```
Inbound events
    |
    v
[ Kafka topic ]
    |
    |-- [ Consumer Group A: Event-sourced aggregate writer ]
    |       - Dedup by event_id (idempotency key in header)
    |       - Save to EventStore (PostgreSQL append-only table)
    |       - Saga choreography: publish domain events on commit
    |
    |-- [ Consumer Group B: Stream processor ]
    |       - Tumbling window aggregation
    |       - Emit WindowResult every N seconds
    |
    v
[ Downstream services / dashboards ]
```

---

## How it works internally

### Capacity math for Kafka

In system design interviews, rough estimates are expected. Three numbers to know:

**Single partition throughput**:
- Producer: ~20–40 MB/s per partition (network / disk limited)
- Consumer: ~50–100 MB/s per partition (read from page cache)

**Rule of thumb formula**:
```
required_partitions = ceil(target_throughput_MB_s / per_partition_throughput_MB_s)

Also constrain by:
  required_partitions >= consumer_parallelism  (one partition per consumer in a group)
```

**Worked example**:
- Target: 500 MB/s ingest, 10 consumer instances
- Per-partition write throughput: 25 MB/s
- required_partitions = ceil(500 / 25) = 20
- max(20, 10 consumer instances) = **20 partitions**

**Retention sizing**:
```
retention_bytes = throughput_MB_s * retention_seconds * replication_factor
               = 25 MB/s * 604800 s (7 days) * 3
               = ~45 TB raw
```

**Message rate to partition**:
```
if avg_message_size = 1 KB
  20 MB/s per partition = 20,000 msg/s per partition
  10 partitions = 200,000 msg/s total
```

### Ordering guarantees — decision tree

```
Do you need strict ordering?
    |
    +-- No: Use multiple partitions freely. Scale out.
    |
    +-- Yes: Ordering within what scope?
             |
             +-- Per entity (order_id, user_id):
             |   → Partition by entity key.
             |     All events for entity A in one partition.
             |
             +-- Global (all events, any entity):
                 → Single partition. Max ~20 MB/s. Usually wrong choice.
                   Reconsider if you actually need global ordering.
```

### Common failure modes and mitigations

```
FAILURE                         SYMPTOM                      MITIGATION
-----------------------------   -------------------------    ----------------------------
Consumer lag spike              Events pile up in topic      Add partitions + consumers;
                                Dashboards go stale          autoscale consumer group

Poison pill message             Consumer crashes in loop;    Dead-letter topic (DLQ);
                                restarts repeatedly          max retries + backoff

Schema drift                    Deserialization errors       Schema registry (Avro/Protobuf);
                                after producer deploy        backward-compatible evolution

Clock skew between producers    Window results miss events   Watermarks + late-arrival
                                that arrive out of order     window with allowed lateness

EventStore hot partition        One aggregate_id gets        Shard aggregates; use
                                10x write load               write-behind cache

Saga partial completion         Some steps done, some not    Idempotent compensations;
                                after crash                  outbox pattern for saga events

Duplicate saga trigger          Two instances process same   Idempotency key on saga start;
                                ORDER_PLACED event           distributed lock or DB unique constraint

Snapshot corruption             Reconstitution produces      Validate snapshot on load;
                                wrong state                  fall back to full replay
```

### Outbox pattern

A critical pattern for reliable saga event publishing. Without it:

```
BAD: two-phase problem
  1. Write to DB  (success)
  2. Publish to Kafka  (crash before publishing)
  Result: DB updated, Kafka never receives event — saga stalls
```

With outbox:

```
GOOD: outbox pattern
  1. In same DB transaction:
     - Write business state
     - Insert event into outbox table
  2. Outbox relay process:
     - Polls outbox table
     - Publishes to Kafka
     - Marks row as published
  Result: DB commit = guaranteed eventual Kafka publish
```

```
Outbox table schema (PostgreSQL example):

  CREATE TABLE outbox (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_id TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ
  );

  CREATE INDEX ON outbox (published_at) WHERE published_at IS NULL;
```

### CQRS and read models

Event sourcing is commonly paired with **CQRS** (Command Query Responsibility Segregation):

```
Write side (command model):
  - Validates commands
  - Applies events to aggregate
  - Persists to EventStore

Read side (query model / projection):
  - Subscribes to event stream
  - Builds denormalized read model (e.g., Postgres table, Redis cache)
  - Optimized for queries, NOT for writes

Tradeoff: eventual consistency
  Read model lags write side by milliseconds to seconds.
  Commands must not depend on read model being current.
```

---

## Worked example: capacity estimation walkthrough

**Scenario**: Design an analytics ingestion system for an e-commerce platform.

Requirements:
- 50,000 events/second peak ingest
- Events average 2 KB each
- 30 days retention
- 5 consumer groups (stream processing, event sourcing, analytics, audit, ML pipeline)
- 99.95% availability target

**Step 1: Throughput**
```
Peak write: 50,000 msg/s * 2 KB = 100 MB/s
```

**Step 2: Partitions (write-bound)**
```
Per-partition write: 20 MB/s (conservative)
Partitions for throughput: ceil(100 / 20) = 5

Also need: max_consumer_instances per group
  If each group scales to 10 consumers: 10 partitions min
  5 groups * 10 consumers each, but consumers in different groups
  don't compete for partitions; only within a group

Final: max(5, 10) = 10 partitions (headroom: double to 20 for future growth)
```

**Step 3: Retention**
```
Raw:           100 MB/s * 86400 s/day * 30 days = ~259 TB
Replication 3: ~777 TB
With compression (LZ4, ~4:1): ~194 TB
```

**Step 4: Consumer lag budget**
```
5 consumer groups, each reading 100 MB/s peak
Total read load: 500 MB/s
Per broker (assume 10 brokers): 50 MB/s read — within page cache if warm
```

**Step 5: Availability**
```
99.95% = ~4.4 hrs downtime/year
Kafka replication factor 3: survives 2 broker failures
Rolling restart: zero downtime (min.insync.replicas=2)
```

**Summary diagram**:

```
                    50k msg/s
                    (100 MB/s)
                        |
                        v
              +--------------------+
              |   Kafka Cluster    |
              |   20 partitions    |
              |   3 replicas       |
              |   30-day retention |
              +--------------------+
              |   |   |   |   |
              v   v   v   v   v
            CG1 CG2 CG3 CG4 CG5
           (10) (10) (10)(10)(10)  consumers each
```

---

## System design interview structure

When answering a system design question involving event-driven architecture, use this structure (10-15 min total):

```
1. Clarify requirements (2 min)
   - What is the throughput? Peak vs average?
   - What are the latency SLOs? (ingest vs query)
   - Ordering requirements?
   - Exactly-once or at-least-once?
   - Retention?

2. High-level design (3 min)
   - Sketch producer → broker → consumer pipeline
   - Name the main components

3. Capacity estimation (3 min)
   - Messages/sec → MB/s
   - Partitions required
   - Storage sizing
   - Consumer group sizing

4. Deep dive on one component (4 min)
   - Interviewers usually guide you here
   - Common: "How do you handle duplicates?" → idempotency
   - Common: "What if a consumer fails mid-saga?" → compensation + outbox
   - Common: "How do you handle late-arriving events?" → watermarks

5. Failure modes and tradeoffs (2 min)
   - Name 2-3 failure scenarios specific to your design
   - Propose mitigations
```

### Common interview questions and strong answers

**Q: "How do you guarantee exactly-once processing?"**

Strong answer: "Exactly-once end-to-end is expensive and rarely truly necessary. I'd design for at-least-once delivery with idempotent consumers. Producers set `enable.idempotence=true`. Consumers store processed event IDs in the same DB transaction as the business write — an `ON CONFLICT DO NOTHING` on a processed_events table. This gives effectively-once semantics with far less coordination overhead."

**Q: "How do you handle a failed saga step?"**

Strong answer: "Each saga step publishes its result event. If a step fails, it publishes a failure event that triggers compensating handlers for all prior steps. Compensating handlers must be idempotent — retrying `PAYMENT_REVERSED` must not double-reverse. The saga start event includes an idempotency key so two consumers don't both start the same saga if the event is delivered twice."

**Q: "Why event sourcing over traditional CRUD?"**

Strong answer: "Event sourcing gives you a full audit log for free, enables temporal queries (what was the state at time T?), and makes it easy to add new read models by replaying history. The cost is increased complexity — you need projections, CQRS discipline, and snapshot management. I'd choose it when audit trails are required by compliance, or when multiple divergent read models need to coexist."

---

## Exercises

1. **Capacity estimate**: a gaming analytics platform ingests player action events. Peak: 200,000 events/second, avg size 500 bytes, 3 consumer groups, 90-day retention with 5:1 compression, replication factor 3. Calculate: required partitions, raw storage, compressed storage, and consumer group sizing.

2. **Failure mode analysis**: given this pipeline:

   ```
   Mobile App → API Gateway → Kafka → Event Sourcing Writer → PostgreSQL
                                   → Stream Processor → Redis
   ```

   Identify three distinct failure modes. For each: describe the symptom, which component fails, and your mitigation.

3. **Outbox implementation**: implement a Python `OutboxRelay` class with methods `enqueue(event: StoredEvent)` and `flush(publish_fn: Callable)`. `enqueue` stores events in an in-memory list. `flush` calls `publish_fn` for each un-published event and marks it published. Write 3 pytest tests.

4. **Schema evolution**: you have a `StoredEvent` with `payload: {"qty": int}` (version 1). Version 2 adds `{"qty": int, "warehouse_id": str}` (default `"WH-MAIN"`). Write an `upcaster(event: StoredEvent) -> StoredEvent` function that upgrades v1 events to v2. The function must be idempotent (running it twice on a v2 event returns the same result).

5. **Interview simulation**: write a 200-word written answer (no code) to: "Design the event-driven backend for a ride-sharing app. Drivers publish location updates at 1 Hz. Riders need ETA updates within 2 seconds. Peak: 100,000 active drivers. Focus on the messaging architecture and ordering strategy."

(Solutions: ../solutions/week-04/day-7.md)

---

## Self-check

- [ ] Can estimate required Kafka partitions given throughput and per-partition limits.
- [ ] Can calculate 30-day retention storage with replication and compression.
- [ ] Can describe the outbox pattern and the two-phase problem it solves.
- [ ] Can explain CQRS and the consistency tradeoff between write and read sides.
- [ ] Can name three common event-driven failure modes and a mitigation for each.
- [ ] Can structure a 15-minute system design answer for an event-driven system.
- [ ] Can articulate when event sourcing is worth the complexity and when it is not.

---

## Week 04 summary

```
Day 1-4 (Days 1-4 of week):
  - Kafka internals, consumer groups, delivery guarantees
  - Producer/consumer configuration

Day 5 (Friday):
  - Choreography vs orchestration — when each wins
  - Saga pattern — forward steps + compensations
  - Idempotency keys and deduplication strategies

Day 6 (Saturday):
  - EventSourcedAggregate base class
  - InventoryItem concrete aggregate
  - In-memory EventStore with snapshots + optimistic concurrency
  - TumblingWindowProcessor
  - Full pytest suite (47 tests)

Day 7 (Sunday):
  - Capacity estimation
  - Outbox pattern
  - CQRS and projections
  - Failure modes and mitigations
  - System design interview structure

Now complete: assessment.md + project.md
```

---

## Supplementary links

- Martin Fowler — CQRS: https://martinfowler.com/bliki/CQRS.html
- Debezium — Outbox pattern (CDC-based): https://debezium.io/blog/2019/02/19/reliable-microservices-data-exchange-with-the-outbox-pattern/
- Confluent — Kafka capacity planning: https://docs.confluent.io/platform/current/kafka/deployment.html#sizing
- Gunnar Morling — Outbox pattern in depth: https://www.morling.dev/blog/outbox-pattern-with-debezium-and-quarkus/
- Apache Kafka — Producer idempotence: https://kafka.apache.org/documentation/#producerconfigs_enable.idempotence
- Designing Data-Intensive Applications (Kleppmann) — Chapter 11: Stream Processing
