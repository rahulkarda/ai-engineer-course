# Phase 6 · Week 02 · Day 4 — Kafka Architecture: Topics, Partitions, Offsets, Consumer Groups, Replication

> Objective: Understand Kafka's internal model well enough to reason about ordering, throughput, and fault tolerance.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

Kafka is a **distributed commit log**, not a traditional message queue. The distinction matters:

- A traditional queue (RabbitMQ, SQS) delivers a message once and deletes it.
- Kafka **appends** messages to a durable log and lets consumers read from any offset, at any time, in any order across consumer groups.

This design enables:
- Multiple independent consumers reading the same data (fan-out)
- Replay: reprocess historical events by seeking to an earlier offset
- Decoupled producers and consumers — neither knows the other exists

Kafka stores data in **topics**. Each topic is sharded into **partitions**. Each partition is an ordered, immutable sequence of records. Every record has a monotonically increasing **offset**. Consumers track their own position; Kafka tracks nothing per-consumer except what the consumer commits.

---

## How it works internally

### Topics
A topic is a named logical channel. Producers write to a topic; consumers subscribe to a topic. A topic has no inherent ordering across partitions — ordering is guaranteed only within a single partition.

### Partitions
A partition is the unit of:
- **Parallelism**: multiple producers can write to different partitions simultaneously
- **Ordering**: records within a partition are strictly ordered by offset
- **Fan-out**: each consumer group gets its own read pointer per partition

The number of partitions is set at topic creation. More partitions = higher throughput but higher overhead (more file handles, more rebalancing cost).

### Offsets
An offset is a 64-bit integer, unique within a partition, assigned by the broker at write time. Offsets are monotonically increasing and never reused. The consumer controls its own offset:
- **Auto-commit**: Kafka commits the latest polled offset periodically (risky — can lose messages on crash)
- **Manual commit**: consumer calls `commit()` after it has fully processed the batch (safer, explicit)

A consumer stores its committed offset in Kafka's internal `__consumer_offsets` topic.

### Consumer Groups
A consumer group is a named set of consumers that collectively consume a topic. The Kafka broker's **group coordinator** assigns partitions to group members (one partition → one member, never shared). This means:
- Scale consumers horizontally up to the partition count
- Beyond that, extra consumers sit idle
- If a consumer dies, the coordinator triggers a **rebalance** and reassigns its partitions

Two separate groups each get all messages — they do not compete.

### Replication
Each partition has one **leader** and zero or more **followers**. All reads and writes go through the leader. Followers pull from the leader and stay in the **in-sync replica set (ISR)**. A follower falls out of ISR if it falls too far behind (`replica.lag.time.max.ms`).

On leader failure:
1. The group coordinator detects the leader is gone (session timeout)
2. A new leader is elected from the ISR
3. Only ISR members are eligible (ensuring no data loss if `acks=all` was used)

The producer's `acks` config determines durability:
- `acks=0`: fire and forget, no guarantee
- `acks=1`: leader acknowledged, follower lag not confirmed
- `acks=all` (or `-1`): all ISR replicas acknowledged before success response

### Retention
Kafka retains data independently of consumption:
- **Time-based**: delete segments older than `retention.ms` (default 7 days)
- **Size-based**: delete oldest segments when topic exceeds `retention.bytes`
- **Log compaction**: instead of deleting by time, keep only the **latest record per key** — useful for changelog topics (e.g., current state of an entity)

### Message flow end-to-end
```
Producer
  |
  |--> serialize(key, value)
  |--> partitioner(key) --> partition N
  |--> send to broker (leader for partition N)
  |--> broker appends to partition log, assigns offset
  |--> replicates to followers (if acks=all, waits for ISR)
  |--> returns offset to producer (delivery callback)

Consumer (poll loop)
  |
  |--> poll(timeout_ms) --> batch of records from assigned partitions
  |--> process each record
  |--> commit offset (manual or auto)
  |--> repeat
```

---

## Worked example

This example uses only Python standard library. It models Kafka concepts with dataclasses and Pydantic, then simulates parallel consumption with an in-memory broker — no real Kafka needed.

```python
"""
kafka_simulation.py

In-memory simulation of Kafka concepts:
  - Topic, Partition, Offset
  - Producer appending records
  - ConsumerGroup with partition assignment
  - Parallel poll across consumers

Run: python kafka_simulation.py
Requires: Python 3.12+, pydantic>=2
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models — typed, Pydantic v2
# ---------------------------------------------------------------------------


class KafkaRecord(BaseModel):
    """A single message written to a partition."""

    key: str | None
    value: dict[str, Any]
    partition: int
    offset: int

    @field_validator("offset")
    @classmethod
    def offset_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("offset must be >= 0")
        return v


class TopicConfig(BaseModel):
    """Topic-level configuration."""

    name: str
    num_partitions: int = 3
    replication_factor: int = 1  # in-memory: always 1

    @field_validator("num_partitions")
    @classmethod
    def at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_partitions must be >= 1")
        return v


# ---------------------------------------------------------------------------
# In-memory broker
# ---------------------------------------------------------------------------


@dataclass
class Partition:
    """Single partition: ordered log of records."""

    partition_id: int
    log: list[KafkaRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        with self._lock:
            offset = len(self.log)
            record = KafkaRecord(
                key=key,
                value=value,
                partition=self.partition_id,
                offset=offset,
            )
            self.log.append(record)
        return record

    def read_from(self, offset: int, max_records: int = 10) -> list[KafkaRecord]:
        with self._lock:
            return self.log[offset : offset + max_records]


@dataclass
class Topic:
    """Topic: named collection of partitions."""

    config: TopicConfig
    partitions: list[Partition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.partitions = [
            Partition(partition_id=i)
            for i in range(self.config.num_partitions)
        ]

    def partition_for_key(self, key: str | None) -> int:
        """Deterministic partitioner: hash of key mod num_partitions."""
        if key is None:
            # Round-robin fallback: use total record count across all partitions
            total = sum(len(p.log) for p in self.partitions)
            return total % self.config.num_partitions
        return hash(key) % self.config.num_partitions


@dataclass
class InMemoryBroker:
    """
    Simulates a single-broker Kafka cluster.

    Supports:
    - Topic creation
    - Producing records (with key-based partitioning)
    - Consumer group offset tracking
    - Polling records per assigned partition
    """

    topics: dict[str, Topic] = field(default_factory=dict)
    # consumer_offsets[group][topic][partition] = committed_offset
    consumer_offsets: dict[str, dict[str, dict[int, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def create_topic(self, config: TopicConfig) -> None:
        with self._lock:
            if config.name in self.topics:
                logger.warning("Topic '%s' already exists, skipping.", config.name)
                return
            self.topics[config.name] = Topic(config=config)
            logger.info(
                "Created topic '%s' with %d partitions.",
                config.name,
                config.num_partitions,
            )

    def produce(
        self,
        topic_name: str,
        key: str | None,
        value: dict[str, Any],
    ) -> KafkaRecord:
        topic = self.topics[topic_name]
        partition_id = topic.partition_for_key(key)
        record = topic.partitions[partition_id].append(key, value)
        logger.debug(
            "Produced to %s[%d]@%d key=%s",
            topic_name,
            partition_id,
            record.offset,
            key,
        )
        return record

    def poll(
        self,
        group_id: str,
        topic_name: str,
        partition_id: int,
        max_records: int = 5,
    ) -> list[KafkaRecord]:
        """Fetch records starting from committed offset for this group/partition."""
        offset = self.consumer_offsets[group_id][topic_name][partition_id]
        records = self.topics[topic_name].partitions[partition_id].read_from(
            offset, max_records
        )
        return records

    def commit(
        self,
        group_id: str,
        topic_name: str,
        partition_id: int,
        offset: int,
    ) -> None:
        """Commit offset for a consumer group + partition."""
        self.consumer_offsets[group_id][topic_name][partition_id] = offset
        logger.debug(
            "Committed %s[%d]@%d for group '%s'",
            topic_name,
            partition_id,
            offset,
            group_id,
        )


# ---------------------------------------------------------------------------
# Consumer group simulation
# ---------------------------------------------------------------------------


def assign_partitions(
    group_members: list[str],
    num_partitions: int,
) -> dict[str, list[int]]:
    """
    Range assignor: distribute partitions as evenly as possible.
    Returns {member_id: [partition_ids]}.
    """
    assignment: dict[str, list[int]] = {m: [] for m in group_members}
    for partition_id in range(num_partitions):
        member = group_members[partition_id % len(group_members)]
        assignment[member].append(partition_id)
    return assignment


def consumer_worker(
    broker: InMemoryBroker,
    group_id: str,
    topic_name: str,
    assigned_partitions: list[int],
    consumer_id: str,
) -> None:
    """
    Simulates one consumer instance: poll -> process -> commit.
    Runs until all assigned partitions are exhausted.
    """
    exhausted = set()
    while len(exhausted) < len(assigned_partitions):
        for partition_id in assigned_partitions:
            if partition_id in exhausted:
                continue
            records = broker.poll(group_id, topic_name, partition_id)
            if not records:
                exhausted.add(partition_id)
                continue
            for record in records:
                logger.info(
                    "[%s] partition=%d offset=%d key=%s value=%s",
                    consumer_id,
                    record.partition,
                    record.offset,
                    record.key,
                    record.value,
                )
            # manual commit: next offset to read = last_offset + 1
            next_offset = records[-1].offset + 1
            broker.commit(group_id, topic_name, partition_id, next_offset)


# ---------------------------------------------------------------------------
# Main: wire it all together
# ---------------------------------------------------------------------------


def main() -> None:
    broker = InMemoryBroker()

    config = TopicConfig(name="orders", num_partitions=3, replication_factor=1)
    broker.create_topic(config)

    # Produce 12 messages with distinct keys (deterministic partitioning)
    orders = [
        ("user-1", {"order_id": f"o{i}", "amount": i * 10})
        for i in range(1, 13)
    ]
    for key, value in orders:
        broker.produce("orders", key, value)

    logger.info("=== Partition log sizes ===")
    for p in broker.topics["orders"].partitions:
        logger.info("  partition %d: %d records", p.partition_id, len(p.log))

    # Consumer group with 2 members consuming 3 partitions
    group_id = "analytics-group"
    members = ["consumer-A", "consumer-B"]
    assignment = assign_partitions(members, num_partitions=3)

    logger.info("=== Partition assignment ===")
    for member, partitions in assignment.items():
        logger.info("  %s -> partitions %s", member, partitions)

    # Run consumers in parallel threads
    threads = [
        threading.Thread(
            target=consumer_worker,
            args=(broker, group_id, "orders", partitions, member),
            name=member,
        )
        for member, partitions in assignment.items()
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info("=== Committed offsets ===")
    for partition_id, offset in broker.consumer_offsets[group_id]["orders"].items():
        logger.info("  partition %d: committed offset %d", partition_id, offset)

    # Second consumer group reads same data independently
    logger.info("=== Second group (audit-group) replaying from offset 0 ===")
    audit_group = "audit-group"
    audit_assignment = assign_partitions(["auditor"], num_partitions=3)
    audit_threads = [
        threading.Thread(
            target=consumer_worker,
            args=(broker, audit_group, "orders", partitions, member),
            name=member,
        )
        for member, partitions in audit_assignment.items()
    ]
    for t in audit_threads:
        t.start()
    for t in audit_threads:
        t.join()


if __name__ == "__main__":
    main()
```

**Expected output (order within group may vary across partitions):**
```
Created topic 'orders' with 3 partitions.
Partition 0: N records, Partition 1: M records, Partition 2: K records
consumer-A reads partitions [0, 2], consumer-B reads partition [1]
Both groups see all 12 records; committed offsets reflect records consumed.
```

---

## Exercises

1. **Ordering guarantee**: Modify `main()` to produce 5 messages with the same key (`"user-42"`). Print which partition they land in and verify all have consecutive offsets. Explain why same-key messages always go to the same partition.

2. **Consumer lag**: After the first consumer group finishes, produce 3 more messages to the topic. Then add a function `consumer_lag(broker, group_id, topic_name)` that returns a dict of `{partition_id: lag}` where lag = `partition_log_length - committed_offset`. Print the lag before and after the second consumer group drains it.

3. **Rebalance simulation**: Start with 3 consumers for 3 partitions (1:1). Remove one consumer mid-run (kill its thread early). Implement a `rebalance()` function that reassigns the dead consumer's partitions to the remaining consumers. Show that no messages are lost and no messages are processed twice (hint: manual commit from last committed offset).

4. **Log compaction**: Implement a `compact(broker, topic_name)` function that, for each partition, retains only the latest record per key (equivalent to Kafka's log compaction). Run it and verify the partition sizes shrink while the latest state per key is preserved.

5. **Replication factor > 1**: Extend `Partition` to hold a `replicas: list[list[KafkaRecord]]` list. When `produce()` is called, write to the leader log and sync all followers. Simulate a leader failure by marking one partition's leader as unavailable and promoting the first follower as the new leader.

(Solutions: ../solutions/week-02/day-4.md)

---

## Self-check

- [ ] I can explain why Kafka is a commit log and not a queue, and name two consequences of that difference.
- [ ] I can describe what an offset is, who tracks it, and what happens when a consumer crashes before committing.
- [ ] I can explain why adding consumers beyond the partition count gives no throughput benefit.
- [ ] I can trace a message from producer call to consumer poll, naming each step and the actor responsible.
- [ ] I can implement `assign_partitions` for a round-robin strategy without reference.
- [ ] I can explain what ISR means and why `acks=all` requires all ISR members, not all replicas.
- [ ] I can distinguish time-based retention from log compaction and name a use case for each.

---

## Supplementary links

- [Kafka design docs — log](https://kafka.apache.org/documentation/#design_log)
- [Kafka design docs — consumer groups](https://kafka.apache.org/documentation/#intro_consumers)
- [Jay Kreps — "The Log"](https://engineering.linkedin.com/distributed-systems/log-what-every-software-engineer-should-know-about-real-time-datas-unifying)
- [Confluent — Kafka internals for the curious](https://developer.confluent.io/courses/architecture/get-started/)
- [KIP-392 — Allow consumers to fetch from closest replica](https://cwiki.apache.org/confluence/display/KAFKA/KIP-392%3A+Allow+consumers+to+fetch+from+closest+replica)
