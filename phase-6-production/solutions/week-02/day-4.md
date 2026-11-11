# Phase 6 · Week 02 · Day 4 — Solutions

> Solutions for: Topics, Partitions, Offsets, Consumer Groups, Replication

---

## Exercise 1 — Ordering guarantee: same-key messages go to same partition

**Problem**: Modify `main()` to produce 5 messages with the same key (`"user-42"`). Print which partition they land in and verify all have consecutive offsets. Explain why same-key messages always go to the same partition.

**Explanation**: `partition_for_key` computes `hash(key) % num_partitions`. Since `hash("user-42")` is deterministic within a Python process, the result is always the same integer. Every call with `"user-42"` maps to the identical partition. This is the core contract: a fixed key → fixed hash → fixed partition. Offsets within a single partition are monotonically increasing, so five consecutive appends produce offsets 0, 1, 2, 3, 4 (or wherever the partition currently sits).

```python
"""
ex1_same_key_ordering.py

Demonstrates that same-key messages always land in the same partition
and have consecutive offsets.

Run: python ex1_same_key_ordering.py
Requires: Python 3.12+, pydantic>=2
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class KafkaRecord(BaseModel):
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
    name: str
    num_partitions: int = 3
    replication_factor: int = 1

    @field_validator("num_partitions")
    @classmethod
    def at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_partitions must be >= 1")
        return v


@dataclass
class Partition:
    partition_id: int
    log: list[KafkaRecord] = field(default_factory=list)

    def append(self, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        offset = len(self.log)
        record = KafkaRecord(key=key, value=value, partition=self.partition_id, offset=offset)
        self.log.append(record)
        return record

    def read_from(self, offset: int, max_records: int = 10) -> list[KafkaRecord]:
        return self.log[offset : offset + max_records]


@dataclass
class Topic:
    config: TopicConfig
    partitions: list[Partition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.partitions = [Partition(partition_id=i) for i in range(self.config.num_partitions)]

    def partition_for_key(self, key: str | None) -> int:
        if key is None:
            total = sum(len(p.log) for p in self.partitions)
            return total % self.config.num_partitions
        return hash(key) % self.config.num_partitions


@dataclass
class InMemoryKafkaBroker:
    topics: dict[str, Topic] = field(default_factory=dict)
    consumer_offsets: dict[str, dict[str, dict[int, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )

    def create_topic(self, config: TopicConfig) -> None:
        if config.name not in self.topics:
            self.topics[config.name] = Topic(config=config)
            logger.info("Created topic '%s' with %d partitions.", config.name, config.num_partitions)

    def produce(self, topic_name: str, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        topic = self.topics[topic_name]
        partition_id = topic.partition_for_key(key)
        return topic.partitions[partition_id].append(key, value)


def main() -> None:
    broker = InMemoryKafkaBroker()
    broker.create_topic(TopicConfig(name="orders", num_partitions=3))

    KEY = "user-42"
    records: list[KafkaRecord] = []

    for i in range(5):
        rec = broker.produce("orders", key=KEY, value={"seq": i, "amount": i * 100})
        records.append(rec)
        logger.info(
            "Produced: key=%s partition=%d offset=%d",
            rec.key,
            rec.partition,
            rec.offset,
        )

    partitions_seen = {r.partition for r in records}
    offsets = [r.offset for r in records]

    logger.info("Partitions used: %s (should be exactly 1)", partitions_seen)
    logger.info("Offsets: %s (should be consecutive)", offsets)

    assert len(partitions_seen) == 1, "FAIL: same key must always land in same partition"
    assert offsets == list(range(len(offsets))), "FAIL: offsets must be consecutive starting at 0"
    logger.info("PASS: all 5 same-key messages → partition %d, offsets 0-4", records[0].partition)

    # Why? hash("user-42") is deterministic within a Python process.
    # partition_for_key returns hash("user-42") % 3 — always the same value.
    # Within that partition the log is append-only so offsets are 0,1,2,3,4.
    logger.info(
        "Explanation: hash('%s') %% 3 = %d (deterministic within process)",
        KEY,
        hash(KEY) % 3,
    )


if __name__ == "__main__":
    main()
```

---

## Exercise 2 — Consumer group rebalancing: add/remove consumers, show partition reassignment

**Problem**: Start with 3 consumers for 3 partitions (1:1). Remove one consumer mid-run (kill its thread early). Implement a `rebalance()` function that reassigns the dead consumer's partitions to the remaining consumers. Show that no messages are lost and no messages are processed twice.

**Explanation**: Rebalancing is triggered when the group membership changes. The key to no-loss / no-duplicate is that we reassign partitions starting from the **last committed offset**, not from 0. Each consumer commits after each batch; on rebalance the replacement consumer picks up exactly where the dead consumer left off.

```python
"""
ex2_consumer_rebalance.py

Simulates consumer group rebalancing when a member dies mid-run.

Key invariants demonstrated:
1. Every message is processed exactly once.
2. Rebalance picks up from the last committed offset — no duplicates, no loss.

Run: python ex2_consumer_rebalance.py
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class KafkaRecord(BaseModel):
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


@dataclass
class Partition:
    partition_id: int
    log: list[KafkaRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        with self._lock:
            offset = len(self.log)
            record = KafkaRecord(key=key, value=value, partition=self.partition_id, offset=offset)
            self.log.append(record)
            return record

    def read_from(self, offset: int, max_records: int = 10) -> list[KafkaRecord]:
        with self._lock:
            return self.log[offset : offset + max_records]

    @property
    def latest_offset(self) -> int:
        with self._lock:
            return len(self.log)


@dataclass
class InMemoryKafkaBroker:
    topics: dict[str, Any] = field(default_factory=dict)
    consumer_offsets: dict[str, dict[str, dict[int, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def create_topic(self, name: str, num_partitions: int) -> None:
        with self._lock:
            if name not in self.topics:
                self.topics[name] = [Partition(partition_id=i) for i in range(num_partitions)]

    def produce(self, topic: str, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        partitions: list[Partition] = self.topics[topic]
        pid = (hash(key) if key else 0) % len(partitions)
        return partitions[pid].append(key, value)

    def poll(self, group: str, topic: str, partition: int, max_records: int = 2) -> list[KafkaRecord]:
        offset = self.consumer_offsets[group][topic][partition]
        return self.topics[topic][partition].read_from(offset, max_records)

    def commit(self, group: str, topic: str, partition: int, offset: int) -> None:
        self.consumer_offsets[group][topic][partition] = offset


def assign_partitions(members: list[str], num_partitions: int) -> dict[str, list[int]]:
    assignment: dict[str, list[int]] = {m: [] for m in members}
    for pid in range(num_partitions):
        assignment[members[pid % len(members)]].append(pid)
    return assignment


def rebalance(
    broker: InMemoryKafkaBroker,
    group: str,
    topic: str,
    dead_member: str,
    remaining_members: list[str],
    current_assignment: dict[str, list[int]],
) -> dict[str, list[int]]:
    """
    Reassign dead member's partitions to remaining members.
    The replacement consumer will start from the dead member's last committed offset.
    No data needs to move — offsets are already stored in broker.consumer_offsets.
    """
    orphaned = current_assignment.pop(dead_member, [])
    logger.info("Rebalance triggered: '%s' died, orphaned partitions: %s", dead_member, orphaned)

    for i, pid in enumerate(orphaned):
        new_owner = remaining_members[i % len(remaining_members)]
        current_assignment[new_owner].append(pid)
        committed = broker.consumer_offsets[group][topic][pid]
        logger.info(
            "  partition %d → '%s' (will resume from committed offset %d)",
            pid,
            new_owner,
            committed,
        )

    logger.info("New assignment: %s", current_assignment)
    return current_assignment


def consumer_worker(
    broker: InMemoryKafkaBroker,
    group: str,
    topic: str,
    partitions: list[int],
    consumer_id: str,
    stop_after: int | None = None,
    processed_log: list[tuple[str, int, int]] | None = None,
) -> None:
    """
    Poll → process → commit loop.
    stop_after: if set, exit after processing this many records (simulates crash).
    processed_log: shared list for recording (consumer_id, partition, offset).
    """
    processed = 0
    exhausted: set[int] = set()

    while len(exhausted) < len(partitions):
        for pid in partitions:
            if pid in exhausted:
                continue
            records = broker.poll(group, topic, pid)
            if not records:
                exhausted.add(pid)
                continue
            for rec in records:
                if stop_after is not None and processed >= stop_after:
                    logger.info("[%s] stopping early after %d records", consumer_id, processed)
                    return
                logger.info(
                    "[%s] partition=%d offset=%d key=%s",
                    consumer_id, rec.partition, rec.offset, rec.key,
                )
                if processed_log is not None:
                    processed_log.append((consumer_id, rec.partition, rec.offset))
                processed += 1
            broker.commit(group, topic, pid, records[-1].offset + 1)


def main() -> None:
    broker = InMemoryKafkaBroker()
    TOPIC = "events"
    GROUP = "test-group"
    NUM_PARTITIONS = 3

    broker.create_topic(TOPIC, NUM_PARTITIONS)

    # Produce 3 messages per partition (9 total), 1 key per partition
    for i in range(9):
        key = f"user-{i % 3}"  # 3 distinct keys → 3 partitions
        broker.produce(TOPIC, key=key, value={"seq": i})

    members = ["consumer-0", "consumer-1", "consumer-2"]
    assignment = assign_partitions(members, NUM_PARTITIONS)
    logger.info("Initial assignment: %s", assignment)

    processed_log: list[tuple[str, int, int]] = []

    # consumer-0 processes 1 record then "dies"
    t0 = threading.Thread(
        target=consumer_worker,
        args=(broker, GROUP, TOPIC, assignment["consumer-0"], "consumer-0"),
        kwargs={"stop_after": 1, "processed_log": processed_log},
        name="consumer-0",
    )
    t1 = threading.Thread(
        target=consumer_worker,
        args=(broker, GROUP, TOPIC, assignment["consumer-1"], "consumer-1"),
        kwargs={"processed_log": processed_log},
        name="consumer-1",
    )
    t2 = threading.Thread(
        target=consumer_worker,
        args=(broker, GROUP, TOPIC, assignment["consumer-2"], "consumer-2"),
        kwargs={"processed_log": processed_log},
        name="consumer-2",
    )

    t0.start(); t1.start(); t2.start()
    t0.join(); t1.join(); t2.join()

    # Rebalance: reassign consumer-0's remaining partitions
    assignment = rebalance(
        broker, GROUP, TOPIC,
        dead_member="consumer-0",
        remaining_members=["consumer-1", "consumer-2"],
        current_assignment=assignment,
    )

    # Drain remaining messages
    t1b = threading.Thread(
        target=consumer_worker,
        args=(broker, GROUP, TOPIC, assignment["consumer-1"], "consumer-1-post-rebalance"),
        kwargs={"processed_log": processed_log},
        name="consumer-1-post",
    )
    t2b = threading.Thread(
        target=consumer_worker,
        args=(broker, GROUP, TOPIC, assignment["consumer-2"], "consumer-2-post-rebalance"),
        kwargs={"processed_log": processed_log},
        name="consumer-2-post",
    )
    t1b.start(); t2b.start()
    t1b.join(); t2b.join()

    # Verify: each (partition, offset) pair must appear exactly once
    seen: dict[tuple[int, int], str] = {}
    duplicates = 0
    for consumer_id, partition, offset in processed_log:
        key = (partition, offset)
        if key in seen:
            logger.error("DUPLICATE: partition=%d offset=%d seen by %s AND %s", partition, offset, seen[key], consumer_id)
            duplicates += 1
        else:
            seen[key] = consumer_id

    total_records = sum(len(broker.topics[TOPIC][p].log) for p in range(NUM_PARTITIONS))
    logger.info("Total records produced: %d", total_records)
    logger.info("Total records processed: %d", len(seen))
    logger.info("Duplicates: %d", duplicates)

    assert duplicates == 0, "FAIL: duplicate processing detected"
    assert len(seen) == total_records, f"FAIL: processed {len(seen)} of {total_records}"
    logger.info("PASS: all %d records processed exactly once across rebalance.", total_records)


if __name__ == "__main__":
    main()
```

---

## Exercise 3 — Log compaction: keep only latest message per key

**Problem**: Implement a `compact(broker, topic_name)` function that, for each partition, retains only the latest record per key. Run it and verify partition sizes shrink while the latest state per key is preserved.

**Explanation**: Log compaction is Kafka's mechanism for changelog topics. Instead of time-based deletion it scans each partition and, for every key, discards all records except the one with the highest offset. The compacted log is smaller but still contains the current state of every key that was ever written. A tombstone (value=`null`) deletes a key from the compacted log entirely.

```python
"""
ex3_log_compaction.py

Implements log compaction on an in-memory partition.
After compaction: exactly one record per key (the latest), in offset order.

Run: python ex3_log_compaction.py
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class KafkaRecord(BaseModel):
    key: str | None
    value: dict[str, Any] | None  # None = tombstone
    partition: int
    offset: int

    @field_validator("offset")
    @classmethod
    def offset_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("offset must be >= 0")
        return v


@dataclass
class Partition:
    partition_id: int
    log: list[KafkaRecord] = field(default_factory=list)

    def append(self, key: str | None, value: dict[str, Any] | None) -> KafkaRecord:
        offset = len(self.log)
        record = KafkaRecord(key=key, value=value, partition=self.partition_id, offset=offset)
        self.log.append(record)
        return record


@dataclass
class InMemoryKafkaBroker:
    topics: dict[str, list[Partition]] = field(default_factory=dict)

    def create_topic(self, name: str, num_partitions: int) -> None:
        if name not in self.topics:
            self.topics[name] = [Partition(partition_id=i) for i in range(num_partitions)]

    def produce(self, topic: str, key: str | None, value: dict[str, Any] | None) -> KafkaRecord:
        partitions = self.topics[topic]
        pid = (hash(key) if key else 0) % len(partitions)
        return partitions[pid].append(key, value)


def compact(broker: InMemoryKafkaBroker, topic_name: str) -> dict[int, int]:
    """
    Perform log compaction on all partitions of topic_name.

    Algorithm (O(n) per partition):
    1. Scan the log forward, tracking latest_offset[key].
    2. Build a new log containing only records at their latest offset.
    3. Tombstones (value=None) remove the key entirely — omit from compacted log.
    4. Offsets in the compacted log are rewritten to be contiguous (simulates
       Kafka's behaviour where compacted segments get new sequential offsets).

    Returns dict[partition_id -> records_removed].
    """
    removed: dict[int, int] = {}

    for partition in broker.topics[topic_name]:
        original_size = len(partition.log)

        # Pass 1: find the latest offset for each key
        latest_offset: dict[str | None, int] = {}
        for record in partition.log:
            latest_offset[record.key] = record.offset

        # Pass 2: retain only records at their latest offset, drop tombstones
        compacted: list[KafkaRecord] = []
        new_offset = 0
        for record in partition.log:
            if record.offset != latest_offset[record.key]:
                continue  # older version — discard
            if record.value is None:
                continue  # tombstone — key is deleted
            # Rewrite offset to be contiguous in the compacted log
            compacted.append(KafkaRecord(
                key=record.key,
                value=record.value,
                partition=partition.partition_id,
                offset=new_offset,
            ))
            new_offset += 1

        records_removed = original_size - len(compacted)
        removed[partition.partition_id] = records_removed

        partition.log = compacted
        logger.info(
            "Partition %d: compacted %d → %d records (%d removed)",
            partition.partition_id,
            original_size,
            len(compacted),
            records_removed,
        )

    return removed


def main() -> None:
    broker = InMemoryKafkaBroker()
    broker.create_topic("user-profiles", num_partitions=1)

    # Simulate multiple updates to the same keys
    updates = [
        ("user-1", {"name": "Alice", "version": 1}),
        ("user-2", {"name": "Bob", "version": 1}),
        ("user-1", {"name": "Alice Smith", "version": 2}),   # update
        ("user-3", {"name": "Carol", "version": 1}),
        ("user-2", {"name": "Bob Jones", "version": 2}),     # update
        ("user-1", {"name": "Alice Smith", "version": 3}),   # update again
        ("user-4", {"name": "Dave", "version": 1}),
        ("user-3", None),                                     # tombstone — delete user-3
        ("user-2", {"name": "Bob Jones", "version": 3}),     # update again
    ]

    for key, value in updates:
        broker.produce("user-profiles", key=key, value=value)

    partitions = broker.topics["user-profiles"]
    logger.info("Before compaction: %d records in partition 0", len(partitions[0].log))

    for rec in partitions[0].log:
        logger.info("  offset=%d key=%s value=%s", rec.offset, rec.key, rec.value)

    removed = compact(broker, "user-profiles")

    logger.info("\nAfter compaction: %d records in partition 0", len(partitions[0].log))
    for rec in partitions[0].log:
        logger.info("  offset=%d key=%s value=%s", rec.offset, rec.key, rec.value)

    # Verify: each key appears at most once, latest value preserved
    keys_in_compacted = {rec.key: rec.value for rec in partitions[0].log}

    assert "user-1" in keys_in_compacted and keys_in_compacted["user-1"]["version"] == 3
    assert "user-2" in keys_in_compacted and keys_in_compacted["user-2"]["version"] == 3
    assert "user-3" not in keys_in_compacted, "tombstoned key must be absent"
    assert "user-4" in keys_in_compacted
    logger.info("PASS: compaction preserved latest state per key, removed tombstones.")


if __name__ == "__main__":
    main()
```

---

## Exercise 4 — Replication simulation: primary + N replicas, reads from primary

**Problem**: Extend `Partition` to hold a `replicas: list[list[KafkaRecord]]` list. When `produce()` is called, write to the leader log and sync all followers. Simulate leader failure by marking one partition's leader as unavailable and promoting the first follower as the new leader.

**Explanation**: In Kafka every write goes to the partition leader. Followers pull from the leader asynchronously. `acks=all` means the producer waits until all ISR replicas acknowledge. On leader failure, the group coordinator elects a new leader from the ISR. The new leader has every record the old leader had (because `acks=all` was used).

```python
"""
ex4_replication.py

Simulates Kafka replication: leader + N follower replicas.
- All writes go through the leader.
- Followers are synchronously synced (simulates acks=all).
- Leader failure triggers ISR election.

Run: python ex4_replication.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class KafkaRecord(BaseModel):
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


@dataclass
class ReplicatedPartition:
    """
    A partition with a configurable replication factor.

    Attributes:
        partition_id: Identifier for this partition.
        replication_factor: Total replica count (leader + followers).
        leader_index: Which replica index is currently the leader.
        replicas: List of replica logs (index 0 = initial leader).
        failed_replicas: Set of replica indices that have "failed" (unavailable).
    """

    partition_id: int
    replication_factor: int = 3
    leader_index: int = 0
    replicas: list[list[KafkaRecord]] = field(default_factory=list)
    failed_replicas: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.replicas = [[] for _ in range(self.replication_factor)]

    @property
    def leader_log(self) -> list[KafkaRecord]:
        return self.replicas[self.leader_index]

    def isr(self) -> list[int]:
        """In-sync replica set: all non-failed replicas."""
        return [i for i in range(self.replication_factor) if i not in self.failed_replicas]

    def append(self, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        """
        Append to leader, then replicate to all ISR followers (acks=all semantics).
        Raises RuntimeError if the leader has failed.
        """
        if self.leader_index in self.failed_replicas:
            raise RuntimeError(
                f"Leader replica {self.leader_index} has failed. Call elect_new_leader() first."
            )

        offset = len(self.leader_log)
        record = KafkaRecord(
            key=key,
            value=value,
            partition=self.partition_id,
            offset=offset,
        )
        self.leader_log.append(record)

        # Replicate to followers in ISR
        for replica_index in self.isr():
            if replica_index == self.leader_index:
                continue
            self.replicas[replica_index].append(record)
            logger.debug(
                "Replicated to replica %d: partition=%d offset=%d",
                replica_index,
                self.partition_id,
                offset,
            )

        return record

    def read(self, offset: int, max_records: int = 10) -> list[KafkaRecord]:
        """Reads always go to the leader."""
        if self.leader_index in self.failed_replicas:
            raise RuntimeError("Leader has failed. Reads unavailable until new leader elected.")
        return self.leader_log[offset : offset + max_records]

    def fail_leader(self) -> int:
        """Mark the current leader as failed. Returns failed leader index."""
        failed = self.leader_index
        self.failed_replicas.add(failed)
        logger.warning(
            "Partition %d: leader replica %d has FAILED (unavailable).",
            self.partition_id,
            failed,
        )
        return failed

    def elect_new_leader(self) -> int:
        """
        Elect a new leader from ISR (first available non-failed replica).
        Raises RuntimeError if ISR is empty (all replicas failed — data loss scenario).
        """
        candidates = self.isr()
        if not candidates:
            raise RuntimeError(
                f"Partition {self.partition_id}: no ISR members available — unrecoverable."
            )
        new_leader = candidates[0]
        old_leader = self.leader_index
        self.leader_index = new_leader
        logger.info(
            "Partition %d: elected replica %d as new leader (was %d). ISR: %s",
            self.partition_id,
            new_leader,
            old_leader,
            candidates,
        )
        return new_leader


def main() -> None:
    partition = ReplicatedPartition(partition_id=0, replication_factor=3)

    # Produce 5 records
    for i in range(5):
        rec = partition.append(key=f"key-{i}", value={"seq": i})
        logger.info(
            "Produced: partition=%d offset=%d replicas=%d",
            rec.partition,
            rec.offset,
            partition.replication_factor,
        )

    # Verify all replicas have the same log
    for idx, replica_log in enumerate(partition.replicas):
        assert len(replica_log) == 5, f"Replica {idx} has {len(replica_log)} records, expected 5"
        logger.info("Replica %d: %d records (in sync)", idx, len(replica_log))

    logger.info("=== Simulating leader failure ===")
    old_leader = partition.fail_leader()

    # Reads now fail
    try:
        partition.read(0)
        assert False, "Should have raised"
    except RuntimeError as e:
        logger.info("Read on failed leader correctly raised: %s", e)

    # Elect new leader
    new_leader = partition.elect_new_leader()
    assert new_leader != old_leader

    # Reads from new leader work and return same data
    records_from_new_leader = partition.read(0, max_records=10)
    assert len(records_from_new_leader) == 5
    logger.info("New leader (replica %d) serves %d records correctly.", new_leader, len(records_from_new_leader))

    # Produce more records after failover
    partition.append(key="key-5", value={"seq": 5})
    assert len(partition.leader_log) == 6
    logger.info("PASS: leader failover succeeded, new records accepted by replica %d.", new_leader)


if __name__ == "__main__":
    main()
```

---

## Exercise 5 — Consumer lag calculation

**Problem**: After the first consumer group finishes, produce 3 more messages. Add `consumer_lag(broker, group_id, topic_name)` returning `{partition_id: lag}` where lag = `partition_log_length - committed_offset`. Print lag before and after the second consumer group drains it.

**Explanation**: Consumer lag is the most important operational metric in Kafka. A lag of N on partition P means the consumer group has N unconsumed records. Lag > 0 after a consumer has run means the consumer is either behind or new records arrived after the consumer finished. A continuously growing lag means the consumer is slower than the producer.

```python
"""
ex5_consumer_lag.py

Demonstrates consumer lag calculation.
Lag per partition = latest_offset - committed_offset for the group.

Run: python ex5_consumer_lag.py
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


class KafkaRecord(BaseModel):
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


@dataclass
class Partition:
    partition_id: int
    log: list[KafkaRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        with self._lock:
            offset = len(self.log)
            record = KafkaRecord(key=key, value=value, partition=self.partition_id, offset=offset)
            self.log.append(record)
            return record

    def read_from(self, offset: int, max_records: int = 10) -> list[KafkaRecord]:
        with self._lock:
            return self.log[offset : offset + max_records]

    @property
    def latest_offset(self) -> int:
        with self._lock:
            return len(self.log)


@dataclass
class InMemoryKafkaBroker:
    topics: dict[str, list[Partition]] = field(default_factory=dict)
    consumer_offsets: dict[str, dict[str, dict[int, int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def create_topic(self, name: str, num_partitions: int) -> None:
        with self._lock:
            if name not in self.topics:
                self.topics[name] = [Partition(partition_id=i) for i in range(num_partitions)]

    def produce(self, topic: str, key: str | None, value: dict[str, Any]) -> KafkaRecord:
        partitions = self.topics[topic]
        pid = (hash(key) if key else 0) % len(partitions)
        return partitions[pid].append(key, value)

    def poll(self, group: str, topic: str, partition: int, max_records: int = 5) -> list[KafkaRecord]:
        offset = self.consumer_offsets[group][topic][partition]
        return self.topics[topic][partition].read_from(offset, max_records)

    def commit(self, group: str, topic: str, partition: int, offset: int) -> None:
        self.consumer_offsets[group][topic][partition] = offset


def consumer_lag(
    broker: InMemoryKafkaBroker,
    group_id: str,
    topic_name: str,
) -> dict[int, int]:
    """
    Calculate per-partition consumer lag for a group.

    lag[partition_id] = latest_offset - committed_offset

    A lag of 0 means the consumer is caught up.
    A positive lag means unread records exist.
    """
    partitions = broker.topics[topic_name]
    lag: dict[int, int] = {}
    for partition in partitions:
        latest = partition.latest_offset
        committed = broker.consumer_offsets[group_id][topic_name][partition.partition_id]
        lag[partition.partition_id] = latest - committed
    return lag


def consumer_worker(
    broker: InMemoryKafkaBroker,
    group: str,
    topic: str,
    partitions: list[int],
    consumer_id: str,
) -> None:
    exhausted: set[int] = set()
    while len(exhausted) < len(partitions):
        for pid in partitions:
            if pid in exhausted:
                continue
            records = broker.poll(group, topic, pid)
            if not records:
                exhausted.add(pid)
                continue
            for rec in records:
                logger.debug("[%s] partition=%d offset=%d", consumer_id, rec.partition, rec.offset)
            broker.commit(group, topic, pid, records[-1].offset + 1)


def assign_partitions(members: list[str], num_partitions: int) -> dict[str, list[int]]:
    assignment: dict[str, list[int]] = {m: [] for m in members}
    for pid in range(num_partitions):
        assignment[members[pid % len(members)]].append(pid)
    return assignment


def main() -> None:
    broker = InMemoryKafkaBroker()
    TOPIC = "orders"
    GROUP_1 = "analytics-group"
    GROUP_2 = "audit-group"
    NUM_PARTITIONS = 3

    broker.create_topic(TOPIC, NUM_PARTITIONS)

    # Produce 12 messages with 3 distinct keys → spread across partitions
    for i in range(12):
        broker.produce(TOPIC, key=f"user-{i % 3}", value={"seq": i})

    # Lag before any consumer has run (all records unconsumed)
    lag_before = consumer_lag(broker, GROUP_1, TOPIC)
    logger.info("Lag before consumer group 1 runs: %s", lag_before)
    total_records = sum(lag_before.values())
    assert total_records == 12, f"Expected lag=12, got {total_records}"

    # Run consumer group 1
    members = ["consumer-A", "consumer-B"]
    assignment = assign_partitions(members, NUM_PARTITIONS)
    threads = [
        threading.Thread(
            target=consumer_worker,
            args=(broker, GROUP_1, TOPIC, assignment[m], m),
            name=m,
        )
        for m in members
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lag_after_group1 = consumer_lag(broker, GROUP_1, TOPIC)
    logger.info("Lag after consumer group 1 finishes: %s", lag_after_group1)
    assert all(v == 0 for v in lag_after_group1.values()), "Group 1 should be fully caught up"

    # Produce 3 more messages — creates lag again for group 1
    for i in range(3):
        broker.produce(TOPIC, key=f"user-{i}", value={"seq": 100 + i})

    lag_new_records = consumer_lag(broker, GROUP_1, TOPIC)
    logger.info("Lag after 3 new records produced (group 1): %s", lag_new_records)
    assert sum(lag_new_records.values()) == 3

    # Group 2 has lag = all 15 records (it never consumed anything)
    lag_group2 = consumer_lag(broker, GROUP_2, TOPIC)
    logger.info("Lag for group 2 (never consumed): %s", lag_group2)
    assert sum(lag_group2.values()) == 15

    # Run group 2 to drain all 15 records
    assignment2 = assign_partitions(["auditor"], NUM_PARTITIONS)
    audit_threads = [
        threading.Thread(
            target=consumer_worker,
            args=(broker, GROUP_2, TOPIC, assignment2[m], m),
            name=m,
        )
        for m in assignment2
    ]
    for t in audit_threads:
        t.start()
    for t in audit_threads:
        t.join()

    lag_group2_after = consumer_lag(broker, GROUP_2, TOPIC)
    logger.info("Lag for group 2 after draining: %s", lag_group2_after)
    assert all(v == 0 for v in lag_group2_after.values())
    logger.info("PASS: consumer lag calculation correct before and after consumption.")


if __name__ == "__main__":
    main()
```

---

## Summary

| Exercise | Core concept | Key insight |
|---|---|---|
| 1 | Partition key routing | `hash(key) % n` is deterministic within a process |
| 2 | Consumer group rebalance | Resume from committed offset — no loss, no duplicates |
| 3 | Log compaction | Keep latest record per key; tombstones delete keys |
| 4 | Replication | Leader handles all I/O; ISR ensures durability; failover from ISR only |
| 5 | Consumer lag | `latest_offset - committed_offset`; zero = caught up |
