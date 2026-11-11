# Phase 6 · Week 01 · Day 7 — Project Completion + Assessment Prep

> Objective: Finalize the distributed KV store project, run all tests, and consolidate the week's concepts for the assessment.  ·  Time: ~3 hr project + 1 hr review  ·  Prereqs: [[day-6]]

## Project finalization (3 hr)

### Completion checklist

Work through each item. A checked box means "done AND tested."

**Core implementation**
- [ ] `src/models.py` — `Entry`, `WriteRequest`, `ReadResult`, `ReplicationMessage` all defined with Pydantic v2, full type hints
- [ ] `src/network_sim.py` — `NetworkSim` with drop_rate, partition pairs, stats
- [ ] `src/node.py` — `KVNode` with idempotent writes, LWW replication, reads
- [ ] `src/cluster.py` — `Cluster` with write, read, quorum_read, sync_replicas

**Tests (all green)**
- [ ] `tests/test_replication.py` — write reaches primary, replicated to both replicas, idempotency, stale-on-drop
- [ ] `tests/test_consistency.py` — stale eventual convergence, quorum read correctness, LWW conflict resolution
- [ ] `tests/test_partition.py` — primary available during partition, convergence after heal, stats accuracy

**Code quality**
- [ ] `ruff check src/ tests/` passes clean
- [ ] No `print()` — only `logging`
- [ ] No hardcoded secrets or model IDs
- [ ] `pyproject.toml` defines project metadata and dev dependencies

### Running the project

```bash
# From the week-01-kvstore/ directory
python -m pip install -e ".[dev]"
python -m pytest tests/ -v --tb=short
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Expected output: all 10 tests pass, coverage ≥ 85%.

### Integration smoke test

Write a short `scripts/smoke.py` that exercises the cluster end-to-end without pytest:

```python
"""
scripts/smoke.py — End-to-end smoke test for the distributed KV store.
Run: python scripts/smoke.py
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)-15s %(message)s")
log = logging.getLogger("smoke")

sys.path.insert(0, "src")  # adjust if running from project root

from src.cluster import Cluster
from src.network_sim import NetworkSim


def main() -> None:
    log.info("=== Scenario 1: healthy cluster ===")
    cluster = Cluster()
    cluster.write("model", "gpt-embedding-v3")
    for node in cluster.node_ids():
        r = cluster.read("model", node=node)
        log.info("  %-12s → %s (stale=%s)", node, r.value, r.is_stale)

    log.info("\n=== Scenario 2: partition + stale replica ===")
    net = NetworkSim()
    cluster2 = Cluster(network=net)
    net.partition("primary", "replica-1")
    cluster2.write("config", "v2")
    r_primary = cluster2.read("config", node="primary")
    r_stale = cluster2.read("config", node="replica-1")
    log.info("  primary   → %s", r_primary.value)
    log.info("  replica-1 → %s (partition)", r_stale.value)

    log.info("\n=== Scenario 3: heal + sync ===")
    net.heal_all()
    cluster2.sync_replicas()
    for node in cluster2.node_ids():
        r = cluster2.read("config", node=node)
        log.info("  %-12s → %s", node, r.value)

    log.info("\n=== Scenario 4: quorum read under partial replication ===")
    cluster3 = Cluster(network=NetworkSim(drop_rate=0.6))
    for _ in range(15):
        cluster3.write("index_ts", "2024-11-01")
    r = cluster3.quorum_read("index_ts")
    log.info("  quorum_read → %s (from %s)", r.value, r.from_node)
    assert r.value == "2024-11-01", "Quorum read must return latest value"

    log.info("\nAll smoke scenarios passed.")


if __name__ == "__main__":
    main()
```

## Week 1 concept review (1 hr)

Work through each concept. Use the self-check as a guide — write brief answers (3–5 sentences each) from memory, then verify against the day's lesson.

### Core concepts review

**Why distributed?**
Summarise the three core motivations (capacity, availability, latency) in your own words. Give one concrete example of each from an AI/ML workload you've seen or can imagine.

**Fallacies**
Name all eight from memory. For each one you struggle to recall, re-read Day 1 and write a one-sentence concrete failure scenario.

**CAP**
Draw the CP/AP decision triangle. Place these systems on it: Zookeeper, Cassandra, Redis (single-node), your distributed KV store from Day 6.

**Consistency spectrum**
Rank these from strongest to weakest: causal, eventual, linearisable, sequential. For each, give the price paid (latency, availability, or complexity).

**Replication**
Explain the difference between synchronous and asynchronous replication. When would you use each? What does replication lag cause in an agentic pipeline?

**Partitioning**
Given a vector database with 100 M embeddings and a hotspot on recent documents, which partition strategy would you use? Why? What mitigation would you apply for the hotspot?

**Raft**
Without looking, draw the state machine (follower/candidate/leader) and write the two rules for granting a vote. What does "majority" mean in a 5-node cluster?

**Failure modes + time**
Explain why physical clocks are insufficient for ordering events. Sketch the rule for updating a Lamport clock on message receive.

**Idempotency**
Describe the idempotency key pattern. Which operations in your KV store simulation are naturally idempotent, and which required explicit idempotency logic?

### Common interview questions for this week's material

1. "Walk me through what happens in your distributed system when a network partition occurs."
2. "How would you design the state storage for a multi-agent system that needs to survive node failures?"
3. "Your vector database has one shard that's getting 80% of the writes. What's causing this and how do you fix it?"
4. "Why can't you just use timestamps to order events in a distributed system?"
5. "What's the difference between at-least-once and exactly-once delivery, and when does it matter?"

### Project debrief

Write a 1-page (≈300 word) technical summary of your KV store implementation covering:

1. **Design decisions**: Why did you choose last-write-wins over version vectors for conflict resolution? What did you sacrifice?
2. **Trade-off demonstrated**: Pick one test. Explain what distributed trade-off it exercises and why the test is structured as it is.
3. **What's missing**: List two features a production system would need that your simulation does not have.

## Self-check

- [ ] All 10 tests pass with `pytest -v`
- [ ] I can walk through each of the four test files and explain what trade-off it demonstrates
- [ ] I can name all eight fallacies and give an AI example for each
- [ ] I can explain the CAP theorem and classify my KV store
- [ ] I am ready for the Week 1 assessment

## Supplementary reading

- [Designing Data-Intensive Applications](https://dataintensive.net/) — Kleppmann (all of Part II: Distributed Data)
- [The Raft Paper](https://raft.github.io/raft.pdf)
- [An Introduction to Distributed Systems](https://github.com/aphyr/distsys-class) — Kyle Kingsbury
