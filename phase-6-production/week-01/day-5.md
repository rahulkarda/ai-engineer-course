# Phase 6 · Week 01 · Day 5 — Failure Modes, Time & Ordering, Idempotency

> Objective: Reason about partial failures and network partitions, use logical clocks for causality, and design idempotent operations.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Failure modes in distributed systems

Distributed systems fail in qualitatively different ways from single-process systems.

**Crash-stop failure**: node stops and never recovers. Clean. Detectable by timeout.

**Crash-recovery failure**: node crashes but may restart. State on disk may be stale. Most real systems.

**Omission failure**: node is running but drops some messages (send omission) or ignores some receives (receive omission). Very common — overwhelmed queues, GC pauses, full buffers.

**Timing failure**: node is too slow (misses a real-time deadline). Common under high load.

**Byzantine failure**: node behaves arbitrarily — sends incorrect messages, lies about state. Rare in internal data-center systems; relevant for blockchain and adversarial environments.

### Partial failure

**Partial failure** is the defining characteristic of distributed systems: some components are up, some are down, and from the outside you often cannot tell which. A single-machine program cannot be "half crashed" — a distributed system routinely is.

Concrete partial failure scenarios:

```
Request sent ──▶ Server  (server processes, but response lost)
                              ↑
Client times out ─────────── did the server do the work or not?
```

This ambiguity is fundamental. It means **at-most-once semantics** (don't retry) risks losing the result; **at-least-once semantics** (retry on timeout) risks duplicating the operation. The solution is **idempotency**.

### Network partitions

A network partition splits the cluster into groups that cannot reach each other. Within each group, nodes appear healthy. From across the partition, they appear dead.

Partitions can last milliseconds (transient packet loss) or hours (misconfigured firewall, datacenter link failure). You must design for both.

AI system partition scenario:
- An agent orchestrator loses contact with the tool-execution service
- The orchestrator times out and retries the tool call via a fallback path
- Meanwhile the original tool call succeeds — the tool has been executed twice

### Time and ordering

Physical clocks drift. Two nodes cannot agree on "what time it is now" to better than ~1 ms on a LAN, or ~100 ms across the internet. This breaks any system that relies on timestamps for ordering.

**Logical clocks** solve this by tracking *causality* rather than wall-clock time.

#### Lamport timestamps

Each node maintains a counter `L`. Rules:
1. Increment `L` before each event
2. When sending a message, attach `L`
3. On receiving a message with timestamp `T`: set `L = max(L, T) + 1`

Lamport timestamps give a **partial order**: if event A caused event B (directly or transitively), then `L(A) < L(B)`. The converse is *not* guaranteed — `L(A) < L(B)` does not imply A caused B.

#### Vector clocks

A vector clock is a vector of counters, one per node. Rules:
1. On an internal event: increment your own component
2. On send: attach the full vector after incrementing your component
3. On receive from node j with vector V: set each component to `max(local[i], V[i])`, then increment your own component

Vector clocks allow you to determine:
- `V_A < V_B` (element-wise): A causally precedes B
- Neither dominates: A and B are **concurrent** (no causal relationship)

This is exactly the information needed to detect write conflicts (Day 2's "version vectors" are vector clocks).

### Idempotency

An operation is **idempotent** if applying it multiple times produces the same result as applying it once.

Examples:
- `PUT /users/123 {name: "Alice"}` — idempotent (sets to known state)
- `POST /orders {item: "book"}` — NOT idempotent (creates a new order each time)
- `DELETE /cache/key` — idempotent (deleting an already-deleted key is a no-op)
- `INSERT INTO events ...` — NOT idempotent by default

### Making operations idempotent

**Idempotency keys**: client generates a unique ID per logical operation. Server stores processed IDs. On duplicate: return stored result without re-executing.

```python
# Client
key = str(uuid.uuid4())
response = post("/tool/call", json={"idempotency_key": key, "params": ...})

# Server
if key in processed_keys:
    return processed_keys[key]
result = execute_tool(params)
processed_keys[key] = result
return result
```

**Natural idempotency patterns for AI**:
- Vector upserts (upsert = insert-or-update) are idempotent; pure inserts are not
- Embedding generation from the same text is deterministic — re-running is safe
- LLM generation is non-deterministic — cache the result keyed by (prompt, params, idempotency_key)
- Agent state checkpoints should be upserts keyed by (session_id, step_index)

## How it works internally

### The two-generals problem

Two generals must coordinate an attack over unreliable messengers. No finite exchange of messages can guarantee both generals attack simultaneously. This is a fundamental impossibility: you cannot achieve agreement over an unreliable channel with certainty.

This is why distributed systems use **probabilistic guarantees** (retry with idempotency) rather than perfect certainty.

### Happens-before relation

Leslie Lamport defined `a → b` ("a happens-before b") as:
- Same process: a comes before b in execution order
- Message passing: a is the send event, b is the receive event of the same message
- Transitivity: if `a → b` and `b → c` then `a → c`

Events where neither `a → b` nor `b → a` are **concurrent** — they could have happened in any order or simultaneously.

## Worked example

```python
"""
day5_logical_clocks.py — Lamport timestamps and vector clocks.

Simulates three nodes exchanging messages and tracks causality.

Run: python day5_logical_clocks.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import NamedTuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-10s %(message)s",
)


# ── Lamport Clock ─────────────────────────────────────────────────────────────

class LamportNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.clock: int = 0
        self.log = logging.getLogger(f"lamport.{node_id}")

    def _tick(self) -> int:
        self.clock += 1
        return self.clock

    def local_event(self, label: str) -> int:
        ts = self._tick()
        self.log.info("LOCAL  event=%s  ts=%d", label, ts)
        return ts

    def send(self, label: str) -> tuple[str, int]:
        ts = self._tick()
        self.log.info("SEND   event=%s  ts=%d", label, ts)
        return (label, ts)

    def receive(self, message: str, sender_ts: int) -> int:
        self.clock = max(self.clock, sender_ts) + 1
        self.log.info("RECV   msg=%s  sender_ts=%d  local_ts=%d", message, sender_ts, self.clock)
        return self.clock


# ── Vector Clock ─────────────────────────────────────────────────────────────

class VectorClock(NamedTuple):
    values: tuple[int, ...]  # one entry per node, indexed by node index

    def increment(self, node_idx: int) -> "VectorClock":
        v = list(self.values)
        v[node_idx] += 1
        return VectorClock(tuple(v))

    def merge(self, other: "VectorClock", node_idx: int) -> "VectorClock":
        merged = tuple(max(a, b) for a, b in zip(self.values, other.values))
        v = list(merged)
        v[node_idx] += 1
        return VectorClock(tuple(v))

    def happens_before(self, other: "VectorClock") -> bool:
        return all(a <= b for a, b in zip(self.values, other.values)) and self != other

    def concurrent_with(self, other: "VectorClock") -> bool:
        return not self.happens_before(other) and not other.happens_before(self)


@dataclass
class VectorClockNode:
    node_id: str
    node_idx: int
    num_nodes: int
    vc: VectorClock = field(init=False)
    _log: logging.Logger = field(init=False)

    def __post_init__(self) -> None:
        self.vc = VectorClock(tuple(0 for _ in range(self.num_nodes)))
        self._log = logging.getLogger(f"vc.{self.node_id}")

    def local_event(self, label: str) -> VectorClock:
        self.vc = self.vc.increment(self.node_idx)
        self._log.info("LOCAL  event=%s  vc=%s", label, list(self.vc.values))
        return self.vc

    def send(self, label: str) -> tuple[str, VectorClock]:
        self.vc = self.vc.increment(self.node_idx)
        self._log.info("SEND   event=%s  vc=%s", label, list(self.vc.values))
        return (label, self.vc)

    def receive(self, message: str, sender_vc: VectorClock) -> VectorClock:
        self.vc = self.vc.merge(sender_vc, self.node_idx)
        self._log.info("RECV   msg=%s  vc=%s", message, list(self.vc.values))
        return self.vc


def demonstrate_lamport() -> None:
    log = logging.getLogger("demo.lamport")
    log.info("=== Lamport Timestamp Demo ===")
    a = LamportNode("A")
    b = LamportNode("B")
    c = LamportNode("C")

    a.local_event("write_embedding")
    msg, ts = a.send("notify_index_ready")
    b.receive("notify_index_ready", ts)
    b.local_event("start_query")
    msg2, ts2 = b.send("query_result")
    c.receive("query_result", ts2)
    c.local_event("aggregate")


def demonstrate_vector_clocks() -> None:
    log = logging.getLogger("demo.vc")
    log.info("\n=== Vector Clock Demo ===")
    num_nodes = 3
    a = VectorClockNode("A", 0, num_nodes)
    b = VectorClockNode("B", 1, num_nodes)
    c = VectorClockNode("C", 2, num_nodes)

    # A writes, tells B
    a.local_event("write_x=1")
    msg_a, vc_a = a.send("replicate_x=1")
    b.receive("replicate_x=1", vc_a)

    # Concurrently, C writes independently
    c.local_event("write_x=2")

    # Now B and C compare their vector clocks
    log.info(
        "B.vc=%s  C.vc=%s  B_before_C=%s  concurrent=%s",
        list(b.vc.values),
        list(c.vc.values),
        b.vc.happens_before(c.vc),
        b.vc.concurrent_with(c.vc),
    )


if __name__ == "__main__":
    demonstrate_lamport()
    demonstrate_vector_clocks()
```

## Exercises

1. In the Lamport demo, after B receives A's message at ts=2, B does a local event (ts=3) and sends to C. Manually trace what C's clock becomes. Verify with code.

2. Create a scenario with three `VectorClockNode` instances where node A and node B perform concurrent writes. Show that `a.vc.concurrent_with(b.vc)` returns `True`. Explain why this means a conflict-detection system must ask the application to resolve the conflict.

3. Implement an `IdempotentToolExecutor` class: it takes a tool function `fn(params) -> result`, stores results by `idempotency_key` in a dict, and never calls `fn` twice for the same key. Write a pytest test that calls the executor twice with the same key and asserts the underlying function was only called once (use `unittest.mock.MagicMock`).

4. A distributed agent framework retries failed tool calls with exponential backoff but does NOT use idempotency keys. Describe three specific failure scenarios this creates in an e-commerce agent that calls a `charge_card(amount, card_id)` tool.

(Solutions: ../solutions/week-01/day-5.md)

## Self-check

- [ ] I can classify failure modes (crash-stop, crash-recovery, omission, Byzantine) and give one AI example of each
- [ ] I can explain why physical clocks are insufficient for ordering in distributed systems
- [ ] I can implement Lamport timestamps and trace a multi-node message exchange
- [ ] I can use vector clocks to determine whether two events are causally ordered or concurrent
- [ ] I can implement an idempotency key pattern and explain its retry semantics

## Supplementary reading

- Designing Data-Intensive Applications (Kleppmann) — Chapter 8: The Trouble with Distributed Systems, Chapter 9: Consistency and Consensus
- [Time, Clocks, and the Ordering of Events in a Distributed System](https://lamport.azurewebsites.net/pubs/time-clocks.pdf) — Leslie Lamport, 1978
- [Dynamo: Amazon's Highly Available Key-value Store](https://dl.acm.org/doi/10.1145/1294261.1294281) — DeCandia et al., SOSP 2007 (vector clocks in practice)
