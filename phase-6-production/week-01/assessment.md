# Week 01 Assessment — Distributed Systems Fundamentals

> Phase 6 · Week 01  ·  Time: ~1.5 hr  ·  Pass: ≥ 75%

## Conceptual (interview-style)

Answer each question in 3–8 sentences. Depth counts more than length.

1. **Fallacies**: A colleague is building an agent orchestration service and says "the tool servers are all on the same LAN, so latency and reliability aren't concerns." Which fallacies does this violate? What specific failure scenario would you show them to change their mind?

2. **CAP**: Your team is designing a distributed session store for a multi-agent pipeline. Agents need to read each other's state to coordinate. Should this store be CP or AP? Defend your choice. What does the other option sacrifice?

3. **Consistency models**: Explain the difference between eventual consistency and causal consistency using a concrete example involving two AI agents writing to a shared document. Why does the distinction matter?

4. **Replication lag**: An agent reads a user's preference vector from a replica immediately after a write to the primary. The replica is 200 ms behind. What does the agent see? Name two patterns that prevent this class of bug and explain how each works.

5. **Consistent hashing**: A vector database cluster has 3 shards using consistent hashing. You add a 4th shard. Approximately what fraction of the 10 M stored embeddings must be moved? How does this compare to naive modulo hashing? Why does the answer matter operationally?

6. **Raft**: A 5-node Raft cluster loses 2 nodes simultaneously. Can it still elect a leader and accept writes? What if it loses 3 nodes? Explain using the majority requirement.

7. **Logical clocks**: Two nodes A and B both write to key `user:42` concurrently during a network partition. Node A's Lamport timestamp is 15; node B's is 12. Last-write-wins says A's write survives. Is this correct? What does a vector clock reveal that Lamport timestamps cannot?

8. **Idempotency**: An agent calls a `charge_payment(amount, account_id)` tool. The tool's HTTP response is lost in transit (timeout). The agent retries. Without idempotency keys, what happens? Describe exactly how to make this safe.

---

## Coding challenge

Implement the `DistributedRateLimiter` class described below.

### Specification

A distributed rate limiter tracks request counts across three replicated nodes. It uses a **gossip** mechanism: each node periodically shares its local count with the others. The limiter should be **eventually consistent** — a node may temporarily allow requests above the global limit, but will converge to the correct state.

```python
# Implement this in a single file: rate_limiter.py
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class RateLimiterNode:
    """
    Single node of a distributed rate limiter.

    window_seconds: length of the sliding window
    limit: max allowed requests across ALL nodes combined in the window
    node_id: unique identifier for this node
    """

    node_id: str
    window_seconds: float
    limit: int

    # TODO: implement internal state — store per-node request counts
    # (hint: a dict mapping node_id → list of request timestamps)

    def record_request(self) -> None:
        """Record one request on this node with the current timestamp."""
        raise NotImplementedError

    def gossip_state(self) -> dict[str, list[float]]:
        """
        Return this node's view of all request timestamps, pruned to the window.
        This payload is sent to peer nodes during gossip.
        """
        raise NotImplementedError

    def receive_gossip(self, payload: dict[str, list[float]]) -> None:
        """
        Merge a gossip payload from a peer node.
        For each node_id in the payload, keep the union of timestamps
        (deduplicated). Prune any timestamps outside the window.
        """
        raise NotImplementedError

    def is_allowed(self) -> bool:
        """
        Return True if a new request should be allowed based on this node's
        current (possibly stale) view of the global request count.
        """
        raise NotImplementedError

    def _prune_window(self) -> None:
        """Remove timestamps older than window_seconds from all buckets."""
        raise NotImplementedError


@dataclass
class DistributedRateLimiter:
    """Coordinator for three RateLimiterNode instances with gossip."""

    window_seconds: float = 1.0
    limit: int = 10
    gossip_drop_rate: float = 0.0  # simulate network drops during gossip

    _nodes: list[RateLimiterNode] = field(init=False)

    def __post_init__(self) -> None:
        self._nodes = [
            RateLimiterNode(f"node-{i}", self.window_seconds, self.limit)
            for i in range(3)
        ]

    def request(self, node_idx: int) -> bool:
        """
        Attempt a request on the given node.
        Returns True if allowed (and records it), False if rate-limited.
        """
        raise NotImplementedError

    def gossip_round(self) -> None:
        """
        Each node shares its state with all other nodes.
        Drop gossip messages with probability self.gossip_drop_rate.
        """
        raise NotImplementedError
```

### Acceptance criteria

1. Under no drops, after 10 requests on node-0, `is_allowed()` on all three nodes returns `False` after a gossip round.
2. Under 100% drops, node-1 and node-2 are unaware of requests on node-0 and incorrectly allow further requests — demonstrating eventual consistency divergence.
3. After 2 gossip rounds with 0% drops following a period of isolation, all nodes converge to the same count.
4. After `window_seconds` elapses, old requests are pruned and the limiter allows new requests.
5. Tests cover all four criteria above.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | What a full-mark answer demonstrates |
|---|---|---|
| Independent implementation | 40% | `DistributedRateLimiter` fully implemented without referencing solutions; all 4 acceptance criteria pass in pytest |
| Conceptual understanding | 25% | Conceptual answers are precise, use correct terminology, give concrete AI examples |
| Debugging & problem-solving | 20% | Acceptance criterion 2 (divergence under drops) is explicitly tested AND the analysis correctly identifies it as an AP trade-off |
| Code quality & testing | 15% | Type hints, logging, Pydantic models where data crosses boundaries, ruff-clean, tests use pytest fixtures |

Answer key: ../solutions/week-01/assessment.md
