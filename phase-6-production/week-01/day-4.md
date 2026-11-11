# Phase 6 · Week 01 · Day 4 — Consensus: Raft Intuition

> Objective: Understand why consensus is hard and how Raft solves it through leader election and log replication.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### The consensus problem

**Consensus** means getting a group of nodes to agree on a single value, even when some nodes fail or messages are delayed. It underpins:

- Distributed locks (who holds the lock?)
- Leader election (who is the primary?)
- Atomic broadcast (in what order did events happen?)
- Distributed transactions (did all nodes commit?)

Without consensus, two nodes could simultaneously believe they are the leader — the **split-brain** problem — leading to conflicting writes, data corruption, or duplicate actions.

Consensus is provably impossible in an asynchronous system if even one node can fail (**FLP impossibility**, Fischer-Lynch-Paterson 1985). Real systems escape this by using **timeouts** to detect failures (even though timeouts can be wrong) and by restricting progress to periods where a majority of nodes are reachable.

### Why Paxos is hard to implement

Paxos (Lamport, 1989) is the theoretical foundation. It proves correctness but:
- Underspecifies the full system (log, leader election, membership changes)
- Is notoriously difficult to implement correctly
- "There is only one consensus protocol, and it's Paxos" — but every implementation adds undocumented extensions

### Raft: consensus for understandable systems

Raft (Ongaro & Ousterhout, 2014) was designed explicitly for **understandability**. It decomposes consensus into three mostly-independent sub-problems:

1. **Leader election** — select one leader per term
2. **Log replication** — leader accepts entries, replicates to followers, commits when majority acknowledges
3. **Safety** — ensure no two nodes commit different values at the same log index

### Leader election

Each server is in one of three states: **follower**, **candidate**, **leader**.

```
          timeout, start election
Follower ──────────────────────────▶ Candidate
   ▲                                      │
   │ discover leader or                   │ receive majority votes
   │ higher-term candidate                ▼
   └──────────────────────────────── Leader
```

**Terms** are logical time units (a monotonically increasing integer). Each election starts a new term. A leader is valid only for its term.

Election process:
1. Follower's election timeout fires (150–300 ms random)
2. Follower increments its term and transitions to Candidate
3. Candidate votes for itself, sends `RequestVote` RPC to all peers
4. Any node that hasn't voted in this term and whose log is no older votes yes
5. Candidate receiving votes from a **majority** (> N/2) becomes the new leader
6. Leader sends periodic `AppendEntries` heartbeats to suppress new elections

**Why random timeouts?** Prevents all followers from starting elections simultaneously, which would split votes repeatedly.

### Log replication

```
Leader log:   [idx=1, cmd=SET x=1] [idx=2, cmd=SET y=2] [idx=3, cmd=DEL x]
                      ↓ AppendEntries RPC
Follower-1 log: [idx=1, cmd=SET x=1] [idx=2, cmd=SET y=2]  ← still replicating idx=3
Follower-2 log: [idx=1, cmd=SET x=1] [idx=2, cmd=SET y=2] [idx=3, cmd=DEL x]
```

An entry is **committed** once a majority of nodes have stored it. The leader tracks the highest committed index (`commitIndex`) and sends it in heartbeats. Followers apply committed entries to their state machine.

**Safety invariant**: if a log entry is committed at index i, every future leader will have that entry at index i. Raft enforces this through its vote restriction: a candidate cannot win if its log is older than a majority of voters' logs.

### What Raft does NOT solve

- Performance at scale (Paxos/Multi-Paxos with batching is often faster)
- Byzantine failures (a lying node can fool Raft — use BFT protocols for that)
- Arbitrary network partitions with unbounded duration

### Raft in production AI systems

| Usage | Example |
|---|---|
| etcd (Kubernetes) | Consensus store for cluster state; agent configs |
| Kafka controller | Leader election for partition leadership |
| CockroachDB | Per-range Raft groups for distributed SQL |
| Zookeeper | Zab protocol (Paxos-like); used by Kafka <= 2.x |
| Vector DB clustering | Qdrant, Weaviate use Raft for cluster membership |

## How it works internally

### AppendEntries RPC

The leader sends `AppendEntries` both as a heartbeat (empty entries) and as a replication mechanism. Each message includes:

- `term`: leader's current term
- `leaderId`: for followers to redirect clients
- `prevLogIndex`, `prevLogTerm`: consistency check — follower must have this entry
- `entries[]`: new log entries to store (empty for heartbeat)
- `leaderCommit`: leader's commit index

If the follower's log doesn't match `prevLogIndex/prevLogTerm`, it rejects the message. The leader backs up and retries — eventually finding the last matching entry and overwriting any divergent suffix.

### Membership changes

Adding or removing nodes is dangerous: a naive approach can create two disjoint majorities. Raft handles this via **joint consensus** (two-phase) or **single-server changes** (safer, one node at a time).

## Worked example

```python
"""
day4_raft_simulation.py — Simplified Raft leader election simulation.

Models the election timeout mechanism and vote-granting logic.
Does NOT implement the full Raft protocol (log replication omitted for clarity).

Run: python day4_raft_simulation.py
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
)

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "99"))
random.seed(RANDOM_SEED)

ELECTION_TIMEOUT_MIN_MS = 150
ELECTION_TIMEOUT_MAX_MS = 300
HEARTBEAT_INTERVAL_MS = 50


class Role(Enum):
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()


class VoteRequest(NamedTuple):
    candidate_id: str
    term: int
    last_log_index: int
    last_log_term: int


class VoteResponse(NamedTuple):
    voter_id: str
    term: int
    granted: bool


@dataclass
class RaftNode:
    node_id: str
    peers: list["RaftNode"] = field(default_factory=list, repr=False)

    # Persistent state
    current_term: int = 0
    voted_for: str | None = None
    log: list[tuple[int, str]] = field(default_factory=list)  # (term, command)

    # Volatile state
    role: Role = Role.FOLLOWER
    commit_index: int = 0
    last_heartbeat: float = field(default_factory=time.monotonic)
    election_timeout_ms: float = field(init=False)

    def __post_init__(self) -> None:
        self.election_timeout_ms = random.uniform(
            ELECTION_TIMEOUT_MIN_MS, ELECTION_TIMEOUT_MAX_MS
        )
        self.log = logging.getLogger(self.node_id)

    def _last_log_index(self) -> int:
        return len(self.log)

    def _last_log_term(self) -> int:
        return self.log[-1][0] if self.log else 0

    def receive_heartbeat(self, leader_term: int) -> None:
        if leader_term >= self.current_term:
            self.current_term = leader_term
            self.role = Role.FOLLOWER
            self.voted_for = None
            self.last_heartbeat = time.monotonic()

    def process_vote_request(self, req: VoteRequest) -> VoteResponse:
        """Grant vote if candidate's term is current and log is at least as up-to-date."""
        grant = False

        if req.term < self.current_term:
            pass  # stale term — deny
        else:
            if req.term > self.current_term:
                self.current_term = req.term
                self.role = Role.FOLLOWER
                self.voted_for = None

            already_voted = self.voted_for not in (None, req.candidate_id)
            candidate_log_ok = (
                req.last_log_term > self._last_log_term()
                or (
                    req.last_log_term == self._last_log_term()
                    and req.last_log_index >= self._last_log_index()
                )
            )
            if not already_voted and candidate_log_ok:
                self.voted_for = req.candidate_id
                self.last_heartbeat = time.monotonic()  # reset timeout
                grant = True

        self.log.debug(
            "vote_request from=%s term=%d grant=%s",
            req.candidate_id, req.term, grant,
        )
        return VoteResponse(voter_id=self.node_id, term=self.current_term, granted=grant)

    def start_election(self) -> bool:
        """Start a new election. Return True if this node becomes leader."""
        self.current_term += 1
        self.role = Role.CANDIDATE
        self.voted_for = self.node_id
        votes_received = 1  # vote for self
        majority = (len(self.peers) + 1) // 2 + 1

        self.log.info(
            "Starting election for term=%d  need=%d/%d votes",
            self.current_term, majority, len(self.peers) + 1,
        )

        req = VoteRequest(
            candidate_id=self.node_id,
            term=self.current_term,
            last_log_index=self._last_log_index(),
            last_log_term=self._last_log_term(),
        )

        for peer in self.peers:
            response = peer.process_vote_request(req)
            if response.term > self.current_term:
                # Discovered higher term — step down
                self.current_term = response.term
                self.role = Role.FOLLOWER
                self.voted_for = None
                self.log.info("Stepped down — higher term %d seen", response.term)
                return False
            if response.granted:
                votes_received += 1
                self.log.debug("Vote granted by %s  total=%d", response.voter_id, votes_received)

        if votes_received >= majority:
            self.role = Role.LEADER
            self.log.info(
                "Elected LEADER  term=%d  votes=%d/%d",
                self.current_term, votes_received, len(self.peers) + 1,
            )
            return True
        else:
            self.role = Role.FOLLOWER
            self.log.info(
                "Election failed  votes=%d/%d needed=%d",
                votes_received, len(self.peers) + 1, majority,
            )
            return False


def run_election_simulation(num_nodes: int = 5) -> None:
    log = logging.getLogger("simulation")

    nodes = [RaftNode(node_id=f"node-{i}") for i in range(num_nodes)]
    for node in nodes:
        node.peers = [p for p in nodes if p is not node]

    # Simulate: node-0's timeout fires first (lowest random timeout value)
    # In a real system this would be driven by actual wall-clock timeouts
    initiator = min(nodes, key=lambda n: n.election_timeout_ms)
    log.info(
        "Simulating election timeout on %s (timeout=%.0f ms)",
        initiator.node_id, initiator.election_timeout_ms,
    )

    won = initiator.start_election()

    log.info("\nFinal node states:")
    for n in nodes:
        log.info("  %-10s role=%-12s term=%d voted_for=%s",
                 n.node_id, n.role.name, n.current_term, n.voted_for)

    if won:
        # Leader sends heartbeats — all followers reset their timeouts
        for follower in initiator.peers:
            follower.receive_heartbeat(initiator.current_term)
        log.info("Leader %s sent heartbeats to all followers", initiator.node_id)


if __name__ == "__main__":
    run_election_simulation()
```

## Exercises

1. Run the simulation with `num_nodes=3`, `num_nodes=5`, and `num_nodes=7`. Verify the majority requirement in each case. What is the minimum number of node failures each cluster can tolerate?

2. Simulate a **split vote**: create 4 nodes with identical `election_timeout_ms` values so two candidates start elections simultaneously. What happens? How does Raft's term mechanism eventually resolve it?

3. Extend `RaftNode` with a `log_append(command: str)` method on the leader that appends an entry. Then implement a simplified `replicate_to_followers()` that sends the new entry to all peers and counts acknowledgements. Only mark the entry as committed when a majority acknowledges.

4. The `process_vote_request` method currently trusts the candidate's claimed `last_log_term` and `last_log_index`. Describe (in comments or a short write-up) what attack a Byzantine node could mount by lying about these fields. Why does standard Raft not protect against this?

(Solutions: ../solutions/week-01/day-4.md)

## Self-check

- [ ] I can explain why consensus is necessary and what "split-brain" means
- [ ] I can describe Raft's three sub-problems and explain leader election step by step
- [ ] I can state the majority quorum requirement and derive the fault-tolerance of a cluster of size N
- [ ] I can explain why random election timeouts prevent repeated split votes
- [ ] I can name three production systems that use Raft or a similar consensus protocol

## Supplementary reading

- [In Search of an Understandable Consensus Algorithm (Extended Version)](https://raft.github.io/raft.pdf) — Ongaro & Ousterhout, 2014
- [The Raft website with visualisation](https://raft.github.io/)
- Designing Data-Intensive Applications (Kleppmann) — Chapter 9: Consistency and Consensus
- [Impossibility of Distributed Consensus with One Faulty Process](https://groups.csail.mit.edu/tds/papers/Lynch/jacm85.pdf) — Fischer, Lynch, Paterson, 1985 (FLP)
