# Solutions — Day 4: Consensus / Raft

## Exercise 1 — Majority requirement by cluster size

| Nodes (N) | Majority needed | Max failures tolerated |
|---|---|---|
| 3 | 2 | 1 |
| 5 | 3 | 2 |
| 7 | 4 | 3 |

Formula: majority = floor(N/2) + 1; fault tolerance = floor((N-1)/2)

```python
def run_multi_size():
    for n in (3, 5, 7):
        nodes = [RaftNode(f"node-{i}") for i in range(n)]
        for node in nodes:
            node.peers = [p for p in nodes if p is not node]
        initiator = nodes[0]
        won = initiator.start_election()
        majority = n // 2 + 1
        print(f"N={n} majority={majority} fault_tolerance={n//2} elected={won}")
```

## Exercise 2 — Split vote with 4 nodes

With 4 nodes (even number), majority = 3. If two candidates each get 2 votes (themselves + 1 peer), neither reaches majority. Resolution:
- Both step back to follower
- Random timeouts cause one to fire first in the next term
- That node starts a new election and wins with 3 votes

```python
def simulate_split_vote():
    nodes = [RaftNode(f"node-{i}") for i in range(4)]
    # Force identical timeout to trigger simultaneous elections
    for n in nodes:
        n.election_timeout_ms = 150.0
        n.peers = [p for p in nodes if p is not n]

    # Simulate two candidates starting simultaneously
    # In a real system they'd send RequestVote at the same time
    # Here we sequence them; both will fail to get majority=3
    won_0 = nodes[0].start_election()  # can get at most 2 votes (self + 1)
    won_2 = nodes[2].start_election()  # term incremented; nodes[0] may now step down

    print(f"node-0 won: {won_0}, node-2 won: {won_2}")
    # Resolution: in the next term, one fires first and wins
    # Raft's random timeouts prevent this from repeating indefinitely
```

## Exercise 3 — Log replication

```python
def log_append(self, command: str) -> int:
    """Append a new entry to the leader's log. Returns the log index."""
    if self.role != Role.LEADER:
        raise RuntimeError(f"{self.node_id} is not the leader")
    term_entry = (self.current_term, command)
    self.log_entries = getattr(self, "log_entries", [])
    self.log_entries.append(term_entry)
    index = len(self.log_entries)
    self.log.info("LOG_APPEND  idx=%d  term=%d  cmd=%s", index, self.current_term, command)
    return index


def replicate_to_followers(self) -> bool:
    """
    Send the latest log entry to all peers.
    Return True if majority (including self) acknowledged.
    """
    if not hasattr(self, "log_entries") or not self.log_entries:
        return False
    entry = self.log_entries[-1]
    index = len(self.log_entries)
    acks = 1  # leader always acks itself
    majority = (len(self.peers) + 1) // 2 + 1

    for peer in self.peers:
        # In a real system: send AppendEntries RPC, wait for response
        # Here: directly call peer's apply method
        if not hasattr(peer, "log_entries"):
            peer.log_entries = []
        peer.log_entries.append(entry)
        acks += 1

    committed = acks >= majority
    if committed:
        self.commit_index = index
        self.log.info("COMMITTED  idx=%d  acks=%d/%d", index, acks, len(self.peers) + 1)
    return committed
```

## Exercise 4 — Byzantine attack on vote granting

A Byzantine node could lie about `last_log_term` and `last_log_index` — claiming a very high log term and index it doesn't actually have. This makes it appear to have the most up-to-date log, winning votes from honest nodes even when its actual log is stale or corrupt.

Once elected leader, the Byzantine node could:
1. Accept writes and confirm them as "committed" without replicating to any follower
2. Send conflicting log entries to different followers, corrupting the cluster's state
3. Refuse to replicate, causing followers' logs to diverge while claiming commits

Standard Raft does not protect against this because it assumes **crash-fault tolerance only** — the fail-stop model where nodes either work correctly or stop responding. Byzantine fault-tolerant (BFT) consensus protocols (e.g., PBFT, HotStuff) require N ≥ 3f+1 nodes to tolerate f Byzantine nodes, and use cryptographic signatures to authenticate messages.

In practice: Raft is used within trusted data centers where Byzantine behavior means software bugs, not malicious actors. For adversarial environments (blockchain, multi-org federations), BFT protocols are needed.
