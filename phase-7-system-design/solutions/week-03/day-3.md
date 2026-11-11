# Solutions — Phase 7 · Week 3 · Day 3 — Replication Strategies

## Exercise 1 — W=2, R=2, N=3: Can You Always Read Your Own Writes?

**Short answer: Yes — W + R > N guarantees at least one overlapping node.**

**Math:**
```
W + R > N
2 + 2 > 3  →  4 > 3  ✓
```

The overlap = W + R - N = 2 + 2 - 3 = **1 node** is always in both the write set and the read set.

**Concrete scenario showing the guarantee:**

Cluster: nodes A, B, C.

Step 1 — Write `user:42 = alice`:
- Coordinator writes to A and B (W=2). Both acknowledge.
- C is NOT written to yet (it may receive it asynchronously later).

Step 2 — Read `user:42` immediately after:
- Coordinator reads from 2 of {A, B, C}. Several possible read sets:
  - {A, B} — both have the write. Returns `alice`. ✓
  - {A, C} — A has it, C doesn't. Coordinator takes the latest version (highest timestamp). Returns `alice`. ✓
  - {B, C} — B has it, C doesn't. Same resolution. Returns `alice`. ✓

**In every case**, at least one node in the read set (A or B) must have the written value because:
- The write set was {A, B}. 
- The read set is any 2 of {A, B, C}.
- Any 2-element subset of a 3-element set shares at least 1 element with any other 2-element subset.

**When does this break?**

If the coordinator uses a **sloppy quorum** — accepting writes to "hint nodes" (e.g., D, E) during a partition, where D and E are not in the normal replica set for this key. Then W + R > N no longer guarantees overlap because the write set and read set are drawn from different populations. The read-your-writes guarantee fails until hinted handoff completes.

---

## Exercise 2 — Quorum Requirements for 5-Node System Tolerating 2 Node Failures

**Goal:** Tolerate 2 simultaneous node failures during both reads and writes.

**N = 5, tolerate f = 2 failures.**

**For writes:**
```
W > f
W > 2  →  W ≥ 3
```
With W=3: a write succeeds even if 2 nodes are down (3 of 5 reachable).

**For reads:**
```
R > f
R > 2  →  R ≥ 3
```
With R=3: a read succeeds even if 2 nodes are down (3 of 5 reachable).

**Verify W + R > N:**
```
3 + 3 = 6 > 5  ✓
```

**Minimum quorum satisfying both fault tolerance and read-your-writes:**

| Parameter | Value |
|---|---|
| N | 5 |
| W (minimum) | 3 |
| R (minimum) | 3 |
| Overlap (W+R-N) | 1 node |
| Nodes that can fail | 2 |

**Trade-offs at W=3, R=3:**
- Write latency: must wait for 3 of 5 nodes — slower than W=2.
- Read latency: must wait for 3 of 5 nodes — slower than R=1.
- Durability: survives 2 simultaneous node failures without data loss.

**Alternative: W=5, R=1 (strong durability, fast reads):**
- All 5 nodes must acknowledge every write — highest write latency, maximum durability.
- Any single node serves reads. Fast reads, but write is a bottleneck.
- Fails if any node is unavailable at write time.

**Asymmetric quorum for read-heavy workloads: W=4, R=2:**
- W+R = 6 > 5. ✓ Overlaps by 1 node.
- Reads query only 2 nodes — lower read latency.
- Writes require 4 of 5 — high write cost.
- Tolerates 1 node failure for writes, 3 node failures for reads (any 2 of remaining 4 suffices).

---

## Exercise 3 — Extend ReplicaRouter with Session Token (Read-Your-Writes)

```python
"""
replica_router_v2.py — ReplicaRouter extended with session-based read-your-writes.

Session tokens track the last write timestamp per session. Reads in the same
session are only routed to replicas that have caught up past that timestamp.

Requires: Python 3.12+
"""

import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

MAX_ACCEPTABLE_LAG_MS = int(os.getenv("MAX_REPLICA_LAG_MS", "500"))
LEADER_HOST = os.getenv("DB_LEADER_HOST", "db-leader.internal")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Replica:
    host: str
    is_leader: bool
    lag_ms: float = 0.0
    last_checked: float = field(default_factory=time.monotonic)

    def refresh_lag(self) -> None:
        """Simulate fetching current replication lag from pg_stat_replication."""
        self.lag_ms = random.uniform(0, 2000)
        self.last_checked = time.monotonic()
        logger.debug("Refreshed lag for %s: %.0f ms", self.host, self.lag_ms)

    def is_caught_up(self, write_timestamp_ms: float) -> bool:
        """
        Return True if this replica has replicated past write_timestamp_ms.

        Logic: a replica that was last_checked at time T has applied all writes
        that arrived before (T - lag_ms). If (last_checked_ms - lag_ms) >= write_timestamp_ms,
        the replica has caught up past the write.
        """
        last_checked_ms = self.last_checked * 1000
        replicated_up_to_ms = last_checked_ms - self.lag_ms
        return replicated_up_to_ms >= write_timestamp_ms


@dataclass
class WriteResult:
    success: bool
    host: str
    write_timestamp_ms: float
    session_id: str


@dataclass
class ReadResult:
    data: Optional[str]
    host: str
    routed_to_leader: bool
    reason: str


# ---------------------------------------------------------------------------
# Session token store
# ---------------------------------------------------------------------------

class SessionStore:
    """
    Tracks the last write timestamp per session ID.

    In production this would live in a shared cache (Redis) so any app server
    can check whether a session's write has propagated to a given replica.
    Here it is in-memory for simplicity.
    """

    def __init__(self) -> None:
        # session_id -> last write monotonic timestamp in ms
        self._last_write: dict[str, float] = {}

    def record_write(self, session_id: str, timestamp_ms: float) -> None:
        """Store or update the last write timestamp for a session."""
        current = self._last_write.get(session_id, 0.0)
        self._last_write[session_id] = max(current, timestamp_ms)
        logger.debug(
            "Session %s last_write_ms updated to %.0f",
            session_id,
            self._last_write[session_id],
        )

    def last_write_ms(self, session_id: str) -> Optional[float]:
        """Return the last write timestamp for a session, or None if no writes."""
        return self._last_write.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Remove session tracking on logout / session expiry."""
        self._last_write.pop(session_id, None)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ReplicaRouter:
    """
    Routes writes to leader, reads to lag-aware replicas.
    Supports session-based read-your-writes via a SessionStore.
    """

    def __init__(
        self,
        replicas: list[Replica],
        max_lag_ms: float = MAX_ACCEPTABLE_LAG_MS,
        session_store: Optional[SessionStore] = None,
    ) -> None:
        self._replicas = replicas
        self._max_lag_ms = max_lag_ms
        self._leader = next(r for r in replicas if r.is_leader)
        self._followers = [r for r in replicas if not r.is_leader]
        self._sessions = session_store or SessionStore()

        logger.info(
            "ReplicaRouter initialized: leader=%s, followers=%d, max_lag_ms=%.0f",
            self._leader.host,
            len(self._followers),
            self._max_lag_ms,
        )

    def write(
        self,
        key: str,
        value: str,
        session_id: Optional[str] = None,
    ) -> WriteResult:
        """
        Write to the leader.

        If session_id provided, records the write timestamp in the session store
        so subsequent reads in this session can enforce read-your-writes.
        """
        logger.info("WRITE key=%s value=%s -> leader=%s", key, value, self._leader.host)
        time.sleep(0.001)  # simulate write latency

        write_ts = time.monotonic() * 1000

        sid = session_id or str(uuid.uuid4())
        self._sessions.record_write(sid, write_ts)

        return WriteResult(
            success=True,
            host=self._leader.host,
            write_timestamp_ms=write_ts,
            session_id=sid,
        )

    def read(
        self,
        key: str,
        session_id: Optional[str] = None,
    ) -> ReadResult:
        """
        Route a read to a healthy replica.

        If session_id is provided and the session has a recorded write, enforces
        read-your-writes: only routes to replicas caught up past the session's
        last write timestamp. Falls back to the leader if no replica qualifies.
        """
        self._refresh_stale_lags()

        # Session-aware read-your-writes
        if session_id is not None:
            last_write = self._sessions.last_write_ms(session_id)
            if last_write is not None:
                return self._session_aware_read(key, session_id, last_write)

        # Standard lag-aware routing
        healthy = self._healthy_followers()
        if not healthy:
            logger.warning("No healthy followers; routing read to leader (fallback)")
            return ReadResult(
                data=self._fetch(self._leader, key),
                host=self._leader.host,
                routed_to_leader=True,
                reason="no healthy follower within lag threshold",
            )

        chosen = random.choice(healthy)
        logger.info(
            "READ key=%s -> follower=%s (lag=%.0f ms)",
            key, chosen.host, chosen.lag_ms,
        )
        return ReadResult(
            data=self._fetch(chosen, key),
            host=chosen.host,
            routed_to_leader=False,
            reason=f"follower lag {chosen.lag_ms:.0f}ms within threshold",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_aware_read(
        self,
        key: str,
        session_id: str,
        last_write_ms: float,
    ) -> ReadResult:
        """
        Route to a replica that has caught up past last_write_ms for this session.
        Falls back to leader if no replica qualifies.
        """
        # Prefer replicas that are (a) within lag threshold AND (b) caught up past write
        qualified = [
            f for f in self._followers
            if f.lag_ms <= self._max_lag_ms and f.is_caught_up(last_write_ms)
        ]

        if qualified:
            chosen = min(qualified, key=lambda r: r.lag_ms)
            logger.info(
                "SESSION READ key=%s session=%s -> follower=%s (lag=%.0f ms, caught up past %.0f ms)",
                key, session_id, chosen.host, chosen.lag_ms, last_write_ms,
            )
            return ReadResult(
                data=self._fetch(chosen, key),
                host=chosen.host,
                routed_to_leader=False,
                reason=(
                    f"follower {chosen.host} caught up past session write "
                    f"(last_write={last_write_ms:.0f}ms, lag={chosen.lag_ms:.0f}ms)"
                ),
            )

        logger.warning(
            "SESSION READ fallback to leader: session=%s no follower caught up past %.0f ms",
            session_id, last_write_ms,
        )
        return ReadResult(
            data=self._fetch(self._leader, key),
            host=self._leader.host,
            routed_to_leader=True,
            reason=(
                f"no follower caught up past session write timestamp "
                f"(last_write={last_write_ms:.0f}ms)"
            ),
        )

    def _healthy_followers(self) -> list[Replica]:
        return [f for f in self._followers if f.lag_ms <= self._max_lag_ms]

    def _refresh_stale_lags(self) -> None:
        """Refresh lag for replicas not checked in the last 1 second."""
        for follower in self._followers:
            if time.monotonic() - follower.last_checked > 1.0:
                follower.refresh_lag()

    def _fetch(self, replica: Replica, key: str) -> str:
        """Stub — simulate row fetch from a replica."""
        return f"value_for_{key}_from_{replica.host}"

    def status(self) -> None:
        logger.info("--- Replica Status ---")
        for r in self._replicas:
            role = "LEADER" if r.is_leader else f"follower (lag={r.lag_ms:.0f}ms)"
            health = "" if r.is_leader else (
                " [OK]" if r.lag_ms <= self._max_lag_ms else " [LAGGING]"
            )
            logger.info("  %s  %s%s", r.host, role, health)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    replicas = [
        Replica(host=LEADER_HOST, is_leader=True),
        Replica(host="db-replica-1.internal", is_leader=False, lag_ms=50),
        Replica(host="db-replica-2.internal", is_leader=False, lag_ms=800),  # lagging
        Replica(host="db-replica-3.internal", is_leader=False, lag_ms=120),
    ]

    router = ReplicaRouter(replicas, max_lag_ms=500)
    router.status()

    logger.info("")
    logger.info("=== Session-based read-your-writes ===")

    session = "sess-user-42"

    # Write in session
    write_result = router.write(key="user:42", value="alice", session_id=session)
    logger.info(
        "Write complete: session=%s ts=%.0f ms",
        write_result.session_id,
        write_result.write_timestamp_ms,
    )

    # Read immediately in same session — must not see stale data
    read_result = router.read(key="user:42", session_id=session)
    logger.info(
        "Read result: host=%s routed_to_leader=%s | %s",
        read_result.host,
        read_result.routed_to_leader,
        read_result.reason,
    )

    logger.info("")
    logger.info("=== Read without session (no RYW guarantee) ===")
    anon_result = router.read(key="user:42")
    logger.info(
        "Anon read: host=%s routed_to_leader=%s",
        anon_result.host,
        anon_result.routed_to_leader,
    )


if __name__ == "__main__":
    main()
```

**Design notes:**
- `SessionStore` is intentionally a separate injectable class — swap it for a Redis-backed implementation in production without changing the router.
- `Replica.is_caught_up()` derives catchup state from `last_checked` + `lag_ms` rather than storing a separate `replicated_up_to_ms` — consistent with how PostgreSQL reports `replay_lag`.
- Leader fallback on session read is logged at WARNING level — it indicates a replica hasn't caught up and is a signal to tune replication or scale up replicas.

---

## Exercise 4 — Multi-Leader Conflict: 3 Resolution Strategies

**Scenario:** User A (on Leader West) renames a document title to `"Q4 Budget Plan"`. Concurrently, User B (on Leader East) renames the same document title to `"FY2025 Budget"`. Both updates commit locally. When leaders sync, both versions exist.

### Strategy 1: Last-Write-Wins (LWW) via Timestamp

**Mechanism:** Each write is tagged with a wall-clock timestamp. On merge, the record with the higher timestamp survives. The other is discarded.

**Example:**
```
Leader West: { title: "Q4 Budget Plan",   updated_at: 1704067200.123 }
Leader East: { title: "FY2025 Budget",    updated_at: 1704067200.456 }
→ East wins: "FY2025 Budget" (higher timestamp)
```

**Tradeoffs:**
- Simple to implement — no domain logic needed.
- Silent data loss: User A's update disappears with no notification.
- Vulnerable to clock skew: NTP drift of even 10ms can flip the winner. A node with a slightly fast clock always "wins" conflicts.
- Works acceptably for: user profile last-updated-at fields where the latest update is semantically correct.

### Strategy 2: Merge with CRDTs (Conflict-Free Replicated Data Types)

**Mechanism:** Use data structures whose merge operation is commutative, associative, and idempotent. The merge always produces a deterministic result regardless of order.

**Example — OR-Set for document tags:**
```
Leader West adds tag "budget" → set = {budget}
Leader East adds tag "finance" → set = {finance}
Merge: union = {budget, finance}  — both survive
```

**For text (e.g., collaborative document title):**
Use a CRDT text sequence (like RGA or LSEQ). Each character has a globally unique position identifier. Concurrent inserts are ordered by position ID, not timestamp. Both edits are merged into the final document without conflict.

**Tradeoffs:**
- No data loss — all concurrent edits are preserved.
- Complex to implement for arbitrary data types. Works natively for counters (G-Counter, PN-Counter), sets (OR-Set), maps (LWW-Map), sequences (RGA).
- Does not work for non-commutative operations (e.g., two conflicting "set title to X" operations cannot be meaningfully merged — you need a different CRDT or a different data model).
- Used by: Riak (CRDT data types), Redis CRDT module, Automerge (collaborative editing).

### Strategy 3: Store Conflict and Surface to Application / User

**Mechanism:** When two versions are detected (via vector clocks or version vectors), store both "sibling" values. Surface the conflict to the application layer, which decides resolution — either programmatically or by prompting the user.

**Example:**
```python
@dataclass
class ConflictedDocument:
    doc_id: str
    versions: list[dict]  # one per conflicting sibling
    vector_clocks: list[dict]

# Application detects conflict, stores siblings:
doc = ConflictedDocument(
    doc_id="doc-42",
    versions=[
        {"title": "Q4 Budget Plan",   "updated_by": "user-A"},
        {"title": "FY2025 Budget",    "updated_by": "user-B"},
    ],
    vector_clocks=[{"west": 5, "east": 3}, {"west": 4, "east": 5}],
)
# Show user: "This document was edited by two people. Choose a version:"
```

**Tradeoffs:**
- Zero data loss — both versions preserved.
- Highest fidelity: the correct resolution is domain-specific and may require human judgment.
- Increases application complexity: every read must handle the "conflict" response type.
- Increases storage: conflicting versions accumulate until resolved.
- Used by: Amazon Dynamo (stores siblings, resolves at read time), CouchDB (detects conflicts, application resolves), Git (manual merge).

---

## Exercise 5 — When Async Replication Becomes Dangerous

### Failure Scenario 1: Leader Crashes Before Replication Completes

**Setup:** Leader (sync=off) accepts write W1, confirms to client, and crashes before the async replication log reaches any follower.

**Failure chain:**
1. Client receives `200 OK` for write W1.
2. Leader crashes.
3. Follower is promoted (newest by log sequence, but W1 is not in its log).
4. Client reads back: W1 is gone.
5. Client retried W1 and it may create a duplicate (if not idempotent) or simply re-apply a lost payment/order.

**Impact:** Committed data loss. Especially dangerous for financial transactions, inventory writes, or any operation that is not safely idempotable.

**Semi-sync mitigation:** With semi-synchronous replication (`synchronous_commit = remote_write` in PostgreSQL, or `sync_binlog=1` + one semi-sync replica in MySQL), the leader waits for at least one replica to acknowledge the write to its WAL before confirming to the client. Even if the leader crashes, the semi-sync replica holds the write and can be safely promoted without data loss.

### Failure Scenario 2: Follower Promoted with Stale State Becomes New Leader

**Setup:** Async replication, follower F2 lags 10 seconds behind leader L1. L1 becomes unreachable. Automated failover (e.g., Patroni, AWS RDS Multi-AZ) promotes F2 (perhaps the only reachable replica).

**Failure chain:**
1. L1 had processed 50,000 writes in the past 10 seconds — none are on F2.
2. F2 is promoted. It accepts new writes from clients.
3. L1 comes back online (network partition healed, not crashed).
4. **Split brain**: both L1 and F2 believe they are leader. Clients write to both.
5. Without fencing tokens (epoch numbers), writes diverge irreversibly.

**Data loss:** 50,000 writes on L1 are lost. Additional writes to both "leaders" diverge further.

**Semi-sync mitigation:**
- With semi-sync, the promoted replica must have at least as many commits as the old leader at the time of failover (it acknowledged all writes before the leader confirmed them to clients). No gap to bridge.
- Fencing tokens (STONITH — Shoot The Other Node In The Head) ensure the old leader is shot down before the new leader accepts writes, preventing split brain even with async replication.

**PostgreSQL `synchronous_commit` levels for reference:**

| Level | Durability | Latency |
|---|---|---|
| `off` | Write may be lost on crash | Lowest (async, fire-and-forget) |
| `local` | Durable on leader only | Low (leader WAL flush) |
| `remote_write` | Durable in replica OS buffer | Medium (replica receives, not fsynced) |
| `remote_apply` | Durable and applied on replica | Higher (replica has applied to its WAL) |
| `on` (default) | Durable and applied on replica | Same as `remote_apply` |

**Financial ledger:** use `remote_apply` (or `on`) — durability is non-negotiable; a few ms of extra latency is acceptable.
**Analytics event stream:** use `off` — losing occasional events is tolerable; throughput and low latency matter more.

---

## Notes

- **W + R > N is a contract, not magic**: it guarantees overlap under normal operation. Sloppy quorums, clock skew, and network partitions can violate the assumption. Design for the failure mode, not just the happy path.
- **Session tokens are cheap insurance**: tracking `{session_id: last_write_timestamp}` in Redis costs one key per session. The cost of a user seeing stale data after their own write (shopping cart reset, profile photo not updated) costs far more in support tickets.
- **CRDT ≠ silver bullet**: CRDTs eliminate conflicts for *specific data types*. They cannot resolve two conflicting "set this field to X" operations. Know which CRDT matches which domain: G-Counter for view counts, OR-Set for tags, LWW-Register for last-updated metadata.
- **Async replication debt accumulates**: every millisecond of lag is a millisecond of potential data loss. Monitor `pg_stat_replication.write_lag`, `flush_lag`, `replay_lag` continuously. Alert on lag > your RPO threshold (e.g., 500ms for most transactional workloads).
- **Interview tip**: When asked about replication, lead with "what's the acceptable RPO (Recovery Point Objective)?" That drives the sync/async decision. Then ask "what's the read/write ratio?" That drives quorum tuning.
