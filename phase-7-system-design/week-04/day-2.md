# Phase 7 · Week 04 · Day 2 — Event Sourcing

> Objective: Implement an append-only event store with aggregate reconstitution, snapshots, time-travel queries, and versioning — all in-memory with no external dependencies.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-1]]

---

## Concept

Event sourcing replaces the "store current state" pattern with "store every state change as an ordered, immutable event." The current state of any entity is derived by replaying its event history from the beginning (or from a snapshot).

### Traditional vs Event-Sourced State

```
Traditional (mutable state):
  Table: accounts
  | account_id | balance | updated_at |
  |    acc-1   |   150   | 2024-11-10 |

  UPDATE accounts SET balance = 150 WHERE account_id = 'acc-1';
  -- Previous state of $200, $250 is gone forever.

Event-Sourced (immutable log):
  Stream: accounts/acc-1
  | seq | event_type      | data              | timestamp          |
  |  0  | AccountOpened   | {balance: 0}      | 2024-11-08 09:00   |
  |  1  | MoneyDeposited  | {amount: 250}     | 2024-11-08 09:01   |
  |  2  | MoneyWithdrawn  | {amount: 100}     | 2024-11-10 14:22   |
  -- Balance of 150 is derived: 0 + 250 - 100
```

### Why Event Sourcing

- **Complete audit trail** — every state change is permanently recorded with who, what, when
- **Temporal queries** — reconstruct state at any point in time ("what was the account balance on November 9th?")
- **Event replay** — re-project state for new use cases without changing the source
- **Debugging** — reproduce exact conditions that led to a bug
- **Integration** — events are natural integration points (cf. Day 1 pub/sub)

### Core Concepts

**Aggregate** — a cluster of domain objects treated as a single transactional unit. Each aggregate has a stream of events. An `Account` is an aggregate; its `AccountId` is the stream key.

**Event** — a named, versioned, immutable fact about what happened. Events are past-tense nouns: `MoneyDeposited`, not `DepositMoney`.

**Event store** — the append-only database that persists event streams. Supports: append, read-stream, read-all (in sequence).

**Reconstitution** — fold (reduce) over an event stream to rebuild current state. `fold(initial_state, events) -> current_state`.

**Snapshot** — a point-in-time cache of an aggregate's state at a given sequence number. Avoids replaying the entire history on every read when streams grow long.

**Optimistic concurrency** — when appending, include the expected version (last-known sequence number). The store rejects the write if the stream has been modified since. Prevents lost updates without locking.

### Event Versioning

Event schemas evolve. Three strategies:

| Strategy | Description | Trade-off |
|---|---|---|
| Weak schema | Ignore unknown fields | Simple, fragile if fields removed |
| Upcasting | Transform old event to new schema on read | Transparent to consumers |
| Copy-transform | Migrate stream by replaying with new handlers | Clean slate, expensive once |

---

## How It Works Internally

### Append Operation

```
append(stream_id, events, expected_version):
  1. Acquire lock on stream (or use optimistic check)
  2. Read current stream head version
  3. If head != expected_version -> raise ConcurrencyError
  4. Assign sequence numbers (head+1, head+2, ...)
  5. Write events atomically
  6. Release lock / increment global position
  7. Notify subscribers (optional — links to Day 1 pub/sub)
```

### Reconstitution with Snapshot

```
load_aggregate(stream_id):
  1. Find latest snapshot for stream_id (if any)
     -> snapshot.state, snapshot.version
  2. Read events from snapshot.version+1 to HEAD
  3. Fold events over snapshot.state
  4. Return current state + version

  Without snapshot: fold ALL events from version 0.
  With snapshot:    fold only delta events since snapshot.
```

### Global Event Log

A single ordered sequence of all events across all streams (the "global log" or "transaction log"). Used for:

- Building read models (projections) — Day 3
- Point-in-time recovery
- Cross-aggregate causality analysis

```
Global position:  1    2    3    4    5    6    7
Stream ID:        acc1 acc2 acc1 acc3 acc1 acc2 acc3
Event type:       Open Open Dep  Open Wdw  Dep  Dep
```

### Snapshot Strategy

Snapshot every N events (e.g., every 50). On load:

- Snapshots reduce reconstitution cost from O(N) to O(N mod snapshot_interval)
- Snapshots are stored separately from the event log (they are not events)
- Snapshots can be deleted safely — state can always be recomputed from events

---

## Worked Example

```python
"""
event_store.py — In-memory event store with aggregate reconstitution, snapshots,
time-travel queries, and optimistic concurrency.

Modeled after EventStoreDB / Marten patterns, but purely in-memory.

Run:
    python event_store.py

No external dependencies beyond Python 3.12 stdlib.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("event_store")


# ---------------------------------------------------------------------------
# Domain events — immutable, past-tense facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    schema_version: int = 1


@dataclass(frozen=True)
class AccountOpened(DomainEvent):
    owner: str = ""
    initial_balance: float = 0.0


@dataclass(frozen=True)
class MoneyDeposited(DomainEvent):
    amount: float = 0.0
    reference: str = ""


@dataclass(frozen=True)
class MoneyWithdrawn(DomainEvent):
    amount: float = 0.0
    reference: str = ""


@dataclass(frozen=True)
class AccountClosed(DomainEvent):
    reason: str = ""


# ---------------------------------------------------------------------------
# Stored event — wraps a domain event with stream metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StoredEvent:
    stream_id: str
    sequence: int          # position within this stream (0-based)
    global_position: int   # position in the global ordered log
    event: DomainEvent


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    stream_id: str
    version: int           # stream sequence at snapshot time
    state: Any             # aggregate state


# ---------------------------------------------------------------------------
# Concurrency error
# ---------------------------------------------------------------------------

class ConcurrencyError(Exception):
    """Raised when expected_version does not match actual stream head."""


# ---------------------------------------------------------------------------
# Event store
# ---------------------------------------------------------------------------

class InMemoryEventStore:
    """
    Append-only event store.

    - Per-stream ordered sequences
    - Global ordered log across all streams
    - Optimistic concurrency control
    - Snapshot storage
    """

    def __init__(self) -> None:
        # stream_id -> list of StoredEvent (ordered by sequence)
        self._streams: dict[str, list[StoredEvent]] = {}
        # global log — all events in append order
        self._global_log: list[StoredEvent] = []
        # stream_id -> Snapshot (latest only; real stores keep multiple)
        self._snapshots: dict[str, Snapshot] = {}
        self._log = logging.getLogger("store")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        stream_id: str,
        events: list[DomainEvent],
        expected_version: int = -1,
    ) -> int:
        """
        Append events to a stream.

        expected_version:
          -1  = stream must not exist (creation)
          N   = stream must currently be at version N
          Any = skip version check (pass None)

        Returns the new stream version (last sequence number).
        """
        if stream_id not in self._streams:
            if expected_version not in (-1,):
                # -1 means "I expect no stream"; any other value on a non-existent
                # stream is only valid if caller passes None to skip the check
                if expected_version is not None and expected_version != -1:
                    raise ConcurrencyError(
                        f"Stream '{stream_id}' does not exist. "
                        f"Expected version {expected_version}."
                    )
            self._streams[stream_id] = []

        current_version = len(self._streams[stream_id]) - 1  # -1 if empty

        if expected_version is not None and expected_version != -2:
            if expected_version != current_version:
                raise ConcurrencyError(
                    f"Stream '{stream_id}': expected version {expected_version}, "
                    f"actual {current_version}."
                )

        for event in events:
            sequence = len(self._streams[stream_id])
            global_pos = len(self._global_log)
            stored = StoredEvent(
                stream_id=stream_id,
                sequence=sequence,
                global_position=global_pos,
                event=event,
            )
            self._streams[stream_id].append(stored)
            self._global_log.append(stored)

        new_version = len(self._streams[stream_id]) - 1
        self._log.debug(
            "Appended %d events to stream '%s', version now %d",
            len(events), stream_id, new_version,
        )
        return new_version

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_stream(
        self,
        stream_id: str,
        from_version: int = 0,
        to_version: int | None = None,
    ) -> list[StoredEvent]:
        """
        Read events from a stream between from_version and to_version (inclusive).
        """
        events = self._streams.get(stream_id, [])
        end = (to_version + 1) if to_version is not None else len(events)
        return events[from_version:end]

    def read_global(
        self,
        from_position: int = 0,
        max_events: int = 1000,
    ) -> list[StoredEvent]:
        """Read from the global log, useful for building projections."""
        return self._global_log[from_position: from_position + max_events]

    def stream_version(self, stream_id: str) -> int:
        """Return current version (-1 if stream does not exist)."""
        return len(self._streams.get(stream_id, [])) - 1

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def save_snapshot(self, stream_id: str, version: int, state: Any) -> None:
        self._snapshots[stream_id] = Snapshot(
            stream_id=stream_id, version=version, state=state
        )
        self._log.debug("Snapshot saved for '%s' at version %d", stream_id, version)

    def load_snapshot(self, stream_id: str) -> Snapshot | None:
        return self._snapshots.get(stream_id)


# ---------------------------------------------------------------------------
# State model for BankAccount aggregate
# ---------------------------------------------------------------------------

@dataclass
class AccountState:
    account_id: str
    owner: str = ""
    balance: float = 0.0
    is_closed: bool = False
    version: int = -1

    @classmethod
    def initial(cls, account_id: str) -> "AccountState":
        return cls(account_id=account_id)


# ---------------------------------------------------------------------------
# Aggregate reconstitution — fold events over state
# ---------------------------------------------------------------------------

StateT = TypeVar("StateT")


def apply_event(state: AccountState, event: DomainEvent) -> AccountState:
    """
    Pure function: given current state + one event, return new state.
    No side effects. This is the 'fold' step.
    """
    match event:
        case AccountOpened(owner=owner, initial_balance=bal):
            return AccountState(
                account_id=state.account_id,
                owner=owner,
                balance=bal,
                is_closed=False,
                version=state.version + 1,
            )
        case MoneyDeposited(amount=amount):
            return AccountState(
                account_id=state.account_id,
                owner=state.owner,
                balance=state.balance + amount,
                is_closed=state.is_closed,
                version=state.version + 1,
            )
        case MoneyWithdrawn(amount=amount):
            return AccountState(
                account_id=state.account_id,
                owner=state.owner,
                balance=state.balance - amount,
                is_closed=state.is_closed,
                version=state.version + 1,
            )
        case AccountClosed():
            return AccountState(
                account_id=state.account_id,
                owner=state.owner,
                balance=state.balance,
                is_closed=True,
                version=state.version + 1,
            )
        case _:
            logger.warning("Unknown event type: %s — state unchanged", type(event).__name__)
            return state


def reconstitute(
    initial: AccountState,
    events: list[StoredEvent],
) -> AccountState:
    """Fold all events over initial state to produce current state."""
    state = initial
    for stored in events:
        state = apply_event(state, stored.event)
    return state


# ---------------------------------------------------------------------------
# Repository — encapsulates store + reconstitution + snapshotting
# ---------------------------------------------------------------------------

SNAPSHOT_EVERY = 5  # take a snapshot every N events


class AccountRepository:
    def __init__(self, store: InMemoryEventStore) -> None:
        self._store = store
        self._log = logging.getLogger("repo.account")

    def load(self, account_id: str) -> AccountState:
        """
        Load current account state.
        Uses snapshot if available, then replays only delta events.
        """
        snapshot = self._store.load_snapshot(account_id)

        if snapshot:
            from_version = snapshot.version + 1
            base_state = snapshot.state
            self._log.debug(
                "Loaded snapshot for %s at version %d", account_id, snapshot.version
            )
        else:
            from_version = 0
            base_state = AccountState.initial(account_id)

        delta_events = self._store.read_stream(account_id, from_version=from_version)
        state = reconstitute(base_state, delta_events)

        # Auto-snapshot after loading if stream is long enough
        current_version = self._store.stream_version(account_id)
        should_snapshot = (
            snapshot is None and current_version >= SNAPSHOT_EVERY - 1
        ) or (
            snapshot is not None
            and (current_version - snapshot.version) >= SNAPSHOT_EVERY
        )
        if should_snapshot:
            self._store.save_snapshot(account_id, current_version, state)
            self._log.debug("Auto-snapshot taken for %s at v%d", account_id, current_version)

        return state

    def save(
        self,
        account_id: str,
        new_events: list[DomainEvent],
        expected_version: int,
    ) -> AccountState:
        """Append new events and return updated state."""
        self._store.append(account_id, new_events, expected_version=expected_version)
        return self.load(account_id)


# ---------------------------------------------------------------------------
# Time-travel query — reconstruct state at a specific timestamp
# ---------------------------------------------------------------------------

def state_at_time(
    store: InMemoryEventStore,
    account_id: str,
    as_of: float,
) -> AccountState:
    """
    Return account state as it was at timestamp `as_of`.
    Only events with timestamp <= as_of are included.
    """
    all_events = store.read_stream(account_id)
    relevant = [e for e in all_events if e.event.timestamp <= as_of]
    return reconstitute(AccountState.initial(account_id), relevant)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_account_lifecycle(store: InMemoryEventStore) -> None:
    logger.info("=== Account Lifecycle Demo ===")
    repo = AccountRepository(store)
    account_id = "acc-alice"

    # Create account — stream must not exist
    repo.save(
        account_id,
        [AccountOpened(owner="Alice", initial_balance=0.0)],
        expected_version=-1,
    )

    # Deposit
    state = repo.load(account_id)
    repo.save(
        account_id,
        [MoneyDeposited(amount=1000.0, reference="salary")],
        expected_version=state.version,
    )

    # Multiple deposits and a withdrawal
    state = repo.load(account_id)
    repo.save(
        account_id,
        [
            MoneyDeposited(amount=200.0, reference="bonus"),
            MoneyWithdrawn(amount=350.0, reference="rent"),
        ],
        expected_version=state.version,
    )

    final_state = repo.load(account_id)
    logger.info(
        "Account %s: owner=%s balance=%.2f version=%d",
        account_id, final_state.owner, final_state.balance, final_state.version,
    )
    assert abs(final_state.balance - 850.0) < 0.01, f"Expected 850, got {final_state.balance}"


def demo_optimistic_concurrency(store: InMemoryEventStore) -> None:
    logger.info("=== Optimistic Concurrency Demo ===")
    repo = AccountRepository(store)
    account_id = "acc-bob"

    store.append(account_id, [AccountOpened(owner="Bob", initial_balance=500.0)], expected_version=-1)
    store.append(account_id, [MoneyDeposited(amount=100.0)], expected_version=0)

    # Simulate two concurrent readers both loading version 1
    state_a = repo.load(account_id)  # version = 1
    # state_b = repo.load(account_id)  # same version = 1

    # First writer succeeds
    store.append(account_id, [MoneyWithdrawn(amount=50.0)], expected_version=state_a.version)
    logger.info("First write succeeded")

    # Second writer tries to write with stale version — must fail
    try:
        store.append(account_id, [MoneyWithdrawn(amount=75.0)], expected_version=state_a.version)
        logger.error("ERROR: second write should have been rejected!")
    except ConcurrencyError as exc:
        logger.info("Concurrency conflict correctly detected: %s", exc)


def demo_time_travel(store: InMemoryEventStore) -> None:
    logger.info("=== Time Travel Query Demo ===")

    account_id = "acc-charlie"
    t0 = time.time()

    # Manually build events with specific timestamps
    e1 = AccountOpened(owner="Charlie", initial_balance=0.0)
    e2 = MoneyDeposited(amount=500.0, reference="initial")
    e3 = MoneyDeposited(amount=200.0, reference="second")
    e4 = MoneyWithdrawn(amount=100.0, reference="withdrawal")

    # Simulate time passing between events
    object.__setattr__(e1, "timestamp", t0 + 0)
    object.__setattr__(e2, "timestamp", t0 + 10)
    object.__setattr__(e3, "timestamp", t0 + 20)
    object.__setattr__(e4, "timestamp", t0 + 30)

    store.append(account_id, [e1, e2, e3, e4], expected_version=-1)

    # Query state at t0+15 (after deposit 2, before deposit 3)
    mid_state = state_at_time(store, account_id, as_of=t0 + 15)
    logger.info(
        "State at t+15s: balance=%.2f (expected 500.00)", mid_state.balance
    )
    assert abs(mid_state.balance - 500.0) < 0.01

    # Query final state
    final_state = state_at_time(store, account_id, as_of=t0 + 35)
    logger.info(
        "State at t+35s: balance=%.2f (expected 600.00)", final_state.balance
    )
    assert abs(final_state.balance - 600.0) < 0.01


def demo_snapshots(store: InMemoryEventStore) -> None:
    logger.info("=== Snapshot Demo ===")
    repo = AccountRepository(store)
    account_id = "acc-dave"

    # Append 12 events (triggers auto-snapshot at load after threshold)
    store.append(account_id, [AccountOpened(owner="Dave", initial_balance=0.0)], expected_version=-1)
    for i in range(11):
        store.append(account_id, [MoneyDeposited(amount=10.0)], expected_version=i)

    # First load: reconstitutes from 12 events, then auto-snapshots
    state = repo.load(account_id)
    logger.info(
        "After 12 events: balance=%.2f version=%d", state.balance, state.version
    )

    snapshot = store.load_snapshot(account_id)
    assert snapshot is not None, "Snapshot should have been created"
    logger.info(
        "Snapshot exists at version %d, balance=%.2f",
        snapshot.version, snapshot.state.balance,
    )

    # Add 3 more events
    for i in range(3):
        store.append(account_id, [MoneyDeposited(amount=5.0)], expected_version=11 + i)

    # Second load: starts from snapshot, replays only 3 delta events
    state2 = repo.load(account_id)
    logger.info(
        "After 3 more events: balance=%.2f (expected %.2f)",
        state2.balance, state.balance + 15.0,
    )


def main() -> None:
    store = InMemoryEventStore()

    demo_account_lifecycle(store)
    print()

    demo_optimistic_concurrency(store)
    print()

    demo_time_travel(store)
    print()

    demo_snapshots(store)
    print()

    logger.info(
        "Global log has %d events across %d streams",
        len(store._global_log),
        len(store._streams),
    )
    logger.info("All demos complete.")


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **Event upcasting.** Add a `schema_version=2` variant of `MoneyDeposited` that includes a `currency` field (default `"USD"`). Write an `upcast(event: StoredEvent) -> DomainEvent` function that transparently upgrades v1 events to v2 when reading. Verify that old streams reconstitute correctly with the new handler.

2. **Event stream subscription.** Extend `InMemoryEventStore` with a `subscribe(stream_id, callback)` method. Whenever events are appended to that stream, `callback(stored_events)` is called. Use this to maintain a running balance cache without full reconstitution on every read.

3. **Copy-transform migration.** Write a `migrate_stream(store, stream_id, transform_fn)` function that reads all events from a stream, applies `transform_fn` to each event (returning a new event or the same), and writes the transformed events into a new stream named `f"{stream_id}-v2"`. Run it to add a `currency="USD"` field to all `MoneyDeposited` events.

4. **Aggregate factory with invariants.** Wrap `AccountRepository` in a command-handling layer: `deposit(account_id, amount, reference)` should reject negative amounts and deposits to closed accounts before writing any events. Write unit tests for both the happy path and each rejection case.

5. **Global projection.** Using `read_global()`, build a function `total_deposits_by_owner(store) -> dict[str, float]` that scans the global log once and returns a map of `owner -> sum of deposits`. This previews the read-model projections from Day 3.

*(Solutions: ../solutions/week-04/day-2.md)*

---

## Self-check

- [ ] I can explain why event sourcing gives you time-travel queries for free
- [ ] I can implement `reconstitute` as a pure fold over a list of events
- [ ] I can describe what a snapshot is, why it exists, and why it is safe to delete
- [ ] I can explain optimistic concurrency and the scenario it prevents
- [ ] I can describe three strategies for evolving event schemas

---

## Supplementary Reading

- [Martin Fowler: Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html) — the canonical reference
- [Greg Young: CQRS and Event Sourcing](https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf) — original paper
- [EventStoreDB documentation](https://developers.eventstore.com/server/v24.2/) — production event store implementation reference
- [Versioning in an Event Sourced System](https://leanpub.com/esversioning/read) — Greg Young (free online)
- [Practical Event Sourcing — Mathias Verraes](https://verraes.net/2014/03/practical-event-sourcing/)
