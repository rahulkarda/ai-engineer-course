# Solutions — Week 04 · Day 2
# Event sourcing

---

## Exercise 1

**Implement a BankAccount aggregate using event sourcing. Events: AccountOpened, MoneyDeposited, MoneyWithdrawn. Reconstitute balance from event log.**

```python
"""
bank_account_es.py

BankAccount aggregate implemented with event sourcing.
The aggregate never stores mutable state directly; all state is derived
by replaying the ordered event log.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainEvent(ABC):
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    aggregate_id: str = ""
    sequence: int = 0

    @property
    @abstractmethod
    def event_type(self) -> str: ...


# ---------------------------------------------------------------------------
# BankAccount events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountOpened(DomainEvent):
    owner: str = ""
    initial_balance: Decimal = Decimal("0.00")

    @property
    def event_type(self) -> str:
        return "AccountOpened"


@dataclass(frozen=True)
class MoneyDeposited(DomainEvent):
    amount: Decimal = Decimal("0.00")

    @property
    def event_type(self) -> str:
        return "MoneyDeposited"


@dataclass(frozen=True)
class MoneyWithdrawn(DomainEvent):
    amount: Decimal = Decimal("0.00")

    @property
    def event_type(self) -> str:
        return "MoneyWithdrawn"


# ---------------------------------------------------------------------------
# Aggregate state (pure, immutable)
# ---------------------------------------------------------------------------


@dataclass
class BankAccountState:
    account_id: str = ""
    owner: str = ""
    balance: Decimal = Decimal("0.00")
    is_open: bool = False
    version: int = 0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class BankAccountAggregate:
    """
    Event-sourced BankAccount aggregate.

    Rules:
    - State is never persisted directly; only events are persisted.
    - `apply()` transitions state given a single event.
    - `reconstitute()` rebuilds state from an ordered event list.
    - Command methods (`open`, `deposit`, `withdraw`) validate and emit events.
    """

    def __init__(self) -> None:
        self._state = BankAccountState()
        self._pending: list[DomainEvent] = []

    # ------------------------------------------------------------------
    # State projection
    # ------------------------------------------------------------------

    @staticmethod
    def _apply(state: BankAccountState, event: DomainEvent) -> BankAccountState:
        """Pure function: state + event -> next state."""
        if isinstance(event, AccountOpened):
            state.account_id = event.aggregate_id
            state.owner = event.owner
            state.balance = event.initial_balance
            state.is_open = True
        elif isinstance(event, MoneyDeposited):
            state.balance += event.amount
        elif isinstance(event, MoneyWithdrawn):
            state.balance -= event.amount
        else:
            logger.warning("Unknown event type: %s", event.event_type)
        state.version = event.sequence
        return state

    @classmethod
    def reconstitute(cls, events: list[DomainEvent]) -> "BankAccountAggregate":
        """Rebuild aggregate from a full event log."""
        aggregate = cls()
        for event in events:
            aggregate._state = cls._apply(aggregate._state, event)
        logger.debug(
            "Reconstituted account %s at version %d balance=%s",
            aggregate._state.account_id,
            aggregate._state.version,
            aggregate._state.balance,
        )
        return aggregate

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def open(
        self,
        account_id: str,
        owner: str,
        initial_balance: Decimal = Decimal("0.00"),
    ) -> None:
        if self._state.is_open:
            raise ValueError("Account already open")
        if initial_balance < Decimal("0"):
            raise ValueError("Initial balance cannot be negative")
        event = AccountOpened(
            aggregate_id=account_id,
            owner=owner,
            initial_balance=initial_balance,
            sequence=self._state.version + 1,
        )
        self._raise(event)

    def deposit(self, amount: Decimal) -> None:
        self._assert_open()
        if amount <= Decimal("0"):
            raise ValueError("Deposit amount must be positive")
        event = MoneyDeposited(
            aggregate_id=self._state.account_id,
            amount=amount,
            sequence=self._state.version + 1,
        )
        self._raise(event)

    def withdraw(self, amount: Decimal) -> None:
        self._assert_open()
        if amount <= Decimal("0"):
            raise ValueError("Withdrawal amount must be positive")
        if self._state.balance < amount:
            raise ValueError(
                f"Insufficient funds: balance={self._state.balance} requested={amount}"
            )
        event = MoneyWithdrawn(
            aggregate_id=self._state.account_id,
            amount=amount,
            sequence=self._state.version + 1,
        )
        self._raise(event)

    def _raise(self, event: DomainEvent) -> None:
        self._state = self._apply(self._state, event)
        self._pending.append(event)

    def _assert_open(self) -> None:
        if not self._state.is_open:
            raise ValueError("Account is not open")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> BankAccountState:
        return self._state

    def pop_pending(self) -> list[DomainEvent]:
        events, self._pending = self._pending, []
        return events


# ---------------------------------------------------------------------------
# In-memory event store
# ---------------------------------------------------------------------------


class InMemoryEventStore:
    def __init__(self) -> None:
        self._streams: dict[str, list[DomainEvent]] = {}

    def append(self, aggregate_id: str, events: list[DomainEvent]) -> None:
        if aggregate_id not in self._streams:
            self._streams[aggregate_id] = []
        self._streams[aggregate_id].extend(events)

    def load(self, aggregate_id: str) -> list[DomainEvent]:
        return list(self._streams.get(aggregate_id, []))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest
from decimal import Decimal


def test_reconstitute_balance_from_event_log() -> None:
    store = InMemoryEventStore()
    account_id = "acc-001"

    # Execute commands
    agg = BankAccountAggregate()
    agg.open(account_id, "Alice", Decimal("100.00"))
    agg.deposit(Decimal("50.00"))
    agg.withdraw(Decimal("30.00"))
    events = agg.pop_pending()
    store.append(account_id, events)

    # Reconstitute from scratch
    raw_events = store.load(account_id)
    restored = BankAccountAggregate.reconstitute(raw_events)

    assert restored.state.balance == Decimal("120.00")
    assert restored.state.owner == "Alice"
    assert restored.state.is_open is True
    assert restored.state.version == 3


def test_insufficient_funds_raises() -> None:
    agg = BankAccountAggregate()
    agg.open("acc-002", "Bob", Decimal("10.00"))
    with pytest.raises(ValueError, match="Insufficient funds"):
        agg.withdraw(Decimal("50.00"))


def test_deposit_must_be_positive() -> None:
    agg = BankAccountAggregate()
    agg.open("acc-003", "Carol", Decimal("0.00"))
    with pytest.raises(ValueError):
        agg.deposit(Decimal("-5.00"))
```

---

## Exercise 2

**Add snapshotting: after every N events, save a snapshot. On reconstitution, load latest snapshot then replay only newer events.**

```python
"""
bank_account_snapshots.py

Extends bank_account_es.py with periodic snapshotting.
After every N events the store saves the current aggregate state as a snapshot.
On reconstitution, the latest snapshot is loaded and only events with
sequence > snapshot.version are replayed.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal

# Re-use types from Exercise 1
from bank_account_es import (
    BankAccountAggregate,
    BankAccountState,
    DomainEvent,
    InMemoryEventStore,
)

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    aggregate_id: str
    version: int
    state: BankAccountState


class SnapshotStore:
    def __init__(self) -> None:
        # account_id -> latest Snapshot
        self._snapshots: dict[str, Snapshot] = {}

    def save(self, snapshot: Snapshot) -> None:
        self._snapshots[snapshot.aggregate_id] = snapshot
        logger.info(
            "Saved snapshot for %s at version %d",
            snapshot.aggregate_id,
            snapshot.version,
        )

    def latest(self, aggregate_id: str) -> Snapshot | None:
        return self._snapshots.get(aggregate_id)


class SnapshotEventStore(InMemoryEventStore):
    """
    Event store that automatically creates snapshots every `snapshot_every`
    events and uses them during reconstitution.
    """

    def __init__(self, snapshot_every: int = 5) -> None:
        super().__init__()
        self._snapshot_store = SnapshotStore()
        self._snapshot_every = snapshot_every

    def append_with_snapshot(
        self,
        aggregate_id: str,
        events: list[DomainEvent],
        current_state: BankAccountState,
    ) -> None:
        self.append(aggregate_id, events)
        if current_state.version % self._snapshot_every == 0:
            snapshot = Snapshot(
                aggregate_id=aggregate_id,
                version=current_state.version,
                state=deepcopy(current_state),
            )
            self._snapshot_store.save(snapshot)

    def reconstitute(self, aggregate_id: str) -> BankAccountAggregate:
        snapshot = self._snapshot_store.latest(aggregate_id)
        all_events = self.load(aggregate_id)

        if snapshot is not None:
            # Start from snapshot state, replay only newer events
            agg = BankAccountAggregate()
            agg._state = deepcopy(snapshot.state)
            tail_events = [
                e for e in all_events if e.sequence > snapshot.version
            ]
            logger.debug(
                "Restoring %s from snapshot@v%d + %d tail events",
                aggregate_id,
                snapshot.version,
                len(tail_events),
            )
            for event in tail_events:
                agg._state = BankAccountAggregate._apply(agg._state, event)
        else:
            agg = BankAccountAggregate.reconstitute(all_events)

        return agg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_snapshot_created_at_boundary() -> None:
    store = SnapshotEventStore(snapshot_every=3)
    account_id = "acc-snap-001"

    agg = BankAccountAggregate()
    agg.open(account_id, "Dave", Decimal("0.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)

    agg.deposit(Decimal("10.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)

    agg.deposit(Decimal("20.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)
    # version==3, snapshot_every==3 => snapshot should be saved now

    snapshot = store._snapshot_store.latest(account_id)
    assert snapshot is not None
    assert snapshot.version == 3
    assert snapshot.state.balance == Decimal("30.00")


def test_reconstitution_uses_snapshot() -> None:
    store = SnapshotEventStore(snapshot_every=3)
    account_id = "acc-snap-002"

    agg = BankAccountAggregate()
    agg.open(account_id, "Eve", Decimal("100.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)

    agg.deposit(Decimal("50.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)

    agg.withdraw(Decimal("20.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)
    # snapshot at v3 balance=130

    agg.deposit(Decimal("5.00"))
    store.append_with_snapshot(account_id, agg.pop_pending(), agg.state)
    # v4, only replayed from snapshot

    restored = store.reconstitute(account_id)
    assert restored.state.balance == Decimal("135.00")
    assert restored.state.version == 4


def test_reconstitution_without_snapshot_falls_back_to_full_replay() -> None:
    store = SnapshotEventStore(snapshot_every=100)  # threshold never reached
    account_id = "acc-snap-003"

    agg = BankAccountAggregate()
    agg.open(account_id, "Frank", Decimal("0.00"))
    agg.deposit(Decimal("99.00"))
    store.append(account_id, agg.pop_pending())

    restored = store.reconstitute(account_id)
    assert restored.state.balance == Decimal("99.00")
```

---

## Exercise 3

**Implement a time-travel query: given an event log and a target timestamp, return the aggregate state at that point.**

```python
"""
bank_account_time_travel.py

Time-travel query: replays events up to a target UTC timestamp and
returns the aggregate state at that point in time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from bank_account_es import (
    BankAccountAggregate,
    BankAccountState,
    DomainEvent,
)

logger = logging.getLogger(__name__)


def state_at(
    events: list[DomainEvent],
    as_of: datetime,
) -> BankAccountState:
    """
    Replay events that occurred at or before `as_of` and return the
    resulting aggregate state.

    Args:
        events: Ordered list of domain events (ascending by sequence).
        as_of: UTC datetime. Events with occurred_at > as_of are excluded.

    Returns:
        BankAccountState at `as_of`. Returns an empty state if no events
        qualify (account did not exist yet at that point).
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (use UTC)")

    filtered = [e for e in events if e.occurred_at <= as_of]

    if not filtered:
        logger.debug("No events before %s; returning empty state", as_of)
        return BankAccountState()

    aggregate = BankAccountAggregate.reconstitute(filtered)
    logger.info(
        "Time-travel to %s: replayed %d/%d events, balance=%s",
        as_of.isoformat(),
        len(filtered),
        len(events),
        aggregate.state.balance,
    )
    return aggregate.state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest
from datetime import timedelta
from decimal import Decimal


def _make_account_events() -> list[DomainEvent]:
    """Helper: create an ordered event log with known timestamps."""
    from bank_account_es import (
        AccountOpened,
        MoneyDeposited,
        MoneyWithdrawn,
    )

    t0 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    events: list[DomainEvent] = [
        AccountOpened(
            aggregate_id="acc-tt-001",
            owner="Grace",
            initial_balance=Decimal("200.00"),
            sequence=1,
            occurred_at=t0,
        ),
        MoneyDeposited(
            aggregate_id="acc-tt-001",
            amount=Decimal("100.00"),
            sequence=2,
            occurred_at=t0 + timedelta(hours=1),
        ),
        MoneyWithdrawn(
            aggregate_id="acc-tt-001",
            amount=Decimal("50.00"),
            sequence=3,
            occurred_at=t0 + timedelta(hours=2),
        ),
        MoneyDeposited(
            aggregate_id="acc-tt-001",
            amount=Decimal("75.00"),
            sequence=4,
            occurred_at=t0 + timedelta(hours=3),
        ),
    ]
    return events


def test_time_travel_returns_state_at_open() -> None:
    events = _make_account_events()
    t0 = events[0].occurred_at
    s = state_at(events, as_of=t0)
    assert s.balance == Decimal("200.00")
    assert s.version == 1


def test_time_travel_midway() -> None:
    events = _make_account_events()
    # Between deposit (t+1h) and withdrawal (t+2h)
    midpoint = events[0].occurred_at + timedelta(hours=1, minutes=30)
    s = state_at(events, as_of=midpoint)
    assert s.balance == Decimal("300.00")  # 200 + 100
    assert s.version == 2


def test_time_travel_full_log() -> None:
    events = _make_account_events()
    future = events[-1].occurred_at + timedelta(days=1)
    s = state_at(events, as_of=future)
    assert s.balance == Decimal("325.00")  # 200+100-50+75


def test_time_travel_before_account_created_returns_empty() -> None:
    events = _make_account_events()
    past = events[0].occurred_at - timedelta(seconds=1)
    s = state_at(events, as_of=past)
    assert s.account_id == ""
    assert s.is_open is False


def test_time_travel_requires_timezone_aware_datetime() -> None:
    events = _make_account_events()
    naive = datetime(2024, 1, 1, 10, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        state_at(events, as_of=naive)
```

---

## Discussion notes

- **Event identity**: Each event carries a stable `event_id` (UUID). Never deduplicate events by sequence alone; two concurrent writers could emit the same sequence number. Use the event_id for idempotency checks.

- **Sequence vs. timestamp for ordering**: Sequence numbers are assigned by the event store at append time and are the canonical ordering key. Timestamps (as used in time-travel) are wall-clock times and can have clock skew between nodes — never sort events by `occurred_at` for replay correctness; use `sequence`. Time-travel uses timestamps only for external queries.

- **Snapshotting trade-offs**:
  - Snapshots improve read performance for large aggregates but add storage and complexity.
  - Snapshots must include the version/sequence so the store knows which tail events to replay.
  - A snapshot is a cache; if the state model changes (new field), old snapshots may be invalid. Version your snapshot schema or always fall back to full replay on deserialization error.
  - Prefer snapshot-on-write (after N events) over snapshot-on-read to keep reads fast.

- **Pure apply function**: The `_apply` method is intentionally a static method that takes state as a parameter. This makes it trivially testable and prevents accidental mutation of fields not touched by a particular event.

- **Open/Closed vs. mutable dataclass**: `BankAccountState` is a mutable dataclass here to keep `_apply` simple. In production, consider making it immutable (`frozen=True`) and returning a new instance from each apply call. This allows safe sharing of state objects across threads and makes the event sourcing pattern structurally explicit.

- **Event store optimistic concurrency**: The real `append` should accept an `expected_version` parameter and raise a concurrency exception if the current version does not match. This prevents lost updates when two processes read the same aggregate and both try to write. This is omitted here for brevity but is essential in production.

---
