# Solutions — Week 04 · Day 3
# CQRS with events

---

## Exercise 1

**Add a read model to the BankAccount aggregate. Project events into a `dict{account_id: balance}`. Rebuild the read model from scratch from the event log.**

```python
"""
cqrs_read_model.py

Read model (query side of CQRS) for BankAccount.
The read model is a simple in-memory projection updated by replaying events.
It is rebuilt from scratch by calling `rebuild()` with the full event log.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from bank_account_es import (
    AccountOpened,
    DomainEvent,
    MoneyDeposited,
    MoneyWithdrawn,
)

logger = logging.getLogger(__name__)


class BalanceReadModel:
    """
    Projection: account_id -> balance (Decimal).

    The read model is the query side of CQRS. It is a denormalised,
    query-optimised view derived entirely from the event log.
    It never contains logic; it only reflects what has happened.
    """

    def __init__(self) -> None:
        self._balances: dict[str, Decimal] = {}
        self._versions: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Projection: apply one event
    # ------------------------------------------------------------------

    def apply(self, event: DomainEvent) -> None:
        """Update projection state for a single event."""
        account_id = event.aggregate_id

        if isinstance(event, AccountOpened):
            self._balances[account_id] = event.initial_balance
            logger.debug("ReadModel: opened %s balance=%s", account_id, event.initial_balance)

        elif isinstance(event, MoneyDeposited):
            self._balances[account_id] = (
                self._balances.get(account_id, Decimal("0")) + event.amount
            )
            logger.debug("ReadModel: deposit %s amount=%s", account_id, event.amount)

        elif isinstance(event, MoneyWithdrawn):
            self._balances[account_id] = (
                self._balances.get(account_id, Decimal("0")) - event.amount
            )
            logger.debug("ReadModel: withdrawal %s amount=%s", account_id, event.amount)

        else:
            logger.warning("ReadModel: unhandled event type %s", event.event_type)

        self._versions[account_id] = event.sequence

    # ------------------------------------------------------------------
    # Rebuild from scratch
    # ------------------------------------------------------------------

    def rebuild(self, events: list[DomainEvent]) -> None:
        """Discard current state and replay all events from the beginning."""
        self._balances.clear()
        self._versions.clear()
        logger.info("ReadModel: rebuilding from %d events", len(events))
        for event in events:
            self.apply(event)
        logger.info("ReadModel: rebuild complete, %d accounts", len(self._balances))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_balance(self, account_id: str) -> Decimal | None:
        return self._balances.get(account_id)

    def all_balances(self) -> dict[str, Decimal]:
        return dict(self._balances)

    def version(self, account_id: str) -> int:
        return self._versions.get(account_id, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from decimal import Decimal
from bank_account_es import InMemoryEventStore, BankAccountAggregate


def _build_two_account_store() -> tuple[InMemoryEventStore, dict[str, Decimal]]:
    """Helper: create two accounts with known balances."""
    store = InMemoryEventStore()

    agg1 = BankAccountAggregate()
    agg1.open("acc-001", "Alice", Decimal("100.00"))
    agg1.deposit(Decimal("50.00"))
    agg1.withdraw(Decimal("20.00"))
    store.append("acc-001", agg1.pop_pending())

    agg2 = BankAccountAggregate()
    agg2.open("acc-002", "Bob", Decimal("0.00"))
    agg2.deposit(Decimal("200.00"))
    store.append("acc-002", agg2.pop_pending())

    expected = {
        "acc-001": Decimal("130.00"),
        "acc-002": Decimal("200.00"),
    }
    return store, expected


def test_read_model_rebuild_reflects_all_events() -> None:
    store, expected = _build_two_account_store()
    all_events: list[DomainEvent] = []
    for account_id in ("acc-001", "acc-002"):
        all_events.extend(store.load(account_id))

    read_model = BalanceReadModel()
    read_model.rebuild(all_events)

    for account_id, balance in expected.items():
        assert read_model.get_balance(account_id) == balance


def test_read_model_incremental_apply() -> None:
    store, expected = _build_two_account_store()
    read_model = BalanceReadModel()

    for account_id in ("acc-001", "acc-002"):
        for event in store.load(account_id):
            read_model.apply(event)

    assert read_model.get_balance("acc-001") == expected["acc-001"]
    assert read_model.get_balance("acc-002") == expected["acc-002"]


def test_read_model_rebuild_is_idempotent() -> None:
    store, expected = _build_two_account_store()
    all_events: list[DomainEvent] = []
    for account_id in ("acc-001", "acc-002"):
        all_events.extend(store.load(account_id))

    read_model = BalanceReadModel()
    read_model.rebuild(all_events)
    read_model.rebuild(all_events)  # rebuild twice

    for account_id, balance in expected.items():
        assert read_model.get_balance(account_id) == balance
```

---

## Exercise 2

**Implement a command handler that validates and dispatches commands to the aggregate, then persists events.**

```python
"""
cqrs_command_handler.py

Command side of CQRS.
Commands are validated, dispatched to the aggregate, and the resulting
events are persisted to the event store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from bank_account_es import BankAccountAggregate, DomainEvent, InMemoryEventStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commands (Plain data objects — no behaviour)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenAccountCommand:
    account_id: str
    owner: str
    initial_balance: Decimal = Decimal("0.00")


@dataclass(frozen=True)
class DepositCommand:
    account_id: str
    amount: Decimal


@dataclass(frozen=True)
class WithdrawCommand:
    account_id: str
    amount: Decimal


Command = OpenAccountCommand | DepositCommand | WithdrawCommand


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


class BankAccountCommandHandler:
    """
    Validates commands, loads the aggregate from the event store,
    executes the business logic, and persists new events.

    This is the write side of CQRS. It has no knowledge of read models.
    """

    def __init__(self, store: InMemoryEventStore) -> None:
        self._store = store

    def handle(self, command: Command) -> list[DomainEvent]:
        """
        Dispatch command, return the list of newly emitted events.
        Raises ValueError on validation failures.
        """
        self._validate(command)

        if isinstance(command, OpenAccountCommand):
            return self._handle_open(command)
        elif isinstance(command, DepositCommand):
            return self._handle_deposit(command)
        elif isinstance(command, WithdrawCommand):
            return self._handle_withdraw(command)
        else:
            raise TypeError(f"Unknown command type: {type(command)}")

    # ------------------------------------------------------------------
    # Validation (before loading the aggregate — cheap checks first)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(command: Command) -> None:
        if isinstance(command, OpenAccountCommand):
            if not command.account_id:
                raise ValueError("account_id is required")
            if not command.owner:
                raise ValueError("owner is required")
            if command.initial_balance < Decimal("0"):
                raise ValueError("initial_balance cannot be negative")
        elif isinstance(command, DepositCommand | WithdrawCommand):
            if not command.account_id:
                raise ValueError("account_id is required")
            if command.amount <= Decimal("0"):
                raise ValueError("amount must be positive")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _load(self, account_id: str) -> BankAccountAggregate:
        events = self._store.load(account_id)
        return BankAccountAggregate.reconstitute(events)

    def _save(self, account_id: str, agg: BankAccountAggregate) -> list[DomainEvent]:
        new_events = agg.pop_pending()
        self._store.append(account_id, new_events)
        logger.info(
            "Persisted %d events for %s (version=%d)",
            len(new_events),
            account_id,
            agg.state.version,
        )
        return new_events

    def _handle_open(self, cmd: OpenAccountCommand) -> list[DomainEvent]:
        existing = self._store.load(cmd.account_id)
        if existing:
            raise ValueError(f"Account {cmd.account_id} already exists")
        agg = BankAccountAggregate()
        agg.open(cmd.account_id, cmd.owner, cmd.initial_balance)
        return self._save(cmd.account_id, agg)

    def _handle_deposit(self, cmd: DepositCommand) -> list[DomainEvent]:
        agg = self._load(cmd.account_id)
        agg.deposit(cmd.amount)
        return self._save(cmd.account_id, agg)

    def _handle_withdraw(self, cmd: WithdrawCommand) -> list[DomainEvent]:
        agg = self._load(cmd.account_id)
        agg.withdraw(cmd.amount)
        return self._save(cmd.account_id, agg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


def test_open_account_command_persists_event() -> None:
    store = InMemoryEventStore()
    handler = BankAccountCommandHandler(store)

    events = handler.handle(
        OpenAccountCommand("acc-cmd-001", "Heidi", Decimal("500.00"))
    )

    assert len(events) == 1
    assert events[0].event_type == "AccountOpened"
    assert len(store.load("acc-cmd-001")) == 1


def test_deposit_command_updates_store() -> None:
    store = InMemoryEventStore()
    handler = BankAccountCommandHandler(store)
    handler.handle(OpenAccountCommand("acc-cmd-002", "Ivan", Decimal("0.00")))

    events = handler.handle(DepositCommand("acc-cmd-002", Decimal("250.00")))

    assert events[0].event_type == "MoneyDeposited"
    assert len(store.load("acc-cmd-002")) == 2


def test_withdraw_more_than_balance_raises() -> None:
    store = InMemoryEventStore()
    handler = BankAccountCommandHandler(store)
    handler.handle(OpenAccountCommand("acc-cmd-003", "Judy", Decimal("50.00")))

    with pytest.raises(ValueError, match="Insufficient funds"):
        handler.handle(WithdrawCommand("acc-cmd-003", Decimal("100.00")))


def test_duplicate_open_raises() -> None:
    store = InMemoryEventStore()
    handler = BankAccountCommandHandler(store)
    handler.handle(OpenAccountCommand("acc-cmd-004", "Karl", Decimal("0.00")))

    with pytest.raises(ValueError, match="already exists"):
        handler.handle(OpenAccountCommand("acc-cmd-004", "Karl", Decimal("0.00")))


def test_validation_rejects_negative_amount() -> None:
    store = InMemoryEventStore()
    handler = BankAccountCommandHandler(store)
    with pytest.raises(ValueError, match="amount must be positive"):
        handler.handle(DepositCommand("acc-cmd-005", Decimal("-10.00")))
```

---

## Exercise 3

**Show eventual consistency: command writes to event log, projection updates asynchronously. Write a test that asserts the projection is stale immediately after a command but consistent after a sync.**

```python
"""
cqrs_eventual_consistency.py

Demonstrates eventual consistency in CQRS:
- Command handler writes events to the store synchronously.
- The read model is updated by an async projector that processes an event queue.
- Immediately after a command the read model may be stale.
- After the projector drains the queue the read model is consistent.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal

from bank_account_es import DomainEvent, InMemoryEventStore
from cqrs_command_handler import (
    BankAccountCommandHandler,
    DepositCommand,
    OpenAccountCommand,
)
from cqrs_read_model import BalanceReadModel

logger = logging.getLogger(__name__)


class AsyncProjector:
    """
    Async read-model updater. Events are pushed onto an asyncio.Queue by the
    command handler bridge, then processed in the background.
    The projector deliberately introduces a small artificial delay to make
    eventual consistency observable in tests.
    """

    def __init__(
        self,
        read_model: BalanceReadModel,
        delay_seconds: float = 0.05,
    ) -> None:
        self._read_model = read_model
        self._queue: asyncio.Queue[DomainEvent] = asyncio.Queue()
        self._delay = delay_seconds
        self._task: asyncio.Task[None] | None = None

    def enqueue(self, events: list[DomainEvent]) -> None:
        for event in events:
            self._queue.put_nowait(event)

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            await asyncio.sleep(self._delay)  # simulated async lag
            self._read_model.apply(event)
            logger.debug("Projector applied %s", event.event_type)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="projector")

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def drain(self) -> None:
        """Wait until all currently queued events have been projected."""
        await self._queue.join()

    # Override Queue.join to mark tasks done after apply
    # We do this by re-implementing with task_done tracking:
    def enqueue_tracked(self, events: list[DomainEvent]) -> None:
        for event in events:
            self._queue.put_nowait(event)

    async def _run_tracked(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await asyncio.sleep(self._delay)
                self._read_model.apply(event)
            finally:
                self._queue.task_done()

    def start_tracked(self) -> None:
        self._task = asyncio.create_task(
            self._run_tracked(), name="projector-tracked"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.asyncio
async def test_projection_stale_immediately_consistent_after_drain() -> None:
    """
    Core eventual consistency test:
    1. Issue a command -> event persisted immediately.
    2. Immediately check read model -> stale (projection not yet applied).
    3. Await drain -> read model updated.
    4. Check read model again -> consistent.
    """
    store = InMemoryEventStore()
    read_model = BalanceReadModel()
    projector = AsyncProjector(read_model, delay_seconds=0.05)
    projector.start_tracked()

    handler = BankAccountCommandHandler(store)

    # Open account -> event emitted, enqueue for projection
    open_events = handler.handle(
        OpenAccountCommand("acc-ec-001", "Lena", Decimal("0.00"))
    )
    projector.enqueue_tracked(open_events)

    # Immediately: read model has NOT yet seen the event
    assert read_model.get_balance("acc-ec-001") is None, (
        "Read model should be stale immediately after command"
    )

    # Drain: projector processes the event
    await projector.drain()

    # Now consistent
    assert read_model.get_balance("acc-ec-001") == Decimal("0.00"), (
        "Read model should be consistent after drain"
    )

    # Issue deposit
    deposit_events = handler.handle(
        DepositCommand("acc-ec-001", Decimal("300.00"))
    )
    projector.enqueue_tracked(deposit_events)

    # Immediately: still shows old balance
    assert read_model.get_balance("acc-ec-001") == Decimal("0.00"), (
        "Read model should still show pre-deposit balance"
    )

    # After drain: consistent
    await projector.drain()
    assert read_model.get_balance("acc-ec-001") == Decimal("300.00")

    projector.stop()


@pytest.mark.asyncio
async def test_multiple_commands_eventually_consistent() -> None:
    """
    Fire multiple commands in quick succession; read model converges
    after a single drain call.
    """
    store = InMemoryEventStore()
    read_model = BalanceReadModel()
    projector = AsyncProjector(read_model, delay_seconds=0.01)
    projector.start_tracked()

    handler = BankAccountCommandHandler(store)

    events = handler.handle(
        OpenAccountCommand("acc-ec-002", "Marco", Decimal("1000.00"))
    )
    projector.enqueue_tracked(events)

    for _ in range(5):
        events = handler.handle(DepositCommand("acc-ec-002", Decimal("100.00")))
        projector.enqueue_tracked(events)

    # Read model is stale: balance < 1500
    # (may have received some events already, but not necessarily all)

    await projector.drain()
    assert read_model.get_balance("acc-ec-002") == Decimal("1500.00")

    projector.stop()
```

---

## Discussion notes

- **CQRS is not mandatory with event sourcing**: They complement each other well but are independent patterns. You can use event sourcing without CQRS (single model for reads and writes) or CQRS without event sourcing (separate read/write databases with synchronisation).

- **Why commands are plain data objects**: Commands are intent, not behaviour. Keeping them as frozen dataclasses means they are serializable, loggable, and replayable. Never put validation logic inside a command class; validation belongs in the command handler.

- **Eventual consistency implications for the UI**:
  - After submitting a form (command), the UI must not immediately re-query the read model — it will be stale.
  - Common patterns: optimistic UI update (assume success), polling until version advances, or WebSocket push from the projector.
  - Inform users: "Your request is being processed" rather than showing a stale state as truth.

- **Projector idempotency**: Projectors must handle duplicate events safely. If the same event is applied twice (e.g., after a crash and replay), the read model must reach the same state. Use `_versions` tracking (as in `BalanceReadModel`) to skip events at or below the already-seen version.

- **Read model rebuild vs. incremental update**:
  - Full rebuild is safe and simple but expensive for large logs.
  - Incremental update is efficient but requires careful at-least-once delivery from the event store.
  - In production, combine: rebuild on deployment from a known checkpoint; apply incremental updates in steady state.

- **Command handler optimistic concurrency**: The handler loads the aggregate, runs the command, and saves. If another writer modifies the aggregate between load and save, the second writer wins (last-write wins) in the simple implementation shown here. In production, pass `expected_version` to `store.append` and retry on conflict.

- **Separation of command and query buses**: In larger systems, commands go through a command bus (sync, returns acknowledgement) and events go through an event bus (async, fires-and-forgets to all projectors). The two buses decouple the write path latency from the read model update latency entirely.

---
