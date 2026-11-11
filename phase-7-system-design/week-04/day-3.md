# Phase 7 · Week 04 · Day 3 — CQRS with Events

> Objective: Implement Command Query Responsibility Segregation on top of an event-sourced store — separate write and read models, build eventual-consistent projections, and sketch the Saga pattern for distributed workflows.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-2]]

---

## Concept

**CQRS** splits a single model into two:

- **Command side (write model)** — handles state-changing operations. Enforces business invariants. Emits events. Optimized for consistency and correctness.
- **Query side (read model)** — handles reads. Built from events. Optimized for the exact query shape needed. May be denormalized, pre-aggregated, cached.

The two sides communicate only through events. The write model never queries the read model; the read model never writes to the command side.

### Why the Split

A single model that is optimal for both writes and reads usually exists only for simple domains. As complexity grows:

- Write model needs strict validation and consistency; reads need different indexing and aggregations
- Reads vastly outnumber writes in most systems (read:write ratios of 100:1 are common)
- Different teams can evolve write and read models independently
- Read models can be rebuilt from scratch by replaying events

```
Write side:                          Read side:
  Command                              Projection
  -> Validate invariants               -> Listen for events
  -> Load aggregate (Day 2)            -> Update read store
  -> Emit events                       -> Serve queries

  Commands are imperative verbs:       Read models are view-optimized:
  OpenAccount, Deposit, Withdraw       AccountSummaryView, OwnerBalanceView
```

### Eventual Consistency

The read model lags behind the write model by the time it takes to process events. This is **eventual consistency**: the read model will be consistent with the write model, eventually — typically milliseconds to seconds.

Implication for callers: a command response (write) may not be immediately visible in a subsequent query. Clients must tolerate or handle this (e.g., return the written state directly from the command handler for immediate feedback, or use polling/WebSocket for updates).

### Projection Rebuilding

Because read models are derived from events, they can be dropped and rebuilt at any time:

```
rebuild(projection, event_store):
  1. Clear projection store
  2. Set checkpoint = 0
  3. Read events from global log starting at checkpoint
  4. For each event: apply projection handler
  5. Advance checkpoint
  6. Repeat until global log is exhausted
```

Rebuilding enables: schema migrations, bug fixes in projections, new aggregations — without touching the source of truth.

### Sagas (intro)

A **saga** coordinates a multi-step, multi-aggregate workflow in an eventually consistent system. It reacts to events and issues commands. Two flavors:

**Choreography** — services emit events; each reacts to others' events directly. No central coordinator. Simple, but hard to trace.

**Orchestration** — a saga object (process manager) listens to events and explicitly commands each step. Easier to trace; single point of coordination.

```
Order Saga (Orchestration):
  1. OrderPlaced event  ->  saga starts
     -> issues: ReserveInventory command
  2. InventoryReserved event -> saga continues
     -> issues: ChargeCreditCard command
  3. PaymentCharged event -> saga continues
     -> issues: ShipOrder command
  4. OrderShipped event -> saga completes

  On failure at any step:
     -> issues compensating commands (ReleaseInventory, RefundPayment)
```

---

## How It Works Internally

### Projection Pipeline

```
Event Store (global log)
     |
     | StoredEvent stream
     v
Projection Dispatcher
     |
     |--- AccountSummaryProjection.handle(event)
     |       -> updates AccountSummaryView store
     |
     |--- OwnerBalanceProjection.handle(event)
             -> updates OwnerBalanceView store

Each projection maintains a checkpoint (global_position last processed).
On restart, replay only from checkpoint forward.
```

### Read Store Options

| Store Type | Use When |
|---|---|
| In-memory dict | Tests, single-node, non-critical |
| Redis hash | Fast point lookups, sessions |
| PostgreSQL | Complex queries, joins |
| Elasticsearch | Full-text, faceted search |
| ClickHouse / BigQuery | Analytics, aggregations |

The read store is chosen based on the query shape, not the write shape — this is the core benefit.

### Command Handler Pattern

```
CommandHandler.handle(command):
  1. Load aggregate from event store (Day 2 reconstitution)
  2. Execute business logic on aggregate
  3. Collect new domain events
  4. Append events to event store (optimistic lock)
  5. Return command result (optional: include new state for immediate read)
```

Write side never queries the read store. Read store never calls command handlers. Communication only via event log.

---

## Worked Example

```python
"""
cqrs_demo.py — CQRS with event sourcing, read model projections, and a
basic saga orchestrator.

Builds on the event store from day-2. All code in one file for standalone run.

Run:
    python cqrs_demo.py

No external dependencies beyond Python 3.12 stdlib.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("cqrs_demo")


# ===========================================================================
# Reuse event store from Day 2 (inline here for standalone execution)
# ===========================================================================

@dataclass(frozen=True)
class DomainEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)


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


@dataclass(frozen=True)
class StoredEvent:
    stream_id: str
    sequence: int
    global_position: int
    event: DomainEvent


class ConcurrencyError(Exception):
    pass


class InMemoryEventStore:
    def __init__(self) -> None:
        self._streams: dict[str, list[StoredEvent]] = {}
        self._global_log: list[StoredEvent] = []

    def append(
        self,
        stream_id: str,
        events: list[DomainEvent],
        expected_version: int = -2,
    ) -> int:
        if stream_id not in self._streams:
            self._streams[stream_id] = []

        current_version = len(self._streams[stream_id]) - 1
        # -2 is the "skip check" sentinel
        if expected_version != -2 and expected_version != current_version:
            raise ConcurrencyError(
                f"Stream '{stream_id}': expected {expected_version}, actual {current_version}"
            )

        for event in events:
            seq = len(self._streams[stream_id])
            gpos = len(self._global_log)
            stored = StoredEvent(stream_id, seq, gpos, event)
            self._streams[stream_id].append(stored)
            self._global_log.append(stored)

        return len(self._streams[stream_id]) - 1

    def read_stream(self, stream_id: str, from_version: int = 0) -> list[StoredEvent]:
        return self._streams.get(stream_id, [])[from_version:]

    def read_global(self, from_position: int = 0) -> list[StoredEvent]:
        return self._global_log[from_position:]

    def stream_version(self, stream_id: str) -> int:
        return len(self._streams.get(stream_id, [])) - 1


# ===========================================================================
# Write side: aggregate state + command handlers
# ===========================================================================

@dataclass
class AccountState:
    account_id: str
    owner: str = ""
    balance: float = 0.0
    is_closed: bool = False
    version: int = -1


def apply_event(state: AccountState, event: DomainEvent) -> AccountState:
    match event:
        case AccountOpened(owner=o, initial_balance=b):
            return AccountState(state.account_id, o, b, False, state.version + 1)
        case MoneyDeposited(amount=a):
            return AccountState(state.account_id, state.owner,
                                state.balance + a, state.is_closed, state.version + 1)
        case MoneyWithdrawn(amount=a):
            return AccountState(state.account_id, state.owner,
                                state.balance - a, state.is_closed, state.version + 1)
        case AccountClosed():
            return AccountState(state.account_id, state.owner,
                                state.balance, True, state.version + 1)
        case _:
            return state


def reconstitute(account_id: str, events: list[StoredEvent]) -> AccountState:
    state = AccountState(account_id=account_id)
    for se in events:
        state = apply_event(state, se.event)
    return state


# ---------------------------------------------------------------------------
# Commands — value objects expressing intent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpenAccountCommand:
    account_id: str
    owner: str
    initial_balance: float = 0.0


@dataclass(frozen=True)
class DepositCommand:
    account_id: str
    amount: float
    reference: str = ""


@dataclass(frozen=True)
class WithdrawCommand:
    account_id: str
    amount: float
    reference: str = ""


@dataclass(frozen=True)
class CloseAccountCommand:
    account_id: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Command result
# ---------------------------------------------------------------------------

@dataclass
class CommandResult:
    success: bool
    account_id: str
    new_version: int
    error: str = ""
    # Return current state so caller doesn't need to query read model immediately
    state: AccountState | None = None


# ---------------------------------------------------------------------------
# Command handler — enforces invariants, emits events
# ---------------------------------------------------------------------------

class AccountCommandHandler:
    def __init__(self, store: InMemoryEventStore) -> None:
        self._store = store
        self._log = logging.getLogger("cmd_handler")

    def _load(self, account_id: str) -> AccountState:
        events = self._store.read_stream(account_id)
        return reconstitute(account_id, events)

    def handle_open(self, cmd: OpenAccountCommand) -> CommandResult:
        if self._store.stream_version(cmd.account_id) >= 0:
            return CommandResult(False, cmd.account_id, -1, "Account already exists")
        if cmd.initial_balance < 0:
            return CommandResult(False, cmd.account_id, -1, "Initial balance cannot be negative")

        events = [AccountOpened(owner=cmd.owner, initial_balance=cmd.initial_balance)]
        version = self._store.append(cmd.account_id, events, expected_version=-1)
        state = self._load(cmd.account_id)
        self._log.info("Account %s opened for %s", cmd.account_id, cmd.owner)
        return CommandResult(True, cmd.account_id, version, state=state)

    def handle_deposit(self, cmd: DepositCommand) -> CommandResult:
        state = self._load(cmd.account_id)
        if state.version < 0:
            return CommandResult(False, cmd.account_id, -1, "Account does not exist")
        if state.is_closed:
            return CommandResult(False, cmd.account_id, state.version, "Account is closed")
        if cmd.amount <= 0:
            return CommandResult(False, cmd.account_id, state.version, "Amount must be positive")

        events = [MoneyDeposited(amount=cmd.amount, reference=cmd.reference)]
        version = self._store.append(cmd.account_id, events, expected_version=state.version)
        new_state = self._load(cmd.account_id)
        self._log.info("Deposited %.2f to %s (balance: %.2f)", cmd.amount, cmd.account_id, new_state.balance)
        return CommandResult(True, cmd.account_id, version, state=new_state)

    def handle_withdraw(self, cmd: WithdrawCommand) -> CommandResult:
        state = self._load(cmd.account_id)
        if state.version < 0:
            return CommandResult(False, cmd.account_id, -1, "Account does not exist")
        if state.is_closed:
            return CommandResult(False, cmd.account_id, state.version, "Account is closed")
        if cmd.amount <= 0:
            return CommandResult(False, cmd.account_id, state.version, "Amount must be positive")
        if state.balance < cmd.amount:
            return CommandResult(False, cmd.account_id, state.version,
                                 f"Insufficient funds: balance={state.balance}, requested={cmd.amount}")

        events = [MoneyWithdrawn(amount=cmd.amount, reference=cmd.reference)]
        version = self._store.append(cmd.account_id, events, expected_version=state.version)
        new_state = self._load(cmd.account_id)
        self._log.info("Withdrew %.2f from %s (balance: %.2f)", cmd.amount, cmd.account_id, new_state.balance)
        return CommandResult(True, cmd.account_id, version, state=new_state)

    def handle_close(self, cmd: CloseAccountCommand) -> CommandResult:
        state = self._load(cmd.account_id)
        if state.version < 0:
            return CommandResult(False, cmd.account_id, -1, "Account does not exist")
        if state.is_closed:
            return CommandResult(False, cmd.account_id, state.version, "Already closed")
        if state.balance != 0:
            return CommandResult(False, cmd.account_id, state.version,
                                 f"Cannot close: balance must be 0, got {state.balance}")

        events = [AccountClosed(reason=cmd.reason)]
        version = self._store.append(cmd.account_id, events, expected_version=state.version)
        self._log.info("Account %s closed", cmd.account_id)
        return CommandResult(True, cmd.account_id, version)


# ===========================================================================
# Read side: projections and read models
# ===========================================================================

@dataclass
class AccountSummaryView:
    account_id: str
    owner: str
    balance: float
    is_closed: bool
    transaction_count: int
    last_updated: float


@dataclass
class OwnerStats:
    owner: str
    total_balance: float
    account_count: int
    total_deposited: float
    total_withdrawn: float


# ---------------------------------------------------------------------------
# Projection: AccountSummary — per-account denormalized view
# ---------------------------------------------------------------------------

class AccountSummaryProjection:
    """
    Maintains a flat view of each account.
    Rebuilt by replaying events from global position = checkpoint.
    """

    def __init__(self) -> None:
        self._store: dict[str, AccountSummaryView] = {}
        self.checkpoint: int = 0
        self._log = logging.getLogger("proj.account_summary")

    def handle(self, stored: StoredEvent) -> None:
        account_id = stored.stream_id
        event = stored.event

        match event:
            case AccountOpened(owner=o, initial_balance=b):
                self._store[account_id] = AccountSummaryView(
                    account_id=account_id,
                    owner=o,
                    balance=b,
                    is_closed=False,
                    transaction_count=0,
                    last_updated=event.timestamp,
                )
            case MoneyDeposited(amount=a):
                if account_id in self._store:
                    v = self._store[account_id]
                    self._store[account_id] = AccountSummaryView(
                        account_id=v.account_id,
                        owner=v.owner,
                        balance=v.balance + a,
                        is_closed=v.is_closed,
                        transaction_count=v.transaction_count + 1,
                        last_updated=event.timestamp,
                    )
            case MoneyWithdrawn(amount=a):
                if account_id in self._store:
                    v = self._store[account_id]
                    self._store[account_id] = AccountSummaryView(
                        account_id=v.account_id,
                        owner=v.owner,
                        balance=v.balance - a,
                        is_closed=v.is_closed,
                        transaction_count=v.transaction_count + 1,
                        last_updated=event.timestamp,
                    )
            case AccountClosed():
                if account_id in self._store:
                    v = self._store[account_id]
                    self._store[account_id] = AccountSummaryView(
                        account_id=v.account_id,
                        owner=v.owner,
                        balance=v.balance,
                        is_closed=True,
                        transaction_count=v.transaction_count,
                        last_updated=event.timestamp,
                    )

        self.checkpoint = stored.global_position + 1

    def get(self, account_id: str) -> AccountSummaryView | None:
        return self._store.get(account_id)

    def all_accounts(self) -> list[AccountSummaryView]:
        return list(self._store.values())


# ---------------------------------------------------------------------------
# Projection: OwnerBalance — aggregate view per owner
# ---------------------------------------------------------------------------

class OwnerBalanceProjection:
    def __init__(self) -> None:
        self._stats: dict[str, OwnerStats] = {}
        # need to track owner per account
        self._account_owner: dict[str, str] = {}
        self.checkpoint: int = 0
        self._log = logging.getLogger("proj.owner_balance")

    def handle(self, stored: StoredEvent) -> None:
        account_id = stored.stream_id
        event = stored.event

        match event:
            case AccountOpened(owner=o, initial_balance=b):
                self._account_owner[account_id] = o
                if o not in self._stats:
                    self._stats[o] = OwnerStats(o, 0.0, 0, 0.0, 0.0)
                s = self._stats[o]
                self._stats[o] = OwnerStats(o, s.total_balance + b, s.account_count + 1,
                                             s.total_deposited + b, s.total_withdrawn)
            case MoneyDeposited(amount=a):
                owner = self._account_owner.get(account_id)
                if owner and owner in self._stats:
                    s = self._stats[owner]
                    self._stats[owner] = OwnerStats(owner, s.total_balance + a, s.account_count,
                                                     s.total_deposited + a, s.total_withdrawn)
            case MoneyWithdrawn(amount=a):
                owner = self._account_owner.get(account_id)
                if owner and owner in self._stats:
                    s = self._stats[owner]
                    self._stats[owner] = OwnerStats(owner, s.total_balance - a, s.account_count,
                                                     s.total_deposited, s.total_withdrawn + a)

        self.checkpoint = stored.global_position + 1

    def get(self, owner: str) -> OwnerStats | None:
        return self._stats.get(owner)


# ---------------------------------------------------------------------------
# Projection runner — catches projections up to current global log head
# ---------------------------------------------------------------------------

class ProjectionRunner:
    def __init__(self, store: InMemoryEventStore) -> None:
        self._store = store
        self._projections: list[AccountSummaryProjection | OwnerBalanceProjection] = []

    def register(self, projection: AccountSummaryProjection | OwnerBalanceProjection) -> None:
        self._projections.append(projection)

    def catch_up(self) -> int:
        """Process all unprocessed global events for all registered projections."""
        # Each projection has its own checkpoint
        total = 0
        for proj in self._projections:
            events = self._store.read_global(from_position=proj.checkpoint)
            for se in events:
                proj.handle(se)
                total += 1
        return total

    def rebuild(
        self,
        projection: AccountSummaryProjection | OwnerBalanceProjection,
    ) -> None:
        """Rebuild a single projection from scratch."""
        projection._store = {}  # type: ignore[attr-defined]
        projection.checkpoint = 0
        all_events = self._store.read_global(from_position=0)
        for se in all_events:
            projection.handle(se)
        logger.info(
            "Rebuilt projection %s: %d events processed",
            type(projection).__name__, len(all_events),
        )


# ===========================================================================
# Saga: Transfer between accounts (orchestration style)
# ===========================================================================

@dataclass
class TransferSagaState:
    saga_id: str
    from_account: str
    to_account: str
    amount: float
    step: str = "pending"   # pending -> withdrawn -> deposited -> complete / failed
    error: str = ""


class TransferSaga:
    """
    Orchestrates a transfer: withdraw from source, deposit to destination.
    On failure, issues compensating commands.
    """

    def __init__(self, handler: AccountCommandHandler) -> None:
        self._handler = handler
        self._log = logging.getLogger("saga.transfer")

    def execute(self, from_account: str, to_account: str, amount: float) -> TransferSagaState:
        saga_id = str(uuid.uuid4())[:8]
        state = TransferSagaState(
            saga_id=saga_id,
            from_account=from_account,
            to_account=to_account,
            amount=amount,
        )

        self._log.info(
            "Saga %s: transferring %.2f from %s to %s",
            saga_id, amount, from_account, to_account,
        )

        # Step 1: Withdraw
        result = self._handler.handle_withdraw(
            WithdrawCommand(account_id=from_account, amount=amount, reference=f"transfer-{saga_id}")
        )
        if not result.success:
            state.step = "failed"
            state.error = f"Withdrawal failed: {result.error}"
            self._log.warning("Saga %s failed at withdrawal: %s", saga_id, result.error)
            return state

        state.step = "withdrawn"

        # Step 2: Deposit — could fail (e.g., account closed)
        result = self._handler.handle_deposit(
            DepositCommand(account_id=to_account, amount=amount, reference=f"transfer-{saga_id}")
        )
        if not result.success:
            # Compensate: return funds to source
            self._log.warning(
                "Saga %s: deposit failed (%s), compensating...", saga_id, result.error
            )
            comp_result = self._handler.handle_deposit(
                DepositCommand(
                    account_id=from_account,
                    amount=amount,
                    reference=f"compensation-{saga_id}",
                )
            )
            state.step = "failed"
            state.error = f"Deposit failed: {result.error}. Compensation: {'ok' if comp_result.success else 'FAILED'}"
            return state

        state.step = "complete"
        self._log.info("Saga %s complete", saga_id)
        return state


# ===========================================================================
# Demo
# ===========================================================================

def demo_commands_and_projections(
    store: InMemoryEventStore,
    handler: AccountCommandHandler,
    runner: ProjectionRunner,
    summary_proj: AccountSummaryProjection,
    owner_proj: OwnerBalanceProjection,
) -> None:
    logger.info("=== Commands + Projections Demo ===")

    # Open two accounts
    handler.handle_open(OpenAccountCommand("acc-alice", "Alice", 1000.0))
    handler.handle_open(OpenAccountCommand("acc-bob", "Bob", 500.0))

    # Transactions
    handler.handle_deposit(DepositCommand("acc-alice", 250.0, "freelance"))
    handler.handle_withdraw(WithdrawCommand("acc-alice", 100.0, "groceries"))
    handler.handle_deposit(DepositCommand("acc-bob", 100.0, "gift"))

    # Catch up projections
    runner.catch_up()

    # Query read model
    alice_view = summary_proj.get("acc-alice")
    bob_view = summary_proj.get("acc-bob")
    alice_stats = owner_proj.get("Alice")

    assert alice_view is not None
    logger.info(
        "Alice view: balance=%.2f transactions=%d",
        alice_view.balance, alice_view.transaction_count,
    )
    assert abs(alice_view.balance - 1150.0) < 0.01, f"Expected 1150, got {alice_view.balance}"

    assert alice_stats is not None
    logger.info(
        "Alice stats: total_deposited=%.2f total_withdrawn=%.2f",
        alice_stats.total_deposited, alice_stats.total_withdrawn,
    )


def demo_invalid_commands(handler: AccountCommandHandler) -> None:
    logger.info("=== Invalid Command Rejection Demo ===")

    # Overdraft attempt
    result = handler.handle_withdraw(WithdrawCommand("acc-alice", 9999.0, "overdraft"))
    logger.info("Overdraft rejected: %s", result.error)
    assert not result.success

    # Negative deposit
    result = handler.handle_deposit(DepositCommand("acc-alice", -50.0))
    logger.info("Negative deposit rejected: %s", result.error)
    assert not result.success

    # Non-existent account
    result = handler.handle_withdraw(WithdrawCommand("acc-ghost", 10.0))
    logger.info("Ghost account rejected: %s", result.error)
    assert not result.success


def demo_projection_rebuild(
    runner: ProjectionRunner,
    summary_proj: AccountSummaryProjection,
) -> None:
    logger.info("=== Projection Rebuild Demo ===")

    # Simulate a bug: corrupt the projection
    for acc_id in list(summary_proj._store.keys()):
        summary_proj._store[acc_id] = AccountSummaryView(
            account_id=acc_id, owner="CORRUPTED", balance=-1.0,
            is_closed=False, transaction_count=0, last_updated=0.0,
        )

    alice_view_corrupted = summary_proj.get("acc-alice")
    assert alice_view_corrupted and alice_view_corrupted.owner == "CORRUPTED"
    logger.info("Projection corrupted (simulated bug)")

    # Rebuild from event store
    runner.rebuild(summary_proj)

    alice_view_rebuilt = summary_proj.get("acc-alice")
    assert alice_view_rebuilt and alice_view_rebuilt.owner == "Alice"
    logger.info(
        "Projection rebuilt: Alice balance=%.2f (correct)",
        alice_view_rebuilt.balance,
    )


def demo_saga(handler: AccountCommandHandler) -> None:
    logger.info("=== Saga: Account Transfer Demo ===")
    saga = TransferSaga(handler)

    # Successful transfer
    state = saga.execute("acc-alice", "acc-bob", 200.0)
    logger.info("Transfer result: step=%s error='%s'", state.step, state.error)
    assert state.step == "complete"

    # Failed transfer: insufficient funds
    state = saga.execute("acc-alice", "acc-bob", 999999.0)
    logger.info(
        "Large transfer result: step=%s error='%s'", state.step, state.error
    )
    assert state.step == "failed"

    # Failed transfer: destination is closed
    handler.handle_withdraw(WithdrawCommand("acc-bob", handler._load("acc-bob").balance))
    handler.handle_close(CloseAccountCommand("acc-bob", "test scenario"))
    state = saga.execute("acc-alice", "acc-bob", 50.0)
    logger.info(
        "Closed-account transfer result: step=%s (compensation triggered)",
        state.step,
    )
    assert state.step == "failed"


def main() -> None:
    store = InMemoryEventStore()
    handler = AccountCommandHandler(store)
    summary_proj = AccountSummaryProjection()
    owner_proj = OwnerBalanceProjection()
    runner = ProjectionRunner(store)
    runner.register(summary_proj)
    runner.register(owner_proj)

    demo_commands_and_projections(store, handler, runner, summary_proj, owner_proj)
    print()

    demo_invalid_commands(handler)
    print()

    demo_projection_rebuild(runner, summary_proj)
    print()

    demo_saga(handler)
    print()

    logger.info("All demos complete.")


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **New projection — top spenders.** Write a `TopSpendersProjection` that tracks the top 5 accounts by total amount withdrawn. Update it via the `ProjectionRunner` and verify after each transaction batch. Add a `get_top_n(n)` method returning a ranked list.

2. **Projection checkpoint persistence.** The current `ProjectionRunner` does not persist checkpoints across restarts. Add a `save_checkpoint(name, position)` / `load_checkpoint(name) -> int` interface backed by a JSON file. On startup, `catch_up` should resume from the persisted checkpoint rather than replaying all events.

3. **Async projection pipeline.** Convert the projection runner to use `asyncio`. `catch_up` should dispatch events to projections concurrently (`asyncio.gather`). Add a simulated slow projection (0.1 s per event) and measure throughput improvement over sequential dispatch.

4. **Saga compensation test.** Write a pytest suite for `TransferSaga` that:
   - Asserts complete saga succeeds and both balances are correct
   - Asserts that when the deposit step fails, the source account balance is restored (compensation worked)
   - Asserts the saga is in `"failed"` state on any rejection

5. **Multiple projections from one event type.** A `MoneyDeposited` event contributes to three projections simultaneously: `AccountSummaryProjection`, `OwnerBalanceProjection`, and a new `DailyVolumeProjection` (sum of all deposits grouped by calendar day). Build `DailyVolumeProjection` and verify it accumulates correctly.

*(Solutions: ../solutions/week-04/day-3.md)*

---

## Self-check

- [ ] I can explain what problem CQRS solves and what the trade-off is
- [ ] I can describe eventual consistency and name one way a caller can handle the read-lag
- [ ] I can implement a projection that handles multiple event types and maintains a denormalized view
- [ ] I can explain projection rebuilding and why it is safe
- [ ] I can describe the difference between choreography and orchestration sagas

---

## Supplementary Reading

- [Martin Fowler: CQRS](https://martinfowler.com/bliki/CQRS.html) — canonical definition
- [Greg Young: CQRS Documents](https://cqrs.files.wordpress.com/2010/11/cqrs_documents.pdf) — original specification
- [Saga pattern — Microsoft Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/reference-architectures/saga/saga)
- [Eventual Consistency Revisited — Werner Vogels](https://www.allthingsdistributed.com/2008/12/eventually_consistent.html)
- [Mauro Servienti: All Our Aggregates Are Wrong](https://www.youtube.com/watch?v=KkzvQSuYd5I) — practical CQRS boundary design
