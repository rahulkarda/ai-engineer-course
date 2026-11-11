# Solutions — Week 06 · Day 4 — Human-in-the-Loop: Approval Gates

> These are reference solutions. Your implementation may differ and still be correct.

## Exercise 1

**Add an "auto-approve" rule: actions under a configurable dollar threshold are auto-approved without human review.**

Design: `ApprovalGate` accepts an `auto_approve_below: float` constructor parameter. Before queuing a request, it checks `request.amount`; if `amount < auto_approve_below`, it bypasses the queue and returns an immediately-resolved `ApprovalDecision` with `approved=True` and `reason="AUTO_APPROVED"`. Amounts at or above the threshold proceed to human review as normal.

```python
"""
day4_ex1.py — Approval gate with dollar-threshold auto-approval.

Run: python day4_ex1.py
Env: ANTHROPIC_API_KEY (not required for this exercise — logic is pure Python)
"""
import logging
import os
import unittest.mock
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    id: str
    description: str
    amount: float = Field(ge=0.0, description="Dollar amount associated with this action")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    request_id: str
    approved: bool
    reason: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewer: str = "human"  # "human" | "auto"


# ---------------------------------------------------------------------------
# Human approval queue
# ---------------------------------------------------------------------------

class HumanApprovalQueue:
    """
    Simulates a queue where human reviewers approve or reject requests.
    In production this would be backed by a database + notification system.
    """

    def __init__(self) -> None:
        self._pending: dict[str, ApprovalRequest] = {}
        self._decisions: dict[str, ApprovalDecision] = {}

    def enqueue(self, request: ApprovalRequest) -> None:
        self._pending[request.id] = request
        logger.info("Queued request id=%s amount=%.2f", request.id, request.amount)

    def decide(self, request_id: str, approved: bool, reason: str) -> ApprovalDecision:
        """Called by a human reviewer (or test mock) to record a decision."""
        if request_id not in self._pending:
            raise KeyError(f"Request '{request_id}' not in queue")
        decision = ApprovalDecision(
            request_id=request_id,
            approved=approved,
            reason=reason,
            reviewer="human",
        )
        self._decisions[request_id] = decision
        del self._pending[request_id]
        logger.info(
            "Decision recorded: id=%s approved=%s reason=%s",
            request_id,
            approved,
            reason,
        )
        return decision

    def get_decision(self, request_id: str) -> ApprovalDecision | None:
        return self._decisions.get(request_id)

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Approval gate with auto-approve threshold
# ---------------------------------------------------------------------------

class ApprovalGate:
    """
    Routes approval requests through a human queue.

    When `request.amount < auto_approve_below`, the action is approved
    automatically without queuing for human review.
    """

    def __init__(
        self,
        queue: HumanApprovalQueue,
        *,
        auto_approve_below: float = 10.0,
    ) -> None:
        self._queue = queue
        self._auto_approve_below = auto_approve_below

    def submit(self, request: ApprovalRequest) -> ApprovalDecision | None:
        """
        Submit a request for approval.

        Returns:
            ApprovalDecision immediately if auto-approved.
            None if routed to the human queue (decision arrives later via queue.decide).
        """
        if request.amount < self._auto_approve_below:
            decision = ApprovalDecision(
                request_id=request.id,
                approved=True,
                reason=(
                    f"AUTO_APPROVED: amount ${request.amount:.2f} is below "
                    f"threshold ${self._auto_approve_below:.2f}"
                ),
                reviewer="auto",
            )
            logger.info(
                "Auto-approved request id=%s amount=%.2f",
                request.id,
                request.amount,
            )
            return decision

        self._queue.enqueue(request)
        logger.info(
            "Queued for human review: id=%s amount=%.2f (threshold=%.2f)",
            request.id,
            request.amount,
            self._auto_approve_below,
        )
        return None  # decision will arrive asynchronously


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0)

    requests = [
        ApprovalRequest(id="req-001", description="Buy coffee", amount=4.50),
        ApprovalRequest(id="req-002", description="Office supplies", amount=9.99),
        ApprovalRequest(id="req-003", description="Software license", amount=10.00),
        ApprovalRequest(id="req-004", description="Server upgrade", amount=250.00),
    ]

    for req in requests:
        decision = gate.submit(req)
        if decision:
            print(f"[{req.id}] AUTO: {decision.reason}")
        else:
            print(f"[{req.id}] PENDING human review (amount=${req.amount:.2f})")

    print(f"\nRequests awaiting human review: {queue.pending_count}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_auto_approve_below_threshold() -> None:
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0)
    req = ApprovalRequest(id="r1", description="Small purchase", amount=5.00)

    decision = gate.submit(req)

    assert decision is not None
    assert decision.approved is True
    assert "AUTO_APPROVED" in decision.reason
    assert decision.reviewer == "auto"
    assert queue.pending_count == 0


def test_auto_approve_exact_threshold_routes_to_human() -> None:
    """Amount == threshold is NOT auto-approved (strict less-than)."""
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0)
    req = ApprovalRequest(id="r2", description="At threshold", amount=10.00)

    decision = gate.submit(req)

    assert decision is None  # queued, not auto-approved
    assert queue.pending_count == 1


def test_above_threshold_routes_to_human_queue() -> None:
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0)
    req = ApprovalRequest(id="r3", description="Large purchase", amount=500.00)

    decision = gate.submit(req)

    assert decision is None
    assert queue.pending_count == 1


def test_human_decision_clears_queue() -> None:
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=5.0)
    req = ApprovalRequest(id="r4", description="Purchase", amount=50.00)

    gate.submit(req)
    assert queue.pending_count == 1

    decision = queue.decide("r4", approved=True, reason="Approved by manager")
    assert decision.approved is True
    assert queue.pending_count == 0


def test_configurable_threshold() -> None:
    """Auto-approve threshold can be adjusted per gate instance."""
    queue = HumanApprovalQueue()
    gate_low = ApprovalGate(queue, auto_approve_below=1.0)
    gate_high = ApprovalGate(queue, auto_approve_below=1000.0)

    req = ApprovalRequest(id="r5", description="Mid-size", amount=50.0)

    decision_low = gate_low.submit(req)
    assert decision_low is None  # 50 >= 1.0 → queued

    queue._pending.pop("r5", None)  # reset queue

    decision_high = gate_high.submit(req)
    assert decision_high is not None  # 50 < 1000 → auto-approved


def test_auto_approve_decision_fields() -> None:
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0)
    req = ApprovalRequest(id="r6", description="Snack", amount=3.00)

    decision = gate.submit(req)

    assert decision is not None
    assert decision.request_id == "r6"
    assert decision.reviewer == "auto"
    assert isinstance(decision.decided_at, datetime)
```


## Exercise 2

**Implement approval expiry: if no decision is made within N seconds, the gate auto-rejects with reason `"TIMEOUT"`.**

Design: `ApprovalGate.wait_for_decision()` is an async method. It polls `queue.get_decision()` at a fixed interval. If the decision arrives within `timeout_seconds`, it is returned. If the timeout elapses first, the gate injects a synthetic `ApprovalDecision(approved=False, reason="TIMEOUT")` into the queue and returns it.

```python
"""
day4_ex2.py — Approval gate with expiry / timeout auto-rejection.

Run: python day4_ex2.py
Env: none required
"""
import asyncio
import logging
import unittest.mock
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.05  # seconds between queue checks (fast for tests)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    id: str
    description: str
    amount: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    request_id: str
    approved: bool
    reason: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewer: str = "human"


# ---------------------------------------------------------------------------
# Human approval queue (sync core, async wait)
# ---------------------------------------------------------------------------

class HumanApprovalQueue:
    def __init__(self) -> None:
        self._pending: dict[str, ApprovalRequest] = {}
        self._decisions: dict[str, ApprovalDecision] = {}

    def enqueue(self, request: ApprovalRequest) -> None:
        self._pending[request.id] = request
        logger.info("Queued: id=%s", request.id)

    def decide(
        self, request_id: str, approved: bool, reason: str, reviewer: str = "human"
    ) -> ApprovalDecision:
        decision = ApprovalDecision(
            request_id=request_id,
            approved=approved,
            reason=reason,
            reviewer=reviewer,
        )
        self._decisions[request_id] = decision
        self._pending.pop(request_id, None)
        logger.info("Decision: id=%s approved=%s", request_id, approved)
        return decision

    def get_decision(self, request_id: str) -> ApprovalDecision | None:
        return self._decisions.get(request_id)

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Approval gate with timeout
# ---------------------------------------------------------------------------

class ApprovalGate:
    """
    Submits requests to a human queue. `wait_for_decision` blocks until either:
    - A human posts a decision via `queue.decide()`, or
    - `timeout_seconds` elapses — gate auto-rejects with reason "TIMEOUT".
    """

    def __init__(
        self,
        queue: HumanApprovalQueue,
        *,
        auto_approve_below: float = 10.0,
        timeout_seconds: float = 30.0,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._queue = queue
        self._auto_approve_below = auto_approve_below
        self._timeout_seconds = timeout_seconds
        self._poll_interval = poll_interval

    def submit(self, request: ApprovalRequest) -> ApprovalDecision | None:
        """Auto-approve if below threshold; otherwise enqueue for human review."""
        if request.amount < self._auto_approve_below:
            decision = ApprovalDecision(
                request_id=request.id,
                approved=True,
                reason=f"AUTO_APPROVED: ${request.amount:.2f} < ${self._auto_approve_below:.2f}",
                reviewer="auto",
            )
            self._queue._decisions[request.id] = decision
            logger.info("Auto-approved id=%s", request.id)
            return decision

        self._queue.enqueue(request)
        return None

    async def wait_for_decision(self, request_id: str) -> ApprovalDecision:
        """
        Async poll until a decision is available or the timeout fires.

        On timeout, injects a TIMEOUT rejection into the queue and returns it.
        """
        deadline = asyncio.get_event_loop().time() + self._timeout_seconds
        logger.info(
            "Waiting up to %.1fs for decision on id=%s",
            self._timeout_seconds,
            request_id,
        )

        while asyncio.get_event_loop().time() < deadline:
            decision = self._queue.get_decision(request_id)
            if decision is not None:
                return decision
            await asyncio.sleep(self._poll_interval)

        # Timeout — auto-reject
        logger.warning("Timeout waiting for decision on id=%s", request_id)
        timeout_decision = self._queue.decide(
            request_id,
            approved=False,
            reason="TIMEOUT",
            reviewer="system",
        )
        return timeout_decision


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0, timeout_seconds=2.0)

    req = ApprovalRequest(id="req-001", description="Office chair", amount=299.00)
    gate.submit(req)

    # Simulate: no human decision arrives within timeout
    print(f"Waiting for decision on {req.id}...")
    decision = await gate.wait_for_decision(req.id)
    print(f"Decision: approved={decision.approved} reason={decision.reason}")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wait_for_decision_resolves_when_human_decides() -> None:
    """Decision arrives before timeout — returned immediately."""
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, timeout_seconds=5.0, poll_interval=0.01)

    req = ApprovalRequest(id="r1", description="Purchase", amount=50.0)
    gate.submit(req)

    async def run() -> ApprovalDecision:
        # Simulate human deciding after 0.05s
        async def human_decide() -> None:
            await asyncio.sleep(0.05)
            queue.decide("r1", approved=True, reason="Looks good")

        asyncio.ensure_future(human_decide())
        return await gate.wait_for_decision("r1")

    decision = asyncio.run(run())
    assert decision.approved is True
    assert decision.reason == "Looks good"


def test_wait_for_decision_times_out() -> None:
    """No human decision — gate auto-rejects with reason='TIMEOUT'."""
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, timeout_seconds=0.1, poll_interval=0.01)

    req = ApprovalRequest(id="r2", description="Purchase", amount=50.0)
    gate.submit(req)

    decision = asyncio.run(gate.wait_for_decision("r2"))

    assert decision.approved is False
    assert decision.reason == "TIMEOUT"
    assert decision.reviewer == "system"


def test_timeout_decision_persisted_in_queue() -> None:
    """After timeout, the decision is retrievable via queue.get_decision."""
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, timeout_seconds=0.05, poll_interval=0.01)

    req = ApprovalRequest(id="r3", description="Purchase", amount=50.0)
    gate.submit(req)
    asyncio.run(gate.wait_for_decision("r3"))

    stored = queue.get_decision("r3")
    assert stored is not None
    assert stored.reason == "TIMEOUT"


def test_auto_approved_skips_wait() -> None:
    """Auto-approved requests have a decision immediately; wait returns instantly."""
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, auto_approve_below=10.0, timeout_seconds=5.0)

    req = ApprovalRequest(id="r4", description="Coffee", amount=4.50)
    gate.submit(req)

    decision = asyncio.run(gate.wait_for_decision("r4"))
    assert decision.approved is True
    assert "AUTO_APPROVED" in decision.reason


def test_timeout_zero_rejects_immediately() -> None:
    """Timeout of 0 seconds rejects without any polling wait."""
    queue = HumanApprovalQueue()
    gate = ApprovalGate(queue, timeout_seconds=0.0, poll_interval=0.001)

    req = ApprovalRequest(id="r5", description="Urgent", amount=100.0)
    gate.submit(req)

    decision = asyncio.run(gate.wait_for_decision("r5"))
    assert decision.approved is False
    assert decision.reason == "TIMEOUT"
```


## Exercise 3

**Add an audit trail: every approval request and decision is appended to an `AuditLog` list with timestamps.**

Design: `AuditLog` is a dataclass holding a `list[AuditEntry]`. Each `AuditEntry` records `event_type` (`"REQUEST_SUBMITTED"`, `"DECISION_MADE"`), the associated `request_id`, and a `timestamp`. `ApprovalGate` and `HumanApprovalQueue` accept an optional `AuditLog` and append entries at each state transition. A `query(request_id)` method retrieves all entries for a given request.

```python
"""
day4_ex3.py — Approval gate with full audit trail.

Run: python day4_ex3.py
Env: none required
"""
import asyncio
import logging
import unittest.mock
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

_POLL_INTERVAL = 0.05


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    id: str
    description: str
    amount: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    request_id: str
    approved: bool
    reason: str
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewer: str = "human"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    event_type: str          # "REQUEST_SUBMITTED" | "DECISION_MADE" | "AUTO_APPROVED" | "TIMEOUT"
    request_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLog:
    """Append-only audit trail for all gate activity."""

    entries: list[AuditEntry] = field(default_factory=list)

    def record(
        self,
        event_type: str,
        request_id: str,
        **details: Any,
    ) -> None:
        entry = AuditEntry(
            event_type=event_type,
            request_id=request_id,
            details=details,
        )
        self.entries.append(entry)
        logger.debug("Audit: %s id=%s details=%s", event_type, request_id, details)

    def query(self, request_id: str) -> list[AuditEntry]:
        """Return all audit entries for a specific request_id, in order."""
        return [e for e in self.entries if e.request_id == request_id]

    def summary(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Queue with audit integration
# ---------------------------------------------------------------------------

class HumanApprovalQueue:
    def __init__(self, audit_log: AuditLog | None = None) -> None:
        self._pending: dict[str, ApprovalRequest] = {}
        self._decisions: dict[str, ApprovalDecision] = {}
        self._audit = audit_log or AuditLog()

    def enqueue(self, request: ApprovalRequest) -> None:
        self._pending[request.id] = request
        self._audit.record(
            "REQUEST_SUBMITTED",
            request.id,
            description=request.description,
            amount=request.amount,
        )
        logger.info("Queued for review: id=%s amount=%.2f", request.id, request.amount)

    def decide(
        self, request_id: str, approved: bool, reason: str, reviewer: str = "human"
    ) -> ApprovalDecision:
        decision = ApprovalDecision(
            request_id=request_id,
            approved=approved,
            reason=reason,
            reviewer=reviewer,
        )
        self._decisions[request_id] = decision
        self._pending.pop(request_id, None)

        event = "TIMEOUT" if reason == "TIMEOUT" else "DECISION_MADE"
        self._audit.record(
            event,
            request_id,
            approved=approved,
            reason=reason,
            reviewer=reviewer,
        )
        logger.info("Decision: id=%s approved=%s", request_id, approved)
        return decision

    def get_decision(self, request_id: str) -> ApprovalDecision | None:
        return self._decisions.get(request_id)

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# Gate with audit + timeout + auto-approve
# ---------------------------------------------------------------------------

class ApprovalGate:
    def __init__(
        self,
        queue: HumanApprovalQueue,
        audit_log: AuditLog | None = None,
        *,
        auto_approve_below: float = 10.0,
        timeout_seconds: float = 30.0,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._queue = queue
        self._audit = audit_log or queue._audit  # share audit log with queue
        self._auto_approve_below = auto_approve_below
        self._timeout_seconds = timeout_seconds
        self._poll_interval = poll_interval

    def submit(self, request: ApprovalRequest) -> ApprovalDecision | None:
        if request.amount < self._auto_approve_below:
            decision = ApprovalDecision(
                request_id=request.id,
                approved=True,
                reason=(
                    f"AUTO_APPROVED: ${request.amount:.2f} < "
                    f"${self._auto_approve_below:.2f}"
                ),
                reviewer="auto",
            )
            self._queue._decisions[request.id] = decision
            self._audit.record(
                "AUTO_APPROVED",
                request.id,
                amount=request.amount,
                threshold=self._auto_approve_below,
            )
            logger.info("Auto-approved id=%s", request.id)
            return decision

        self._queue.enqueue(request)
        return None

    async def wait_for_decision(self, request_id: str) -> ApprovalDecision:
        deadline = asyncio.get_event_loop().time() + self._timeout_seconds
        while asyncio.get_event_loop().time() < deadline:
            decision = self._queue.get_decision(request_id)
            if decision is not None:
                return decision
            await asyncio.sleep(self._poll_interval)

        logger.warning("Timeout on id=%s", request_id)
        return self._queue.decide(
            request_id, approved=False, reason="TIMEOUT", reviewer="system"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=10.0, timeout_seconds=1.0)

    requests = [
        ApprovalRequest(id="r1", description="Coffee", amount=4.50),
        ApprovalRequest(id="r2", description="Server rack", amount=3000.00),
    ]

    for req in requests:
        gate.submit(req)

    # Simulate human approving r2
    queue.decide("r2", approved=True, reason="Budget approved")

    print(f"\nAudit log ({len(audit.entries)} entries):")
    for entry in audit.entries:
        print(f"  [{entry.timestamp.isoformat()}] {entry.event_type} id={entry.request_id} {entry.details}")

    print("\nAudit summary:", audit.summary())
    print("\nHistory for r1:", [(e.event_type, e.details) for e in audit.query("r1")])


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_audit_log_records_submission() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=5.0)

    req = ApprovalRequest(id="r1", description="Purchase", amount=50.0)
    gate.submit(req)

    entries = audit.query("r1")
    assert len(entries) == 1
    assert entries[0].event_type == "REQUEST_SUBMITTED"
    assert entries[0].request_id == "r1"


def test_audit_log_records_auto_approval() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=10.0)

    req = ApprovalRequest(id="r2", description="Coffee", amount=4.50)
    gate.submit(req)

    entries = audit.query("r2")
    assert len(entries) == 1
    assert entries[0].event_type == "AUTO_APPROVED"


def test_audit_log_records_human_decision() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=5.0)

    req = ApprovalRequest(id="r3", description="Equipment", amount=100.0)
    gate.submit(req)
    queue.decide("r3", approved=True, reason="Budget OK")

    entries = audit.query("r3")
    event_types = [e.event_type for e in entries]
    assert "REQUEST_SUBMITTED" in event_types
    assert "DECISION_MADE" in event_types


def test_audit_log_records_timeout() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(
        queue, audit_log=audit, auto_approve_below=5.0,
        timeout_seconds=0.05, poll_interval=0.01
    )

    req = ApprovalRequest(id="r4", description="Urgent purchase", amount=200.0)
    gate.submit(req)
    asyncio.run(gate.wait_for_decision("r4"))

    entries = audit.query("r4")
    event_types = [e.event_type for e in entries]
    assert "TIMEOUT" in event_types


def test_audit_query_returns_only_matching_request() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=5.0)

    r_a = ApprovalRequest(id="ra", description="A", amount=50.0)
    r_b = ApprovalRequest(id="rb", description="B", amount=50.0)
    gate.submit(r_a)
    gate.submit(r_b)

    entries_a = audit.query("ra")
    assert all(e.request_id == "ra" for e in entries_a)
    assert len(entries_a) == 1


def test_audit_summary_counts_event_types() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=10.0)

    # 1 auto-approve, 1 submitted + 1 decision
    gate.submit(ApprovalRequest(id="r5", description="Small", amount=5.0))
    gate.submit(ApprovalRequest(id="r6", description="Large", amount=100.0))
    queue.decide("r6", approved=False, reason="Over budget")

    summary = audit.summary()
    assert summary.get("AUTO_APPROVED", 0) == 1
    assert summary.get("REQUEST_SUBMITTED", 0) == 1
    assert summary.get("DECISION_MADE", 0) == 1


def test_audit_entries_have_timestamps() -> None:
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=5.0)
    gate.submit(ApprovalRequest(id="r7", description="test", amount=50.0))

    for entry in audit.entries:
        assert isinstance(entry.timestamp, datetime)


def test_audit_log_entries_ordered_by_insertion() -> None:
    """Entries appear in chronological insertion order."""
    audit = AuditLog()
    queue = HumanApprovalQueue(audit_log=audit)
    gate = ApprovalGate(queue, audit_log=audit, auto_approve_below=5.0)

    req = ApprovalRequest(id="r8", description="Test", amount=50.0)
    gate.submit(req)
    queue.decide("r8", approved=True, reason="OK")

    entries = audit.query("r8")
    assert entries[0].event_type == "REQUEST_SUBMITTED"
    assert entries[1].event_type == "DECISION_MADE"
