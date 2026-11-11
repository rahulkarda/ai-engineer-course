# Phase 5 · Week 06 · Day 6 — Approval Workflows for Sensitive Actions

> Objective: Design and implement a complete multi-step approval workflow system for sensitive agent actions, with state machine, audit logging, timeout handling, and retry logic.  ·  Time: ~4 hr  ·  Prereqs: [[day-4]], [[day-5]]

## Concept

Some actions an agent can take are irreversible or high-stakes: sending a bulk email, executing a financial transfer, modifying production config. Before the agent acts, a human (or chain of humans) must explicitly approve. This is an **approval workflow**.

An approval workflow is a state machine with these core properties:

1. **States are explicit.** A request is always in exactly one state (`draft`, `pending`, `approved`, `rejected`, `expired`). Implicit state (a boolean flag) breaks down the moment you add a second approver.
2. **Approval chains are ordered.** Manager must approve before compliance. Each approver sees the previous approvers' decisions as context.
3. **Time-limited approvals are required.** An approval granted six months ago for a one-time action should not still be valid. Expiry forces re-approval for long-delayed actions.
4. **Audit logs are immutable append-only records.** Every state transition — who made it, when, what they said — is written and never deleted.
5. **Rejection is a first-class outcome.** Agents must handle rejection gracefully, with configurable retry logic.

### State machine

```
draft → pending → approved → (action executed)
               ↘ rejected  → (retry or abort)
               ↘ expired   → (re-submit required)
```

## How it works internally

`WorkflowState` is a `str` enum — transitions are validated explicitly in `WorkflowEngine.advance()`. Invalid transitions raise `InvalidTransitionError` rather than silently succeeding.

`ApprovalChain` is an ordered list of approver callables. Each approver receives an `ApprovalRequest` (context, prior decisions) and returns an `ApprovalResponse` (decision + note). Approvers are injected dependencies — they can be async human-approval UI calls, automated policy checks, or mocks in tests.

`AuditLog` is a list of `AuditEntry` records with `datetime` timestamps (UTC). Each entry is appended, never mutated — the audit trail is complete and tamper-evident.

Timeout is implemented with `asyncio.wait_for`. The `WorkflowEngine.run()` coroutine runs the full chain; if it exceeds `timeout_seconds`, `asyncio.TimeoutError` is caught, state transitions to `expired`, and an audit entry records the expiry.

Pydantic v2 `model_config = ConfigDict(frozen=True)` on `AuditEntry` prevents accidental mutation after creation.

## Worked example

```python
"""
day6_approval_workflow.py — Multi-step approval workflow engine.

Run:  python day6_approval_workflow.py
Env:  ANTHROPIC_API_KEY  (used only in the demo prompt-generation step)
"""
import asyncio
import logging
import os
from datetime import UTC, datetime
from enum import Enum
from typing import Callable, Awaitable

import anthropic
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Exceptions
# ---------------------------------------------------------------------------

class WorkflowState(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Valid transitions: from_state → set of allowed to_states
_VALID_TRANSITIONS: dict[WorkflowState, set[WorkflowState]] = {
    WorkflowState.DRAFT: {WorkflowState.PENDING},
    WorkflowState.PENDING: {
        WorkflowState.APPROVED,
        WorkflowState.REJECTED,
        WorkflowState.EXPIRED,
    },
    WorkflowState.APPROVED: set(),   # terminal
    WorkflowState.REJECTED: {WorkflowState.PENDING},  # retry allowed
    WorkflowState.EXPIRED: {WorkflowState.PENDING},   # re-submit allowed
}


class InvalidTransitionError(Exception):
    pass


class WorkflowTimeoutError(Exception):
    pass


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    actor: str
    from_state: WorkflowState
    to_state: WorkflowState
    note: str = ""


class ApprovalRequest(BaseModel):
    request_id: str
    action: str                          # human-readable description of what will happen
    payload: dict                        # structured action data
    requester: str
    prior_decisions: list["ApprovalResponse"] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApprovalResponse(BaseModel):
    approver: str
    approved: bool
    note: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ApprovalChain(BaseModel):
    approvers: list[str]                 # ordered approver names
    timeout_seconds: float = 30.0
    max_retries: int = 2


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class AuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self,
        actor: str,
        from_state: WorkflowState,
        to_state: WorkflowState,
        note: str = "",
    ) -> AuditEntry:
        entry = AuditEntry(
            actor=actor,
            from_state=from_state,
            to_state=to_state,
            note=note,
        )
        self._entries.append(entry)
        log.info(
            "AUDIT [%s] %s → %s by %s | %s",
            entry.timestamp.isoformat(),
            from_state.value,
            to_state.value,
            actor,
            note or "(no note)",
        )
        return entry

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)  # defensive copy

    def summary(self) -> str:
        lines = [
            f"  {e.timestamp.strftime('%H:%M:%S')} | {e.from_state.value:8s} → {e.to_state.value:8s} | {e.actor} | {e.note}"
            for e in self._entries
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Approver type alias
# ---------------------------------------------------------------------------

# An approver is an async callable: (request) → ApprovalResponse
ApproverFn = Callable[[ApprovalRequest], Awaitable[ApprovalResponse]]


# ---------------------------------------------------------------------------
# Workflow Engine
# ---------------------------------------------------------------------------

class WorkflowEngine:
    def __init__(
        self,
        chain: ApprovalChain,
        approver_fns: dict[str, ApproverFn],
    ) -> None:
        self.chain = chain
        self.approver_fns = approver_fns
        self.audit = AuditLog()
        self._state = WorkflowState.DRAFT

    @property
    def state(self) -> WorkflowState:
        return self._state

    def _transition(self, to: WorkflowState, actor: str, note: str = "") -> None:
        allowed = _VALID_TRANSITIONS[self._state]
        if to not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from {self._state.value} to {to.value}"
            )
        self.audit.record(actor, self._state, to, note)
        self._state = to

    async def _run_chain(self, request: ApprovalRequest) -> list[ApprovalResponse]:
        """Walk the approval chain in order. Short-circuits on first rejection."""
        decisions: list[ApprovalResponse] = []
        for approver_name in self.chain.approvers:
            fn = self.approver_fns.get(approver_name)
            if fn is None:
                raise ValueError(f"No approver function registered for '{approver_name}'")

            # Provide prior decisions as context
            request_with_context = request.model_copy(
                update={"prior_decisions": list(decisions)}
            )
            log.info("Awaiting approval from: %s", approver_name)
            response = await fn(request_with_context)
            decisions.append(response)
            log.info(
                "%s %s | note: %s",
                approver_name,
                "APPROVED" if response.approved else "REJECTED",
                response.note or "(none)",
            )
            if not response.approved:
                return decisions  # short-circuit on first rejection
        return decisions

    async def run(self, request: ApprovalRequest) -> WorkflowState:
        """
        Drive the workflow to completion (approved/rejected/expired).
        Handles retries up to chain.max_retries on rejection.
        Returns the final state.
        """
        self._transition(WorkflowState.PENDING, actor="system", note="Workflow started")

        for attempt in range(1, self.chain.max_retries + 2):  # +1 for initial attempt
            log.info("Approval attempt %d / %d", attempt, self.chain.max_retries + 1)
            try:
                decisions = await asyncio.wait_for(
                    self._run_chain(request),
                    timeout=self.chain.timeout_seconds,
                )
            except asyncio.TimeoutError:
                self._transition(
                    WorkflowState.EXPIRED,
                    actor="system",
                    note=f"Timed out after {self.chain.timeout_seconds}s on attempt {attempt}",
                )
                raise WorkflowTimeoutError(
                    f"Approval workflow expired after {self.chain.timeout_seconds}s"
                )

            all_approved = all(d.approved for d in decisions)
            if all_approved:
                self._transition(
                    WorkflowState.APPROVED,
                    actor="system",
                    note=f"All {len(decisions)} approver(s) approved on attempt {attempt}",
                )
                return WorkflowState.APPROVED

            # Rejected — retry if attempts remain
            rejection = next(d for d in decisions if not d.approved)
            if attempt <= self.chain.max_retries:
                self._transition(
                    WorkflowState.REJECTED,
                    actor=rejection.approver,
                    note=f"Rejected: {rejection.note}. Retrying ({attempt}/{self.chain.max_retries}).",
                )
                self._transition(
                    WorkflowState.PENDING,
                    actor="system",
                    note=f"Re-submitted for retry attempt {attempt + 1}",
                )
            else:
                self._transition(
                    WorkflowState.REJECTED,
                    actor=rejection.approver,
                    note=f"Rejected: {rejection.note}. No retries remaining.",
                )
                return WorkflowState.REJECTED

        return self._state  # should be unreachable


# ---------------------------------------------------------------------------
# Demo: approver functions
# ---------------------------------------------------------------------------

async def manager_approver(request: ApprovalRequest) -> ApprovalResponse:
    """Simulate manager approval: auto-approve if amount < 1000."""
    amount = request.payload.get("amount", 0)
    approved = amount < 1000
    return ApprovalResponse(
        approver="manager",
        approved=approved,
        note="Amount within delegated authority." if approved else f"Amount {amount} exceeds limit of 1000.",
    )


async def compliance_approver(request: ApprovalRequest) -> ApprovalResponse:
    """Simulate compliance check: always approves in demo."""
    return ApprovalResponse(
        approver="compliance",
        approved=True,
        note="No regulatory flags detected.",
    )


async def generate_action_description(payload: dict) -> str:
    """Use Claude to generate a human-readable description of the action."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarise this financial action in one plain sentence for an approver:\n{payload}"
                ),
            }
        ],
    )
    return resp.content[0].text.strip()


async def main() -> None:
    payload = {"type": "wire_transfer", "amount": 500, "recipient": "vendor-42", "currency": "USD"}
    action_desc = await generate_action_description(payload)

    request = ApprovalRequest(
        request_id="req-001",
        action=action_desc,
        payload=payload,
        requester="agent-billing",
    )

    chain = ApprovalChain(
        approvers=["manager", "compliance"],
        timeout_seconds=10.0,
        max_retries=1,
    )
    engine = WorkflowEngine(
        chain=chain,
        approver_fns={
            "manager": manager_approver,
            "compliance": compliance_approver,
        },
    )

    print(f"\nAction: {action_desc}")
    final_state = await engine.run(request)
    print(f"\nFinal state: {final_state.value}")
    print(f"\nAudit trail:\n{engine.audit.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
```

Key design decisions:
- `_VALID_TRANSITIONS` dict makes the state machine explicit and centrally auditable.
- `AuditEntry` is `frozen=True` — records are immutable after creation.
- Approver functions are injected as `dict[str, ApproverFn]` — test mocks replace real implementations with no subclassing.
- `asyncio.wait_for` provides timeout without threading complexity.
- Retry loop is bounded by `max_retries` — no infinite loops.
- `prior_decisions` in `ApprovalRequest` lets later approvers see what earlier ones decided.

## Exercises

1. The current `AuditLog` only lives in memory. Implement `PersistentAuditLog(AuditLog)` that appends each `AuditEntry` as a JSON line to a file at `log_path: pathlib.Path`. Write a test that creates a workflow, drives it to `approved`, and asserts the file contains the correct number of lines with valid JSON.

2. Add an `expiry_at: datetime | None` field to `ApprovalRequest`. In `WorkflowEngine.run()`, check at the start of each attempt whether `datetime.now(UTC) > request.expiry_at`; if so, transition to `expired` immediately. Write a test that sets `expiry_at` to the past and asserts the workflow expires on the first attempt.

3. The demo `manager_approver` is a plain async function. Wrap it in a `PolicyApprover` class that takes `max_amount: float` and `approver_name: str`, and exposes `__call__` so it satisfies `ApproverFn`. Write two tests: one where the amount is under the limit (approved), one where it is over (rejected).

4. Implement `WorkflowEngine.replay(entries: list[AuditEntry]) -> WorkflowState` that reconstructs the workflow's final state from an audit log without re-running any approvers. This is useful for disaster recovery and reporting. Write a test with a known audit trail and assert the replayed state matches.

5. Currently all approvers must approve for the workflow to reach `approved`. Add a `quorum_required: int` field to `ApprovalChain` that specifies how many approvals are needed (e.g. 2-of-3). Update `_run_chain` to continue through all approvers even after a rejection, then check the count at the end. Write tests for majority-approved and majority-rejected scenarios.

(Solutions: ../solutions/week-06/day-6.md)

## Self-check

- [ ] I can draw the approval workflow state machine from memory, including all valid transitions
- [ ] I can explain why `AuditEntry` should be immutable (`frozen=True`)
- [ ] I can implement an `ApprovalChain` with ordered approvers that short-circuits on rejection
- [ ] I understand how `asyncio.wait_for` provides timeout without blocking the event loop
- [ ] I can inject mock approver functions in tests without subclassing `WorkflowEngine`
- [ ] I can explain the difference between `max_retries` (rejection retry) and `timeout` (time expiry)
- [ ] I can implement a persistent append-only audit log
