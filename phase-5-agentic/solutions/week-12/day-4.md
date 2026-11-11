# Week 12 · Day 4 Solutions — HITL + Audit + Security

## Exercise 1 — Timeout enforcement with background thread

```python
# src/finops_agent/hitl/gate.py — extended with timeout thread
import threading
from datetime import datetime, timezone

class HumanApprovalGate:
    def __init__(self, auto_approve_levels: set[str] | None = None, timeout_seconds: int = 300):
        self.auto_approve_levels = auto_approve_levels or {"low"}
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, ApprovalRequest] = {}
        self._timers: dict[str, threading.Timer] = {}

    def request_approval(self, req: ApprovalRequest) -> ApprovalRequest:
        if req.risk_level in self.auto_approve_levels:
            req.status = ApprovalStatus.APPROVED
            req.reviewed_by = "auto"
            req.reviewed_at = datetime.now(timezone.utc).isoformat()
            return req

        self._pending[req.request_id] = req
        timer = threading.Timer(self.timeout_seconds, self._timeout_request, args=(req.request_id,))
        timer.daemon = True
        self._timers[req.request_id] = timer
        timer.start()
        logger.warning(
            "HITL gate: awaiting approval for %s (id=%s, timeout=%ds)",
            req.tool_name, req.request_id, self.timeout_seconds
        )
        return req

    def _timeout_request(self, request_id: str) -> None:
        req = self._pending.pop(request_id, None)
        self._timers.pop(request_id, None)
        if req:
            req.status = ApprovalStatus.TIMED_OUT
            req.reviewed_by = "system_timeout"
            req.reviewed_at = datetime.now(timezone.utc).isoformat()
            logger.warning("Approval %s timed out for tool %s", request_id, req.tool_name)

    def resolve(self, request_id: str, approved: bool, reviewer: str) -> ApprovalRequest:
        timer = self._timers.pop(request_id, None)
        if timer:
            timer.cancel()
        req = self._pending.pop(request_id, None)
        if not req:
            raise ValueError(f"No pending approval with id={request_id}")
        req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        req.reviewed_by = reviewer
        req.reviewed_at = datetime.now(timezone.utc).isoformat()
        return req
```

## Exercise 2 — read_events_for_session + unit test

```python
# src/finops_agent/audit/logger.py — addition
def read_events_for_session(self, session_id: str) -> list[AuditEvent]:
    return [e for e in self.read_events() if e.session_id == session_id]
```

Test:
```python
# tests/unit/test_audit_logger.py
import pytest
from pathlib import Path

def test_read_events_for_session(tmp_path):
    from finops_agent.audit.logger import AuditLogger

    audit = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
    # Write 2 events for session-A
    audit.log_event("tool_call", session_id="session-A", tool_name="retrieve_invoice")
    audit.log_event("tool_result", session_id="session-A", tool_name="retrieve_invoice")
    # Write 3 events for session-B
    for i in range(3):
        audit.log_event("tool_call", session_id="session-B", tool_name=f"tool-{i}")

    a_events = audit.read_events_for_session("session-A")
    b_events = audit.read_events_for_session("session-B")
    assert len(a_events) == 2
    assert len(b_events) == 3
    assert all(e.session_id == "session-A" for e in a_events)
    assert all(e.session_id == "session-B" for e in b_events)
```

## Exercise 3 — validate_amount

```python
# src/finops_agent/core/sanitizer.py — addition
ENTERPRISE_PAYMENT_LIMIT = 1_000_000.0

def validate_amount(amount: float) -> float:
    if amount <= 0:
        raise ValueError(f"Amount must be positive, got {amount}")
    if amount > ENTERPRISE_PAYMENT_LIMIT:
        raise ValueError(f"Amount {amount} exceeds enterprise limit of {ENTERPRISE_PAYMENT_LIMIT}")
    return amount
```

## Exercise 4 — Unit tests for approve_payment

```python
# tests/unit/test_controlled_tools.py
import pytest
from unittest.mock import MagicMock
from finops_agent.hitl.gate import ApprovalRequest, ApprovalStatus

def make_pending_gate(req_id: str) -> MagicMock:
    gate = MagicMock()
    pending_req = MagicMock()
    pending_req.status = ApprovalStatus.PENDING
    pending_req.request_id = req_id
    gate.request_approval.return_value = pending_req
    return gate

def make_approved_gate(req_id: str) -> MagicMock:
    gate = MagicMock()
    approved_req = MagicMock()
    approved_req.status = ApprovalStatus.APPROVED
    approved_req.request_id = req_id
    gate.request_approval.return_value = approved_req
    return gate

def test_approve_payment_returns_pending_when_critical():
    from finops_agent.tools.controlled import approve_payment
    gate = make_pending_gate("req-001")
    result = approve_payment(invoice_id="INV-001", amount=15000.0, session_id="s1", gate=gate)
    assert result["approved"] is False
    assert result["pending"] is True
    assert "request_id" in result

def test_approve_payment_returns_approved_after_resolve():
    from finops_agent.tools.controlled import approve_payment
    gate = make_approved_gate("req-001")
    result = approve_payment(invoice_id="INV-001", amount=15000.0, session_id="s1", gate=gate)
    assert result["approved"] is True
    assert result["invoice_id"] == "INV-001"
    assert result["amount"] == 15000.0
```
