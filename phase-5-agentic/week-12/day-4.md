# Phase 5 · Week 12 · Day 4 — HITL Approval Gates + Audit Logging + Security

> Objective: Implement human-in-the-loop approval gates, thread-safe audit logging, and input sanitization for enterprise-grade agent security.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Autonomous agents operating on financial data need three safety layers:

1. **HITL gates** — block irreversible or high-value actions until a human explicitly approves or rejects
2. **Audit logs** — append-only JSONL record of every significant event for compliance and debugging
3. **Input sanitization** — strip credentials and validate inputs before they reach the LLM or tool layer

These are not optional features — they are architectural requirements for any agent touching real financial workflows. Regulators, security teams, and incident responders all depend on these controls existing and being trustworthy.

**Why HITL is architectural, not bolted on:**
The `ToolSchema.risk_level` field is set at registration time, not at call time. This means the gate decision is made by the framework, not by the agent's reasoning. An agent cannot accidentally bypass the gate by choosing a different phrasing.

**Audit log properties that matter:**
- **Append-only**: no delete, no update — only `open("a")`
- **Thread-safe**: multiple agent threads may log concurrently; use a `threading.Lock`
- **Structured**: JSONL (one JSON object per line) for easy streaming and `jq` queries
- **Typed**: `AuditEvent` Pydantic model enforces field names and types at write time

## How it works internally

**HITL decision flow:**

```
tool call requested
        │
        ▼
ToolSchema.risk_level ──── low/medium ──── auto_approve_levels? ──► APPROVED
        │                                                                │
        │ high/critical                                                  │
        ▼                                                                │
_pending[request_id] = req                                              │
        │                                                                │
        ▼                                                                │
return ApprovalRequest(status=PENDING) ◄── caller polls /hitl/pending   │
        │                                                                │
human reviews                                                           │
        │                                                                │
gate.resolve(request_id, approved, reviewer)                            │
        │                                                                │
        ▼                                                                │
status = APPROVED / REJECTED ───────────────────────────────────────────┘
```

**Risk levels and their default gate behavior:**

| risk_level | Auto-approved? | Notes |
|---|---|---|
| `low` | Yes (default) | Read-only lookups, status checks |
| `medium` | Configurable | Validation, non-destructive writes |
| `high` | No — human required | Financial approvals under threshold |
| `critical` | No — human required | Large payments, escalations, deletions |

**Audit event lifecycle:**
`AuditLogger.log_event("agent_run_start", session_id=..., input=...)` is called at agent entry. Tool calls, approval requests, approval resolutions, and errors each emit their own event type. A compliance reviewer can reconstruct the full decision trail by filtering `audit.jsonl`.

**Security principles applied:**
- `sanitize_input` runs before any user text reaches the LLM
- `validate_invoice_id` rejects malformed IDs before touching the tool layer
- Credentials are never logged — `_SENSITIVE_PATTERNS` redacts known formats
- `approve_payment` requires a `HumanApprovalGate` injection (testable, not global state)

## Worked example

HITL approval gate:

```python
# src/finops_agent/hitl/gate.py
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"

class ApprovalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    tool_name: str
    parameters: dict
    risk_level: str
    reason: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: str | None = None

class HumanApprovalGate:
    def __init__(self, auto_approve_levels: set[str] | None = None, timeout_seconds: int = 300):
        # "low" can auto-approve; "medium", "high", "critical" need human by default
        self.auto_approve_levels = auto_approve_levels or {"low"}
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, ApprovalRequest] = {}

    def request_approval(self, req: ApprovalRequest) -> ApprovalRequest:
        if req.risk_level in self.auto_approve_levels:
            req.status = ApprovalStatus.APPROVED
            req.reviewed_by = "auto"
            req.reviewed_at = datetime.now(timezone.utc).isoformat()
            logger.info("Auto-approved %s (risk=%s)", req.tool_name, req.risk_level)
            return req

        self._pending[req.request_id] = req
        logger.warning(
            "HITL gate: awaiting human approval for %s (risk=%s, id=%s)",
            req.tool_name, req.risk_level, req.request_id
        )
        return req

    def resolve(self, request_id: str, approved: bool, reviewer: str) -> ApprovalRequest:
        req = self._pending.pop(request_id, None)
        if not req:
            raise ValueError(f"No pending approval with id={request_id}")
        req.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        req.reviewed_by = reviewer
        req.reviewed_at = datetime.now(timezone.utc).isoformat()
        logger.info("Resolved approval %s: status=%s by %s", request_id, req.status, reviewer)
        return req

    def get_pending(self) -> list[ApprovalRequest]:
        return list(self._pending.values())
```

Thread-safe append-only audit logger:

```python
# src/finops_agent/audit/logger.py
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class AuditEvent(BaseModel):
    timestamp: str
    event_type: str
    session_id: str | None = None
    user_id: str | None = None
    tool_name: str | None = None
    risk_level: str | None = None
    status: str | None = None
    details: dict = {}

class AuditLogger:
    def __init__(self, log_path: str = "logs/audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        logger.info("AuditLogger writing to %s", self.log_path)

    def log_event(self, event_type: str, **kwargs) -> AuditEvent:
        event = AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            **{k: v for k, v in kwargs.items() if k in AuditEvent.model_fields},
            details={k: v for k, v in kwargs.items() if k not in AuditEvent.model_fields}
        )
        with self._lock:
            with self.log_path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
        return event

    def read_events(self, event_type: str | None = None) -> list[AuditEvent]:
        if not self.log_path.exists():
            return []
        events = []
        with self.log_path.open() as f:
            for line in f:
                e = AuditEvent.model_validate_json(line.strip())
                if event_type is None or e.event_type == event_type:
                    events.append(e)
        return events
```

Controlled tool with gate integration:

```python
# src/finops_agent/tools/controlled.py
import logging
from finops_agent.hitl.gate import ApprovalRequest, ApprovalStatus, HumanApprovalGate

logger = logging.getLogger(__name__)

def approve_payment(
    invoice_id: str,
    amount: float,
    session_id: str,
    gate: HumanApprovalGate,
) -> dict:
    req = ApprovalRequest(
        session_id=session_id,
        tool_name="approve_payment",
        parameters={"invoice_id": invoice_id, "amount": amount},
        risk_level="critical",
        reason=f"Payment approval for {invoice_id} (${amount:.2f})",
    )
    result = gate.request_approval(req)
    if result.status == ApprovalStatus.APPROVED:
        logger.info("Payment approved for %s: $%.2f", invoice_id, amount)
        return {"approved": True, "invoice_id": invoice_id, "amount": amount, "approval_id": result.request_id}
    elif result.status == ApprovalStatus.PENDING:
        return {"approved": False, "pending": True, "request_id": result.request_id, "message": "Awaiting human review"}
    else:
        return {"approved": False, "rejected": True, "reason": f"Status: {result.status}"}
```

Input sanitization:

```python
# src/finops_agent/core/sanitizer.py
import logging
import re

logger = logging.getLogger(__name__)

# Never log tokens or credentials
_SENSITIVE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),           # API keys
    re.compile(r"\b[A-Z0-9]{20,}\b"),              # AWS-style secrets
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}"),  # Bearer tokens
]

def sanitize_input(text: str) -> str:
    """Strip known credential patterns before LLM submission."""
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text

def validate_invoice_id(invoice_id: str) -> str:
    if not re.match(r"^INV-\d{3,10}$", invoice_id):
        raise ValueError(f"Invalid invoice_id format: {invoice_id!r}")
    return invoice_id
```

FastAPI HITL management endpoints:

```python
# src/finops_agent/api/hitl_router.py
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from finops_agent.hitl.gate import HumanApprovalGate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/hitl", tags=["hitl"])

class ResolveRequest(BaseModel):
    approved: bool
    reviewer: str

def make_hitl_router(gate: HumanApprovalGate) -> APIRouter:
    @router.get("/pending")
    def get_pending():
        return {"pending": [r.model_dump() for r in gate.get_pending()]}

    @router.post("/resolve/{request_id}")
    def resolve_approval(request_id: str, body: ResolveRequest):
        try:
            result = gate.resolve(request_id, body.approved, body.reviewer)
            return result.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return router
```

**How the full HITL flow looks end-to-end:**

```python
# Illustrative integration — not a runnable script without the full framework wired
import os
import anthropic
from finops_agent.hitl.gate import HumanApprovalGate
from finops_agent.tools.controlled import approve_payment
from finops_agent.audit.logger import AuditLogger

gate = HumanApprovalGate(auto_approve_levels={"low"})  # critical requires human
audit = AuditLogger("logs/audit.jsonl")

# Agent calls the tool — returns pending because risk_level="critical"
result = approve_payment("INV-001", 15000.0, session_id="sess-123", gate=gate)
# result = {"approved": False, "pending": True, "request_id": "uuid-...", ...}

audit.log_event("approval_requested", session_id="sess-123",
                tool_name="approve_payment", risk_level="critical")

# Human reviewer hits the API endpoint or calls resolve() directly
pending = gate.get_pending()
gate.resolve(pending[0].request_id, approved=True, reviewer="jane.doe@corp.com")

audit.log_event("approval_resolved", session_id="sess-123",
                status="approved", user_id="jane.doe@corp.com")
```

## Exercises

1. Add `timeout_seconds` enforcement to `HumanApprovalGate.request_approval`. When a request is added to `_pending`, start a background `threading.Timer` that calls a private `_timeout_request(request_id)` method after `self.timeout_seconds` seconds. `_timeout_request` should set the request's `status` to `TIMED_OUT` and remove it from `_pending`. Handle the race condition where `resolve()` and the timer fire at the same time.

2. Extend `AuditLogger` with a `read_events_for_session(session_id: str) -> list[AuditEvent]` method. Write a unit test that creates a temp-file-backed `AuditLogger`, writes 5 events (2 for `"session-A"`, 3 for `"session-B"`), and verifies `read_events_for_session("session-A")` returns exactly 2 events and `read_events_for_session("session-B")` returns exactly 3.

3. Add `validate_amount(amount: float) -> float` to `sanitizer.py` that raises `ValueError` for: (a) negative or zero values, (b) values over `1_000_000` (enterprise payment limit). Write 4 unit tests: boundary at 0, a normal value, exactly `1_000_000`, and `1_000_001`.

4. Write a unit test for `approve_payment` that uses `unittest.mock.MagicMock` to mock `HumanApprovalGate`. Verify: (a) calling `approve_payment` with a `critical` gate that has no human reviewer returns `{"pending": True, ...}`; (b) after calling `gate.resolve(request_id, True, "reviewer")`, a second invocation with a gate configured to auto-approve `"critical"` returns `{"approved": True, ...}`. Hint: configure `auto_approve_levels={"low", "medium", "high", "critical"}` for the second test case.

## Self-check

- [ ] I can implement a HITL gate with auto-approve for low-risk and human-block for critical actions
- [ ] I can write a thread-safe append-only audit logger using `threading.Lock` and JSONL format
- [ ] I understand how `sanitize_input` prevents credential leakage into LLM context
- [ ] I can trace the `risk_level` → gate decision → approval resolution flow end-to-end
