# Solutions — Week 06 · Day 6 — Approval Workflows for Sensitive Actions

> These are reference solutions. Your implementation may differ and still be correct.

---

## Exercise 1 — Sequential Approval Chain with State Machine

Approver 2 only receives the request after Approver 1 has approved.

```python
# src/workflow/sequential_approval.py
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class ApprovalState(str, Enum):
    PENDING_APPROVER1 = "pending_approver1"
    PENDING_APPROVER2 = "pending_approver2"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    requester: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ApprovalRecord(BaseModel):
    approver_id: str
    decision: str           # "approve" | "reject"
    comment: str = ""
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SequentialApprovalChain:
    """
    Two-stage sequential approval.
    Approver 2 is notified only after Approver 1 approves.
    """

    # Valid transitions
    _TRANSITIONS: dict[ApprovalState, list[ApprovalState]] = {
        ApprovalState.PENDING_APPROVER1: [
            ApprovalState.PENDING_APPROVER2,
            ApprovalState.REJECTED,
        ],
        ApprovalState.PENDING_APPROVER2: [
            ApprovalState.APPROVED,
            ApprovalState.REJECTED,
        ],
        ApprovalState.APPROVED: [],
        ApprovalState.REJECTED: [],
    }

    def __init__(
        self,
        request: ApprovalRequest,
        approver1_id: str,
        approver2_id: str,
    ) -> None:
        self.request = request
        self.approver1_id = approver1_id
        self.approver2_id = approver2_id
        self.state: ApprovalState = ApprovalState.PENDING_APPROVER1
        self.records: list[ApprovalRecord] = []

    def _transition(self, new_state: ApprovalState) -> None:
        allowed = self._TRANSITIONS[self.state]
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition {self.state} → {new_state}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        logger.info(
            "approval_chain state_transition request_id=%s %s → %s",
            self.request.request_id,
            self.state.value,
            new_state.value,
        )
        self.state = new_state

    def decide_approver1(self, decision: str, comment: str = "") -> None:
        if self.state != ApprovalState.PENDING_APPROVER1:
            raise ValueError(f"Not waiting for approver1 (state={self.state.value})")
        record = ApprovalRecord(
            approver_id=self.approver1_id, decision=decision, comment=comment
        )
        self.records.append(record)

        if decision == "approve":
            self._transition(ApprovalState.PENDING_APPROVER2)
            logger.info(
                "Approver1 approved; notifying approver2=%s", self.approver2_id
            )
            self._notify_approver2()
        elif decision == "reject":
            self._transition(ApprovalState.REJECTED)
        else:
            raise ValueError(f"Unknown decision: {decision!r}")

    def decide_approver2(self, decision: str, comment: str = "") -> None:
        if self.state != ApprovalState.PENDING_APPROVER2:
            raise ValueError(f"Not waiting for approver2 (state={self.state.value})")
        record = ApprovalRecord(
            approver_id=self.approver2_id, decision=decision, comment=comment
        )
        self.records.append(record)

        if decision == "approve":
            self._transition(ApprovalState.APPROVED)
        elif decision == "reject":
            self._transition(ApprovalState.REJECTED)
        else:
            raise ValueError(f"Unknown decision: {decision!r}")

    def _notify_approver2(self) -> None:
        """Stub — real impl sends email or pushes to message queue."""
        logger.info(
            "notify approver2=%s for request_id=%s",
            self.approver2_id,
            self.request.request_id,
        )
```

### Tests

```python
# tests/unit/test_sequential_approval.py
import pytest

from src.workflow.sequential_approval import (
    ApprovalRequest,
    ApprovalState,
    SequentialApprovalChain,
)


@pytest.fixture()
def chain() -> SequentialApprovalChain:
    req = ApprovalRequest(
        description="Deploy to production",
        requester="engineer@example.com",
    )
    return SequentialApprovalChain(req, approver1_id="mgr1", approver2_id="mgr2")


def test_initial_state_pending_approver1(chain: SequentialApprovalChain) -> None:
    assert chain.state == ApprovalState.PENDING_APPROVER1


def test_approver2_cannot_decide_before_approver1(chain: SequentialApprovalChain) -> None:
    with pytest.raises(ValueError, match="Not waiting for approver2"):
        chain.decide_approver2("approve")


def test_approver1_approve_moves_to_pending_approver2(chain: SequentialApprovalChain) -> None:
    chain.decide_approver1("approve")
    assert chain.state == ApprovalState.PENDING_APPROVER2


def test_full_approval_path(chain: SequentialApprovalChain) -> None:
    chain.decide_approver1("approve", comment="Looks good")
    chain.decide_approver2("approve", comment="LGTM")
    assert chain.state == ApprovalState.APPROVED
    assert len(chain.records) == 2


def test_approver1_reject_terminates(chain: SequentialApprovalChain) -> None:
    chain.decide_approver1("reject", comment="Not ready")
    assert chain.state == ApprovalState.REJECTED
    # approver2 should not be contactable after rejection
    with pytest.raises(ValueError):
        chain.decide_approver2("approve")


def test_approver2_reject_terminates(chain: SequentialApprovalChain) -> None:
    chain.decide_approver1("approve")
    chain.decide_approver2("reject", comment="Policy violation")
    assert chain.state == ApprovalState.REJECTED


def test_invalid_decision_raises(chain: SequentialApprovalChain) -> None:
    with pytest.raises(ValueError, match="Unknown decision"):
        chain.decide_approver1("maybe")


def test_double_decide_approver1_raises(chain: SequentialApprovalChain) -> None:
    chain.decide_approver1("approve")
    with pytest.raises(ValueError, match="Not waiting for approver1"):
        chain.decide_approver1("approve")
```

---

## Exercise 2 — Email Notification Stubs

`notify_approver` logs but does not send real email; wired into `WorkflowEngine` on PENDING.

```python
# src/workflow/notifications.py
from __future__ import annotations

import logging

from src.workflow.sequential_approval import ApprovalRequest

logger = logging.getLogger(__name__)


def notify_approver(approver_id: str, request: ApprovalRequest) -> None:
    """
    Stub notification. In production replace the log with an email/Slack call.
    Intentionally returns None so callers don't depend on return value.
    """
    logger.info(
        "NOTIFY approver=%s request_id=%s description=%r",
        approver_id,
        request.request_id,
        request.description,
    )
```

```python
# src/workflow/engine.py
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from src.workflow.notifications import notify_approver
from src.workflow.sequential_approval import (
    ApprovalRequest,
    ApprovalState,
    SequentialApprovalChain,
)

logger = logging.getLogger(__name__)


class WorkflowEntry(BaseModel):
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    chain: Any  # SequentialApprovalChain not serialisable — stored by reference
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True


class WorkflowEngine:
    def __init__(self) -> None:
        self._workflows: dict[str, SequentialApprovalChain] = {}

    def submit(
        self,
        request: ApprovalRequest,
        approver1_id: str,
        approver2_id: str,
    ) -> str:
        chain = SequentialApprovalChain(request, approver1_id, approver2_id)
        self._workflows[request.request_id] = chain
        logger.info("Workflow submitted request_id=%s", request.request_id)
        # Notify first approver immediately on PENDING
        notify_approver(approver1_id, request)
        return request.request_id

    def decide(
        self, request_id: str, approver_id: str, decision: str, comment: str = ""
    ) -> ApprovalState:
        chain = self._get(request_id)

        if chain.state == ApprovalState.PENDING_APPROVER1:
            if approver_id != chain.approver1_id:
                raise PermissionError(f"{approver_id} is not approver1 for {request_id}")
            chain.decide_approver1(decision, comment)
            if chain.state == ApprovalState.PENDING_APPROVER2:
                notify_approver(chain.approver2_id, chain.request)

        elif chain.state == ApprovalState.PENDING_APPROVER2:
            if approver_id != chain.approver2_id:
                raise PermissionError(f"{approver_id} is not approver2 for {request_id}")
            chain.decide_approver2(decision, comment)
        else:
            raise ValueError(f"Workflow {request_id} is in terminal state {chain.state.value}")

        return chain.state

    def _get(self, request_id: str) -> SequentialApprovalChain:
        try:
            return self._workflows[request_id]
        except KeyError as exc:
            raise KeyError(f"No workflow found for request_id={request_id}") from exc
```

### Tests

```python
# tests/unit/test_workflow_engine.py
from unittest.mock import call, patch

import pytest

from src.workflow.engine import WorkflowEngine
from src.workflow.sequential_approval import ApprovalRequest, ApprovalState


@pytest.fixture()
def engine() -> WorkflowEngine:
    return WorkflowEngine()


@pytest.fixture()
def request() -> ApprovalRequest:
    return ApprovalRequest(description="DB migration", requester="dev@example.com")


def test_submit_notifies_approver1(engine: WorkflowEngine, request: ApprovalRequest) -> None:
    with patch("src.workflow.engine.notify_approver") as mock_notify:
        engine.submit(request, approver1_id="a1", approver2_id="a2")
        mock_notify.assert_called_once_with("a1", request)


def test_approver1_approve_notifies_approver2(
    engine: WorkflowEngine, request: ApprovalRequest
) -> None:
    with patch("src.workflow.engine.notify_approver") as mock_notify:
        rid = engine.submit(request, approver1_id="a1", approver2_id="a2")
        engine.decide(rid, "a1", "approve")
        # First call: submit → notify a1; second call: a1 approve → notify a2
        assert mock_notify.call_count == 2
        calls = [c.args[0] for c in mock_notify.call_args_list]
        assert calls == ["a1", "a2"]


def test_full_approval(engine: WorkflowEngine, request: ApprovalRequest) -> None:
    with patch("src.workflow.engine.notify_approver"):
        rid = engine.submit(request, approver1_id="a1", approver2_id="a2")
        engine.decide(rid, "a1", "approve")
        state = engine.decide(rid, "a2", "approve")
    assert state == ApprovalState.APPROVED


def test_wrong_approver_raises(engine: WorkflowEngine, request: ApprovalRequest) -> None:
    with patch("src.workflow.engine.notify_approver"):
        rid = engine.submit(request, approver1_id="a1", approver2_id="a2")
        with pytest.raises(PermissionError):
            engine.decide(rid, "a2", "approve")  # a2 can't decide at stage 1
```

---

## Exercise 3 — Workflow Expiry via asyncio

Requests older than `timeout_seconds` auto-transition to EXPIRED.

```python
# src/workflow/expiry.py
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.workflow.sequential_approval import ApprovalRequest, ApprovalState

logger = logging.getLogger(__name__)


class ExtendedState(str, Enum):
    PENDING_APPROVER1 = ApprovalState.PENDING_APPROVER1.value
    PENDING_APPROVER2 = ApprovalState.PENDING_APPROVER2.value
    APPROVED = ApprovalState.APPROVED.value
    REJECTED = ApprovalState.REJECTED.value
    EXPIRED = "expired"


TERMINAL_STATES = {
    ExtendedState.APPROVED,
    ExtendedState.REJECTED,
    ExtendedState.EXPIRED,
}


class ExpiryWorkflow(BaseModel):
    request: ApprovalRequest
    state: ExtendedState = ExtendedState.PENDING_APPROVER1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True

    def is_expired(self, timeout_seconds: float) -> bool:
        if self.state in TERMINAL_STATES:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > timeout_seconds


class ExpiryManager:
    def __init__(self, timeout_seconds: float = 86400.0, poll_interval: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self._workflows: dict[str, ExpiryWorkflow] = {}
        self._task: asyncio.Task[None] | None = None

    def register(self, workflow: ExpiryWorkflow) -> None:
        self._workflows[workflow.request.request_id] = workflow
        logger.info("Registered workflow request_id=%s", workflow.request.request_id)

    async def _expire_loop(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval)
            self._check_and_expire()

    def _check_and_expire(self) -> None:
        for wf in self._workflows.values():
            if wf.is_expired(self.timeout_seconds):
                logger.warning(
                    "Expiring request_id=%s (age > %s s)",
                    wf.request.request_id,
                    self.timeout_seconds,
                )
                wf.state = ExtendedState.EXPIRED

    def start(self) -> None:
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._expire_loop())
        logger.info("ExpiryManager started (timeout=%ss)", self.timeout_seconds)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None
```

### Tests

```python
# tests/unit/test_expiry.py
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from src.workflow.expiry import ExpiryManager, ExpiryWorkflow, ExtendedState
from src.workflow.sequential_approval import ApprovalRequest


def _old_workflow(seconds_old: float) -> ExpiryWorkflow:
    req = ApprovalRequest(description="Old request", requester="x@example.com")
    wf = ExpiryWorkflow(request=req)
    wf.created_at = datetime.now(timezone.utc) - timedelta(seconds=seconds_old)
    return wf


def test_non_expired_workflow_not_marked() -> None:
    manager = ExpiryManager(timeout_seconds=3600)
    wf = _old_workflow(seconds_old=100)
    manager.register(wf)
    manager._check_and_expire()
    assert wf.state == ExtendedState.PENDING_APPROVER1


def test_expired_workflow_marked_expired() -> None:
    manager = ExpiryManager(timeout_seconds=60)
    wf = _old_workflow(seconds_old=120)
    manager.register(wf)
    manager._check_and_expire()
    assert wf.state == ExtendedState.EXPIRED


def test_terminal_state_not_re_expired() -> None:
    manager = ExpiryManager(timeout_seconds=60)
    wf = _old_workflow(seconds_old=120)
    wf.state = ExtendedState.APPROVED
    manager.register(wf)
    manager._check_and_expire()
    assert wf.state == ExtendedState.APPROVED  # not changed


@pytest.mark.asyncio
async def test_expire_loop_runs() -> None:
    manager = ExpiryManager(timeout_seconds=60, poll_interval=0.05)
    wf = _old_workflow(seconds_old=120)
    manager.register(wf)
    manager.start()
    await asyncio.sleep(0.15)
    manager.stop()
    assert wf.state == ExtendedState.EXPIRED
```

---

## Exercise 4 — Workflow Stats Dashboard

```python
# src/workflow/stats.py
from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel

from src.workflow.engine import WorkflowEngine
from src.workflow.sequential_approval import ApprovalState

logger = logging.getLogger(__name__)


class WorkflowStats(BaseModel):
    total: int
    by_state: dict[str, int]
    average_approval_seconds: float | None  # None if no completed workflows


def get_workflow_stats(engine: WorkflowEngine) -> WorkflowStats:
    """Compute dashboard statistics from a WorkflowEngine."""
    by_state: dict[str, int] = {s.value: 0 for s in ApprovalState}
    approval_durations: list[float] = []

    for chain in engine._workflows.values():
        by_state[chain.state.value] = by_state.get(chain.state.value, 0) + 1

        if chain.state == ApprovalState.APPROVED and chain.records:
            # Duration = time from request creation to last approval record
            last_decision_time = chain.records[-1].decided_at
            duration = (last_decision_time - chain.request.created_at).total_seconds()
            approval_durations.append(duration)

    avg = (
        sum(approval_durations) / len(approval_durations)
        if approval_durations
        else None
    )

    stats = WorkflowStats(
        total=sum(by_state.values()),
        by_state={k: v for k, v in by_state.items() if v > 0},
        average_approval_seconds=avg,
    )
    logger.info("workflow_stats total=%d avg_approval=%.1f", stats.total, avg or 0)
    return stats
```

### Tests

```python
# tests/unit/test_workflow_stats.py
from unittest.mock import patch

import pytest

from src.workflow.engine import WorkflowEngine
from src.workflow.sequential_approval import ApprovalRequest, ApprovalState
from src.workflow.stats import WorkflowStats, get_workflow_stats


def _submit(engine: WorkflowEngine, desc: str = "req") -> str:
    req = ApprovalRequest(description=desc, requester="test@example.com")
    with patch("src.workflow.engine.notify_approver"):
        return engine.submit(req, approver1_id="a1", approver2_id="a2")


def test_empty_engine_returns_zero_stats() -> None:
    engine = WorkflowEngine()
    stats = get_workflow_stats(engine)
    assert stats.total == 0
    assert stats.average_approval_seconds is None


def test_pending_counted_correctly() -> None:
    engine = WorkflowEngine()
    _submit(engine)
    _submit(engine)
    stats = get_workflow_stats(engine)
    assert stats.total == 2
    assert stats.by_state.get(ApprovalState.PENDING_APPROVER1.value, 0) == 2


def test_approved_average_computed() -> None:
    engine = WorkflowEngine()
    rid = _submit(engine)
    with patch("src.workflow.engine.notify_approver"):
        engine.decide(rid, "a1", "approve")
        engine.decide(rid, "a2", "approve")
    stats = get_workflow_stats(engine)
    assert stats.by_state.get(ApprovalState.APPROVED.value, 0) == 1
    assert stats.average_approval_seconds is not None
    assert stats.average_approval_seconds >= 0


def test_mixed_states() -> None:
    engine = WorkflowEngine()
    rid1 = _submit(engine, "approve-me")
    rid2 = _submit(engine, "reject-me")
    rid3 = _submit(engine, "pending")
    with patch("src.workflow.engine.notify_approver"):
        engine.decide(rid1, "a1", "approve")
        engine.decide(rid1, "a2", "approve")
        engine.decide(rid2, "a1", "reject")
    stats = get_workflow_stats(engine)
    assert stats.total == 3
    assert stats.by_state[ApprovalState.APPROVED.value] == 1
    assert stats.by_state[ApprovalState.REJECTED.value] == 1
```
