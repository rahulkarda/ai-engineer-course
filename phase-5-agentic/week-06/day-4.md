# Phase 5 · Week 06 · Day 4 — Human-in-the-Loop: Approval Gates and Interrupts

> Objective: Add approval gates to an agent workflow so sensitive actions pause and wait for human review before proceeding.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Fully autonomous agents occasionally propose irreversible or high-risk actions (send email, delete record, make payment). Human-in-the-loop (HITL) patterns intercept these actions before execution and require explicit human approval.

Three primitives:

1. **ApprovalGate** — wraps a proposed action. Starts `PENDING`, transitions to `APPROVED` or `REJECTED` when a human responds.
2. **HumanApprovalQueue** — a queue of pending gates; humans dequeue, inspect, and decide.
3. **AgentWorkflow** — the agent loop. Before executing a sensitive action it submits a gate, blocks (or pauses), and resumes only on approval.

This keeps dangerous side-effects out of the hot path while letting safe actions run uninterrupted.

## How it works internally

**Gate lifecycle**:

```
PENDING → APPROVED → (action executes)
        ↘ REJECTED → (action aborts, agent receives rejection reason)
```

**Blocking vs. async**: synchronous workflows call `gate.wait()` (blocks the thread). Async workflows `await gate.wait_async()`. Both poll the gate's internal event — set by the reviewer's `approve()` / `reject()` call.

**Queue mechanics**: `HumanApprovalQueue` is a simple `asyncio.Queue` / thread-safe `queue.Queue`. Reviewers consume from it in a separate loop (or a UI). The queue decouples producers (agents) from consumers (humans) so agents don't need to know who reviews.

**Sensitive-action detection**: `AgentWorkflow` keeps a set of action names (`SENSITIVE_ACTIONS`). Any proposed action in that set is intercepted; others run directly. This is the cheapest classification strategy — extend with LLM scoring (see exercises) for unknown action names.

**Why this beats post-hoc audit**: intercepting before execution means the harmful side-effect never happens. Post-hoc audit can only alert; it cannot undo a sent email.

## Worked example

```python
"""
Human-in-the-loop approval gates for an agent workflow.

Run (interactive demo — reviewer runs in a background thread):
    export ANTHROPIC_API_KEY="sk-ant-..."
    python day4_hitl_approval_gates.py
"""

import logging
import os
import queue
import threading
import time
import uuid
from enum import Enum

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────

class GateState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_name: str
    action_payload: dict
    requester: str
    rationale: str


class ApprovalDecision(BaseModel):
    request_id: str
    state: GateState  # APPROVED or REJECTED
    reviewer: str
    comment: str = ""


# ── ApprovalGate ──────────────────────────────────────────────────────────────

class ApprovalGate:
    """
    Wraps a single proposed action.
    Thread-safe: agent calls wait(); reviewer calls approve()/reject().
    """

    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        self._state = GateState.PENDING
        self._decision: ApprovalDecision | None = None
        self._event = threading.Event()

    @property
    def state(self) -> GateState:
        return self._state

    @property
    def decision(self) -> ApprovalDecision | None:
        return self._decision

    def approve(self, reviewer: str, comment: str = "") -> None:
        self._decision = ApprovalDecision(
            request_id=self.request.request_id,
            state=GateState.APPROVED,
            reviewer=reviewer,
            comment=comment,
        )
        self._state = GateState.APPROVED
        self._event.set()
        logger.info("Gate APPROVED by %s: %s", reviewer, self.request.action_name)

    def reject(self, reviewer: str, comment: str = "") -> None:
        self._decision = ApprovalDecision(
            request_id=self.request.request_id,
            state=GateState.REJECTED,
            reviewer=reviewer,
            comment=comment,
        )
        self._state = GateState.REJECTED
        self._event.set()
        logger.info("Gate REJECTED by %s: %s — %s", reviewer, self.request.action_name, comment)

    def wait(self, timeout: float | None = 60.0) -> GateState:
        """Block until a decision is made. Returns final GateState."""
        resolved = self._event.wait(timeout=timeout)
        if not resolved:
            logger.warning("Gate timed out for action: %s", self.request.action_name)
            self._state = GateState.REJECTED
        return self._state


# ── HumanApprovalQueue ────────────────────────────────────────────────────────

class HumanApprovalQueue:
    """
    Thread-safe queue that decouples agents (producers) from reviewers (consumers).
    """

    def __init__(self) -> None:
        self._q: queue.Queue[ApprovalGate] = queue.Queue()

    def submit(self, gate: ApprovalGate) -> None:
        logger.info(
            "Queued for review: action=%s id=%s",
            gate.request.action_name,
            gate.request.request_id[:8],
        )
        self._q.put(gate)

    def next_pending(self, timeout: float = 5.0) -> ApprovalGate | None:
        """Pop the next gate. Returns None on timeout (no items)."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def pending_count(self) -> int:
        return self._q.qsize()


# ── AgentWorkflow ─────────────────────────────────────────────────────────────

SENSITIVE_ACTIONS: frozenset[str] = frozenset(
    {"send_email", "delete_record", "make_payment", "update_permissions"}
)


class AgentWorkflow:
    """
    Simulates an agent that proposes actions.
    Sensitive actions are intercepted and sent to the approval queue.
    """

    def __init__(
        self,
        approval_queue: HumanApprovalQueue,
        model: str = "claude-sonnet-5-5",
    ) -> None:
        self._queue = approval_queue
        self._model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ------------------------------------------------------------------
    # Internal action stubs (would call real services in production)
    # ------------------------------------------------------------------

    def _execute_send_email(self, payload: dict) -> str:
        return f"Email sent to {payload.get('to')} — subject: {payload.get('subject')}"

    def _execute_delete_record(self, payload: dict) -> str:
        return f"Record {payload.get('record_id')} deleted from {payload.get('table')}"

    def _execute_make_payment(self, payload: dict) -> str:
        return f"Payment of {payload.get('amount')} {payload.get('currency')} processed"

    def _execute_safe_lookup(self, payload: dict) -> str:
        return f"Lookup result for {payload.get('query')}: [mock data]"

    _EXECUTORS = {
        "send_email": _execute_send_email,
        "delete_record": _execute_delete_record,
        "make_payment": _execute_make_payment,
        "safe_lookup": _execute_safe_lookup,
    }

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def propose_action(
        self,
        action_name: str,
        payload: dict,
        rationale: str = "",
    ) -> str:
        """
        Propose an action. If sensitive, submit to the approval queue and
        wait for a human decision before executing.
        """
        if action_name in SENSITIVE_ACTIONS:
            return self._gated_execute(action_name, payload, rationale)
        return self._direct_execute(action_name, payload)

    def _direct_execute(self, action_name: str, payload: dict) -> str:
        executor = self._EXECUTORS.get(action_name)
        if executor is None:
            return f"[no executor for '{action_name}']"
        logger.info("Direct execute: %s", action_name)
        return executor(self, payload)

    def _gated_execute(self, action_name: str, payload: dict, rationale: str) -> str:
        request = ApprovalRequest(
            action_name=action_name,
            action_payload=payload,
            requester="AgentWorkflow",
            rationale=rationale or f"Agent proposes: {action_name}",
        )
        gate = ApprovalGate(request)
        self._queue.submit(gate)

        logger.info("Waiting for human approval on: %s …", action_name)
        final_state = gate.wait(timeout=30.0)

        if final_state == GateState.APPROVED:
            executor = self._EXECUTORS.get(action_name)
            if executor is None:
                return f"[approved but no executor for '{action_name}']"
            result = executor(self, payload)
            logger.info("Action executed after approval: %s", action_name)
            return result
        else:
            comment = gate.decision.comment if gate.decision else "no comment"
            logger.info("Action aborted (rejected/timeout): %s — %s", action_name, comment)
            return f"Action '{action_name}' was not approved: {comment}"

    def run_llm_task(self, user_request: str) -> str:
        """Ask the LLM what action to take, then propose it."""
        logger.info("LLM deciding action for: %r", user_request)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=128,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"You are an agent. Decide ONE action to take for this request.\n"
                        f"Actions available: send_email, delete_record, make_payment, safe_lookup.\n"
                        f"Reply with JSON only: "
                        f'{{ "action": "<name>", "rationale": "<one sentence>" }}\n\n'
                        f"Request: {user_request}"
                    ),
                }
            ],
        )
        import json

        parsed = json.loads(response.content[0].text.strip())
        action = parsed["action"]
        rationale = parsed.get("rationale", "")
        # Construct a minimal payload for demo purposes
        payload: dict = {"query": user_request, "request": user_request}
        return self.propose_action(action, payload, rationale)


# ── Simulated reviewer loop (runs in a background thread) ─────────────────────

def simulated_reviewer(approval_queue: HumanApprovalQueue, stop_event: threading.Event) -> None:
    """
    Mimics a human reviewer consuming gates and making decisions.
    In production this would be a UI or Slack bot.
    """
    reviewer_id = "human-reviewer-1"
    while not stop_event.is_set():
        gate = approval_queue.next_pending(timeout=2.0)
        if gate is None:
            continue
        # Simulate human thinking time
        time.sleep(1.0)
        action = gate.request.action_name
        # Approve everything except payments > $1000 (demo rule)
        payload = gate.request.action_payload
        amount = payload.get("amount", 0)
        if action == "make_payment" and isinstance(amount, (int, float)) and amount > 1000:
            gate.reject(reviewer_id, comment="Amount exceeds auto-approval limit")
        else:
            gate.approve(reviewer_id, comment="Looks good")


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    approval_queue = HumanApprovalQueue()
    workflow = AgentWorkflow(approval_queue)

    # Start the simulated reviewer in a background thread
    stop_event = threading.Event()
    reviewer_thread = threading.Thread(
        target=simulated_reviewer,
        args=(approval_queue, stop_event),
        daemon=True,
    )
    reviewer_thread.start()

    # --- Scenario 1: safe action — no gate needed ---
    print("\n--- Scenario 1: Safe lookup (no gate) ---")
    result = workflow.propose_action("safe_lookup", {"query": "user profile for id=42"})
    print(f"Result: {result}")

    # --- Scenario 2: sensitive action — gate intercepts, reviewer approves ---
    print("\n--- Scenario 2: Send email (gated, approved) ---")
    result = workflow.propose_action(
        "send_email",
        {"to": "alice@example.com", "subject": "Q3 report", "body": "Please find attached."},
        rationale="User requested Q3 report be emailed to Alice",
    )
    print(f"Result: {result}")

    # --- Scenario 3: payment above threshold — reviewer rejects ---
    print("\n--- Scenario 3: Large payment (gated, rejected) ---")
    result = workflow.propose_action(
        "make_payment",
        {"amount": 5000, "currency": "USD", "recipient": "vendor-x"},
        rationale="Settle vendor invoice",
    )
    print(f"Result: {result}")

    stop_event.set()
    reviewer_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
```

## Exercises

1. Replace the `threading.Event` gate with an `asyncio.Event` and rewrite `ApprovalGate.wait_async` as a coroutine. Update `AgentWorkflow` to use `async def propose_action`. Verify the same three scenarios still pass.
2. Add a `deadline: datetime` field to `ApprovalRequest`. If the gate is still `PENDING` when `deadline` is reached, auto-reject it and log a warning. Write a pytest test that mocks `time.sleep` to trigger the deadline.
3. Extend `HumanApprovalQueue` to support priority levels (`HIGH`, `NORMAL`, `LOW`) using a `PriorityQueue`. Payments become `HIGH`, emails `NORMAL`, lookups `LOW`. Verify that a `HIGH` gate submitted after a `NORMAL` gate is processed first.
4. Add a `max_retries: int` parameter to `AgentWorkflow._gated_execute`. When the gate is rejected, re-submit the gate up to `max_retries` times with an auto-modified rationale (e.g. appending "Retry N/3"). Write a test that rejects the first two attempts and approves the third.

(Solutions: ../solutions/week-06/day-4.md)

## Self-check

- [ ] I can explain the PENDING → APPROVED/REJECTED gate lifecycle and why it prevents irreversible side-effects.
- [ ] I can implement `ApprovalGate` with a `threading.Event` and correct `wait()` semantics from memory.
- [ ] I can describe how `HumanApprovalQueue` decouples agents from reviewers.
- [ ] I can implement `AgentWorkflow.propose_action` with sensitive-action detection and gated execution without reference.
- [ ] I can articulate the trade-offs between synchronous blocking gates and async/callback-based approval patterns.
