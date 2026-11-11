# Phase 5 · Week 06 · Day 7 — Week 6 Project: Routing + HITL System

> Objective: Integrate the week's components into a complete Routing + Human-in-the-Loop system: specialist agent routing, confidence-based escalation, and approval gates on sensitive actions.  ·  Time: ~3 hr project + 1 hr assessment prep  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]], [[day-6]]

## Concept

This day has no worked example. You build the system from scratch, integrating everything from the week.

The target architecture:

```
User request
    │
    ▼
ConfidenceRouter  ──── confidence < escalate_threshold ──────→ HumanEscalation
    │                                                                 ▲
    │ confidence >= auto_threshold                                    │
    ▼                                                                 │
SpecialistRouter ──── routes by domain ─────┐                        │
    │                                        ▼                        │
    │                           BillingAgent / TechAgent / GeneralAgent
    │                                        │
    │                         sensitive action detected?
    │                                        │ yes
    │                                        ▼
    │                              ApprovalWorkflow
    │                                        │
    │                         approved?  ────┤
    │                                   no  │
    │                                   ▼   ▼
    │                             abort  execute
    └──────────────────────────────────────────→ response
```

### Key integration points

- `ConfidenceRouter` (day-5) gates every request before it reaches a specialist.
- `SpecialistRouter` maps domain labels (from the LLM or rule-based classifier) to agent instances.
- Each specialist agent checks whether its intended action is sensitive; if so, it submits an `ApprovalRequest` to `WorkflowEngine` (day-6) before acting.
- The `HumanEscalation` path (low-confidence or rejected approval) produces a structured `EscalationRecord` for a human operator queue.
- A shared `AuditLog` instance threads through all components so the full trace is in one place.

## How it works internally

The integration challenge is not any individual component — it is **threading shared state** without coupling the components:

1. **Dependency injection over global state.** Pass `AuditLog`, `ThresholdSettings`, and `WorkflowEngine` factory into the top-level orchestrator at construction time. Components receive what they need; they do not import globals.

2. **Typed message passing.** Every handoff between components should be a Pydantic model: `IncomingRequest → RoutingDecision → SpecialistResult → ApprovalRequest → WorkflowState`. Untyped dicts at boundaries hide bugs.

3. **Async throughout.** Approval workflows are async (day-6). The specialist agents should be async too. The orchestrator uses `asyncio.gather` or sequential `await` depending on whether parallelism is appropriate.

4. **Graceful degradation.** If `WorkflowEngine` raises `WorkflowTimeoutError`, the orchestrator catches it and routes to human escalation — it does not crash. Every external call has a fallback.

5. **Observability first.** Every routing decision, every approval event, every escalation is logged at `INFO`. The audit log is written to disk. At the end of a run, a summary is printable from the log alone — no need to re-run.

## Project structure

```
week-06-project/
├── src/
│   ├── config.py          # ThresholdSettings, WorkflowSettings (pydantic-settings)
│   ├── models.py           # All shared Pydantic models
│   ├── audit.py            # AuditLog, PersistentAuditLog
│   ├── confidence.py       # ConfidenceRouter (from day-5)
│   ├── workflow.py         # WorkflowEngine, ApprovalChain (from day-6)
│   ├── agents/
│   │   ├── base.py         # SpecialistAgent base class / protocol
│   │   ├── billing.py      # BillingAgent
│   │   ├── technical.py    # TechnicalAgent
│   │   └── general.py      # GeneralAgent
│   └── orchestrator.py     # Top-level HITLOrchestrator
├── tests/
│   ├── unit/
│   │   ├── test_confidence.py
│   │   ├── test_workflow.py
│   │   └── test_agents.py
│   └── integration/
│       └── test_orchestrator.py
├── .env.example
└── pyproject.toml
```

## Project checklist

Work through these in order. Check each off as you complete it.

### Setup
- [ ] Repository initialised, `pyproject.toml` with `anthropic`, `pydantic`, `pydantic-settings`, `pytest`, `pytest-asyncio`
- [ ] `.env.example` present with `ANTHROPIC_API_KEY=`, `CONFIDENCE_AUTO_THRESHOLD=`, `CONFIDENCE_ESCALATE_THRESHOLD=`
- [ ] `.env` in `.gitignore`
- [ ] `src/` package structure created with `__init__.py` files

### Models (`src/models.py`)
- [ ] `IncomingRequest` — `request_id`, `user_input`, `domain: str | None`
- [ ] `SpecialistResult` — `agent_name`, `answer`, `requires_approval`, `action_payload: dict | None`
- [ ] `EscalationRecord` — `request_id`, `reason`, `confidence_score: float | None`, `timestamp`
- [ ] `OrchestratorResponse` — `request_id`, `outcome: Literal["answered", "escalated", "rejected"]`, `answer: str | None`, `escalation: EscalationRecord | None`

### Config (`src/config.py`)
- [ ] `ThresholdSettings` from day-5 (or import it)
- [ ] `WorkflowSettings` — `timeout_seconds`, `max_retries`, `sensitive_action_types: list[str]`

### Confidence layer (`src/confidence.py`)
- [ ] `ConfidenceRouter` from day-5 integrated and importable
- [ ] Unit tests in `tests/unit/test_confidence.py` — all three routing branches covered with mocked LLM

### Workflow layer (`src/workflow.py`)
- [ ] `WorkflowEngine` from day-6 integrated and importable
- [ ] `AuditLog` / `PersistentAuditLog` in `src/audit.py`
- [ ] Unit tests in `tests/unit/test_workflow.py` — approved, rejected, expired paths covered

### Specialist agents (`src/agents/`)
- [ ] `SpecialistAgent` protocol: `name: str`, `async def handle(request: IncomingRequest) -> SpecialistResult`
- [ ] `BillingAgent` — handles billing queries; marks `requires_approval=True` for amounts > threshold
- [ ] `TechnicalAgent` — handles technical queries; marks `requires_approval=True` for destructive operations
- [ ] `GeneralAgent` — handles everything else; never requires approval
- [ ] Unit tests mock LLM calls and assert `requires_approval` flag is set correctly

### Orchestrator (`src/orchestrator.py`)
- [ ] `HITLOrchestrator.__init__` takes `ThresholdSettings`, `WorkflowSettings`, `AuditLog`, `dict[str, SpecialistAgent]`
- [ ] `async def handle(request: IncomingRequest) -> OrchestratorResponse` — full decision tree
- [ ] Low confidence → `EscalationRecord` with `reason="low_confidence"`
- [ ] Approval timeout → `EscalationRecord` with `reason="approval_timeout"`
- [ ] Approval rejected → `OrchestratorResponse(outcome="rejected")`
- [ ] All paths produce an audit entry

### Integration tests (`tests/integration/test_orchestrator.py`)
- [ ] Happy path: billing query, medium confidence, approval granted → `outcome="answered"`
- [ ] Low confidence path: general query, low confidence → `outcome="escalated"`
- [ ] Approval rejection path: billing query, high confidence, approval rejected → `outcome="rejected"`
- [ ] Timeout path: workflow times out → `outcome="escalated"` with correct reason

### Assessment prep
- [ ] Run `pytest -v` — all tests green
- [ ] Produce a `runs/audit.jsonl` by running the orchestrator against 3 test requests
- [ ] Review the audit log — does every state transition make sense?
- [ ] Read through your `orchestrator.py` — can you explain every line?

## Exercises

1. **Wire up the full system.** Implement `HITLOrchestrator` as described in the checklist. Start with `GeneralAgent` only (no approval needed), get one end-to-end test green, then add `BillingAgent` and the approval workflow.

2. **Add a fourth specialist.** Implement an `HRAgent` that handles HR-related queries (leave, payroll questions). It requires approval for any request that mentions compensation amounts. Register it in the orchestrator's specialist map. Write a unit test for the approval-trigger logic.

3. **Test the full HITL path.** Write an integration test for the scenario: billing request, amount > threshold, approval chain has one approver who rejects on first attempt but approves on retry. Assert the final `OrchestratorResponse.outcome == "answered"` and the audit log shows the rejection and retry entries.

4. **Observability report.** Add a `OrchestratorMetrics` dataclass that counts: `total_requests`, `auto_handled`, `escalated_to_human`, `approved`, `rejected`, `expired`. Populate it inside `HITLOrchestrator.handle()`. After running the demo, print a formatted metrics summary. Write a test that runs three mock requests and asserts the counts.

(Solutions: ../solutions/week-06/day-7.md)

## Self-check

**Routing & Confidence (day-5)**
- [ ] I can explain the three-level routing decision and when each fires
- [ ] I can set per-environment thresholds without touching code

**Approval Workflows (day-6)**
- [ ] I can draw the workflow state machine from memory
- [ ] I can implement a two-approver chain with retry and timeout

**Integration**
- [ ] I can describe the full request lifecycle — from `IncomingRequest` to `OrchestratorResponse` — at every branch
- [ ] I can explain how dependency injection keeps the components testable in isolation
- [ ] I can write integration tests that mock LLM and approver calls end-to-end

**Architecture**
- [ ] I can explain why typed Pydantic models at every boundary beat untyped dicts
- [ ] I can describe where the audit log lives and why it is shared rather than per-component
- [ ] I can add a new specialist agent in under 30 minutes: create the class, register it, write the tests
