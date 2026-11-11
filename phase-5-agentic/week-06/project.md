# Project: RouteMaster — Intelligent Routing & Human-in-the-Loop Platform

## Overview

RouteMaster is a production-grade agentic platform that classifies incoming user requests by intent, dynamically selects the appropriate specialist handler, and enforces human-in-the-loop approval gates for high-risk actions. The system applies confidence thresholds to reject or escalate low-certainty classifications before any downstream processing occurs. A persistent approval-queue component decouples the agent workflow from human review timing, so the pipeline can handle concurrent requests without blocking. The project integrates all five Week 6 concepts — router agents, delegation patterns, orchestrator-worker architecture, dynamic tool selection, and HITL — into a single deployable service.

---

## Architecture Diagram

```
                          ┌─────────────────────────────────────┐
                          │            RouteMaster               │
                          │                                      │
  User Request ──────────►│  ┌──────────────────────────────┐   │
                          │  │     IntentClassifier          │   │
                          │  │  (LLM-based, returns intent   │   │
                          │  │   + confidence score)         │   │
                          │  └──────────────┬───────────────┘   │
                          │                 │                    │
                          │         confidence < 0.6?            │
                          │           ┌─────┴─────┐             │
                          │         YES           NO             │
                          │           │            │             │
                          │      ┌────▼────┐  ┌───▼──────────┐  │
                          │      │ Reject/ │  │   Router     │  │
                          │      │Escalate │  │ (selects     │  │
                          │      └─────────┘  │  handler)    │  │
                          │                   └───┬──────────┘  │
                          │          ┌────────────┼────────────┐ │
                          │          │            │            │ │
                          │   ┌──────▼──┐  ┌─────▼──┐  ┌─────▼─┐│
                          │   │ Billing │  │ Tech   │  │General││
                          │   │ Handler │  │ Handler│  │Handler││
                          │   └──────┬──┘  └─────┬──┘  └─────┬─┘│
                          │          └────────────┼────────────┘ │
                          │                       │              │
                          │               ┌───────▼──────────┐  │
                          │               │   ApprovalGate   │  │
                          │               │ requires_approval?│  │
                          │               └───────┬──────────┘  │
                          │                  ┌────┴────┐        │
                          │                YES        NO        │
                          │                  │         │        │
                          │           ┌──────▼──┐  ┌───▼──────┐ │
                          │           │Approval │  │ Respond  │ │
                          │           │  Queue  │  │  to User │ │
                          │           │(async)  │  └──────────┘ │
                          │           └──────┬──┘               │
                          │                  │ Human approves   │
                          │           ┌──────▼──────┐           │
                          │           │  Execute &  │           │
                          │           │  Respond    │           │
                          │           └─────────────┘           │
                          └─────────────────────────────────────┘
```

---

## Required Components

### 1. `IntentClassifier`
- Class accepting `query: str`
- Returns `IntentResult(intent, confidence, raw_scores)` — Pydantic v2 model
- `intent: Literal["billing", "technical", "general", "unknown"]`
- `confidence: float` validated in range `[0.0, 1.0]`
- `raw_scores: dict[str, float]` — per-class probability map
- Must be injectable (accept a callable LLM backend) so tests can mock without real API calls
- Log classification results at `DEBUG` level; never log query content at `INFO` or above (PII)

### 2. `Router`
- Pure function: `route(result: IntentResult) -> RouteDecision`
- `RouteDecision(handler_name: str, rejected: bool, rejection_reason: str | None)`
- Rejects (`rejected=True`) when `confidence < 0.6`
- Handler name must map to a registered handler in the `HandlerRegistry`
- Raise `UnknownIntentError` (custom exception) for intents not in the registry

### 3. `HandlerRegistry`
- A Pydantic v2 model or typed dict mapping intent string to handler callable
- Supports runtime registration: `registry.register(intent, handler_fn)`
- Supports listing registered intents: `registry.list_intents() -> list[str]`

### 4. Specialist Handlers
Three handlers, each typed `(query: str, context: RequestContext) -> HandlerResponse`:
- `billing_handler` — handles payment, invoice, refund queries; sets `requires_approval=True` for any query containing `"refund"` or `"chargeback"`
- `technical_handler` — handles bug reports, how-to, configuration queries; `requires_approval` always `False`
- `general_handler` — fallback for all other queries; `requires_approval` always `False`

`HandlerResponse` Pydantic model fields:
- `handler: str`
- `response: str`
- `requires_approval: bool`
- `action_type: str | None` — e.g. `"refund"`, `"password_reset"`, `None`
- `metadata: dict[str, Any]`

### 5. `ApprovalGate`
- Class with an async-compatible in-memory queue (`asyncio.Queue` or `collections.deque`)
- `check(response: HandlerResponse) -> ApprovalStatus` — synchronous gate check
- `approve(request_id: str) -> ApprovalStatus` — marks pending request as approved
- `reject(request_id: str, reason: str) -> ApprovalStatus` — marks as rejected
- `list_pending() -> list[PendingApproval]` — exposes queue contents for a human-review UI
- `ApprovalStatus` Pydantic model: `approved: bool`, `reason: str`, `request_id: str`, `timestamp: datetime`

### 6. `Pipeline`
- `run_pipeline(query: str, context: RequestContext) -> PipelineResult` — top-level orchestrator
- `PipelineResult` Pydantic model: `query`, `request_id`, `intent`, `confidence`, `routed_to`, `handler_response`, `approval_status`, `elapsed_ms`
- Must use `logging` (not `print`); structured log entries at each stage
- `RequestContext` Pydantic model: `user_id: str`, `session_id: str`, `timestamp: datetime`, `metadata: dict[str, Any]`

### 7. Configuration (`config.py`)
- `pydantic-settings` `Settings` class
- Fields: `confidence_threshold: float = 0.6`, `approval_required_actions: list[str]`, `log_level: str = "INFO"`
- Load from `.env`; provide `.env.example` with placeholder values only

### 8. Tests (`tests/`)
- Minimum 15 unit tests; all external calls (LLM, HTTP) mocked with `unittest.mock` or `pytest-mock`
- At minimum cover: classifier output, routing logic for all intents + reject path, each handler, approval gate transitions (pending → approved, pending → rejected), full pipeline happy path, full pipeline reject path

---

## Acceptance Criteria

1. `run_pipeline` returns a fully populated `PipelineResult` with no `None` fields for a valid high-confidence query routed to any of the three handlers.
2. Any query where the classifier returns `confidence < 0.6` results in `PipelineResult.routed_to == "reject"` and no handler is invoked.
3. A billing query containing the word `"refund"` results in `ApprovalStatus.approved == False` and `reason == "human_review_required"` on the first pipeline run.
4. Calling `gate.approve(request_id)` for a pending refund request and re-running returns `ApprovalStatus.approved == True`.
5. `HandlerRegistry.register` successfully adds a new handler at runtime; `route()` dispatches to it on the next call with matching intent.
6. `ApprovalGate.list_pending()` returns exactly the requests that have not yet been approved or rejected.
7. All configuration values load from environment variables via `pydantic-settings`; the service starts without any hardcoded credentials or API keys.
8. All 15+ unit tests pass with `pytest` using only mocked LLM backends (zero real API calls in the test suite).
9. `ruff check src/ tests/` reports zero violations; all public functions and classes have type annotations.
10. The `README.md` includes a local-run quickstart (venv setup, `.env.example` copy, `pytest`, `uvicorn` start command if applicable) that a new contributor can follow without prior context.

---

## Suggested File Structure

```
route-master/
├── src/
│   └── route_master/
│       ├── __init__.py
│       ├── main.py                  # entry point / FastAPI app (optional)
│       ├── config.py                # pydantic-settings Settings
│       ├── models/
│       │   ├── __init__.py
│       │   ├── intent.py            # IntentResult, RouteDecision
│       │   ├── handler.py           # HandlerResponse, RequestContext
│       │   ├── approval.py          # ApprovalStatus, PendingApproval
│       │   └── pipeline.py          # PipelineResult
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── classifier.py        # IntentClassifier
│       │   ├── router.py            # Router, HandlerRegistry
│       │   └── pipeline.py          # run_pipeline orchestrator
│       ├── handlers/
│       │   ├── __init__.py
│       │   ├── billing.py           # billing_handler
│       │   ├── technical.py         # technical_handler
│       │   └── general.py           # general_handler
│       ├── gates/
│       │   ├── __init__.py
│       │   └── approval.py          # ApprovalGate
│       └── prompts/
│           ├── __init__.py
│           └── classification.py    # prompt templates for IntentClassifier
├── tests/
│   ├── conftest.py                  # shared fixtures, mock LLM factory
│   ├── unit/
│   │   ├── test_classifier.py
│   │   ├── test_router.py
│   │   ├── test_handlers.py
│   │   ├── test_approval_gate.py
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_pipeline_e2e.py     # mocked end-to-end, no real API
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Grading Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

---

## Stretch Goals

1. **Async approval queue with webhook notification** — Replace the in-memory queue with an async implementation that calls a configurable webhook URL (e.g., Slack or email) when a new item enters the pending state. Include a mock webhook target in the test suite.

2. **Multi-level confidence tiers** — Extend the router to support three tiers: `high (≥0.85)` routes directly, `medium (0.60–0.84)` routes but logs a warning and adds a `low_confidence` flag to `PipelineResult`, `low (<0.60)` rejects. Add a `ConfidenceTierPolicy` Pydantic model to make thresholds configurable without code changes.

3. **FastAPI admin endpoint** — Expose a `/admin/approvals` REST API with `GET /pending`, `POST /approve/{request_id}`, and `POST /reject/{request_id}` endpoints protected by HTTP Bearer token auth (token loaded from env, never hardcoded). Include OpenAPI schema and at least three `httpx`-based integration tests.

---

Answer key: ../solutions/week-06/project.md
