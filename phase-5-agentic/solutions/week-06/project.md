# Solutions — Week 06 Project — RouteMaster

> These are reference solutions. Your implementation may differ and still be correct.

---

## Reference Implementation Outline

### `src/route_master/models.py`

All Pydantic v2 data models shared across the system. Contains:

- `UserMessage` — raw user input with session context
- `RoutingResult` — intent label, confidence score, raw text
- `SpecialistResponse` — answer text, specialist name, processing metadata
- `GateDecision` — `requires_approval: bool`, `hard_block: bool`, tier enum
- `WorkflowRequest` / `WorkflowState` — approval workflow lifecycle models
- `OrchestratorResponse` — final response returned to caller, includes `human_involved` flag and audit trail

All cross-boundary data goes through these models. No untyped dicts leave a function.

---

### `src/route_master/config.py`

`pydantic-settings` `Settings` class:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str = ""
    router_model: str = "claude-sonnet-5-5"
    specialist_model: str = "claude-sonnet-5-5"
    confidence_threshold: float = 0.60
    hitl_threshold: float = 0.40
    workflow_timeout_seconds: float = 86400.0

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
```

Single source of truth for all thresholds and model IDs.

---

### `src/route_master/router.py`

`IntentRouter` class:
- Takes an `anthropic.Anthropic` client (injected, never instantiated inside).
- `route(message: str) -> RoutingResult` — calls LLM with a classification system prompt, parses JSON response, normalises intent label.
- `_parse_response(raw: str) -> tuple[str, float]` — private parser, separated for unit testing without a live client.
- Validates confidence is in [0, 1]; defaults to 0.0 on parse failure and logs a warning.

---

### `src/route_master/specialists.py`

`Specialist` protocol + concrete `LLMSpecialist`:
- `SpecialistRegistry` maps intent strings to `Specialist` instances.
- `register(intent, specialist)` and `get(intent) -> Specialist | None`.
- Falls back to `"general"` specialist if intent not found.
- Each `LLMSpecialist` takes `name` and `system_prompt` at construction; shares the injected client.
- `answer(message: str) -> SpecialistResponse` — single method.

---

### `src/route_master/gates.py`

Stateless functions — no class needed:
- `classify_tier(confidence: float) -> ConfidenceTier` — maps float to tier enum.
- `evaluate_gate(intent: str, confidence: float, action_type: str | None) -> GateDecision` — combines tier and action sensitivity.
- `SENSITIVE_ACTIONS: frozenset[str]` — defined at module level for easy extension.

No side effects; pure functions make unit testing trivial.

---

### `src/route_master/workflow.py`

`WorkflowEngine` + `ExpiryManager`:
- `WorkflowEngine.submit(request, approver1_id, approver2_id) -> str` — creates chain, calls `notify_approver`, returns `request_id`.
- `WorkflowEngine.decide(request_id, approver_id, decision, comment) -> ApprovalState` — validates approver identity, advances state machine, calls `notify_approver` when stage transitions.
- `WorkflowEngine.get_stats() -> WorkflowStats` — counts by state, average approval time.
- `ExpiryManager` — `asyncio` background task; polls `_check_and_expire()` every `poll_interval` seconds.
- `notify_approver` stub lives in `notifications.py` — logs only, no real email.

---

### `src/route_master/orchestrator.py`

`Orchestrator` — top-level coordinator:

```
handle(message: str) -> OrchestratorResponse
    │
    ├─ router.route(message) → RoutingResult
    │
    ├─ evaluate_gate(intent, confidence, action_type) → GateDecision
    │       │
    │       ├─ hard_block=True → submit workflow, return PENDING response
    │       └─ soft gate → log, continue
    │
    ├─ registry.get(intent) → Specialist
    │
    └─ specialist.answer(message) → SpecialistResponse → OrchestratorResponse
```

Dependency-injected: takes `router`, `registry`, `workflow_engine`, `settings` at construction. Never instantiates Anthropic client directly.

---

## Key Architectural Decisions

### Decision 1 — Dependency injection for all external clients

All `anthropic.Anthropic` clients are injected via constructor rather than instantiated inside classes. This makes every component unit-testable by passing a `MagicMock` instead of a real client, with zero monkey-patching at the module level. The composition root in `main.py` is the only place where real clients are created.

### Decision 2 — Single shared `Settings` instance for all thresholds

Thresholds (`confidence_threshold`, `hitl_threshold`, `workflow_timeout_seconds`) defined once in `config.py` and consumed everywhere. Avoids the split-threshold bug (different values in router vs. gate) that commonly causes silent misrouting. Env-var override works via `pydantic-settings` without code changes.

### Decision 3 — State machine in `SequentialApprovalChain`, validation in `WorkflowEngine`

The chain owns state-transition logic; the engine owns identity validation (which approver can act). This separation means the chain is testable in isolation (no engine needed) and the engine's permission logic can be changed without touching the chain's state machine.

### Decision 4 — Pure functions in `gates.py`

`evaluate_gate` and `classify_tier` are stateless functions, not methods on a class. They have no side effects, take and return Pydantic models, and require no mocking in tests — just call them with a constructed `RoutingResult`. This lowers test friction and makes the gate logic easy to read and audit.

### Decision 5 — `ExpiryManager` as an opt-in background task, not woven into `WorkflowEngine`

Expiry is a cross-cutting concern. Embedding it inside `WorkflowEngine` would make unit tests of the engine require running an event loop. Separating it into `ExpiryManager` means the engine is synchronous and testable; expiry is activated by calling `manager.start()` in the app lifespan hook.

---

## Integration Test Suite

```python
# tests/integration/test_route_master.py
"""
Integration tests for RouteMaster.
All Anthropic API calls mocked via unittest.mock.patch.

Happy paths tested:
1. Billing query — high confidence, autonomous, no gate.
2. Low-confidence query — confidence gate fires, escalates.
3. Refund — sensitive action, HITL gate, approval workflow completes.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.route_master.config import Settings
from src.route_master.gates import evaluate_gate
from src.route_master.models import RoutingResult
from src.route_master.orchestrator import Orchestrator
from src.route_master.router import IntentRouter
from src.route_master.specialists import LLMSpecialist, SpecialistRegistry
from src.route_master.workflow import WorkflowEngine
from src.workflow.sequential_approval import ApprovalRequest, ApprovalState


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _block(text: str) -> MagicMock:
    b = MagicMock()
    b.text = text
    return b


def _msg(*texts: str) -> list[MagicMock]:
    """Build a side_effect list for client.messages.create."""
    results = []
    for text in texts:
        m = MagicMock()
        m.content = [_block(text)]
        results.append(m)
    return results


def _make_client(*texts: str) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = _msg(*texts)
    return client


def _make_orchestrator(client: MagicMock) -> Orchestrator:
    settings = Settings(
        anthropic_api_key="test-key",
        confidence_threshold=0.60,
        hitl_threshold=0.40,
    )
    router = IntentRouter(client=client)
    registry = SpecialistRegistry()
    for intent in ("billing", "technical", "refund", "general"):
        registry.register(intent, LLMSpecialist(name=intent, client=client))
    engine = WorkflowEngine()
    return Orchestrator(
        router=router,
        registry=registry,
        workflow_engine=engine,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Test 1 — Billing query, high confidence, handled autonomously
# ---------------------------------------------------------------------------

class TestBillingQuery:
    def test_happy_path(self) -> None:
        client = _make_client(
            json.dumps({"intent": "billing", "confidence": 0.92}),
            "Your invoice resets on the 1st of each month.",
        )
        orchestrator = _make_orchestrator(client)

        with patch("src.route_master.workflow.notify_approver"):
            result = orchestrator.handle("When does my billing reset?")

        assert result.intent == "billing"
        assert result.confidence == pytest.approx(0.92)
        assert not result.human_involved
        assert result.answer == "Your invoice resets on the 1st of each month."

    def test_specialist_called_with_original_message(self) -> None:
        client = _make_client(
            json.dumps({"intent": "billing", "confidence": 0.85}),
            "Billing answer.",
        )
        orchestrator = _make_orchestrator(client)

        with patch("src.route_master.workflow.notify_approver"):
            orchestrator.handle("Show me my last invoice")

        # Second call to create should carry the user's message
        second_call = client.messages.create.call_args_list[1]
        assert "Show me my last invoice" in str(second_call)


# ---------------------------------------------------------------------------
# Test 2 — Low confidence, escalation
# ---------------------------------------------------------------------------

class TestLowConfidenceEscalation:
    def test_below_threshold_triggers_human_flag(self) -> None:
        client = _make_client(
            json.dumps({"intent": "general", "confidence": 0.30}),
        )
        orchestrator = _make_orchestrator(client)

        with patch("src.route_master.workflow.notify_approver"):
            result = orchestrator.handle("I have some kind of problem maybe")

        assert result.human_involved is True
        assert "human" in result.answer.lower() or "review" in result.answer.lower()

    def test_specialist_not_called_on_escalation(self) -> None:
        client = _make_client(
            json.dumps({"intent": "general", "confidence": 0.25}),
        )
        orchestrator = _make_orchestrator(client)

        with patch("src.route_master.workflow.notify_approver"):
            orchestrator.handle("???")

        # Only one LLM call: the router. Specialist never called.
        assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Test 3 — Refund with approval workflow
# ---------------------------------------------------------------------------

class TestRefundApprovalWorkflow:
    def test_refund_triggers_hitl_gate(self) -> None:
        client = _make_client(
            json.dumps({"intent": "refund", "confidence": 0.75}),
            "Refund processed.",
        )
        orchestrator = _make_orchestrator(client)

        with patch("src.route_master.workflow.notify_approver"):
            result = orchestrator.handle("I want a refund for order #42")

        gate = evaluate_gate("refund", 0.75, action_type="refund")
        assert gate.requires_approval is True
        assert result.human_involved is True

    def test_approval_workflow_reaches_approved_state(self) -> None:
        engine = WorkflowEngine()
        req = ApprovalRequest(
            description="Refund for order #42",
            requester="user@example.com",
        )
        with patch("src.route_master.workflow.notify_approver"):
            rid = engine.submit(req, approver1_id="mgr1", approver2_id="mgr2")

        with patch("src.route_master.workflow.notify_approver"):
            engine.decide(rid, "mgr1", "approve", comment="Looks valid")
            final = engine.decide(rid, "mgr2", "approve", comment="Approved")

        assert final == ApprovalState.APPROVED

    def test_refund_rejected_by_approver1_terminates(self) -> None:
        engine = WorkflowEngine()
        req = ApprovalRequest(description="Suspicious refund", requester="x@example.com")

        with patch("src.route_master.workflow.notify_approver"):
            rid = engine.submit(req, approver1_id="mgr1", approver2_id="mgr2")
            final = engine.decide(rid, "mgr1", "reject", comment="Policy violation")

        assert final == ApprovalState.REJECTED
```

---

## Common Pitfalls and Solutions

### Pitfall 1 — Model IDs pointing to retired `claude-3-*` models

**Symptom:** `anthropic.BadRequestError: model not found` or unexpected costs from wrong model tier.

**Solution:** Always use `claude-sonnet-5-5` or `claude-opus-5-5`. Store model IDs in `config.py` — never inline them. Add a startup assertion:
```python
assert settings.router_model.startswith("claude-"), "Invalid model ID"
```

---

### Pitfall 2 — Confidence threshold inconsistency across components

**Symptom:** Router escalates at 0.60, gate checks at 0.50 — requests with confidence 0.55 pass the orchestrator but the gate flags them, causing duplicate HITL triggers.

**Solution:** Both the orchestrator and the gate consume `settings.confidence_threshold`. The gate uses `settings.hitl_threshold` for its own tier cut-off. A single `Settings` instance prevents drift.

---

### Pitfall 3 — Approval workflow `notify_approver` not called on stage transitions

**Symptom:** Approver 2 never receives the request; workflow sits in `PENDING_APPROVER2` indefinitely.

**Solution:** `WorkflowEngine.decide()` calls `notify_approver(chain.approver2_id, request)` after a successful Approver 1 approval. Covered by the test `test_approver1_approve_notifies_approver2`.

---

### Pitfall 4 — `ExpiryManager` starts before the event loop is running

**Symptom:** `RuntimeError: no running event loop` on application startup.

**Solution:** Call `manager.start()` inside a FastAPI `lifespan` context manager (not at module import time), which runs inside an active asyncio event loop:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    expiry_manager.start()
    yield
    expiry_manager.stop()
```

---

### Pitfall 5 — Integration tests hitting real Anthropic API

**Symptom:** Tests are slow, flaky (rate limits), and expensive. CI fails when `ANTHROPIC_API_KEY` is not set.

**Solution:** Every test that exercises any code path touching `anthropic.Anthropic.messages.create` must mock it. Use `_make_client(*texts)` helper pattern shown above — build `MagicMock` objects with `.content[0].text` attribute, set as `side_effect` list so each call consumes the next response. Add a CI lint rule: grep for `anthropic.Anthropic()` without a mock in the test file.

---

## Grading Notes per Rubric Dimension

| Dimension | Weight | What to look for |
|---|---|---|
| Independent implementation | 40% | Original code — not copy-pasted from exercises; components correctly wired together; `Orchestrator` delegates to the right component at the right time |
| Conceptual understanding | 25% | Architecture decisions are justified (not just described); student can explain why dependency injection is used, why the state machine lives in the chain not the engine, why thresholds are centralised |
| Debugging | 20% | Student can identify and fix: split threshold bug, wrong approver acting, expiry manager not started, mock returning dict instead of object |
| Code quality | 15% | `ruff`-clean; type hints on every function; Pydantic v2 syntax; `logging` not `print`; `os.getenv` / `pydantic-settings` for secrets; no files > 300 lines |

**Automatic zero for:**
- Hardcoded `ANTHROPIC_API_KEY` in source code.
- Using `claude-3-*` model IDs.
- No tests (or tests that call the real API without mocking).

**Common partial-credit deductions:**
- Missing type hints: -2 pts per untyped public function.
- `print()` instead of `logging`: -3 pts per occurrence.
- Pydantic v1 `@validator` syntax: -3 pts (shows copy-paste from old examples).
- `WorkflowEngine` and `Orchestrator` in a single file > 300 lines: -5 pts (separation of concerns).
- No test for the low-confidence escalation path: -8 pts.
