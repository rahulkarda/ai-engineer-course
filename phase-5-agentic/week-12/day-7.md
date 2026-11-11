# Phase 5 · Week 12 · Day 7 — Build the Enterprise Agent Orchestration Framework (Full Capstone Build)

> Objective: Build the complete FinOps Agent Orchestration Framework end-to-end.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]], [[day-5]], [[day-6]]

## Concept

Capstone integration day. You build everything from day-1 through day-6 into a working, testable, Dockerized framework. This is portfolio-grade output: clean Python API, multi-agent orchestration, MCP tool server, A2A messaging, HITL gates, audit logging, evaluation suite, and observability — all wired together.

The objective is not perfection but integration: every subsystem must connect. You finish with a runnable `docker compose up` and a passing `pytest` suite.

Each subsystem was built and tested in isolation across days 1–6. Today the goal is to verify that they compose correctly: that `ToolRegistry` feeds `ReActPlanner`, which feeds `MultiAgentCoordinator`, which the `SupervisorAgent` orchestrates, all behind the FastAPI layer with audit and metrics attached at every seam.

## How it works internally

The full request lifecycle:

```
User → POST /agent/run
  → sanitize_input()
  → audit.log_event("agent_request")
  → metrics.start_span("api.agent_run")
  → router.route(task) → agent_name
  → coordinator.dispatch([SubTask(agent_name, task)])
    → agent_fn(payload)
      → planner.plan(session_id, payload)
        → [LLM call → parse step → tool call → memory update] × N
        → for critical tools: gate.request_approval() → PENDING
  → audit.log_event("agent_response")
  → return AgentResponse
```

Key integration contracts:

- `ToolRegistry` → `ReActPlanner`: planner calls `registry.get(tool_name)` to resolve tool callables and their schemas
- `HumanApprovalGate` → `ToolRegistry`: gate checks `ToolSchema.risk_level` before allowing execution
- `AuditLogger` → all subsystems: every subsystem receives an audit logger instance, not a module-level logger
- `MetricsCollector` → `ReActPlanner` + API layer: spans nest as `api.agent_run` → `planner.run` → `tool.call`
- `SQLiteMemoryStore` → `ReActPlanner`: planner reads previous observations for context, writes new ones after each step

## Worked example

Integration wiring — the `build_framework()` function connects all subsystems:

```python
# src/finops_agent/api/full_integration.py
"""
Integration wiring: connects coordinator, supervisor, planner, router, and gate.
This module is the 'main brain' — it imports every subsystem and shows the wiring.
"""
import logging
import os
import anthropic

from finops_agent.config import AgentSettings
from finops_agent.core.registry import ToolRegistry, ToolSchema
from finops_agent.core.planner import ReActPlanner
from finops_agent.memory.store import SQLiteMemoryStore
from finops_agent.routing.router import AgentRouter
from finops_agent.hitl.gate import HumanApprovalGate
from finops_agent.audit.logger import AuditLogger
from finops_agent.orchestration.coordinator import MultiAgentCoordinator, SubTask
from finops_agent.orchestration.supervisor import SupervisorAgent
from finops_agent.observability.metrics import MetricsCollector
from finops_agent.tools.finance import retrieve_invoice, validate_payment, check_budget
from finops_agent.tools.controlled import approve_payment
from finops_agent.evaluation.harness import EvalHarness

logger = logging.getLogger(__name__)


def build_framework():
    settings = AgentSettings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    audit = AuditLogger()
    gate = HumanApprovalGate(auto_approve_levels={"low", "medium"})
    memory = SQLiteMemoryStore()
    metrics = MetricsCollector()
    registry = ToolRegistry()

    registry.register(ToolSchema(name="retrieve_invoice", description="Fetch invoice", parameters={"invoice_id": "str"}, risk_level="low"), retrieve_invoice)
    registry.register(ToolSchema(name="validate_payment", description="Validate payment", parameters={"invoice_id": "str", "amount": "float"}, risk_level="low"), validate_payment)
    registry.register(ToolSchema(name="check_budget", description="Check budget", parameters={"department": "str", "amount": "float"}, risk_level="medium"), check_budget)

    planner = ReActPlanner(client=client, model=settings.primary_model, registry=registry, memory=memory)
    router = AgentRouter()

    def finance_agent(payload: str) -> str:
        steps = planner.plan(session_id="finance", task=payload)
        final = next((s.final_answer for s in steps if s.is_final), "No final answer")
        return final or "Completed"

    def ops_agent(payload: str) -> str:
        return f"Ops processed: {payload}"

    coordinator = MultiAgentCoordinator(agents={"finance_agent": finance_agent, "ops_agent": ops_agent})
    supervisor = SupervisorAgent(client=client, coordinator=coordinator, model=settings.planning_model)

    return {
        "settings": settings,
        "client": client,
        "audit": audit,
        "gate": gate,
        "memory": memory,
        "metrics": metrics,
        "registry": registry,
        "planner": planner,
        "router": router,
        "coordinator": coordinator,
        "supervisor": supervisor,
    }
```

Integration tests covering the full stack with mocked LLM:

```python
# tests/integration/test_full_stack.py
import pytest
from unittest.mock import MagicMock, patch

def test_full_stack_invoice_retrieval():
    """Full stack: route → finance agent → retrieve_invoice tool."""
    from finops_agent.routing.router import AgentRouter
    from finops_agent.tools.finance import retrieve_invoice

    router = AgentRouter()
    decision = router.route("retrieve invoice INV-001 and validate")
    assert decision.agent_name == "finance_agent"

    result = retrieve_invoice("INV-001")
    assert result.invoice_id == "INV-001"
    assert result.amount == 15000.0


def test_full_stack_hitl_gate():
    """Approval gate blocks critical actions."""
    from finops_agent.hitl.gate import HumanApprovalGate, ApprovalRequest, ApprovalStatus

    gate = HumanApprovalGate(auto_approve_levels={"low"})
    req = ApprovalRequest(session_id="s1", tool_name="approve_payment",
                          parameters={"invoice_id": "INV-001", "amount": 15000.0},
                          risk_level="critical", reason="test")
    result = gate.request_approval(req)
    assert result.status == ApprovalStatus.PENDING
    assert len(gate.get_pending()) == 1

    resolved = gate.resolve(req.request_id, approved=True, reviewer="admin")
    assert resolved.status == ApprovalStatus.APPROVED
    assert len(gate.get_pending()) == 0


def test_full_stack_audit_log(tmp_path):
    """Audit logger writes and reads events correctly."""
    from finops_agent.audit.logger import AuditLogger

    audit = AuditLogger(log_path=str(tmp_path / "audit.jsonl"))
    audit.log_event("test_event", session_id="s1", tool_name="retrieve_invoice")
    events = audit.read_events("test_event")
    assert len(events) == 1
    assert events[0].session_id == "s1"


def test_eval_harness_mock():
    """Eval harness runs cases and computes pass_rate."""
    import json
    import tempfile
    import os
    from finops_agent.evaluation.harness import EvalHarness

    def mock_agent(task: str):
        return (["retrieve_invoice"], "INV-001 retrieved successfully")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"case_id": "t1", "input": "retrieve INV-001", "expected_tool_calls": ["retrieve_invoice"], "expected_answer_contains": ["INV-001"]}) + "\n")
        tmp_path = f.name

    harness = EvalHarness(agent_fn=mock_agent, dataset_path=tmp_path)
    results = harness.run_all()
    report = harness.report(results)
    os.unlink(tmp_path)
    assert report["pass_rate"] == 1.0
```

## Exercises

Build tasks — 4 hours to build the complete framework:

**Hour 1 — Scaffold + core (day-1 material)**

1. Create the full project directory structure: `src/finops_agent/` with subdirectories `api/`, `core/`, `memory/`, `routing/`, `hitl/`, `audit/`, `orchestration/`, `observability/`, `tools/`, `evaluation/`. Add `__init__.py` to each.
2. Implement `AgentSettings` using `pydantic-settings` with fields: `anthropic_api_key`, `primary_model` (default `"claude-sonnet-5-5"`), `planning_model` (default `"claude-opus-5-5"`). Values must come from environment variables only.
3. Implement `ToolRegistry` with `register(schema, fn)` and `get(name)` methods. Implement `ToolSchema` as a Pydantic v2 model.
4. Write unit tests for `ToolRegistry.register` (registers correctly), `ToolRegistry.get` (raises on missing tool), and `AgentSettings` loading from environment variables using `monkeypatch`.

**Hour 2 — Tools + memory + routing (day-2 material)**

5. Implement `SQLiteMemoryStore` with `write(session_id, key, value)` and `read(session_id, key)` methods backed by SQLite.
6. Implement `ReActPlanner` that accepts a mocked LLM callable in tests. The planner must call tools via `ToolRegistry` and write observations to `SQLiteMemoryStore`.
7. Implement `AgentRouter` with a `route(task: str) -> RoutingDecision` method. Route to `"finance_agent"` for tasks containing invoice/payment/budget keywords, `"ops_agent"` otherwise.
8. Implement the three finance tools: `retrieve_invoice(invoice_id)`, `validate_payment(invoice_id, amount)`, `check_budget(department, amount)`. Return typed Pydantic models.
9. Write unit tests for `AgentRouter` (keyword routing), `SQLiteMemoryStore` (read/write/missing key), and each finance tool.

**Hour 3 — Multi-agent + HITL + audit (days 3–4)**

10. Implement `MultiAgentCoordinator` with a `dispatch(subtasks: list[SubTask]) -> list[str]` method that runs each subtask's agent function and returns results.
11. Implement `SupervisorAgent` that uses a mocked LLM call in tests to decompose a task into subtasks, dispatches via coordinator, and returns a combined result.
12. Implement `HumanApprovalGate` with `request_approval`, `resolve`, and `get_pending` methods. Critical-risk tools must always return `PENDING` status.
13. Implement `AuditLogger` that appends structured JSONL events with timestamp, event type, session ID, and metadata. Include a `read_events(event_type)` method.
14. Implement `controlled.approve_payment` as a tool with `risk_level="critical"`. It must pass through `HumanApprovalGate` before executing.
15. Write tests for coordinator dispatch, approval gate (pending/resolve cycle), and audit log read/write.

**Hour 4 — Eval + observability + Docker (days 5–6)**

16. Implement `EvalHarness` as shown in day-5. Write `eval/dataset.jsonl` with at least 6 cases covering finance retrieval, approval, budget check, error paths.
17. Implement `MetricsCollector` with `increment`, `start_span`, `get_counters`, and `get_trace` methods.
18. Wire the full `main.py` FastAPI app with lifespan, all endpoints (`/agent/run`, `/health`, `/metrics`, `/hitl/pending`, `/hitl/resolve/{id}`), and observability on every request.
19. Write the `Dockerfile` (multi-stage) and `docker-compose.yml`. Run `docker build .` and verify it exits 0.
20. Run the full pytest suite with `pytest tests/` — all tests must pass before submission.

## Self-check

- [ ] All modules import without error (`python -c "from finops_agent.api.full_integration import build_framework"`)
- [ ] `pytest tests/` passes with no failures
- [ ] `docker build .` completes without error
- [ ] HITL gate blocks all critical-risk tool calls
- [ ] Audit log contains entries for every agent run
- [ ] Eval harness reports `pass_rate` from a 6-case dataset
