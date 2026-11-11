# Week 12 Assessment — Answer Key

## Conceptual answers

**1. Role of ToolSchema.risk_level:**
`risk_level` is set when a tool is registered in `ToolRegistry`. When the agent dispatches a tool call, it checks the schema's `risk_level`. If the level is `"high"` or `"critical"`, `HumanApprovalGate.request_approval()` is called instead of executing the tool directly. The gate creates a pending `ApprovalRequest`. The `AuditLogger` records both the request creation event and its resolution. This creates a traceable chain: tool registration → gate decision → audit record.

**2. Session memory vs. long-term memory:**
Session memory (e.g., `SQLiteMemoryStore` scoped by `session_id`) holds the context of a single conversation: recent steps, tool results, current task. It is fast and ephemeral. Long-term memory persists across sessions — it might store summarized past tasks, user preferences, or learned domain facts. Use session memory to maintain coherence within a single agent run; use long-term memory to give the agent a persistent identity and reduce repeated lookups across runs.

**3. Full request lifecycle:**
`POST /agent/run` → `sanitize_input()` (security) → `audit.log_event("agent_request")` → `metrics.start_span()` → `AgentRouter.route(task)` → route decision → `MultiAgentCoordinator.dispatch([SubTask])` → sub-agent function called → `ReActPlanner.plan()` → [LLM call → parse step → tool dispatch → `ToolRegistry.get()` → check `risk_level` → optional `HumanApprovalGate.request_approval()` → tool function executed → observation appended → `SQLiteMemoryStore.save_session()`] × N steps → final answer → `AuditLogger.log_event("agent_response")` → `MetricsCollector.increment()` → `AgentResponse` returned.

**4. Pydantic vs. dicts at seams:**
Raw dicts have no schema validation, no type checking, and silently accept wrong keys. Pydantic models reject invalid input at construction time, provide IDE autocompletion, and document the contract between modules. Concrete example: `ApprovalRequest` enforces that `risk_level` is a string and `parameters` is a dict. If a caller passes `risk_level=5` (wrong type), Pydantic raises `ValidationError` immediately rather than propagating a silent bug downstream.

**5. MCP vs. A2A:**
MCP (Model Context Protocol) is a tool-exposure protocol: it exposes callable functions as JSON-RPC-style endpoints that external clients (including LLMs) can discover and invoke. Use MCP when you want tools to be callable from outside your framework. A2A (Agent-to-Agent) is a message-passing protocol: structured envelopes (sender, recipient, payload, correlation_id) that decouple agents from each other. Use A2A for asynchronous coordination between agents in a multi-agent system. MCP is for tool invocation; A2A is for agent coordination.

**6. Eval harness vs. unit tests:**
Agent outputs are non-deterministic (LLM sampling) and multi-step (the correct behavior spans a chain of tool calls). Unit tests check exact code paths with deterministic inputs. The eval harness checks structural properties — "was `retrieve_invoice` called?", "does the final answer contain 'INV-001'?" — against a labeled dataset. Exact string match fails on paraphrases; property checks pass on equivalent correct answers. Eval also measures latency, pass rates across a dataset, and detects regressions across model versions.

**7. Two security controls:**
(a) Input sanitization (`sanitize_input`): strips credential patterns (API keys, bearer tokens) from user input before LLM submission. Mitigates prompt injection attacks where a user embeds a real credential hoping it will be logged or echoed. (b) HITL gate for critical tools: any tool with `risk_level="critical"` (e.g., `approve_payment`) is blocked until a human explicitly approves it. Mitigates autonomous agent financial fraud or accidental large payments without oversight.

**8. FastAPI lifespan vs. global state:**
Global singletons (`db = SQLiteMemoryStore()` at module level) are initialized at import time, cannot be swapped for test doubles, and create hidden coupling. Lifespan creates all instances once at startup and attaches them to `app.state`. Tests can create a fresh `app` instance with different `app.state` values, or override lifespan entirely. This makes the API fully testable in isolation without touching production databases or sending real API calls.

## Coding challenge — ReflectionAgent

```python
# src/finops_agent/core/reflection.py
import logging
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class ReflectionReport(BaseModel):
    task: str
    total_steps: int
    tool_calls: list[str]
    success: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    confidence: float

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def reflect(task: str, steps: list) -> ReflectionReport:
    """Analyze a completed planner run and return a structured improvement report.
    Pure Python — no LLM calls.
    """
    tool_calls = [s.action for s in steps if s.action is not None]
    success = any(s.is_final for s in steps)
    issues: list[str] = []
    suggestions: list[str] = []

    if not steps:
        issues.append("No steps executed")
        suggestions.append("Verify planner received a valid task and LLM is reachable")
    elif not success:
        issues.append("No final answer reached")
        suggestions.append("Increase max_steps or improve task decomposition prompt")

    # Check for steps that had no observation (tool call failed silently)
    failed_tool_steps = [s for s in steps if s.action and s.observation is None and not s.is_final]
    if failed_tool_steps:
        issues.append(f"Tool call(s) with missing observation: {[s.action for s in failed_tool_steps]}")
        suggestions.append("Review tool error handling and ensure all tool calls produce observations")

    observations_present = sum(1 for s in steps if s.observation is not None)
    confidence = observations_present / max(1, len(steps))

    logger.info("Reflection: task=%r success=%s issues=%d confidence=%.2f", task, success, len(issues), confidence)

    return ReflectionReport(
        task=task,
        total_steps=len(steps),
        tool_calls=tool_calls,
        success=success,
        issues=issues,
        suggestions=suggestions if issues else [],
        confidence=confidence,
    )
```

Tests:
```python
# tests/unit/test_reflection.py
import pytest
from finops_agent.core.planner import PlanStep
from finops_agent.core.reflection import reflect, ReflectionReport

def make_step(action=None, observation=None, is_final=False, final_answer=None):
    return PlanStep(thought="t", action=action, observation=observation, is_final=is_final, final_answer=final_answer)

def test_reflect_all_success():
    steps = [
        make_step(action="retrieve_invoice", observation="INV-001 found"),
        make_step(is_final=True, final_answer="Invoice is valid"),
    ]
    report = reflect("check invoice", steps)
    assert report.success is True
    assert report.issues == []
    assert report.suggestions == []
    assert 0.0 <= report.confidence <= 1.0

def test_reflect_no_final_answer():
    steps = [make_step(action="retrieve_invoice", observation="result"), make_step(action="validate_payment", observation="result")]
    report = reflect("validate payment", steps)
    assert report.success is False
    assert any("final answer" in i.lower() for i in report.issues)
    assert len(report.suggestions) >= 1

def test_reflect_empty_steps():
    report = reflect("empty task", [])
    assert report.total_steps == 0
    assert report.success is False
    assert any("no steps" in i.lower() for i in report.issues)
    assert report.confidence == 0.0

def test_reflect_steps_with_errors():
    steps = [
        make_step(action="retrieve_invoice", observation=None),  # tool called, no observation
        make_step(action="validate_payment", observation=None),
    ]
    report = reflect("validate", steps)
    assert any("missing observation" in i.lower() for i in report.issues)
    assert len(report.suggestions) >= 1

def test_reflect_confidence_clamped():
    # Force confidence >1 scenario by overriding — validator clamps it
    r = ReflectionReport(task="t", total_steps=0, tool_calls=[], success=False, confidence=1.5)
    assert r.confidence == 1.0
    r2 = ReflectionReport(task="t", total_steps=0, tool_calls=[], success=False, confidence=-0.5)
    assert r2.confidence == 0.0
```
