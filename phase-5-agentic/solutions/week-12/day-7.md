# Week 12 · Day 7 Solutions — Capstone Build Reference

Day 7 is a build day — there are no discrete exercises with single answers. This file provides reference implementations and a "done" checklist for each build hour.

## Hour 1 — Scaffold verification

Expected directory structure after Hour 1:
```
src/finops_agent/
├── __init__.py          ✓
├── config.py            ✓
├── core/
│   ├── __init__.py
│   ├── agent.py         ✓
│   ├── registry.py      ✓
│   └── planner.py       ✓
├── memory/
│   ├── __init__.py
│   └── store.py         ✓
├── tools/
│   ├── __init__.py
│   ├── finance.py       ✓
│   └── controlled.py    ✓
├── routing/
│   ├── __init__.py
│   └── router.py        ✓
├── hitl/
│   ├── __init__.py
│   └── gate.py          ✓
├── audit/
│   ├── __init__.py
│   └── logger.py        ✓
├── orchestration/
│   ├── __init__.py
│   ├── coordinator.py   ✓
│   └── supervisor.py    ✓
├── mcp/
│   ├── __init__.py
│   └── server.py        ✓
├── a2a/
│   ├── __init__.py
│   └── protocol.py      ✓
├── evaluation/
│   ├── __init__.py
│   └── harness.py       ✓
├── observability/
│   ├── __init__.py
│   └── metrics.py       ✓
└── api/
    ├── __init__.py
    └── main.py          ✓
tests/
├── unit/
│   ├── test_registry.py
│   ├── test_router.py
│   ├── test_memory.py
│   ├── test_hitl.py
│   ├── test_audit.py
│   └── test_planner_parser.py
├── integration/
│   ├── test_full_stack.py
│   ├── test_eval_harness.py
│   └── test_api.py
eval/
└── dataset.jsonl        ✓ (6+ cases)
```

## LLM mock pattern (reference for all tests)

```python
# Reusable mock factory for anthropic.Anthropic
from unittest.mock import MagicMock, patch

def make_mock_client(response_text: str = "Final Answer: done") -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_client

# Usage in tests:
def test_planner_with_mock_llm():
    from finops_agent.core.planner import ReActPlanner
    from finops_agent.core.registry import build_default_registry
    from finops_agent.memory.store import SQLiteMemoryStore
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = SQLiteMemoryStore(db_path=os.path.join(tmpdir, "mem.db"))
        registry = build_default_registry()
        client = make_mock_client("Thought: done.\nFinal Answer: Invoice retrieved.")
        planner = ReActPlanner(client=client, model="claude-sonnet-5-5", registry=registry, memory=memory)
        steps = planner.plan(session_id="s1", task="retrieve INV-001")
        assert any(s.is_final for s in steps)
```

## Minimum passing test suite (12 tests)

```python
# tests/unit/test_minimum_suite.py
"""Minimum 12 tests for pass gate — expand in your own test files."""
import pytest

def test_tool_registry_register_and_get():
    from finops_agent.core.registry import ToolRegistry, ToolSchema
    r = ToolRegistry()
    def fn(x: str) -> str: return x
    s = ToolSchema(name="t", description="d", parameters={}, risk_level="low")
    r.register(s, fn)
    assert r.get("t") is not None

def test_tool_registry_list_schemas():
    from finops_agent.core.registry import ToolRegistry, ToolSchema
    r = ToolRegistry()
    s = ToolSchema(name="t2", description="d", parameters={}, risk_level="low")
    r.register(s, lambda: None)
    assert any(sc.name == "t2" for sc in r.list_schemas())

def test_agent_router_finance():
    from finops_agent.routing.router import AgentRouter
    r = AgentRouter()
    d = r.route("process invoice INV-001")
    assert d.agent_name == "finance_agent"

def test_agent_router_ops():
    from finops_agent.routing.router import AgentRouter
    r = AgentRouter()
    d = r.route("create ops ticket for outage")
    assert d.agent_name == "ops_agent"

def test_hitl_auto_approve_low():
    from finops_agent.hitl.gate import HumanApprovalGate, ApprovalRequest, ApprovalStatus
    gate = HumanApprovalGate(auto_approve_levels={"low"})
    req = ApprovalRequest(session_id="s", tool_name="t", parameters={}, risk_level="low", reason="r")
    result = gate.request_approval(req)
    assert result.status == ApprovalStatus.APPROVED

def test_hitl_blocks_critical():
    from finops_agent.hitl.gate import HumanApprovalGate, ApprovalRequest, ApprovalStatus
    gate = HumanApprovalGate(auto_approve_levels={"low"})
    req = ApprovalRequest(session_id="s", tool_name="t", parameters={}, risk_level="critical", reason="r")
    result = gate.request_approval(req)
    assert result.status == ApprovalStatus.PENDING

def test_audit_log_write_read(tmp_path):
    from finops_agent.audit.logger import AuditLogger
    a = AuditLogger(log_path=str(tmp_path / "a.jsonl"))
    a.log_event("test", session_id="s1")
    events = a.read_events()
    assert len(events) == 1

def test_memory_sqlite_save_and_get(tmp_path):
    from finops_agent.memory.store import SQLiteMemoryStore
    m = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"))
    m.save_session("s1", "task1", [])
    ctx = m.get_context("s1")
    assert "task1" in ctx

def test_retrieve_invoice_valid():
    from finops_agent.tools.finance import retrieve_invoice
    result = retrieve_invoice("INV-001")
    assert result.invoice_id == "INV-001"
    assert result.amount > 0

def test_retrieve_invoice_invalid():
    from finops_agent.tools.finance import retrieve_invoice
    with pytest.raises(ValueError, match="not found"):
        retrieve_invoice("INV-999")

def test_a2a_send_receive():
    from finops_agent.a2a.protocol import A2AChannel, A2AMessage
    ch = A2AChannel()
    msg = A2AMessage(sender="a", recipient="b", payload={})
    ch.send(msg)
    msgs = ch.receive("b")
    assert len(msgs) == 1

def test_eval_harness_report_shape():
    from finops_agent.evaluation.harness import EvalResult, EvalHarness
    results = [EvalResult(case_id="c1", passed=True, latency_ms=100.0, score=1.0),
               EvalResult(case_id="c2", passed=False, latency_ms=50.0, score=0.5)]
    # Directly test report() without loading dataset
    harness = EvalHarness(agent_fn=lambda x: ([], ""), dataset_path="/dev/null")
    report = harness.report(results)
    assert report["total"] == 2
    assert report["passed"] == 1
    assert report["pass_rate"] == 0.5
```
