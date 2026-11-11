# Week 12 · Day 2 Solutions — Orchestration Core

## Exercise 1 — Model-based routing with claude-haiku-5-5

```python
# src/finops_agent/routing/router.py — extended
import json
import logging
import os
import anthropic
from pydantic import BaseModel

logger = logging.getLogger(__name__)

VALID_AGENTS = {"finance_agent", "ops_agent", "approval_agent", "reporting_agent"}

ROUTE_PROMPT = """Classify this task into exactly one of these agent names:
finance_agent, ops_agent, approval_agent, reporting_agent

Task: {task}

Reply with ONLY the agent name — no explanation, no punctuation."""

class RouteDecision(BaseModel):
    agent_name: str
    confidence: float
    reason: str

class AgentRouter:
    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        return self._client

    def route(self, task: str) -> RouteDecision:
        # Fast keyword path first
        task_lower = task.lower()
        rules = [
            (["invoice", "payment", "budget", "expense", "finance"], "finance_agent"),
            (["ticket", "service", "outage", "ops", "incident"], "ops_agent"),
            (["approve", "escalate", "reject", "authorize"], "approval_agent"),
            (["report", "summary", "analytics", "kpi"], "reporting_agent"),
        ]
        for keywords, agent in rules:
            if any(kw in task_lower for kw in keywords):
                return RouteDecision(agent_name=agent, confidence=0.85, reason=f"keyword: {keywords[0]}")

        # Model-based fallback
        return self._model_route(task)

    def _model_route(self, task: str) -> RouteDecision:
        client = self._get_client()
        try:
            response = client.messages.create(
                model="claude-haiku-5-5",
                max_tokens=32,
                messages=[{"role": "user", "content": ROUTE_PROMPT.format(task=task)}],
            )
            raw = response.content[0].text.strip().lower()
            # Validate
            if raw not in VALID_AGENTS:
                logger.warning("LLM returned unknown agent %r, defaulting to finance_agent", raw)
                return RouteDecision(agent_name="finance_agent", confidence=0.3, reason="llm_invalid_fallback")
            return RouteDecision(agent_name=raw, confidence=0.75, reason="llm_classification")
        except Exception as e:
            logger.error("Model-based routing failed: %s", e)
            return RouteDecision(agent_name="finance_agent", confidence=0.3, reason=f"llm_error: {e}")
```

## Exercise 2 — RedisMemoryStore stub

```python
# src/finops_agent/memory/redis_backend.py
import logging
import json
from finops_agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)

class RedisMemoryStore(MemoryStore):
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        import redis  # type: ignore
        self._redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        logger.info("RedisMemoryStore connected to %s:%d db=%d", host, port, db)

    def get_context(self, session_id: str) -> str:
        key = f"session:{session_id}:history"
        entries = self._redis.lrange(key, -5, -1)  # last 5 entries
        if not entries:
            return "No prior context."
        return "\n".join(f"- {e}" for e in entries)

    def save_session(self, session_id: str, task: str, steps: list) -> None:
        key = f"session:{session_id}:history"
        summary = f"{task} | {len(steps)} steps"
        self._redis.rpush(key, summary)
        self._redis.expire(key, 86400)  # 24h TTL
        logger.info("Saved session %s to Redis", session_id)
```

Test with mock:
```python
# tests/unit/test_redis_memory.py
from unittest.mock import MagicMock, patch
import pytest

def test_redis_memory_get_context():
    with patch("redis.Redis") as mock_redis_cls:
        mock_r = MagicMock()
        mock_redis_cls.return_value = mock_r
        mock_r.lrange.return_value = ["task1 | 3 steps", "task2 | 2 steps"]

        from finops_agent.memory.redis_backend import RedisMemoryStore
        store = RedisMemoryStore()
        context = store.get_context("s1")
        assert "task1" in context
        assert "task2" in context

def test_redis_memory_save_session():
    with patch("redis.Redis") as mock_redis_cls:
        mock_r = MagicMock()
        mock_redis_cls.return_value = mock_r

        from finops_agent.memory.redis_backend import RedisMemoryStore
        store = RedisMemoryStore()
        store.save_session("s1", "pay invoice", [])
        mock_r.rpush.assert_called_once()
        mock_r.expire.assert_called_once()
```

## Exercise 3 — Regex-based _parse_step

```python
import re
from finops_agent.core.planner import PlanStep

_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)\(([^)]*)\)", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*?)$", re.DOTALL)

def parse_step(raw: str) -> PlanStep:
    final_match = _FINAL_RE.search(raw)
    if final_match:
        thought_match = _THOUGHT_RE.search(raw)
        thought = thought_match.group(1).strip() if thought_match else ""
        return PlanStep(thought=thought, is_final=True, final_answer=final_match.group(1).strip())

    thought_match = _THOUGHT_RE.search(raw)
    thought = thought_match.group(1).strip() if thought_match else raw.strip()

    action_match = _ACTION_RE.search(raw)
    action_name = action_match.group(1) if action_match else None

    return PlanStep(thought=thought, action=action_name)
```

Tests:
```python
# tests/unit/test_planner_parser.py
def test_parse_final_answer():
    raw = "Thought: All done.\nFinal Answer: Invoice INV-001 is valid."
    step = parse_step(raw)
    assert step.is_final is True
    assert "INV-001" in step.final_answer

def test_parse_action():
    raw = "Thought: I need to retrieve.\nAction: retrieve_invoice(invoice_id=INV-001)"
    step = parse_step(raw)
    assert step.action == "retrieve_invoice"
    assert step.is_final is False

def test_parse_no_action():
    raw = "Thought: Let me think about this..."
    step = parse_step(raw)
    assert step.action is None
    assert step.is_final is False

def test_parse_empty():
    step = parse_step("")
    assert step.is_final is False
    assert step.action is None
```

## Exercise 4 — build_default_registry factory

```python
# src/finops_agent/core/registry.py — addition

def build_default_registry() -> ToolRegistry:
    """Factory that wires all default finance tools into a ToolRegistry."""
    from finops_agent.tools.finance import retrieve_invoice, validate_payment, check_budget

    registry = ToolRegistry()
    registry.register(
        ToolSchema(
            name="retrieve_invoice",
            description="Retrieve an invoice by ID from the finance system",
            parameters={"invoice_id": "str"},
            requires_approval=False,
            risk_level="low",
        ),
        retrieve_invoice,
    )
    registry.register(
        ToolSchema(
            name="validate_payment",
            description="Validate that a payment amount matches the invoice",
            parameters={"invoice_id": "str", "amount": "float"},
            requires_approval=False,
            risk_level="low",
        ),
        validate_payment,
    )
    registry.register(
        ToolSchema(
            name="check_budget",
            description="Check whether a department has budget for a given amount",
            parameters={"department": "str", "amount": "float"},
            requires_approval=False,
            risk_level="medium",
        ),
        check_budget,
    )
    return registry
```
