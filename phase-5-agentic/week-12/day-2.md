# Phase 5 · Week 12 · Day 2 — Orchestration Core: Agents, Tools, Memory, Routing

> Objective: Implement the ReAct execution loop, SQLite memory store, tool registry wiring, and keyword-based task router.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

The orchestration core is the engine that moves a task from user input to final answer. It connects four subsystems:

- **ReActPlanner** — LLM-driven reason-then-act loop that produces `PlanStep` objects
- **ToolRegistry** — maps tool names to `(schema, callable)` pairs; decouples planner from implementation
- **MemoryStore** — persists session summaries so the planner has prior context on re-entry
- **AgentRouter** — classifies an incoming task and assigns it to the right specialist agent

These subsystems are deliberately independent: the planner does not import the router, the router does not import memory. They communicate only through the typed models that cross their boundaries.

**ReAct loop overview:**

```
user task
    │
    ▼
[Planner: Thought]  ← memory context injected here
    │
    ├─ Action: tool_name(params)
    │       │
    │       ▼
    │   [ToolRegistry: dispatch]
    │       │
    │       ▼
    │   Observation (tool result)
    │       │
    │       └──► appended to messages → loop repeats
    │
    └─ Final Answer → memory.save_session() → return steps
```

## How it works internally

**Step-by-step execution:**

1. `ReActPlanner.plan(session_id, task)` fetches prior context from `MemoryStore`
2. Builds a system prompt listing available tools
3. Calls `claude-sonnet-5-5` with the task and context
4. `_parse_step(raw)` parses the LLM text into a `PlanStep`
5. If `is_final=True` → loop exits, `memory.save_session()` is called, steps returned
6. If an `action` is present → `_execute_tool(name, inputs)` dispatches through `ToolRegistry`
7. Tool result becomes the next `Observation:` message → loop continues
8. Hard cap at `max_steps=10` to prevent runaway loops

**ToolRegistry decoupling:**
The planner calls `registry.get(name)` which returns a `(ToolSchema, Callable)` tuple. The planner never imports `tools/finance.py` directly. This means tools can be swapped or mocked without touching the planner.

**Memory as first-class module:**
`MemoryStore` is an ABC — the planner depends on the interface, not the implementation. `SQLiteMemoryStore` is the default concrete backend. Swapping to Redis requires only a new subclass.

**Router patterns:**
- **Keyword routing** (implemented below): O(1), deterministic, no LLM call. Good for clear domain separation.
- **Embedding-based routing**: compute cosine similarity between task embedding and agent description embeddings. More flexible but adds latency.
- **Model-based routing**: ask a fast model (`claude-haiku-5-5`) to classify. Most flexible, ~100ms overhead.

## Worked example

ReAct planner:

```python
# src/finops_agent/core/planner.py
import logging
from typing import Any
from pydantic import BaseModel
import anthropic

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial operations agent. You have access to tools.
Use ReAct format:
Thought: your reasoning
Action: tool_name(param1=value1, param2=value2)
Observation: [tool result will be inserted here]
...
Final Answer: your conclusion

Available tools: {tool_list}
"""

class PlanStep(BaseModel):
    thought: str
    action: str | None = None
    action_input: dict[str, Any] = {}
    observation: str | None = None
    is_final: bool = False
    final_answer: str | None = None

class ReActPlanner:
    def __init__(self, client: anthropic.Anthropic, model: str, registry, memory):
        self.client = client
        self.model = model
        self.registry = registry
        self.memory = memory
        self.max_steps = 10

    def plan(self, session_id: str, task: str) -> list[PlanStep]:
        context = self.memory.get_context(session_id)
        tool_list = ", ".join(s.name for s in self.registry.list_schemas())
        system = SYSTEM_PROMPT.format(tool_list=tool_list)
        messages = [{"role": "user", "content": f"Context: {context}\nTask: {task}"}]
        steps: list[PlanStep] = []

        for step_num in range(self.max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            raw = response.content[0].text
            step = self._parse_step(raw)
            steps.append(step)
            logger.debug("Step %d: action=%s final=%s", step_num, step.action, step.is_final)

            if step.is_final:
                break

            # Execute tool and append observation
            if step.action:
                obs = self._execute_tool(step.action, step.action_input)
                step.observation = obs
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Observation: {obs}"})

        self.memory.save_session(session_id, task, steps)
        return steps

    def _parse_step(self, raw: str) -> PlanStep:
        # Minimal parser — production would use regex
        if "Final Answer:" in raw:
            answer = raw.split("Final Answer:")[-1].strip()
            return PlanStep(thought=raw, is_final=True, final_answer=answer)
        thought = raw.split("Action:")[0].replace("Thought:", "").strip() if "Thought:" in raw else raw
        action_part = raw.split("Action:")[-1].split("\n")[0].strip() if "Action:" in raw else None
        action_name = action_part.split("(")[0].strip() if action_part else None
        return PlanStep(thought=thought, action=action_name)

    def _execute_tool(self, name: str, inputs: dict) -> str:
        entry = self.registry.get(name)
        if not entry:
            return f"Error: tool '{name}' not found"
        schema, fn = entry
        try:
            result = fn(**inputs)
            return str(result)
        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return f"Error: {e}"
```

Memory store (ABC + SQLite backend):

```python
# src/finops_agent/memory/store.py
import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MemoryEntry(BaseModel):
    session_id: str
    task: str
    summary: str
    step_count: int

class MemoryStore(ABC):
    @abstractmethod
    def get_context(self, session_id: str) -> str: ...
    @abstractmethod
    def save_session(self, session_id: str, task: str, steps: list) -> None: ...

class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS sessions
                (session_id TEXT, task TEXT, summary TEXT, step_count INTEGER)""")
        logger.info("SQLite memory store initialized at %s", self.db_path)

    def get_context(self, session_id: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT task, summary FROM sessions WHERE session_id=? ORDER BY rowid DESC LIMIT 5",
                (session_id,)
            ).fetchall()
        if not rows:
            return "No prior context."
        return "\n".join(f"- {r[0]}: {r[1]}" for r in rows)

    def save_session(self, session_id: str, task: str, steps: list) -> None:
        summary = f"{len(steps)} steps; final={'yes' if steps and steps[-1].is_final else 'no'}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?,?,?,?)",
                (session_id, task, summary, len(steps))
            )
        logger.info("Saved session %s (%d steps)", session_id, len(steps))
```

Keyword-based router:

```python
# src/finops_agent/routing/router.py
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    agent_name: str
    confidence: float
    reason: str

ROUTING_RULES: list[tuple[list[str], str]] = [
    (["invoice", "payment", "budget", "expense", "finance"], "finance_agent"),
    (["ticket", "service", "outage", "ops", "incident"], "ops_agent"),
    (["approve", "escalate", "reject", "authorize"], "approval_agent"),
    (["report", "summary", "analytics", "kpi"], "reporting_agent"),
]

class AgentRouter:
    def route(self, task: str) -> RouteDecision:
        task_lower = task.lower()
        for keywords, agent in ROUTING_RULES:
            if any(kw in task_lower for kw in keywords):
                logger.info("Routed task to %s via keyword match", agent)
                return RouteDecision(agent_name=agent, confidence=0.85, reason=f"keyword: {keywords[0]}")
        logger.warning("No route matched, defaulting to finance_agent")
        return RouteDecision(agent_name="finance_agent", confidence=0.5, reason="default fallback")
```

Finance tools (synthetic data, no real systems):

```python
# src/finops_agent/tools/finance.py
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class InvoiceResult(BaseModel):
    invoice_id: str
    vendor: str
    amount: float
    status: str
    currency: str = "USD"

# Synthetic data — no real systems
_INVOICES: dict[str, InvoiceResult] = {
    "INV-001": InvoiceResult(invoice_id="INV-001", vendor="Acme Corp", amount=15000.0, status="pending"),
    "INV-002": InvoiceResult(invoice_id="INV-002", vendor="TechSupply Ltd", amount=4200.0, status="approved"),
}

def retrieve_invoice(invoice_id: str) -> InvoiceResult:
    logger.info("Retrieving invoice %s", invoice_id)
    if invoice_id not in _INVOICES:
        raise ValueError(f"Invoice {invoice_id} not found")
    return _INVOICES[invoice_id]

def validate_payment(invoice_id: str, amount: float) -> dict:
    inv = retrieve_invoice(invoice_id)
    valid = abs(inv.amount - amount) < 0.01
    logger.info("Validated payment for %s: valid=%s", invoice_id, valid)
    return {"valid": valid, "expected": inv.amount, "provided": amount}

def check_budget(department: str, amount: float) -> dict:
    budgets = {"engineering": 50000.0, "marketing": 20000.0, "ops": 30000.0}
    limit = budgets.get(department, 10000.0)
    approved = amount <= limit
    logger.info("Budget check %s: amount=%.2f limit=%.2f approved=%s", department, amount, limit, approved)
    return {"approved": approved, "budget_limit": limit, "requested": amount}
```

**Wiring it together in a factory function:**

```python
# src/finops_agent/factory.py
import anthropic
import logging
import os
from finops_agent.config import AgentSettings
from finops_agent.core.agent import AgentCore
from finops_agent.core.registry import ToolRegistry, ToolSchema
from finops_agent.core.planner import ReActPlanner
from finops_agent.memory.store import SQLiteMemoryStore
from finops_agent.routing.router import AgentRouter
from finops_agent.audit.logger import AuditLogger
from finops_agent.tools.finance import retrieve_invoice, validate_payment, check_budget

logger = logging.getLogger(__name__)

def build_finops_agent() -> AgentCore:
    settings = AgentSettings()
    registry = ToolRegistry()
    registry.register(
        ToolSchema(name="retrieve_invoice", description="Fetch invoice by ID",
                   parameters={"invoice_id": "str"}, risk_level="low"),
        retrieve_invoice,
    )
    registry.register(
        ToolSchema(name="validate_payment", description="Validate payment amount against invoice",
                   parameters={"invoice_id": "str", "amount": "float"}, risk_level="low"),
        validate_payment,
    )
    registry.register(
        ToolSchema(name="check_budget", description="Check if amount is within department budget",
                   parameters={"department": "str", "amount": "float"}, risk_level="low"),
        check_budget,
    )
    memory = SQLiteMemoryStore()
    audit = AuditLogger(settings.audit_log_path)
    return AgentCore(settings, registry, memory, audit)
```

## Exercises

1. Extend `AgentRouter` with a `route_with_model` method that calls `claude-haiku-5-5` to classify a task into one of the four agent names (`finance_agent`, `ops_agent`, `approval_agent`, `reporting_agent`). Write the prompt. How would you handle the LLM returning an unexpected category (not in the four known names)?

2. Add a `RedisMemoryStore` subclass of `MemoryStore` with stub implementations of `get_context` and `save_session` that use `redis.Redis`. The stubs should call `redis.Redis(host=..., port=...)` but the actual connection should be injectable so tests can mock it. Write the class signature and both method bodies.

3. The `ReActPlanner._parse_step` is fragile (relies on string splitting). Rewrite it using `re` module regex to robustly extract `Thought:`, `Action:`, and `Final Answer:` sections, handling whitespace variation and multi-line thoughts. Write at least 3 unit tests covering: (a) a step with action, (b) a final answer step, (c) a step with neither action nor final answer.

4. Write a `build_default_registry()` factory function that creates a `ToolRegistry` and registers all three finance tools (`retrieve_invoice`, `validate_payment`, `check_budget`) with appropriate schemas. Where should this function live in the module tree?

## Self-check

- [ ] I can trace a full ReAct loop step by step on paper without reference
- [ ] I can implement `MemoryStore` ABC and a concrete SQLite backend from scratch
- [ ] I understand why `AgentRouter` returns a `confidence` score and how a caller would use it
- [ ] I can register tools in `ToolRegistry` and explain how it decouples the planner from tool implementation
