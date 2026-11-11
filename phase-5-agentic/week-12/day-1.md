# Phase 5 · Week 12 · Day 1 — Framework Architecture & Module Wiring

> Objective: Understand the capstone framework layer structure and how each subsystem wires together via typed interfaces.  ·  Time: ~1 hr  ·  Prereqs: [[week-11/day-5]]

## Concept

The FinOpsAgent framework is a multi-layer enterprise agent system built around separation of concerns. Each subsystem owns exactly one responsibility and communicates with adjacent subsystems through Pydantic model seams — never raw dicts.

The framework layers, from outermost to innermost:

| Layer | Module | Responsibility |
|---|---|---|
| API Gateway | `main.py` / FastAPI | Accept requests, auth, rate limiting |
| Router | `routing/router.py` | Classify task → assign to agent |
| Agent Core | `core/agent.py` | ReAct loop, state management |
| Tool Registry | `core/registry.py` | Register, look up, execute tools |
| Memory | `memory/store.py` | Session + long-term context |
| HITL Gate | `hitl/gate.py` | Block high-risk actions for human review |
| Audit Logger | `audit/logger.py` | Append-only JSONL event record |
| Orchestration | `orchestration/coordinator.py` | Multi-agent dispatch and aggregation |
| MCP Server | `mcp/server.py` | Expose tools over HTTP to external consumers |
| A2A Protocol | `a2a/protocol.py` | Cross-agent structured messaging |
| Evaluation | `evaluation/harness.py` | Offline agent quality measurement |
| Observability | `observability/metrics.py` | Metrics and tracing stubs |

The public API surface is controlled through `__init__.py` — only explicitly listed names are importable by consumers.

## How it works internally

**Module dependency graph (simplified):**

```
config.py
    └─> core/agent.py
            ├─> core/registry.py
            ├─> memory/store.py
            └─> audit/logger.py
    └─> routing/router.py
    └─> hitl/gate.py
    └─> orchestration/coordinator.py
            ├─> core/agent.py
            └─> a2a/protocol.py
    └─> mcp/server.py
            └─> tools/finance.py
            └─> tools/ops.py
            └─> tools/controlled.py
                    └─> hitl/gate.py
```

**Request flow (happy path):**

1. `POST /agent/run` hits FastAPI → `AgentSettings` loaded from env
2. `AgentRouter.route(task)` classifies the task → returns `RouteDecision`
3. `AgentCore.run(session_id, user_input)` initializes `AgentState`
4. `ReActPlanner.plan()` calls LLM → parses `PlanStep` → dispatches to `ToolRegistry`
5. `ToolRegistry.get(name)` returns `(ToolSchema, Callable)` → callable executed
6. If `ToolSchema.requires_approval` and `risk_level` is high/critical → `HumanApprovalGate` blocks
7. Tool result appended to `AgentState.messages`
8. `MemoryStore.save_session()` persists summary
9. `AuditLogger.log_event()` writes JSONL entry
10. `AgentState` returned → serialized as JSON response

**Why Pydantic at every seam:**
- Serialization/deserialization is free and validated
- Type errors surface at boundary crossing, not deep in business logic
- `model_dump()` / `model_validate()` make persistence and testing straightforward
- `pydantic-settings` gives the same discipline to environment config

**`__init__.py` as public API contract:**
Every module exports only what downstream layers need. Anything not in `__all__` is internal. This prevents accidental coupling to implementation details.

## Worked example

Public API surface:

```python
# src/finops_agent/__init__.py
from finops_agent.core.agent import AgentCore
from finops_agent.core.registry import ToolRegistry
from finops_agent.memory.store import MemoryStore
from finops_agent.routing.router import AgentRouter
from finops_agent.hitl.gate import HumanApprovalGate
from finops_agent.audit.logger import AuditLogger
from finops_agent.orchestration.coordinator import MultiAgentCoordinator

__all__ = [
    "AgentCore", "ToolRegistry", "MemoryStore", "AgentRouter",
    "HumanApprovalGate", "AuditLogger", "MultiAgentCoordinator",
]
```

Settings via `pydantic-settings` (secrets from environment, never hardcoded):

```python
# src/finops_agent/config.py
import logging
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    primary_model: str = "claude-sonnet-5-5"
    planning_model: str = "claude-opus-5-5"
    fast_model: str = "claude-haiku-5-5"
    max_tool_retries: int = 3
    hitl_timeout_seconds: int = 300
    audit_log_path: str = "logs/audit.jsonl"
    memory_backend: str = "sqlite"  # or "redis"
    enable_mcp: bool = True
    enable_a2a: bool = True
```

Tool registry — schema-first, then callable:

```python
# src/finops_agent/core/registry.py
import logging
from dataclasses import dataclass, field
from typing import Any, Callable
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False
    risk_level: str = "low"  # low | medium | high | critical

@dataclass
class ToolRegistry:
    _tools: dict[str, tuple[ToolSchema, Callable]] = field(default_factory=dict)

    def register(self, schema: ToolSchema, fn: Callable) -> None:
        self._tools[schema.name] = (schema, fn)
        logger.info("Registered tool %s (risk=%s)", schema.name, schema.risk_level)

    def get(self, name: str) -> tuple[ToolSchema, Callable] | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[ToolSchema]:
        return [s for s, _ in self._tools.values()]
```

Agent core with typed state:

```python
# src/finops_agent/core/agent.py
import logging
from typing import Any
from pydantic import BaseModel, Field
import anthropic

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class AgentCore:
    def __init__(self, settings, registry, memory, audit_logger):
        self.settings = settings
        self.registry = registry
        self.memory = memory
        self.audit = audit_logger
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.info("AgentCore initialized with model=%s", settings.primary_model)

    def run(self, session_id: str, user_input: str) -> AgentState:
        state = AgentState(session_id=session_id)
        state.messages.append({"role": "user", "content": user_input})
        self.audit.log_event("agent_run_start", session_id=session_id, input=user_input)
        # tool loop handled by orchestration layer
        return state
```

Full project directory layout:

```
src/finops_agent/
├── __init__.py
├── config.py
├── core/
│   ├── agent.py          # AgentCore, AgentState
│   ├── registry.py       # ToolRegistry, ToolSchema
│   └── planner.py        # ReActPlanner
├── memory/
│   ├── store.py          # MemoryStore ABC
│   └── sqlite_backend.py
├── tools/
│   ├── finance.py        # retrieve_invoice, validate_payment, check_budget
│   ├── ops.py            # check_service_status, create_ticket
│   └── controlled.py     # approve_payment, escalate_issue (require HITL)
├── routing/
│   └── router.py         # AgentRouter: classify and route tasks
├── hitl/
│   └── gate.py           # HumanApprovalGate
├── audit/
│   └── logger.py         # AuditLogger (JSONL)
├── orchestration/
│   ├── coordinator.py    # MultiAgentCoordinator
│   └── supervisor.py     # SupervisorAgent
├── mcp/
│   └── server.py         # MCP tool server
├── a2a/
│   └── protocol.py       # A2A message format
├── evaluation/
│   └── harness.py        # EvalHarness
└── observability/
    └── metrics.py        # metrics + tracing stubs
tests/
├── unit/
└── integration/
```

## Exercises

1. Add a `DataValidationAgent` entry point to `__init__.py`. What `ToolSchema` would you define for a `validate_expense_report` tool with `requires_approval=False` and `risk_level="medium"`? Write the schema definition including realistic `parameters`.

2. Extend `AgentState` with a `context_window_tokens: int` field and an `is_over_budget` property that returns `True` when `context_window_tokens > 100_000`. How does Pydantic v2 handle computed properties?

3. Sketch the module dependency graph for the framework as a plain-text diagram (arrows = "depends on"). Which module(s) have zero inbound dependencies (pure leaves)?

4. The `AgentSettings.primary_model` defaults to `"claude-sonnet-5-5"`. Propose a validation approach using Pydantic v2 `@field_validator` that rejects any model string not starting with `"claude-"`. Write the validator and a test that passes a bad value and verifies `ValidationError` is raised.

## Self-check

- [ ] I can draw the full module dependency graph from memory
- [ ] I can explain why Pydantic models are used at module seams rather than dicts
- [ ] I can scaffold the full project layout without reference
- [ ] I understand how `AgentSettings` avoids hardcoded secrets
