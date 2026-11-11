# Week 12 · Day 1 Solutions — Framework Architecture & Module Wiring

## Exercise 1

```python
# Addition to src/finops_agent/__init__.py
from finops_agent.tools.finance import validate_expense_report  # hypothetical function

# ToolSchema for validate_expense_report
from finops_agent.core.registry import ToolSchema

validate_expense_schema = ToolSchema(
    name="validate_expense_report",
    description="Validates a submitted expense report against policy rules",
    parameters={
        "report_id": "str",
        "submitter_id": "str",
        "total_amount": "float",
    },
    requires_approval=False,
    risk_level="medium",
)
```

The `DataValidationAgent` entry in `__init__.py`:
```python
# src/finops_agent/__init__.py  (extended)
from finops_agent.agents.validation import DataValidationAgent  # new agent module

__all__ = [
    "AgentCore", "ToolRegistry", "MemoryStore", "AgentRouter",
    "HumanApprovalGate", "AuditLogger", "MultiAgentCoordinator",
    "DataValidationAgent",
]
```

## Exercise 2

```python
# src/finops_agent/core/agent.py — extended AgentState
from pydantic import BaseModel, Field, computed_field

class AgentState(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    context_window_tokens: int = 0

    @computed_field
    @property
    def is_over_budget(self) -> bool:
        return self.context_window_tokens > 100_000
```

Note: `@computed_field` is Pydantic v2's way to expose a property as a field in `.model_dump()`. Alternatively, use a plain `@property` if you do not need it in serialization:
```python
    @property
    def is_over_budget(self) -> bool:
        return self.context_window_tokens > 100_000
```

## Exercise 3

Plain-text dependency graph:

```
config.py            (leaf — no inbound imports from framework modules)
core/registry.py     (leaf — depends only on stdlib + pydantic)
core/agent.py        → config.py, core/registry.py, memory/store.py, audit/logger.py
core/planner.py      → core/registry.py, memory/store.py
memory/store.py      (leaf — depends only on stdlib + pydantic)
audit/logger.py      (leaf — depends only on stdlib + pydantic)
hitl/gate.py         (leaf — depends only on stdlib + pydantic)
routing/router.py    (leaf — depends only on stdlib + pydantic)
tools/finance.py     (leaf — depends only on pydantic)
tools/controlled.py  → hitl/gate.py
orchestration/coordinator.py → (no framework deps — takes callables)
orchestration/supervisor.py  → orchestration/coordinator.py
mcp/server.py        → tools/finance.py
a2a/protocol.py      (leaf)
evaluation/harness.py → (takes callable — no framework deps)
observability/metrics.py (leaf)
api/main.py          → ALL of the above (wiring layer)
```

Pure leaves (zero inbound from other framework modules):
- `config.py`, `core/registry.py`, `memory/store.py`, `audit/logger.py`, `hitl/gate.py`, `routing/router.py`, `tools/finance.py`, `a2a/protocol.py`, `observability/metrics.py`

## Exercise 4

```python
# src/finops_agent/config.py — extended with field_validator
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""
    primary_model: str = "claude-sonnet-5-5"
    planning_model: str = "claude-opus-5-5"
    fast_model: str = "claude-haiku-5-5"
    max_tool_retries: int = 3
    hitl_timeout_seconds: int = 300
    audit_log_path: str = "logs/audit.jsonl"
    memory_backend: str = "sqlite"
    enable_mcp: bool = True
    enable_a2a: bool = True

    @field_validator("primary_model", "planning_model", "fast_model")
    @classmethod
    def must_be_claude_model(cls, v: str) -> str:
        if not v.startswith("claude-"):
            raise ValueError(f"Model ID must start with 'claude-', got: {v!r}")
        return v
```

Test:
```python
import pytest
from pydantic import ValidationError

def test_invalid_model_rejected():
    from finops_agent.config import AgentSettings
    with pytest.raises(ValidationError, match="must start with 'claude-'"):
        AgentSettings(primary_model="gpt-4o", anthropic_api_key="dummy")

def test_valid_model_accepted():
    s = AgentSettings(primary_model="claude-sonnet-5-5", anthropic_api_key="dummy")
    assert s.primary_model == "claude-sonnet-5-5"
```
