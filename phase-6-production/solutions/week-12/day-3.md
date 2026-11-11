# Week 12 · Day 3 — Solutions: Tool Abuse, Excessive Agency & Sandboxing

## Exercise 1

`ToolSandbox` with Pydantic v2 models, parameter validation, and audit logging.

```python
"""
solutions/week-12/tool_sandbox.py

ToolSandbox — allowlist-based parameter validation with audit logging.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ── Parameter constraint models ───────────────────────────────────────────────

class ParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    PATH = "path"
    URL = "url"
    ENUM = "enum"


class ParamConstraint(BaseModel):
    name: str
    param_type: ParamType
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    max_length: int | None = None
    allowed_values: list[str] | None = None   # for enum type
    path_allowlist: list[str] | None = None   # for path type
    url_prefix_allowlist: list[str] | None = None  # for url type


class ToolConfig(BaseModel):
    tool_name: str
    description: str
    param_constraints: list[ParamConstraint]
    max_execution_ms: int = 5_000
    requires_human_approval: bool = False


class AuditEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: float = Field(default_factory=time.time)
    tool_name: str
    params: dict[str, Any]
    outcome: str          # "allowed" | "blocked" | "error"
    block_reason: str | None = None
    agent_id: str | None = None


# ── Validation logic ──────────────────────────────────────────────────────────

class ValidationError(Exception):
    pass


def _validate_param(value: Any, constraint: ParamConstraint) -> Any:
    """Validate a single parameter value against its constraint. Raises ValidationError."""
    if value is None:
        if constraint.required:
            raise ValidationError(f"Required parameter '{constraint.name}' is missing.")
        return value

    match constraint.param_type:
        case ParamType.INTEGER | ParamType.FLOAT:
            try:
                v = int(value) if constraint.param_type == ParamType.INTEGER else float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"'{constraint.name}' must be numeric, got {type(value).__name__}"
                ) from exc
            if constraint.min_value is not None and v < constraint.min_value:
                raise ValidationError(
                    f"'{constraint.name}' = {v} is below minimum {constraint.min_value}"
                )
            if constraint.max_value is not None and v > constraint.max_value:
                raise ValidationError(
                    f"'{constraint.name}' = {v} exceeds maximum {constraint.max_value}"
                )
            return v

        case ParamType.STRING:
            if not isinstance(value, str):
                raise ValidationError(f"'{constraint.name}' must be a string.")
            if constraint.max_length and len(value) > constraint.max_length:
                raise ValidationError(
                    f"'{constraint.name}' length {len(value)} exceeds max {constraint.max_length}"
                )
            return value

        case ParamType.ENUM:
            if constraint.allowed_values is None:
                raise ValidationError(f"'{constraint.name}' has no allowed_values defined.")
            if str(value) not in constraint.allowed_values:
                raise ValidationError(
                    f"'{constraint.name}' = '{value}' not in allowed values: {constraint.allowed_values}"
                )
            return str(value)

        case ParamType.PATH:
            p = Path(str(value)).resolve()
            if constraint.path_allowlist:
                if not any(
                    str(p).startswith(str(Path(allowed).resolve()))
                    for allowed in constraint.path_allowlist
                ):
                    raise ValidationError(
                        f"'{constraint.name}' path '{p}' outside allowed directories: {constraint.path_allowlist}"
                    )
            return str(p)

        case ParamType.URL:
            url_str = str(value)
            if constraint.url_prefix_allowlist:
                if not any(url_str.startswith(prefix) for prefix in constraint.url_prefix_allowlist):
                    raise ValidationError(
                        f"'{constraint.name}' URL '{url_str}' not in allowed prefixes: {constraint.url_prefix_allowlist}"
                    )
            return url_str

        case _:
            return value


class ToolSandbox:
    """
    Validates tool call parameters before execution.
    All calls (allowed or blocked) produce an AuditEntry.
    """

    def __init__(self, tools: list[ToolConfig]) -> None:
        self._tools: dict[str, ToolConfig] = {t.tool_name: t for t in tools}
        self._audit_log: list[AuditEntry] = []

    def validate(
        self,
        tool_name: str,
        params: dict[str, Any],
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate params for a named tool.
        Returns sanitized params dict on success.
        Raises ValidationError and logs a blocked AuditEntry on failure.
        """
        if tool_name not in self._tools:
            self._record_audit(tool_name, params, "blocked", f"Unknown tool: {tool_name}", agent_id)
            raise ValidationError(f"Tool '{tool_name}' is not registered.")

        config = self._tools[tool_name]
        sanitized: dict[str, Any] = {}

        try:
            for constraint in config.param_constraints:
                raw = params.get(constraint.name)
                sanitized[constraint.name] = _validate_param(raw, constraint)
        except ValidationError as exc:
            self._record_audit(tool_name, params, "blocked", str(exc), agent_id)
            raise

        self._record_audit(tool_name, sanitized, "allowed", None, agent_id)
        logger.info("Tool call allowed. tool=%s agent=%s", tool_name, agent_id)
        return sanitized

    def _record_audit(
        self,
        tool_name: str,
        params: dict[str, Any],
        outcome: str,
        block_reason: str | None,
        agent_id: str | None,
    ) -> None:
        entry = AuditEntry(
            tool_name=tool_name,
            params=params,
            outcome=outcome,
            block_reason=block_reason,
            agent_id=agent_id,
        )
        self._audit_log.append(entry)
        if outcome == "blocked":
            logger.warning(
                "Tool call blocked. tool=%s reason=%s agent=%s",
                tool_name, block_reason, agent_id,
            )

    @property
    def audit_log(self) -> list[AuditEntry]:
        return list(self._audit_log)
```

---

## Exercise 2

Least-privilege tool registry mapping agent roles to allowed tools.

```python
"""
solutions/week-12/tool_registry.py

Least-privilege tool registry — each agent role gets a scoped view.
"""
from __future__ import annotations

import logging
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    WRITER = "writer"
    ADMIN = "admin"


class ToolPermission(BaseModel):
    tool_name: str
    allowed_roles: list[AgentRole]


# Define the permission matrix
TOOL_PERMISSIONS: list[ToolPermission] = [
    ToolPermission(tool_name="web_search", allowed_roles=[AgentRole.RESEARCHER, AgentRole.PLANNER]),
    ToolPermission(tool_name="read_file", allowed_roles=[AgentRole.RESEARCHER, AgentRole.WRITER]),
    ToolPermission(tool_name="write_file", allowed_roles=[AgentRole.WRITER]),
    ToolPermission(tool_name="send_email", allowed_roles=[AgentRole.ADMIN]),
    ToolPermission(tool_name="execute_code", allowed_roles=[AgentRole.ADMIN]),
    ToolPermission(tool_name="list_tasks", allowed_roles=[AgentRole.PLANNER, AgentRole.ADMIN]),
]


class ToolRegistry:
    def __init__(self, permissions: list[ToolPermission]) -> None:
        self._permissions = {tp.tool_name: tp for tp in permissions}

    def allowed_tools(self, role: AgentRole) -> list[str]:
        return [
            name
            for name, perm in self._permissions.items()
            if role in perm.allowed_roles
        ]

    def check(self, role: AgentRole, tool_name: str) -> bool:
        perm = self._permissions.get(tool_name)
        if perm is None:
            logger.warning("Unknown tool: %s", tool_name)
            return False
        allowed = role in perm.allowed_roles
        if not allowed:
            logger.warning(
                "Access denied. role=%s tool=%s allowed_roles=%s",
                role.value, tool_name, [r.value for r in perm.allowed_roles],
            )
        return allowed
```

---

## Exercise 3

Pytest tests for the sandbox validation logic.

```python
"""
tests/test_tool_sandbox.py
"""
import pytest
from tool_sandbox import ToolSandbox, ToolConfig, ParamConstraint, ParamType, ValidationError


@pytest.fixture
def sandbox() -> ToolSandbox:
    return ToolSandbox(tools=[
        ToolConfig(
            tool_name="read_file",
            description="Read a file from the allowed data directory",
            param_constraints=[
                ParamConstraint(
                    name="path",
                    param_type=ParamType.PATH,
                    path_allowlist=["/tmp/data"],
                ),
            ],
        ),
        ToolConfig(
            tool_name="fetch_url",
            description="Fetch a URL from an allowed domain",
            param_constraints=[
                ParamConstraint(
                    name="url",
                    param_type=ParamType.URL,
                    url_prefix_allowlist=["https://api.example.com/"],
                ),
            ],
        ),
        ToolConfig(
            tool_name="set_temperature",
            description="Set a numeric temperature parameter",
            param_constraints=[
                ParamConstraint(
                    name="temperature",
                    param_type=ParamType.FLOAT,
                    min_value=0.0,
                    max_value=1.0,
                ),
            ],
        ),
    ])


def test_valid_path_allowed(sandbox: ToolSandbox, tmp_path) -> None:
    # Create a file inside /tmp/data equivalent using tmp_path
    # For this test we just check path normalization within allowed prefix
    # We mock the allowlist to use tmp_path
    from tool_sandbox import ToolConfig, ParamConstraint, ParamType
    sb = ToolSandbox(tools=[
        ToolConfig(
            tool_name="read_file",
            description="Read",
            param_constraints=[
                ParamConstraint(
                    name="path",
                    param_type=ParamType.PATH,
                    path_allowlist=[str(tmp_path)],
                )
            ],
        )
    ])
    result = sb.validate("read_file", {"path": str(tmp_path / "data.txt")})
    assert "data.txt" in result["path"]


def test_path_traversal_blocked(sandbox: ToolSandbox) -> None:
    with pytest.raises(ValidationError, match="outside allowed directories"):
        sandbox.validate("read_file", {"path": "/etc/passwd"})


def test_disallowed_url_blocked(sandbox: ToolSandbox) -> None:
    with pytest.raises(ValidationError, match="not in allowed prefixes"):
        sandbox.validate("fetch_url", {"url": "https://evil.com/steal"})


def test_allowed_url_passes(sandbox: ToolSandbox) -> None:
    result = sandbox.validate("fetch_url", {"url": "https://api.example.com/v1/data"})
    assert result["url"].startswith("https://api.example.com/")


def test_float_out_of_range_blocked(sandbox: ToolSandbox) -> None:
    with pytest.raises(ValidationError, match="exceeds maximum"):
        sandbox.validate("set_temperature", {"temperature": 1.5})


def test_valid_float_passes(sandbox: ToolSandbox) -> None:
    result = sandbox.validate("set_temperature", {"temperature": 0.7})
    assert result["temperature"] == pytest.approx(0.7)


def test_unknown_tool_blocked(sandbox: ToolSandbox) -> None:
    with pytest.raises(ValidationError, match="not registered"):
        sandbox.validate("delete_database", {"confirm": "yes"})


def test_audit_log_records_blocked_calls(sandbox: ToolSandbox) -> None:
    try:
        sandbox.validate("read_file", {"path": "/root/.ssh/id_rsa"})
    except ValidationError:
        pass
    blocked = [e for e in sandbox.audit_log if e.outcome == "blocked"]
    assert len(blocked) == 1
    assert "outside allowed" in (blocked[0].block_reason or "")
```
