# Solutions — Week 09 · Day 6 — MCP + A2A Together & Standards Landscape

## Exercise 1: CodeExecutionAgent end-to-end

```python
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class A2ATask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    payload: str = ""
    result: str = ""
    status: str = "submitted"
    metadata: dict[str, Any] = field(default_factory=dict)


class MockMCPClient:
    def __init__(self, tools: dict[str, Callable[..., str]]) -> None:
        self._tools = tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            return {"error": f"Tool not found: {name}"}
        return {"content": [{"type": "text", "text": self._tools[name](**arguments)}]}


class CodeExecutionAgent:
    def __init__(self) -> None:
        self._mcp = MockMCPClient(
            tools={
                "run_python": lambda code: "Mean: 42.5, Std: 3.1",
            }
        )

    def get_skills(self) -> list[str]:
        return ["code-execution", "statistics"]

    async def handle(self, task: A2ATask) -> A2ATask:
        task.status = "working"
        code_snippet = task.metadata.get("code", "print('hello')")
        result = await self._mcp.call_tool("run_python", {"code": code_snippet})
        task.result = result["content"][0]["text"]
        task.status = "completed"
        return task


# Test
import pytest


@pytest.mark.asyncio
async def test_code_execution_agent():
    agent = CodeExecutionAgent()
    task = A2ATask(
        payload="Run statistics",
        metadata={"code": "import statistics; print(statistics.mean([40, 42, 45]))"},
    )
    result_task = await agent.handle(task)
    assert result_task.status == "completed"
    assert "42.5" in result_task.result
    assert "3.1" in result_task.result


@pytest.mark.asyncio
async def test_three_phase_orchestration():
    """End-to-end test: financial + search in parallel, then code-execution phase."""
    from unittest.mock import AsyncMock

    financial_agent = AsyncMock()
    financial_agent.handle = AsyncMock(return_value=A2ATask(
        result="Revenue: $2.4M", status="completed"
    ))

    search_agent = AsyncMock()
    search_agent.handle = AsyncMock(return_value=A2ATask(
        result="Market growing 15%", status="completed"
    ))

    code_agent = CodeExecutionAgent()

    # Phase 1: parallel
    financial_task = A2ATask(payload="financials")
    search_task = A2ATask(payload="search")
    fin, srch = await asyncio.gather(
        financial_agent.handle(financial_task),
        search_agent.handle(search_task),
    )

    # Phase 2: sequential — code execution
    code_task = A2ATask(
        payload="compute stats",
        metadata={"code": "mean = 42.5"},
    )
    code_result = await code_agent.handle(code_task)

    assert fin.result == "Revenue: $2.4M"
    assert srch.result == "Market growing 15%"
    assert "42.5" in code_result.result
```

## Exercise 2: Correlation ID propagation

```python
import logging


class CorrelationFilter(logging.Filter):
    def __init__(self, correlation_id: str) -> None:
        super().__init__()
        self.correlation_id = correlation_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"[cid:{self.correlation_id[:8]}] {record.msg}"
        return True


def setup_correlation_logging(correlation_id: str, logger: logging.Logger) -> None:
    """Attach correlation filter to logger."""
    f = CorrelationFilter(correlation_id)
    for handler in logging.root.handlers:
        handler.addFilter(f)
    logger.addFilter(f)


def test_correlation_id_propagation(caplog):
    import uuid
    cid = str(uuid.uuid4())
    agent_logger = logging.getLogger("test_agent")

    with caplog.at_level(logging.INFO, logger="test_agent"):
        f = CorrelationFilter(cid)
        agent_logger.addFilter(f)
        agent_logger.info("Starting task")
        agent_logger.info("Task complete")
        agent_logger.removeFilter(f)

    cid_prefix = f"[cid:{cid[:8]}]"
    assert all(cid_prefix in record.message for record in caplog.records)
```

## Exercise 3: AuthContext

```python
from typing import Literal
from pydantic import BaseModel
import pytest


class AuthContext(BaseModel):
    scheme: Literal["none", "bearer", "api_key"]
    token: str | None = None

    def is_valid(self) -> bool:
        if self.scheme == "none":
            return True
        return self.token is not None and len(self.token) > 0


class SecureAgent:
    REQUIRED_SCHEME: Literal["bearer"] = "bearer"

    async def handle(self, task: A2ATask, auth: AuthContext) -> A2ATask:
        if auth.scheme != self.REQUIRED_SCHEME or not auth.is_valid():
            raise PermissionError(f"Expected bearer auth, got scheme={auth.scheme!r}")
        task.status = "completed"
        task.result = "Secure result"
        return task


@pytest.mark.asyncio
async def test_secure_agent_with_valid_auth():
    agent = SecureAgent()
    auth = AuthContext(scheme="bearer", token="valid-token-123")
    task = A2ATask(payload="secure request")
    result = await agent.handle(task, auth)
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_secure_agent_rejects_no_auth():
    agent = SecureAgent()
    auth = AuthContext(scheme="none")
    task = A2ATask(payload="attempt")
    with pytest.raises(PermissionError):
        await agent.handle(task, auth)


@pytest.mark.asyncio
async def test_secure_agent_rejects_wrong_scheme():
    agent = SecureAgent()
    auth = AuthContext(scheme="api_key", token="my-key")
    task = A2ATask(payload="attempt")
    with pytest.raises(PermissionError):
        await agent.handle(task, auth)
```

## Exercise 4: ASCII architecture diagram

```python
"""
Three-hop A2A + MCP architecture:

UserAgent
    │
    │  A2A protocol (HTTPS, bearer token)
    │  POST /tasks/send  →  Authorization: Bearer <token-A>
    ▼
OrchestratorAgent   (A2A server + A2A client)
    │  Credentials for OrchestratorAgent are provisioned at deployment.
    │  UserAgent's token is NOT forwarded — new auth context begins here.
    │
    │  A2A protocol (HTTPS, bearer token)
    │  POST /tasks/send  →  Authorization: Bearer <token-B>
    ▼
SpecialistAgent   (A2A server + MCP client)
    │  token-B is verified by SpecialistAgent.
    │  SpecialistAgent opens its OWN MCP session with separate credentials.
    │
    │  MCP protocol (JSON-RPC 2.0 / HTTP+SSE, API key)
    │  initialize → list_tools → call_tool
    │  X-MCP-Key: <mcp-server-key>  (never seen by UserAgent or Orchestrator)
    ▼
MCPServer   (tool / data source)
    Returns tool results to SpecialistAgent only.

Auth boundaries:
  - UserAgent ↔ Orchestrator:  token-A  (user-scoped OAuth or JWT)
  - Orchestrator ↔ Specialist: token-B  (service-to-service bearer)
  - Specialist ↔ MCP:          mcp-key  (infra credential, lowest scope)

Each hop has independent credentials.
Compromise of token-A does NOT expose token-B or mcp-key.
"""
```
