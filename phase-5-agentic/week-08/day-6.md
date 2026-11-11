# Phase 5 · Week 08 · Day 6 (Saturday) — Build: MCP Server + Client + Agent Integration

> Objective: 4-hour deep build — complete MCP server with tools/resources/prompts, client adapter, LangGraph agent, and full test suite  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] – [[day-5]]

## Overview

Today you build a production-quality MCP system from scratch:

1. **MCP Server** (`task_server.py`) — a task management server with tools, resources, prompts
2. **MCP Client Adapter** (`mcp_adapter.py`) — reusable adapter converting MCP ↔ LangChain
3. **LangGraph Agent** (`task_agent.py`) — an agent that uses the server to manage tasks
4. **Test suite** (`tests/`) — full coverage with in-process mocked MCP sessions

Estimated time split: Server (1hr), Adapter (45min), Agent (45min), Tests (1.5hr).

## Part 1 — MCP Task Server (1 hr)

### Specification

Build `task_server.py` using FastMCP. The server manages a to-do list.

**Tools**:

| Tool | Args | Returns |
|---|---|---|
| `create_task` | `title: str, description: str, priority: "low"\|"medium"\|"high" = "medium"` | task ID |
| `complete_task` | `task_id: str` | success/error message |
| `list_tasks` | `status: "pending"\|"completed"\|"all" = "pending"` | formatted list |
| `get_task_detail` | `task_id: str` | full task JSON |
| `add_note_to_task` | `task_id: str, note: str` | success/error |

**Resources**:

| Resource URI | Content |
|---|---|
| `tasks://all` | JSON array of all tasks |
| `tasks://pending` | JSON array of pending tasks |
| `tasks://task/{task_id}` | JSON of a single task |
| `tasks://stats` | JSON summary (counts by status, priority) |

**Prompts**:

| Prompt | Args | Returns |
|---|---|---|
| `prioritize_tasks` | `context: str` | User message asking LLM to suggest priority order |
| `daily_summary` | `date: str = "today"` | User message asking LLM to generate a daily standup summary |

**Data model** (use Pydantic):

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
import uuid

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str
    priority: Literal["low", "medium", "high"] = "medium"
    status: Literal["pending", "completed"] = "pending"
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
```

### Starter Code

```python
# task_server.py
import json
import logging
import os
from datetime import datetime
from typing import Literal
import uuid

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    description: str
    priority: Literal["low", "medium", "high"] = "medium"
    status: Literal["pending", "completed"] = "pending"
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None


# In-memory store
TASKS: dict[str, Task] = {}


# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="task-manager",
    version="1.0.0",
    instructions=(
        "Task management server. Use create_task to add tasks, "
        "list_tasks to see pending work, complete_task when done. "
        "Resources provide JSON views; prompts help with planning."
    ),
)


# ── TODO: Implement tools ─────────────────────────────────────────────────────
# @mcp.tool()
# async def create_task(...) -> str:
#     ...
# (implement all 5 tools per spec above)


# ── TODO: Implement resources ─────────────────────────────────────────────────
# @mcp.resource("tasks://all", mime_type="application/json")
# def get_all_tasks() -> str:
#     ...
# (implement all 4 resources)


# ── TODO: Implement prompts ───────────────────────────────────────────────────
# @mcp.prompt()
# def prioritize_tasks(context: str) -> str:
#     ...
# (implement both prompts)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        mcp.run(transport="sse", host="127.0.0.1", port=8001)
    else:
        mcp.run(transport="stdio")
```

### Checkpoints

After Part 1:
- [ ] `python task_server.py` starts without error
- [ ] All 5 tools are registered
- [ ] All 4 resources are registered
- [ ] Both prompts are registered
- [ ] Pydantic model validates all task data

## Part 2 — MCP Client Adapter (45 min)

Build `mcp_adapter.py` — a reusable, tested adapter that:

1. Connects to any MCP server via stdio subprocess
2. Loads all tools as LangChain `StructuredTool` objects
3. Exposes a `read_resource(uri) -> str` helper
4. Exposes a `get_prompt(name, **kwargs) -> str` helper (returns rendered user message)
5. Handles `isError: True` by returning error strings (not raising)

```python
# mcp_adapter.py
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)


def _build_args_model(input_schema: dict[str, Any]) -> type[BaseModel]:
    """Convert JSON Schema to a Pydantic model for StructuredTool."""
    # TODO: implement — handle string, integer, number, boolean, array types
    # Required fields get ... as default; optional get None
    raise NotImplementedError


async def load_tools(session: ClientSession) -> list[StructuredTool]:
    """Load all MCP tools from a session as LangChain StructuredTools."""
    # TODO: implement
    raise NotImplementedError


async def read_resource(session: ClientSession, uri: str) -> str:
    """Read a resource URI and return its text content."""
    # TODO: implement — handle mimeType, binary blobs
    raise NotImplementedError


async def get_prompt(
    session: ClientSession, name: str, arguments: dict[str, Any]
) -> str:
    """Get a rendered prompt and return the first user message text."""
    # TODO: implement
    raise NotImplementedError


@asynccontextmanager
async def connect_stdio(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
) -> AsyncGenerator[tuple[ClientSession, list[StructuredTool]], None]:
    """Context manager: connect to an MCP server via stdio, yield (session, tools)."""
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_tools(session)
            yield session, tools
```

### Checkpoints

- [ ] `_build_args_model` handles all five JSON Schema primitive types
- [ ] `load_tools` returns one `StructuredTool` per MCP tool
- [ ] `read_resource` handles missing resources gracefully
- [ ] `get_prompt` returns the first user message text

## Part 3 — LangGraph Agent (45 min)

Build `task_agent.py` — a LangGraph agent that uses the task server.

```python
# task_agent.py
import asyncio
import logging
import os
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from mcp_adapter import connect_stdio

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(tools):
    """Build a ReAct agent with the given tools."""
    # TODO: build LangGraph agent
    # - LLM: claude-haiku-5-5, bound to tools
    # - call_llm node: invoke LLM, return {messages: [response]}
    # - should_continue: tool_calls present? → "tools" else END
    # - tool_node: ToolNode(tools)
    # - graph: START → llm → (tools → llm)* → END
    raise NotImplementedError


async def run_agent(query: str) -> str:
    """Connect to task server and answer query."""
    async with connect_stdio("python", ["task_server.py"]) as (session, tools):
        agent = build_agent(tools)
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
        return result["messages"][-1].content


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "Create a task called 'Write MCP tests' with high priority, then list all pending tasks."
    answer = asyncio.run(run_agent(query))
    print(answer)
```

### Checkpoints

- [ ] Agent connects to server, loads tools, runs a query
- [ ] Agent calls `create_task` when asked to create
- [ ] Agent calls `list_tasks` when asked to list
- [ ] Conversation ends without infinite loop

## Part 4 — Test Suite (1.5 hr)

Build `tests/test_task_server.py` and `tests/test_mcp_adapter.py`.

### test_task_server.py — Unit tests (no MCP transport)

Test the handler functions directly. All tests must pass without internet or `ANTHROPIC_API_KEY`.

```python
# tests/test_task_server.py
import pytest
import asyncio
import json

# Import handler functions directly — don't start the server
# Adjust imports once task_server.py is implemented


@pytest.mark.asyncio
async def test_create_task_returns_id():
    """create_task returns a non-empty ID string."""
    ...  # TODO


@pytest.mark.asyncio
async def test_list_tasks_shows_pending_only():
    """list_tasks with status='pending' excludes completed tasks."""
    ...  # TODO


@pytest.mark.asyncio
async def test_complete_task_updates_status():
    """complete_task marks task as completed."""
    ...  # TODO


@pytest.mark.asyncio
async def test_complete_nonexistent_task_returns_error():
    """complete_task on unknown ID returns error string, not raises."""
    ...  # TODO


@pytest.mark.asyncio
async def test_all_tasks_resource_is_valid_json():
    """tasks://all resource returns parseable JSON."""
    ...  # TODO


@pytest.mark.asyncio
async def test_prioritize_prompt_includes_context():
    """prioritize_tasks prompt embeds the context string."""
    ...  # TODO
```

### test_mcp_adapter.py — Adapter tests (mocked sessions)

```python
# tests/test_mcp_adapter.py
import pytest
from unittest.mock import AsyncMock, MagicMock


def make_mock_tool(name: str, description: str, schema: dict):
    """Helper to create a mock MCP tool object."""
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema
    return t


@pytest.mark.asyncio
async def test_load_tools_correct_names():
    """load_tools returns tools with correct names from session."""
    from mcp_adapter import load_tools
    session = AsyncMock()
    list_response = MagicMock()
    list_response.tools = [
        make_mock_tool("create_task", "Create a task", {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        })
    ]
    session.list_tools.return_value = list_response
    tools = await load_tools(session)
    assert len(tools) == 1
    assert tools[0].name == "create_task"


@pytest.mark.asyncio
async def test_read_resource_returns_text():
    """read_resource returns the text content of the resource."""
    from mcp_adapter import read_resource
    session = AsyncMock()
    content_item = MagicMock()
    content_item.text = '{"id":"abc","title":"Test"}'
    resource_result = MagicMock()
    resource_result.contents = [content_item]
    session.read_resource.return_value = resource_result
    result = await read_resource(session, "tasks://task/abc")
    assert "Test" in result


@pytest.mark.asyncio
async def test_tool_error_returns_string_not_raises():
    """Tool with isError=True returns error string; does not raise."""
    from mcp_adapter import load_tools
    session = AsyncMock()
    list_response = MagicMock()
    list_response.tools = [
        make_mock_tool("complete_task", "Complete a task", {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        })
    ]
    session.list_tools.return_value = list_response

    error_result = MagicMock()
    error_result.isError = True
    err_content = MagicMock()
    err_content.text = "Task not found"
    error_result.content = [err_content]
    session.call_tool.return_value = error_result

    tools = await load_tools(session)
    result = await tools[0].acall({"task_id": "nonexistent"})
    assert "error" in result.lower() or "not found" in result.lower()
```

### Full test run

```bash
pytest tests/ -v --tb=short
```

All tests must pass before considering Part 4 complete.

## Deliverables Checklist

- [ ] `task_server.py` — complete, runnable, all primitives implemented
- [ ] `mcp_adapter.py` — complete, no `NotImplementedError` remaining
- [ ] `task_agent.py` — complete, runnable (requires `ANTHROPIC_API_KEY`)
- [ ] `tests/test_task_server.py` — ≥ 6 passing tests
- [ ] `tests/test_mcp_adapter.py` — ≥ 3 passing tests
- [ ] All tests pass without real LLM or MCP subprocess
- [ ] No hardcoded secrets; `ANTHROPIC_API_KEY` via `os.getenv`

## Self-check

- [ ] I built a complete FastMCP server from scratch with all three primitive types
- [ ] I built a reusable adapter converting MCP tools to LangChain tools
- [ ] I connected a LangGraph agent to an MCP server via the adapter
- [ ] I wrote tests that cover tool execution, error handling, and resource reading
- [ ] All tests pass using mocked sessions — no subprocess, no LLM, no internet

---

**Supplementary**: [FastMCP docs](https://gofastmcp.com) · [pytest-asyncio](https://pytest-asyncio.readthedocs.io) · [LangGraph ToolNode](https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode)
