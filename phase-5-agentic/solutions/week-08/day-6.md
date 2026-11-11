# Solutions — Week 08 · Day 6

This file contains complete reference implementations for all four parts of the Saturday build.

## Part 1 — task_server.py (complete)

```python
# task_server.py
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Literal

from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel, Field, field_validator

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


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def create_task(
    title: str,
    description: str,
    priority: Literal["low", "medium", "high"] = "medium",
    ctx: Context = None,
) -> str:
    """Create a new task with a title, description, and priority level.

    Returns the new task ID and title on success.
    """
    if not title.strip():
        return "Error: title cannot be empty."
    if not description.strip():
        return "Error: description cannot be empty."

    task = Task(title=title.strip(), description=description.strip(), priority=priority)
    TASKS[task.id] = task
    logger.info("Task created: %s (priority=%s)", task.id, priority)
    if ctx:
        await ctx.info(f"Task created: {task.id}")
    return f"Created task {task.id}: {task.title}"


@mcp.tool()
async def complete_task(task_id: str, ctx: Context = None) -> str:
    """Mark a task as completed by its ID.

    Returns the task title on success, or an error if the task does not exist.
    """
    if task_id not in TASKS:
        return f"Task {task_id!r} not found."
    task = TASKS[task_id]
    if task.status == "completed":
        return f"Task {task_id!r} is already completed."
    task.status = "completed"
    task.completed_at = datetime.now().isoformat()
    logger.info("Task completed: %s", task_id)
    if ctx:
        await ctx.info(f"Task completed: {task_id}")
    return f"Completed: {task.title}"


@mcp.tool()
async def list_tasks(
    status: Literal["pending", "completed", "all"] = "pending",
    ctx: Context = None,
) -> str:
    """List tasks filtered by status.

    Returns a formatted list of matching tasks with IDs, titles, and priorities.
    """
    if status == "all":
        tasks = list(TASKS.values())
    else:
        tasks = [t for t in TASKS.values() if t.status == status]

    if not tasks:
        return f"No tasks found with status={status!r}."

    lines = [f"Tasks ({status}):"]
    for t in sorted(tasks, key=lambda x: x.created_at):
        lines.append(f"  [{t.id}] [{t.priority.upper()}] {t.title} — {t.status}")
    return "\n".join(lines)


@mcp.tool()
async def get_task_detail(task_id: str, ctx: Context = None) -> str:
    """Get the full details of a task as JSON, including notes and timestamps."""
    if task_id not in TASKS:
        return f"Task {task_id!r} not found."
    task = TASKS[task_id]
    return task.model_dump_json(indent=2)


@mcp.tool()
async def add_note_to_task(task_id: str, note: str, ctx: Context = None) -> str:
    """Add a text note to an existing task.

    Returns success message or error if task not found.
    """
    if task_id not in TASKS:
        return f"Task {task_id!r} not found."
    if not note.strip():
        return "Error: note cannot be empty."
    TASKS[task_id].notes.append(note.strip())
    logger.info("Note added to task %s", task_id)
    return f"Note added to '{TASKS[task_id].title}'."


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("tasks://all", mime_type="application/json")
def get_all_tasks() -> str:
    """JSON array of all tasks."""
    return json.dumps([t.model_dump() for t in TASKS.values()], indent=2)


@mcp.resource("tasks://pending", mime_type="application/json")
def get_pending_tasks() -> str:
    """JSON array of all pending tasks."""
    pending = [t.model_dump() for t in TASKS.values() if t.status == "pending"]
    return json.dumps(pending, indent=2)


@mcp.resource("tasks://task/{task_id}", mime_type="application/json")
def get_task_resource(task_id: str) -> str:
    """JSON of a single task by ID."""
    if task_id not in TASKS:
        return json.dumps({"error": f"Task {task_id!r} not found."})
    return TASKS[task_id].model_dump_json(indent=2)


@mcp.resource("tasks://stats", mime_type="application/json")
def get_task_stats() -> str:
    """JSON summary: total count, breakdown by status and priority."""
    by_status: dict[str, int] = {"pending": 0, "completed": 0}
    by_priority: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    for t in TASKS.values():
        by_status[t.status] = by_status.get(t.status, 0) + 1
        by_priority[t.priority] = by_priority.get(t.priority, 0) + 1
    return json.dumps({
        "total": len(TASKS),
        "by_status": by_status,
        "by_priority": by_priority,
    }, indent=2)


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def prioritize_tasks(context: str) -> str:
    """Generate a prompt asking the LLM to prioritize pending tasks.

    Args:
        context: Current context or constraints (deadline, team size, etc.)
    """
    pending = [t for t in TASKS.values() if t.status == "pending"]
    if not pending:
        return f"No pending tasks to prioritize. Context: {context}"

    task_list = "\n".join(
        f"  - [{t.id}] {t.title} (priority={t.priority})"
        for t in pending
    )
    return (
        f"You are a project manager assistant. Given the following context and "
        f"list of pending tasks, suggest the optimal order to work on them.\n\n"
        f"Context: {context}\n\n"
        f"Pending tasks:\n{task_list}\n\n"
        f"Explain your ordering rationale briefly."
    )


@mcp.prompt()
def daily_summary(date: str = "today") -> str:
    """Generate a daily standup summary prompt.

    Args:
        date: Date to summarize (default: 'today').
    """
    completed = [t for t in TASKS.values() if t.status == "completed"]
    pending = [t for t in TASKS.values() if t.status == "pending"]

    completed_list = "\n".join(
        f"  - {t.title}" for t in completed
    ) or "  (none)"
    pending_list = "\n".join(
        f"  - [{t.priority.upper()}] {t.title}" for t in pending
    ) or "  (none)"

    return (
        f"Write a concise daily standup update for {date}.\n\n"
        f"Completed:\n{completed_list}\n\n"
        f"In progress / pending:\n{pending_list}\n\n"
        f"Format: What did we do? What are we doing next? Any blockers?"
    )


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "sse":
        mcp.run(transport="sse", host="127.0.0.1", port=int(os.getenv("MCP_PORT", "8001")))
    else:
        mcp.run(transport="stdio")
```

## Part 2 — mcp_adapter.py (complete)

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

_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
}


def _build_args_model(input_schema: dict[str, Any]) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    for name, prop in props.items():
        json_type = prop.get("type", "string")
        if json_type == "array":
            item_type = prop.get("items", {}).get("type", "string")
            py_type = list[_JSON_TYPE_MAP.get(item_type, str)]
        else:
            py_type = _JSON_TYPE_MAP.get(json_type, str)
        default = ... if name in required else None
        fields[name] = (py_type, default)
    if not fields:
        # Empty schema — create a sentinel field so create_model doesn't fail
        fields["_no_args"] = (str | None, None)
    return create_model("ToolArgs", **fields)


async def load_tools(session: ClientSession) -> list[StructuredTool]:
    response = await session.list_tools()
    lc_tools: list[StructuredTool] = []
    for mcp_tool in response.tools:
        tool_name = mcp_tool.name
        tool_desc = mcp_tool.description or f"MCP tool: {tool_name}"
        args_model = _build_args_model(mcp_tool.inputSchema or {})

        def _make_caller(name: str, sess: ClientSession):
            async def _call(**kwargs) -> str:
                kwargs.pop("_no_args", None)
                logger.info("MCP tool call: %s(%s)", name, list(kwargs.keys()))
                result = await sess.call_tool(name, arguments=kwargs)
                texts = [
                    c.text for c in result.content if hasattr(c, "text")
                ]
                if result.isError:
                    return "Tool error: " + "\n".join(texts)
                return "\n".join(texts)
            return _call

        lc_tools.append(
            StructuredTool(
                name=tool_name,
                description=tool_desc,
                coroutine=_make_caller(tool_name, session),
                args_schema=args_model,
            )
        )
    logger.info("Loaded %d tools from MCP session", len(lc_tools))
    return lc_tools


async def read_resource(session: ClientSession, uri: str) -> str:
    try:
        result = await session.read_resource(uri)
        texts = [c.text for c in result.contents if hasattr(c, "text") and c.text]
        return "\n".join(texts) if texts else "[empty resource]"
    except Exception as exc:
        logger.warning("Resource read failed for %s: %s", uri, exc)
        return f"Resource not found: {uri}"


async def get_prompt(
    session: ClientSession, name: str, arguments: dict[str, Any]
) -> str:
    result = await session.get_prompt(name, arguments=arguments)
    for msg in result.messages:
        if msg.role == "user":
            content = msg.content
            if hasattr(content, "text"):
                return content.text
    return "[no user message in prompt]"


@asynccontextmanager
async def connect_stdio(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
) -> AsyncGenerator[tuple[ClientSession, list[StructuredTool]], None]:
    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_tools(session)
            yield session, tools
```

## Part 3 — task_agent.py (complete)

```python
# task_agent.py
import asyncio
import logging
import os
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from mcp_adapter import connect_stdio

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_agent(tools: list[StructuredTool]) -> Any:
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    ).bind_tools(tools)

    def call_llm(state: AgentState) -> dict:
        logger.info("LLM node: %d messages", len(state["messages"]))
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()


async def run_agent(query: str) -> str:
    async with connect_stdio("python", ["task_server.py"]) as (session, tools):
        agent = build_agent(tools)
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
        last = result["messages"][-1]
        return last.content if hasattr(last, "content") else str(last)


if __name__ == "__main__":
    query = (
        "Create a task called 'Write MCP tests' with description "
        "'Cover all tools and resources' and high priority. "
        "Then list all pending tasks."
    )
    answer = asyncio.run(run_agent(query))
    print(answer)
```

## Part 4 — Tests (complete)

```python
# tests/test_task_server.py
import json
import pytest
from task_server import (
    TASKS,
    Task,
    create_task,
    complete_task,
    list_tasks,
    get_task_detail,
    add_note_to_task,
    get_all_tasks,
    get_pending_tasks,
    get_task_resource,
    get_task_stats,
    prioritize_tasks,
    daily_summary,
)


@pytest.fixture(autouse=True)
def clear_tasks():
    TASKS.clear()
    yield
    TASKS.clear()


@pytest.mark.asyncio
async def test_create_task_returns_id():
    result = await create_task(title="Test task", description="A description")
    assert "Test task" in result
    assert len(TASKS) == 1


@pytest.mark.asyncio
async def test_create_task_empty_title_returns_error():
    result = await create_task(title="  ", description="desc")
    assert "error" in result.lower()
    assert len(TASKS) == 0


@pytest.mark.asyncio
async def test_complete_task_success():
    await create_task(title="Task A", description="desc")
    task_id = list(TASKS.keys())[0]
    result = await complete_task(task_id=task_id)
    assert "Task A" in result
    assert TASKS[task_id].status == "completed"


@pytest.mark.asyncio
async def test_complete_nonexistent_task_returns_error():
    result = await complete_task(task_id="does-not-exist")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_list_tasks_pending_only():
    await create_task(title="Pending", description="d")
    await create_task(title="To Complete", description="d")
    task_ids = list(TASKS.keys())
    await complete_task(task_id=task_ids[1])

    result = await list_tasks(status="pending")
    assert "Pending" in result
    assert "To Complete" not in result


@pytest.mark.asyncio
async def test_add_note_to_task():
    await create_task(title="Noted task", description="d")
    task_id = list(TASKS.keys())[0]
    result = await add_note_to_task(task_id=task_id, note="Important note")
    assert "Noted task" in result
    assert "Important note" in TASKS[task_id].notes


def test_all_tasks_resource_is_valid_json():
    TASKS["t1"] = Task(title="T1", description="d")
    result = get_all_tasks()
    data = json.loads(result)
    assert isinstance(data, list)
    assert data[0]["title"] == "T1"


def test_stats_resource_counts_correctly():
    TASKS["t1"] = Task(title="T1", description="d", priority="high")
    TASKS["t2"] = Task(title="T2", description="d", priority="low", status="completed")
    result = get_task_stats()
    data = json.loads(result)
    assert data["total"] == 2
    assert data["by_status"]["pending"] == 1
    assert data["by_status"]["completed"] == 1
    assert data["by_priority"]["high"] == 1


def test_prioritize_prompt_includes_task_titles():
    TASKS["t1"] = Task(title="Critical bug fix", description="d", priority="high")
    TASKS["t2"] = Task(title="Write docs", description="d", priority="low")
    result = prioritize_tasks(context="release tomorrow")
    assert "Critical bug fix" in result
    assert "release tomorrow" in result
```

```python
# tests/test_mcp_adapter.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from mcp_adapter import _build_args_model, load_tools, read_resource, get_prompt


def make_tool(name, desc, schema):
    t = MagicMock()
    t.name = name
    t.description = desc
    t.inputSchema = schema
    return t


@pytest.mark.asyncio
async def test_load_tools_names():
    session = AsyncMock()
    resp = MagicMock()
    resp.tools = [
        make_tool("create_task", "Create a task", {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        })
    ]
    session.list_tools.return_value = resp
    tools = await load_tools(session)
    assert len(tools) == 1
    assert tools[0].name == "create_task"


@pytest.mark.asyncio
async def test_load_tools_calls_session():
    session = AsyncMock()
    resp = MagicMock()
    resp.tools = []
    session.list_tools.return_value = resp

    call_result = MagicMock()
    call_result.isError = False
    content = MagicMock()
    content.text = "Created task abc"
    call_result.content = [content]
    session.call_tool.return_value = call_result

    tools = await load_tools(session)
    assert tools == []


@pytest.mark.asyncio
async def test_tool_error_returns_string():
    session = AsyncMock()
    resp = MagicMock()
    resp.tools = [make_tool("bad_tool", "fails", {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    })]
    session.list_tools.return_value = resp

    err = MagicMock()
    err.isError = True
    c = MagicMock()
    c.text = "not found"
    err.content = [c]
    session.call_tool.return_value = err

    tools = await load_tools(session)
    result = await tools[0].acall({"id": "x"})
    assert "error" in result.lower() or "not found" in result.lower()


@pytest.mark.asyncio
async def test_read_resource_returns_text():
    session = AsyncMock()
    item = MagicMock()
    item.text = "task data"
    result_mock = MagicMock()
    result_mock.contents = [item]
    session.read_resource.return_value = result_mock

    text = await read_resource(session, "tasks://task/abc")
    assert "task data" in text


def test_build_args_model_required_field():
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    Model = _build_args_model(schema)
    instance = Model(title="hello")
    assert instance.title == "hello"
```
