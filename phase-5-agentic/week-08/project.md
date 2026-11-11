# Week 08 Project — MCP Task Management System

> Build a working MCP server + client + agent for task management. Fully tested.  ·  Time: ~3 hr (Sunday session)

## Objective

By the end of this project you have a working system where a LangGraph agent manages tasks by talking to an MCP server. All critical paths are covered by tests that run without real LLM calls or MCP subprocesses.

## Deliverables

```
week-08-project/
├── task_server.py          # FastMCP server: tools + resources + prompts
├── mcp_adapter.py          # Reusable MCP → LangChain adapter
├── task_agent.py           # LangGraph agent using the MCP server
├── tests/
│   ├── __init__.py
│   ├── test_task_server.py    # Unit tests for tool/resource/prompt handlers
│   └── test_mcp_adapter.py   # Tests for adapter using mocked sessions
└── pyproject.toml          # or requirements.txt
```

## Specification

### task_server.py

**Data model**:

```python
class Task(BaseModel):
    id: str             # 8-char hex, auto-generated
    title: str
    description: str
    priority: Literal["low", "medium", "high"]
    status: Literal["pending", "completed"]
    notes: list[str]
    created_at: str     # ISO datetime
    completed_at: str | None
```

**Tools** (all must validate with Pydantic; bad input → `isError: True`, not crash):

| Tool | Args | Returns on success | Returns on error |
|---|---|---|---|
| `create_task` | `title, description, priority="medium"` | `"Created task {id}: {title}"` | validation error message |
| `complete_task` | `task_id` | `"Completed: {title}"` | `"Task {task_id} not found"` |
| `list_tasks` | `status="pending"` | formatted list with IDs and titles | `"No tasks found"` |
| `get_task_detail` | `task_id` | JSON string of Task | `"Task {task_id} not found"` |
| `add_note_to_task` | `task_id, note` | `"Note added to {title}"` | `"Task {task_id} not found"` |

**Resources**:

| URI | MIME | Content |
|---|---|---|
| `tasks://all` | `application/json` | JSON array of all Task dicts |
| `tasks://pending` | `application/json` | JSON array of pending tasks |
| `tasks://task/{task_id}` | `application/json` | Single Task JSON or `"{task_id} not found"` |
| `tasks://stats` | `application/json` | `{total, by_status: {pending, completed}, by_priority: {low, medium, high}}` |

**Prompts**:

| Prompt | Args | What the rendered message contains |
|---|---|---|
| `prioritize_tasks` | `context: str` | Lists all pending task titles + priorities; asks LLM to order them given the context |
| `daily_summary` | `date: str = "today"` | Lists all tasks (pending and completed) for the date; asks LLM to write a standup update |

### mcp_adapter.py

Implement all four public functions:

```python
def _build_args_model(input_schema: dict) -> type[BaseModel]: ...
async def load_tools(session: ClientSession) -> list[StructuredTool]: ...
async def read_resource(session: ClientSession, uri: str) -> str: ...
async def get_prompt(session: ClientSession, name: str, arguments: dict) -> str: ...

@asynccontextmanager
async def connect_stdio(command, args, env=None) -> AsyncGenerator[tuple[ClientSession, list[StructuredTool]], None]: ...
```

Contract:
- `load_tools`: tools with `isError: True` result return a string starting with `"Tool error:"`, not raise
- `read_resource`: if the session raises (unknown URI), return `"Resource not found: {uri}"`
- `get_prompt`: return the `text` of the first `user` message in the rendered prompt

### task_agent.py

```python
async def build_agent(tools: list[StructuredTool]) -> CompiledGraph:
    """ReAct agent: claude-haiku-5-5, ToolNode, loop until no tool calls."""
    ...

async def run_agent(query: str) -> str:
    """Connect to task_server.py via stdio, run query, return final message."""
    ...
```

The agent must handle multi-turn interactions (create a task, then list it in one query).

### Tests

**test_task_server.py** — minimum 6 tests, all async, no server startup:

1. `create_task` returns string with the ID
2. `complete_task` marks task completed and returns title
3. `complete_task` on unknown ID returns error string
4. `list_tasks` with `status="pending"` excludes completed tasks
5. `tasks://all` resource returns valid JSON array
6. `prioritize_tasks` prompt includes pending task titles in the output

**test_mcp_adapter.py** — minimum 4 tests with mocked `ClientSession`:

1. `load_tools` returns correct tool count and names
2. Tool invocation calls `session.call_tool` with correct args
3. `isError: True` result returns error string, does not raise
4. `read_resource` returns text content from session response

## Constraints

- Python 3.12+
- `mcp`, `langchain-anthropic`, `langgraph`, `pydantic ≥ 2.0`, `pytest`, `pytest-asyncio`
- No `print()` — use `logging`
- No hardcoded values — task store is a module-level dict, cleared between tests via fixture
- `ANTHROPIC_API_KEY` via `os.getenv("ANTHROPIC_API_KEY")` — never hardcoded
- Model ID: `claude-haiku-5-5` for the agent LLM
- All tests pass without real LLM or MCP subprocess

## Grading

Use the Week 08 Assessment rubric. For this project specifically:

| Criterion | Pass bar |
|---|---|
| All 5 tools implemented and tested | Required |
| All 4 resources implemented and tested | Required |
| At least 1 prompt implemented and tested | Required |
| Adapter functions all implemented | Required |
| Agent connects and runs a 2-step query | Required |
| ≥ 10 total tests passing | Required |
| `ruff check .` clean | Strongly encouraged |

## Extensions (optional, for extra depth)

1. **Persistence**: Replace the in-memory dict with a SQLite-backed store using `aiosqlite`. Tests can use a `:memory:` database.

2. **HTTP transport**: Add a second entry point that runs the server over SSE. Write an integration test that connects the adapter via HTTP.

3. **Multi-server agent**: Connect the task agent to a second MCP server (e.g., a simple note-taking server you build) and demonstrate the agent using both.

4. **Streaming progress**: Use `ctx.report_progress(current, total)` in a tool that processes multiple items (e.g., bulk-creating tasks from a list). Write a test that captures the progress notifications.

5. **Auth**: Add a simple Bearer token check to the HTTP SSE endpoint using a middleware wrapper. The token should come from `os.getenv("MCP_SERVER_TOKEN")`.
