# Solutions — Week 08 Project

This file contains the complete reference implementation for the Week 08 project: MCP Task Management System.

For the full server and adapter implementations, see [day-6.md](./day-6.md) — the Saturday build contains the complete `task_server.py`, `mcp_adapter.py`, `task_agent.py`, and full test suites.

Below are supplementary notes, common mistakes, and the grading key.

## Common Implementation Mistakes

### 1. Forgetting to clear TASKS between tests

```python
# Wrong — tests share state
def test_create():
    result = asyncio.run(create_task(title="X", description="d"))
    assert len(TASKS) == 1

def test_list():
    # Fails if test_create ran first — TASKS has 1 item
    result = asyncio.run(list_tasks())
    assert "No tasks" in result

# Correct — use a fixture
@pytest.fixture(autouse=True)
def clear_tasks():
    TASKS.clear()
    yield
    TASKS.clear()
```

### 2. Tool closure bug — all tools call the same server

```python
# WRONG — all tools close over the last value of `tool_name`
for mcp_tool in response.tools:
    tool_name = mcp_tool.name
    async def _call(**kwargs):  # captures tool_name by reference
        result = await session.call_tool(tool_name, ...)  # always last tool_name!
    ...

# CORRECT — use a factory function to capture per-iteration value
def _make_caller(name: str, sess: ClientSession):
    async def _call(**kwargs):
        return await sess.call_tool(name, ...)
    return _call
```

### 3. Blocking in async tool handler

```python
# WRONG — blocks the event loop
@mcp.tool()
async def slow_tool(path: str) -> str:
    with open(path) as f:           # blocks
        return f.read()

# CORRECT — use asyncio.to_thread for blocking I/O
import asyncio

@mcp.tool()
async def slow_tool(path: str) -> str:
    return await asyncio.to_thread(lambda: open(path).read())
```

### 4. Writing logs to stdout in a stdio server

```python
# WRONG — corrupts the MCP stdio channel
print("Server started")  # goes to stdout = MCP channel

# CORRECT — explicitly use stderr
import logging, sys
handler = logging.StreamHandler(sys.stderr)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Server started")  # goes to stderr, not MCP channel
```

### 5. Resource URI template parameter mismatch

```python
# WRONG — template param 'task_id' doesn't match function param 'id'
@mcp.resource("tasks://task/{task_id}")
def get_task(id: str) -> str:  # parameter name must match template!
    ...

# CORRECT
@mcp.resource("tasks://task/{task_id}")
def get_task(task_id: str) -> str:
    ...
```

## Grading Key

| Criterion | Full marks | Partial marks | Zero |
|---|---|---|---|
| All 5 tools | Correct behavior, isError on bad input, logged | 3-4 tools correct | <3 tools |
| All 4 resources | Correct JSON, meaningful URIs | 2-3 resources | <2 resources |
| ≥1 prompt | Correct structure, embeds context | Prompt renders but content thin | No prompt |
| Adapter | All 4 functions impl'd, closure bug absent | 2-3 functions | 0-1 functions |
| Agent | Connects, runs 2-step query correctly | Connects but incorrect tool use | Does not connect |
| ≥10 tests | All pass, cover happy+error paths | 7-9 passing tests | <7 |
| ruff clean | 0 violations | 1-3 minor violations | Systemic issues |

**Automatic zero on**:
- Hardcoded `ANTHROPIC_API_KEY` value in any file
- `print()` used for logging in server or adapter
- Tests that require real network or real LLM to pass

## Extension Notes

### Persistence with aiosqlite

```python
# Sketch — replace in-memory TASKS dict
import aiosqlite

async def init_db(db_path: str = ":memory:") -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            priority TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            completed_at TEXT
        )
    """)
    await conn.commit()
    return conn

# In tests: pass ":memory:" to get a fresh DB each test
```

### HTTP transport entry point

```python
# In task_server.py — add to __main__:
if os.getenv("MCP_TRANSPORT") == "sse":
    mcp.run(
        transport="sse",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8001")),
    )

# In mcp_adapter.py — add connect_sse:
from mcp.client.sse import sse_client

@asynccontextmanager
async def connect_sse(url: str) -> AsyncGenerator[tuple[ClientSession, list[StructuredTool]], None]:
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_tools(session)
            yield session, tools
```
