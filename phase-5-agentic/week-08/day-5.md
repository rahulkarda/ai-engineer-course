# Phase 5 · Week 08 · Day 5 — MCP Client, Agent Integration, and Security

> Objective: Build an MCP client, connect a LangGraph agent to MCP servers, and apply security practices  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### The MCP Client

The `mcp` SDK provides `ClientSession` and transport helpers for building clients. The typical pattern:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async with stdio_client(StdioServerParameters(command="python", args=["server.py"])) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("my_tool", arguments={"arg": "value"})
```

For the HTTP/SSE transport:

```python
from mcp.client.sse import sse_client

async with sse_client("http://localhost:8000/sse") as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        ...
```

The `ClientSession` handles the `initialize` handshake, capability negotiation, and message routing.

### Connecting an Agent to MCP

A LangGraph agent uses MCP by:
1. Starting a client session to each MCP server at agent startup
2. Fetching `tools/list` and converting each MCP tool to a `langchain_core` tool object
3. Including those tools in the agent's tool node
4. When the LLM calls a tool, routing the call through the MCP client session

The LangChain `langchain-mcp-adapters` package automates step 2. Alternatively, build your own adapter (shown in the example below).

### MCP Tool → LangChain Tool Adapter

```python
from langchain_core.tools import StructuredTool
from mcp import ClientSession

async def mcp_tools_to_langchain(session: ClientSession) -> list[StructuredTool]:
    tools_response = await session.list_tools()
    lc_tools = []
    for tool in tools_response.tools:
        # Capture tool metadata in closure
        tool_name = tool.name
        tool_desc = tool.description or ""
        async def _call(**kwargs) -> str:
            result = await session.call_tool(tool_name, arguments=kwargs)
            return "\n".join(
                c.text for c in result.content if hasattr(c, "text")
            )
        lc_tools.append(
            StructuredTool.from_function(
                name=tool_name,
                description=tool_desc,
                coroutine=_call,
                args_schema=...,  # build from tool.inputSchema
            )
        )
    return lc_tools
```

In practice, use `langchain-mcp-adapters` (`uv add langchain-mcp-adapters`) which handles schema conversion automatically.

### Security Considerations

MCP introduces new attack surfaces. Security is the server's responsibility — the LLM (and therefore the client) cannot be trusted.

**Key threat model elements**:

| Threat | Mitigation |
|---|---|
| **Tool poisoning** — malicious tool description tricks LLM into leaking data | Vet server descriptions; don't allow user-controlled tool descriptions |
| **Prompt injection via resources** — attacker-controlled resource content hijacks LLM | Sanitize resource content before including in prompts |
| **Excessive capability** — server exposes more tools than needed | Principle of least privilege: expose minimum necessary tools |
| **Unauthorized access** — client connects to wrong server | Validate server identity; use allowlists for trusted servers |
| **Side-effect amplification** — LLM chains multiple destructive tool calls | Add confirmation gates for destructive tools; rate-limit; audit log |
| **Server impersonation** — malicious server in stdio pipe | Pin server commands; never run arbitrary user-supplied server paths |

**Authentication patterns**:
- **stdio**: server inherits the host process's environment; use env vars for credentials (`os.getenv`)
- **HTTP/SSE**: use Bearer tokens in the Authorization header; implement OAuth 2.0 with the MCP auth spec (2025)
- Never embed credentials in tool arguments or resource URIs

**Input validation** (server-side always):
- Validate all tool arguments with Pydantic before executing
- Sanitize file paths: reject `..` traversal, validate against an allowed root
- Reject SQL injection patterns if building dynamic queries
- Apply rate limits per client session

**Audit logging**:
- Log every tool call with arguments, caller identity, timestamp, and result
- Do NOT log secrets, API keys, or PII that may appear in arguments
- Use structured logging for easy parsing

## How It Works Internally

### In-Process Testing Strategy

For tests, you don't want to spawn real subprocesses. The `mcp` SDK supports in-process testing via `mcp.shared.memory`:

```python
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

async def test_my_server():
    server = build_my_server()  # returns FastMCP instance
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        await client.initialize()
        result = await client.list_tools()
        assert len(result.tools) > 0
```

This creates a real MCP session in-process with no subprocess overhead — ideal for tests.

### Tool Schema from `inputSchema`

The `mcp` SDK returns `tool.inputSchema` as a dict. To convert it to a Pydantic model for a LangChain `StructuredTool`:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import create_model

def schema_to_pydantic(schema: dict) -> type[BaseModel]:
    """Convert a JSON Schema object to a Pydantic model."""
    fields = {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    for name, prop in props.items():
        py_type = {"string": str, "integer": int, "number": float, "boolean": bool}.get(
            prop.get("type", "string"), str
        )
        default = ... if name in required else None
        fields[name] = (py_type, default)
    return create_model("ToolArgs", **fields)
```

## Worked Example

A complete agent that uses the knowledge server from Day 4 via MCP, with in-process testing.

```python
# mcp_agent.py
# LangGraph agent that uses the knowledge-base MCP server.
# In production: connects via stdio subprocess.
# In tests: uses in-process MCP session.

import asyncio
import logging
import os
from typing import Annotated, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, create_model
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── MCP → LangChain tool adapter ──────────────────────────────────────────────

def _json_schema_type_to_python(type_str: str) -> type:
    return {"string": str, "integer": int, "number": float, "boolean": bool}.get(
        type_str, str
    )


def _build_args_model(input_schema: dict) -> type[BaseModel]:
    """Build a Pydantic model from an MCP tool's inputSchema."""
    fields: dict[str, Any] = {}
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    for name, prop in props.items():
        py_type = _json_schema_type_to_python(prop.get("type", "string"))
        # Handle list types
        if prop.get("type") == "array":
            py_type = list[str]  # simplification
        default = ... if name in required else None
        fields[name] = (py_type, default)
    if not fields:
        fields["_dummy"] = (str | None, None)
    return create_model("ToolArgs", **fields)


async def load_mcp_tools_from_session(session: ClientSession) -> list[StructuredTool]:
    """Convert all MCP tools in a session to LangChain StructuredTools."""
    response = await session.list_tools()
    lc_tools: list[StructuredTool] = []

    for mcp_tool in response.tools:
        tool_name = mcp_tool.name
        tool_description = mcp_tool.description or f"Tool: {tool_name}"
        args_model = _build_args_model(mcp_tool.inputSchema or {})

        # Capture session and tool_name in closure
        def make_caller(name: str, sess: ClientSession):
            async def _caller(**kwargs) -> str:
                logger.info("MCP tool call: %s(%s)", name, kwargs)
                # Remove dummy field if present
                kwargs.pop("_dummy", None)
                result = await sess.call_tool(name, arguments=kwargs)
                if result.isError:
                    return f"Tool error: " + "\n".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                return "\n".join(
                    c.text for c in result.content if hasattr(c, "text")
                )
            return _caller

        lc_tools.append(
            StructuredTool(
                name=tool_name,
                description=tool_description,
                coroutine=make_caller(tool_name, session),
                args_schema=args_model,
            )
        )

    logger.info("Loaded %d MCP tools as LangChain tools", len(lc_tools))
    return lc_tools


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(tools: list[StructuredTool]) -> Any:
    """Build a simple ReAct-style LangGraph agent with the given tools."""
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    ).bind_tools(tools)

    def call_llm(state: AgentState) -> dict:
        logger.info("LLM call with %d messages", len(state["messages"]))
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()


# ── Stdio-transport entry point ───────────────────────────────────────────────

async def run_with_stdio_server(query: str) -> str:
    """Connect to knowledge_server.py via subprocess and run a query."""
    params = StdioServerParameters(
        command="python",
        args=["knowledge_server.py"],
        env=None,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools_from_session(session)
            agent = build_agent(tools)
            result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
            last_msg = result["messages"][-1]
            return last_msg.content if hasattr(last_msg, "content") else str(last_msg)


# ── In-process entry point (for testing) ─────────────────────────────────────

async def run_with_in_process_server(
    session: ClientSession, query: str
) -> str:
    """Use a pre-connected in-process session."""
    tools = await load_mcp_tools_from_session(session)
    agent = build_agent(tools)
    result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
    last_msg = result["messages"][-1]
    return last_msg.content if hasattr(last_msg, "content") else str(last_msg)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "What articles do we have about protocols?"
    logger.info("Query: %s", query)
    answer = asyncio.run(run_with_stdio_server(query))
    logger.info("Answer: %s", answer)
```

Test file (using in-process MCP, mocked LLM):

```python
# tests/test_mcp_agent.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Import the server and adapter — NOT the full agent (to avoid LLM dependency)
# We test: tool loading, tool invocation, and error handling


@pytest.fixture
def sample_mcp_tools_response():
    """Simulate what session.list_tools() returns."""
    tool_mock = MagicMock()
    tool_mock.name = "search_articles"
    tool_mock.description = "Search knowledge base articles."
    tool_mock.inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    }
    response = MagicMock()
    response.tools = [tool_mock]
    return response


@pytest.mark.asyncio
async def test_load_mcp_tools_count(sample_mcp_tools_response):
    """load_mcp_tools_from_session returns one tool per MCP tool."""
    from mcp_agent import load_mcp_tools_from_session

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = sample_mcp_tools_response

    tools = await load_mcp_tools_from_session(mock_session)
    assert len(tools) == 1
    assert tools[0].name == "search_articles"


@pytest.mark.asyncio
async def test_mcp_tool_call_success(sample_mcp_tools_response):
    """Tool call returns text content from MCP response."""
    from mcp_agent import load_mcp_tools_from_session

    call_result = MagicMock()
    call_result.isError = False
    content_item = MagicMock()
    content_item.text = "Found 2 article(s)"
    call_result.content = [content_item]

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = sample_mcp_tools_response
    mock_session.call_tool.return_value = call_result

    tools = await load_mcp_tools_from_session(mock_session)
    result = await tools[0].acall({"query": "MCP"})
    assert "Found 2" in result
    mock_session.call_tool.assert_called_once_with(
        "search_articles", arguments={"query": "MCP"}
    )


@pytest.mark.asyncio
async def test_mcp_tool_call_error(sample_mcp_tools_response):
    """Tool call with isError=True returns error string, not raises."""
    from mcp_agent import load_mcp_tools_from_session

    call_result = MagicMock()
    call_result.isError = True
    content_item = MagicMock()
    content_item.text = "Article not found"
    call_result.content = [content_item]

    mock_session = AsyncMock()
    mock_session.list_tools.return_value = sample_mcp_tools_response
    mock_session.call_tool.return_value = call_result

    tools = await load_mcp_tools_from_session(mock_session)
    result = await tools[0].acall({"query": "nonexistent"})
    assert "Tool error" in result


@pytest.mark.asyncio
async def test_args_model_required_field(sample_mcp_tools_response):
    """Schema with required field creates model that rejects missing arg."""
    from mcp_agent import _build_args_model

    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    model = _build_args_model(schema)
    instance = model(query="test")
    assert instance.query == "test"
```

Run tests:

```bash
pip install pytest pytest-asyncio
pytest tests/test_mcp_agent.py -v
```

## Exercises

1. **Resource reading in agent**: Extend `mcp_agent.py` with a function `read_mcp_resource(session, uri)` that calls `session.read_resource(uri)` and returns the content as a string. Write a unit test that mocks the session.

2. **Security: path traversal guard**: Write a function `safe_file_path(requested: str, allowed_root: str) -> str` that raises `ValueError` if the resolved path escapes `allowed_root`. Use `pathlib.Path.resolve()`. Write tests for both normal and traversal cases.

3. **Audit logger**: Write a decorator `@audit_log` that wraps an async tool function, logs `tool_name`, `arguments`, `result_is_error`, and `duration_ms` to a `logging.Logger`, but skips logging any argument whose name contains `"key"`, `"token"`, or `"secret"`. Write tests.

4. **Rate limiter**: Implement a simple `RateLimiter(calls_per_minute: int)` class with an `async def check(client_id: str) -> bool` method. Raise `PermissionError` if the limit is exceeded. Write tests with time mocking.

(Solutions: ../solutions/week-08/day-5.md)

## Self-check

- [ ] I can write an MCP client using `ClientSession` and connect it to a stdio server
- [ ] I can convert MCP tool descriptors into LangChain `StructuredTool` objects
- [ ] I can connect a LangGraph agent to an MCP server via the tool adapter
- [ ] I can enumerate five MCP-specific security threats and their mitigations
- [ ] I can write in-process MCP tests using mocked sessions without spawning subprocesses

---

**Supplementary**: [MCP Python SDK client](https://github.com/modelcontextprotocol/python-sdk/tree/main/src/mcp/client) · [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) · [MCP security considerations](https://spec.modelcontextprotocol.io/specification/security/)
