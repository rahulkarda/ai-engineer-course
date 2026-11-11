# Solutions — Week 08 · Day 2

## Exercise 1 — Transport comparison

```
stdio is better when:
1. Local dev tool — server runs on the same machine as the host; no need for networking overhead or firewall config.
2. Single-tenant desktop app — Claude Desktop spawning per-user tool servers; each user gets their own process, no sharing/isolation worries.
3. Security-sensitive tools — filesystem access, SSH keys, credentials from env; no network exposure, no auth tokens to manage.

HTTP+SSE is better when:
1. Shared remote service — a company-wide knowledge base MCP server; many users connect to one server, cheaper to run one process.
2. Serverless/cloud deployment — server runs in a container or Lambda; stdin/stdout are not available in those runtimes.
3. Tool that requires its own auth/scaling — a payment processor MCP server that needs to be independently deployed, scaled, and monitored.
```

## Exercise 2 — list_all_tool_names + test

```python
# In MockMCPHost:

async def list_all_tool_names(self) -> list[str]:
    """Return prefixed tool names: 'server_name.tool_name'."""
    names: list[str] = []
    for server_name, client in self._clients.items():
        tools = await client.list_tools()
        for tool in tools:
            names.append(f"{server_name}.{tool['name']}")
    return sorted(names)
```

```python
# tests/test_host_tools.py
import pytest
import asyncio
from mcp_architecture_sim import MockMCPHost, build_weather_server, build_calculator_server


@pytest.mark.asyncio
async def test_list_all_tool_names_format():
    host = MockMCPHost()
    host.add_server(build_weather_server())
    host.add_server(build_calculator_server())

    names = await host.list_all_tool_names()

    assert "weather.get_weather" in names
    assert "calculator.add" in names
    assert "calculator.multiply" in names
    # All names are prefixed
    assert all("." in name for name in names)


@pytest.mark.asyncio
async def test_list_all_tool_names_empty_host():
    host = MockMCPHost()
    names = await host.list_all_tool_names()
    assert names == []
```

## Exercise 3 — Client state machine

```python
from dataclasses import dataclass, field
from typing import Literal

ClientState = Literal["uninitialized", "ready", "closed"]


@dataclass
class StatefulMockMCPClient:
    server: MockMCPServer
    _state: ClientState = field(default="uninitialized", init=False)

    def _require_state(self, *states: ClientState) -> None:
        if self._state not in states:
            raise RuntimeError(
                f"Operation not allowed in state '{self._state}'. "
                f"Required: {states}"
            )

    async def connect(self) -> None:
        self._require_state("uninitialized")
        self._state = "ready"

    async def close(self) -> None:
        self._require_state("ready")
        self._state = "closed"

    async def list_tools(self) -> list[dict]:
        self._require_state("ready")
        return await self.server.handle_tools_list()

    async def call_tool(self, name: str, args: dict) -> dict:
        self._require_state("ready")
        return await self.server.handle_tools_call(name, args)
```

```python
# tests/test_client_state.py
import pytest
import asyncio
from mcp_architecture_sim import MockMCPServer, build_weather_server


def make_client(server):
    from mcp_architecture_sim import StatefulMockMCPClient
    return StatefulMockMCPClient(server=server)


@pytest.mark.asyncio
async def test_list_tools_before_connect_raises():
    client = make_client(build_weather_server())
    with pytest.raises(RuntimeError, match="uninitialized"):
        await client.list_tools()


@pytest.mark.asyncio
async def test_list_tools_after_connect_succeeds():
    client = make_client(build_weather_server())
    await client.connect()
    tools = await client.list_tools()
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_call_tool_after_close_raises():
    client = make_client(build_weather_server())
    await client.connect()
    await client.close()
    with pytest.raises(RuntimeError, match="closed"):
        await client.call_tool("get_weather", {"city": "Berlin"})


@pytest.mark.asyncio
async def test_connect_twice_raises():
    client = make_client(build_weather_server())
    await client.connect()
    with pytest.raises(RuntimeError):
        await client.connect()
```

## Exercise 4 — ToolCallError

```python
class ToolCallError(Exception):
    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"Tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name
        self.message = message


# In MockMCPHost:
async def invoke_tool(self, server_name: str, tool_name: str, arguments: dict) -> dict:
    if server_name not in self._clients:
        raise ValueError(f"No client for server: {server_name}")
    result = await self._clients[server_name].call_tool(tool_name, arguments)
    if result.get("isError"):
        msg = result["content"][0]["text"] if result.get("content") else "Unknown error"
        raise ToolCallError(tool_name, msg)
    return result
```

```python
@pytest.mark.asyncio
async def test_invoke_tool_raises_on_error():
    from mcp_architecture_sim import MockMCPHost, MockMCPServer, ToolCallError

    server = MockMCPServer(name="test")

    async def failing_tool(**kwargs):
        # Return isError result
        raise ValueError("deliberate failure")

    # Patch handle_tools_call to return isError
    async def bad_call(name, args):
        return {"isError": True, "content": [{"type": "text", "text": "deliberate failure"}]}

    server.handle_tools_call = bad_call
    server.register_tool("bad_tool", "fails", lambda: None)

    host = MockMCPHost()
    host.add_server(server)

    with pytest.raises(ToolCallError) as exc_info:
        await host.invoke_tool("test", "bad_tool", {})

    assert exc_info.value.tool_name == "bad_tool"
    assert "deliberate failure" in str(exc_info.value)
```
