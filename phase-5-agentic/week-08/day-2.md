# Phase 5 · Week 08 · Day 2 — MCP Architecture: Hosts, Clients, Servers, and Transports

> Objective: Map the three-role MCP architecture and understand both transport mechanisms  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### The Three Roles

MCP defines three distinct roles. A single process can occupy multiple roles, but keeping them separate is the clean mental model.

**Host**
The user-facing application. Examples: Claude Desktop, a LangGraph agent loop, an IDE plugin. The host:
- Owns the user session and context
- Spawns or connects to MCP servers
- Contains one or more MCP clients
- Enforces security policy (which servers are trusted, which capabilities are allowed)
- Decides what context to inject into LLM prompts from server results

**Client**
A component inside the host. Each client maintains exactly one 1:1 connection to exactly one server. The client:
- Manages the JSON-RPC session lifecycle (initialize, operate, shutdown)
- Translates host requests ("run this tool") into MCP protocol messages
- Receives responses and server-initiated notifications
- Does NOT make routing decisions — the host does

**Server**
An independent process (or network service) that exposes capabilities. The server:
- Declares capabilities at initialization
- Responds to requests: tools/list, tools/call, resources/list, resources/read, prompts/list, prompts/get
- May initiate notifications (e.g., resource changed, log messages, progress)
- Has no knowledge of how many clients are connected or what the host application does with results

### Topology

A host typically manages several clients simultaneously:

```
Host Application
├── MCP Client A  ──(stdio)──► Server: filesystem
├── MCP Client B  ──(stdio)──► Server: git
├── MCP Client C  ──(HTTP)──►  Server: remote weather API
└── MCP Client D  ──(stdio)──► Server: custom Postgres tools
```

Each connection is independent. Servers don't know about each other. The host decides how to combine results from multiple servers when constructing an LLM prompt.

### Transport 1: stdio

The server is launched as a **subprocess** of the host. Communication happens over the process's **stdin/stdout** pipes.

```
Host Process
│
├── spawn subprocess: python my_server.py
│     │
│     └── child process I/O
│          stdin  ◄─── host writes JSON-RPC messages (newline-delimited)
│          stdout ───► host reads JSON-RPC responses/notifications
│          stderr ───► host captures for logging (not part of protocol)
```

Properties:
- Single tenant: one client per server process
- No network required: ideal for local tools
- Process lifecycle tied to host: server exits when host kills it
- Secure by default: no network exposure

Message framing: messages are newline-delimited JSON. Each message is one JSON object, one line.

### Transport 2: HTTP + SSE (Server-Sent Events)

The server is a network service. Two channels:

1. **Client → Server**: standard HTTP POST requests to a `/messages` endpoint
2. **Server → Client**: HTTP GET to a `/sse` endpoint, which the server keeps open and streams newline-delimited `data:` events

```
Host / Client                       MCP Server (HTTP)
     │                                    │
     │──GET /sse ──────────────────────►  │
     │◄─────────── SSE stream (open) ─────│
     │                                    │
     │──POST /messages (request) ───────► │
     │◄─────────── SSE event (response) ──│
     │                                    │
     │──POST /messages (another req) ───► │
     │◄─────────── SSE event ─────────────│
```

Properties:
- Multi-tenant capable: many clients can connect to one server process
- Network-accessible: deploy once, use from anywhere
- Firewall-friendly: only outbound HTTP from client perspective
- Slightly more complex: must correlate POST request IDs to SSE events

### Transport 3: Streamable HTTP (2025)

The updated MCP spec introduces **Streamable HTTP** — a single endpoint that handles both requests and streaming responses over one HTTP connection using chunked transfer encoding or SSE on the response stream. This simplifies the two-endpoint design. Check `spec.modelcontextprotocol.io` for the current state.

## How It Works Internally

### Session State Machine

```
UNCONNECTED
    │ connect()
    ▼
INITIALIZING ──► send initialize request
    │ receive initialize response
    ▼
INITIALIZED ──► send initialized notification
    │
    ▼
OPERATING  ◄──► tools/call, resources/read, etc.
    │ shutdown signal
    ▼
CLOSED
```

The client tracks this state. Any request sent in the wrong state is a protocol error.

### Message Framing Details (stdio)

Each JSON-RPC message is a single JSON object followed by `\n`. The reader accumulates bytes until it sees `\n`, then parses the JSON. No length prefix, no delimiter escaping. The constraint: JSON values must not contain raw newlines (they should be `\n` escape sequences inside strings, which JSON requires anyway).

```python
# Pseudocode for reading messages from a subprocess stdout
import json

def read_messages(stdout_stream):
    for line in stdout_stream:  # line ends with \n
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        yield msg
```

### SSE Message Format

SSE is a simple HTTP streaming format:

```
data: {"jsonrpc":"2.0","id":1,"result":{...}}\n\n
```

Each event is `data: <payload>\n\n`. The `\n\n` (blank line) terminates the event. The client reads the response stream and splits on blank lines.

### Concurrency Model

For stdio transport with the `mcp` Python SDK:
- The SDK runs the server in an async event loop (`asyncio`)
- Requests from the client are dispatched concurrently using `asyncio.create_task`
- Long-running tools should use `asyncio.to_thread` for blocking I/O so they don't block other requests

## Worked Example

This example demonstrates the architectural concepts by building a minimal in-process simulation — no real transport yet, just demonstrating how host, client, and server roles compose.

```python
# mcp_architecture_sim.py
# Simulates host/client/server role separation without a real transport.
# Shows how the host orchestrates multiple clients, each connected to one server.

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ── Server role ──────────────────────────────────────────────────────────────

@dataclass
class MockMCPServer:
    """Represents an MCP server's capability registry."""

    name: str
    tools: dict[str, Any] = field(default_factory=dict)

    def register_tool(
        self,
        name: str,
        description: str,
        handler,  # callable
    ) -> None:
        self.tools[name] = {"description": description, "handler": handler}

    async def handle_tools_list(self) -> list[dict[str, Any]]:
        return [
            {"name": k, "description": v["description"]}
            for k, v in self.tools.items()
        ]

    async def handle_tools_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name not in self.tools:
            return {"isError": True, "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}
        result = await self.tools[tool_name]["handler"](**arguments)
        return {"isError": False, "content": [{"type": "text", "text": str(result)}]}


# ── Client role ───────────────────────────────────────────────────────────────

class MockMCPClient:
    """One-to-one connection to one server."""

    def __init__(self, server: MockMCPServer) -> None:
        self._server = server
        self._logger = logging.getLogger(f"client.{server.name}")

    async def list_tools(self) -> list[dict[str, Any]]:
        self._logger.info("Requesting tools/list from %s", self._server.name)
        tools = await self._server.handle_tools_list()
        self._logger.info("Received %d tools", len(tools))
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self._logger.info("Calling %s/%s with %s", self._server.name, tool_name, arguments)
        result = await self._server.handle_tools_call(tool_name, arguments)
        self._logger.info("Result isError=%s", result["isError"])
        return result


# ── Host role ─────────────────────────────────────────────────────────────────

class MockMCPHost:
    """Manages multiple clients, enforces policy, builds agent context."""

    def __init__(self) -> None:
        self._clients: dict[str, MockMCPClient] = {}
        self._logger = logging.getLogger("host")

    def add_server(self, server: MockMCPServer) -> None:
        """Host creates a client for each server it trusts."""
        client = MockMCPClient(server)
        self._clients[server.name] = client
        self._logger.info("Registered server: %s", server.name)

    async def discover_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        """Aggregate tools from all connected servers."""
        all_tools: dict[str, list[dict[str, Any]]] = {}
        for server_name, client in self._clients.items():
            all_tools[server_name] = await client.list_tools()
        return all_tools

    async def invoke_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a tool call to the appropriate client."""
        if server_name not in self._clients:
            raise ValueError(f"No client for server: {server_name}")
        return await self._clients[server_name].call_tool(tool_name, arguments)


# ── Concrete servers ──────────────────────────────────────────────────────────

def build_weather_server() -> MockMCPServer:
    server = MockMCPServer(name="weather")

    async def get_weather(city: str) -> str:
        # In real life: call a weather API
        return f"{city}: 18°C, clear"

    server.register_tool(
        "get_weather",
        "Get current weather for a city.",
        get_weather,
    )
    return server


def build_calculator_server() -> MockMCPServer:
    server = MockMCPServer(name="calculator")

    async def add(a: float, b: float) -> float:
        return a + b

    async def multiply(a: float, b: float) -> float:
        return a * b

    server.register_tool("add", "Add two numbers.", add)
    server.register_tool("multiply", "Multiply two numbers.", multiply)
    return server


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger = logging.getLogger("main")

    # Build host with two servers
    host = MockMCPHost()
    host.add_server(build_weather_server())
    host.add_server(build_calculator_server())

    # Discover all available tools (as a real host would do at startup)
    all_tools = await host.discover_all_tools()
    for server_name, tools in all_tools.items():
        logger.info("Server '%s' exposes: %s", server_name, [t["name"] for t in tools])

    # Invoke tools via host (simulating agent requests)
    weather = await host.invoke_tool("weather", "get_weather", {"city": "Tokyo"})
    logger.info("Weather result: %s", weather["content"][0]["text"])

    calc = await host.invoke_tool("calculator", "add", {"a": 3.5, "b": 2.1})
    logger.info("Calc result: %s", calc["content"][0]["text"])

    # Simulate error: wrong server name
    try:
        await host.invoke_tool("nonexistent", "something", {})
    except ValueError as exc:
        logger.warning("Expected error: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
python mcp_architecture_sim.py
```

Study the output. The host routes, clients translate, servers execute. Nothing crosses these boundaries — that isolation is the core of MCP's design.

## Exercises

1. **Transport comparison**: Write a markdown comment block listing three scenarios where stdio transport is the better choice and three where HTTP+SSE is better. Justify each.

2. **Multi-client host**: Extend `MockMCPHost` with a method `list_all_tool_names() -> list[str]` that returns a flat list of all tool names across all servers, prefixed with the server name (e.g., `"weather.get_weather"`). Write a test for this method using `pytest` and `asyncio`.

3. **Client state machine**: Add a `_state` field to `MockMCPClient` that tracks `"uninitialized" | "ready" | "closed"`. Make `list_tools` and `call_tool` raise `RuntimeError` if called in the wrong state. Add `connect()` and `close()` methods that transition state. Write tests for the state transitions.

4. **Error routing**: Modify `MockMCPHost.invoke_tool` to catch errors from the server's `isError: True` response and raise a `ToolCallError(tool_name, message)` exception. Write tests.

(Solutions: ../solutions/week-08/day-2.md)

## Self-check

- [ ] I can name the three MCP roles and state the single most important constraint of each
- [ ] I can draw the host → client → server topology for a multi-server setup
- [ ] I can describe how messages are framed over stdio transport
- [ ] I can explain why SSE needs two channels (POST + SSE stream) while stdio uses one
- [ ] I can explain the MCP session state machine (unconnected → initializing → operating → closed)

---

**Supplementary**: [MCP architecture spec](https://spec.modelcontextprotocol.io/specification/architecture/) · [Transports](https://spec.modelcontextprotocol.io/specification/basic/transports/)
