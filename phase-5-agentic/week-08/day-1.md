# Phase 5 · Week 08 · Day 1 — What is MCP and Why It Exists

> Objective: Understand the problem MCP solves and where it fits in the AI ecosystem  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-7]]

## Concept

### The Integration Problem Before MCP

Every AI application that needs to use tools — search, databases, code execution, APIs — used to wire those tools directly into the application. Each LLM provider had its own tool/function-calling schema. Each agent framework invented its own plugin system. The result:

- A tool written for LangChain couldn't be reused in a Claude application without rewriting.
- Every new data source required bespoke connector code per application.
- Security, auth, and transport were reinvented in each integration.

This is the **M×N integration problem**: M applications × N tools = M×N bespoke connectors.

### What MCP Is

**Model Context Protocol (MCP)** is an open protocol (introduced by Anthropic, November 2024) that defines a standard interface between AI applications (hosts/clients) and capability providers (servers). It reduces M×N to M+N:

- Write a tool or data source once as an **MCP server**.
- Any MCP-compatible host (Claude Desktop, a LangGraph agent, VS Code, etc.) can use it without modification.

MCP is transport-agnostic, language-agnostic, and designed for both local (subprocess) and remote (HTTP) deployments.

### Why It Matters

| Without MCP | With MCP |
|---|---|
| Tool code tied to framework | Tool code stands alone |
| Auth/transport per integration | Standardized negotiation |
| Hard to share tools across teams | Publish once, use everywhere |
| Security posture inconsistent | Security model is part of spec |

MCP's position in the stack:

```
┌─────────────────────────────────────┐
│        AI Host Application          │  ← Claude Desktop, your agent, IDE plugin
│  (contains MCP Client)              │
└─────────────────┬───────────────────┘
                  │ MCP (JSON-RPC 2.0 over stdio / HTTP+SSE)
        ┌─────────┴──────────┐
        │                    │
  ┌─────▼─────┐       ┌──────▼──────┐
  │ MCP Server│       │ MCP Server  │
  │ (tools)   │       │ (resources) │
  └─────┬─────┘       └──────┬──────┘
        │                    │
  ┌─────▼─────┐       ┌──────▼──────┐
  │ Database  │       │  File system│
  └───────────┘       └─────────────┘
```

### The MCP Ecosystem (2024-2025)

**Official integrations** ship with Claude Desktop: filesystem, fetch, git, GitHub, Google Drive, Postgres, Slack, and more.

**Community ecosystem**: hundreds of published servers at `modelcontextprotocol.io/servers` covering everything from Kubernetes to Stripe to Jira.

**IDE integration**: VS Code Copilot, JetBrains, Cursor — all adding MCP support for agents that can reach your workspace.

The protocol is governed by the MCP specification at `spec.modelcontextprotocol.io`. The reference Python SDK is `mcp` (installable via `uv add mcp`).

## How It Works Internally

### Protocol Foundations

MCP messages are **JSON-RPC 2.0**. Every message is either:

- A **request**: `{ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {...} }`
- A **response**: `{ "jsonrpc": "2.0", "id": 1, "result": {...} }` or `{ "error": {...} }`
- A **notification**: like a request but no `id`, no response expected

Session lifecycle:

1. **initialize** — client sends capabilities, server responds with its capabilities, protocol version negotiated
2. **initialized** (notification) — client confirms
3. **Normal operation** — tools/list, tools/call, resources/list, resources/read, prompts/list, prompts/get, sampling requests
4. **Shutdown** — clean teardown

The protocol is **stateful**: a session has negotiated capabilities, server-side context (like an open database connection), and the server can send notifications back to the client (e.g., resource changed, progress updates).

### Capability Negotiation

During `initialize`, client declares what it supports (e.g., `sampling`, `roots`, `experimental`) and server declares its capabilities (e.g., `tools`, `resources`, `prompts`, `logging`). This allows older clients to work with newer servers and vice versa — unknown capabilities are ignored.

### Transport Layer

Two official transports:

- **stdio**: server is a subprocess, communication over stdin/stdout. Ideal for local tools.
- **HTTP + SSE (Server-Sent Events)**: server is a network service, client sends requests via HTTP POST, server streams responses/notifications via SSE. Ideal for remote/shared services.

A third transport (**Streamable HTTP**, 2025 revision) replaces the SSE transport for bidirectional streaming over a single HTTP connection — check the latest spec.

## Worked Example

No server to run yet — this example shows what raw MCP messages look like so you understand the protocol before abstracting it with the SDK.

```python
# mcp_protocol_demo.py
# Shows raw JSON-RPC message shapes used by MCP (no network I/O — educational only)

import json
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def make_request(method: str, params: dict[str, Any], request_id: int = 1) -> str:
    """Build a JSON-RPC 2.0 request string."""
    msg = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    return json.dumps(msg, indent=2)


def make_response(result: dict[str, Any], request_id: int = 1) -> str:
    """Build a JSON-RPC 2.0 success response string."""
    msg = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }
    return json.dumps(msg, indent=2)


def make_error_response(
    code: int, message: str, request_id: int = 1
) -> str:
    """Build a JSON-RPC 2.0 error response string."""
    msg = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
    return json.dumps(msg, indent=2)


def demo_initialize_handshake() -> None:
    """Show what the initialize request/response looks like."""
    logger.info("=== MCP Initialize Handshake ===")

    # Client -> Server
    init_request = make_request(
        method="initialize",
        params={
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "sampling": {},       # client can handle sampling requests
                "roots": {"listChanged": True},
            },
            "clientInfo": {"name": "demo-client", "version": "0.1.0"},
        },
    )
    logger.info("CLIENT -> SERVER:\n%s", init_request)

    # Server -> Client
    init_response = make_response(
        result={
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": False, "listChanged": True},
                "prompts": {"listChanged": False},
                "logging": {},
            },
            "serverInfo": {"name": "demo-server", "version": "1.0.0"},
        }
    )
    logger.info("SERVER -> CLIENT:\n%s", init_response)


def demo_tools_list() -> None:
    """Show tools/list exchange."""
    logger.info("\n=== tools/list ===")

    request = make_request("tools/list", params={}, request_id=2)
    logger.info("CLIENT -> SERVER:\n%s", request)

    response = make_response(
        result={
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                }
            ]
        },
        request_id=2,
    )
    logger.info("SERVER -> CLIENT:\n%s", response)


def demo_tools_call() -> None:
    """Show a tool call exchange."""
    logger.info("\n=== tools/call ===")

    request = make_request(
        method="tools/call",
        params={"name": "get_weather", "arguments": {"city": "Berlin"}},
        request_id=3,
    )
    logger.info("CLIENT -> SERVER:\n%s", request)

    response = make_response(
        result={
            "content": [
                {"type": "text", "text": "Berlin: 12°C, partly cloudy"}
            ],
            "isError": False,
        },
        request_id=3,
    )
    logger.info("SERVER -> CLIENT:\n%s", response)


if __name__ == "__main__":
    demo_initialize_handshake()
    demo_tools_list()
    demo_tools_call()
```

Run it:

```bash
python mcp_protocol_demo.py
```

You should see the initialize handshake, a tools listing, and a tool call — all as raw JSON-RPC. This is exactly what the `mcp` SDK serializes/deserializes for you.

## Exercises

1. **Protocol mapping**: Draw (on paper or in a comment block) the full message sequence for a client that: (a) initializes, (b) lists available resources, (c) reads one resource, (d) calls one tool. What messages flow in each direction?

2. **Capability design**: You are building an MCP server that wraps a PostgreSQL database. List the MCP capabilities it should advertise (tools vs resources vs prompts) and justify each choice. Write your answer as comments in a Python file.

3. **Ecosystem research**: Visit `modelcontextprotocol.io/servers`. Find three servers relevant to data/ML work. For each: what does it expose as tools? As resources? Note your findings in a Python docstring.

4. **Error handling sketch**: Extend `mcp_protocol_demo.py` with a `demo_error_case()` function that shows the message flow when `tools/call` is called with a missing required argument. Use the appropriate JSON-RPC error code (hint: -32602 is "Invalid params").

(Solutions: ../solutions/week-08/day-1.md)

## Self-check

- [ ] I can explain the M×N integration problem and how MCP reduces it to M+N
- [ ] I can describe what JSON-RPC 2.0 is and how it maps to MCP messages
- [ ] I can name the three MCP primitive types (tools, resources, prompts)
- [ ] I can describe the two official MCP transports (stdio, HTTP+SSE)
- [ ] I can sketch the initialize handshake from memory

---

**Supplementary**: [MCP official docs](https://modelcontextprotocol.io) · [MCP spec](https://spec.modelcontextprotocol.io) · [MCP servers list](https://modelcontextprotocol.io/servers)
