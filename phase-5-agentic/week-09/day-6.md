# Phase 5 · Week 09 · Day 6 — MCP + A2A Together & the Standards Landscape

> Objective: Combine MCP (tools/data access) with A2A (agent delegation) in a single architecture, and understand the current standards landscape.  ·  Time: ~1 hr  ·  Prereqs: [[day-5]], [[week-05]]

## Concept

MCP and A2A solve different problems and complement each other:

| | MCP | A2A |
|---|---|---|
| **What it connects** | Agent ↔ Tool / Data source | Agent ↔ Agent |
| **Direction** | Agent calls tool | Agent delegates to agent |
| **Responder type** | Deterministic function or DB query | Another intelligent agent |
| **Response time** | Milliseconds | Seconds to hours |
| **State** | Stateless per call | Stateful task lifecycle |
| **Discovery** | Tool schema in manifest | Agent card at `/.well-known/agent.json` |

**Combined pattern — "MCP-inside, A2A-outside":**

```
User
  │
  ▼
Orchestrator Agent
  │  (A2A protocol)          (A2A protocol)
  ├──→ SQL Agent ────────── Finance DB (via MCP)
  ├──→ Search Agent ─────── Web Search Tool (via MCP)
  └──→ Writer Agent ─────── File System (via MCP)
```

Each specialist agent uses MCP to access its tools and data. The orchestrator uses A2A to delegate to each specialist. Users see only the orchestrator — the internals are fully encapsulated.

**The standards landscape (as of 2025):**

- **A2A** — Google-initiated, now community-governed. HTTP-based, JSON, SSE. Focused on interoperability between agents from different providers.
- **MCP (Model Context Protocol)** — Anthropic-initiated, community-governed. JSON-RPC 2.0. Focused on connecting agents to tools and data.
- **OpenAI Agent Protocol** — OpenAI's agent interoperability work (similar goals to A2A, different wire format).
- **ANP (Agent Network Protocol)** — Emerging, more decentralized, DID-based identity.
- **LangChain Agent Protocol** — Earlier attempt, less structured than A2A.

The trajectory: **convergence around A2A + MCP** as the de-facto pairing, similar to how REST + OAuth became de-facto for web APIs in the 2010s.

## How it works internally

When an orchestrator delegates via A2A, the receiving specialist may itself be a LangGraph agent that uses MCP tools. The specialist's internal architecture is invisible to the orchestrator — only the A2A interface matters.

**Practical integration:**

```python
# Inside a specialist agent (uses MCP for tools)
async def handle_task(task: Task) -> Artifact:
    async with MCPClient(server_url="http://mcp-server/sse") as mcp:
        tools = await mcp.list_tools()
        result = await run_agent_with_tools(task, tools, mcp)
    return Artifact(parts=[TextPart(text=result)])
```

**Authentication flow:**
- Orchestrator → A2A server: bearer token in `Authorization` header
- A2A server → MCP server: separate credentials (never forwarded from caller)
- Each hop has its own auth context

**Observability:** Each agent logs with a shared `correlation_id` (passed in A2A task metadata), enabling distributed tracing across the chain.

## Worked example

```python
"""
day_6_mcp_a2a_combined.py
Demonstrates MCP + A2A architecture:
- MockMCPClient simulates tool calls
- Specialist agents use MCP tools internally
- Orchestrator uses A2A to delegate to specialists
- Correlation ID threads through for observability
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ──────────────────────────────────────────────
# Mock MCP layer
# ──────────────────────────────────────────────

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPToolResult:
    content: list[dict[str, Any]]
    is_error: bool = False


class MockMCPClient:
    """
    Simulates an MCP server. In production, connects via HTTP/SSE
    to a real MCP server (database, web search, filesystem, etc.).
    """

    def __init__(self, server_name: str, tools: dict[str, Callable[..., str]]) -> None:
        self._name = server_name
        self._tool_fns = tools
        self._logger = logging.getLogger(f"mcp.{server_name}")

    async def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(name=name, description=f"{name} tool", input_schema={})
            for name in self._tool_fns
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        self._logger.info("Calling tool '%s' with args: %s", name, arguments)
        if name not in self._tool_fns:
            return MCPToolResult(
                content=[{"type": "text", "text": f"Tool not found: {name}"}],
                is_error=True,
            )
        result = self._tool_fns[name](**arguments)
        return MCPToolResult(content=[{"type": "text", "text": result}])


# ──────────────────────────────────────────────
# A2A types (minimal, reusing day-5 patterns)
# ──────────────────────────────────────────────

@dataclass
class A2ATask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""   # For distributed tracing
    payload: str = ""
    result: str = ""
    status: str = "submitted"
    metadata: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Specialist agents (A2A servers that use MCP)
# ──────────────────────────────────────────────

class DataAnalystAgent:
    """
    A2A server for data analysis.
    Uses MCP internally to query a database and run calculations.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("agent.data-analyst")
        self._mcp = MockMCPClient(
            server_name="analytics-db",
            tools={
                "query_revenue": lambda period: f"Revenue for {period}: $2.4M (Q3), $2.8M (Q4)",
                "calculate_growth": lambda q1, q2: f"Growth from {q1} to {q2}: +16.7%",
            },
        )

    def get_skills(self) -> list[str]:
        return ["financial-analysis", "revenue-reporting"]

    async def handle(self, task: A2ATask) -> A2ATask:
        self._logger.info("[%s] Handling task: %s", task.correlation_id[:8], task.payload)
        task.status = "working"

        # Use MCP tools internally (caller never sees this)
        revenue_result = await self._mcp.call_tool(
            "query_revenue", {"period": "last-two-quarters"}
        )
        growth_result = await self._mcp.call_tool(
            "calculate_growth", {"q1": "Q3-2024", "q2": "Q4-2024"}
        )

        revenue_text = revenue_result.content[0]["text"]
        growth_text = growth_result.content[0]["text"]

        task.result = f"Financial Analysis: {revenue_text}. {growth_text}."
        task.status = "completed"
        self._logger.info("[%s] Task completed", task.correlation_id[:8])
        return task


class WebSearchAgent:
    """
    A2A server for web search.
    Uses MCP to access a search tool and a scraping tool.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("agent.web-search")
        self._mcp = MockMCPClient(
            server_name="web-tools",
            tools={
                "search": lambda query: f"Top results for '{query}': [1] Competitor X launches..., [2] Market grows 25%...",
                "fetch_page": lambda url: f"Page content from {url}: Full article text here.",
            },
        )

    def get_skills(self) -> list[str]:
        return ["web-search", "competitive-intelligence"]

    async def handle(self, task: A2ATask) -> A2ATask:
        self._logger.info("[%s] Searching: %s", task.correlation_id[:8], task.payload)
        task.status = "working"

        search_result = await self._mcp.call_tool("search", {"query": task.payload})
        task.result = search_result.content[0]["text"]
        task.status = "completed"
        return task


class ReportWriterAgent:
    """
    A2A server for writing structured reports.
    Uses MCP to read a template from the filesystem and write the output.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("agent.report-writer")
        self._mcp = MockMCPClient(
            server_name="filesystem",
            tools={
                "read_template": lambda name: f"Template '{name}': # {{title}}\n## Summary\n{{summary}}\n## Details\n{{details}}",
                "write_file": lambda path, content: f"Written {len(content)} chars to {path}",
            },
        )

    def get_skills(self) -> list[str]:
        return ["report-generation", "document-writing"]

    async def handle(self, task: A2ATask) -> A2ATask:
        self._logger.info("[%s] Writing report", task.correlation_id[:8])
        task.status = "working"

        template_result = await self._mcp.call_tool(
            "read_template", {"name": "executive-summary"}
        )
        template = template_result.content[0]["text"]

        # Fill template with injected context from metadata
        context = task.metadata.get("context", {})
        report = template.format(
            title=context.get("title", "Executive Report"),
            summary=context.get("summary", task.payload),
            details=context.get("details", "See appendix."),
        )

        write_result = await self._mcp.call_tool(
            "write_file", {"path": "reports/output.md", "content": report}
        )
        task.result = f"Report generated. {write_result.content[0]['text']}"
        task.status = "completed"
        return task


# ──────────────────────────────────────────────
# Orchestrator (A2A client)
# ──────────────────────────────────────────────

class ResearchOrchestrator:
    """
    Coordinates DataAnalyst, WebSearch, and ReportWriter via A2A.
    Each specialist uses MCP internally — orchestrator doesn't know or care.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger("orchestrator")
        self._analysts: dict[str, Any] = {
            "financial-analysis": DataAnalystAgent(),
            "web-search": WebSearchAgent(),
            "report-generation": ReportWriterAgent(),
        }

    async def run(self, query: str) -> str:
        correlation_id = str(uuid.uuid4())
        self._logger.info("[%s] Starting research pipeline for: %s",
                          correlation_id[:8], query)

        # Phase 1: Parallel information gathering
        financial_task = A2ATask(
            payload=f"Analyze financials for: {query}",
            correlation_id=correlation_id,
        )
        search_task = A2ATask(
            payload=f"competitive landscape for {query}",
            correlation_id=correlation_id,
        )

        financial_result, search_result = await asyncio.gather(
            self._analysts["financial-analysis"].handle(financial_task),
            self._analysts["web-search"].handle(search_task),
        )

        self._logger.info(
            "[%s] Phase 1 done: financial=%s, search=%s",
            correlation_id[:8],
            financial_result.status,
            search_result.status,
        )

        # Phase 2: Sequential — write report using phase 1 outputs
        report_task = A2ATask(
            payload=f"Executive report for {query}",
            correlation_id=correlation_id,
            metadata={
                "context": {
                    "title": f"Market Analysis: {query}",
                    "summary": financial_result.result,
                    "details": search_result.result,
                }
            },
        )
        report_result = await self._analysts["report-generation"].handle(report_task)

        self._logger.info("[%s] Pipeline complete", correlation_id[:8])
        return (
            f"=== Research Complete (correlation: {correlation_id[:8]}) ===\n"
            f"Financial: {financial_result.result}\n"
            f"Market: {search_result.result}\n"
            f"Output: {report_result.result}"
        )


# ──────────────────────────────────────────────
# Standards landscape summary (printed, not tested)
# ──────────────────────────────────────────────

def print_standards_landscape() -> None:
    standards = {
        "A2A (Agent-to-Agent)": {
            "initiator": "Google → community",
            "transport": "HTTP/SSE",
            "focus": "Agent-to-agent delegation",
            "status": "Active, v0.2+",
        },
        "MCP (Model Context Protocol)": {
            "initiator": "Anthropic → community",
            "transport": "JSON-RPC 2.0 / HTTP+SSE",
            "focus": "Agent-to-tool/data",
            "status": "Active, widely adopted",
        },
        "OpenAI Agent Protocol": {
            "initiator": "OpenAI",
            "transport": "HTTP",
            "focus": "Agent interoperability",
            "status": "Emerging",
        },
        "ANP (Agent Network Protocol)": {
            "initiator": "Community",
            "transport": "DID / P2P",
            "focus": "Decentralized agent identity",
            "status": "Early draft",
        },
    }
    logger = logging.getLogger("standards")
    for name, props in standards.items():
        logger.info("%-35s | %-12s | %s", name, props["status"], props["focus"])


async def main() -> None:
    print_standards_landscape()
    orchestrator = ResearchOrchestrator()
    result = await orchestrator.run("ACME Corp Q4 2024")
    print("\n" + result)


if __name__ == "__main__":
    asyncio.run(main())
```

## Exercises

1. Add a `CodeExecutionAgent` that uses a mock MCP `run_python` tool. Register it with the orchestrator and add a third phase after the report: execute a Python snippet to compute a summary statistic. Mock the tool to return `"Mean: 42.5, Std: 3.1"`. Wire it end-to-end with a pytest test.

2. Implement `correlation_id` propagation: when the orchestrator creates tasks, embed the same `correlation_id` in all sub-tasks. Add a `LogFilter` that prefixes all log lines with `[cid:{correlation_id[:8]}]`. Write a test confirming all log records share the same prefix.

3. Design an `AuthContext` Pydantic model with `scheme: Literal["none","bearer","api_key"]` and `token: str | None`. Update each specialist agent to require the correct auth in its A2A handler (raise `PermissionError` if missing). Update the orchestrator to inject credentials. Write tests for both authenticated and unauthenticated paths.

4. Draw (as a commented ASCII diagram in a Python docstring) a three-hop architecture: UserAgent → OrchestratorAgent (A2A) → SpecialistAgent (A2A) → MCPServer. Label which protocols are used at each hop and where credentials are needed.

(Solutions: ../solutions/week-09/day-6.md)

## Self-check
- [ ] I can explain the MCP/A2A division of responsibility in one sentence
- [ ] I can implement a specialist agent that uses MCP internally and exposes A2A externally
- [ ] I can thread a correlation ID through a multi-agent pipeline for observability
- [ ] I can name four agent interoperability standards and their key difference

## Supplementary links
- [Anthropic MCP documentation](https://modelcontextprotocol.io/)
- [A2A + MCP integration guide](https://a2aproject.github.io/A2A/topics/a2a-and-mcp/)
- [OpenAI agent protocol announcement](https://openai.com/index/agents/)
