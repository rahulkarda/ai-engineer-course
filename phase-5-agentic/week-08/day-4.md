# Phase 5 · Week 08 · Day 4 — Building an MCP Server in Python

> Objective: Build a functional MCP server using the `mcp` Python SDK  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### The `mcp` Python SDK

The official MCP Python SDK (`mcp`) is the reference implementation. Install via:

```bash
uv add mcp
# or
pip install mcp
```

The SDK provides:
- `mcp.server.Server` — low-level server with explicit handler registration
- `mcp.server.fastmcp.FastMCP` — high-level decorator-based API (recommended for new servers)
- Transport implementations: `stdio`, `sse`
- Client-side classes (covered in Day 5)

We use **FastMCP** today. It's analogous to FastAPI: define tools, resources, and prompts with decorators; the SDK handles protocol negotiation, message routing, and serialization.

### FastMCP Server Structure

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("server-name")

@mcp.tool()
def my_tool(arg: str) -> str:
    """Tool description visible to the LLM."""
    return f"result: {arg}"

@mcp.resource("scheme://path/{param}")
def my_resource(param: str) -> str:
    """Resource content."""
    return f"data for {param}"

@mcp.prompt()
def my_prompt(arg: str) -> str:
    """Prompt template."""
    return f"Please analyze: {arg}"
```

### Tool Registration Details

When you decorate a function with `@mcp.tool()`:
- The function's **docstring** becomes the tool description sent to the LLM
- The function's **type hints** are converted to JSON Schema for `inputSchema`
- The **return value** is wrapped in a `TextContent` response
- The function can be sync or async
- Pydantic types in the signature are fully supported

For tools that may fail: return a `list[TextContent | ImageContent | EmbeddedResource]` or raise an exception (SDK converts it to `isError: True`).

### Resource Registration

`@mcp.resource("uri-template")`:
- URI template uses `{param}` placeholders
- Parameters become function arguments
- Supports `mime_type` kwarg: `@mcp.resource("...", mime_type="application/json")`
- Return `str` (text) or `bytes` (binary/blob)

### Prompt Registration

`@mcp.prompt()`:
- Docstring becomes the prompt description
- Parameters become prompt arguments
- Return `str` (rendered into a user message) or `list[Message]` for multi-turn

### Running the Server

```python
if __name__ == "__main__":
    mcp.run()               # defaults to stdio transport
    # or:
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
```

For stdio: the server reads from stdin, writes to stdout, runs until the pipe closes.

## How It Works Internally

### Decorator Internals

`@mcp.tool()` calls `mcp.add_tool(func)` which:
1. Inspects the function signature via `inspect.signature`
2. Uses Pydantic's `TypeAdapter` or `create_model` to build the JSON Schema
3. Registers the tool in an internal registry keyed by function name
4. Wraps the function in an async handler that validates arguments before calling

### Request Dispatch

When the server receives `{"method": "tools/call", "params": {"name": "my_tool", "arguments": {...}}}`:
1. SDK looks up `my_tool` in the registry
2. Validates `arguments` against the tool's input schema
3. Calls the handler (wrapping sync functions with `asyncio.to_thread`)
4. Serializes the return value to `ToolResult`
5. Writes the JSON-RPC response to stdout

### Error Handling

If the handler raises any exception, the SDK catches it and returns:
```json
{"content": [{"type": "text", "text": "<error message>"}], "isError": true}
```

This is important: **tool errors should not crash the server**. The LLM receives the error message and can decide how to proceed.

### Context Object

FastMCP passes an optional `Context` object if you include `ctx: Context` in the signature:

```python
from mcp.server.fastmcp import FastMCP, Context

@mcp.tool()
async def my_tool(query: str, ctx: Context) -> str:
    await ctx.info(f"Searching for: {query}")  # sends log notification to client
    await ctx.report_progress(0, 100)           # sends progress notification
    return "result"
```

The `Context` enables logging, progress reporting, and reading other resources from within a tool.

## Worked Example

This is a complete, runnable MCP server. It exposes tools, resources, and a prompt.

```python
# knowledge_server.py
# A complete MCP server for a knowledge base.
# Run: python knowledge_server.py
# It communicates over stdio — pipe it to an MCP client to use.

import logging
import os
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP, Context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    # Log to stderr so it doesn't pollute the stdio MCP channel
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Server definition ─────────────────────────────────────────────────────────

mcp = FastMCP(
    name="knowledge-base",
    version="1.0.0",
    instructions=(
        "A knowledge base server. Use search_articles to find articles, "
        "get_article_stats for counts, and read articles via resources."
    ),
)

# ── Simulated data store ──────────────────────────────────────────────────────

ARTICLES: dict[str, dict[str, Any]] = {
    "mcp-intro": {
        "id": "mcp-intro",
        "title": "Introduction to MCP",
        "category": "protocols",
        "content": "MCP standardizes tool and data integration for AI applications...",
        "tags": ["mcp", "ai", "protocols"],
        "created_at": "2024-11-15",
    },
    "langgraph-basics": {
        "id": "langgraph-basics",
        "title": "LangGraph Basics",
        "category": "frameworks",
        "content": "LangGraph is a graph-based agent framework built on LangChain...",
        "tags": ["langgraph", "agents", "python"],
        "created_at": "2024-10-01",
    },
    "rag-patterns": {
        "id": "rag-patterns",
        "title": "RAG Design Patterns",
        "category": "retrieval",
        "content": "Retrieval-Augmented Generation combines vector search with LLMs...",
        "tags": ["rag", "embeddings", "retrieval"],
        "created_at": "2024-09-20",
    },
}


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_articles(
    query: str,
    category: str | None = None,
    max_results: int = 5,
    ctx: Context = None,
) -> str:
    """Search knowledge base articles by keyword. Optionally filter by category.

    Returns a list of matching article titles and IDs.
    """
    if ctx:
        await ctx.info(f"Searching for '{query}' in category={category}")

    query_lower = query.lower()
    results = []
    for article in ARTICLES.values():
        if category and article["category"] != category:
            continue
        if (
            query_lower in article["title"].lower()
            or query_lower in article["content"].lower()
            or any(query_lower in tag for tag in article["tags"])
        ):
            results.append(article)

    results = results[:max_results]
    if not results:
        return f"No articles found for query: {query!r}"

    lines = [f"Found {len(results)} article(s):"]
    for art in results:
        lines.append(f"  - [{art['id']}] {art['title']} ({art['category']})")
    return "\n".join(lines)


@mcp.tool()
async def get_article_stats(ctx: Context = None) -> str:
    """Get statistics about the knowledge base: total articles, categories, and tag counts."""
    if ctx:
        await ctx.info("Computing article stats")

    categories: dict[str, int] = {}
    all_tags: dict[str, int] = {}
    for article in ARTICLES.values():
        categories[article["category"]] = categories.get(article["category"], 0) + 1
        for tag in article["tags"]:
            all_tags[tag] = all_tags.get(tag, 0) + 1

    lines = [
        f"Total articles: {len(ARTICLES)}",
        "Categories:",
    ]
    for cat, count in sorted(categories.items()):
        lines.append(f"  {cat}: {count}")
    lines.append("Top tags:")
    for tag, count in sorted(all_tags.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  {tag}: {count}")
    return "\n".join(lines)


@mcp.tool()
async def add_article(
    id: str,
    title: str,
    category: str,
    content: str,
    tags: list[str] | None = None,
    ctx: Context = None,
) -> str:
    """Add a new article to the knowledge base.

    Returns the created article ID on success.
    """
    if id in ARTICLES:
        return f"Error: article with id '{id}' already exists."
    if not title.strip() or not content.strip():
        return "Error: title and content cannot be empty."

    ARTICLES[id] = {
        "id": id,
        "title": title,
        "category": category,
        "content": content,
        "tags": tags or [],
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    }
    logger.info("Article added: %s", id)
    if ctx:
        await ctx.info(f"Article added: {id}")
    return f"Article '{id}' created successfully."


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("kb://articles/{article_id}")
def get_article(article_id: str) -> str:
    """Return the full content of a specific article."""
    if article_id not in ARTICLES:
        return f"Article not found: {article_id}"
    art = ARTICLES[article_id]
    return (
        f"# {art['title']}\n"
        f"Category: {art['category']}\n"
        f"Tags: {', '.join(art['tags'])}\n"
        f"Created: {art['created_at']}\n\n"
        f"{art['content']}"
    )


@mcp.resource("kb://articles", mime_type="application/json")
def list_article_metadata() -> str:
    """Return JSON listing of all article metadata (no content)."""
    import json
    listing = [
        {k: v for k, v in art.items() if k != "content"}
        for art in ARTICLES.values()
    ]
    return json.dumps(listing, indent=2)


@mcp.resource("kb://categories/{category}")
def get_category_articles(category: str) -> str:
    """Return all articles in a specific category."""
    articles = [a for a in ARTICLES.values() if a["category"] == category]
    if not articles:
        return f"No articles in category: {category}"
    lines = [f"Articles in '{category}':"]
    for art in articles:
        lines.append(f"  - [{art['id']}] {art['title']}")
    return "\n".join(lines)


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def research_assistant(topic: str, depth: str = "overview") -> str:
    """Generate a research prompt for a given topic.

    Args:
        topic: The subject to research.
        depth: 'overview' for a summary or 'deep-dive' for thorough analysis.
    """
    if depth == "deep-dive":
        instruction = (
            "Provide a thorough, structured analysis including: background, "
            "key concepts, current best practices, trade-offs, and open questions."
        )
    else:
        instruction = "Provide a clear, concise overview suitable for someone new to the topic."

    return (
        f"You are a knowledgeable research assistant. {instruction}\n\n"
        f"Topic: {topic}\n\n"
        f"Draw on your knowledge and any available context. "
        f"Structure your response with clear headings."
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    logger.info("Starting knowledge-base MCP server, transport=%s", transport)
    if transport == "sse":
        host = os.getenv("MCP_HOST", "127.0.0.1")
        port = int(os.getenv("MCP_PORT", "8000"))
        mcp.run(transport="sse", host=host, port=port)
    else:
        mcp.run(transport="stdio")
```

Test the server manually with the MCP CLI (if installed):

```bash
# Install MCP CLI
uv tool install mcp

# Inspect the server
mcp dev knowledge_server.py
```

Or test programmatically (see Day 5 for the full client). For now, verify it starts:

```bash
# Should print nothing and wait for input — Ctrl+C to exit
python knowledge_server.py
```

## Exercises

1. **Add a tool**: Add a `delete_article(id: str) -> str` tool to `knowledge_server.py`. It should return an error message if the article doesn't exist, and a success message with the deleted title if it does. Include a `ctx.warning(...)` log call.

2. **Resource template**: Add a resource at `"kb://search/{query}"` that returns articles matching `query` as formatted text (reuse the search logic). Test it by reading the resource in a unit test.

3. **Prompt with resources**: Add a `"summarize_category"` prompt that accepts a `category` string, reads the matching articles from `ARTICLES`, and returns a prompt message asking the LLM to summarize the main themes in that category.

4. **Unit-test the tools**: Write `tests/test_knowledge_server.py` with pytest. Test `search_articles`, `get_article_stats`, and `add_article`. Do NOT start the real MCP server — call the handler functions directly (they're just async Python functions). Use `pytest-asyncio` or `anyio`.

(Solutions: ../solutions/week-08/day-4.md)

## Self-check

- [ ] I can create a FastMCP server with at least one tool, one resource, and one prompt
- [ ] I know how to type-hint tool arguments so FastMCP generates the correct JSON Schema
- [ ] I know how to use `Context` for logging and progress inside a tool
- [ ] I can run an MCP server in stdio mode and understand why stderr is used for logs
- [ ] I can write pytest tests for tool handlers without starting the real server

---

**Supplementary**: [FastMCP docs](https://gofastmcp.com) · [mcp Python SDK](https://github.com/modelcontextprotocol/python-sdk) · [MCP server examples](https://github.com/modelcontextprotocol/servers)
