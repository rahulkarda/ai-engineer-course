# Phase 5 · Week 08 · Day 3 — MCP Primitives: Tools, Resources, and Prompts

> Objective: Master the three MCP primitive types and know when to use each  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Why Three Primitives?

MCP doesn't expose one generic "do something" primitive. It distinguishes three fundamentally different kinds of capability, because:

- The **host/LLM** needs to know what it can call (tools), what context it can read (resources), and what interaction patterns are available (prompts)
- **Security policy** differs: tools have side effects, resources are read-only data, prompts are templates
- **UI representation** differs: Claude Desktop shows tools as "available actions", resources as "attached files", prompts as "suggested starters"

### Primitive 1: Tools

**What**: Executable functions the LLM can invoke. They *do* things — they may have side effects.

**Schema**: Each tool has a `name`, `description`, and `inputSchema` (JSON Schema object). The description is critical: the LLM reads it to decide when to call the tool.

**Invocation**: `tools/call` request with `name` and `arguments`.

**Response**: A list of `content` items (text, image, embedded resource). Has an `isError` flag.

```json
{
  "name": "create_file",
  "description": "Create a new file at the given path with given content.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "path": { "type": "string" },
      "content": { "type": "string" }
    },
    "required": ["path", "content"]
  }
}
```

**Tool result content types**:
- `{ "type": "text", "text": "..." }` — plain text
- `{ "type": "image", "data": "<base64>", "mimeType": "image/png" }` — image
- `{ "type": "resource", "resource": { "uri": "...", "text": "..." } }` — embedded resource reference

### Primitive 2: Resources

**What**: Read-only data the client/host can attach to context. Resources are *not* invocable by the LLM directly. The host fetches them and includes them in the LLM's context window.

**Identification**: Each resource has a `uri` (e.g., `file:///project/README.md`, `db://customers/42`, `git://main/src/app.py`).

**Listing**: `resources/list` returns all available resources. For large collections, use `resources/list` with pagination cursors.

**Reading**: `resources/read` with a `uri` returns `contents` — a list of `{ "uri": "...", "mimeType": "...", "text": "..." }` or `{ "uri": "...", "blob": "<base64>" }`.

**Resource templates**: `resources/templates/list` returns URI templates (RFC 6570) that clients can expand to construct resource URIs dynamically. Example: `"file:///{+path}"`.

**Subscriptions**: If the server advertises `resources.subscribe: true`, clients can subscribe to a URI and receive `notifications/resources/updated` when the content changes.

```json
{
  "uri": "db://inventory/products",
  "name": "Product catalog",
  "description": "Full product inventory",
  "mimeType": "application/json"
}
```

### Primitive 3: Prompts

**What**: Parameterized message templates. The server defines reusable prompt patterns; the client retrieves a rendered prompt and injects it into the conversation.

**Listing**: `prompts/list` returns available prompts with their argument schemas.

**Getting**: `prompts/get` with `name` and `arguments` returns a list of `{ "role": "user"|"assistant", "content": {...} }` messages — ready to prepend to an LLM conversation.

Use case: encode expert workflows as prompts. Example: a code review server might have a `review_pr` prompt that takes `(diff, style_guide)` and returns a structured set of system+user messages that prime the LLM to do a thorough review.

```json
{
  "name": "analyze_logs",
  "description": "Prime the LLM to analyze application logs for errors.",
  "arguments": [
    { "name": "log_content", "description": "Raw log text", "required": true },
    { "name": "severity_filter", "description": "Minimum severity", "required": false }
  ]
}
```

### Choosing the Right Primitive

| You want to... | Use |
|---|---|
| Perform an action (write, call API, compute) | Tool |
| Provide background data/context (file, DB record, config) | Resource |
| Give the LLM a structured starting prompt | Prompt |
| Execute a read that has side effects | Tool (even if it "reads") |
| Stream large data the LLM should reason over | Resource |
| Encode a domain expert workflow | Prompt |

## How It Works Internally

### Tool Input Validation

The server receives `arguments` as an unvalidated `dict`. Best practice: validate immediately against a Pydantic model derived from the `inputSchema`, raise a structured error if invalid.

```python
from pydantic import BaseModel, ValidationError

class WeatherArgs(BaseModel):
    city: str
    units: str = "celsius"

def validate_tool_args(args: dict) -> WeatherArgs:
    try:
        return WeatherArgs.model_validate(args)
    except ValidationError as exc:
        raise ValueError(f"Invalid arguments: {exc}") from exc
```

### Resource URI Design

URIs should be stable, hierarchical, and meaningful:

```
file:///absolute/path/to/file.txt          # filesystem
db://database_name/table_name/row_id       # database records
git://branch/path/in/repo                  # git objects
http://internal-api/endpoint               # external data
mem://session_id/memory_key                # in-memory data
```

Avoid opaque IDs like `resource://x/abc123def`. The LLM may read the URI and use it to reason about what the resource is.

### Prompt Message Format

A prompt response contains `messages` — an ordered list ready for a chat API:

```json
{
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Please analyze these logs for errors:\n<logs>..."
      }
    }
  ]
}
```

The host can prepend these to its own conversation messages before calling the LLM. The rendered prompt may embed resource content fetched by the server.

## Worked Example

```python
# mcp_primitives_demo.py
# Demonstrates all three MCP primitives with typed schemas and Pydantic validation.
# This is a standalone simulation — no real MCP transport.

import logging
from typing import Any
from pydantic import BaseModel, ValidationError, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Tool primitive ─────────────────────────────────────────────────────────────

class SearchArgs(BaseModel):
    query: str
    max_results: int = 5

    @field_validator("max_results")
    @classmethod
    def clamp_results(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_results must be >= 1")
        return min(v, 20)


TOOL_SCHEMA: dict[str, Any] = {
    "name": "search_documents",
    "description": (
        "Search the internal document store. Returns ranked snippets."
        " Use this when the user asks to find documents or information."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string"},
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (1–20)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


def handle_search_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool handler: validate, execute, return MCP-shaped result."""
    try:
        args = SearchArgs.model_validate(arguments)
    except ValidationError as exc:
        return {
            "isError": True,
            "content": [{"type": "text", "text": f"Invalid arguments: {exc}"}],
        }

    # Simulated search results
    results = [f"Doc {i}: result for '{args.query}'" for i in range(args.max_results)]
    text = "\n".join(results)
    logger.info("Tool search_documents: query=%r results=%d", args.query, len(results))
    return {
        "isError": False,
        "content": [{"type": "text", "text": text}],
    }


# ── Resource primitive ─────────────────────────────────────────────────────────

RESOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "docs://catalog/overview": {
        "uri": "docs://catalog/overview",
        "name": "Document Catalog Overview",
        "description": "High-level summary of all documents in the system.",
        "mimeType": "text/plain",
    },
    "docs://catalog/schema": {
        "uri": "docs://catalog/schema",
        "name": "Document Schema",
        "description": "JSON Schema for a document object.",
        "mimeType": "application/json",
    },
}

RESOURCE_CONTENTS: dict[str, str] = {
    "docs://catalog/overview": (
        "The document catalog contains 1,247 technical documents.\n"
        "Categories: architecture (312), api-reference (455), tutorials (480)."
    ),
    "docs://catalog/schema": (
        '{"type":"object","properties":{"id":{"type":"string"},'
        '"title":{"type":"string"},"category":{"type":"string"},'
        '"content":{"type":"string"}}}'
    ),
}


def handle_resources_list() -> list[dict[str, Any]]:
    return list(RESOURCE_REGISTRY.values())


def handle_resources_read(uri: str) -> dict[str, Any]:
    if uri not in RESOURCE_REGISTRY:
        return {"error": f"Unknown resource URI: {uri}"}
    meta = RESOURCE_REGISTRY[uri]
    text = RESOURCE_CONTENTS[uri]
    logger.info("Resource read: %s", uri)
    return {
        "contents": [
            {"uri": uri, "mimeType": meta["mimeType"], "text": text}
        ]
    }


# ── Prompt primitive ──────────────────────────────────────────────────────────

class DocumentSummaryArgs(BaseModel):
    document_uri: str
    audience: str = "technical"
    max_words: int = 200


PROMPT_SCHEMA: dict[str, Any] = {
    "name": "summarize_document",
    "description": "Generate a structured prompt to summarize a document for a given audience.",
    "arguments": [
        {"name": "document_uri", "description": "URI of the resource to summarize", "required": True},
        {"name": "audience", "description": "Target audience: technical or executive", "required": False},
        {"name": "max_words", "description": "Maximum word count for summary", "required": False},
    ],
}


def handle_prompts_get(arguments: dict[str, Any]) -> dict[str, Any]:
    """Render a prompt by fetching the referenced resource and building messages."""
    try:
        args = DocumentSummaryArgs.model_validate(arguments)
    except ValidationError as exc:
        return {"error": str(exc)}

    # Fetch resource content to embed in prompt
    resource_result = handle_resources_read(args.document_uri)
    if "error" in resource_result:
        return {"error": resource_result["error"]}

    content_text = resource_result["contents"][0]["text"]
    logger.info("Prompt get: summarize_document audience=%s", args.audience)

    return {
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Please summarize the following document for a {args.audience} audience"
                        f" in at most {args.max_words} words.\n\n"
                        f"Document:\n{content_text}"
                    ),
                },
            }
        ]
    }


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== Tool Demo ===")
    result = handle_search_tool({"query": "LangGraph state management"})
    logger.info("Tool result: %s", result)

    bad_result = handle_search_tool({"query": "test", "max_results": 0})
    logger.info("Bad args result: %s", bad_result)

    logger.info("\n=== Resource Demo ===")
    resources = handle_resources_list()
    logger.info("Available resources: %s", [r["uri"] for r in resources])

    content = handle_resources_read("docs://catalog/overview")
    logger.info("Resource content:\n%s", content["contents"][0]["text"])

    missing = handle_resources_read("docs://catalog/nonexistent")
    logger.info("Missing resource: %s", missing)

    logger.info("\n=== Prompt Demo ===")
    prompt_result = handle_prompts_get({
        "document_uri": "docs://catalog/overview",
        "audience": "executive",
        "max_words": 100,
    })
    logger.info("Rendered prompt message:\n%s", prompt_result["messages"][0]["content"]["text"])


if __name__ == "__main__":
    main()
```

Run:

```bash
python mcp_primitives_demo.py
```

Observe: each primitive has a distinct shape. Tools validate input and may error. Resources return raw content. Prompts render messages ready for an LLM.

## Exercises

1. **Tool schema design**: Design the `inputSchema` JSON for a tool called `run_sql_query` that takes a `query` string and an optional `database` string (default: `"default"`). Write the schema and the Pydantic model for argument validation.

2. **Resource URI scheme**: You are building an MCP server for a GitHub repository. Define a URI scheme and list (as Python dicts) at least five resources it should expose. What `mimeType` should each use?

3. **Prompt embedding**: Extend `handle_prompts_get` with a second prompt called `"compare_documents"` that takes `uri_a` and `uri_b`, reads both resources, and returns a two-message prompt (system + user) asking the LLM to compare them. Write a unit test (using `pytest`, mocking `handle_resources_read`).

4. **Primitive classification**: For each of the following capabilities, decide whether it should be a tool, resource, or prompt — and justify:
   - Retrieve the current stock price of a ticker symbol
   - The company's internal style guide (static Markdown file)
   - A multi-step code review checklist template
   - Delete a record from a database
   - Fetch today's calendar events (read-only, but dynamic)

(Solutions: ../solutions/week-08/day-3.md)

## Self-check

- [ ] I can describe the three MCP primitives and name a concrete use case for each
- [ ] I know the JSON shape of a tool descriptor, resource descriptor, and prompt descriptor
- [ ] I can implement argument validation using Pydantic for tool calls
- [ ] I can design a URI scheme for a domain-specific MCP resource set
- [ ] I can explain why resources are fetched by the host, not called by the LLM

---

**Supplementary**: [MCP tools spec](https://spec.modelcontextprotocol.io/specification/server/tools/) · [Resources](https://spec.modelcontextprotocol.io/specification/server/resources/) · [Prompts](https://spec.modelcontextprotocol.io/specification/server/prompts/)
