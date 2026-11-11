# Solutions — Week 08 · Day 3

## Exercise 1 — run_sql_query schema and Pydantic model

```python
# JSON Schema
RUN_SQL_QUERY_SCHEMA = {
    "name": "run_sql_query",
    "description": "Execute a read-only SQL SELECT query against the specified database.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT statement to execute.",
            },
            "database": {
                "type": "string",
                "description": "Target database name.",
                "default": "default",
            },
        },
        "required": ["query"],
    },
}

# Pydantic model
from pydantic import BaseModel, field_validator
import re

class RunSqlQueryArgs(BaseModel):
    query: str
    database: str = "default"

    @field_validator("query")
    @classmethod
    def must_be_select(cls, v: str) -> str:
        stripped = v.strip().upper()
        if not stripped.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")
        # Reject obvious injection patterns
        for keyword in ("DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE"):
            if keyword in stripped:
                raise ValueError(f"Forbidden keyword in query: {keyword}")
        return v

    @field_validator("database")
    @classmethod
    def safe_db_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Database name must be alphanumeric with underscores only.")
        return v
```

## Exercise 2 — GitHub repo resource URI scheme

```python
GITHUB_RESOURCES = [
    {
        "uri": "github://repos/owner/repo/readme",
        "name": "Repository README",
        "description": "README.md of the repository",
        "mimeType": "text/markdown",
    },
    {
        "uri": "github://repos/owner/repo/pulls/open",
        "name": "Open Pull Requests",
        "description": "List of open pull requests with titles and numbers",
        "mimeType": "application/json",
    },
    {
        "uri": "github://repos/owner/repo/issues/open",
        "name": "Open Issues",
        "description": "List of open issues",
        "mimeType": "application/json",
    },
    {
        "uri": "github://repos/owner/repo/contents/{path}",
        "name": "File Content",
        "description": "Raw content of a file at the given path in the default branch",
        "mimeType": "text/plain",  # caller should check file extension for actual type
    },
    {
        "uri": "github://repos/owner/repo/commits/main",
        "name": "Recent Commits on Main",
        "description": "Last 20 commits on the main branch",
        "mimeType": "application/json",
    },
]
```

## Exercise 3 — compare_documents prompt + test

```python
# In mcp_primitives_demo.py — add to handle_prompts_get or as a second handler:

class CompareDocumentsArgs(BaseModel):
    uri_a: str
    uri_b: str

def handle_compare_documents_prompt(arguments: dict) -> dict:
    """Render a prompt comparing two documents."""
    try:
        args = CompareDocumentsArgs.model_validate(arguments)
    except ValidationError as exc:
        return {"error": str(exc)}

    result_a = handle_resources_read(args.uri_a)
    result_b = handle_resources_read(args.uri_b)

    if "error" in result_a:
        return {"error": f"Resource A: {result_a['error']}"}
    if "error" in result_b:
        return {"error": f"Resource B: {result_b['error']}"}

    text_a = result_a["contents"][0]["text"]
    text_b = result_b["contents"][0]["text"]

    return {
        "messages": [
            {
                "role": "system",
                "content": {
                    "type": "text",
                    "text": "You are a technical analyst. Compare documents objectively.",
                },
            },
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": (
                        f"Compare the following two documents.\n\n"
                        f"Document A ({args.uri_a}):\n{text_a}\n\n"
                        f"Document B ({args.uri_b}):\n{text_b}\n\n"
                        f"Identify key differences, similarities, and which is more complete."
                    ),
                },
            },
        ]
    }
```

```python
# tests/test_compare_prompt.py
import pytest
from unittest.mock import patch

def test_compare_documents_two_messages():
    """compare_documents returns system + user messages."""
    from mcp_primitives_demo import handle_compare_documents_prompt

    with patch("mcp_primitives_demo.handle_resources_read") as mock_read:
        def side_effect(uri):
            return {"contents": [{"text": f"Content of {uri}"}]}
        mock_read.side_effect = side_effect

        result = handle_compare_documents_prompt({
            "uri_a": "docs://catalog/overview",
            "uri_b": "docs://catalog/schema",
        })

    assert "messages" in result
    assert len(result["messages"]) == 2
    roles = [m["role"] for m in result["messages"]]
    assert "system" in roles
    assert "user" in roles


def test_compare_documents_embeds_content():
    """compare_documents embeds both document contents in the user message."""
    from mcp_primitives_demo import handle_compare_documents_prompt

    with patch("mcp_primitives_demo.handle_resources_read") as mock_read:
        mock_read.side_effect = lambda uri: {
            "contents": [{"text": f"text_for_{uri.split('/')[-1]}"}]
        }

        result = handle_compare_documents_prompt({
            "uri_a": "docs://catalog/overview",
            "uri_b": "docs://catalog/schema",
        })

    user_msg = next(m for m in result["messages"] if m["role"] == "user")
    text = user_msg["content"]["text"]
    assert "text_for_overview" in text
    assert "text_for_schema" in text


def test_compare_documents_unknown_resource():
    """compare_documents returns error if a resource URI is unknown."""
    from mcp_primitives_demo import handle_compare_documents_prompt

    result = handle_compare_documents_prompt({
        "uri_a": "docs://catalog/overview",
        "uri_b": "docs://catalog/nonexistent",
    })

    assert "error" in result
```

## Exercise 4 — Primitive classification

```python
"""
Classification rationale:

1. Current stock price → TOOL
   Reason: calls an external API at invocation time; the result is dynamic and ephemeral.
   Even though it's read-only, "fetch current data on demand" = tool.

2. Internal style guide (static Markdown) → RESOURCE
   Reason: static file the host can attach to context; no execution needed.
   URI: docs://style-guide (or file:///path/to/style-guide.md)

3. Multi-step code review checklist template → PROMPT
   Reason: encodes a reusable LLM workflow pattern; takes (diff, guidelines) as args
   and returns structured messages that prime the LLM for a specific review mode.

4. Delete a record from a database → TOOL
   Reason: has side effects (destructs data). Always a tool, never a resource.
   Should include a confirmation field in the schema.

5. Today's calendar events (read-only, dynamic) → TOOL or RESOURCE depending on usage
   - As RESOURCE: if the host wants to attach "today's events" as background context
     before every LLM call. URI: calendar://events/today
   - As TOOL: if the LLM should decide when to fetch (e.g., "what's on my calendar?").
   Both are valid; the choice depends on whether the host or LLM drives the fetch.
"""
```
