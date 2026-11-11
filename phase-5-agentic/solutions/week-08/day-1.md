# Solutions — Week 08 · Day 1

## Exercise 1 — Protocol message sequence

```
CLIENT                          SERVER
  │──initialize ──────────────►  │  (request, id=1)
  │◄──── initialize response ────│  (result with server capabilities)
  │──initialized ─────────────►  │  (notification, no id, no response)
  │                               │
  │──resources/list ───────────►  │  (request, id=2)
  │◄──── resources/list result ──│  (list of resource descriptors)
  │                               │
  │──resources/read ───────────►  │  (request, id=3, params: {uri: "..."})
  │◄──── resources/read result ──│  (contents: [{uri, mimeType, text}])
  │                               │
  │──tools/call ───────────────►  │  (request, id=4, params: {name, arguments})
  │◄──── tools/call result ──────│  (content: [{type, text}], isError: bool)
```

Key points:
- `initialized` is a notification (no `id` field) — the server does NOT respond
- All other exchanges are request/response pairs identified by matching `id`
- The client increments `id` for each new request

## Exercise 2 — PostgreSQL server capability design

```python
"""
PostgreSQL MCP server capability design.

Tools (executable, may have side effects):
  - execute_query(sql: str, params: list = []) -> str
      Why tool: reads data but query execution is an action; parameterized to prevent injection.
  - execute_write(sql: str, params: list = []) -> str
      Why tool: DML (INSERT/UPDATE/DELETE) has side effects.
  - explain_query(sql: str) -> str
      Why tool: EXPLAIN runs a query plan analysis (technically read-only but an action).
  - list_tables(schema: str = "public") -> str
      Why tool: dynamic — queries pg_tables at call time; host may cache via resource instead.

Resources (read-only reference data):
  - db://schemas/{schema}/tables  → JSON list of table names in a schema
      Why resource: static metadata, suitable for context injection.
  - db://schemas/{schema}/tables/{table}/columns  → JSON column definitions
      Why resource: DDL rarely changes; good for background context.
  - db://schemas/{schema}/tables/{table}/sample  → first 5 rows as JSON
      Why resource: sample data for LLM reasoning; host controls when to fetch.
  - db://version  → PostgreSQL version string
      Why resource: truly static, never needs to be a tool.

Prompts (parameterized templates):
  - query_builder(table: str, task_description: str)
      Why prompt: encodes the expert pattern of providing schema + asking LLM to write SQL.
  - explain_slow_query(query: str, explain_output: str)
      Why prompt: packages a multi-step analysis workflow.

Security notes:
  - execute_write should require explicit confirmation capability from host
  - execute_query should be rate-limited
  - All SQL goes through parameterized execution, never string interpolation
"""
```

## Exercise 3 — Ecosystem research (sample findings)

```python
"""
MCP servers relevant to data/ML work (from modelcontextprotocol.io/servers):

1. Postgres MCP Server
   Tools: query (SELECT only), describe_table
   Resources: db://schemas, db://tables/{name}/schema
   Use case: give LLM read access to a database without write risk.

2. Filesystem MCP Server
   Tools: read_file, write_file, list_directory, search_files, create_directory
   Resources: file:///{path} (any file in allowed root)
   Use case: agent that reads and writes local files (notebooks, configs, data).

3. GitHub MCP Server
   Tools: create_issue, create_pull_request, search_code, get_file_contents,
          push_files, list_commits, create_branch
   Resources: github://repos/{owner}/{repo}/contents/{path}
   Use case: ML ops agent that can open PRs, check CI, review code changes.
"""
```

## Exercise 4 — Error case demo

```python
# Append to mcp_protocol_demo.py

def demo_error_case() -> None:
    """Show message flow when tools/call is called with missing required argument."""
    logger.info("\n=== tools/call with missing required argument ===")

    # Client sends a call missing the required 'city' argument
    request = make_request(
        method="tools/call",
        params={"name": "get_weather", "arguments": {}},  # missing 'city'
        request_id=4,
    )
    logger.info("CLIENT -> SERVER (bad args):\n%s", request)

    # Server responds with JSON-RPC error
    # -32602 = Invalid params (standard JSON-RPC error code)
    error_response = make_error_response(
        code=-32602,
        message="Invalid params: 'city' is required but was not provided.",
        request_id=4,
    )
    logger.info("SERVER -> CLIENT (error):\n%s", error_response)

    # Note: this is a PROTOCOL error (bad request shape).
    # For TOOL execution errors (tool ran but returned bad data),
    # the correct pattern is a successful JSON-RPC response with isError: true:
    tool_execution_error = make_response(
        result={
            "content": [{"type": "text", "text": "City 'Unknown' not found in weather service."}],
            "isError": True,
        },
        request_id=5,
    )
    logger.info(
        "Contrast — tool execution error (successful RPC, tool-level error):\n%s",
        tool_execution_error,
    )
```

Key distinction:
- JSON-RPC `-32602` error: the *protocol* request was malformed (missing required field before even calling the tool)
- `isError: true` in the result: the *tool* ran but the tool logic failed (e.g., city not found)
