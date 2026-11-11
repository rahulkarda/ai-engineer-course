# Solutions — Week 08 · Day 7

## Exercise 1 — Design review notes (reference)

Sample assessment of `task_server.py` tool docstrings vs. what the LLM needs:

```python
"""
Tool: create_task
Current docstring: "Create a new task with a title, description, and priority level."
LLM needs to know: call this when user asks to add/create/track a new piece of work.
Verdict: adequate — action-oriented, parameter descriptions are clear.

Tool: complete_task
Current docstring: "Mark a task as completed by its ID."
LLM needs to know: use this after finding the task ID via list_tasks or get_task_detail.
Improvement: "Mark a task as completed. Use list_tasks first to find the task ID."

Tool: list_tasks
Current: "List tasks filtered by status."
Improvement: "List tasks. Call with status='pending' to see what needs doing,
status='completed' to review what was done, status='all' for everything."

Tool: get_task_detail
Current: "Get the full details of a task as JSON, including notes and timestamps."
Verdict: good — explains what extra info it provides over list_tasks.

Tool: add_note_to_task
Current: "Add a text note to an existing task."
Verdict: adequate. Could add: "Use to record progress updates or relevant information."
"""
```

## Exercise 2 — Transport migration sketch

```python
"""
Changes required to switch task_server.py from stdio to HTTP+SSE:

Changes to server code:
  1. Entry point: replace mcp.run(transport="stdio") with
     mcp.run(transport="sse", host="0.0.0.0", port=8001)
  2. Logging: no change needed — stderr is already separate from transport
  3. CORS: may need to add CORS headers if client is from a different origin
  4. Auth: add Bearer token middleware (not in stdio model)
  5. Process lifecycle: server now runs as a long-lived service, not a subprocess

Changes to client (mcp_adapter.py):
  1. Replace stdio_client import with sse_client
  2. Replace StdioServerParameters with the SSE endpoint URL
  3. connect_stdio context manager becomes connect_sse
  4. Remove command/args parameters; add url parameter
  5. Handle connection retries (network is unreliable; stdin/stdout pipe is not)

What stays the same:
  - All FastMCP tool/resource/prompt decorators — no change
  - ClientSession API — identical
  - load_tools, read_resource, get_prompt — no change
  - test mocks — no change (tests mock the session, not the transport)
  - TASKS in-memory store — no change (though HTTP = multi-process = need shared DB)

What breaks:
  - In-memory TASKS dict: each process has its own copy
    Fix: move to Redis, SQLite, or PostgreSQL
  - Single-process assumption: HTTP servers can have multiple workers
    Fix: use a proper persistence layer
"""
```

## Exercise 3 — Multi-server agent design

```python
"""
Multi-server agent design: task_server + calendar_server

Problem: when the LLM calls a tool, how does it know which server to use?

Option A — Namespace tools by server
  Tool names: "tasks.create_task", "tasks.list_tasks",
              "calendar.get_events", "calendar.create_event"
  Pros: LLM sees clear ownership; no ambiguity
  Cons: tool names are longer; some models prefer short names

Option B — Unique names per server (no prefix)
  Tool names: "create_task", "list_tasks", "get_calendar_events", "create_calendar_event"
  Pros: shorter, more natural names
  Cons: requires careful naming at server design time; breaks if servers use same names

Recommended: Option A (namespacing) for a multi-server agent.

Adapter changes:
  - load_tools receives server_name parameter
  - Prepend server_name to tool name: f"{server_name}__{tool.name}"
  - Or wrap tool names with a prefix at StructuredTool creation time

Host orchestration:
  - Maintain two separate ClientSession objects (one per server)
  - Build tools list by loading from both sessions
  - Pass combined tools list to the agent
  - The StructuredTool closure already captures the correct session per tool

Code sketch:
  async with connect_stdio("python", ["task_server.py"]) as (task_session, task_tools):
      async with connect_stdio("python", ["calendar_server.py"]) as (cal_session, cal_tools):
          all_tools = [
              rename_tool(t, f"tasks_{t.name}") for t in task_tools
          ] + [
              rename_tool(t, f"calendar_{t.name}") for t in cal_tools
          ]
          agent = build_agent(all_tools)
          result = await agent.ainvoke(...)
"""
```

## Exercise 4 — Assessment preparation guidance

The most commonly weak dimension is **Debugging & problem-solving** (20%). Focus areas:

```python
"""
Common bugs in MCP implementations and how to catch them:

1. Tool handler crashes → server dies
   How to catch: test with invalid arguments; assert server returns isError=True
   Fix: wrap handler body in try/except, return isError response

2. stdio server writing to stdout for logging
   How to catch: run server, check that stdout contains only valid JSON-RPC
   Fix: configure logging handler to use stderr only

3. Resource URI template not matching
   How to catch: call the resource with a known param; check response
   Fix: ensure {param} in URI template matches function parameter name exactly

4. Pydantic validation silent failure
   How to catch: pass wrong type (e.g., string for int); check error message
   Fix: use model_validate (raises ValidationError) not model_construct

5. asyncio blocking in tool handler
   How to catch: add a tool that does time.sleep(1); call two tools concurrently;
                 measure if they serialize (bad) or run concurrently (good)
   Fix: use asyncio.to_thread for blocking calls

6. Tool closure bug (loop variable capture)
   How to catch: load two tools, call each — do they use each other's session?
   Fix: use a factory function (_make_caller(name, session)) not a lambda
"""
```
