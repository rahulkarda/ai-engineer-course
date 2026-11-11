# Week 08 Assessment — Model Context Protocol

## Conceptual (interview-style)

Answer each question as if in a technical interview. Aim for clarity and precision — not length.

1. A colleague says "MCP is just another plugin system, like LangChain tools." How do you respond? What does MCP provide that a framework-specific plugin system does not?

2. Explain the role of `initialize` / `initialized` in the MCP session lifecycle. What would go wrong if a client sent `tools/call` before the handshake completed?

3. You are designing an MCP server for a vector database. Classify each of the following as a tool, resource, or prompt, and justify:
   - Semantic search (given a query, return top-K documents)
   - The collection's current index statistics (document count, dimension)
   - A query planning template that primes the LLM to decompose a complex question into sub-queries

4. A tool's inputSchema marks `file_path` as required. An attacker supplies `../../etc/passwd` as the value. What layer of defense should your server implement, and at what point in the request handling pipeline?

5. Your MCP server is behind an HTTP/SSE transport. Describe the two-channel design: what goes over the GET /sse channel versus the POST /messages channel, and why are two channels needed?

6. Explain the difference between a `ToolResult` with `isError: false` containing an error message in the text, versus one with `isError: true`. When should each be used?

7. An LLM-based agent is connected to an MCP server that wraps a third-party data source. The data source returns user-controlled text (e.g., document content). What attack vector does this create, and what is the mitigation?

## Coding Challenge

**Spec: Build a minimal but complete MCP server and test it in-process.**

Implement `metrics_server.py` — an MCP server using FastMCP that tracks numeric metrics.

### Requirements

**Data model**: A `Metric` Pydantic model with fields: `name: str`, `value: float`, `unit: str`, `recorded_at: str` (ISO timestamp).

**Tools**:

| Tool | Signature | Behavior |
|---|---|---|
| `record_metric` | `name: str, value: float, unit: str = "count"` | Store the metric, return `"Recorded {name}={value} {unit}"` |
| `get_metric_history` | `name: str, last_n: int = 10` | Return last N values as formatted string, or error if unknown metric |
| `get_metric_average` | `name: str` | Return `"Average of {name}: {avg:.2f} {unit}"` or error if unknown |
| `list_metric_names` | _(no args)_ | Return newline-separated list of all tracked metric names |

**Resources**:

| URI | Content |
|---|---|
| `metrics://all` | JSON array of all metrics (newest first) |
| `metrics://metric/{name}` | JSON of all records for one metric name |
| `metrics://summary` | JSON object: `{name: {count, min, max, avg, unit}}` for each metric |

**Prompts**:

| Prompt | Args | Returns |
|---|---|---|
| `analyze_metric` | `name: str, goal: str` | User message: "Analyze [name] history. Goal: [goal]. Data: [embedded resource content]" |

**Test requirements** (in `tests/test_metrics_server.py`):

Write at least 8 tests. Required coverage:
- `record_metric` creates a metric and returns expected string
- `get_metric_history` returns correct count when last_n < total records
- `get_metric_history` returns error string (not raises) for unknown metric
- `get_metric_average` returns correct average (float math, assert with tolerance)
- `list_metric_names` returns all names after multiple records
- `metrics://all` resource returns valid JSON list
- `metrics://summary` resource contains min, max, avg keys for a recorded metric
- `analyze_metric` prompt embeds metric name and goal in the returned message

**Code quality requirements**:
- Type hints on all functions
- Pydantic validation on all tool inputs
- `logging` not `print`
- All resource URIs are meaningful
- No hardcoded values (metric store is in-memory dict, not hardcoded)

### Acceptance criteria

- All 8+ tests pass with `pytest tests/ -v`
- Tests run without `ANTHROPIC_API_KEY` and without spawning subprocesses
- `ruff check .` passes with no errors
- Server starts with `python metrics_server.py` (even if you immediately Ctrl+C)

## Rubric (pass ≥ 75%)

| Dimension | Weight | What earns full marks |
|---|---|---|
| Independent implementation | 40% | All tools, resources, and prompt implemented correctly per spec; test suite passes |
| Conceptual understanding | 25% | Conceptual questions answered precisely, with correct terminology |
| Debugging & problem-solving | 20% | Error cases handled gracefully (isError, validation, unknown metric); no crashes |
| Code quality & testing | 15% | Type hints, Pydantic, logging, ≥8 tests, ruff clean |

**Pass gate**: score ≥ 75% overall; the independent implementation dimension alone must score ≥ 60%.

Answer key: ../solutions/week-08/assessment.md
