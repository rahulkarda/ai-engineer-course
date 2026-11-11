# Phase 5 · Week 08 · Day 7 (Sunday) — Project + Review

> Objective: Complete the week project, reflect on the MCP design, and prepare for assessment  ·  Time: ~3 hr project + 1 hr review  ·  Prereqs: [[day-6]]

## Part 1 — Week Project Completion (3 hr)

See [`project.md`](./project.md) for the full project specification. Today you finalize the project deliverables, clean up the code, add missing tests, and write a brief architectural reflection.

### Checklist before submission

- [ ] All tools, resources, and prompts are implemented with docstrings
- [ ] All tool handlers validate inputs with Pydantic; bad args return `isError: True` responses, they do not crash the server
- [ ] All resources use meaningful URI schemes (not opaque IDs)
- [ ] Test suite covers: tool happy path, tool error case, resource read, prompt render, adapter tool loading
- [ ] Zero `print()` calls — use `logging`
- [ ] No hardcoded secrets; `ANTHROPIC_API_KEY` via `os.getenv`
- [ ] `pyproject.toml` (or `requirements.txt`) lists all dependencies: `mcp`, `langchain-anthropic`, `langgraph`, `pydantic`, `pytest`, `pytest-asyncio`

### Code quality pass

Run before finalizing:

```bash
# Lint (ruff must be installed)
ruff check .

# Type check (optional but recommended)
mypy . --ignore-missing-imports

# Tests
pytest tests/ -v
```

Fix any ruff violations. If mypy reports errors, investigate — don't just add `type: ignore` without understanding why.

### Reflection document

Write a brief comment block (or docstring in `task_server.py`) answering:

1. What security controls does your server implement?
2. What would you change if this server needed to be multi-tenant (many users, each with their own tasks)?
3. What capability would you add if this were a production tool?

## Part 2 — Week Review (1 hr)

### Core concepts to lock in

Work through these questions from memory before looking back at the lessons.

**MCP fundamentals**
- What is the M×N integration problem? How does MCP reduce it?
- What protocol does MCP build on? What are the three message types?
- What happens during the `initialize` handshake?

**Architecture**
- What is the exact constraint on a client-to-server relationship?
- How does the host enforce security policy in an MCP system?
- What are the two official MCP transports and when is each appropriate?

**Primitives**
- A tool vs a resource: what is the key difference?
- Give two examples of capabilities that look like tools but are actually resources.
- What does a prompt response contain, exactly?

**Building servers**
- How does FastMCP convert a Python type hint to a JSON Schema field?
- What happens when a tool handler raises an exception?
- Why should MCP server log output go to stderr, not stdout?

**Building clients and agents**
- What is the lifecycle of a `ClientSession`?
- How do you convert an MCP tool descriptor into a LangChain `StructuredTool`?
- What are three prompt injection risks specific to MCP?

### Common mistakes review

| Mistake | Correct pattern |
|---|---|
| Returning raw exceptions from tool handlers | Catch and return `isError: True` with a message |
| Logging to stdout in a stdio server | Always log to stderr |
| Trusting tool arguments without validation | Validate with Pydantic before executing |
| Using opaque resource URIs (`resource://x/abc123`) | Use descriptive URIs (`tasks://pending`, `db://users/42`) |
| Embedding secrets in resource URIs | Credentials in env vars, never in protocol messages |
| Writing tool descriptions that are vague | Precise, action-oriented descriptions — the LLM decides based on this |

### Looking ahead

**Week 09** builds on MCP with the **Agent-to-Agent (A2A) protocol** — how multiple agents communicate and delegate tasks to each other. MCP handles tool/data integration; A2A handles agent-to-agent coordination.

Key question to hold in mind: when an agent calls another agent via A2A, how is that different from the same agent calling an MCP tool? (Hint: state, autonomy, and delegation semantics.)

## Exercises

1. **Design review**: Read through your `task_server.py`. For each tool, write one sentence explaining exactly what the LLM needs to know to decide when to call that tool. Is the docstring sufficient?

2. **Transport migration**: Your task server currently uses stdio. Sketch what changes would be required to switch to HTTP+SSE. What breaks? What stays the same?

3. **Multi-server agent**: Design (don't implement) an agent that connects to both the task server and a calendar MCP server. What does the host need to do to help the LLM route tool calls to the right server? Sketch the tool naming strategy.

4. **Assessment preparation**: Review the assessment rubric in `assessment.md`. Identify your weakest dimension and spend 20 minutes on targeted practice.

(Solutions: ../solutions/week-08/day-7.md)

## Self-check

- [ ] I completed all project deliverables and they pass tests
- [ ] I can answer all week review questions from memory
- [ ] I understand the distinction between tools, resources, and prompts at an intuitive level
- [ ] I can explain the MCP security threat model and enumerate key mitigations
- [ ] I am ready for the assessment

---

**Supplementary**: [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) · [MCP specification](https://spec.modelcontextprotocol.io) · [MCP community servers](https://modelcontextprotocol.io/servers)
