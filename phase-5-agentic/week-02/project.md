# Week 02 Project — Typed Tool-Calling Framework

> Build a production-quality, reusable typed tool-calling framework for the Anthropic API.

---

## Project Overview

You will design and implement a Python package (`tool_framework`) that abstracts the Anthropic tool-use loop behind a clean, typed API. The framework must be usable by any downstream agent — import the package, register tools with a decorator, and call `loop.run()` to get an answer.

This is not a toy. The framework should be robust enough that you could ship it as an internal library.

---

## Background

The raw Anthropic API requires consumers to:
1. Maintain a growing message list
2. Detect `stop_reason == "tool_use"` manually
3. Dispatch tools and format results correctly
4. Retry on transient failures
5. Guard against infinite loops

A framework removes all of that from application code. Application code should look like this:

```python
registry = ToolRegistry()

@registry.register
def get_price(sku: str) -> dict:
    """Look up the current price for a product SKU."""
    ...

loop = ToolLoop(client=anthropic_client, registry=registry)
answer = loop.run("What is the price of SKU-12345?")
```

---

## Requirements

### R1 — ToolRegistry

- `register` decorator: zero-config registration of any Python function
- Auto-extract parameter names, types, and defaults from the function signature using `inspect` and `typing.get_type_hints`
- Build JSON Schema `input_schema` automatically (string, integer, number, boolean, array, object)
- Use first line of docstring as the tool description (required — raise `ValueError` at registration time if docstring is missing)
- Create a dynamic Pydantic `BaseModel` for each tool's inputs using `create_model`
- `get_schema() -> list[dict]`: return Anthropic-compatible tool schema list
- `get_tool(name: str) -> ToolEntry`: return entry or raise `KeyError`
- `__contains__` and `__len__` support

### R2 — ToolDispatcher

- `dispatch(tool_id, name, raw_inputs) -> dict`: always returns a `tool_result` block, never raises
- Validate inputs with the tool's Pydantic model before calling the function
- Return `is_error: True` result for: unknown tool, validation failure, unhandled exception
- Accept a `RetryConfig` for configurable retry on transient exceptions
- Log every dispatch at INFO level (tool name, id, success/failure)

### R3 — ToolLoop

- `run(user_message: str, system: str | None = None) -> str`
- Drive the full tool_use loop: send → detect tool_use → dispatch all → send results → repeat
- Dispatch multiple tool calls per iteration (parallel calls)
- `max_iterations` guard (default 10) — return `""` when exhausted, or raise `LoopExhaustedError` if `raise_on_exhausted=True`
- Accept `model`, `max_tokens`, `retry_config` at construction time
- Never hard-code model IDs — default to `"claude-sonnet-5-5"` but make it overridable

### R4 — Configuration

- `pydantic-settings` `FrameworkConfig` class
- Read `ANTHROPIC_API_KEY` from environment (never hard-code)
- Configurable via env vars: model, max_tokens, max_iterations, retry attempts
- `.env.example` file listing all variables

### R5 — Error handling

- Typed exception hierarchy: `ToolFrameworkError` base, with `ToolNotFoundError`, `ToolValidationError`, `ToolExecutionError`, `LoopExhaustedError`
- `RetryConfig` dataclass: `max_attempts`, `backoff_seconds`, `retryable_exceptions`
- `call_with_retry` function: exponential backoff, logs every retry attempt

### R6 — Logging

- `logging` throughout — no `print` statements in framework code
- Every tool dispatch logged at INFO
- Retry attempts logged at WARNING
- Loop completion (iterations used) logged at INFO
- Exceptions logged at ERROR or EXCEPTION

### R7 — Test suite

- `tests/unit/test_registry.py` — registry tests
- `tests/unit/test_dispatcher.py` — dispatcher + retry tests
- `tests/unit/test_loop.py` — loop integration tests
- `tests/unit/test_config.py` — config tests
- All LLM calls mocked with `unittest.mock.MagicMock`
- No real API calls in any test
- `pytest.fixture` for registry and mock client
- Coverage target: 80%+ lines

---

## Project Structure

```
src/
  tool_framework/
    __init__.py         ← public API re-exports
    registry.py
    dispatcher.py
    loop.py
    retry.py
    exceptions.py
    config.py
tests/
  unit/
    test_registry.py
    test_dispatcher.py
    test_loop.py
    test_config.py
demo_agent.py           ← runnable demo (uses real API if ANTHROPIC_API_KEY set)
.env.example
pyproject.toml
```

---

## Deliverables Checklist

- [ ] `src/tool_framework/__init__.py` exports all public names
- [ ] `ToolRegistry` with `register`, `get_schema`, `get_tool`, `__contains__`, `__len__`
- [ ] `ToolDispatcher` with `dispatch` (validation + retry)
- [ ] `ToolLoop` with `run` (full loop, max_iterations guard)
- [ ] `RetryConfig` dataclass + `call_with_retry` function
- [ ] Typed exception hierarchy (`ToolFrameworkError` and 4 subclasses)
- [ ] `FrameworkConfig` via `pydantic-settings`
- [ ] `.env.example` with all env vars documented
- [ ] `pyproject.toml` with `[project]` + `[tool.ruff]` sections
- [ ] 4 test files, all LLM calls mocked
- [ ] `demo_agent.py` with at least 3 registered tools and a demo loop
- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] No hardcoded API keys anywhere

---

## Acceptance Criteria

| Criterion | Pass condition |
|---|---|
| Type safety | Every function, method, and class attribute is typed; no `Any` in public API signatures |
| No hardcoded keys | `grep -r "sk-ant" .` returns nothing; API key only via `os.getenv` |
| Logging | No `print()` calls in framework code |
| Model IDs | Only `claude-opus-5-5`, `claude-sonnet-5-5`, or `claude-haiku-5-5` appear in code |
| Tests pass | `pytest tests/` exits 0 |
| Ruff clean | `ruff check src/ tests/` exits 0 |
| Error paths covered | Unknown tool, validation failure, exhausted loop each have a test |
| No real API calls | `grep -r "anthropic.Anthropic()" tests/` finds no real client construction |
| Docstrings enforced | Registry raises `ValueError` if a function has no docstring |
| Demo works | `python demo_agent.py` runs without errors when `ANTHROPIC_API_KEY` is set |

---

## Stretch Goals

Complete these after the core requirements pass.

**S1 — Tool timeout**
Add `timeout_seconds: float | None = None` to `ToolDispatcher`. Use `concurrent.futures.ThreadPoolExecutor` to enforce a wall-clock limit per call. Return an `is_error` result on timeout.

**S2 — Async support**
Add `ToolLoop.run_async(user_message: str) -> Coroutine[str]` that dispatches all tool calls in a single iteration concurrently using `asyncio.gather`. Add `AsyncToolDispatcher` that wraps async tool functions.

**S3 — Streaming stub**
Add `ToolLoop.run_streaming(user_message: str) -> Iterator[str]` that yields text chunks using the Anthropic streaming API. The tool dispatch logic inside the stream handler should be identical to the non-streaming version.

**S4 — OpenTelemetry tracing**
Add optional OpenTelemetry span creation around each tool call and loop iteration. Controlled by `FrameworkConfig.enable_tracing: bool = False`.

**S5 — Tool metadata**
Extend the `register` decorator to accept optional `tags: list[str]` and `version: str`. Expose `ToolRegistry.list_tools(tag: str | None = None) -> list[str]` for filtering.

---

## Evaluation (same as assessment rubric)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Pass mark: 75%

---

Solution reference: ../solutions/week-02/project.md
