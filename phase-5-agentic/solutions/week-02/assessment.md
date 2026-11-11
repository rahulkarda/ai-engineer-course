# Answer Key — Week 2 Assessment — Tool & Function Calling

---

## Conceptual Questions

### Q1. What is the difference between `tool_use` and `tool_result` content blocks?

**`tool_use`** is a content block that appears in an **assistant** message. Claude produces it when it decides to invoke a tool. It contains three fields:
- `id` — a unique identifier for this specific invocation (e.g., `"toolu_01abc..."`).
- `name` — the name of the tool to call (must match a registered tool).
- `input` — a JSON object of arguments matching the tool's `input_schema`.

**`tool_result`** is a content block that appears in a **user** message sent *back to* Claude after the tool runs. It contains:
- `tool_use_id` — the `id` from the corresponding `tool_use` block (required to correlate request to response).
- `content` — the string output of the tool execution.
- `is_error` (optional, boolean) — set to `true` if the tool raised an exception or returned an error condition.

In short: `tool_use` is Claude's *request*, `tool_result` is the application's *response*.

---

### Q2. How does Claude decide which tool to call? What role does the tool description play?

Claude is a language model; it reads the full context — the system prompt, conversation history, and the `tools` array — and generates the most likely next tokens. It does **not** perform keyword matching or routing logic.

The **tool description** is the primary signal. Claude uses it to understand:
- *What the tool does* — a precise, verb-led summary (e.g., "Execute a SQL SELECT query and return rows as JSON").
- *When to use it* — implicit from the description's scope. Overlapping descriptions confuse the model.
- *What the inputs mean* — descriptions on individual `input_schema` properties (via the `description` field in JSON Schema) further guide argument selection.

Practical rules:
1. Be specific: "Search the web for recent news" is better than "Search".
2. Describe the *output* too: "Returns a float representing degrees Celsius".
3. Distinguish tools that might seem similar (e.g., `read_file` vs `search_files`).

---

### Q3. What does `stop_reason: "tool_use"` mean vs `"end_turn"`?

| `stop_reason` | Meaning | Required action |
|---|---|---|
| `"tool_use"` | Claude produced one or more `tool_use` blocks and is *paused*, waiting for results before it can continue reasoning. | Execute every `tool_use` block in `response.content`, collect the `tool_result` dicts, append an assistant message (the raw response content) and a user message (the tool results), then call the API again. |
| `"end_turn"` | Claude has finished reasoning and produced its final response. There are no pending tool calls. | Extract the `text` block from `response.content` and return it to the user. The agentic loop is complete. |

Other possible values: `"max_tokens"` (output token limit hit — may need to truncate or increase `max_tokens`) and `"stop_sequence"` (a custom stop sequence matched).

---

### Q4. Why should tool inputs be validated with Pydantic rather than used directly?

Claude generates `input` as a JSON object, but:
1. **Type coercion errors** — the model might send `"42"` (string) for an `int` field.
2. **Missing required fields** — the model might omit a field it "thinks" is optional.
3. **Malformed values** — an `expression` field might contain arbitrary code instead of a math expression.
4. **Security** — unvalidated inputs passed directly to `subprocess`, `eval`, SQL, or file paths create injection vulnerabilities.

Pydantic provides:
- Automatic type coercion with clear error messages.
- `required` enforcement — `ValidationError` if a field is missing.
- Custom validators (`@field_validator`) for domain rules.
- A structured `ValidationError` you can serialize and return as a `tool_result` with `is_error: true`, giving Claude the chance to correct its call.

Using `**raw_input` directly means a `TypeError` or silent wrong-type behavior with no useful error propagation back to the model.

---

### Q5. When Claude returns two `tool_use` blocks in one response, how many user messages should you send back?

**One user message**, containing a list of two `tool_result` content blocks — one for each `tool_use` block.

```python
# Correct — one user message, multiple tool_result blocks
messages.append({
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "result A"},
        {"type": "tool_result", "tool_use_id": "toolu_02", "content": "result B"},
    ]
})
```

Sending two separate user messages (one per result) would violate the Anthropic API's message alternation requirement (user/assistant/user/...) and would also cause the second result to appear in a new conversational turn, losing the association with the assistant message that requested both tools. Claude requires all results for a single assistant turn to be bundled in one user reply.

---

### Q6. What is idempotency? Give one example of an idempotent tool and one non-idempotent tool.

**Idempotency**: An operation is idempotent if calling it multiple times with the same inputs produces the same result and leaves the system in the same state as calling it once.

**Idempotent tool example: `get_weather(city: str) -> dict`**
Calling it 3 times for `"Berlin"` returns the current weather 3 times — no side effects, no state change. (Assuming read-only access to a weather API.)

**Non-idempotent tool example: `send_email(to: str, subject: str, body: str) -> str`**
Calling it 3 times sends 3 emails. Each call has an irreversible side effect. If the agentic loop retries due to a transient error (network timeout *after* the email was sent), the recipient receives duplicates.

**Why it matters for tool design:**
- Idempotent tools are safe to retry — use `RetryConfig` freely.
- Non-idempotent tools need deduplication logic (idempotency keys, check-before-act patterns) or should not be retried automatically.
- Tag non-idempotent tools clearly in their description so Claude (and developers) understand the consequences.

---

## Coding Challenge Solution

### `mini_framework.py`

```python
"""Mini tool-calling framework — assessment coding challenge solution."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from typing import Any, get_type_hints

import anthropic
from pydantic import ValidationError, create_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolLoopError(RuntimeError):
    """Raised when the agentic loop exceeds max_iterations."""


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def _py_to_json_schema(annotation: Any) -> dict:
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }.get(annotation, {"type": "string"})


class ToolRegistry:
    """Decorator-based tool registry with auto-schema generation."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._meta: dict[str, dict] = {}

    def register(self, fn: Callable) -> Callable:
        """Register *fn* as a tool (use as a decorator)."""
        name = fn.__name__
        description = (inspect.getdoc(fn) or "").strip()

        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            annotation = hints.get(param_name, str)
            properties[param_name] = _py_to_json_schema(annotation)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        input_schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            input_schema["required"] = required

        self._tools[name] = fn
        self._meta[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
        return fn

    def get_tools_list(self) -> list[dict]:
        """Return Anthropic-format tools array."""
        return list(self._meta.values())

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------


class ToolDispatcher:
    """Validates inputs with Pydantic, executes the tool, handles errors."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def _build_model(self, tool_name: str):
        fn = self._registry.get_tool(tool_name)
        if fn is None:
            return None
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            annotation = hints.get(param_name, str)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (annotation, default)
        return create_model(f"{tool_name}_model", **fields)

    def dispatch(self, tool_use_block: dict) -> dict:
        tool_id = tool_use_block.get("id", "")
        tool_name = tool_use_block.get("name", "")
        raw_input = tool_use_block.get("input", {})

        fn = self._registry.get_tool(tool_name)
        if fn is None:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Unknown tool: {tool_name}",
                "is_error": True,
            }

        model_cls = self._build_model(tool_name)
        try:
            validated = model_cls(**raw_input)
        except ValidationError as exc:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(exc),
                "is_error": True,
            }

        try:
            result = fn(**validated.model_dump())
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            }
        except Exception as exc:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Error: {exc}",
                "is_error": True,
            }


# ---------------------------------------------------------------------------
# ToolLoop
# ---------------------------------------------------------------------------


class ToolLoop:
    """Agentic loop: send messages, handle tool_use, return final text."""

    def __init__(self, registry: ToolRegistry, dispatcher: ToolDispatcher) -> None:
        self._registry = registry
        self._dispatcher = dispatcher
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-5-5",
        max_iterations: int = 10,
    ) -> str:
        tools = self._registry.get_tools_list()
        for _ in range(max_iterations):
            response = self._client.messages.create(
                model=model, max_tokens=4096, tools=tools, messages=messages
            )
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""
            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                results = [
                    self._dispatcher.dispatch(
                        {"id": b.id, "name": b.name, "input": b.input}
                    )
                    for b in response.content
                    if b.type == "tool_use"
                ]
                messages.append({"role": "user", "content": results})
        raise ToolLoopError(f"Exceeded max_iterations={max_iterations}")
```

---

### `test_mini_framework.py`

```python
"""Assessment test suite — at least 5 tests, all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mini_framework import ToolDispatcher, ToolLoop, ToolLoopError, ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def square(n: int) -> int:
        """Return n squared."""
        return n * n

    @reg.register
    def concat(a: str, b: str) -> str:
        """Concatenate two strings."""
        return a + b

    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(registry)


# --- Test 1: registry stores and returns correct metadata ---
def test_registry_tools_list(registry: ToolRegistry) -> None:
    tools = registry.get_tools_list()
    names = [t["name"] for t in tools]
    assert "square" in names
    assert "concat" in names
    square_tool = next(t for t in tools if t["name"] == "square")
    assert square_tool["input_schema"]["properties"]["n"]["type"] == "integer"
    assert square_tool["input_schema"]["required"] == ["n"]


# --- Test 2: dispatcher executes successfully ---
def test_dispatcher_success(dispatcher: ToolDispatcher) -> None:
    result = dispatcher.dispatch({"id": "t1", "name": "square", "input": {"n": 5}})
    assert result["content"] == "25"
    assert "is_error" not in result


# --- Test 3: dispatcher returns is_error on validation failure ---
def test_dispatcher_validation_error(dispatcher: ToolDispatcher) -> None:
    result = dispatcher.dispatch(
        {"id": "t2", "name": "square", "input": {"n": "not_an_int"}}
    )
    assert result["is_error"] is True


# --- Test 4: dispatcher returns is_error on runtime exception ---
def test_dispatcher_runtime_error(registry: ToolRegistry) -> None:
    @registry.register
    def explode(msg: str) -> str:
        """Always raises."""
        raise RuntimeError("boom")

    d = ToolDispatcher(registry)
    result = d.dispatch({"id": "t3", "name": "explode", "input": {"msg": "x"}})
    assert result["is_error"] is True
    assert "boom" in result["content"]


# --- Test 5: loop returns text on end_turn ---
@patch("mini_framework.anthropic.Anthropic")
def test_loop_end_turn(mock_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher) -> None:
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    text_block = MagicMock()
    text_block.text = "Done!"
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    mock_client.messages.create.return_value = response

    loop = ToolLoop(registry, dispatcher)
    result = loop.run([{"role": "user", "content": "hi"}])
    assert result == "Done!"


# --- Test 6: loop handles tool_use then end_turn ---
@patch("mini_framework.anthropic.Anthropic")
def test_loop_tool_call_cycle(mock_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher) -> None:
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "t1"
    tool_block.name = "square"
    tool_block.input = {"n": 9}

    text_block = MagicMock()
    text_block.text = "81"

    r1 = MagicMock()
    r1.stop_reason = "tool_use"
    r1.content = [tool_block]

    r2 = MagicMock()
    r2.stop_reason = "end_turn"
    r2.content = [text_block]

    mock_client.messages.create.side_effect = [r1, r2]

    loop = ToolLoop(registry, dispatcher)
    result = loop.run([{"role": "user", "content": "square 9"}])
    assert result == "81"
    assert mock_client.messages.create.call_count == 2


# --- Test 7: loop raises ToolLoopError on max_iterations ---
@patch("mini_framework.anthropic.Anthropic")
def test_loop_max_iterations(mock_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher) -> None:
    mock_client = MagicMock()
    mock_cls.return_value = mock_client

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "t1"
    tool_block.name = "square"
    tool_block.input = {"n": 1}

    r = MagicMock()
    r.stop_reason = "tool_use"
    r.content = [tool_block]
    mock_client.messages.create.return_value = r

    loop = ToolLoop(registry, dispatcher)
    with pytest.raises(ToolLoopError):
        loop.run([{"role": "user", "content": "loop"}], max_iterations=2)
```

---

## Grading Rubric

| Criterion | Points |
|---|---|
| Q1–Q6 conceptual answers, each fully correct | 10 pts each (60 total) |
| `ToolRegistry.register` decorator works, correct schema | 10 pts |
| `ToolDispatcher.dispatch` with Pydantic validation | 10 pts |
| `ToolLoop.run` with max_iterations guard | 10 pts |
| Test suite: ≥5 tests, all LLM calls mocked | 10 pts |
| **Total** | **100 pts** |

Partial credit awarded for conceptual questions that are correct in substance but imprecise in terminology.
