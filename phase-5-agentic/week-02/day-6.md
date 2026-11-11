# Phase 5 · Week 02 · Day 6 — Build a Typed Tool-Calling Framework Part 1 (Saturday Deep Practice)

> Objective: Build a production-ready typed tool-calling framework from scratch — ToolRegistry, ToolDispatcher, and ToolLoop — with full pytest coverage.  ·  Time: ~4 hrs  ·  Prereqs: [[day-5]]

---

## Overview

Today you build three components that together form a reusable, typed tool-calling framework:

| Component | Responsibility |
|---|---|
| `ToolRegistry` | Register Python functions as tools; auto-generate JSON schema |
| `ToolDispatcher` | Validate inputs with Pydantic; call function; return typed result |
| `ToolLoop` | Drive the full Anthropic tool_use loop until `end_turn` |

All three are tested with a mocked Anthropic client — no real API calls.

Estimated time breakdown:
- Part 1 — ToolRegistry: ~90 min
- Part 2 — ToolDispatcher: ~60 min
- Part 3 — ToolLoop: ~60 min
- Tests + polish: ~30 min

---

## Concept

### Why build a framework?

The day-5 worked example hard-codes tool dispatch. A real agent codebase needs:
- Register tools at import time (decorator pattern)
- Validate inputs before calling (Pydantic)
- Consistent error handling across all tools
- Loop logic reusable for any set of tools

### Architecture

```
ToolRegistry
  .register(fn)          → adds tool, generates schema
  .get_schema()          → returns list[dict] for API call
  .get_tool(name)        → returns callable + input model

ToolDispatcher
  .dispatch(tool_use_block) → validates input, calls fn, returns tool_result block

ToolLoop
  .run(messages)         → drives tool_use loop, returns final text
```

---

## Part 1 — ToolRegistry

### Concept

The registry wraps Python functions into the Anthropic tool schema format. It:
1. Extracts parameter names and types via `inspect` + `typing`
2. Builds the JSON schema for each parameter
3. Stores a Pydantic model for runtime validation

### How it works internally

```python
"""
framework/registry.py
"""
import inspect
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

# Python type → JSON Schema type mapping
_PY_TO_JSON: dict[type, str] = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _py_type_to_json_schema(annotation: type) -> dict[str, str]:
    """Convert a Python type annotation to a JSON Schema type dict."""
    json_type = _PY_TO_JSON.get(annotation, "string")
    return {"type": json_type}


class ToolEntry:
    """Holds everything needed to call a registered tool."""

    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        input_model: type[BaseModel],
        schema: dict[str, Any],
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.input_model = input_model
        self.schema = schema  # full Anthropic tool schema dict


class ToolRegistry:
    """Registry that maps tool names to ToolEntry objects.

    Usage:
        registry = ToolRegistry()

        @registry.register
        def search(query: str, max_results: int = 5) -> dict:
            '''Search the product catalogue by keyword.'''
            ...
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(self, fn: Callable) -> Callable:
        """Decorator: register a function as a tool.

        Extracts name, description (from docstring), and parameter schema
        from the function signature and type hints.
        """
        name = fn.__name__
        description = (fn.__doc__ or "").strip().split("\n")[0]  # first line only
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)

        # Build properties dict and required list for JSON Schema
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            annotation = hints.get(param_name, str)
            prop = _py_type_to_json_schema(annotation)

            # Use param default as description hint if it is a string (rare, but explicit)
            prop["description"] = param_name.replace("_", " ").capitalize()

            properties[param_name] = prop

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build a dynamic Pydantic model for input validation
        field_definitions: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            annotation = hints.get(param_name, str)
            if param.default is inspect.Parameter.empty:
                field_definitions[param_name] = (annotation, ...)
            else:
                field_definitions[param_name] = (annotation, param.default)

        input_model: type[BaseModel] = create_model(
            f"{name.title().replace('_', '')}Input",
            **field_definitions,
        )

        schema: dict[str, Any] = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        entry = ToolEntry(
            fn=fn,
            name=name,
            description=description,
            input_model=input_model,
            schema=schema,
        )
        self._tools[name] = entry
        logger.debug("Registered tool: %s", name)
        return fn  # return original function unmodified

    def get_schema(self) -> list[dict[str, Any]]:
        """Return list of Anthropic tool schema dicts for all registered tools."""
        return [entry.schema for entry in self._tools.values()]

    def get_tool(self, name: str) -> ToolEntry:
        """Return ToolEntry by name, raise KeyError if not found."""
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name!r}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
```

### Worked example — registry in action

```python
# demo_registry.py
import logging
from framework.registry import ToolRegistry

logging.basicConfig(level=logging.DEBUG)

registry = ToolRegistry()


@registry.register
def search_products(query: str, max_results: int = 5) -> dict:
    """Search the product catalogue by keyword. Returns matching products."""
    return {"results": []}


@registry.register
def calculate_total(prices: list, quantity: int) -> dict:
    """Calculate subtotal, tax, and total for a list of prices."""
    subtotal = sum(prices) * quantity
    return {"subtotal": subtotal, "tax": subtotal * 0.08, "total": subtotal * 1.08}


print("Registered tools:", len(registry))
import json
print(json.dumps(registry.get_schema(), indent=2))
```

Output (formatted):
```json
[
  {
    "name": "search_products",
    "description": "Search the product catalogue by keyword. Returns matching products.",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Query"},
        "max_results": {"type": "integer", "description": "Max results"}
      },
      "required": ["query"]
    }
  },
  ...
]
```

---

## Part 2 — ToolDispatcher

### Concept

The dispatcher takes a `tool_use` block from Claude's response, validates inputs against the Pydantic model, calls the function, and returns a properly formatted `tool_result` block. It centralises error handling so the loop never crashes on bad tool calls.

```python
"""
framework/dispatcher.py
"""
import logging
from typing import Any

from pydantic import ValidationError

from framework.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolCallError(Exception):
    """Raised when a tool call fails irrecoverably."""


class ToolDispatcher:
    """Validates tool inputs, dispatches calls, and formats results.

    Args:
        registry: The ToolRegistry containing registered tools.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def dispatch(self, tool_id: str, tool_name: str, raw_inputs: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs, call the tool, return a tool_result block.

        Returns a dict conforming to the Anthropic tool_result content block.
        Never raises — errors are returned as is_error tool_results.
        """
        logger.info("Dispatching tool=%s id=%s", tool_name, tool_id)

        # Look up tool
        try:
            entry = self._registry.get_tool(tool_name)
        except KeyError:
            return self._error_result(tool_id, f"Unknown tool: {tool_name!r}")

        # Validate inputs with Pydantic
        try:
            validated = entry.input_model(**raw_inputs)
        except ValidationError as exc:
            logger.warning("Validation failed for tool=%s: %s", tool_name, exc)
            return self._error_result(tool_id, f"Input validation failed: {exc}")

        # Call the function
        try:
            result = entry.fn(**validated.model_dump())
            logger.debug("Tool %s returned: %s", tool_name, result)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            }
        except Exception as exc:
            logger.exception("Tool %s raised: %s", tool_name, exc)
            return self._error_result(tool_id, f"Tool raised {type(exc).__name__}: {exc}")

    @staticmethod
    def _error_result(tool_id: str, message: str) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": message,
            "is_error": True,
        }
```

### Worked example — dispatcher in isolation

```python
# demo_dispatcher.py
from framework.registry import ToolRegistry
from framework.dispatcher import ToolDispatcher

registry = ToolRegistry()

@registry.register
def add_numbers(left: float, right: float) -> dict:
    """Add two numbers together."""
    return {"result": left + right}

dispatcher = ToolDispatcher(registry)

# Happy path
result = dispatcher.dispatch("tu_001", "add_numbers", {"left": 3.0, "right": 4.5})
print(result)
# {'type': 'tool_result', 'tool_use_id': 'tu_001', 'content': "{'result': 7.5}"}

# Validation error
result = dispatcher.dispatch("tu_002", "add_numbers", {"left": "not_a_number", "right": 4.5})
print(result)
# {'type': 'tool_result', 'tool_use_id': 'tu_002', 'content': 'Input validation failed: ...', 'is_error': True}

# Unknown tool
result = dispatcher.dispatch("tu_003", "nonexistent", {})
print(result)
# {'type': 'tool_result', 'tool_use_id': 'tu_003', 'content': "Unknown tool: 'nonexistent'", 'is_error': True}
```

---

## Part 3 — ToolLoop

### Concept

The loop orchestrates the full Anthropic message cycle: send a message, detect `tool_use`, dispatch, send results, repeat until `end_turn` (or a max-iteration guard).

```python
"""
framework/loop.py
"""
import logging
from typing import Any

import anthropic

from framework.dispatcher import ToolDispatcher
from framework.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # guard against infinite loops


class ToolLoop:
    """Drives the full Anthropic tool_use loop.

    Args:
        client: Anthropic client (real or mock).
        registry: ToolRegistry with tools to expose.
        model: Model ID to use for all calls.
        max_tokens: Max tokens per response.
        max_iterations: Safety cap on loop iterations.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        registry: ToolRegistry,
        model: str = "claude-opus-5-5",
        max_tokens: int = 1024,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self._client = client
        self._registry = registry
        self._dispatcher = ToolDispatcher(registry)
        self._model = model
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations

    def run(self, user_message: str) -> str:
        """Run the tool loop for a single user message. Return the final text."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        tools = self._registry.get_schema()

        for iteration in range(self._max_iterations):
            logger.info("Loop iteration %d", iteration + 1)

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                tools=tools,
                messages=messages,
            )
            logger.debug("stop_reason=%s", response.stop_reason)

            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            if response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                logger.info("Dispatching %d tool call(s)", len(tool_calls))

                tool_results = [
                    self._dispatcher.dispatch(tc.id, tc.name, tc.input)
                    for tc in tool_calls
                ]

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                logger.warning("Unexpected stop_reason=%s, aborting loop", response.stop_reason)
                break

        logger.error("Max iterations (%d) reached without end_turn", self._max_iterations)
        return ""

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text from the first text content block."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
```

### Worked example — full loop with mock

```python
"""
demo_full_loop.py
Complete demo: registry → dispatcher → loop, all mocked.
"""
import logging
from unittest.mock import MagicMock, patch

import anthropic

from framework.registry import ToolRegistry
from framework.loop import ToolLoop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Register tools ---------------------------------------------------------

registry = ToolRegistry()

@registry.register
def get_weather(city: str) -> dict:
    """Get current weather for a city (stub)."""
    logger.info("get_weather: city=%s", city)
    return {"city": city, "temp_c": 18, "condition": "partly cloudy"}

@registry.register
def convert_celsius_to_fahrenheit(celsius: float) -> dict:
    """Convert a Celsius temperature to Fahrenheit."""
    return {"fahrenheit": celsius * 9 / 5 + 32}

# --- Build mock client ------------------------------------------------------

def build_mock_client() -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)

    # Turn 1: tool_use for get_weather
    weather_block = MagicMock()
    weather_block.type = "tool_use"
    weather_block.id = "tu_weather_1"
    weather_block.name = "get_weather"
    weather_block.input = {"city": "Berlin"}

    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.content = [weather_block]

    # Turn 2: tool_use for convert_celsius_to_fahrenheit
    convert_block = MagicMock()
    convert_block.type = "tool_use"
    convert_block.id = "tu_convert_1"
    convert_block.name = "convert_celsius_to_fahrenheit"
    convert_block.input = {"celsius": 18.0}

    resp2 = MagicMock()
    resp2.stop_reason = "tool_use"
    resp2.content = [convert_block]

    # Turn 3: final answer
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Berlin is 18°C (64.4°F) and partly cloudy."

    resp3 = MagicMock()
    resp3.stop_reason = "end_turn"
    resp3.content = [text_block]

    client.messages.create.side_effect = [resp1, resp2, resp3]
    return client


if __name__ == "__main__":
    mock_client = build_mock_client()
    loop = ToolLoop(client=mock_client, registry=registry)
    answer = loop.run("What is the weather in Berlin in Fahrenheit?")
    logger.info("Answer: %s", answer)
```

---

## Tests

```python
"""
tests/test_framework_day6.py
Full pytest suite for ToolRegistry, ToolDispatcher, and ToolLoop.
All LLM calls are mocked — no real API calls.
"""
import pytest
from unittest.mock import MagicMock, call
from typing import Any

import anthropic

from framework.registry import ToolRegistry, ToolEntry
from framework.dispatcher import ToolDispatcher
from framework.loop import ToolLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def add(left: float, right: float) -> dict:
        """Add two numbers."""
        return {"result": left + right}

    @reg.register
    def greet(name: str, formal: bool = False) -> dict:
        """Greet a person by name."""
        prefix = "Dear" if formal else "Hi"
        return {"message": f"{prefix} {name}"}

    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(registry)


def make_mock_client(responses: list) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = responses
    return client


def tool_use_block(tool_id: str, name: str, inputs: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = inputs
    return block


def text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def end_turn_response(answer: str) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [text_block(answer)]
    return resp


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_adds_tool(self, registry: ToolRegistry) -> None:
        assert "add" in registry
        assert "greet" in registry

    def test_len(self, registry: ToolRegistry) -> None:
        assert len(registry) == 2

    def test_get_schema_returns_list(self, registry: ToolRegistry) -> None:
        schemas = registry.get_schema()
        assert isinstance(schemas, list)
        assert len(schemas) == 2

    def test_schema_has_required_keys(self, registry: ToolRegistry) -> None:
        schema = registry.get_schema()[0]
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_required_params_listed(self, registry: ToolRegistry) -> None:
        add_schema = next(s for s in registry.get_schema() if s["name"] == "add")
        assert "left" in add_schema["input_schema"]["required"]
        assert "right" in add_schema["input_schema"]["required"]

    def test_optional_param_not_required(self, registry: ToolRegistry) -> None:
        greet_schema = next(s for s in registry.get_schema() if s["name"] == "greet")
        assert "formal" not in greet_schema["input_schema"]["required"]

    def test_get_tool_returns_entry(self, registry: ToolRegistry) -> None:
        entry = registry.get_tool("add")
        assert isinstance(entry, ToolEntry)
        assert entry.name == "add"

    def test_get_tool_raises_for_unknown(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError):
            registry.get_tool("nonexistent")

    def test_description_from_docstring(self, registry: ToolRegistry) -> None:
        entry = registry.get_tool("add")
        assert "Add two numbers" in entry.description

    def test_input_model_validates_correct(self, registry: ToolRegistry) -> None:
        entry = registry.get_tool("add")
        model = entry.input_model(left=1.0, right=2.0)
        assert model.left == 1.0

    def test_input_model_rejects_wrong_type(self, registry: ToolRegistry) -> None:
        from pydantic import ValidationError
        entry = registry.get_tool("add")
        with pytest.raises(ValidationError):
            entry.input_model(left="not_a_float", right=2.0)

    def test_register_returns_original_function(self, registry: ToolRegistry) -> None:
        """Decorator must not wrap the function — calling it directly still works."""
        entry = registry.get_tool("add")
        result = entry.fn(left=1.0, right=2.0)
        assert result == {"result": 3.0}


# ---------------------------------------------------------------------------
# ToolDispatcher tests
# ---------------------------------------------------------------------------

class TestToolDispatcher:
    def test_happy_path(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch("tu_1", "add", {"left": 1.0, "right": 2.0})
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tu_1"
        assert "is_error" not in result

    def test_unknown_tool_returns_error(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch("tu_2", "unknown", {})
        assert result["is_error"] is True
        assert "Unknown tool" in result["content"]

    def test_validation_failure_returns_error(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch("tu_3", "add", {"left": "bad", "right": 2.0})
        assert result["is_error"] is True
        assert "validation" in result["content"].lower()

    def test_tool_exception_returns_error(self, registry: ToolRegistry) -> None:
        @registry.register
        def explode(x: int) -> dict:
            """Always raises."""
            raise RuntimeError("boom")

        dispatcher = ToolDispatcher(registry)
        result = dispatcher.dispatch("tu_4", "explode", {"x": 1})
        assert result["is_error"] is True
        assert "RuntimeError" in result["content"]

    def test_result_content_is_string(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch("tu_5", "greet", {"name": "Alice"})
        assert isinstance(result["content"], str)


# ---------------------------------------------------------------------------
# ToolLoop tests
# ---------------------------------------------------------------------------

class TestToolLoop:
    def test_single_tool_call_then_end(self, registry: ToolRegistry) -> None:
        add_block = tool_use_block("tu_add", "add", {"left": 3.0, "right": 4.0})
        resp1 = MagicMock(stop_reason="tool_use", content=[add_block])
        resp2 = end_turn_response("3 + 4 = 7")

        client = make_mock_client([resp1, resp2])
        loop = ToolLoop(client=client, registry=registry)
        answer = loop.run("What is 3 + 4?")

        assert answer == "3 + 4 = 7"
        assert client.messages.create.call_count == 2

    def test_direct_end_turn(self, registry: ToolRegistry) -> None:
        resp = end_turn_response("No tools needed.")
        client = make_mock_client([resp])
        loop = ToolLoop(client=client, registry=registry)
        answer = loop.run("Say hello")
        assert answer == "No tools needed."

    def test_max_iterations_guard(self, registry: ToolRegistry) -> None:
        """Loop must abort after max_iterations even if stop_reason is always tool_use."""
        add_block = tool_use_block("tu_add", "add", {"left": 1.0, "right": 1.0})
        infinite_resp = MagicMock(stop_reason="tool_use", content=[add_block])

        client = make_mock_client([infinite_resp] * 20)
        loop = ToolLoop(client=client, registry=registry, max_iterations=3)
        answer = loop.run("Keep adding forever")

        assert answer == ""  # no end_turn, returns empty
        assert client.messages.create.call_count == 3

    def test_tool_results_sent_in_user_message(self, registry: ToolRegistry) -> None:
        add_block = tool_use_block("tu_add", "add", {"left": 1.0, "right": 2.0})
        resp1 = MagicMock(stop_reason="tool_use", content=[add_block])
        resp2 = end_turn_response("Answer is 3")

        client = make_mock_client([resp1, resp2])
        loop = ToolLoop(client=client, registry=registry)
        loop.run("Add 1 and 2")

        # Second call should have a user message containing tool_result
        second_call_args = client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        last_user_msg = messages[-1]
        assert last_user_msg["role"] == "user"
        assert isinstance(last_user_msg["content"], list)
        assert last_user_msg["content"][0]["type"] == "tool_result"

    def test_parallel_tool_calls_dispatched(self, registry: ToolRegistry) -> None:
        block1 = tool_use_block("tu_1", "add", {"left": 1.0, "right": 2.0})
        block2 = tool_use_block("tu_2", "greet", {"name": "Bob"})
        resp1 = MagicMock(stop_reason="tool_use", content=[block1, block2])
        resp2 = end_turn_response("Both done")

        client = make_mock_client([resp1, resp2])
        loop = ToolLoop(client=client, registry=registry)
        answer = loop.run("Do both tasks")

        assert answer == "Both done"
        # Both tool_results should be in the follow-up user message
        second_call_args = client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_results = messages[-1]["content"]
        assert len(tool_results) == 2
```

---

## Exercises (no answers)

1. Extend `ToolRegistry` to support tools with `list[str]` parameters — currently the schema just records `"type": "array"`. Add an `items` field with `{"type": "string"}` when the annotation is `list[str]`.

2. Add a `ToolRegistry.unregister(name: str)` method and write tests for it (including the error case when the tool doesn't exist).

3. Modify `ToolDispatcher.dispatch` to accept a `timeout_seconds: float = 5.0` parameter. Use `concurrent.futures` or `asyncio` with a timeout to abort tool calls that take too long, returning an error result.

4. Write a `ToolLoop.run_async` variant that dispatches all tool calls in a single iteration concurrently using `asyncio.gather`.

5. Add a `tool_call_count` property to `ToolLoop` that tracks how many tool calls have been dispatched in the current run.

6. Extend the registry decorator to accept an explicit `description` parameter that overrides the docstring: `@registry.register(description="Custom description")`.

(Solutions: ../solutions/week-02/day-6.md)

---

## Self-check

- [ ] Can you explain why `create_model` from Pydantic is used instead of a fixed BaseModel subclass?
- [ ] What is `inspect.Parameter.empty` checking for and why does it matter for the `required` list?
- [ ] Why does `ToolDispatcher.dispatch` never raise an exception?
- [ ] Where in `ToolLoop.run` does the assistant turn get appended to messages and why is it critical?
- [ ] What is the risk of setting `max_iterations` too high? Too low?
- [ ] How would you add support for tools defined as Pydantic BaseModel subclasses instead of bare functions?

---

## Supplementary links

- [Anthropic tool use — full guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Pydantic `create_model` docs](https://docs.pydantic.dev/latest/api/main/#pydantic.main.create_model)
- [Python `inspect` module](https://docs.python.org/3/library/inspect.html)
- [Python `typing.get_type_hints`](https://docs.python.org/3/library/typing.html#typing.get_type_hints)
- [pytest fixtures](https://docs.pytest.org/en/stable/reference/fixtures.html)
- [JSON Schema — object type](https://json-schema.org/understanding-json-schema/reference/object.html)
