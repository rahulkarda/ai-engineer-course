# Solutions — Phase 5 · Week 2 · Day 6 — Typed Tool-Calling Framework (Reference Implementation)

## Overview

Complete reference implementation of the typed tool-calling framework. All code lives under `src/tool_framework/`. Tests use `unittest.mock` — no real LLM calls.

---

## Module: `src/tool_framework/registry.py`

```python
"""ToolRegistry: decorator-based registration with auto-schema generation."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import create_model

logger = logging.getLogger(__name__)


def _python_type_to_json_schema(annotation: Any) -> dict:
    """Map a Python type annotation to a JSON Schema type dict."""
    mapping: dict[Any, dict] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    return mapping.get(annotation, {"type": "string"})


def _build_input_schema(fn: Callable) -> dict:
    """Build an Anthropic-compatible input_schema from a function's type hints."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, str)
        properties[param_name] = _python_type_to_json_schema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    """Registry of callable tools with auto-extracted metadata."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def tool(self, fn: Callable) -> Callable:
        """Decorator: register a function as a tool.

        Extracts name from function name, description from docstring,
        and builds JSON schema from type hints.
        """
        name = fn.__name__
        description = (inspect.getdoc(fn) or "").strip()
        schema = _build_input_schema(fn)

        self._tools[name] = fn
        self._schemas[name] = {
            "name": name,
            "description": description,
            "input_schema": schema,
        }
        logger.debug("Registered tool: %s", name)
        return fn

    def get_tools_list(self) -> list[dict]:
        """Return Anthropic-format tools array."""
        return list(self._schemas.values())

    def get_tool(self, name: str) -> Callable | None:
        """Return the callable for *name*, or None if not registered."""
        return self._tools.get(name)
```

---

## Module: `src/tool_framework/dispatcher.py`

```python
"""ToolDispatcher: validates inputs and calls the registered tool."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ValidationError, create_model

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


def _make_validation_model(registry: ToolRegistry, tool_name: str) -> type[BaseModel] | None:
    """Build a Pydantic model matching the tool's input schema."""
    import inspect
    from typing import get_type_hints

    fn = registry.get_tool(tool_name)
    if fn is None:
        return None

    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, str)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, ...)
        else:
            fields[param_name] = (annotation, param.default)

    return create_model(f"{tool_name}_inputs", **fields)


class ToolDispatcher:
    """Dispatches tool_use blocks to registered tools."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def dispatch(self, tool_use_block: dict) -> dict:
        """Execute the tool described by *tool_use_block*.

        Args:
            tool_use_block: dict with keys ``id``, ``name``, ``input``.

        Returns:
            A tool_result dict suitable for the Anthropic messages API.
        """
        tool_id: str = tool_use_block.get("id", "")
        tool_name: str = tool_use_block.get("name", "")
        raw_input: dict = tool_use_block.get("input", {})

        fn = self._registry.get_tool(tool_name)
        if fn is None:
            logger.warning("Unknown tool: %s", tool_name)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Unknown tool: {tool_name}",
                "is_error": True,
            }

        # Validate with Pydantic
        model_cls = _make_validation_model(self._registry, tool_name)
        if model_cls is None:
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": "Could not build validation model",
                "is_error": True,
            }

        try:
            validated = model_cls(**raw_input)
        except ValidationError as exc:
            logger.error("Validation error for %s: %s", tool_name, exc)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(exc),
                "is_error": True,
            }

        # Execute the tool
        try:
            result = fn(**validated.model_dump())
            logger.debug("Tool %s returned: %r", tool_name, result)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            }
        except Exception as exc:
            logger.exception("Runtime error in tool %s: %s", tool_name, exc)
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": f"Tool execution error: {exc}",
                "is_error": True,
            }
```

---

## Module: `src/tool_framework/exceptions.py`

```python
"""Custom exceptions for the tool framework."""


class ToolLoopError(RuntimeError):
    """Raised when the agentic loop exceeds max_iterations."""
```

---

## Module: `src/tool_framework/loop.py`

```python
"""ToolLoop: agentic loop that drives Claude with tools."""

from __future__ import annotations

import logging
import os

import anthropic

from .dispatcher import ToolDispatcher
from .exceptions import ToolLoopError
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolLoop:
    """Drives the Anthropic tool-use loop until end_turn or max_iterations."""

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
        """Run the agentic loop.

        Args:
            messages: Initial messages list (mutated in-place during the loop).
            model: Claude model ID.
            max_iterations: Guard against infinite loops.

        Returns:
            The final text response from the model.

        Raises:
            ToolLoopError: If max_iterations is exceeded.
        """
        tools = self._registry.get_tools_list()

        for iteration in range(max_iterations):
            logger.debug("ToolLoop iteration %d", iteration)

            response = self._client.messages.create(
                model=model,
                max_tokens=4096,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                # Extract final text content
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if response.stop_reason == "tool_use":
                # Append assistant message
                messages.append(
                    {"role": "assistant", "content": response.content}
                )

                # Dispatch all tool_use blocks
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_block = {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        result = self._dispatcher.dispatch(tool_block)
                        tool_results.append(result)

                # Append user message with all tool results
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop_reason
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        raise ToolLoopError(f"ToolLoop exceeded max_iterations={max_iterations}")
```

---

## Tests: `tests/test_tool_framework_day6.py`

```python
"""Day-6 test suite — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tool_framework.dispatcher import ToolDispatcher
from src.tool_framework.exceptions import ToolLoopError
from src.tool_framework.loop import ToolLoop
from src.tool_framework.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def add(x: int, y: int) -> int:
        """Add two integers together."""
        return x + y

    @reg.tool
    def greet(name: str) -> str:
        """Return a greeting for the given name."""
        return f"Hello, {name}!"

    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(registry)


# ---------------------------------------------------------------------------
# TestToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_register_stores_callable(self, registry: ToolRegistry) -> None:
        fn = registry.get_tool("add")
        assert fn is not None
        assert fn(2, 3) == 5

    def test_get_tools_list_length(self, registry: ToolRegistry) -> None:
        tools = registry.get_tools_list()
        assert len(tools) == 2

    def test_get_tools_list_schema_structure(self, registry: ToolRegistry) -> None:
        tools = registry.get_tools_list()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_schema_for_typed_function(self, registry: ToolRegistry) -> None:
        tools = {t["name"]: t for t in registry.get_tools_list()}
        add_tool = tools["add"]
        props = add_tool["input_schema"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "integer"
        assert add_tool["input_schema"]["required"] == ["x", "y"]

    def test_description_from_docstring(self, registry: ToolRegistry) -> None:
        tools = {t["name"]: t for t in registry.get_tools_list()}
        assert tools["add"]["description"] == "Add two integers together."

    def test_get_tool_unknown_returns_none(self, registry: ToolRegistry) -> None:
        assert registry.get_tool("nonexistent") is None


# ---------------------------------------------------------------------------
# TestToolDispatcher
# ---------------------------------------------------------------------------


class TestToolDispatcher:
    def test_successful_dispatch(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "tool_1", "name": "add", "input": {"x": 3, "y": 4}}
        )
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_1"
        assert result["content"] == "7"
        assert "is_error" not in result

    def test_validation_error_returns_is_error(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "tool_2", "name": "add", "input": {"x": "not_an_int", "y": 4}}
        )
        assert result["is_error"] is True
        assert result["tool_use_id"] == "tool_2"

    def test_runtime_exception_returns_is_error(self, registry: ToolRegistry) -> None:
        @registry.tool
        def boom(msg: str) -> str:
            """Always raises."""
            raise ValueError("Intentional error")

        dispatcher = ToolDispatcher(registry)
        result = dispatcher.dispatch(
            {"id": "tool_3", "name": "boom", "input": {"msg": "test"}}
        )
        assert result["is_error"] is True
        assert "Intentional error" in result["content"]

    def test_unknown_tool_returns_is_error(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "tool_4", "name": "unknown_tool", "input": {}}
        )
        assert result["is_error"] is True


# ---------------------------------------------------------------------------
# TestToolLoop
# ---------------------------------------------------------------------------


def _make_mock_response(stop_reason: str, content: list) -> MagicMock:
    """Build a fake Anthropic response object."""
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(
    tool_id: str, name: str, input_data: dict
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


class TestToolLoop:
    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_end_turn_immediately(
        self, mock_anthropic_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            "end_turn", [_make_text_block("All done.")]
        )

        loop = ToolLoop(registry, dispatcher)
        result = loop.run([{"role": "user", "content": "hello"}])
        assert result == "All done."
        assert mock_client.messages.create.call_count == 1

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_single_tool_call_cycle(
        self, mock_anthropic_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        tool_block = _make_tool_use_block("t1", "add", {"x": 2, "y": 3})
        mock_client.messages.create.side_effect = [
            _make_mock_response("tool_use", [tool_block]),
            _make_mock_response("end_turn", [_make_text_block("Result: 5")]),
        ]

        loop = ToolLoop(registry, dispatcher)
        result = loop.run([{"role": "user", "content": "What is 2 + 3?"}])
        assert result == "Result: 5"
        assert mock_client.messages.create.call_count == 2

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_max_iterations_raises(
        self, mock_anthropic_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher
    ) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # Always return tool_use so the loop never ends
        tool_block = _make_tool_use_block("t1", "add", {"x": 1, "y": 1})
        mock_client.messages.create.return_value = _make_mock_response(
            "tool_use", [tool_block]
        )

        loop = ToolLoop(registry, dispatcher)
        with pytest.raises(ToolLoopError):
            loop.run(
                [{"role": "user", "content": "loop forever"}],
                max_iterations=3,
            )
        assert mock_client.messages.create.call_count == 3

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_multiple_tool_blocks_dispatched(
        self, mock_anthropic_cls: MagicMock, registry: ToolRegistry, dispatcher: ToolDispatcher
    ) -> None:
        """When a response contains two tool_use blocks, both are dispatched."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        t1 = _make_tool_use_block("t1", "add", {"x": 1, "y": 2})
        t2 = _make_tool_use_block("t2", "add", {"x": 10, "y": 20})
        mock_client.messages.create.side_effect = [
            _make_mock_response("tool_use", [t1, t2]),
            _make_mock_response("end_turn", [_make_text_block("Done")]),
        ]

        loop = ToolLoop(registry, dispatcher)
        messages = [{"role": "user", "content": "two calls"}]
        loop.run(messages)

        # The user message appended after the tool_use response should have 2 results
        tool_results_msg = messages[-1]
        assert tool_results_msg["role"] == "user"
        assert len(tool_results_msg["content"]) == 2
```

---

## Key Design Decisions

- `_build_input_schema` iterates `inspect.signature` — order matches the function signature and required/optional is determined by presence of a default value.
- `_make_validation_model` in the dispatcher rebuilds a Pydantic model on every dispatch call; for production use, cache it keyed by tool name.
- `ToolLoop` mutates the `messages` list in-place so callers can inspect the full conversation history after `run()` returns.
- Tests patch `anthropic.Anthropic` at the import path in `loop.py`, not at the `anthropic` package level, to ensure the patch applies to the exact object used.
