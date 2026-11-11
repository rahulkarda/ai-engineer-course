# Project Solution — Week 2 — Typed Tool-Calling Framework

## Directory Structure

```
src/tool_framework/
    __init__.py
    registry.py
    dispatcher.py
    loop.py
    models.py
    config.py
    exceptions.py
    retry.py
tests/
    unit/
        test_registry.py
        test_dispatcher.py
        test_loop.py
        test_retry.py
    integration/
        test_framework.py
demo/
    agent.py
pyproject.toml
```

---

## `src/tool_framework/exceptions.py`

```python
"""Custom exceptions for the tool framework."""


class ToolFrameworkError(Exception):
    """Base class for all tool-framework exceptions."""


class ToolNotFoundError(ToolFrameworkError):
    """Raised when a requested tool name is not registered."""


class ToolValidationError(ToolFrameworkError):
    """Raised when tool inputs fail Pydantic validation."""


class ToolExecutionError(ToolFrameworkError):
    """Raised when a tool raises an unexpected exception during execution."""


class ToolLoopError(ToolFrameworkError):
    """Raised when the agentic loop exceeds max_iterations."""
```

---

## `src/tool_framework/models.py`

```python
"""Pydantic models for tool-framework data structures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool_use block from an Anthropic response."""

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Represents a tool_result block sent back to Claude."""

    type: str = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a dict compatible with the Anthropic messages API."""
        d: dict[str, Any] = {
            "type": self.type,
            "tool_use_id": self.tool_use_id,
            "content": self.content,
        }
        if self.is_error:
            d["is_error"] = True
        return d


class ToolError(BaseModel):
    """Structured error returned by a tool execution."""

    tool_name: str
    error_type: str
    message: str
```

---

## `src/tool_framework/config.py`

```python
"""Framework configuration loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class FrameworkConfig(BaseSettings):
    """Runtime configuration for the tool framework.

    All fields can be overridden via environment variables prefixed with
    ``TOOL_FRAMEWORK_``, e.g. ``TOOL_FRAMEWORK_DEFAULT_MODEL=claude-haiku-5-5``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TOOL_FRAMEWORK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-5-5"
    max_iterations: int = 10
    log_level: str = "INFO"
```

---

## `src/tool_framework/retry.py`

```python
"""Retry utilities with exponential back-off."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig(BaseModel):
    """Configuration for retry behaviour."""

    max_attempts: int = 3
    backoff_factor: float = 0.5


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    config: RetryConfig,
    **kwargs: Any,
) -> T:
    """Call *fn* with retry on any exception.

    Sleep time on attempt *n* (0-indexed): ``config.backoff_factor * 2 ** n``.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        config: Retry configuration.
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn* on success.

    Raises:
        Exception: The last exception raised by *fn* after all attempts fail.
    """
    last_exc: Exception | None = None
    for attempt in range(config.max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < config.max_attempts - 1:
                sleep_time = config.backoff_factor * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed (%s: %s). Retrying in %.3fs.",
                    attempt + 1,
                    config.max_attempts,
                    type(exc).__name__,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    "All %d attempts failed. Last error: %s",
                    config.max_attempts,
                    exc,
                )
    assert last_exc is not None
    raise last_exc
```

---

## `src/tool_framework/registry.py`

```python
"""ToolRegistry: decorator-based registration with auto-schema generation."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)


def _python_type_to_json_schema(annotation: Any) -> dict:
    """Map a Python type annotation to a JSON Schema type descriptor."""
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
    """Build an Anthropic-compatible ``input_schema`` from a function signature."""
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
    """Registry of callable tools.

    Usage::

        registry = ToolRegistry()

        @registry.tool
        def my_tool(x: int, y: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return x + y
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def tool(self, fn: Callable) -> Callable:
        """Decorator: register *fn* as a tool.

        Auto-extracts name from ``fn.__name__``, description from docstring,
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
        """Return the Anthropic-format tools array."""
        return list(self._schemas.values())

    def get_tool(self, name: str) -> Callable | None:
        """Return the callable registered under *name*, or None."""
        return self._tools.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
```

---

## `src/tool_framework/dispatcher.py`

```python
"""ToolDispatcher: validates inputs with Pydantic, executes tools, handles errors."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import ValidationError, create_model

from .exceptions import ToolNotFoundError
from .models import ToolResult
from .registry import ToolRegistry
from .retry import RetryConfig, call_with_retry

logger = logging.getLogger(__name__)


def _make_validation_model(fn) -> type:
    """Build a Pydantic model matching *fn*'s signature."""
    import inspect
    from typing import get_type_hints

    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    return create_model(f"{fn.__name__}_inputs", **fields)


class ToolDispatcher:
    """Validates inputs and calls the registered tool.

    Args:
        registry: The ``ToolRegistry`` containing registered tools.
        retry_config: Optional retry configuration. When provided, tool
            execution errors trigger retries with exponential back-off.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._registry = registry
        self._retry_config = retry_config
        self._model_cache: dict[str, type] = {}

    def _get_model(self, tool_name: str) -> type | None:
        if tool_name not in self._model_cache:
            fn = self._registry.get_tool(tool_name)
            if fn is None:
                return None
            self._model_cache[tool_name] = _make_validation_model(fn)
        return self._model_cache[tool_name]

    def dispatch(self, tool_use_block: dict) -> dict:
        """Execute the tool described by *tool_use_block*.

        Args:
            tool_use_block: Dict with keys ``id``, ``name``, ``input``.

        Returns:
            A tool_result dict for the Anthropic messages API.
        """
        tool_id: str = tool_use_block.get("id", "")
        tool_name: str = tool_use_block.get("name", "")
        raw_input: dict = tool_use_block.get("input", {})

        fn = self._registry.get_tool(tool_name)
        if fn is None:
            logger.warning("Dispatch: unknown tool '%s'", tool_name)
            return ToolResult(
                tool_use_id=tool_id,
                content=f"Unknown tool: {tool_name}",
                is_error=True,
            ).to_dict()

        model_cls = self._get_model(tool_name)
        if model_cls is None:
            return ToolResult(
                tool_use_id=tool_id,
                content="Could not build validation model",
                is_error=True,
            ).to_dict()

        try:
            validated = model_cls(**raw_input)
        except ValidationError as exc:
            logger.error("Validation error for tool '%s': %s", tool_name, exc)
            return ToolResult(
                tool_use_id=tool_id,
                content=str(exc),
                is_error=True,
            ).to_dict()

        kwargs = validated.model_dump()
        try:
            if self._retry_config is not None:
                result = call_with_retry(fn, **kwargs, config=self._retry_config)
            else:
                result = fn(**kwargs)
            logger.debug("Tool '%s' succeeded: %r", tool_name, result)
            return ToolResult(
                tool_use_id=tool_id,
                content=str(result),
            ).to_dict()
        except Exception as exc:
            logger.exception("Runtime error in tool '%s': %s", tool_name, exc)
            return ToolResult(
                tool_use_id=tool_id,
                content=f"Tool execution error: {exc}",
                is_error=True,
            ).to_dict()
```

---

## `src/tool_framework/loop.py`

```python
"""ToolLoop: agentic loop that drives Claude with registered tools."""

from __future__ import annotations

import logging
import os

import anthropic

from .config import FrameworkConfig
from .dispatcher import ToolDispatcher
from .exceptions import ToolLoopError
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolLoop:
    """Drives the Anthropic tool-use loop until end_turn or max_iterations.

    Args:
        registry: Registry of available tools.
        dispatcher: Dispatcher for executing tool_use blocks.
        config: Framework configuration. If not provided, defaults are used.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        config: FrameworkConfig | None = None,
    ) -> None:
        self._registry = registry
        self._dispatcher = dispatcher
        self._config = config or FrameworkConfig()
        self._client = anthropic.Anthropic(
            api_key=self._config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        )

    def run(
        self,
        messages: list[dict],
        model: str | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """Run the agentic loop.

        Args:
            messages: Conversation history. Mutated in-place during the loop.
            model: Model ID override. Falls back to ``config.default_model``.
            max_iterations: Iteration limit override. Falls back to
                ``config.max_iterations``.

        Returns:
            The final text response from the model.

        Raises:
            ToolLoopError: If the loop exceeds *max_iterations* without
                reaching ``end_turn``.
        """
        resolved_model = model or self._config.default_model
        resolved_max = (
            max_iterations
            if max_iterations is not None
            else self._config.max_iterations
        )
        tools = self._registry.get_tools_list()

        for iteration in range(resolved_max):
            logger.debug(
                "ToolLoop iteration %d/%d (model=%s)",
                iteration + 1,
                resolved_max,
                resolved_model,
            )

            response = self._client.messages.create(
                model=resolved_model,
                max_tokens=4096,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        logger.debug("ToolLoop finished at iteration %d", iteration + 1)
                        return block.text
                return ""

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._dispatcher.dispatch(
                            {"id": block.id, "name": block.name, "input": block.input}
                        )
                        tool_results.append(result)
                        logger.debug(
                            "Dispatched tool '%s' -> is_error=%s",
                            block.name,
                            result.get("is_error", False),
                        )
                messages.append({"role": "user", "content": tool_results})
                continue

            logger.warning("Unexpected stop_reason: '%s'", response.stop_reason)
            break

        raise ToolLoopError(
            f"ToolLoop exceeded max_iterations={resolved_max} without end_turn"
        )
```

---

## `src/tool_framework/__init__.py`

```python
"""Tool framework public API."""

from .config import FrameworkConfig
from .dispatcher import ToolDispatcher
from .exceptions import (
    ToolExecutionError,
    ToolFrameworkError,
    ToolLoopError,
    ToolNotFoundError,
    ToolValidationError,
)
from .loop import ToolLoop
from .models import ToolCall, ToolError, ToolResult
from .registry import ToolRegistry
from .retry import RetryConfig, call_with_retry

__all__ = [
    "ToolRegistry",
    "ToolDispatcher",
    "ToolLoop",
    "FrameworkConfig",
    "RetryConfig",
    "call_with_retry",
    "ToolCall",
    "ToolResult",
    "ToolError",
    "ToolFrameworkError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolLoopError",
]
```

---

## `demo/agent.py`

```python
"""Demo agent: calculator, string_reverse, word_count tools."""

from __future__ import annotations

import ast
import logging
import sys

from src.tool_framework import FrameworkConfig, ToolDispatcher, ToolLoop, ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build registry
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.tool
def calculator(expression: str) -> float:
    """Safely evaluate a mathematical expression and return the numeric result.

    Supports: +, -, *, /, //, %, ** and numeric literals.
    Does not support function calls, attribute access, or imports.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    _ALLOWED_NODES = frozenset(
        {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
        }
    )

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise ValueError(
                f"Expression contains disallowed construct: {type(node).__name__}"
            )

    result = eval(compile(tree, "<expression>", "eval"))  # noqa: S307
    return float(result)


@registry.tool
def string_reverse(text: str) -> str:
    """Return the input string with all characters in reversed order."""
    return text[::-1]


@registry.tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in the given text."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(prompt: str | None = None) -> str:
    """Run the demo agent with *prompt* (or a default if not provided)."""
    config = FrameworkConfig()
    dispatcher = ToolDispatcher(registry)
    loop = ToolLoop(registry, dispatcher, config)

    user_prompt = prompt or (
        "Please do three things: "
        "(1) calculate 144 / (2 ** 3 + 4), "
        "(2) reverse the string 'Tool calling is powerful', "
        "(3) count how many words are in the sentence "
        "'the quick brown fox jumps over the lazy dog'."
    )

    logger.info("Starting demo agent")
    messages = [{"role": "user", "content": user_prompt}]
    answer = loop.run(messages)
    logger.info("Agent final answer:\n%s", answer)
    return answer


if __name__ == "__main__":
    prompt_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    print(main(prompt_arg))
```

---

## `tests/unit/test_registry.py`

```python
"""Unit tests for ToolRegistry."""

from __future__ import annotations

import pytest

from src.tool_framework.registry import ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def add(x: int, y: int) -> int:
        """Add two integers."""
        return x + y

    @reg.tool
    def greet(name: str, formal: bool = False) -> str:
        """Return a greeting. Pass formal=True for a formal greeting."""
        prefix = "Good day" if formal else "Hello"
        return f"{prefix}, {name}!"

    return reg


class TestToolRegistry:
    def test_register_stores_callable(self, registry: ToolRegistry) -> None:
        fn = registry.get_tool("add")
        assert fn is not None
        assert fn(3, 4) == 7

    def test_len(self, registry: ToolRegistry) -> None:
        assert len(registry) == 2

    def test_contains(self, registry: ToolRegistry) -> None:
        assert "add" in registry
        assert "missing" not in registry

    def test_get_tool_unknown_returns_none(self, registry: ToolRegistry) -> None:
        assert registry.get_tool("nonexistent") is None

    def test_get_tools_list_structure(self, registry: ToolRegistry) -> None:
        tools = registry.get_tools_list()
        assert len(tools) == 2
        for t in tools:
            assert {"name", "description", "input_schema"} <= set(t)
            assert t["input_schema"]["type"] == "object"

    def test_schema_required_fields(self, registry: ToolRegistry) -> None:
        tools = {t["name"]: t for t in registry.get_tools_list()}
        assert tools["add"]["input_schema"]["required"] == ["x", "y"]

    def test_schema_optional_fields_not_required(self, registry: ToolRegistry) -> None:
        tools = {t["name"]: t for t in registry.get_tools_list()}
        required = tools["greet"]["input_schema"].get("required", [])
        assert "formal" not in required
        assert "name" in required

    def test_schema_integer_type(self, registry: ToolRegistry) -> None:
        tools = {t["name"]: t for t in registry.get_tools_list()}
        props = tools["add"]["input_schema"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "integer"

    def test_description_from_docstring(self, registry: ToolRegistry) -> None:
        tools = {t["name"]: t for t in registry.get_tools_list()}
        assert tools["add"]["description"] == "Add two integers."

    def test_register_no_docstring(self) -> None:
        reg = ToolRegistry()

        @reg.tool
        def no_doc(x: str) -> str:
            return x

        tools = {t["name"]: t for t in reg.get_tools_list()}
        assert tools["no_doc"]["description"] == ""

    def test_float_type_in_schema(self) -> None:
        reg = ToolRegistry()

        @reg.tool
        def divide(a: float, b: float) -> float:
            """Divide a by b."""
            return a / b

        tools = {t["name"]: t for t in reg.get_tools_list()}
        props = tools["divide"]["input_schema"]["properties"]
        assert props["a"]["type"] == "number"
        assert props["b"]["type"] == "number"
```

---

## `tests/unit/test_dispatcher.py`

```python
"""Unit tests for ToolDispatcher."""

from __future__ import annotations

import pytest

from src.tool_framework.dispatcher import ToolDispatcher
from src.tool_framework.registry import ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def multiply(x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y

    @reg.tool
    def upper(text: str) -> str:
        """Uppercase a string."""
        return text.upper()

    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(registry)


class TestToolDispatcher:
    def test_successful_dispatch_float(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "t1", "name": "multiply", "input": {"x": 3.0, "y": 4.0}}
        )
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "t1"
        assert result["content"] == "12.0"
        assert "is_error" not in result

    def test_successful_dispatch_string(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "t2", "name": "upper", "input": {"text": "hello"}}
        )
        assert result["content"] == "HELLO"

    def test_validation_error_wrong_type(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "t3", "name": "multiply", "input": {"x": "abc", "y": 2.0}}
        )
        assert result["is_error"] is True
        assert result["tool_use_id"] == "t3"

    def test_validation_error_missing_field(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "t4", "name": "multiply", "input": {"x": 2.0}}
        )
        assert result["is_error"] is True

    def test_runtime_exception_returns_is_error(self, registry: ToolRegistry) -> None:
        @registry.tool
        def divide(a: float, b: float) -> float:
            """Divide a by b."""
            return a / b

        d = ToolDispatcher(registry)
        result = d.dispatch(
            {"id": "t5", "name": "divide", "input": {"a": 1.0, "b": 0.0}}
        )
        assert result["is_error"] is True
        assert "division by zero" in result["content"].lower()

    def test_unknown_tool_returns_is_error(self, dispatcher: ToolDispatcher) -> None:
        result = dispatcher.dispatch(
            {"id": "t6", "name": "ghost_tool", "input": {}}
        )
        assert result["is_error"] is True
        assert "ghost_tool" in result["content"]

    def test_model_cached(self, dispatcher: ToolDispatcher) -> None:
        """Validation model is built once and cached."""
        dispatcher.dispatch({"id": "t7", "name": "multiply", "input": {"x": 1.0, "y": 2.0}})
        dispatcher.dispatch({"id": "t8", "name": "multiply", "input": {"x": 3.0, "y": 4.0}})
        assert "multiply" in dispatcher._model_cache
```

---

## `tests/unit/test_loop.py`

```python
"""Unit tests for ToolLoop — all LLM calls mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tool_framework.config import FrameworkConfig
from src.tool_framework.dispatcher import ToolDispatcher
from src.tool_framework.exceptions import ToolLoopError
from src.tool_framework.loop import ToolLoop
from src.tool_framework.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(stop_reason: str, content: list) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    return r


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(tid: str, name: str, inp: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = tid
    b.name = name
    b.input = inp
    return b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def echo(msg: str) -> str:
        """Echo the message back."""
        return msg

    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(registry)


@pytest.fixture()
def config() -> FrameworkConfig:
    return FrameworkConfig(anthropic_api_key="test-key-unit")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolLoop:
    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_end_turn_immediately(
        self, mock_cls: MagicMock, registry, dispatcher, config
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_response(
            "end_turn", [_text_block("Final answer")]
        )
        loop = ToolLoop(registry, dispatcher, config)
        result = loop.run([{"role": "user", "content": "hi"}])
        assert result == "Final answer"
        assert mock_client.messages.create.call_count == 1

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_single_tool_call_cycle(
        self, mock_cls: MagicMock, registry, dispatcher, config
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        tb = _tool_use_block("t1", "echo", {"msg": "ping"})
        mock_client.messages.create.side_effect = [
            _make_response("tool_use", [tb]),
            _make_response("end_turn", [_text_block("pong")]),
        ]
        loop = ToolLoop(registry, dispatcher, config)
        result = loop.run([{"role": "user", "content": "echo ping"}])
        assert result == "pong"
        assert mock_client.messages.create.call_count == 2

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_max_iterations_raises(
        self, mock_cls: MagicMock, registry, dispatcher, config
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        tb = _tool_use_block("t1", "echo", {"msg": "x"})
        mock_client.messages.create.return_value = _make_response("tool_use", [tb])
        loop = ToolLoop(registry, dispatcher, config)
        with pytest.raises(ToolLoopError):
            loop.run([{"role": "user", "content": "loop"}], max_iterations=3)
        assert mock_client.messages.create.call_count == 3

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_multiple_tool_blocks_in_one_response(
        self, mock_cls: MagicMock, registry, dispatcher, config
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        t1 = _tool_use_block("t1", "echo", {"msg": "a"})
        t2 = _tool_use_block("t2", "echo", {"msg": "b"})
        mock_client.messages.create.side_effect = [
            _make_response("tool_use", [t1, t2]),
            _make_response("end_turn", [_text_block("done")]),
        ]
        loop = ToolLoop(registry, dispatcher, config)
        messages = [{"role": "user", "content": "two tools"}]
        loop.run(messages)
        # user message with tool results should have 2 items
        assert len(messages[-1]["content"]) == 2

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_model_override(
        self, mock_cls: MagicMock, registry, dispatcher, config
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_response(
            "end_turn", [_text_block("ok")]
        )
        loop = ToolLoop(registry, dispatcher, config)
        loop.run([{"role": "user", "content": "hi"}], model="claude-haiku-5-5")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-5-5"

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_empty_content_end_turn_returns_empty_string(
        self, mock_cls: MagicMock, registry, dispatcher, config
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        block = MagicMock(spec=[])  # no 'text' attribute
        mock_client.messages.create.return_value = _make_response("end_turn", [block])
        loop = ToolLoop(registry, dispatcher, config)
        result = loop.run([{"role": "user", "content": "hi"}])
        assert result == ""
```

---

## `tests/unit/test_retry.py`

```python
"""Unit tests for retry utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.tool_framework.retry import RetryConfig, call_with_retry


class TestRetryConfig:
    def test_defaults(self) -> None:
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.backoff_factor == 0.5

    def test_custom_values(self) -> None:
        config = RetryConfig(max_attempts=5, backoff_factor=1.0)
        assert config.max_attempts == 5
        assert config.backoff_factor == 1.0


class TestCallWithRetry:
    @patch("src.tool_framework.retry.time.sleep")
    def test_succeeds_on_first_attempt(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(return_value="ok")
        result = call_with_retry(fn, config=RetryConfig(max_attempts=3))
        assert result == "ok"
        fn.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("src.tool_framework.retry.time.sleep")
    def test_retries_and_eventually_succeeds(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[ValueError("e1"), ValueError("e2"), "success"])
        result = call_with_retry(fn, config=RetryConfig(max_attempts=3, backoff_factor=1.0))
        assert result == "success"
        assert fn.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("src.tool_framework.retry.time.sleep")
    def test_raises_after_all_attempts_exhausted(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=RuntimeError("always"))
        with pytest.raises(RuntimeError, match="always"):
            call_with_retry(fn, config=RetryConfig(max_attempts=2))
        assert fn.call_count == 2

    @patch("src.tool_framework.retry.time.sleep")
    def test_backoff_formula_correct(self, mock_sleep: MagicMock) -> None:
        """Sleep times: factor*1, factor*2 for attempts 0,1."""
        fn = MagicMock(side_effect=[Exception(), Exception(), "ok"])
        call_with_retry(fn, config=RetryConfig(max_attempts=3, backoff_factor=0.25))
        mock_sleep.assert_has_calls([call(0.25), call(0.5)])

    @patch("src.tool_framework.retry.time.sleep")
    def test_passes_args_and_kwargs(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(return_value=42)
        result = call_with_retry(fn, 1, 2, config=RetryConfig(), key="val")
        fn.assert_called_once_with(1, 2, key="val")
        assert result == 42

    @patch("src.tool_framework.retry.time.sleep")
    def test_single_attempt_no_sleep(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=ValueError("fail"))
        with pytest.raises(ValueError):
            call_with_retry(fn, config=RetryConfig(max_attempts=1))
        mock_sleep.assert_not_called()
```

---

## `tests/integration/test_framework.py`

```python
"""Integration tests: registry + dispatcher + loop working together."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tool_framework import (
    FrameworkConfig,
    ToolDispatcher,
    ToolLoop,
    ToolRegistry,
)
from src.tool_framework.exceptions import ToolLoopError
from src.tool_framework.retry import RetryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def square(n: int) -> int:
        """Return n squared."""
        return n * n

    @reg.tool
    def concat(a: str, b: str) -> str:
        """Concatenate two strings."""
        return a + b

    return reg


def _make_response(stop_reason: str, content: list) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    return r


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(tid: str, name: str, inp: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = tid
    b.name = name
    b.input = inp
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_tool_call_and_final_answer(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        tb = _tool_use_block("t1", "square", {"n": 7})
        mock_client.messages.create.side_effect = [
            _make_response("tool_use", [tb]),
            _make_response("end_turn", [_text_block("7 squared is 49")]),
        ]

        registry = _build_registry()
        dispatcher = ToolDispatcher(registry)
        config = FrameworkConfig(anthropic_api_key="integration-test-key")
        loop = ToolLoop(registry, dispatcher, config)

        result = loop.run([{"role": "user", "content": "What is 7 squared?"}])
        assert result == "7 squared is 49"

        # Verify the tool result was passed back to the model
        second_call_messages = mock_client.messages.create.call_args_list[1].kwargs[
            "messages"
        ]
        # Last message is the user message with tool results
        tool_result_msg = second_call_messages[-1]
        assert tool_result_msg["role"] == "user"
        tool_result_content = tool_result_msg["content"][0]
        assert tool_result_content["content"] == "49"
        assert "is_error" not in tool_result_content

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_tool_validation_error_propagates_to_model(self, mock_cls: MagicMock) -> None:
        """When dispatch returns is_error=True, the loop continues (Claude self-corrects)."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # First call: tool with wrong type
        tb_bad = _tool_use_block("t1", "square", {"n": "not_an_int"})
        # Second call: corrected tool call
        tb_good = _tool_use_block("t2", "square", {"n": 5})
        mock_client.messages.create.side_effect = [
            _make_response("tool_use", [tb_bad]),
            _make_response("tool_use", [tb_good]),
            _make_response("end_turn", [_text_block("5 squared is 25")]),
        ]

        registry = _build_registry()
        dispatcher = ToolDispatcher(registry)
        config = FrameworkConfig(anthropic_api_key="test")
        loop = ToolLoop(registry, dispatcher, config)
        result = loop.run([{"role": "user", "content": "square 5"}])
        assert result == "5 squared is 25"

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_retry_dispatcher_in_pipeline(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        registry = ToolRegistry()
        call_count = 0

        @registry.tool
        def flaky_square(n: int) -> int:
            """Square n, but fails once first."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient network error")
            return n * n

        tb = _tool_use_block("t1", "flaky_square", {"n": 6})
        mock_client.messages.create.side_effect = [
            _make_response("tool_use", [tb]),
            _make_response("end_turn", [_text_block("36")]),
        ]

        from unittest.mock import patch as _patch

        with _patch("src.tool_framework.retry.time.sleep"):
            dispatcher = ToolDispatcher(
                registry, retry_config=RetryConfig(max_attempts=3, backoff_factor=0.01)
            )
            config = FrameworkConfig(anthropic_api_key="test")
            loop = ToolLoop(registry, dispatcher, config)
            result = loop.run([{"role": "user", "content": "square 6"}])

        assert result == "36"
        assert call_count == 2

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_max_iterations_integration(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        tb = _tool_use_block("t1", "square", {"n": 1})
        mock_client.messages.create.return_value = _make_response("tool_use", [tb])

        registry = _build_registry()
        dispatcher = ToolDispatcher(registry)
        config = FrameworkConfig(anthropic_api_key="test", max_iterations=3)
        loop = ToolLoop(registry, dispatcher, config)

        with pytest.raises(ToolLoopError):
            loop.run([{"role": "user", "content": "loop"}])
        assert mock_client.messages.create.call_count == 3
```

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tool-framework"
version = "0.1.0"
description = "Typed tool-calling framework for Claude (Phase 5 Week 2 reference)"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.40.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "ruff",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "S"]
ignore = ["S307"]  # eval inside AST-whitelisted calculator
```

---

## Running the Tests

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=src/tool_framework --cov-report=term-missing

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

---

## Architecture Notes

- **`models.py`** keeps `ToolResult.to_dict()` as the single serialization point — `ToolDispatcher` never constructs raw dicts.
- **`ToolDispatcher` caches validation models** in `_model_cache` to avoid rebuilding Pydantic models on every dispatch call.
- **`ToolLoop` mutates `messages` in-place** so callers can inspect the full conversation trace after `run()` returns.
- **`FrameworkConfig`** uses env prefix `TOOL_FRAMEWORK_` to avoid collisions. `ToolLoop` falls back to bare `ANTHROPIC_API_KEY` as a secondary source for convenience.
- **`RetryConfig`** is a Pydantic `BaseModel` so it can be embedded in larger config hierarchies or passed via JSON.
- **`demo/agent.py`** uses an AST whitelist for the `calculator` tool rather than `eval` on raw input, keeping it safe while still supporting full arithmetic expressions.
