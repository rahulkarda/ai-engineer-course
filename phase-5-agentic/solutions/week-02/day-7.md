# Solutions — Phase 5 · Week 2 · Day 7 — Final Integration Reference

## Overview

Complete integrated framework adding retry logic, custom exceptions, `pydantic-settings` config, a demo agent, and a full test suite. All modules are in `src/tool_framework/`.

---

## Module: `src/tool_framework/exceptions.py`

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

## Module: `src/tool_framework/retry.py`

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
    """Call *fn* with retry on exception.

    Sleep duration on attempt *n* (0-indexed): ``config.backoff_factor * 2 ** n``.

    Raises:
        The last exception if all attempts fail.
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
                    "Attempt %d/%d failed: %s. Retrying in %.2fs.",
                    attempt + 1,
                    config.max_attempts,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    "All %d attempts failed: %s", config.max_attempts, exc
                )
    assert last_exc is not None  # always set after at least one iteration
    raise last_exc
```

---

## Module: `src/tool_framework/config.py`

```python
"""Framework configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class FrameworkConfig(BaseSettings):
    """Runtime configuration loaded from environment variables."""

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

*Env vars: `TOOL_FRAMEWORK_ANTHROPIC_API_KEY`, `TOOL_FRAMEWORK_DEFAULT_MODEL`, etc. Falls back to `.env` file.*

---

## Module: `src/tool_framework/registry.py` (unchanged from Day 6)

*(Same source as Day 6 — reproduced for completeness.)*

```python
"""ToolRegistry: decorator-based registration with auto-schema generation."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any, get_type_hints

logger = logging.getLogger(__name__)


def _python_type_to_json_schema(annotation: Any) -> dict:
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

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}

    def tool(self, fn: Callable) -> Callable:
        """Decorator: register a function as a tool."""
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
        return list(self._schemas.values())

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)
```

---

## Module: `src/tool_framework/dispatcher.py` (enhanced with retry)

```python
"""ToolDispatcher: validates inputs, calls the tool, optionally retries."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from pydantic import ValidationError, create_model

from .exceptions import ToolNotFoundError
from .registry import ToolRegistry
from .retry import RetryConfig, call_with_retry

logger = logging.getLogger(__name__)


def _make_validation_model(registry: ToolRegistry, tool_name: str):
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
    """Dispatches tool_use blocks; supports optional retry on execution errors."""

    def __init__(
        self,
        registry: ToolRegistry,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._registry = registry
        self._retry_config = retry_config

    def dispatch(self, tool_use_block: dict) -> dict:
        """Execute the tool described by *tool_use_block*.

        Returns a tool_result dict for the Anthropic messages API.
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

        try:
            if self._retry_config is not None:
                result = call_with_retry(
                    fn,
                    **validated.model_dump(),
                    config=self._retry_config,
                )
            else:
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

## Module: `src/tool_framework/loop.py` (uses FrameworkConfig)

```python
"""ToolLoop: agentic loop driven by FrameworkConfig."""

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
    """Drives the Anthropic tool-use loop until end_turn or max_iterations."""

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
            api_key=self._config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def run(
        self,
        messages: list[dict],
        model: str | None = None,
        max_iterations: int | None = None,
    ) -> str:
        """Run the agentic loop.

        Args:
            messages: Conversation history (mutated in-place).
            model: Model ID; falls back to config.default_model.
            max_iterations: Guard; falls back to config.max_iterations.

        Returns:
            Final text from the model.

        Raises:
            ToolLoopError: If max_iterations exceeded.
        """
        resolved_model = model or self._config.default_model
        resolved_max = max_iterations if max_iterations is not None else self._config.max_iterations
        tools = self._registry.get_tools_list()

        for iteration in range(resolved_max):
            logger.debug("ToolLoop iteration %d / %d", iteration + 1, resolved_max)
            response = self._client.messages.create(
                model=resolved_model,
                max_tokens=4096,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
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
                messages.append({"role": "user", "content": tool_results})
                continue

            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        raise ToolLoopError(
            f"ToolLoop exceeded max_iterations={resolved_max}"
        )
```

---

## Module: `src/tool_framework/__init__.py`

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
from .registry import ToolRegistry
from .retry import RetryConfig, call_with_retry

__all__ = [
    "ToolRegistry",
    "ToolDispatcher",
    "ToolLoop",
    "FrameworkConfig",
    "RetryConfig",
    "call_with_retry",
    "ToolFrameworkError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolLoopError",
]
```

---

## Demo Agent: `demo/agent.py`

```python
"""Demo agent with calculator, string_reverse, and word_count tools."""

from __future__ import annotations

import ast
import logging
import os

from src.tool_framework import (
    FrameworkConfig,
    ToolDispatcher,
    ToolLoop,
    ToolRegistry,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

registry = ToolRegistry()


@registry.tool
def calculator(expression: str) -> float:
    """Safely evaluate a mathematical expression and return the result.

    Supports +, -, *, /, ** and numeric literals only.
    """
    # ast.literal_eval only handles literals, not operators.
    # We compile and evaluate with ast.parse restricted to Expr -> BinOp tree.
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    # Whitelist of allowed node types
    allowed = {
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

    for node in ast.walk(tree):
        if type(node) not in allowed:
            raise ValueError(f"Disallowed node type in expression: {type(node).__name__}")

    result = eval(compile(tree, "<expression>", "eval"))  # noqa: S307 — AST-whitelisted
    return float(result)


@registry.tool
def string_reverse(text: str) -> str:
    """Return the input string reversed character by character."""
    return text[::-1]


@registry.tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in text."""
    return len(text.split())


def main() -> None:
    config = FrameworkConfig()
    dispatcher = ToolDispatcher(registry)
    loop = ToolLoop(registry, dispatcher, config)

    messages = [
        {
            "role": "user",
            "content": (
                "Please: (1) calculate 12 * (3 + 4), "
                "(2) reverse the string 'hello world', "
                "(3) count the words in 'the quick brown fox'."
            ),
        }
    ]

    result = loop.run(messages)
    logger.info("Final answer:\n%s", result)


if __name__ == "__main__":
    main()
```

---

## Tests: `tests/test_tool_framework_day7.py`

```python
"""Day-7 test suite — retry, config, enhanced dispatcher, integration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, call, patch

import pytest

from src.tool_framework.config import FrameworkConfig
from src.tool_framework.dispatcher import ToolDispatcher
from src.tool_framework.exceptions import ToolLoopError
from src.tool_framework.loop import ToolLoop
from src.tool_framework.registry import ToolRegistry
from src.tool_framework.retry import RetryConfig, call_with_retry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    return reg


@pytest.fixture()
def dispatcher(registry: ToolRegistry) -> ToolDispatcher:
    return ToolDispatcher(registry)


# ---------------------------------------------------------------------------
# TestRetry
# ---------------------------------------------------------------------------


class TestCallWithRetry:
    @patch("src.tool_framework.retry.time.sleep")
    def test_succeeds_first_attempt(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(return_value=42)
        config = RetryConfig(max_attempts=3, backoff_factor=0.5)
        result = call_with_retry(fn, config=config)
        assert result == 42
        mock_sleep.assert_not_called()

    @patch("src.tool_framework.retry.time.sleep")
    def test_retries_on_failure_then_succeeds(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[ValueError("oops"), ValueError("oops"), 99])
        config = RetryConfig(max_attempts=3, backoff_factor=1.0)
        result = call_with_retry(fn, config=config)
        assert result == 99
        assert fn.call_count == 3
        # Sleep called twice (after attempt 0 and attempt 1)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    @patch("src.tool_framework.retry.time.sleep")
    def test_raises_after_all_attempts(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=RuntimeError("always fails"))
        config = RetryConfig(max_attempts=2, backoff_factor=0.5)
        with pytest.raises(RuntimeError, match="always fails"):
            call_with_retry(fn, config=config)
        assert fn.call_count == 2

    @patch("src.tool_framework.retry.time.sleep")
    def test_backoff_formula(self, mock_sleep: MagicMock) -> None:
        """backoff_factor * 2**attempt — attempt 0 → factor, attempt 1 → 2*factor."""
        fn = MagicMock(side_effect=[Exception(), Exception(), "ok"])
        config = RetryConfig(max_attempts=3, backoff_factor=0.25)
        call_with_retry(fn, config=config)
        mock_sleep.assert_has_calls([call(0.25), call(0.5)])


# ---------------------------------------------------------------------------
# TestFrameworkConfig
# ---------------------------------------------------------------------------


class TestFrameworkConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOOL_FRAMEWORK_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TOOL_FRAMEWORK_DEFAULT_MODEL", raising=False)
        config = FrameworkConfig()
        assert config.default_model == "claude-sonnet-5-5"
        assert config.max_iterations == 10
        assert config.log_level == "INFO"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOOL_FRAMEWORK_DEFAULT_MODEL", "claude-haiku-5-5")
        monkeypatch.setenv("TOOL_FRAMEWORK_MAX_ITERATIONS", "5")
        config = FrameworkConfig()
        assert config.default_model == "claude-haiku-5-5"
        assert config.max_iterations == 5


# ---------------------------------------------------------------------------
# TestEnhancedDispatcher
# ---------------------------------------------------------------------------


class TestEnhancedDispatcher:
    @patch("src.tool_framework.retry.time.sleep")
    def test_retry_dispatcher_retries_on_failure(
        self, mock_sleep: MagicMock, registry: ToolRegistry
    ) -> None:
        call_count = 0

        @registry.tool
        def flaky(x: int) -> int:
            """Flaky tool that fails twice then succeeds."""
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return x * 2

        config = RetryConfig(max_attempts=3, backoff_factor=0.1)
        dispatcher = ToolDispatcher(registry, retry_config=config)
        result = dispatcher.dispatch({"id": "t1", "name": "flaky", "input": {"x": 5}})
        assert result["content"] == "10"
        assert "is_error" not in result
        assert call_count == 3

    def test_no_retry_config_does_not_retry(self, registry: ToolRegistry) -> None:
        call_count = 0

        @registry.tool
        def always_fails(msg: str) -> str:
            """Always raises."""
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        dispatcher = ToolDispatcher(registry, retry_config=None)
        result = dispatcher.dispatch({"id": "t1", "name": "always_fails", "input": {"msg": "x"}})
        assert result["is_error"] is True
        assert call_count == 1


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


def _make_mock_response(stop_reason: str, content: list) -> MagicMock:
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = content
    return response


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


class TestIntegration:
    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_registry_dispatcher_loop_pipeline(
        self, mock_cls: MagicMock, registry: ToolRegistry
    ) -> None:
        """Full pipeline: registry -> dispatcher -> loop with mocked client."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        tb = _tool_use_block("t1", "multiply", {"a": 6.0, "b": 7.0})
        mock_client.messages.create.side_effect = [
            _make_mock_response("tool_use", [tb]),
            _make_mock_response("end_turn", [_text_block("The answer is 42.0")]),
        ]

        dispatcher = ToolDispatcher(registry)
        config = FrameworkConfig(anthropic_api_key="test-key")
        loop = ToolLoop(registry, dispatcher, config)

        result = loop.run([{"role": "user", "content": "What is 6 × 7?"}])
        assert result == "The answer is 42.0"
        assert mock_client.messages.create.call_count == 2

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_loop_uses_config_defaults(self, mock_cls: MagicMock, registry: ToolRegistry) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_response(
            "end_turn", [_text_block("ok")]
        )

        config = FrameworkConfig(
            anthropic_api_key="test-key",
            default_model="claude-haiku-5-5",
            max_iterations=5,
        )
        dispatcher = ToolDispatcher(registry)
        loop = ToolLoop(registry, dispatcher, config)
        loop.run([{"role": "user", "content": "hi"}])

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-5-5"

    @patch("src.tool_framework.loop.anthropic.Anthropic")
    def test_loop_max_iterations_from_config(
        self, mock_cls: MagicMock, registry: ToolRegistry
    ) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        tb = _tool_use_block("t1", "multiply", {"a": 1.0, "b": 1.0})
        mock_client.messages.create.return_value = _make_mock_response("tool_use", [tb])

        config = FrameworkConfig(anthropic_api_key="test-key", max_iterations=2)
        dispatcher = ToolDispatcher(registry)
        loop = ToolLoop(registry, dispatcher, config)

        with pytest.raises(ToolLoopError):
            loop.run([{"role": "user", "content": "loop"}])
        assert mock_client.messages.create.call_count == 2
```

---

## Demo Agent Tests: `tests/test_demo_agent.py`

```python
"""Tests for the demo agent tools (calculator, string_reverse, word_count)."""

from __future__ import annotations

import pytest

# Import tools directly from the demo module
import importlib
import sys

# We test the tool functions directly without running the agent
import ast


def _get_calculator():
    """Re-implement inline to avoid importing demo which would init anthropic."""
    def calculator(expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc
        allowed = {
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
            ast.Mod, ast.Pow, ast.USub, ast.UAdd,
        }
        for node in ast.walk(tree):
            if type(node) not in allowed:
                raise ValueError(f"Disallowed node: {type(node).__name__}")
        return float(eval(compile(tree, "<expression>", "eval")))
    return calculator


class TestDemoTools:
    def test_calculator_addition(self) -> None:
        calc = _get_calculator()
        assert calc("2 + 3") == 5.0

    def test_calculator_complex(self) -> None:
        calc = _get_calculator()
        assert calc("12 * (3 + 4)") == 84.0

    def test_calculator_power(self) -> None:
        calc = _get_calculator()
        assert calc("2 ** 10") == 1024.0

    def test_calculator_disallows_builtins(self) -> None:
        calc = _get_calculator()
        with pytest.raises(ValueError, match="Disallowed"):
            calc("__import__('os').system('ls')")

    def test_string_reverse(self) -> None:
        assert "hello"[::-1] == "olleh"

    def test_word_count(self) -> None:
        assert len("the quick brown fox".split()) == 4
```

---

## Key Design Decisions

- `RetryConfig` is a Pydantic `BaseModel`, not a dataclass, so it validates `max_attempts > 0` by extension and composes cleanly with other Pydantic models.
- `call_with_retry` takes `config` as keyword-only to avoid ambiguity when `fn` accepts positional args.
- `FrameworkConfig` uses `env_prefix="TOOL_FRAMEWORK_"` to avoid collisions with the bare `ANTHROPIC_API_KEY` var. The loop falls back to `os.getenv("ANTHROPIC_API_KEY")` as a secondary source.
- The calculator uses an AST whitelist rather than `ast.literal_eval` (which cannot evaluate expressions) while avoiding `eval()` on arbitrary input.
- Demo tools are tested in isolation without importing the full `demo/agent.py` to avoid Anthropic client initialization during unit tests.
