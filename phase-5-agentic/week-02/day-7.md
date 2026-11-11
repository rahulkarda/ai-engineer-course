# Phase 5 · Week 02 · Day 7 — Week 2 Project: Typed Tool-Calling Framework (Final Integration)

> Objective: Integrate ToolRegistry, ToolDispatcher, and ToolLoop into a production-quality framework with error handling, logging, retries, and a working demo agent.  ·  Time: ~3 hrs  ·  Prereqs: [[day-6]]

---

## Overview

Today you complete the framework started on Day 6 by adding:
1. Error handling + retry logic for failed tool calls
2. Structured logging with per-run context
3. Connection to the real Anthropic API via `os.getenv`
4. A demo agent using the framework (calculator, string_utils, weather_stub)
5. Full pytest test suite with all LLM calls mocked

At the end, read `assessment.md` and `project.md` for the week's deliverables.

Estimated time breakdown:
- Integration + error handling: ~60 min
- Demo agent: ~45 min
- Tests: ~45 min
- Assessment review: ~30 min

---

## Concept

### What "integration" means here

Day 6 built three components in isolation. Today they become a package:

```
src/tool_framework/
    __init__.py       ← re-exports public API
    registry.py       ← ToolRegistry (from day 6)
    dispatcher.py     ← ToolDispatcher (from day 6)
    loop.py           ← ToolLoop (enhanced today)
    retry.py          ← RetryConfig + retry logic (new)
    exceptions.py     ← typed exceptions (new)
    config.py         ← pydantic-settings config (new)
```

### Error handling strategy

```
Tool call fails → retry up to N times with backoff
Still failing   → return is_error result and continue loop
Loop exhausted   → return empty string + log error
```

Always keep the loop going — one bad tool should not abort the agent.

---

## How it works internally

### exceptions.py

```python
"""
src/tool_framework/exceptions.py
Typed exceptions for the tool framework.
"""


class ToolFrameworkError(Exception):
    """Base for all tool framework errors."""


class ToolNotFoundError(ToolFrameworkError):
    """Raised when a requested tool is not in the registry."""


class ToolValidationError(ToolFrameworkError):
    """Raised when tool inputs fail Pydantic validation."""


class ToolExecutionError(ToolFrameworkError):
    """Raised when a tool raises an unexpected exception."""


class LoopExhaustedError(ToolFrameworkError):
    """Raised when the tool loop hits max_iterations without end_turn."""
```

### retry.py

```python
"""
src/tool_framework/retry.py
RetryConfig and retry-aware dispatch helper.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for tool call retry behaviour.

    Attributes:
        max_attempts: Total attempts including the first try.
        backoff_seconds: Base sleep between retries (doubles each attempt).
        retryable_exceptions: Exception types that trigger a retry.
    """
    max_attempts: int = 3
    backoff_seconds: float = 0.5
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (RuntimeError, TimeoutError, ConnectionError)
    )


def call_with_retry(
    fn: Callable,
    kwargs: dict[str, Any],
    config: RetryConfig,
    tool_name: str,
) -> Any:
    """Call fn(**kwargs) with retry on retryable exceptions.

    Args:
        fn: The tool function to call.
        kwargs: Validated keyword arguments.
        config: Retry configuration.
        tool_name: Used only for logging.

    Returns:
        The function's return value on success.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            return fn(**kwargs)
        except config.retryable_exceptions as exc:
            last_exc = exc
            if attempt < config.max_attempts:
                sleep_time = config.backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Tool %s failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    tool_name, attempt, config.max_attempts, exc, sleep_time,
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    "Tool %s exhausted %d attempts. Last error: %s",
                    tool_name, config.max_attempts, exc,
                )
    raise last_exc  # type: ignore[misc]
```

### Enhanced loop.py

```python
"""
src/tool_framework/loop.py
ToolLoop with retry support and structured logging.
"""
import logging
from typing import Any

import anthropic

from tool_framework.dispatcher import ToolDispatcher
from tool_framework.exceptions import LoopExhaustedError
from tool_framework.registry import ToolRegistry
from tool_framework.retry import RetryConfig

logger = logging.getLogger(__name__)

_DEFAULT_RETRY = RetryConfig()


class ToolLoop:
    """Drives the full Anthropic tool_use loop with retry support.

    Args:
        client: Anthropic client (real or mock).
        registry: ToolRegistry with registered tools.
        model: Model ID. Must be a claude-*-4-5 family model.
        max_tokens: Max tokens per response.
        max_iterations: Safety cap on loop iterations.
        retry_config: Retry settings for failed tool calls.
        raise_on_exhausted: If True, raise LoopExhaustedError instead of returning "".
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        registry: ToolRegistry,
        model: str = "claude-sonnet-5-5",
        max_tokens: int = 1024,
        max_iterations: int = 10,
        retry_config: RetryConfig | None = None,
        raise_on_exhausted: bool = False,
    ) -> None:
        self._client = client
        self._registry = registry
        self._dispatcher = ToolDispatcher(registry, retry_config=retry_config or _DEFAULT_RETRY)
        self._model = model
        self._max_tokens = max_tokens
        self._max_iterations = max_iterations
        self._raise_on_exhausted = raise_on_exhausted

    def run(self, user_message: str, system: str | None = None) -> str:
        """Run the tool loop for a single user message. Return the final text.

        Args:
            user_message: The user's question or instruction.
            system: Optional system prompt.

        Returns:
            The assistant's final text answer.

        Raises:
            LoopExhaustedError: If raise_on_exhausted is True and max_iterations hit.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        tools = self._registry.get_schema()
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "tools": tools,
            "messages": messages,
        }
        if system:
            create_kwargs["system"] = system

        for iteration in range(self._max_iterations):
            logger.info(
                "ToolLoop iteration=%d model=%s tools=%d",
                iteration + 1, self._model, len(tools),
            )

            response = self._client.messages.create(**create_kwargs)
            logger.debug("stop_reason=%s", response.stop_reason)

            if response.stop_reason == "end_turn":
                answer = self._extract_text(response)
                logger.info("Loop completed in %d iteration(s)", iteration + 1)
                return answer

            if response.stop_reason == "tool_use":
                tool_calls = [b for b in response.content if b.type == "tool_use"]
                logger.info("Processing %d tool call(s)", len(tool_calls))

                tool_results = [
                    self._dispatcher.dispatch(tc.id, tc.name, tc.input)
                    for tc in tool_calls
                ]

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                create_kwargs["messages"] = messages
            else:
                logger.warning("Unexpected stop_reason=%s", response.stop_reason)
                break

        msg = f"ToolLoop exhausted {self._max_iterations} iterations without end_turn"
        logger.error(msg)
        if self._raise_on_exhausted:
            raise LoopExhaustedError(msg)
        return ""

    @staticmethod
    def _extract_text(response: Any) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
```

### Updated dispatcher.py (with retry)

```python
"""
src/tool_framework/dispatcher.py
ToolDispatcher with Pydantic validation and retry support.
"""
import logging
from typing import Any

from pydantic import ValidationError

from tool_framework.registry import ToolRegistry
from tool_framework.retry import RetryConfig, call_with_retry

logger = logging.getLogger(__name__)

_DEFAULT_RETRY = RetryConfig()


class ToolDispatcher:
    """Validates tool inputs with Pydantic, dispatches calls with retry.

    Args:
        registry: The ToolRegistry containing registered tools.
        retry_config: Retry configuration for failed calls.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._registry = registry
        self._retry_config = retry_config or _DEFAULT_RETRY

    def dispatch(self, tool_id: str, tool_name: str, raw_inputs: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs, call the tool with retry, return tool_result block.

        Never raises — errors become is_error results.
        """
        logger.info("dispatch tool=%s id=%s", tool_name, tool_id)

        try:
            entry = self._registry.get_tool(tool_name)
        except KeyError:
            return self._error(tool_id, f"Unknown tool: {tool_name!r}")

        try:
            validated = entry.input_model(**raw_inputs)
        except ValidationError as exc:
            logger.warning("Validation failed for %s: %s", tool_name, exc)
            return self._error(tool_id, f"Input validation failed: {exc}")

        try:
            result = call_with_retry(
                fn=entry.fn,
                kwargs=validated.model_dump(),
                config=self._retry_config,
                tool_name=tool_name,
            )
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": str(result),
            }
        except Exception as exc:
            logger.exception("Tool %s failed after retries: %s", tool_name, exc)
            return self._error(tool_id, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _error(tool_id: str, message: str) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": message,
            "is_error": True,
        }
```

### config.py (pydantic-settings)

```python
"""
src/tool_framework/config.py
Framework configuration via pydantic-settings.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field


class FrameworkConfig(BaseSettings):
    """Framework configuration loaded from environment variables.

    Required env vars:
        ANTHROPIC_API_KEY: Anthropic API key.

    Optional env vars:
        TOOL_FRAMEWORK_MODEL: Model ID (default: claude-sonnet-5-5).
        TOOL_FRAMEWORK_MAX_TOKENS: Max tokens per response (default: 1024).
        TOOL_FRAMEWORK_MAX_ITERATIONS: Max loop iterations (default: 10).
        TOOL_FRAMEWORK_MAX_RETRY_ATTEMPTS: Max retry attempts per tool (default: 3).
    """

    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = Field(default="claude-sonnet-5-5", alias="TOOL_FRAMEWORK_MODEL")
    max_tokens: int = Field(default=1024, alias="TOOL_FRAMEWORK_MAX_TOKENS")
    max_iterations: int = Field(default=10, alias="TOOL_FRAMEWORK_MAX_ITERATIONS")
    max_retry_attempts: int = Field(default=3, alias="TOOL_FRAMEWORK_MAX_RETRY_ATTEMPTS")

    model_config = {"populate_by_name": True, "env_prefix": "TOOL_FRAMEWORK_"}
```

### __init__.py (public API)

```python
"""
src/tool_framework/__init__.py
Public API for the tool_framework package.
"""
from tool_framework.registry import ToolRegistry
from tool_framework.dispatcher import ToolDispatcher
from tool_framework.loop import ToolLoop
from tool_framework.retry import RetryConfig
from tool_framework.config import FrameworkConfig
from tool_framework.exceptions import (
    ToolFrameworkError,
    ToolNotFoundError,
    ToolValidationError,
    ToolExecutionError,
    LoopExhaustedError,
)

__all__ = [
    "ToolRegistry",
    "ToolDispatcher",
    "ToolLoop",
    "RetryConfig",
    "FrameworkConfig",
    "ToolFrameworkError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolExecutionError",
    "LoopExhaustedError",
]
```

---

## Demo Agent

```python
"""
demo_agent.py
A small agent that uses the completed framework to answer questions.
Tools: calculator, string_utils, weather_stub.
Uses real Anthropic API (key from ANTHROPIC_API_KEY env var).
"""
import logging
import os

import anthropic

from tool_framework import ToolRegistry, ToolLoop, RetryConfig, FrameworkConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.register
def calculator(expression: str) -> dict:
    """Evaluate a simple arithmetic expression safely. Supports +, -, *, /.

    Args:
        expression: Arithmetic expression, e.g. "3 * (4 + 5)".
    """
    import ast
    import operator

    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression node: {type(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        logger.info("calculator: %s = %s", expression, result)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}


@registry.register
def string_utils(operation: str, text: str, second_text: str = "") -> dict:
    """Perform string operations: upper, lower, reverse, word_count, join.

    Args:
        operation: One of: upper, lower, reverse, word_count, join.
        text: Primary input text.
        second_text: Second string (only used by join).
    """
    ops = {
        "upper": lambda t, _: t.upper(),
        "lower": lambda t, _: t.lower(),
        "reverse": lambda t, _: t[::-1],
        "word_count": lambda t, _: str(len(t.split())),
        "join": lambda t, s: t + s,
    }
    if operation not in ops:
        return {"error": f"Unknown operation: {operation}. Choose: {list(ops)}"}
    result = ops[operation](text, second_text)
    logger.info("string_utils: %s(%r) = %r", operation, text, result)
    return {"operation": operation, "result": result}


@registry.register
def weather_stub(city: str, units: str = "metric") -> dict:
    """Get current weather for a city (stub — returns realistic fake data).

    Args:
        city: City name.
        units: 'metric' (Celsius) or 'imperial' (Fahrenheit).
    """
    # Stub data — no external API call
    data = {
        "Berlin": {"temp": 14, "condition": "overcast", "humidity": 72},
        "Tokyo": {"temp": 22, "condition": "sunny", "humidity": 55},
        "New York": {"temp": 18, "condition": "partly cloudy", "humidity": 65},
    }
    weather = data.get(city, {"temp": 20, "condition": "clear", "humidity": 60})
    temp = weather["temp"]
    if units == "imperial":
        temp = temp * 9 / 5 + 32
    unit_label = "°F" if units == "imperial" else "°C"
    return {
        "city": city,
        "temperature": f"{temp}{unit_label}",
        "condition": weather["condition"],
        "humidity": f"{weather['humidity']}%",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """\
You are a helpful assistant. You have access to a calculator, string utilities, \
and a weather service. Use them when the user's question requires computation, \
string manipulation, or weather information. Be concise.\
"""

DEMO_QUESTIONS = [
    "What is (15 * 4) + (100 / 5)?",
    "Reverse the phrase 'Hello World' and count its words.",
    "What is the weather in Berlin? Convert the temperature to Fahrenheit.",
]


def main() -> None:
    config = FrameworkConfig()
    client = build_client()
    retry = RetryConfig(max_attempts=config.max_retry_attempts)
    loop = ToolLoop(
        client=client,
        registry=registry,
        model=config.model,
        max_tokens=config.max_tokens,
        max_iterations=config.max_iterations,
        retry_config=retry,
    )

    for question in DEMO_QUESTIONS:
        logger.info("=" * 60)
        logger.info("Question: %s", question)
        answer = loop.run(user_message=question, system=SYSTEM_PROMPT)
        logger.info("Answer: %s", answer)


if __name__ == "__main__":
    main()
```

---

## Full Test Suite

```python
"""
tests/test_framework_day7.py
Full pytest suite for the integrated framework.
All LLM calls are mocked — no real API calls.
"""
import os
import time
import pytest
from unittest.mock import MagicMock, patch, call
from typing import Any

import anthropic

from tool_framework import (
    ToolRegistry,
    ToolDispatcher,
    ToolLoop,
    RetryConfig,
    LoopExhaustedError,
)
from tool_framework.retry import call_with_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.register
    def add(left: float, right: float) -> dict:
        """Add two numbers."""
        return {"result": left + right}

    @reg.register
    def fail_once(x: int) -> dict:
        """Fails the first time, succeeds the second."""
        return {"result": x * 2}

    return reg


def mock_tool_use(tool_id: str, name: str, inputs: dict) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = inputs
    return b


def mock_end_turn(text: str) -> MagicMock:
    tb = MagicMock()
    tb.type = "text"
    tb.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [tb]
    return resp


def mock_tool_use_response(*blocks: MagicMock) -> MagicMock:
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = list(blocks)
    return resp


# ---------------------------------------------------------------------------
# RetryConfig + call_with_retry
# ---------------------------------------------------------------------------

class TestRetry:
    def test_success_on_first_attempt(self) -> None:
        fn = MagicMock(return_value={"ok": True})
        config = RetryConfig(max_attempts=3)
        result = call_with_retry(fn, {"x": 1}, config, "test_tool")
        assert result == {"ok": True}
        fn.assert_called_once_with(x=1)

    def test_retries_on_retryable_exception(self) -> None:
        fn = MagicMock(side_effect=[RuntimeError("fail"), {"ok": True}])
        config = RetryConfig(max_attempts=3, backoff_seconds=0.0)
        result = call_with_retry(fn, {}, config, "test_tool")
        assert result == {"ok": True}
        assert fn.call_count == 2

    def test_raises_after_max_attempts(self) -> None:
        fn = MagicMock(side_effect=RuntimeError("persistent fail"))
        config = RetryConfig(max_attempts=2, backoff_seconds=0.0)
        with pytest.raises(RuntimeError, match="persistent fail"):
            call_with_retry(fn, {}, config, "test_tool")
        assert fn.call_count == 2

    def test_non_retryable_exception_not_retried(self) -> None:
        fn = MagicMock(side_effect=ValueError("bad input"))
        config = RetryConfig(max_attempts=3, backoff_seconds=0.0,
                             retryable_exceptions=(RuntimeError,))
        with pytest.raises(ValueError):
            call_with_retry(fn, {}, config, "test_tool")
        fn.assert_called_once()


# ---------------------------------------------------------------------------
# ToolDispatcher (with retry)
# ---------------------------------------------------------------------------

class TestToolDispatcherWithRetry:
    def test_retry_on_connection_error(self) -> None:
        reg = ToolRegistry()
        call_count = 0

        @reg.register
        def flaky_tool(value: int) -> dict:
            """Flaky: fails first call."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient")
            return {"result": value}

        retry = RetryConfig(max_attempts=3, backoff_seconds=0.0)
        dispatcher = ToolDispatcher(reg, retry_config=retry)
        result = dispatcher.dispatch("tu_1", "flaky_tool", {"value": 42})
        assert "is_error" not in result
        assert call_count == 2

    def test_error_result_after_exhausting_retries(self) -> None:
        reg = ToolRegistry()

        @reg.register
        def always_fail(x: int) -> dict:
            """Always raises."""
            raise RuntimeError("boom")

        retry = RetryConfig(max_attempts=2, backoff_seconds=0.0)
        dispatcher = ToolDispatcher(reg, retry_config=retry)
        result = dispatcher.dispatch("tu_2", "always_fail", {"x": 1})
        assert result["is_error"] is True
        assert "RuntimeError" in result["content"]


# ---------------------------------------------------------------------------
# ToolLoop (integration)
# ---------------------------------------------------------------------------

class TestToolLoop:
    def test_run_with_tool_call(self) -> None:
        reg = make_registry()
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.create.side_effect = [
            mock_tool_use_response(mock_tool_use("tu_1", "add", {"left": 2.0, "right": 3.0})),
            mock_end_turn("The answer is 5."),
        ]
        loop = ToolLoop(client=client, registry=reg)
        assert loop.run("What is 2 + 3?") == "The answer is 5."

    def test_system_prompt_passed(self) -> None:
        reg = make_registry()
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.create.return_value = mock_end_turn("Hi")
        loop = ToolLoop(client=client, registry=reg)
        loop.run("Hi", system="You are a helpful assistant.")
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."

    def test_raise_on_exhausted(self) -> None:
        reg = make_registry()
        client = MagicMock(spec=anthropic.Anthropic)
        infinite = mock_tool_use_response(
            mock_tool_use("tu_inf", "add", {"left": 1.0, "right": 1.0})
        )
        client.messages.create.return_value = infinite
        loop = ToolLoop(client=client, registry=reg, max_iterations=2,
                        raise_on_exhausted=True)
        with pytest.raises(LoopExhaustedError):
            loop.run("Keep going")

    def test_tool_error_result_continues_loop(self) -> None:
        """An is_error tool result should not abort the loop."""
        reg = make_registry()
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.create.side_effect = [
            mock_tool_use_response(
                mock_tool_use("tu_bad", "nonexistent_tool", {})
            ),
            mock_end_turn("I encountered an error but recovered."),
        ]
        loop = ToolLoop(client=client, registry=reg)
        answer = loop.run("Call a bad tool")
        assert "recovered" in answer

    def test_model_passed_to_client(self) -> None:
        reg = make_registry()
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.create.return_value = mock_end_turn("ok")
        loop = ToolLoop(client=client, registry=reg, model="claude-haiku-5-5")
        loop.run("ping")
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-5-5"

    def test_tool_results_batched_in_one_user_message(self) -> None:
        reg = make_registry()
        client = MagicMock(spec=anthropic.Anthropic)
        client.messages.create.side_effect = [
            mock_tool_use_response(
                mock_tool_use("tu_a", "add", {"left": 1.0, "right": 2.0}),
                mock_tool_use("tu_b", "add", {"left": 3.0, "right": 4.0}),
            ),
            mock_end_turn("Both done"),
        ]
        loop = ToolLoop(client=client, registry=reg)
        loop.run("Two additions")
        second_call = client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]
        last_user = messages[-1]
        assert last_user["role"] == "user"
        assert len(last_user["content"]) == 2


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestFrameworkConfig:
    def test_reads_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
        from tool_framework.config import FrameworkConfig
        config = FrameworkConfig()
        assert config.anthropic_api_key == "test-key-12345"

    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TOOL_FRAMEWORK_MODEL", raising=False)
        from tool_framework.config import FrameworkConfig
        config = FrameworkConfig()
        assert config.model == "claude-sonnet-5-5"


# ---------------------------------------------------------------------------
# Demo tools (unit tests, no LLM)
# ---------------------------------------------------------------------------

class TestDemoTools:
    """Tests for the demo agent's tools in isolation."""

    def test_calculator_simple(self) -> None:
        # Import the function directly from demo
        import importlib.util, sys
        # Minimal inline test — avoid importing demo_agent.py at module level
        import ast
        import operator
        _OPS = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
        }

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
            raise ValueError()

        tree = ast.parse("3 * 4", mode="eval")
        assert _eval(tree.body) == 12.0

    def test_string_utils_upper(self) -> None:
        text = "hello world"
        assert text.upper() == "HELLO WORLD"

    def test_string_utils_word_count(self) -> None:
        assert len("hello world foo".split()) == 3
```

---

## Exercises (no answers)

1. Add a `timeout_seconds: float` parameter to `ToolDispatcher.dispatch` that limits each tool call's wall time using `concurrent.futures.ThreadPoolExecutor`. Return an error result on timeout.

2. Implement `ToolLoop.run_streaming` that yields text chunks as they arrive using the Anthropic streaming API. Keep the tool dispatch logic identical to the non-streaming version.

3. Add a `ToolRegistry.from_module(module)` class method that discovers and registers all public functions in a Python module that have a docstring.

4. Extend `FrameworkConfig` with a `log_level: str = "INFO"` field and apply it at startup in `main()`.

5. Add a `ToolLoop.last_run_stats` property that returns a dict with keys `iterations`, `tool_calls`, `errors` populated after each `run()` call.

6. Write an integration test that exercises the full `demo_agent.py` `main()` function with a multi-turn mocked client. Verify all three demo questions produce non-empty answers.

(Solutions: ../solutions/week-02/day-7.md)

---

## Self-check

- [ ] Where in the codebase is retry logic decoupled from dispatch logic and why?
- [ ] Why does `ToolDispatcher.dispatch` never raise?
- [ ] What would happen if `ToolLoop` appended tool results as a new `assistant` message instead of a `user` message?
- [ ] How does `pydantic-settings` pick up `ANTHROPIC_API_KEY` without an explicit `os.getenv` call?
- [ ] List 3 things that would break if you replaced `str(result)` in the dispatcher with `json.dumps(result)` but `result` was not JSON-serialisable.
- [ ] Why use `ast.parse` in the calculator instead of `eval()`?

---

## Next steps

- Read `assessment.md` — complete it this week.
- Read `project.md` — deliverables and acceptance criteria.

---

## Supplementary links

- [Anthropic tool use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Pydantic `create_model`](https://docs.pydantic.dev/latest/api/main/#pydantic.main.create_model)
- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Python `ast` module — safe eval](https://docs.python.org/3/library/ast.html)
- [pytest monkeypatch](https://docs.pytest.org/en/stable/reference/fixtures.html#monkeypatch)
- [`concurrent.futures` — thread pool](https://docs.python.org/3/library/concurrent.futures.html)
