# Solutions — Phase 5 · Week 2 · Day 4 — Input/Output Validation and Error Handling

---

## Exercise 1 — `CalculatorInput` + `run_calculator`

```python
import logging
import re
from typing import Any

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Re-use ToolResult and ToolError defined in Exercise 2 below.
# Importing here for self-contained demonstration:


class ToolResult(BaseModel):
    """Successful tool execution result."""
    tool_name: str
    output: str
    is_error: bool = False


class ToolError(BaseModel):
    """Failed tool execution result."""
    tool_name: str
    error_message: str
    is_error: bool = True

    def to_tool_result(self) -> ToolResult:
        """Convert to a ToolResult with is_error=True for the API message."""
        return ToolResult(
            tool_name=self.tool_name,
            output=self.error_message,
            is_error=True,
        )


class CalculatorInput(BaseModel):
    """Validated input for the calculator tool.

    Only allows digits and the operators +-*/()., preventing code injection.
    """
    expression: str

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Allow only digits, basic operators, parentheses, dots, and spaces."""
        allowed_pattern = re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$")
        if not allowed_pattern.match(v):
            raise ValueError(
                f"Expression contains disallowed characters: {v!r}. "
                "Only digits and operators +-*/().  are permitted."
            )
        # Additional safety: no consecutive operators that could confuse eval
        if re.search(r"[+\-*/]{2,}", v.replace(" ", "")):
            raise ValueError(f"Expression has consecutive operators: {v!r}")
        return v.strip()


def run_calculator(raw_input: dict[str, Any]) -> ToolResult:
    """
    Validate and evaluate a math expression.

    Args:
        raw_input: Raw dict from the model's tool_use block (e.g. {"expression": "2 + 3 * 4"}).

    Returns:
        ToolResult with the numeric result, or ToolResult(is_error=True) on failure.
    """
    tool_name = "calculator"

    try:
        validated = CalculatorInput(**raw_input)
    except ValueError as exc:
        logger.warning("Calculator validation failed: %s", exc)
        return ToolError(tool_name=tool_name, error_message=str(exc)).to_tool_result()

    try:
        # eval is safe here because CalculatorInput.validate_expression
        # enforces a strict allowlist of characters before this point.
        result = eval(validated.expression)  # noqa: S307 — allowlist-validated
        logger.info("Calculator: %s = %s", validated.expression, result)
        return ToolResult(tool_name=tool_name, output=str(result))

    except ZeroDivisionError:
        return ToolError(tool_name=tool_name, error_message="Division by zero.").to_tool_result()
    except Exception as exc:
        return ToolError(
            tool_name=tool_name, error_message=f"Evaluation error: {exc}"
        ).to_tool_result()


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(run_calculator({"expression": "2 + 3 * 4"}))        # output='14'
    print(run_calculator({"expression": "(10 / 2) + 3"}))     # output='8.0'
    print(run_calculator({"expression": "1 / 0"}))            # is_error=True
    print(run_calculator({"expression": "__import__('os')"})) # is_error=True (blocked)
```

---

## Exercise 2 — `ToolError`, `ToolResult`, and `@validate_tool_io` Decorator

```python
import functools
import logging
from typing import Any, Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

InputModel = TypeVar("InputModel", bound=BaseModel)
OutputModel = TypeVar("OutputModel", bound=BaseModel)


# ── Shared result models ──────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """Successful (or error) tool execution result for the Anthropic tool_result message."""
    tool_name: str
    output: str
    is_error: bool = False

    def to_api_content(self) -> dict[str, Any]:
        """Serialize for use in a tool_result content block."""
        return {"result": self.output, "is_error": self.is_error}


class ToolError(BaseModel):
    """Structured error from a tool execution failure."""
    tool_name: str
    error_message: str
    error_type: str = "ToolExecutionError"
    is_error: bool = True

    def to_tool_result(self) -> ToolResult:
        """Convert to ToolResult so callers always get the same type."""
        return ToolResult(
            tool_name=self.tool_name,
            output=self.error_message,
            is_error=True,
        )


# ── Decorator ─────────────────────────────────────────────────────────────────

def validate_tool_io(
    input_model: Type[BaseModel],
    output_model: Type[BaseModel],
) -> Callable[[Callable[..., Any]], Callable[..., ToolResult]]:
    """
    Decorator that validates tool input and output using Pydantic models.

    Usage:
        @validate_tool_io(MyInputModel, MyOutputModel)
        def my_tool(validated_input: MyInputModel) -> MyOutputModel:
            ...

    The decorated function receives a VALIDATED input model instance (not a raw dict).
    The decorator wraps it so the caller passes a raw dict and gets a ToolResult back.

    - Input validation failure → ToolResult(is_error=True) with the validation message.
    - Output validation failure → ToolResult(is_error=True) with the validation message.
    - Any unhandled exception → ToolResult(is_error=True) with the exception message.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., ToolResult]:
        tool_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(raw_input: dict[str, Any]) -> ToolResult:
            # 1. Validate input
            try:
                validated_input = input_model(**raw_input)
            except ValidationError as exc:
                error_msg = f"Input validation failed: {exc}"
                logger.warning("[%s] %s", tool_name, error_msg)
                return ToolError(
                    tool_name=tool_name,
                    error_message=error_msg,
                    error_type="InputValidationError",
                ).to_tool_result()

            # 2. Execute the tool function
            try:
                raw_output = fn(validated_input)
            except Exception as exc:
                error_msg = f"Execution error: {type(exc).__name__}: {exc}"
                logger.error("[%s] %s", tool_name, error_msg)
                return ToolError(
                    tool_name=tool_name,
                    error_message=error_msg,
                    error_type=type(exc).__name__,
                ).to_tool_result()

            # 3. Validate output
            try:
                if not isinstance(raw_output, output_model):
                    # Coerce: allow returning a dict that matches the output model
                    if isinstance(raw_output, dict):
                        validated_output = output_model(**raw_output)
                    else:
                        validated_output = output_model.model_validate(raw_output)
                else:
                    validated_output = raw_output
            except (ValidationError, Exception) as exc:
                error_msg = f"Output validation failed: {exc}"
                logger.error("[%s] %s", tool_name, error_msg)
                return ToolError(
                    tool_name=tool_name,
                    error_message=error_msg,
                    error_type="OutputValidationError",
                ).to_tool_result()

            return ToolResult(
                tool_name=tool_name,
                output=validated_output.model_dump_json(),
            )

        return wrapper

    return decorator


# ── Example usage of the decorator ───────────────────────────────────────────

class AddInput(BaseModel):
    a: float
    b: float


class AddOutput(BaseModel):
    sum: float


@validate_tool_io(AddInput, AddOutput)
def add_numbers(inp: AddInput) -> AddOutput:
    """Add two numbers together."""
    return AddOutput(sum=inp.a + inp.b)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(add_numbers({"a": 3, "b": 4}))            # ToolResult(output='{"sum":7.0}')
    print(add_numbers({"a": "not_a_number", "b": 4}))  # ToolResult(is_error=True)
```

---

## Exercise 3 — Pytest: Invalid Tool Input → `is_error: true`

```python
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

# Import implementations from this module (in a real project these would be imports)
# from day4_solutions import CalculatorInput, ToolResult, ToolError, run_calculator
# For self-contained testing we inline the needed pieces.


# ── Re-import / inline the pieces needed for tests ────────────────────────────

import re
from pydantic import BaseModel, field_validator


class _ToolResult(BaseModel):
    tool_name: str
    output: str
    is_error: bool = False


class _ToolError(BaseModel):
    tool_name: str
    error_message: str
    is_error: bool = True

    def to_tool_result(self) -> "_ToolResult":
        return _ToolResult(tool_name=self.tool_name, output=self.error_message, is_error=True)


class _CalculatorInput(BaseModel):
    expression: str

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        if not re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$").match(v):
            raise ValueError(f"Disallowed characters in expression: {v!r}")
        return v.strip()


def _run_calculator(raw_input: dict[str, Any]) -> _ToolResult:
    try:
        validated = _CalculatorInput(**raw_input)
    except ValueError as exc:
        return _ToolError(tool_name="calculator", error_message=str(exc)).to_tool_result()
    try:
        result = eval(validated.expression)  # noqa: S307
        return _ToolResult(tool_name="calculator", output=str(result))
    except Exception as exc:
        return _ToolError(tool_name="calculator", error_message=str(exc)).to_tool_result()


def _build_tool_result_block(
    tool_use_id: str, content: str, is_error: bool = False
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCalculatorInvalidInput:
    """Tests for the calculator tool's validation and error handling."""

    def test_invalid_input_returns_is_error_true(self) -> None:
        """Sending a non-numeric expression returns is_error=True."""
        result = _run_calculator({"expression": "__import__('os').system('ls')"})

        assert result.is_error is True
        assert "Disallowed characters" in result.output or "disallowed" in result.output.lower()

    def test_tool_result_message_has_is_error_flag(self) -> None:
        """build_tool_result_block propagates is_error into the message content block."""
        error_result = _run_calculator({"expression": "abc + def"})

        # Simulate building the Anthropic tool_result message
        tool_result_block = _build_tool_result_block(
            tool_use_id="toolu_test_invalid",
            content=error_result.output,
            is_error=error_result.is_error,
        )

        assert tool_result_block["is_error"] is True
        assert "toolu_test_invalid" == tool_result_block["tool_use_id"]
        assert len(tool_result_block["content"]) > 0  # error message is present

    def test_valid_expression_does_not_have_error(self) -> None:
        """A valid expression returns is_error=False."""
        result = _run_calculator({"expression": "2 + 2"})

        assert result.is_error is False
        assert result.output == "4"

    def test_division_by_zero_returns_is_error_true(self) -> None:
        """Division by zero is caught and returns is_error=True."""
        result = _run_calculator({"expression": "10 / 0"})

        assert result.is_error is True
        assert "zero" in result.output.lower() or "error" in result.output.lower()

    def test_missing_required_field_returns_is_error_true(self) -> None:
        """Missing required field raises ValidationError caught as is_error=True."""
        # expression field is required; passing empty dict should error
        result = _run_calculator({})

        assert result.is_error is True

    def test_error_message_in_tool_result_content(self) -> None:
        """The tool_result content block includes a descriptive error message string."""
        result = _run_calculator({"expression": "2 ++ 3"})  # consecutive operators

        block = _build_tool_result_block(
            tool_use_id="toolu_consecutive",
            content=result.output,
            is_error=result.is_error,
        )

        # The content field carries the error description
        assert isinstance(block["content"], str)
        assert len(block["content"]) > 10  # not empty/trivial


# ── Integration: mock full Anthropic call with invalid tool input ─────────────

class TestMockedAPIWithInvalidToolInput:
    """Integration test: model sends bad tool input, we return is_error, model handles it."""

    def test_full_flow_with_invalid_tool_input(self) -> None:
        """End-to-end: API returns tool_use with bad input → is_error in tool_result."""
        # Simulate the Anthropic API returning a tool_use block with invalid input
        mock_tool_use_block = MagicMock()
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.id = "toolu_bad_input"
        mock_tool_use_block.name = "calculator"
        mock_tool_use_block.input = {"expression": "exec('import os')"}  # injection attempt

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_tool_use_block]

        # Process the tool call
        raw_input = mock_tool_use_block.input
        result = _run_calculator(raw_input)

        # Build the API tool_result message
        tool_result_msg = {
            "role": "user",
            "content": [
                _build_tool_result_block(
                    tool_use_id=mock_tool_use_block.id,
                    content=result.output,
                    is_error=result.is_error,
                )
            ],
        }

        # Assertions
        assert result.is_error is True, "Invalid expression must return is_error=True"
        tool_result_block = tool_result_msg["content"][0]
        assert tool_result_block["is_error"] is True
        assert tool_result_block["tool_use_id"] == "toolu_bad_input"
        assert len(tool_result_block["content"]) > 0, "Error message must be non-empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Key design decisions:**
- `CalculatorInput.validate_expression` uses a strict allowlist regex — rejects anything not in `[\d\s+\-*/().]`. This makes `eval` safe by construction.
- `@validate_tool_io` separates validation concerns from business logic — the tool function only ever sees a typed Pydantic model.
- Tests use `MagicMock` — zero real API calls. The mock simulates the SDK's response object structure.
