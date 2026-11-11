# Phase 5 · Week 02 · Day 4 — Input/Output Validation and Error Handling in Tools

> Objective: Validate tool inputs and outputs with Pydantic v2, handle errors gracefully, and return structured errors back to the LLM via the `is_error` field.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-02/day-3]]

## Concept

The LLM generates tool arguments as JSON. That JSON can be:

- **Syntactically valid** but semantically wrong (`"temperature": "hot"` instead of a number)
- **Missing required fields** (the LLM omitted a field it should have included)
- **Out of range** (`page: -1`, `limit: 99999`)
- **Correct** but the tool's downstream call fails (network error, file not found)

Without validation, any of these silently corrupt your tool's behaviour or raise an unhandled exception that crashes the agent loop. With Pydantic, you parse the raw `tool_use.input` dict into a typed model *before* executing any logic — validation errors become explicit, catchable, and reportable.

### Two categories of tool error

| Category | Cause | Handling strategy |
|---|---|---|
| **Validation error** | Bad input from the LLM | Return `is_error: True` tool_result — the model can self-correct |
| **Execution error** | Tool logic failed (network, file, DB) | Return `is_error: True` tool_result with context — model decides what to do |
| **System error** | Infrastructure failure (OOM, disk full) | Re-raise as Python exception — the agent loop handles it at a higher level |

The distinction between the first two and the last is important: if you `raise` from a tool, the agent loop crashes. If you return `is_error: True`, the model sees the error and can respond intelligently — "I tried to read that file but it didn't exist; would you like me to try a different path?"

### The `is_error` field

```python
{
    "type": "tool_result",
    "tool_use_id": "<id>",
    "content": "Error: file '/tmp/data.csv' not found.",
    "is_error": True,   # tells the model the tool failed
}
```

When `is_error` is `True`, the model treats `content` as an error description rather than a result. It can then:
- Acknowledge the error to the user
- Try a different approach
- Ask the user for clarification

### Structured error types

Use Pydantic models for error payloads too. JSON-serialisable, self-documenting, and easy to test.

```python
class ToolError(BaseModel):
    error_code: str
    message: str
    field: str | None = None  # for validation errors: which field failed

class ToolSuccess(BaseModel):
    ...  # your result fields
```

## How it works internally

### Validation flow

```
tool_use.input (dict from LLM)
        │
        ▼
MyToolInput.model_validate(input_dict)
        │
   ┌────┴────┐
   │         │
   ▼         ▼
success   ValidationError
   │         │
run tool   build ToolError
   │         │
return     return is_error tool_result
result
```

`model_validate()` raises `pydantic.ValidationError` on failure. This is deterministic and testable — you can write tests that feed bad inputs and assert the correct error structure.

### Output validation

Tool outputs should also be typed. A return type annotation + Pydantic model:

1. Documents the contract
2. Catches bugs where the tool returns unexpected keys or types
3. Makes the serialised `content` string predictable for tests

```python
def get_weather(input: WeatherInput) -> WeatherOutput:
    ...
    return WeatherOutput(city=..., temperature=..., condition=...)

# Serialise with model_dump_json() before putting in tool_result content
```

### Pydantic validators for business rules

Use `@field_validator` or `model_validator` for constraints that JSON Schema cannot express:

```python
from pydantic import field_validator

class DivideInput(BaseModel):
    a: float
    b: float

    @field_validator("b")
    @classmethod
    def b_must_not_be_zero(cls, v: float) -> float:
        if v == 0.0:
            raise ValueError("b must not be zero — division by zero is undefined")
        return v
```

## Worked example

A `read_file` tool with full input validation, output validation, and error handling.

```python
# phase-5-agentic/week-02/examples/day4_tool_validation.py
"""
Tool input/output validation and error handling with Pydantic v2.
Run: python day4_tool_validation.py
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# 1. Input model — validated before the tool runs
# ---------------------------------------------------------------------------


class ReadFileInput(BaseModel):
    """
    Validated input for the read_file tool.
    The LLM's tool_use.input dict is parsed into this model.
    """

    path: str = Field(description="Absolute path to the file to read.")
    max_bytes: int = Field(
        default=4096,
        ge=1,        # must be >= 1
        le=1_000_000,  # at most 1 MB
        description="Maximum bytes to read. 1–1,000,000.",
    )
    encoding: str = Field(
        default="utf-8",
        description="File encoding, e.g. 'utf-8' or 'latin-1'.",
    )

    @field_validator("path")
    @classmethod
    def path_must_be_absolute(cls, v: str) -> str:
        if not Path(v).is_absolute():
            raise ValueError(
                f"path must be an absolute path, got relative path: {v!r}"
            )
        return v

    @field_validator("encoding")
    @classmethod
    def encoding_must_be_known(cls, v: str) -> str:
        import codecs
        try:
            codecs.lookup(v)
        except LookupError:
            raise ValueError(f"Unknown encoding: {v!r}")
        return v


# ---------------------------------------------------------------------------
# 2. Output model — returned by the tool on success
# ---------------------------------------------------------------------------


class ReadFileOutput(BaseModel):
    """Structured success result from read_file."""

    path: str
    content: str
    bytes_read: int
    truncated: bool = Field(
        description="True if the file was larger than max_bytes and was truncated."
    )


# ---------------------------------------------------------------------------
# 3. Error model — returned on any failure
# ---------------------------------------------------------------------------


class ToolError(BaseModel):
    """Structured error payload — serialised into the tool_result content."""

    error_code: str = Field(description="Machine-readable error identifier.")
    message: str = Field(description="Human-readable description.")
    field: str | None = Field(
        default=None,
        description="For validation errors: the field that failed.",
    )


ERROR_CODES = {
    "VALIDATION_ERROR": "VALIDATION_ERROR",
    "FILE_NOT_FOUND": "FILE_NOT_FOUND",
    "PERMISSION_DENIED": "PERMISSION_DENIED",
    "READ_ERROR": "READ_ERROR",
}


# ---------------------------------------------------------------------------
# 4. Tool implementation — uses the validated input model
# ---------------------------------------------------------------------------


def read_file(validated_input: ReadFileInput) -> ReadFileOutput:
    """
    Read a file from disk. Raises OSError on filesystem errors.
    Never call this with an unvalidated dict — always pass a ReadFileInput.
    """
    p = Path(validated_input.path)
    logger.info("read_file: path=%s max_bytes=%d", p, validated_input.max_bytes)

    # Read raw bytes
    raw = p.read_bytes()
    truncated = len(raw) > validated_input.max_bytes
    chunk = raw[: validated_input.max_bytes]

    content = chunk.decode(validated_input.encoding, errors="replace")
    logger.info(
        "read_file: read %d bytes (truncated=%s)", len(chunk), truncated
    )
    return ReadFileOutput(
        path=str(p),
        content=content,
        bytes_read=len(chunk),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# 5. Dispatcher — validates input, calls tool, returns tool_result dict
# ---------------------------------------------------------------------------


def dispatch_read_file(tool_use_id: str, raw_input: dict[str, Any]) -> dict[str, Any]:
    """
    Full dispatch pipeline:
    1. Validate raw_input with Pydantic
    2. Call the tool
    3. Validate the output
    4. Return a tool_result dict (with is_error if anything failed)

    Never raises — all errors are converted to is_error tool_result dicts.
    """
    # Step 1 — Input validation
    try:
        validated = ReadFileInput.model_validate(raw_input)
    except ValidationError as exc:
        # Extract the first error for a concise message
        first_error = exc.errors()[0]
        field_name = ".".join(str(loc) for loc in first_error["loc"])
        error = ToolError(
            error_code=ERROR_CODES["VALIDATION_ERROR"],
            message=first_error["msg"],
            field=field_name,
        )
        logger.warning("Validation error for read_file: %s", error)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": error.model_dump_json(),
            "is_error": True,
        }

    # Step 2 — Execution
    try:
        output = read_file(validated)
    except FileNotFoundError:
        error = ToolError(
            error_code=ERROR_CODES["FILE_NOT_FOUND"],
            message=f"File not found: {validated.path!r}",
        )
        logger.warning("File not found: %s", validated.path)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": error.model_dump_json(),
            "is_error": True,
        }
    except PermissionError:
        error = ToolError(
            error_code=ERROR_CODES["PERMISSION_DENIED"],
            message=f"Permission denied: {validated.path!r}",
        )
        logger.warning("Permission denied: %s", validated.path)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": error.model_dump_json(),
            "is_error": True,
        }
    except OSError as exc:
        error = ToolError(
            error_code=ERROR_CODES["READ_ERROR"],
            message=str(exc),
        )
        logger.error("OS error reading %s: %s", validated.path, exc)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": error.model_dump_json(),
            "is_error": True,
        }

    # Step 3 — Serialize validated output
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": output.model_dump_json(),
    }


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Case 1: Valid input, real file (create a temp file for the demo)
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        tmp.write("Hello from the demo file!\n" * 10)
        tmp_path = tmp.name

    logger.info("=== Case 1: valid input ===")
    result = dispatch_read_file("id_001", {"path": tmp_path, "max_bytes": 50})
    parsed = json.loads(result["content"])
    logger.info("is_error: %s", result.get("is_error", False))
    logger.info("bytes_read: %s", parsed.get("bytes_read"))
    logger.info("truncated: %s", parsed.get("truncated"))
    Path(tmp_path).unlink()

    # Case 2: Relative path — validation error
    logger.info("=== Case 2: relative path (validation error) ===")
    result2 = dispatch_read_file("id_002", {"path": "relative/path.txt"})
    logger.info("is_error: %s", result2.get("is_error"))
    logger.info("content: %s", result2["content"])

    # Case 3: File does not exist — execution error
    logger.info("=== Case 3: file not found ===")
    result3 = dispatch_read_file("id_003", {"path": "/nonexistent/file.txt"})
    logger.info("is_error: %s", result3.get("is_error"))
    logger.info("content: %s", result3["content"])

    # Case 4: max_bytes out of range — validation error
    logger.info("=== Case 4: max_bytes=0 (out of range) ===")
    result4 = dispatch_read_file(
        "id_004", {"path": "/tmp/x.txt", "max_bytes": 0}
    )
    logger.info("is_error: %s", result4.get("is_error"))
    error_data = json.loads(result4["content"])
    logger.info("error_code: %s  field: %s", error_data["error_code"], error_data["field"])
```

### Minimal tests

```python
# tests/unit/phase5/week02/test_day4.py
"""Tests for day4_tool_validation.py — no real files except controlled tempfiles."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from phase_5_agentic.week_02.examples.day4_tool_validation import (
    ReadFileInput,
    ReadFileOutput,
    ToolError,
    dispatch_read_file,
    read_file,
)


# ---------------------------------------------------------------------------
# Input model tests
# ---------------------------------------------------------------------------


def test_valid_input_parses() -> None:
    inp = ReadFileInput.model_validate({"path": "/tmp/file.txt"})
    assert inp.path == "/tmp/file.txt"
    assert inp.max_bytes == 4096
    assert inp.encoding == "utf-8"


def test_relative_path_raises_validation_error() -> None:
    with pytest.raises(ValidationError, match="absolute"):
        ReadFileInput.model_validate({"path": "relative/path.txt"})


def test_max_bytes_zero_raises() -> None:
    with pytest.raises(ValidationError):
        ReadFileInput.model_validate({"path": "/tmp/f.txt", "max_bytes": 0})


def test_max_bytes_too_large_raises() -> None:
    with pytest.raises(ValidationError):
        ReadFileInput.model_validate({"path": "/tmp/f.txt", "max_bytes": 2_000_000})


def test_bad_encoding_raises() -> None:
    with pytest.raises(ValidationError, match="Unknown encoding"):
        ReadFileInput.model_validate({"path": "/tmp/f.txt", "encoding": "not-a-codec"})


# ---------------------------------------------------------------------------
# Tool execution tests (use real temp files)
# ---------------------------------------------------------------------------


def test_read_file_success() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    try:
        inp = ReadFileInput.model_validate({"path": path})
        out = read_file(inp)
        assert isinstance(out, ReadFileOutput)
        assert "hello" in out.content
        assert not out.truncated
    finally:
        Path(path).unlink(missing_ok=True)


def test_read_file_truncates() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("a" * 100)
        path = f.name
    try:
        inp = ReadFileInput.model_validate({"path": path, "max_bytes": 10})
        out = read_file(inp)
        assert out.truncated
        assert out.bytes_read == 10
    finally:
        Path(path).unlink(missing_ok=True)


def test_read_file_missing_raises_file_not_found() -> None:
    inp = ReadFileInput.model_validate({"path": "/nonexistent/missing_file.txt"})
    with pytest.raises(FileNotFoundError):
        read_file(inp)


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


def test_dispatcher_returns_success_result() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("data")
        path = f.name
    try:
        result = dispatch_read_file("id_001", {"path": path})
        assert result.get("is_error") is not True
        assert result["tool_use_id"] == "id_001"
        data = json.loads(result["content"])
        assert "content" in data
    finally:
        Path(path).unlink(missing_ok=True)


def test_dispatcher_validation_error_sets_is_error() -> None:
    result = dispatch_read_file("id_002", {"path": "not/absolute"})
    assert result["is_error"] is True
    error = json.loads(result["content"])
    assert error["error_code"] == "VALIDATION_ERROR"
    assert error["field"] == "path"


def test_dispatcher_file_not_found_sets_is_error() -> None:
    result = dispatch_read_file("id_003", {"path": "/no/such/file.txt"})
    assert result["is_error"] is True
    error = json.loads(result["content"])
    assert error["error_code"] == "FILE_NOT_FOUND"


def test_dispatcher_out_of_range_max_bytes() -> None:
    result = dispatch_read_file("id_004", {"path": "/tmp/x.txt", "max_bytes": 0})
    assert result["is_error"] is True
    error = json.loads(result["content"])
    assert error["error_code"] == "VALIDATION_ERROR"
    assert error["field"] == "max_bytes"


def test_dispatcher_never_raises() -> None:
    """dispatch_read_file must return a dict, never raise."""
    for bad_input in [
        {},
        {"path": None},
        {"path": "/tmp/f.txt", "max_bytes": -1},
        {"path": "/nonexistent_12345/file.txt"},
    ]:
        result = dispatch_read_file("id_x", bad_input)
        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] == "tool_result"
```

## Exercises

1. **Pydantic input model for a web search tool.** Write a `WebSearchInput` model with:
   - `query: str` (required, min_length=1, max_length=500)
   - `num_results: int` (optional, default=5, ge=1, le=20)
   - `safe_search: bool` (optional, default=True)
   
   Add a `@field_validator` that strips leading/trailing whitespace from `query` and raises `ValueError` if the stripped result is empty. Write tests for: valid input, empty query, query too long, num_results=0, num_results=21.

2. **Structured output model.** Define a `WebSearchOutput` model with `results: list[SearchResult]` where `SearchResult` has `title: str`, `url: str`, `snippet: str`. Write a `mock_web_search` function that accepts `WebSearchInput` and returns `WebSearchOutput` with two fake results. Verify that `mock_web_search(...).model_dump_json()` produces valid JSON and that both results appear in the `"results"` list.

3. **Error handling strategy comparison.** You have a `send_email` tool. Two implementation strategies:
   - **Strategy A**: if the SMTP server is unreachable, `raise ConnectionError("SMTP unreachable")`
   - **Strategy B**: return `{"type": "tool_result", "is_error": True, "content": "{\"error_code\": \"SMTP_UNREACHABLE\", ...}"}`
   
   For each strategy: describe what happens to the agent loop, describe what the LLM sees, and argue when each strategy is the right choice. Then implement Strategy B for a stub `send_email` dispatcher that always simulates an SMTP failure, and write a test that asserts `is_error == True` and `error_code == "SMTP_UNREACHABLE"`.

(Solutions: ../solutions/week-02/day-4.md)

## Self-check

- [ ] I can explain the difference between a validation error and an execution error and when to handle each as `is_error` vs. re-raise
- [ ] I can write a Pydantic input model with `field_validator` and parse a raw `tool_use.input` dict into it
- [ ] I can implement a dispatcher that catches `ValidationError` and `OSError` separately and returns the correct `is_error` tool_result in each case
- [ ] I can define a `ToolError` Pydantic model and serialise it with `model_dump_json()` for use in `tool_result.content`

## Supplementary links

- [Anthropic tool use — error handling](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#error-handling)
- [Pydantic v2 — validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Pydantic v2 — model_validate()](https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_validate)
- [Pydantic v2 — ValidationError](https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.ValidationError)
- [JSON Schema — constraints (minimum, maximum, minLength)](https://json-schema.org/understanding-json-schema/reference/numeric.html)
