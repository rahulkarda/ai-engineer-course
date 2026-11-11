# Phase 5 · Week 02 · Day 2 — Defining Tool Schemas

> Objective: Write precise, LLM-readable tool schemas by hand and auto-generate them from Python type hints using Pydantic v2.  ·  Time: ~1 hr  ·  Prereqs: [[phase-5-agentic/week-02/day-1]]

## Concept

A **tool schema** is the contract between your code and the LLM. It tells the model:

1. The tool's **name** — the identifier it will echo back in `tool_use.name`
2. A **description** — *the LLM reads this to decide whether and when to call the tool*
3. An **input_schema** — JSON Schema describing every parameter the tool accepts

The quality of descriptions is the single biggest lever on model accuracy. A vague description leads to wrong tool selection; a precise one leads to reliable routing.

### The Anthropic tools list format

```python
tools = [
    {
        "name": "tool_name",           # snake_case, unique per request
        "description": "...",          # crucial — full sentence(s)
        "input_schema": {              # JSON Schema object
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "...",  # per-parameter docstring
                },
            },
            "required": ["param_name"],    # list of required param names
        },
    }
]
```

### Writing effective descriptions

The LLM treats your `description` field like a function docstring in a code-completion context. It uses it to decide:

- **Should I call this tool at all?** ("use this when the user asks for live weather data")
- **Which tool among several?** (distinguish `search_web` from `search_database`)
- **What values to pass?** (per-property descriptions set expectations)

**Rules of thumb:**

| Do | Don't |
|---|---|
| "Use this when the user requests current weather for a location." | "Gets weather." |
| "ISO 8601 date string, e.g. '2024-03-15'." | "A date." |
| "Returns one of: 'sunny', 'cloudy', 'rain', 'snow'." | "The weather condition." |
| Name the trigger condition explicitly. | Leave the model to guess. |
| List enum values in the property description. | Rely on the enum field alone. |

### JSON Schema for tool inputs

Tool parameters use a subset of JSON Schema Draft 7. The `input_schema` must be:

```
{
  "type": "object",
  "properties": { ... },   # each property is itself a JSON Schema node
  "required": [ ... ]      # names of properties that must be present
}
```

Supported property types: `"string"`, `"number"`, `"integer"`, `"boolean"`, `"array"`, `"object"`.

Optional but useful:
- `"enum": [...]` — restrict to a fixed set of values
- `"description"` — per-property docstring read by the LLM
- `"default"` — informational only (the API does not auto-fill defaults)

## How it works internally

### Manual schema vs. auto-generation

**Manual** — write the dict by hand. Full control, verbose, easy to drift from the Python implementation.

**Auto-generated** — derive the schema from the Python function signature using Pydantic v2's `model_json_schema()`. The schema stays in sync with the code automatically.

### Auto-generation pipeline

```
Python function signature
        │
        ▼
Pydantic BaseModel (one field per parameter, types from hints)
        │
        ▼
model_json_schema()  →  JSON Schema dict
        │
        ▼
Anthropic tool dict  (wrap in {"name": ..., "description": ..., "input_schema": ...})
```

Pydantic translates Python types to JSON Schema types:

| Python type | JSON Schema type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |
| `Optional[str]` | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` |

Pydantic also picks up `Field(description="...")` and uses it as the property description — the LLM reads these.

## Worked example

Three tools defined both manually and via Pydantic auto-generation: `calculator`, `file_reader`, and `send_email`.

```python
# phase-5-agentic/week-02/examples/day2_tool_schemas.py
"""
Tool schema definition — manual JSON Schema vs. Pydantic auto-generation.
Run: python day2_tool_schemas.py
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ===========================================================================
# APPROACH 1 — Manual JSON Schema dicts
# ===========================================================================

CALCULATOR_TOOL_MANUAL: dict[str, Any] = {
    "name": "calculator",
    "description": (
        "Perform basic arithmetic (add, subtract, multiply, divide) on two numbers. "
        "Use this whenever the user needs a precise numeric calculation. "
        "Do not use for symbolic math or algebra."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "Arithmetic operation to perform.",
            },
            "a": {
                "type": "number",
                "description": "Left-hand operand.",
            },
            "b": {
                "type": "number",
                "description": "Right-hand operand. Must not be 0 when operation is 'divide'.",
            },
        },
        "required": ["operation", "a", "b"],
    },
}

FILE_READER_TOOL_MANUAL: dict[str, Any] = {
    "name": "file_reader",
    "description": (
        "Read the text contents of a local file given its absolute path. "
        "Use this when the user refers to a file on disk by path. "
        "Returns the raw UTF-8 text content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the file, e.g. '/home/user/notes.txt'.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum number of bytes to read. Defaults to 4096.",
            },
        },
        "required": ["path"],
    },
}

SEND_EMAIL_TOOL_MANUAL: dict[str, Any] = {
    "name": "send_email",
    "description": (
        "Send an email to one recipient. "
        "Use this when the user explicitly asks to send, draft-and-send, or email someone. "
        "Do NOT use just to compose a draft."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "body": {
                "type": "string",
                "description": "Plain-text email body.",
            },
            "cc": {
                "type": "string",
                "description": "Optional CC address.",
            },
        },
        "required": ["to", "subject", "body"],
    },
}


# ===========================================================================
# APPROACH 2 — Pydantic auto-generation
# ===========================================================================
# Strategy: define a Pydantic input model for each tool, then derive the
# JSON Schema from it. Keep the tool description in a docstring/constant.


class CalculatorInput(BaseModel):
    """Input model for the calculator tool."""

    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        description="Arithmetic operation to perform."
    )
    a: float = Field(description="Left-hand operand.")
    b: float = Field(
        description="Right-hand operand. Must not be 0 when operation is 'divide'."
    )


class FileReaderInput(BaseModel):
    """Input model for the file_reader tool."""

    path: str = Field(
        description="Absolute path to the file, e.g. '/home/user/notes.txt'."
    )
    max_bytes: int = Field(
        default=4096,
        description="Maximum number of bytes to read. Defaults to 4096.",
    )


class SendEmailInput(BaseModel):
    """Input model for the send_email tool."""

    to: str = Field(description="Recipient email address.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Plain-text email body.")
    cc: str | None = Field(default=None, description="Optional CC address.")


def pydantic_model_to_tool(
    model: type[BaseModel],
    tool_name: str,
    tool_description: str,
) -> dict[str, Any]:
    """
    Convert a Pydantic v2 BaseModel into an Anthropic tool dict.

    Steps:
    1. Call model.model_json_schema() to get the full JSON Schema.
    2. Strip Pydantic metadata fields ($defs, title, etc.) not needed by the API.
    3. Wrap in {"name": ..., "description": ..., "input_schema": ...}.
    """
    raw_schema = model.model_json_schema()

    # Anthropic expects the input_schema to be a plain object schema.
    # model_json_schema() returns {"title": ..., "type": "object", "properties": ..., ...}
    # We keep only the keys the API cares about.
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": raw_schema.get("properties", {}),
    }
    if "required" in raw_schema:
        input_schema["required"] = raw_schema["required"]
    # Handle $defs for nested models (not needed for flat models)
    if "$defs" in raw_schema:
        input_schema["$defs"] = raw_schema["$defs"]

    return {
        "name": tool_name,
        "description": tool_description,
        "input_schema": input_schema,
    }


CALCULATOR_TOOL_PYDANTIC = pydantic_model_to_tool(
    CalculatorInput,
    tool_name="calculator",
    tool_description=(
        "Perform basic arithmetic (add, subtract, multiply, divide) on two numbers. "
        "Use this whenever the user needs a precise numeric calculation."
    ),
)

FILE_READER_TOOL_PYDANTIC = pydantic_model_to_tool(
    FileReaderInput,
    tool_name="file_reader",
    tool_description=(
        "Read the text contents of a local file given its absolute path. "
        "Returns the raw UTF-8 text content."
    ),
)

SEND_EMAIL_TOOL_PYDANTIC = pydantic_model_to_tool(
    SendEmailInput,
    tool_name="send_email",
    tool_description=(
        "Send an email to one recipient. "
        "Use only when the user explicitly asks to send an email."
    ),
)


# ===========================================================================
# BONUS — introspection helper: build a schema from a bare Python function
# ===========================================================================


def function_to_tool_schema(fn: Any, description: str) -> dict[str, Any]:
    """
    Derive a minimal Anthropic tool schema from a Python function's type hints.
    Only supports flat str/int/float/bool parameters.
    For production use prefer the Pydantic approach above.
    """
    hints = fn.__annotations__.copy()
    hints.pop("return", None)  # exclude return type

    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        py_type = hints.get(name, str)
        json_type = type_map.get(py_type, "string")
        properties[name] = {"type": json_type, "description": f"Parameter '{name}'."}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": fn.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def sample_fn(city: str, days: int = 7) -> str:
    """Example function for schema introspection demo."""
    return f"Forecast for {city} over {days} days"


if __name__ == "__main__":
    logger.info("=== Manual schema ===")
    logger.info(json.dumps(CALCULATOR_TOOL_MANUAL, indent=2))

    logger.info("=== Pydantic auto-generated schema ===")
    logger.info(json.dumps(CALCULATOR_TOOL_PYDANTIC, indent=2))

    logger.info("=== Diff check — required fields match? ===")
    manual_req = set(CALCULATOR_TOOL_MANUAL["input_schema"]["required"])
    pydantic_req = set(CALCULATOR_TOOL_PYDANTIC["input_schema"].get("required", []))
    logger.info("Manual required: %s", manual_req)
    logger.info("Pydantic required: %s", pydantic_req)
    assert manual_req == pydantic_req, "Required fields differ!"

    logger.info("=== Introspection-based schema for sample_fn ===")
    auto = function_to_tool_schema(sample_fn, "Get a weather forecast for a city.")
    logger.info(json.dumps(auto, indent=2))
```

### Minimal test

```python
# tests/unit/phase5/week02/test_day2.py
"""Tests for day2_tool_schemas.py — no LLM calls."""
from __future__ import annotations

import pytest

from phase_5_agentic.week_02.examples.day2_tool_schemas import (
    CALCULATOR_TOOL_MANUAL,
    CALCULATOR_TOOL_PYDANTIC,
    FILE_READER_TOOL_PYDANTIC,
    SEND_EMAIL_TOOL_PYDANTIC,
    CalculatorInput,
    SendEmailInput,
    function_to_tool_schema,
    pydantic_model_to_tool,
    sample_fn,
)


def _assert_valid_tool(tool: dict) -> None:
    """Every Anthropic tool dict must have these three top-level keys."""
    assert "name" in tool
    assert "description" in tool
    assert "input_schema" in tool
    schema = tool["input_schema"]
    assert schema["type"] == "object"
    assert "properties" in schema


def test_manual_calculator_schema_is_valid() -> None:
    _assert_valid_tool(CALCULATOR_TOOL_MANUAL)
    assert "operation" in CALCULATOR_TOOL_MANUAL["input_schema"]["properties"]
    assert set(CALCULATOR_TOOL_MANUAL["input_schema"]["required"]) == {"operation", "a", "b"}


def test_pydantic_calculator_schema_is_valid() -> None:
    _assert_valid_tool(CALCULATOR_TOOL_PYDANTIC)
    assert set(CALCULATOR_TOOL_PYDANTIC["input_schema"].get("required", [])) == {
        "operation", "a", "b"
    }


def test_manual_and_pydantic_required_fields_match() -> None:
    manual_req = set(CALCULATOR_TOOL_MANUAL["input_schema"]["required"])
    pydantic_req = set(CALCULATOR_TOOL_PYDANTIC["input_schema"].get("required", []))
    assert manual_req == pydantic_req


def test_pydantic_file_reader_max_bytes_not_required() -> None:
    schema = FILE_READER_TOOL_PYDANTIC["input_schema"]
    required = schema.get("required", [])
    assert "path" in required
    assert "max_bytes" not in required


def test_pydantic_send_email_cc_not_required() -> None:
    schema = SEND_EMAIL_TOOL_PYDANTIC["input_schema"]
    required = schema.get("required", [])
    assert "cc" not in required


def test_calculator_input_model_validates() -> None:
    data = CalculatorInput(operation="add", a=3.0, b=4.0)
    assert data.operation == "add"


def test_calculator_input_model_rejects_bad_operation() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        CalculatorInput(operation="power", a=2.0, b=3.0)


def test_send_email_cc_defaults_to_none() -> None:
    m = SendEmailInput(to="a@b.com", subject="Hi", body="Hello")
    assert m.cc is None


def test_function_introspection_marks_required() -> None:
    schema = function_to_tool_schema(sample_fn, "Forecast tool")
    required = schema["input_schema"]["required"]
    assert "city" in required
    assert "days" not in required  # has default


def test_pydantic_model_to_tool_preserves_name() -> None:
    tool = pydantic_model_to_tool(CalculatorInput, "my_tool", "Does something.")
    assert tool["name"] == "my_tool"
    assert tool["description"] == "Does something."
```

## Exercises

1. **Schema from scratch.** Define a manual Anthropic tool schema for a `web_search` tool that accepts a `query` (required string), `num_results` (optional integer, default 5), and `safe_search` (optional boolean, default True). Write the description following the rules in the Concept section — trigger condition, input format expectations, output summary. Verify your schema is valid by parsing it with `json.loads(json.dumps(schema))`.

2. **Pydantic auto-generation.** Create a Pydantic `BaseModel` called `DatabaseQueryInput` with fields: `sql` (str, required, with a Field description), `database` (Literal of at least three database names, required), and `timeout_seconds` (int, optional, default 30). Call `pydantic_model_to_tool()` with it and assert that `"sql"` and `"database"` appear in `required` but `"timeout_seconds"` does not.

3. **Description impact experiment.** Take the `send_email` tool schema (either version) and intentionally write a *bad* description — vague, no trigger condition, no output summary. Then write the *good* version following the rules above. In comments, predict: for the user message "Draft me an email to Alice about the meeting but don't send it yet", which description version would cause the LLM to incorrectly call `send_email`? Why?

(Solutions: ../solutions/week-02/day-2.md)

## Self-check

- [ ] I can explain all three top-level keys of an Anthropic tool dict and what each is used for
- [ ] I can write a `description` that includes a trigger condition, input format hint, and output summary
- [ ] I can define a Pydantic BaseModel with Field descriptions and call `pydantic_model_to_tool()` to get a valid schema
- [ ] I can explain why `Optional[str]` fields should not appear in `required` and verify this with `model_json_schema()`

## Supplementary links

- [Anthropic tool use — defining tools](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#defining-tools)
- [Pydantic v2 — model_json_schema()](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Pydantic v2 — Field](https://docs.pydantic.dev/latest/api/fields/)
- [JSON Schema — type keywords](https://json-schema.org/understanding-json-schema/reference/type.html)
- [JSON Schema — enum](https://json-schema.org/understanding-json-schema/reference/enum.html)
