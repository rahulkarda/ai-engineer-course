# Solutions — Phase 5 · Week 2 · Day 2 — Defining Tool Schemas

---

## Exercise 1 — `auto_tool_schema(func)` — Introspect a Function into a Tool Schema

```python
import inspect
import logging
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)


# Map Python types to JSON Schema type strings
_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    origin = getattr(py_type, "__origin__", None)

    # Handle list[X]
    if origin is list:
        args = getattr(py_type, "__args__", (Any,))
        item_schema = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}

    # Handle dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Primitives
    json_type = _PYTHON_TO_JSON_TYPE.get(py_type, "string")
    return {"type": json_type}


def auto_tool_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Extract a full Anthropic tool schema from a Python function.

    Uses:
    - func.__name__ as the tool name
    - func.__doc__ (first line) as the tool description
    - Type hints + default values to build input_schema

    Returns a dict ready to pass to client.messages.create(tools=[...]).
    """
    name: str = func.__name__

    # Parse docstring: first non-empty line is the description
    raw_doc: str = inspect.getdoc(func) or ""
    description: str = raw_doc.split("\n")[0].strip() if raw_doc else f"Call {name}"

    hints: dict[str, Any] = get_type_hints(func)
    hints.pop("return", None)  # exclude return type

    sig: inspect.Signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type = hints.get(param_name, str)
        prop_schema = _python_type_to_json_schema(py_type)

        # Try to extract per-param description from docstring (Google style)
        # e.g. "    directory: Root directory to list files in."
        prop_desc = ""
        for line in raw_doc.split("\n"):
            stripped = line.strip()
            if stripped.startswith(f"{param_name}:"):
                prop_desc = stripped[len(param_name) + 1:].strip()
                break
        if prop_desc:
            prop_schema["description"] = prop_desc

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop_schema["default"] = param.default

        properties[param_name] = prop_schema

    schema: dict[str, Any] = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    logger.debug("Generated schema for %s: %s", name, schema)
    return schema


# --- Sample function to demonstrate auto_tool_schema ---

def get_stock_price(ticker: str, currency: str = "USD") -> dict[str, Any]:
    """
    Retrieve the current stock price for a given ticker symbol.

    ticker: Stock ticker symbol, e.g. 'AAPL' or 'MSFT'.
    currency: Currency for the returned price. Defaults to 'USD'.
    """
    # Stub — real implementation calls a financial API
    return {"ticker": ticker, "price": 182.45, "currency": currency}


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.DEBUG)
    schema = auto_tool_schema(get_stock_price)
    print(json.dumps(schema, indent=2))
```

**Expected output:**
```json
{
  "name": "get_stock_price",
  "description": "Retrieve the current stock price for a given ticker symbol.",
  "input_schema": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "Stock ticker symbol, e.g. 'AAPL' or 'MSFT'."
      },
      "currency": {
        "type": "string",
        "description": "Currency for the returned price. Defaults to 'USD'.",
        "default": "USD"
      }
    },
    "required": ["ticker"]
  }
}
```

---

## Exercise 2 — File Operation Tool Schemas with Pydantic Models

```python
import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Pydantic input models ──────────────────────────────────────────────────────

class ListFilesInput(BaseModel):
    """Validated input for list_files."""
    directory: str = Field(..., description="Absolute or relative path to directory to list.")
    pattern: str = Field("*", description="Glob pattern to filter filenames, e.g. '*.py'.")


class ReadFileInput(BaseModel):
    """Validated input for read_file."""
    path: str = Field(..., description="Path to the file to read.")
    encoding: str = Field("utf-8", description="File encoding. Defaults to 'utf-8'.")


class WriteFileInput(BaseModel):
    """Validated input for write_file."""
    path: str = Field(..., description="Destination file path.")
    content: str = Field(..., description="Text content to write.")
    overwrite: bool = Field(False, description="If False, raise error if file already exists.")


# ── JSON schema dicts for the Anthropic API ────────────────────────────────────

list_files_tool: dict[str, Any] = {
    "name": "list_files",
    "description": (
        "List files in a directory, optionally filtered by a glob pattern. "
        "Returns a list of matching file paths. "
        "Use this to explore directory structure before reading or writing files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Absolute or relative path to directory to list.",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter filenames, e.g. '*.py'. Defaults to '*' (all files).",
                "default": "*",
            },
        },
        "required": ["directory"],
    },
}

read_file_tool: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read the full text content of a file. "
        "Returns the file content as a string. "
        "Use this after list_files to inspect specific files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "encoding": {
                "type": "string",
                "description": "File encoding. Defaults to 'utf-8'.",
                "default": "utf-8",
            },
        },
        "required": ["path"],
    },
}

write_file_tool: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Write text content to a file. "
        "Returns True on success. "
        "Set overwrite=True to replace an existing file; otherwise the call will fail if the file exists. "
        "Use this to create or update files with generated content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination file path.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "If False (default), fail when the file already exists.",
                "default": False,
            },
        },
        "required": ["path", "content"],
    },
}


# ── Stub implementations that validate via Pydantic ───────────────────────────

import fnmatch
import pathlib


def list_files(raw: dict[str, Any]) -> list[str]:
    inp = ListFilesInput(**raw)
    base = pathlib.Path(inp.directory)
    return [str(p) for p in base.iterdir() if fnmatch.fnmatch(p.name, inp.pattern)]


def read_file(raw: dict[str, Any]) -> str:
    inp = ReadFileInput(**raw)
    return pathlib.Path(inp.path).read_text(encoding=inp.encoding)


def write_file(raw: dict[str, Any]) -> bool:
    inp = WriteFileInput(**raw)
    p = pathlib.Path(inp.path)
    if p.exists() and not inp.overwrite:
        raise FileExistsError(f"{inp.path} already exists. Pass overwrite=True to replace.")
    p.write_text(inp.content, encoding="utf-8")
    return True


ALL_FILE_TOOLS = [list_files_tool, read_file_tool, write_file_tool]
```

---

## Exercise 3 — Good vs Bad Tool Descriptions for `send_email`

```python
from typing import Any

# ── Bad description ───────────────────────────────────────────────────────────

bad_send_email_tool: dict[str, Any] = {
    "name": "send_email",
    "description": "send email",          # <-- too vague
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}

# ── Improved description ──────────────────────────────────────────────────────

good_send_email_tool: dict[str, Any] = {
    "name": "send_email",
    "description": (
        "Send an email message to one or more recipients. "
        "Use this when the user explicitly asks to send, draft-and-send, or forward an email. "
        "Do NOT use this just to show a draft — only call it when the user confirms they want it sent. "
        "Returns a message ID on success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": (
                    "Recipient email address, or comma-separated list for multiple recipients. "
                    "Example: 'alice@example.com' or 'alice@example.com, bob@example.com'."
                ),
            },
            "subject": {
                "type": "string",
                "description": "Email subject line. Keep it concise and descriptive.",
            },
            "body": {
                "type": "string",
                "description": "Plain-text body of the email. Markdown is not rendered.",
            },
            "cc": {
                "type": "string",
                "description": "Optional CC recipients, comma-separated.",
            },
        },
        "required": ["to", "subject", "body"],
    },
}
```

### Why the improved description leads to better LLM behavior

| Aspect | Bad (`"send email"`) | Good (improved) |
|---|---|---|
| **When to call** | Ambiguous — model may call on any email mention | Explicit: only on confirmed send intent |
| **When NOT to call** | Not stated — model may call prematurely | States "do NOT call just to show a draft" |
| **Return value** | Not mentioned — model can't relay success to user | "Returns a message ID on success" |
| **Parameter guidance** | No descriptions — model guesses formats | Concrete examples for `to`, format notes for `body` |
| **Side-effect warning** | None — dangerous for irreversible actions | Implicit: "only call when user confirms" |

**Rule of thumb:** A good tool description answers three questions for the model:
1. *When should I call this?*
2. *When should I NOT call this?*
3. *What will I get back, and how should I use it?*

Poor descriptions cause models to call tools too eagerly, pass incorrectly-formatted inputs, or fail to communicate results properly.
