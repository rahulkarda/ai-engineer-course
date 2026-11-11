# Solutions — Phase 5 · Week 2 · Day 5 — Parallel Tool Calls + Tool Design Best Practices

---

## Exercise 1 — `process_parallel_tool_calls`

Process all `tool_use` blocks in a single response and return a list of `tool_result` dicts.

```python
import json
import logging
import traceback
from typing import Any, Callable

logger = logging.getLogger(__name__)


def build_tool_result_block(
    tool_use_id: str,
    content: str,
    is_error: bool = False,
) -> dict[str, Any]:
    """Build a single tool_result content block."""
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def process_parallel_tool_calls(
    response_content: list[dict[str, Any]],
    tool_fns: dict[str, Callable[..., Any]],
) -> list[dict[str, Any]]:
    """
    Process ALL tool_use blocks in a model response, return list of tool_result dicts.

    The Anthropic API allows the model to emit multiple tool_use blocks in a single
    response (parallel tool calls). Each must be dispatched and ALL results must be
    returned in the *same* user message on the next turn — the API rejects partial replies.

    Args:
        response_content: The list of content blocks from the assistant response.
                          May contain a mix of text and tool_use blocks.
        tool_fns:         Map of tool_name -> callable. Each callable accepts
                          keyword arguments matching the tool's input schema.

    Returns:
        A list of tool_result content block dicts, one per tool_use block.
        Non-tool_use blocks (e.g. text) are skipped.
    """
    tool_results: list[dict[str, Any]] = []

    for block in response_content:
        # Handle both raw dict (from tests) and SDK objects (real responses)
        if isinstance(block, dict):
            block_type = block.get("type")
            block_id = block.get("id", "")
            block_name = block.get("name", "")
            block_input = block.get("input", {})
        else:
            block_type = getattr(block, "type", None)
            block_id = getattr(block, "id", "")
            block_name = getattr(block, "name", "")
            block_input = getattr(block, "input", {})

        if block_type != "tool_use":
            continue  # skip text blocks and other content types

        logger.info("Dispatching tool: %s (id=%s) input=%s", block_name, block_id, block_input)

        if block_name not in tool_fns:
            error_msg = f"Unknown tool: '{block_name}'. Available: {list(tool_fns.keys())}"
            logger.warning(error_msg)
            tool_results.append(build_tool_result_block(block_id, error_msg, is_error=True))
            continue

        try:
            result = tool_fns[block_name](**block_input)
            content = result if isinstance(result, str) else json.dumps(result)
            logger.info("Tool '%s' succeeded: %s", block_name, content[:200])
            tool_results.append(build_tool_result_block(block_id, content))

        except Exception as exc:
            error_msg = f"Tool '{block_name}' raised {type(exc).__name__}: {exc}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            tool_results.append(build_tool_result_block(block_id, error_msg, is_error=True))

    logger.info("Processed %d parallel tool call(s)", len(tool_results))
    return tool_results


def build_parallel_results_message(
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Wrap multiple tool_result blocks into the single user message the API requires.

    All tool_results from a parallel batch MUST go into the same message.
    Sending them as separate messages causes an API error.
    """
    return {"role": "user", "content": tool_results}


# ── Example stub tool implementations ─────────────────────────────────────────

def search_documents(query: str) -> dict[str, Any]:
    return {"query": query, "results": [{"id": "doc1", "title": f"Result for '{query}'"}]}


def calculate(expression: str) -> dict[str, Any]:
    import re
    if not re.match(r"^[\d\s+\-*/().]+$", expression):
        raise ValueError(f"Disallowed characters: {expression!r}")
    return {"expression": expression, "result": eval(expression)}  # noqa: S307


def lookup_user(user_id: str) -> dict[str, Any]:
    return {"user_id": user_id, "name": "Alice Example", "email": "alice@example.com"}


TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "search_documents": search_documents,
    "calculate": calculate,
    "lookup_user": lookup_user,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate a response with 3 parallel tool calls
    parallel_response_content: list[dict[str, Any]] = [
        {"type": "tool_use", "id": "toolu_01", "name": "search_documents", "input": {"query": "Pydantic v2"}},
        {"type": "tool_use", "id": "toolu_02", "name": "calculate", "input": {"expression": "42 * 100"}},
        {"type": "tool_use", "id": "toolu_03", "name": "lookup_user", "input": {"user_id": "usr_abc"}},
    ]

    results = process_parallel_tool_calls(parallel_response_content, TOOL_REGISTRY)
    msg = build_parallel_results_message(results)
    print(json.dumps(msg, indent=2))
```

---

## Exercise 2 — Pytest: 3 Parallel Tool Calls Dispatched and Returned

```python
import json
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest


# ── Helper: build a mock tool_use block ──────────────────────────────────────

def _mock_tool_use(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return a tool_use block as a plain dict (matches what process_parallel_tool_calls expects)."""
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


# ── Stub tool functions ───────────────────────────────────────────────────────

def _stub_search(query: str) -> dict[str, Any]:
    return {"results": [f"Result for '{query}'"]}


def _stub_calculate(expression: str) -> dict[str, Any]:
    import re
    if not re.match(r"^[\d\s+\-*/().]+$", expression):
        raise ValueError("Disallowed characters")
    return {"result": eval(expression)}  # noqa: S307


def _stub_lookup(user_id: str) -> dict[str, Any]:
    return {"user_id": user_id, "name": "Test User"}


STUB_TOOLS: dict[str, Callable[..., Any]] = {
    "search_documents": _stub_search,
    "calculate": _stub_calculate,
    "lookup_user": _stub_lookup,
}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestProcessParallelToolCalls:

    def test_all_three_results_returned(self) -> None:
        """process_parallel_tool_calls returns exactly 3 tool_result blocks."""
        response_content = [
            _mock_tool_use("toolu_01", "search_documents", {"query": "AI agents"}),
            _mock_tool_use("toolu_02", "calculate", {"expression": "10 * 5"}),
            _mock_tool_use("toolu_03", "lookup_user", {"user_id": "u_42"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)

        assert len(results) == 3

    def test_tool_use_ids_match(self) -> None:
        """Each tool_result block has the correct tool_use_id."""
        response_content = [
            _mock_tool_use("toolu_01", "search_documents", {"query": "AI agents"}),
            _mock_tool_use("toolu_02", "calculate", {"expression": "10 * 5"}),
            _mock_tool_use("toolu_03", "lookup_user", {"user_id": "u_42"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)
        result_ids = {r["tool_use_id"] for r in results}

        assert result_ids == {"toolu_01", "toolu_02", "toolu_03"}

    def test_all_results_are_tool_result_type(self) -> None:
        """Every returned block has type='tool_result'."""
        response_content = [
            _mock_tool_use("toolu_01", "search_documents", {"query": "test"}),
            _mock_tool_use("toolu_02", "calculate", {"expression": "1 + 1"}),
            _mock_tool_use("toolu_03", "lookup_user", {"user_id": "u_1"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)

        for r in results:
            assert r["type"] == "tool_result"

    def test_successful_results_have_no_is_error(self) -> None:
        """Successful tool calls do not have is_error=True."""
        response_content = [
            _mock_tool_use("toolu_01", "search_documents", {"query": "test"}),
            _mock_tool_use("toolu_02", "calculate", {"expression": "2 + 2"}),
            _mock_tool_use("toolu_03", "lookup_user", {"user_id": "u_1"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)

        for r in results:
            assert r.get("is_error", False) is False

    def test_results_packaged_in_single_user_message(self) -> None:
        """build_parallel_results_message wraps all results in one user message."""
        response_content = [
            _mock_tool_use("toolu_01", "search_documents", {"query": "test"}),
            _mock_tool_use("toolu_02", "calculate", {"expression": "3 * 3"}),
            _mock_tool_use("toolu_03", "lookup_user", {"user_id": "u_1"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)
        user_message = build_parallel_results_message(results)

        assert user_message["role"] == "user"
        assert len(user_message["content"]) == 3

    def test_unknown_tool_returns_is_error_true(self) -> None:
        """Unknown tool name results in is_error=True for that block only."""
        response_content = [
            _mock_tool_use("toolu_01", "search_documents", {"query": "test"}),
            _mock_tool_use("toolu_bad", "nonexistent_tool", {}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)

        assert len(results) == 2
        error_results = [r for r in results if r.get("is_error")]
        assert len(error_results) == 1
        assert error_results[0]["tool_use_id"] == "toolu_bad"

    def test_text_blocks_in_response_are_skipped(self) -> None:
        """Text blocks alongside tool_use blocks are ignored (not included in results)."""
        response_content: list[dict[str, Any]] = [
            {"type": "text", "text": "I'll look that up for you."},
            _mock_tool_use("toolu_01", "search_documents", {"query": "test"}),
            {"type": "text", "text": "And also calculate:"},
            _mock_tool_use("toolu_02", "calculate", {"expression": "5 + 5"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)

        # Only tool_use blocks become tool_results
        assert len(results) == 2

    def test_tool_content_is_json_serializable_string(self) -> None:
        """Tool result content is a JSON string (for dict results)."""
        response_content = [
            _mock_tool_use("toolu_01", "lookup_user", {"user_id": "u_test"}),
        ]

        results = process_parallel_tool_calls(response_content, STUB_TOOLS)

        content = results[0]["content"]
        # Should be a JSON string, parseable back to dict
        parsed = json.loads(content)
        assert "user_id" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Exercise 3 — Tool Design Critique: `do_everything` vs Specific Tools

### Option A — One God Tool (Bad Design)

```python
from typing import Any

# Anti-pattern: single tool with a string "action" selector
do_everything_tool: dict[str, Any] = {
    "name": "do_everything",
    "description": "Perform document operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: 'search', 'create', 'delete'",
            },
            "args": {
                "type": "object",
                "description": "Arguments for the chosen action.",
            },
        },
        "required": ["action"],
    },
}
```

### Option B — Specific Tools (Good Design)

```python
search_documents_tool: dict[str, Any] = {
    "name": "search_documents",
    "description": (
        "Search the document store for content matching a query. "
        "Returns a list of matching documents with IDs and titles. "
        "Use before create_document to check if the document already exists."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Full-text search query string.",
            },
        },
        "required": ["query"],
    },
}

create_document_tool: dict[str, Any] = {
    "name": "create_document",
    "description": (
        "Create a new document in the document store. "
        "Returns the new document's ID. "
        "Do NOT call this if search_documents already found an existing document with the same title."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Document title. Must be unique.",
            },
            "content": {
                "type": "string",
                "description": "Full text body of the document.",
            },
        },
        "required": ["title", "content"],
    },
}

delete_document_tool: dict[str, Any] = {
    "name": "delete_document",
    "description": (
        "Permanently delete a document by its ID. This action is irreversible. "
        "Always confirm the document ID with search_documents before calling this. "
        "Returns True on success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "The unique document ID returned by search_documents or create_document.",
            },
        },
        "required": ["doc_id"],
    },
}
```

### Critique and Analysis

| Criterion | `do_everything(action, *args)` | Separate specific tools |
|---|---|---|
| **Schema clarity** | Model must remember valid `action` strings with no schema enforcement | Each tool has a clear, typed schema — the API validates inputs |
| **Parallel calls** | Cannot parallelize — single tool, one call | Model can call `search_documents` + `lookup_user` simultaneously |
| **Input validation** | `args` is an untyped `object` — Pydantic can't validate per-action | Each tool's `input_schema` is fully typed and validates at call time |
| **Error isolation** | All errors look the same; hard to tell which action failed | Error messages name the specific tool and its inputs |
| **Permission / security** | All-or-nothing grant: giving the tool = giving all actions | Fine-grained: you can give `search_documents` without `delete_document` |
| **Model reasoning** | Model must hold action names in implicit memory; no schema hints | Descriptions guide when to use each tool, including safety guardrails |
| **Code maintainability** | One giant dispatch function with if/elif branches | One function per tool — easy to test, extend, or remove |
| **Parallel tool calls** | Impossible: one tool handles all operations sequentially | Multiple specific tools can be called in parallel in one response |

**Verdict: Option B is significantly better.**

The god-tool pattern (Option A) is the tool equivalent of a "do everything" API endpoint — it collapses all operations into one schema-less interface. This prevents the model from using parallel calls, prevents fine-grained validation, and removes the per-operation guidance that helps the model decide when to call what.

Specific tools (Option B) follow the principle of **one tool, one responsibility**. Each tool has:
1. A clear trigger condition ("when to use me")
2. A typed schema the API can validate
3. Safety guardrails built into the description ("irreversible — confirm first")
4. Independence, enabling parallel execution when multiple operations are needed

**Design rule:** If your tool has an `action: str` parameter that selects behavior, split it into N tools.
