# Day 3 Solutions — Tool Schemas and Agentic Tool-Use Loop

## Exercise 1: Tool Schema Dict for `search_documents`

```python
# tools/schemas.py
"""Tool schema definitions for the Anthropic tool-use API."""

from __future__ import annotations

SEARCH_DOCUMENTS_SCHEMA: dict = {
    "name": "search_documents",
    "description": (
        "Search a document store using a natural language query. "
        "Returns the top matching document excerpts ranked by relevance. "
        "Use this when the user asks about specific topics, facts, or needs "
        "information that may exist in stored documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The natural language search query. Be specific and include "
                    "key terms relevant to the information you need."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    },
}

GET_DOCUMENT_SCHEMA: dict = {
    "name": "get_document",
    "description": (
        "Retrieve the full content of a document by its ID. "
        "Use after search_documents to get the complete text of a relevant result."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The unique identifier of the document to retrieve.",
            },
        },
        "required": ["document_id"],
    },
}

SUMMARIZE_TEXT_SCHEMA: dict = {
    "name": "summarize_text",
    "description": (
        "Summarize a block of text to a specified length. "
        "Use when document content is too long and you need a condensed version."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summarize.",
            },
            "max_sentences": {
                "type": "integer",
                "description": "Target summary length in sentences. Defaults to 3.",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["text"],
    },
}

ALL_TOOLS: list[dict] = [
    SEARCH_DOCUMENTS_SCHEMA,
    GET_DOCUMENT_SCHEMA,
    SUMMARIZE_TEXT_SCHEMA,
]
```

---

## Exercise 2: Full Tool-Use Agentic Loop

```python
# agent/tool_loop.py
"""Multi-tool agentic loop for Anthropic's tool-use API."""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from tools.schemas import ALL_TOOLS
from tools.validators import validate_tool_input

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Concrete tool implementations (stubs — replace with real logic)
# ---------------------------------------------------------------------------

def search_documents(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search document store. Returns list of {id, title, excerpt}."""
    logger.info("search_documents called: query=%r max_results=%d", query, max_results)
    # Real implementation would call a vector store or search API
    return [
        {"id": f"doc-{i}", "title": f"Result {i} for '{query}'", "excerpt": f"...excerpt {i}..."}
        for i in range(1, min(max_results, 3) + 1)
    ]


def get_document(document_id: str) -> dict[str, str]:
    """Retrieve full document by ID."""
    logger.info("get_document called: document_id=%r", document_id)
    return {"id": document_id, "title": "Example Document", "content": "Full content here..."}


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Summarize text to max_sentences sentences."""
    logger.info("summarize_text called: len(text)=%d max_sentences=%d", len(text), max_sentences)
    sentences = text.split(". ")
    return ". ".join(sentences[:max_sentences]) + ("." if sentences else "")


TOOL_REGISTRY: dict[str, Any] = {
    "search_documents": search_documents,
    "get_document": get_document,
    "summarize_text": summarize_text,
}


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
    """Validate and execute a tool by name.

    Args:
        tool_name: Name matching one of the registered tools.
        tool_input: Raw input dict from the LLM tool_use block.

    Returns:
        The tool's return value (will be serialized to string for the API).

    Raises:
        ValueError: If the tool name is unknown or input fails validation.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name!r}. Available: {list(TOOL_REGISTRY)}")

    # Pydantic validation before execution (Exercise 3)
    validated = validate_tool_input(tool_name, tool_input)
    logger.debug("Executing tool %r with validated input: %s", tool_name, validated)

    fn = TOOL_REGISTRY[tool_name]
    return fn(**validated)


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(
    user_message: str,
    system_prompt: str = "You are a helpful research assistant. Use the provided tools to answer questions accurately.",
    model: str = "claude-opus-5-5",
    max_tokens: int = 4096,
) -> str:
    """Run a multi-turn agentic loop until the model stops calling tools.

    The loop:
    1. Send messages to Claude with all tools available.
    2. If stop_reason == "tool_use", execute each requested tool.
    3. Append tool results and send back to Claude.
    4. Repeat until stop_reason == "end_turn" or max iterations reached.

    Args:
        user_message: The user's initial question or request.
        system_prompt: System-level instructions for the agent.
        model: Claude model ID to use.
        max_tokens: Maximum tokens per API call.

    Returns:
        The final text response from the model.

    Raises:
        RuntimeError: If max iterations are exceeded without a final answer.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message}
    ]

    for iteration in range(MAX_ITERATIONS):
        logger.info("Agent iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=ALL_TOOLS,
            messages=messages,
        )

        # Append the assistant's full response turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract the final text block
            for block in response.content:
                if hasattr(block, "text"):
                    logger.info("Agent finished after %d iterations", iteration + 1)
                    return block.text
            return ""

        if response.stop_reason != "tool_use":
            logger.warning("Unexpected stop_reason: %r", response.stop_reason)
            break

        # Process all tool_use blocks in this response turn
        tool_results: list[dict[str, Any]] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            logger.info("Tool requested: %s (id=%s)", block.name, block.id)
            try:
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })
                logger.debug("Tool %r returned: %r", block.name, str(result)[:200])
            except (ValueError, TypeError) as exc:
                logger.error("Tool %r failed: %s", block.name, exc)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {exc}",
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"Agent exceeded maximum iterations ({MAX_ITERATIONS}) without producing a final answer."
    )
```

---

## Exercise 3: Pydantic Validation for Tool Inputs

```python
# tools/validators.py
"""Pydantic v2 models for validating tool inputs before execution."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class SearchDocumentsInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("query must not be blank after stripping whitespace")
        return stripped


class GetDocumentInput(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=100)

    @field_validator("document_id")
    @classmethod
    def no_path_traversal(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError(f"document_id contains invalid characters: {v!r}")
        return v


class SummarizeTextInput(BaseModel):
    text: str = Field(..., min_length=1)
    max_sentences: int = Field(default=3, ge=1, le=10)


_VALIDATORS: dict[str, type[BaseModel]] = {
    "search_documents": SearchDocumentsInput,
    "get_document": GetDocumentInput,
    "summarize_text": SummarizeTextInput,
}


def validate_tool_input(tool_name: str, raw_input: dict[str, Any]) -> dict[str, Any]:
    """Validate tool input against its Pydantic schema and return the validated dict.

    Args:
        tool_name: The name of the tool being called.
        raw_input: The input dict provided by the LLM.

    Returns:
        Validated and normalized input dict (with defaults applied).

    Raises:
        ValueError: If tool_name is unknown or validation fails.
    """
    model_cls = _VALIDATORS.get(tool_name)
    if model_cls is None:
        raise ValueError(f"No validator defined for tool: {tool_name!r}")

    validated = model_cls.model_validate(raw_input)
    logger.debug("Validated %s input: %s", tool_name, validated.model_dump())
    return validated.model_dump()
```

---

## Exercise 4: Complete Test Suite for Tool-Use Loop

```python
# tests/test_tool_loop.py
"""Full test suite for the multi-tool agentic loop."""

from __future__ import annotations

from typing import Any
from unittest import mock

import anthropic
import pytest

from agent.tool_loop import execute_tool, run_agent
from tools.validators import validate_tool_input


# ---------------------------------------------------------------------------
# Mock builder helpers
# ---------------------------------------------------------------------------

def _text_block(text: str) -> mock.MagicMock:
    """Simulate a TextBlock from the API."""
    block = mock.MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(
    name: str, tool_id: str, tool_input: dict[str, Any]
) -> mock.MagicMock:
    """Simulate a ToolUseBlock from the API."""
    block = mock.MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = tool_id
    block.input = tool_input
    return block


def _response(
    stop_reason: str, content: list[mock.MagicMock]
) -> mock.MagicMock:
    """Simulate an anthropic.types.Message."""
    resp = mock.MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# Case 1: end_turn immediately — no tools used
# ---------------------------------------------------------------------------

class TestRunAgentNoTools:
    def test_returns_text_on_end_turn(self) -> None:
        """Agent returns text directly when LLM answers without calling tools."""
        final_text = "The answer is 42."

        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _response(
                "end_turn", [_text_block(final_text)]
            )

            result = run_agent("What is the answer?")

        assert result == final_text
        mock_client.messages.create.assert_called_once()

    def test_calls_api_with_correct_model(self) -> None:
        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _response(
                "end_turn", [_text_block("done")]
            )
            run_agent("hi", model="claude-opus-5-5")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-5-5"


# ---------------------------------------------------------------------------
# Case 2: one tool call then end_turn
# ---------------------------------------------------------------------------

class TestRunAgentSingleToolCall:
    def test_tool_called_and_result_sent_back(self) -> None:
        """Agent calls search_documents once, then gets a final answer."""
        tool_block = _tool_use_block(
            "search_documents", "tu_001", {"query": "climate change", "max_results": 3}
        )
        first_response = _response("tool_use", [tool_block])
        second_response = _response("end_turn", [_text_block("Here are the results...")])

        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = [first_response, second_response]

            result = run_agent("Search for climate change documents")

        assert result == "Here are the results..."
        assert mock_client.messages.create.call_count == 2

        # Verify tool result was appended in second call
        second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
        # Last user message should contain tool_result
        tool_result_turn = second_call_messages[-1]
        assert tool_result_turn["role"] == "user"
        assert any(
            item.get("type") == "tool_result" and item.get("tool_use_id") == "tu_001"
            for item in tool_result_turn["content"]
        )

    def test_tool_error_is_returned_as_error_result(self) -> None:
        """If a tool raises, the loop sends is_error=True result and continues."""
        tool_block = _tool_use_block(
            "get_document", "tu_002", {"document_id": "../etc/passwd"}  # path traversal
        )
        first_response = _response("tool_use", [tool_block])
        second_response = _response("end_turn", [_text_block("Could not retrieve.")])

        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = [first_response, second_response]

            result = run_agent("Get document ../etc/passwd")

        assert result == "Could not retrieve."
        # Check that the tool result had is_error=True
        second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_turn = second_call_messages[-1]["content"]
        error_result = next(
            item for item in tool_result_turn if item.get("tool_use_id") == "tu_002"
        )
        assert error_result.get("is_error") is True


# ---------------------------------------------------------------------------
# Case 3: multiple sequential tool calls then end_turn
# ---------------------------------------------------------------------------

class TestRunAgentMultipleToolCalls:
    def test_three_tool_calls_converge_to_answer(self) -> None:
        """Agent chains three tool calls before giving a final answer."""
        responses = [
            _response(
                "tool_use",
                [_tool_use_block("search_documents", "tu_1", {"query": "LLMs"})],
            ),
            _response(
                "tool_use",
                [_tool_use_block("get_document", "tu_2", {"document_id": "doc-1"})],
            ),
            _response(
                "tool_use",
                [_tool_use_block("summarize_text", "tu_3", {"text": "Long text here..."})],
            ),
            _response("end_turn", [_text_block("Final summary.")]),
        ]

        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = responses

            result = run_agent("Summarize LLM research")

        assert result == "Final summary."
        assert mock_client.messages.create.call_count == 4


# ---------------------------------------------------------------------------
# Case 4: max iterations exceeded
# ---------------------------------------------------------------------------

class TestRunAgentMaxIterations:
    def test_raises_runtime_error_on_max_iterations(self) -> None:
        """Loop raises RuntimeError when it never gets end_turn."""
        # Always returns tool_use — simulates an agent stuck in a loop
        infinite_tool = _response(
            "tool_use",
            [_tool_use_block("search_documents", "tu_inf", {"query": "loop"})],
        )

        with mock.patch("anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = infinite_tool

            with pytest.raises(RuntimeError, match="maximum iterations"):
                run_agent("Loop forever")


# ---------------------------------------------------------------------------
# Unit tests for validate_tool_input
# ---------------------------------------------------------------------------

class TestValidateToolInput:
    def test_search_documents_valid(self) -> None:
        result = validate_tool_input("search_documents", {"query": "AI safety"})
        assert result["query"] == "AI safety"
        assert result["max_results"] == 5  # default applied

    def test_search_documents_invalid_max_results(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            validate_tool_input("search_documents", {"query": "q", "max_results": 999})

    def test_get_document_blocks_path_traversal(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="invalid characters"):
            validate_tool_input("get_document", {"document_id": "../etc/passwd"})

    def test_unknown_tool_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="No validator defined"):
            validate_tool_input("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Unit tests for execute_tool
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_search_documents_returns_list(self) -> None:
        result = execute_tool("search_documents", {"query": "test", "max_results": 2})
        assert isinstance(result, list)

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool"):
            execute_tool("does_not_exist", {})

    def test_invalid_input_raises_before_execution(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            execute_tool("search_documents", {"query": "", "max_results": 5})
```
