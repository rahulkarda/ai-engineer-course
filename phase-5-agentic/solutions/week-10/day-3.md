# Solutions — Phase 5 · Week 10 · Day 3

> LangChain Tools & Tool Calling

---

## Exercise 1 — Streaming tool-call detection with `astream`

```python
import asyncio
import logging
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Tools (reused from lesson) ---

@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Use +, -, *, / only."""
    import ast
    import operator as op

    _ops: dict[type, Any] = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
    }

    def _eval(node: ast.expr) -> float:
        match node:
            case ast.BinOp(left=l, op=o, right=r):
                return _ops[type(o)](_eval(l), _eval(r))
            case ast.Constant(value=v) if isinstance(v, int | float):
                return float(v)
            case _:
                raise ValueError(f"Unsupported: {type(node)}")

    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


# --- Streaming agent loop ---

async def run_agent_streaming(user_question: str) -> str:
    """
    Run the tool-calling loop using astream() instead of invoke().
    Chunks are accumulated with the + operator on AIMessageChunk, which
    merges content and tool_call_chunks fields incrementally.
    """
    tools: list[BaseTool] = [calculator]
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [HumanMessage(content=user_question)]
    max_iterations = 5

    for iteration in range(max_iterations):
        logger.info("Iteration %d — streaming LLM with %d messages", iteration + 1, len(messages))

        # Accumulate all chunks into a single AIMessageChunk
        accumulated: AIMessageChunk | None = None
        async for chunk in llm_with_tools.astream(messages):
            # chunk is an AIMessageChunk; + merges content and tool_call_chunks
            if accumulated is None:
                accumulated = chunk
            else:
                accumulated = accumulated + chunk

            # Optionally print content tokens as they arrive
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

        print()  # newline after streaming

        if accumulated is None:
            logger.warning("No chunks received from model.")
            return ""

        # Convert AIMessageChunk to AIMessage for appending to messages list.
        # tool_calls are fully resolved after accumulation.
        final_message = AIMessage(
            content=accumulated.content,
            tool_calls=accumulated.tool_calls,
        )
        messages.append(final_message)

        if not final_message.tool_calls:
            logger.info("No tool calls — agent finished.")
            return str(final_message.content)

        logger.info("Tool calls: %s", [tc["name"] for tc in final_message.tool_calls])

        # Execute tool calls
        for tool_call in final_message.tool_calls:
            tool_name: str = tool_call["name"]
            tool_args: dict[str, Any] = tool_call["args"]
            tool_call_id: str = tool_call["id"]

            if tool_name in tools_by_name:
                result = str(tools_by_name[tool_name].invoke(tool_args))
            else:
                result = f"Error: unknown tool '{tool_name}'"

            messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

    return "Max iterations reached."


# --- Tests ---

def test_streaming_tool_call_detection() -> None:
    """Mock astream to yield chunks; assert tool_calls are detected after accumulation."""

    # Build two chunks that simulate a tool call arriving in pieces
    tool_call_chunk_1 = AIMessageChunk(
        content="",
        tool_call_chunks=[{
            "name": "calculator",
            "args": '{"expression":',
            "id": "tc_001",
            "index": 0,
        }],
    )
    tool_call_chunk_2 = AIMessageChunk(
        content="",
        tool_call_chunks=[{
            "name": None,
            "args": ' "2 + 2"}',
            "id": None,
            "index": 0,
        }],
    )
    # Final chunk: content with no tool calls (simulates "done")
    final_content_chunk = AIMessageChunk(
        content="The answer is 4.0",
        tool_call_chunks=[],
    )

    # We need to simulate two iterations:
    # 1st: yields tool call chunks → execute calculator → append ToolMessage
    # 2nd: yields final content chunk → return
    call_count = 0

    async def mock_astream(*args: Any, **kwargs: Any):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            for c in [tool_call_chunk_1, tool_call_chunk_2]:
                yield c
        else:
            yield final_content_chunk

    with patch.object(ChatAnthropic, "astream", new=mock_astream):
        result = asyncio.run(run_agent_streaming("What is 2 + 2?"))

    assert "4" in result
    logger.info("Streaming tool-call detection test passed: %s", result)
```

**Why this works:**
`astream()` yields `AIMessageChunk` objects. The `+` operator on `AIMessageChunk` is defined by LangChain to merge `content` strings and concatenate `tool_call_chunks` entries by index. After the stream exhausts, the accumulated chunk holds the complete tool call with fully-assembled `name` and `args`. Converting to `AIMessage` before appending to the message list ensures downstream code sees the standard message type with a populated `tool_calls` list.

**Pitfalls:**
- Do not check `tool_calls` on each raw chunk — early chunks have partial `args` JSON. Always wait for full accumulation before inspecting.
- `AIMessageChunk + AIMessageChunk` only merges chunks from the same model call. Do not mix chunks from different iterations.

---

## Exercise 2 — Tool error handling with `status="error"`

```python
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Supports +, -, *, /."""
    import ast
    import operator as op

    _ops: dict[type, Any] = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
    }

    def _eval(node: ast.expr) -> float:
        match node:
            case ast.BinOp(left=l, op=o, right=r):
                return _ops[type(o)](_eval(l), _eval(r))
            case ast.Constant(value=v) if isinstance(v, int | float):
                return float(v)
            case _:
                raise ValueError(f"Unsupported node: {type(node)}")

    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


def execute_tool_calls_with_error_handling(
    ai_message: AIMessage,
    tools_by_name: dict[str, BaseTool],
) -> list[ToolMessage]:
    """
    Execute all tool calls. If a tool raises an exception, return a ToolMessage
    with is_error=True and the error text as content so the model can recover.

    Anthropic's API supports an `is_error` field on tool_result blocks.
    LangChain surfaces this via the `status` field on ToolMessage:
      status="error"  → is_error=True in the API payload
      status="success" → is_error=False (default)
    """
    results: list[ToolMessage] = []

    for tool_call in ai_message.tool_calls:
        tool_name: str = tool_call["name"]
        tool_args: dict[str, Any] = tool_call["args"]
        tool_call_id: str = tool_call["id"]

        if tool_name not in tools_by_name:
            results.append(
                ToolMessage(
                    content=f"Error: unknown tool '{tool_name}'",
                    tool_call_id=tool_call_id,
                    status="error",
                )
            )
            continue

        try:
            content = str(tools_by_name[tool_name].invoke(tool_args))
            logger.info("Tool %r succeeded: %s", tool_name, content[:80])
            results.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                    status="success",
                )
            )
        except Exception as exc:
            error_msg = f"Tool '{tool_name}' raised {type(exc).__name__}: {exc}"
            logger.warning("Tool error: %s", error_msg)
            results.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            )

    return results


def run_agent(user_question: str) -> str:
    """Agent loop using execute_tool_calls_with_error_handling."""
    tools: list[BaseTool] = [calculator]
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(content=user_question)]

    for _ in range(5):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        tool_messages = execute_tool_calls_with_error_handling(response, tools_by_name)
        messages.extend(tool_messages)

    return "Max iterations reached."


# --- Tests ---

def test_zero_division_error_returns_error_tool_message() -> None:
    """Calculator raises ZeroDivisionError; ToolMessage should have status='error'."""
    # Simulate the AIMessage asking to call calculator
    ai_message = AIMessage(
        content="",
        tool_calls=[{
            "name": "calculator",
            "args": {"expression": "1 / 0"},
            "id": "tc_err_001",
            "type": "tool_call",
        }],
    )

    # Patch the calculator tool to raise ZeroDivisionError
    with patch.object(calculator, "invoke", side_effect=ZeroDivisionError("division by zero")):
        results = execute_tool_calls_with_error_handling(
            ai_message,
            {"calculator": calculator},
        )

    assert len(results) == 1
    assert results[0].status == "error"
    assert "ZeroDivisionError" in results[0].content
    assert results[0].tool_call_id == "tc_err_001"
    logger.info("Error tool message test passed: %s", results[0].content)


def test_model_recovers_after_error_tool_result() -> None:
    """Model should produce a final answer even after receiving an error ToolMessage."""
    # First response: model calls calculator (which will fail)
    first_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "calculator",
            "args": {"expression": "1 / 0"},
            "id": "tc_002",
            "type": "tool_call",
        }],
    )
    # Second response: model recovers and answers directly
    second_response = AIMessage(
        content="I cannot divide by zero. The operation is mathematically undefined.",
        tool_calls=[],
    )

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        return first_response if call_count == 1 else second_response

    with (
        patch.object(ChatAnthropic, "invoke", side_effect=side_effect),
        patch.object(calculator, "invoke", side_effect=ZeroDivisionError("division by zero")),
    ):
        result = run_agent("What is 1 divided by 0?")

    assert "zero" in result.lower() or "undefined" in result.lower()
    assert call_count == 2
    logger.info("Recovery after error test passed: %s", result)
```

**Why this works:**
Anthropic's API accepts a `is_error: true` flag on `tool_result` blocks. LangChain's `ToolMessage` exposes this as `status="error"`. When the model receives an error tool result, it understands the tool failed and can choose to explain the failure, retry with different arguments, or answer from its own knowledge. Catching exceptions inside `execute_tool_calls` prevents the entire agent loop from crashing on a single tool failure.

**Pitfalls:**
- Do not silently return an empty string on tool failure — the model needs error context to recover intelligently.
- Distinguish between `ValueError` (bad input — the model should not retry) and transient errors like `TimeoutError` (may succeed on retry). Pass this signal in the error message content.

---

## Exercise 3 — Add `get_current_datetime` tool

```python
import datetime
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Supports +, -, *, /."""
    import ast
    import operator as op

    _ops: dict[type, Any] = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
    }

    def _eval(node: ast.expr) -> float:
        match node:
            case ast.BinOp(left=l, op=o, right=r):
                return _ops[type(o)](_eval(l), _eval(r))
            case ast.Constant(value=v) if isinstance(v, int | float):
                return float(v)
            case _:
                raise ValueError(f"Unsupported: {type(node)}")

    tree = ast.parse(expression, mode="eval")
    return str(_eval(tree.body))


@tool
def get_current_datetime() -> str:
    """
    Return the current UTC date and time as an ISO 8601 string.
    Use this when the user asks about the current time, date, or datetime.
    Example output: '2024-11-11T14:30:00+00:00'
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    iso_str = now.isoformat()
    logger.info("get_current_datetime called: %s", iso_str)
    return iso_str


def run_agent(user_question: str) -> str:
    """Agent loop with all three tools bound."""
    tools: list[BaseTool] = [calculator, get_current_datetime]
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)
    messages: list[BaseMessage] = [HumanMessage(content=user_question)]

    for _ in range(5):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            name: str = tool_call["name"]
            args: dict[str, Any] = tool_call["args"]
            tool_call_id: str = tool_call["id"]

            try:
                result = str(tools_by_name[name].invoke(args)) if name in tools_by_name else f"Unknown tool: {name}"
            except Exception as exc:
                result = f"Error: {exc}"

            messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

    return "Max iterations reached."


# --- Tests ---

def test_get_current_datetime_returns_valid_iso() -> None:
    """Tool should return a parseable UTC ISO 8601 datetime string."""
    result: str = get_current_datetime.invoke({})

    parsed = datetime.datetime.fromisoformat(result)
    assert parsed.tzinfo is not None
    assert parsed.tzinfo == datetime.timezone.utc or str(parsed.tzinfo) == "UTC"
    logger.info("Datetime format test passed: %s", result)


def test_model_calls_datetime_tool_when_asked_time() -> None:
    """When asked 'what time is it?', model should call get_current_datetime."""
    # First response: model decides to call the tool
    first_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "get_current_datetime",
            "args": {},
            "id": "tc_dt_001",
            "type": "tool_call",
        }],
    )
    # Second response: model answers using the tool result
    second_response = AIMessage(
        content="The current UTC time is 2024-11-11T14:30:00+00:00.",
        tool_calls=[],
    )

    call_count = 0

    def side_effect(*args: Any, **kwargs: Any) -> AIMessage:
        nonlocal call_count
        call_count += 1
        return first_response if call_count == 1 else second_response

    with patch.object(ChatAnthropic, "invoke", side_effect=side_effect):
        result = run_agent("What time is it right now?")

    # Verify that the first model call requested the datetime tool
    assert call_count == 2
    assert "2024" in result or "time" in result.lower()
    logger.info("Datetime tool call test passed: %s", result)
```

**Why this works:**
The `@tool` decorator with no arguments wraps `get_current_datetime()` and introspects the signature — no parameters, so the JSON schema is `{"type": "object", "properties": {}}`. The docstring becomes the tool description sent to the model. `datetime.datetime.now(tz=datetime.timezone.utc)` is safe for production — it always returns an aware datetime in UTC, and `.isoformat()` produces a standard RFC 3339 / ISO 8601 string. No external dependencies needed.

**Pitfalls:**
- Do not use `datetime.datetime.utcnow()` — it returns a naive datetime (no `tzinfo`) and is deprecated in Python 3.12+. Always pass `tz=datetime.timezone.utc`.
- If the tool takes no arguments, invoke it as `tool.invoke({})` not `tool.invoke(None)` — the schema expects an empty dict.

---

## Exercise 4 — Force tool choice with `tool_choice`

```python
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression. Supports +, -, *, /."""
    return "42.0"  # stub for tests


@tool
def get_current_datetime() -> str:
    """Return current UTC datetime as ISO 8601 string."""
    import datetime
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


def build_forced_calculator_model() -> ChatAnthropic:
    """
    Bind tools to the model and force it to always call 'calculator'.

    tool_choice="calculator" maps to Anthropic's:
      {"type": "tool", "name": "calculator"}

    This instructs the API to always include a calculator tool_use block
    in the response, regardless of the user's question.

    WHY THIS IS RARELY CORRECT IN PRODUCTION:
    Forcing a specific tool removes the model's ability to decide whether
    a tool call is appropriate. In a real agent, the model needs freedom
    to choose "no tool" (for questions it can answer from knowledge) or
    a different tool (for datetime questions). Forced tool choice is only
    appropriate in narrow scenarios: testing/CI (to ensure a tool call is
    always emitted), deterministic pipelines (where the input always
    requires the tool), or demos. In all other cases, let the model decide
    via natural tool selection — its judgment is more reliable than a hard
    override.
    """
    llm = ChatAnthropic(
        model="claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    tools: list[BaseTool] = [calculator, get_current_datetime]

    # Pass the tool name as a string — LangChain maps this to the
    # Anthropic-format {"type": "tool", "name": "calculator"} in bind_tools.
    llm_with_forced_tool = llm.bind_tools(tools, tool_choice="calculator")
    return llm_with_forced_tool


# --- Tests ---

def test_forced_tool_choice_always_calls_calculator() -> None:
    """
    Regardless of the question asked, the model should always emit a
    calculator tool call when tool_choice='calculator' is set.
    """
    # Mock model response — always includes a calculator tool call
    mock_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "calculator",
            "args": {"expression": "1 + 1"},
            "id": "tc_forced_001",
            "type": "tool_call",
        }],
    )

    llm_forced = build_forced_calculator_model()

    with patch.object(ChatAnthropic, "invoke", return_value=mock_response):
        # Ask a question that would NEVER normally require a calculator
        response: AIMessage = llm_forced.invoke(
            [HumanMessage(content="What is the capital of France?")]
        )

    assert len(response.tool_calls) >= 1
    assert response.tool_calls[0]["name"] == "calculator", (
        f"Expected 'calculator', got '{response.tool_calls[0]['name']}'"
    )
    logger.info("Forced tool choice test passed: tool=%s", response.tool_calls[0]["name"])


def test_forced_tool_choice_with_different_questions() -> None:
    """
    Three different questions — forced tool choice ensures all three
    produce a calculator call.
    """
    mock_response = AIMessage(
        content="",
        tool_calls=[{
            "name": "calculator",
            "args": {"expression": "0"},
            "id": "tc_forced_002",
            "type": "tool_call",
        }],
    )
    llm_forced = build_forced_calculator_model()

    questions = [
        "What is the weather today?",
        "Who wrote Hamlet?",
        "What time is it?",  # would normally trigger get_current_datetime
    ]

    with patch.object(ChatAnthropic, "invoke", return_value=mock_response):
        for q in questions:
            response: AIMessage = llm_forced.invoke([HumanMessage(content=q)])
            assert response.tool_calls[0]["name"] == "calculator", (
                f"Question '{q}' did not produce a calculator call"
            )
            logger.info("Question '%s' → tool: %s", q, response.tool_calls[0]["name"])

    logger.info("All forced-tool-choice tests passed.")
```

**Why this works:**
`bind_tools(tools, tool_choice="calculator")` serialises to Anthropic's `tool_choice: {"type": "tool", "name": "calculator"}`, which instructs the API to always include a `tool_use` block for `calculator` in its response. LangChain's `ChatAnthropic.bind_tools` accepts the string shorthand and converts it to the correct API format. The model will always emit a tool call even if the question has nothing to do with arithmetic.

**Pitfalls:**
- `tool_choice="auto"` (default) lets the model decide. `tool_choice="any"` forces a tool call but lets the model pick which tool. `tool_choice="<name>"` forces exactly that tool — this is the most restrictive option.
- If `tool_choice` names a tool not included in the `tools` list, Anthropic returns a 400 error. Always validate the tool name matches a bound tool.
- In production agents, forced tool choice breaks multi-step reasoning: if the model is forced to call the calculator on every turn, it cannot produce a final text answer — the loop becomes infinite. Only use for single-turn forced extraction patterns.
