# Phase 5 · Week 10 · Day 3 — LangChain Tools & Tool Calling

> Objective: Define LangChain tools with typed schemas, bind them to Claude, parse tool call responses, and build a complete LCEL tool-execution loop.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

---

## Concept

**Tool calling** (also called function calling) lets an LLM declare that it wants to invoke a specific function with specific arguments instead of generating free text. The model does not execute the function — it emits a structured `ToolCall` object in its response. Your code executes the tool and feeds the result back into the next LLM call.

LangChain provides:
- Decorators and base classes to define tools with Pydantic-validated schemas.
- `bind_tools()` to attach tool definitions to a chat model.
- `AIMessage.tool_calls` to inspect which tools the model requested.
- `ToolMessage` to return results back to the model.

This builds the foundation for agents: a loop of `[LLM → tool calls → execute → LLM → ...]` until the model returns a final answer with no tool calls.

---

## How It Works Internally

### Defining Tools

**Method 1: `@tool` decorator**

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b
```

LangChain introspects the docstring (tool description), function signature (argument names and types), and return annotation. It generates a JSON schema automatically.

**Method 2: `BaseTool` subclass**

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="A mathematical expression to evaluate, e.g. '2 + 2 * 3'")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate a safe mathematical expression. Input must be a valid Python arithmetic expression."
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        # Implementation in worked example
        ...

    async def _arun(self, expression: str) -> str:
        return self._run(expression)
```

Use `BaseTool` when you need:
- Custom `name` separate from the function name.
- Injected dependencies (e.g., a database client stored as a field).
- Shared state between calls.

**Tool metadata:**
- `name` — identifier the LLM uses to reference the tool (snake_case, no spaces).
- `description` — natural language description; quality here directly affects whether the LLM chooses the tool correctly.
- `args_schema` — Pydantic model; each field's `description` becomes the parameter description sent to the LLM.

### `bind_tools()`

```python
llm_with_tools = llm.bind_tools([web_search, calculator])
```

`bind_tools()` serialises tool schemas to the format the underlying API expects (Anthropic `tools` array) and attaches them to the model's default config. The return type is still `ChatAnthropic` — a `Runnable[LanguageModelInput, AIMessage]`.

Under the hood, `ChatAnthropic.bind_tools()` converts each `BaseTool` to an Anthropic-format tool definition:

```json
{
  "name": "calculator",
  "description": "...",
  "input_schema": { "type": "object", "properties": { ... } }
}
```

### Tool Call Flow

1. `llm_with_tools.invoke(messages)` → returns `AIMessage`.
2. Check `AIMessage.tool_calls: list[ToolCall]`. Each `ToolCall` has `id`, `name`, `args` (already-parsed dict).
3. Execute the tool: `tool.invoke(tool_call["args"])` → returns a string (or auto-coerced to string).
4. Wrap result in `ToolMessage(content=result, tool_call_id=tool_call["id"])`.
5. Append both the `AIMessage` and the `ToolMessage` to the message list.
6. Re-invoke the LLM with the updated list. Repeat until `AIMessage.tool_calls` is empty.

### AgentExecutor vs Raw LCEL Tool Loops

| | AgentExecutor | Raw LCEL loop |
|---|---|---|
| Setup | High-level: pass agent + tools | Manual: write the loop |
| Control | Opaque; hard to customise mid-loop | Full control over every step |
| Streaming | Partial support | Full streaming per chunk |
| Debugging | LangSmith traces + verbose flag | Explicit logging |
| When to use | Rapid prototyping | Production agents, custom logic |

For production agents, prefer LangGraph (covered in later weeks) over either. `AgentExecutor` is maintained for compatibility but considered legacy. Raw LCEL loops are fine for simple cases; LangGraph is better for multi-step stateful agents.

---

## Worked Example

Define two tools, bind them to `claude-haiku-5-5`, parse tool calls, execute tools, and return the final answer.

**Dependencies (add via uv):**
```
uv add langchain langchain-anthropic
```

**`tool_calling.py`**

```python
import ast
import logging
import operator
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Tool 1: Web search stub (no real API key needed for this exercise) ---

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web")
    max_results: int = Field(default=3, description="Maximum number of results to return (1-10)", ge=1, le=10)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for current information. Use this when you need facts, "
        "recent events, or data you are not confident about."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 3) -> str:
        # Stub: in production, call a real search API (e.g. Tavily, SerpAPI)
        logger.info("web_search called: query=%r max_results=%d", query, max_results)
        return (
            f"[STUB] Search results for '{query}':\n"
            f"1. Result one about {query}\n"
            f"2. Result two about {query}\n"
            f"3. Result three about {query}\n"
        )[:max_results * 60]

    async def _arun(self, query: str, max_results: int = 3) -> str:
        return self._run(query, max_results)


# --- Tool 2: Safe calculator ---

_SAFE_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """Evaluate a simple arithmetic expression without using eval()."""
    tree = ast.parse(expr, mode="eval")

    def _visit(node: ast.expr) -> float:
        match node:
            case ast.BinOp(left=l, op=op, right=r):
                fn = _SAFE_OPS.get(type(op))
                if fn is None:
                    raise ValueError(f"Unsupported operator: {op}")
                return fn(_visit(l), _visit(r))
            case ast.UnaryOp(op=op, operand=operand):
                fn = _SAFE_OPS.get(type(op))
                if fn is None:
                    raise ValueError(f"Unsupported operator: {op}")
                return fn(_visit(operand))
            case ast.Constant(value=v) if isinstance(v, int | float):
                return float(v)
            case _:
                raise ValueError(f"Unsupported node type: {type(node)}")

    return _visit(tree.body)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a safe mathematical expression.
    Supports +, -, *, /, ** operators and numeric literals only.
    Example: '2 + 3 * 4' returns '14.0'.
    Do NOT pass variables or function calls.
    """
    logger.info("calculator called: expression=%r", expression)
    try:
        result = _safe_eval(expression)
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


# --- Tool executor ---

def execute_tool_calls(
    ai_message: AIMessage,
    tools_by_name: dict[str, BaseTool],
) -> list[ToolMessage]:
    """Execute all tool calls in an AIMessage and return ToolMessages."""
    results: list[ToolMessage] = []
    for tool_call in ai_message.tool_calls:
        tool_name: str = tool_call["name"]
        tool_args: dict[str, Any] = tool_call["args"]
        tool_call_id: str = tool_call["id"]

        if tool_name not in tools_by_name:
            content = f"Error: unknown tool '{tool_name}'"
        else:
            try:
                content = str(tools_by_name[tool_name].invoke(tool_args))
            except Exception as exc:
                content = f"Error executing {tool_name}: {exc}"
            logger.info("Tool %r returned: %s", tool_name, content[:120])

        results.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    return results


# --- Main agent loop ---

def run_agent(user_question: str) -> str:
    """Run a simple tool-calling loop until the model returns a final answer."""
    tools: list[BaseTool] = [WebSearchTool(), calculator]
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
        logger.info("Iteration %d — invoking LLM with %d messages", iteration + 1, len(messages))
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            logger.info("No tool calls — agent finished.")
            return str(response.content)

        logger.info("Tool calls requested: %s", [tc["name"] for tc in response.tool_calls])
        tool_messages = execute_tool_calls(response, tools_by_name)
        messages.extend(tool_messages)

    logger.warning("Max iterations (%d) reached without final answer.", max_iterations)
    return "Agent did not produce a final answer within the iteration limit."


def main() -> None:
    questions: list[str] = [
        "What is 17 * 23 + 144?",
        "Search for recent news about LangChain and then calculate 42 ** 2.",
    ]
    for q in questions:
        logger.info("Question: %s", q)
        answer = run_agent(q)
        logger.info("Answer: %s\n", answer)


if __name__ == "__main__":
    main()
```

**Key points:**
- `WebSearchTool` uses `BaseTool` subclass — allows injecting a real API client as a field later.
- `calculator` uses `@tool` decorator — simple, annotation-driven, docstring becomes description.
- `_safe_eval` uses `ast.parse` + structural pattern matching instead of `eval()` — avoids the security flag.
- `execute_tool_calls` is a pure function: takes an `AIMessage` and a tool registry, returns `list[ToolMessage]`.
- The agent loop appends both `AIMessage` and `ToolMessage` to `messages` before re-invoking — required by the Anthropic API.
- Loop is bounded by `max_iterations` to prevent infinite cycles.
- `bind_tools()` is called once; the result is reused across all iterations.

---

## Exercises

_(Solutions: ../solutions/week-10/day-3.md)_

1. **Streaming tool-call detection.** Rewrite `run_agent` to use `llm_with_tools.astream()`. Accumulate chunks into a final `AIMessage` using `AIMessageChunk` concatenation (use the `+` operator on chunks). Confirm tool calls are still detected correctly after accumulation. Write a unit test with a mocked stream.

2. **Tool error handling.** Modify `execute_tool_calls` to return a `ToolMessage` with `status="error"` when the tool raises an exception (Anthropic supports a `is_error` field on tool results). Verify the model can recover and produce a final answer after receiving an error tool result. Write an integration test that mocks the calculator to raise `ZeroDivisionError`.

3. **Add a third tool.** Implement a `get_current_datetime` tool (no arguments) that returns the current UTC datetime as an ISO 8601 string. Use `@tool` and `datetime.datetime.now(tz=datetime.timezone.utc)`. Bind all three tools to the model. Write a test asserting the model calls this tool when asked "what time is it?".

4. **Tool choice forcing.** Use `bind_tools(tools, tool_choice="calculator")` to force the model to always call the calculator. Verify via unit test that `AIMessage.tool_calls[0]["name"] == "calculator"` regardless of the question. Then explain in a comment why forcing tool choice is rarely correct in production agents.

---

## Self-check

- What is the difference between using `@tool` and subclassing `BaseTool`? When do you prefer each?
- What does `bind_tools()` do to the model — does it change the model instance?
- Why must both the `AIMessage` (with tool calls) and the `ToolMessage` (with results) be appended to the message list before re-invoking the LLM?
- What field on `AIMessage` tells you the model wants to call tools, and what fields does each entry have?
- What is the key production concern with `AgentExecutor`, and what should you use instead for complex agents?

---

## Supplementary Links

- [LangChain tool concept guide](https://python.langchain.com/docs/concepts/tools/)
- [How to create tools](https://python.langchain.com/docs/how_to/custom_tools/)
- [bind_tools() reference](https://python.langchain.com/docs/how_to/tool_calling/)
- [Tool calling with Anthropic](https://python.langchain.com/docs/integrations/chat/anthropic/#tool-calling)
- [AgentExecutor migration to LangGraph](https://python.langchain.com/docs/how_to/migrate_agent/)
