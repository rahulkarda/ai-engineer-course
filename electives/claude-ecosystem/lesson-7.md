# Elective: Claude Ecosystem · Lesson 7 — Tool Use & Server-Side Tools

> Objective: Implement the tool-use loop and understand server-side tools (web search, code execution).
> Time: ~1 hr · Prereqs: [[lesson-1.md]], [[lesson-2.md]]

## Concept

**Tool use** lets Claude call functions you define — or tools Anthropic hosts — to get
real-world data or take actions. The model signals it wants a tool via `stop_reason="tool_use"`;
you execute and return results; the loop continues until `stop_reason="end_turn"`.

Two kinds of tools:

| Kind | Runs where | You implement? |
|------|-----------|---------------|
| Client tools | Your code | Yes — you define the schema and the function |
| Server-side tools | Anthropic servers | No — declare in `tools`, results come back in the response |

**Server-side tools covered here:**
- `web_search_20260209` — live web search (results in `web_search_tool_result` blocks).
- `code_execution_20260521` — Python execution in a sandbox (results in `bash_code_execution_tool_result`).

**Availability:** Server-side tools require models that support them (Opus 5.5, Sonnet 5.5,
Haiku 5.5 and their predecessors). Not available on Bedrock or Vertex.

## How the tool loop works internally

```
1. messages.create(tools=[...], messages=[...])
   → stop_reason="tool_use"  (response.content includes tool_use blocks)

2. For each tool_use block:
   - Extract tool_use.name, tool_use.input (dict)
   - Execute your function
   - Build tool_result block: {type: "tool_result", tool_use_id: ..., content: ...}

3. Append assistant response + all tool_result blocks to messages
4. messages.create(tools=[...], messages=[updated messages])
   → loop back to 1 or → stop_reason="end_turn"
```

**Parallel tool use:** One assistant message can contain multiple `tool_use` blocks.
Execute them concurrently; return **all** results in a **single** user message.
Never split results across multiple messages.

**Error tool results:** Set `"is_error": True` on the `tool_result` block.
Never drop a failed tool call — always return a result.

**Strict schemas:** Add `"strict": True` to a tool definition along with
`"additionalProperties": False` and all fields in `"required"`.
Claude's output is guaranteed to validate against the schema.

**`tool_choice` rules for 5.5 models:** `{"type": "any"}` and `{"type": "tool", "name": ...}`
return a 400 on Opus 5.5, Sonnet 5.5. Use `{"type": "auto"}` and steer via the prompt.

## Worked example

```python
# lesson_7_demo.py
import json
import logging
import os
from collections.abc import Callable

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Client tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_stock_price",
        "description": "Return the current price of a stock ticker symbol.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_company_info",
        "description": "Return basic info (name, sector, description) for a ticker.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
            "additionalProperties": False,
        },
    },
]


# Mock implementations (replace with real data sources in production)
def get_stock_price(ticker: str) -> dict:
    prices = {"AAPL": 195.40, "MSFT": 415.20, "GOOG": 175.60}
    price = prices.get(ticker.upper())
    if price is None:
        return {"error": f"Unknown ticker: {ticker}"}
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}


def get_company_info(ticker: str) -> dict:
    info = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
        "MSFT": {"name": "Microsoft Corp.", "sector": "Technology"},
    }
    return info.get(ticker.upper(), {"error": f"No info for {ticker}"})


TOOL_REGISTRY: dict[str, Callable] = {
    "get_stock_price": get_stock_price,
    "get_company_info": get_company_info,
}


# ---------------------------------------------------------------------------
# Manual agentic loop
# ---------------------------------------------------------------------------

def execute_tool(name: str, tool_input: dict) -> str:
    """Dispatch to the correct tool function; return JSON result string."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**tool_input)
        return json.dumps(result)
    except Exception as e:
        log.exception("Tool %s failed", name)
        return json.dumps({"error": str(e)})


def run_agent(user_message: str, max_iterations: int = 10) -> str:
    """Run the tool-use loop until end_turn or max_iterations."""
    messages: list[dict] = [{"role": "user", "content": user_message}]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        log.info("Iteration %d — stop_reason=%s", iteration, response.stop_reason)

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            log.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

        # Collect all tool_use blocks (may be multiple — parallel tool use)
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        # Append the assistant's turn
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tools and return all results in one user message
        tool_results = []
        for block in tool_use_blocks:
            result_str = execute_tool(block.name, block.input)
            log.info("Tool %s(%s) → %s", block.name, block.input, result_str[:100])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})

    return next(
        (b.text for b in response.content if b.type == "text"),
        "[No text response]",
    )


# ---------------------------------------------------------------------------
# Server-side tool: web search (conceptual — requires a supported model)
# ---------------------------------------------------------------------------

def web_search_demo(query: str) -> str:
    """
    Server-side web search: Claude calls it automatically.
    Results arrive as web_search_tool_result blocks.
    """
    response = client.messages.create(
        model="claude-opus-5-5",  # Web search requires Opus/Sonnet 5.5+
        max_tokens=2048,
        tools=[{
            "type": "web_search_20260209",
            "name": "web_search",
        }],
        messages=[{"role": "user", "content": query}],
    )

    # Server tools run automatically; just read the text answer
    return next(b.text for b in response.content if b.type == "text")


# ---------------------------------------------------------------------------
# Server-side tool: code execution (conceptual)
# ---------------------------------------------------------------------------

def code_execution_demo(task: str) -> str:
    """
    Claude writes and executes Python in a server-side sandbox.
    Results arrive as bash_code_execution_tool_result blocks.
    """
    response = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2048,
        tools=[{
            "type": "code_execution_20260521",
            "name": "code_execution",
        }],
        messages=[{"role": "user", "content": task}],
    )

    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "bash_code_execution_tool_result":
            result = block.content
            if hasattr(result, "stdout") and result.stdout:
                text_parts.append(f"[stdout] {result.stdout}")
            if hasattr(result, "stderr") and result.stderr:
                text_parts.append(f"[stderr] {result.stderr}")

    return "\n".join(text_parts)


if __name__ == "__main__":
    # Client tool loop
    answer = run_agent(
        "What is the current price of AAPL and what sector is it in?"
    )
    log.info("Final answer: %s", answer)
```

Key observations:
- All `tool_result` blocks for one turn go in a **single** user message.
- Use `tool_use_id` (not `id`) to link results to calls.
- Server-side tools execute automatically on Anthropic's servers — no loop code needed.
- Check `block.type` before accessing type-specific attributes.

## Exercises

1. **Parallel tool test.** Write a unit test that injects a mocked response with two
   `tool_use` blocks (for `get_stock_price` and `get_company_info`) and verifies that
   `run_agent` returns both results in a single user message (not two separate messages).

2. **Error tool result.** Modify `execute_tool` to set `"is_error": True` on the
   `tool_result` block when the tool raises an exception.
   Write a test that mocks a tool that throws `ValueError` and asserts the `is_error` flag.

3. **Tool registry from schema.** Write a decorator `@register_tool(description, schema)`
   that adds a function to `TOOL_REGISTRY` and appends its definition to `TOOLS`.
   Register `get_stock_price` and `get_company_info` with it.
   Test that `TOOLS` contains both definitions after decoration.

4. **Max iterations guard.** Write a test that mocks the client to always return
   `stop_reason="tool_use"` and verifies that `run_agent` stops after `max_iterations`
   without an infinite loop.

(Solutions: solutions/lesson-7.md)

## Self-check

- [ ] I can define a tool with a strict JSON schema and register it in a loop.
- [ ] I understand the full tool-use loop: send → receive tool_use → execute → return results → repeat.
- [ ] I know how to handle parallel tool calls in a single assistant turn.
- [ ] I can implement error tool results with `is_error: True`.
- [ ] I understand the conceptual difference between client tools and server-side tools.

## Supplementary links

- [Tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Tool schemas reference](https://docs.anthropic.com/en/api/messages#body-tools)
- [Web search guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/web-search-tool)
- [Code execution guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/code-execution-tool)
