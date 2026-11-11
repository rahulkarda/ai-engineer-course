# Phase 5 · Week 02 · Day 5 — Parallel Tool Calls + Tool Design Best Practices

> Objective: Detect and dispatch parallel tool calls from Claude, then master the design principles that make tools reliable and composable.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

Claude can request **multiple tools in a single response**. Instead of one `tool_use` block in `content`, you get several. All of them arrive before Claude has seen any results — it decided to fan out the work.

Why this matters:
- Reduces round-trips (latency drops, fewer API calls)
- Claude naturally parallelises independent subtasks (search + format, fetch + calculate)
- Your dispatcher must handle *lists* of tool calls, not assume exactly one

The message flow with parallel calls:

```
You  →  assistant message (tool_use ×N)
         → dispatch all N tools concurrently
You  →  user message (tool_result ×N in one message)
         → Claude continues with all results
```

A single `tool_result` per message is NOT wrong, but batching them in one user message is the expected pattern and avoids confusing Claude with interleaved turn ownership.

---

## How it works internally

### Detecting parallel calls

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    tools=[...],
    messages=[{"role": "user", "content": "..."}],
)

# stop_reason == "tool_use" means ≥1 tool call present
if response.stop_reason == "tool_use":
    tool_calls = [b for b in response.content if b.type == "tool_use"]
    # tool_calls may have 1 or many entries
```

Each `tool_use` block has:
- `id`   — unique per call, used to match results back
- `name` — tool name
- `input` — dict of arguments

### Dispatching in parallel (asyncio)

```python
import asyncio
from anthropic.types import ToolUseBlock

async def dispatch(tool_call: ToolUseBlock) -> dict:
    result = await run_tool(tool_call.name, tool_call.input)
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.id,   # must match the call id
        "content": str(result),
    }

async def dispatch_all(tool_calls: list[ToolUseBlock]) -> list[dict]:
    return await asyncio.gather(*[dispatch(tc) for tc in tool_calls])
```

### Sending results back

All results go in **one user message** as a list of `tool_result` blocks:

```python
tool_results = asyncio.run(dispatch_all(tool_calls))

follow_up = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    tools=[...],
    messages=[
        {"role": "user",      "content": original_user_message},
        {"role": "assistant", "content": response.content},   # include full content
        {"role": "user",      "content": tool_results},       # list of tool_result dicts
    ],
)
```

Passing `response.content` (not just text) is critical — it preserves the `tool_use` blocks so Claude can correlate results.

---

## Tool Design Best Practices

### 1. Idempotency
A tool called twice with the same arguments should produce the same result. This is especially important because Claude may retry on ambiguous responses.

Bad: `create_record(name)` — creates a duplicate on retry.  
Good: `upsert_record(name)` or check-then-act inside the tool.

### 2. Clear, discriminating descriptions
Claude selects tools based on descriptions. Vague descriptions cause wrong selections.

```python
# Too vague
{"description": "Get information"}

# Good
{"description": "Search a product catalogue by keyword. Returns a list of matching product names, SKUs, and prices. Does NOT check live inventory."}
```

Include: what it does, what it returns, what it does NOT do.

### 3. Minimal footprint
Each tool should do one thing. Avoid tools that branch on a `mode` or `action` parameter — that's a signal to split.

```python
# Anti-pattern: multi-mode tool
def manage_user(action: str, user_id: str, data: dict | None = None): ...

# Better: separate tools
def get_user(user_id: str): ...
def update_user(user_id: str, data: UserUpdate): ...
def delete_user(user_id: str): ...
```

### 4. Avoid side effects when possible
Prefer read operations in tools; wrap writes with confirmation steps or use agentic checkpoints. Mark write tools explicitly in their description.

### 5. Parameter naming
Use `snake_case`, explicit names, never `x` or `val`. Claude maps natural language to parameters; clear names reduce hallucinated values.

```python
# Unclear
def calc(x: float, y: float, op: str) -> float: ...

# Clear
def calculate(left_operand: float, right_operand: float, operator: str) -> float: ...
```

### 6. When to split vs. combine tools

| Split when | Combine when |
|---|---|
| Operations are independently useful | Steps are always done together |
| One step may fail while other succeeds | Atomicity is required |
| You want parallel dispatch | Parameters are always co-present |
| Different permission levels needed | Round-trip cost exceeds value |

### 7. Return structured data
Return JSON-serialisable dicts, not raw strings. Claude handles structured returns better and you can validate them.

```python
# Fragile
return f"Found {count} results"

# Better
return {"count": count, "results": [...]}
```

---

## Worked Example

Full mock walkthrough: two parallel tool calls (`search` + `calculate`), dispatch, send results, get final answer.

```python
"""
day_5_parallel_tools.py
Demonstrates parallel tool call dispatch with a mocked Anthropic client.
"""
import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tool implementations ---------------------------------------------------

def search_products(query: str, max_results: int = 5) -> dict[str, Any]:
    """Mock product search."""
    logger.info("search_products called: query=%s", query)
    return {
        "results": [
            {"name": "Widget Pro", "price": 29.99, "sku": "WP-001"},
            {"name": "Widget Lite", "price": 14.99, "sku": "WL-002"},
        ][:max_results]
    }


def calculate_total(prices: list[float], quantity: int) -> dict[str, float]:
    """Mock price calculation."""
    logger.info("calculate_total called: prices=%s qty=%d", prices, quantity)
    subtotal = sum(prices) * quantity
    tax = subtotal * 0.08
    return {"subtotal": subtotal, "tax": tax, "total": subtotal + tax}


TOOL_REGISTRY: dict[str, Any] = {
    "search_products": search_products,
    "calculate_total": calculate_total,
}

# --- Dispatch ---------------------------------------------------------------

async def dispatch_tool(tool_id: str, name: str, inputs: dict) -> dict:
    """Dispatch one tool call, return a tool_result block."""
    logger.info("Dispatching tool %s (id=%s)", name, tool_id)
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"type": "tool_result", "tool_use_id": tool_id,
                "content": f"Unknown tool: {name}", "is_error": True}
    try:
        result = fn(**inputs)
        return {"type": "tool_result", "tool_use_id": tool_id, "content": str(result)}
    except Exception as exc:
        logger.exception("Tool %s raised: %s", name, exc)
        return {"type": "tool_result", "tool_use_id": tool_id,
                "content": str(exc), "is_error": True}


async def dispatch_parallel(tool_calls: list) -> list[dict]:
    """Dispatch all tool calls concurrently."""
    tasks = [dispatch_tool(tc.id, tc.name, tc.input) for tc in tool_calls]
    results = await asyncio.gather(*tasks)
    logger.info("All %d tools completed", len(results))
    return list(results)


# --- Mock LLM responses -----------------------------------------------------

def build_mock_client() -> MagicMock:
    """Build a mock Anthropic client that returns parallel tool calls on first
    call, then a final text answer on second call."""
    client = MagicMock(spec=anthropic.Anthropic)

    # First response: two parallel tool_use blocks
    search_block = MagicMock()
    search_block.type = "tool_use"
    search_block.id = "tu_search_001"
    search_block.name = "search_products"
    search_block.input = {"query": "widget", "max_results": 2}

    calc_block = MagicMock()
    calc_block.type = "tool_use"
    calc_block.id = "tu_calc_002"
    calc_block.name = "calculate_total"
    calc_block.input = {"prices": [29.99, 14.99], "quantity": 3}

    first_response = MagicMock()
    first_response.stop_reason = "tool_use"
    first_response.content = [search_block, calc_block]

    # Second response: final text answer
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = (
        "Found 2 widgets. Buying 3 of each costs $134.96 including tax."
    )

    second_response = MagicMock()
    second_response.stop_reason = "end_turn"
    second_response.content = [text_block]

    client.messages.create.side_effect = [first_response, second_response]
    return client


# --- Main loop --------------------------------------------------------------

def run_agent(user_message: str, client: anthropic.Anthropic) -> str:
    """Run a single agentic turn with parallel tool dispatch."""
    tools_schema = [
        {
            "name": "search_products",
            "description": (
                "Search the product catalogue by keyword. "
                "Returns matching product names, SKUs, and prices. "
                "Does NOT check live inventory."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "max_results": {"type": "integer", "description": "Max items to return", "default": 5},
                },
                "required": ["query"],
            },
        },
        {
            "name": "calculate_total",
            "description": "Calculate subtotal, tax (8%), and total for a list of prices and a quantity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "prices": {"type": "array", "items": {"type": "number"}, "description": "Unit prices"},
                    "quantity": {"type": "integer", "description": "Number of units to buy"},
                },
                "required": ["prices", "quantity"],
            },
        },
    ]

    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=1024,
            tools=tools_schema,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text_blocks = [b for b in response.content if b.type == "text"]
            return text_blocks[0].text if text_blocks else ""

        if response.stop_reason == "tool_use":
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            logger.info("Parallel tool calls requested: %d", len(tool_calls))

            tool_results = asyncio.run(dispatch_parallel(tool_calls))

            # Append assistant turn (with tool_use blocks) + tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            break

    return ""


if __name__ == "__main__":
    mock_client = build_mock_client()
    answer = run_agent("Search for widgets and calculate total for 3 of each", mock_client)
    logger.info("Final answer: %s", answer)
```

---

## Exercises (no answers)

1. Extend `TOOL_REGISTRY` with a third tool `format_currency(amount: float, currency: str) -> str` and update the mock so Claude requests all three in parallel.

2. Modify `dispatch_parallel` to record each tool's wall-clock execution time and log a summary after all tasks complete.

3. Implement a `dispatch_sequential` variant that calls tools one at a time and compare the interface — is the signature identical? What would need to change to swap them?

4. Add error injection: make `calculate_total` raise a `ValueError` when `quantity < 1`. Update the dispatcher to mark the result `is_error: True` and verify the mock loop still reaches the final answer.

5. Design a tool schema for a `send_email` tool. Write its JSON schema with at least 4 parameters. Then write a critique of your own design against the 7 best-practice rules from this lesson.

6. Refactor the worked example so `run_agent` is async end-to-end (no `asyncio.run` inside a sync function).

(Solutions: ../solutions/week-02/day-5.md)

---

## Self-check

- [ ] Can you identify a parallel tool call response from `stop_reason` alone? What else should you check?
- [ ] What happens if you send tool results in separate user messages instead of batching them?
- [ ] Name two signs that a tool should be split into two tools.
- [ ] Why must `tool_use_id` in a `tool_result` exactly match the `id` in the `tool_use` block?
- [ ] What does "idempotency" mean for a tool and why is it important in an agentic loop?
- [ ] A tool's description says "processes data" — what is wrong with this and how would you fix it?

---

## Supplementary links

- [Anthropic tool use docs — multiple tools](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Anthropic cookbook — parallel tool use](https://github.com/anthropics/anthropic-cookbook)
- [Python asyncio.gather docs](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
- [JSON Schema reference](https://json-schema.org/understanding-json-schema/)
- [Tool design patterns — function calling guide](https://platform.openai.com/docs/guides/function-calling) (provider-agnostic design principles apply)
