# Solution: Lesson 7 — Tool Use & Server-Side Tools

## Exercise 1: Parallel tool test

```python
from unittest.mock import MagicMock, call
import json

def make_tool_use_block(tool_id: str, name: str, input_dict: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def make_end_turn_response(text: str = "Here is the answer"):
    block = MagicMock(); block.type = "text"; block.text = text
    resp = MagicMock()
    resp.stop_reason = "end_turn"
    resp.content = [block]
    return resp


def make_tool_use_response(tool_blocks: list):
    resp = MagicMock()
    resp.stop_reason = "tool_use"
    resp.content = tool_blocks
    return resp


def test_parallel_tool_results_in_single_message():
    """
    When the response has two tool_use blocks, run_agent must
    return both results in ONE user message (not two separate messages).
    """
    client = MagicMock()

    # First call: two parallel tool_use blocks
    price_block  = make_tool_use_block("tu_1", "get_stock_price",  {"ticker": "AAPL"})
    info_block   = make_tool_use_block("tu_2", "get_company_info", {"ticker": "AAPL"})
    tool_resp    = make_tool_use_response([price_block, info_block])
    final_resp   = make_end_turn_response("AAPL is $195.40 and is a Technology company.")

    client.messages.create.side_effect = [tool_resp, final_resp]

    answer = run_agent("What is AAPL's price and sector?", client)

    assert answer == "AAPL is $195.40 and is a Technology company."

    # Second call's messages argument should contain a user message
    # with TWO tool_result blocks in one message.
    second_call_messages = client.messages.create.call_args_list[1][1]["messages"]
    last_user_msg = next(
        m for m in reversed(second_call_messages) if m["role"] == "user"
    )
    assert isinstance(last_user_msg["content"], list)
    assert len(last_user_msg["content"]) == 2
    assert all(r["type"] == "tool_result" for r in last_user_msg["content"])


def run_agent(user_message: str, client, max_iterations: int = 10) -> str:
    """Extracted for testability — same logic as lesson_7_demo."""
    from lesson_7_demo import TOOLS, execute_tool  # import from lesson module
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in tool_use_blocks:
            result_str = execute_tool(block.name, block.input)
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
```

## Exercise 2: Error tool result

```python
import json
import logging

log = logging.getLogger(__name__)


def execute_tool_with_error_flag(name: str, tool_input: dict, tool_registry: dict) -> dict:
    """
    Execute a tool. Return a tool_result dict with is_error=True on exception.
    """
    fn = tool_registry.get(name)
    if fn is None:
        return {
            "content": json.dumps({"error": f"Unknown tool: {name}"}),
            "is_error": True,
        }
    try:
        result = fn(**tool_input)
        return {"content": json.dumps(result), "is_error": False}
    except Exception as exc:
        log.exception("Tool %s raised %s", name, exc)
        return {
            "content": json.dumps({"error": str(exc)}),
            "is_error": True,
        }


def test_is_error_set_on_exception():
    def bad_tool(**kwargs):
        raise ValueError("Data source unavailable")

    registry = {"bad_tool": bad_tool}
    result = execute_tool_with_error_flag("bad_tool", {}, registry)

    assert result["is_error"] is True
    data = json.loads(result["content"])
    assert "Data source unavailable" in data["error"]


def test_is_error_false_on_success():
    def good_tool(ticker: str):
        return {"price": 100.0}

    registry = {"good_tool": good_tool}
    result = execute_tool_with_error_flag("good_tool", {"ticker": "X"}, registry)

    assert result["is_error"] is False
    data = json.loads(result["content"])
    assert data["price"] == 100.0
```

## Exercise 3: Tool registry decorator

```python
from typing import Callable

TOOL_REGISTRY: dict[str, Callable] = {}
TOOLS: list[dict] = []


def register_tool(description: str, schema: dict):
    """
    Decorator that registers a function as a tool.
    Adds it to TOOL_REGISTRY and appends its definition to TOOLS.
    """
    def decorator(fn: Callable) -> Callable:
        TOOL_REGISTRY[fn.__name__] = fn
        TOOLS.append({
            "name": fn.__name__,
            "description": description,
            "strict": True,
            "input_schema": {
                **schema,
                "additionalProperties": False,
            },
        })
        return fn
    return decorator


@register_tool(
    description="Return the current price of a stock ticker.",
    schema={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
)
def get_stock_price(ticker: str) -> dict:
    return {"ticker": ticker, "price": 195.40}


@register_tool(
    description="Return basic info for a ticker.",
    schema={
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
)
def get_company_info(ticker: str) -> dict:
    return {"ticker": ticker, "name": "Apple Inc."}


def test_registry_contains_both_tools():
    assert "get_stock_price" in TOOL_REGISTRY
    assert "get_company_info" in TOOL_REGISTRY


def test_tools_list_has_correct_names():
    names = [t["name"] for t in TOOLS]
    assert "get_stock_price" in names
    assert "get_company_info" in names


def test_tools_have_additionalproperties_false():
    for tool in TOOLS:
        assert tool["input_schema"].get("additionalProperties") is False
```

## Exercise 4: Max iterations guard

```python
def test_max_iterations_prevents_infinite_loop():
    """Mock client always returns tool_use; verify loop stops at max_iterations."""
    client = MagicMock()

    tool_block = make_tool_use_block("tu_1", "get_stock_price", {"ticker": "AAPL"})
    always_tool_use = make_tool_use_response([tool_block])

    # Always return tool_use — never end_turn
    client.messages.create.return_value = always_tool_use

    MAX = 5
    # run_agent should stop after MAX iterations without infinite loop
    result = run_agent("What is AAPL?", client, max_iterations=MAX)

    # create() was called at most MAX times
    assert client.messages.create.call_count <= MAX
    # Result is the fallback string (no text block in final response)
    assert result == "[No text response]"
```
