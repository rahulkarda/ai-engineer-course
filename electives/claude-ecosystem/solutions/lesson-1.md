# Solution: Lesson 1 — Messages API Mechanics

## Exercise 1: Stop reason guard with auto-retry

```python
# tests/test_lesson1.py  (excerpt)
import pytest
from unittest.mock import MagicMock, patch, call

def single_turn_with_retry(client, user_text: str, max_tokens: int = 512) -> str:
    """
    Call the API; if stop_reason is max_tokens, retry with doubled max_tokens
    (capped at 4096).
    """
    MAX_CAP = 4096
    current_max = max_tokens

    while True:
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=current_max,
            system="You are a concise technical assistant.",
            messages=[{"role": "user", "content": user_text}],
        )
        if response.stop_reason != "max_tokens":
            break
        new_max = min(current_max * 2, MAX_CAP)
        if new_max == current_max:
            break  # already at cap, give up
        current_max = new_max

    return next(b.text for b in response.content if b.type == "text")


def make_mock_response(stop_reason: str, text: str = "hello"):
    """Build a minimal mock Message object."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = [block]
    resp.stop_details = None
    return resp


def test_retries_on_max_tokens():
    """When stop_reason is max_tokens, the function should retry with doubled max_tokens."""
    client = MagicMock()

    truncated = make_mock_response("max_tokens", "truncated")
    full = make_mock_response("end_turn", "full answer")

    client.messages.create.side_effect = [truncated, full]

    result = single_turn_with_retry(client, "explain something", max_tokens=128)

    assert result == "full answer"
    assert client.messages.create.call_count == 2

    first_call_kwargs = client.messages.create.call_args_list[0][1]
    second_call_kwargs = client.messages.create.call_args_list[1][1]
    assert first_call_kwargs["max_tokens"] == 128
    assert second_call_kwargs["max_tokens"] == 256


def test_respects_max_cap():
    """Retry stops when max_tokens is already at the cap."""
    client = MagicMock()
    always_truncated = make_mock_response("max_tokens", "still truncated")
    client.messages.create.return_value = always_truncated

    # Start at cap — should call exactly once
    result = single_turn_with_retry(client, "explain", max_tokens=4096)
    assert client.messages.create.call_count == 1


def test_no_retry_on_end_turn():
    client = MagicMock()
    client.messages.create.return_value = make_mock_response("end_turn", "answer")
    single_turn_with_retry(client, "hello")
    assert client.messages.create.call_count == 1
```

## Exercise 2: Content block extractor

```python
from collections import defaultdict
from anthropic.types import Message

def extract_blocks(response) -> dict[str, list]:
    """
    Return a dict keyed by block.type containing lists of matching blocks.
    Works with any response object whose .content is iterable.
    """
    result: dict[str, list] = defaultdict(list)
    for block in response.content:
        result[block.type].append(block)
    return dict(result)


def test_extract_blocks_mixed():
    thinking = MagicMock(); thinking.type = "thinking"; thinking.thinking = "..."
    text1 = MagicMock(); text1.type = "text"; text1.text = "Hello"
    text2 = MagicMock(); text2.type = "text"; text2.text = "World"
    tool = MagicMock(); tool.type = "tool_use"

    resp = MagicMock()
    resp.content = [thinking, text1, text2, tool]

    blocks = extract_blocks(resp)
    assert len(blocks["thinking"]) == 1
    assert len(blocks["text"]) == 2
    assert len(blocks["tool_use"]) == 1


def test_extract_blocks_empty():
    resp = MagicMock(); resp.content = []
    assert extract_blocks(resp) == {}
```

## Exercise 3: Streaming token accounting

```python
def get_streaming_cost(client, prompt: str, model: str = "claude-haiku-5-5") -> dict:
    """Stream a response and return usage + estimated cost."""
    PRICES = {
        "claude-haiku-5-5": (0.10, 0.50),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-opus-5-5": (4.00, 20.00),
    }

    with client.messages.stream(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for _ in stream.text_stream:
            pass  # consume stream
        final = stream.get_final_message()

    input_price, output_price = PRICES.get(model, (4.00, 20.00))
    cost = (
        final.usage.input_tokens * input_price / 1_000_000
        + final.usage.output_tokens * output_price / 1_000_000
    )
    return {
        "input_tokens": final.usage.input_tokens,
        "output_tokens": final.usage.output_tokens,
        "estimated_cost_usd": round(cost, 8),
    }
```

## Exercise 4: System prompt effect (mocked)

```python
def test_different_system_prompts_differ():
    client = MagicMock()

    resp_fr = MagicMock()
    resp_fr.stop_reason = "end_turn"
    fr_block = MagicMock(); fr_block.type = "text"; fr_block.text = "Bonjour"
    resp_fr.content = [fr_block]

    resp_es = MagicMock()
    resp_es.stop_reason = "end_turn"
    es_block = MagicMock(); es_block.type = "text"; es_block.text = "Hola"
    resp_es.content = [es_block]

    client.messages.create.side_effect = [resp_fr, resp_es]

    def call(system: str) -> str:
        response = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=64,
            system=system,
            messages=[{"role": "user", "content": "Say hello"}],
        )
        return next(b.text for b in response.content if b.type == "text")

    answer_fr = call("Reply in French.")
    answer_es = call("Reply in Spanish.")
    assert answer_fr != answer_es
```
