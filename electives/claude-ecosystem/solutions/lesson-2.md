# Solution: Lesson 2 — Model Family & Extended Thinking

## Exercise 1: Cost calculator

```python
# models cost table — matches skill reference (Nov 2026)
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5":   (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5":  (0.10,  0.50),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return USD cost for a request. Raises KeyError for unknown models."""
    input_price, output_price = MODEL_PRICES[model]
    return (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )


# tests/test_lesson2.py
import pytest

@pytest.mark.parametrize("model,inp,out,expected", [
    ("claude-haiku-5-5",  1_000_000, 1_000_000, 0.10 + 0.50),
    ("claude-sonnet-5-5", 1_000_000, 1_000_000, 2.00 + 10.00),
    ("claude-opus-5-5",   1_000_000, 1_000_000, 4.00 + 20.00),
    ("claude-haiku-5-5",  500, 100, 500 * 0.10 / 1e6 + 100 * 0.50 / 1e6),
])
def test_estimate_cost(model, inp, out, expected):
    assert abs(estimate_cost(model, inp, out) - expected) < 1e-9


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_cost("claude-fake-99", 100, 100)
```

## Exercise 2: Effort parameter assertion

```python
from unittest.mock import MagicMock, patch

def test_reason_hard_uses_xhigh_effort():
    """Assert that reason_hard sends output_config={"effort": "xhigh"}."""
    client = MagicMock()

    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "Deep thought..."

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "42"

    mock_response = MagicMock()
    mock_response.content = [thinking_block, text_block]
    mock_response.usage = MagicMock(input_tokens=100, output_tokens=50)
    client.messages.create.return_value = mock_response

    # Call the function from lesson_2_demo using the mocked client
    from electives.claude_ecosystem.lesson_2_demo import reason_hard as _reason_hard
    # Patch the module-level client
    with patch("electives.claude_ecosystem.lesson_2_demo.client", client):
        thinking, answer = _reason_hard("What is the answer to life?")

    call_kwargs = client.messages.create.call_args[1]
    assert call_kwargs.get("output_config") == {"effort": "xhigh"}
    assert call_kwargs.get("model") == "claude-opus-5-5"
    assert answer == "42"
    assert thinking == "Deep thought..."
```

## Exercise 3: Thinking block extraction (handles missing thinking)

```python
def extract_thinking_and_text(response) -> tuple[str, str]:
    """
    Return (thinking_text, answer_text).
    Returns empty string for missing thinking block.
    """
    thinking_text = ""
    answer_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking or ""
        elif block.type == "text":
            answer_text = block.text
    return thinking_text, answer_text


def test_extract_with_both_blocks():
    t = MagicMock(); t.type = "thinking"; t.thinking = "I think..."
    a = MagicMock(); a.type = "text"; a.text = "Answer"
    resp = MagicMock(); resp.content = [t, a]
    thinking, answer = extract_thinking_and_text(resp)
    assert thinking == "I think..."
    assert answer == "Answer"


def test_extract_without_thinking():
    a = MagicMock(); a.type = "text"; a.text = "Just text"
    resp = MagicMock(); resp.content = [a]
    thinking, answer = extract_thinking_and_text(resp)
    assert thinking == ""
    assert answer == "Just text"


def test_extract_empty_thinking_field():
    """Handles thinking block with empty .thinking string (display="omitted")."""
    t = MagicMock(); t.type = "thinking"; t.thinking = ""
    a = MagicMock(); a.type = "text"; a.text = "Answer"
    resp = MagicMock(); resp.content = [t, a]
    thinking, _ = extract_thinking_and_text(resp)
    assert thinking == ""
```

## Exercise 4: Multi-task router with cost tracking

```python
from dataclasses import dataclass, field


@dataclass
class MultiTaskResult:
    task: str
    answer: str
    model: str
    estimated_cost_usd: float


TIER_MAP = {
    "simple":  ("claude-haiku-5-5",  "low"),
    "medium":  ("claude-sonnet-5-5", "high"),
    "complex": ("claude-opus-5-5",   "xhigh"),
}


def route_tasks(
    tasks: list[tuple[str, str]],   # [(task_text, complexity), ...]
    client,
) -> tuple[list[MultiTaskResult], float]:
    """
    Route each task to the appropriate model tier.
    Returns (results, total_estimated_cost_usd).
    Mocked usage: each call returns 200 input + 100 output tokens.
    """
    results: list[MultiTaskResult] = []
    total_cost = 0.0

    for task_text, complexity in tasks:
        model, effort = TIER_MAP.get(complexity, ("claude-opus-5-5", "high"))

        response = client.messages.create(
            model=model,
            max_tokens=2048,
            output_config={"effort": effort},
            messages=[{"role": "user", "content": task_text}],
        )

        answer = next(b.text for b in response.content if b.type == "text")
        cost = estimate_cost(model, response.usage.input_tokens, response.usage.output_tokens)
        total_cost += cost
        results.append(MultiTaskResult(task_text, answer, model, cost))

    return results, total_cost


def test_route_tasks():
    client = MagicMock()

    def make_resp(text):
        block = MagicMock(); block.type = "text"; block.text = text
        resp = MagicMock()
        resp.content = [block]
        resp.usage = MagicMock(input_tokens=200, output_tokens=100)
        return resp

    client.messages.create.side_effect = [
        make_resp("easy answer"),
        make_resp("complex answer"),
    ]

    tasks = [
        ("Classify this as spam or not", "simple"),
        ("Design a microservices architecture", "complex"),
    ]

    results, total_cost = route_tasks(tasks, client)

    assert results[0].model == "claude-haiku-5-5"
    assert results[1].model == "claude-opus-5-5"
    # Haiku: 200*0.10/1e6 + 100*0.50/1e6 = 0.00007
    # Opus:  200*4.00/1e6 + 100*20.0/1e6 = 0.00280
    assert abs(total_cost - (0.00007 + 0.00280)) < 1e-7
```
