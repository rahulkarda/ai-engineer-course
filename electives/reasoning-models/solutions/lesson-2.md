# Solutions · Lesson 2 — Test-Time Compute Scaling

## Exercise 1 — Budget recommendation function

```python
"""solutions/lesson_2_ex1.py"""
from __future__ import annotations
import re


def recommend_budget(problem: str) -> int:
    """
    Classify problem difficulty and return a thinking budget.

    Easy  → 1024
    Medium → 4000
    Hard  → 10000
    """
    text = problem.lower()
    words = text.split()
    word_count = len(words)

    math_keywords = {"prove", "proof", "calculate", "derive", "solve", "integral",
                     "differentiate", "matrix", "eigenvalue", "algorithm", "complexity"}
    logic_keywords = {"if", "then", "therefore", "deduce", "constraint", "schedule",
                      "optimize", "optimise", "verify", "formal"}

    math_hits = sum(1 for w in words if w.rstrip(".,?:") in math_keywords)
    logic_hits = sum(1 for w in words if w.rstrip(".,?:") in logic_keywords)
    number_count = len(re.findall(r"\b\d+\b", text))

    hard_score = math_hits * 3 + logic_hits * 2 + (number_count > 4) * 2 + (word_count > 80) * 1

    if hard_score >= 6:
        return 10000
    if hard_score >= 2 or word_count > 40:
        return 4000
    return 1024
```

```python
"""tests/test_lesson_2_ex1.py"""
import pytest
from solutions.lesson_2_ex1 import recommend_budget


@pytest.mark.parametrize("problem,expected", [
    ("What is the capital of France?", 1024),
    ("Translate this sentence into Spanish.", 1024),
    ("Summarize this paragraph in one sentence.", 1024),
    (
        "A customer placed three orders. Each order has a 10% discount applied after a $5 fee. "
        "Calculate the total cost for orders of $20, $35, and $50.",
        4000,
    ),
    (
        "Given a binary tree, determine whether it is height-balanced. "
        "Describe your approach and analyze the time complexity.",
        4000,
    ),
    (
        "Prove using mathematical induction that the sum of the first n odd numbers equals n^2. "
        "Show each step formally.",
        10000,
    ),
    (
        "Design and verify a lock-free queue algorithm. "
        "Prove it is linearizable and deadlock-free under all interleavings.",
        10000,
    ),
    (
        "Derive the closed-form solution for the integral of x^2 * e^x dx using integration by parts.",
        10000,
    ),
])
def test_recommend_budget(problem: str, expected: int):
    result = recommend_budget(problem)
    assert result == expected, f"Problem: {problem!r}\nExpected: {expected}, Got: {result}"
```

---

## Exercise 2 — Streaming with budget

```python
"""solutions/lesson_2_ex2.py"""
from __future__ import annotations
import os
import anthropic


def stream_with_thinking(problem: str, budget: int) -> tuple[str, str]:
    """Stream a thinking response; return (thinking, answer)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    thinking_parts: list[str] = []
    text_parts: list[str] = []

    with client.messages.stream(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": problem}],
    ) as stream:
        for event in stream:
            if not hasattr(event, "type"):
                continue
            if event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "thinking_delta":
                    thinking_parts.append(delta.thinking)
                elif delta.type == "text_delta":
                    text_parts.append(delta.text)

    return "".join(thinking_parts), "".join(text_parts)
```

```python
"""tests/test_lesson_2_ex2.py"""
from unittest.mock import MagicMock, patch, __version__
from solutions.lesson_2_ex2 import stream_with_thinking


def _make_events(thinking_chunks: list[str], text_chunks: list[str]) -> list:
    events = []
    for chunk in thinking_chunks:
        delta = MagicMock(); delta.type = "thinking_delta"; delta.thinking = chunk
        event = MagicMock(); event.type = "content_block_delta"; event.delta = delta
        events.append(event)
    for chunk in text_chunks:
        delta = MagicMock(); delta.type = "text_delta"; delta.text = chunk
        event = MagicMock(); event.type = "content_block_delta"; event.delta = delta
        events.append(event)
    return events


def test_streaming_captures_both_streams():
    events = _make_events(
        thinking_chunks=["Step 1. ", "Step 2."],
        text_chunks=["The answer ", "is 42."],
    )

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter(events))
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch("solutions.lesson_2_ex2.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.stream.return_value = mock_stream
        thinking, answer = stream_with_thinking("2+2?", budget=2000)

    assert thinking == "Step 1. Step 2."
    assert answer == "The answer is 42."
```

---

## Exercise 3 — Multi-turn thinking

```python
"""solutions/lesson_2_ex3.py"""
from __future__ import annotations
import os
import anthropic


def two_turn_thinking(
    question1: str,
    question2: str,
    budget: int = 4000,
) -> tuple[str, str]:
    """Two-turn conversation with thinking; returns (answer1, answer2)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    messages: list[dict] = [{"role": "user", "content": question1}]

    r1 = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=messages,
    )
    # Preserve ALL blocks including thinking
    messages.append({"role": "assistant", "content": r1.content})
    messages.append({"role": "user", "content": question2})

    r2 = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=messages,
    )

    answer1 = next(b.text for b in r1.content if b.type == "text")
    answer2 = next(b.text for b in r2.content if b.type == "text")
    return answer1, answer2
```

```python
"""tests/test_lesson_2_ex3.py"""
from unittest.mock import MagicMock, call, patch
from solutions.lesson_2_ex3 import two_turn_thinking


def _make_response(thinking_text: str, answer_text: str) -> MagicMock:
    t = MagicMock(); t.type = "thinking"; t.thinking = thinking_text
    a = MagicMock(); a.type = "text"; a.text = answer_text
    r = MagicMock(); r.content = [t, a]
    return r


def test_thinking_blocks_included_in_second_call():
    r1 = _make_response("thinking1", "answer1")
    r2 = _make_response("thinking2", "answer2")

    with patch("solutions.lesson_2_ex3.anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.side_effect = [r1, r2]

        two_turn_thinking("Q1", "Q2", budget=2000)

    # Second call's messages should contain the thinking block from turn 1
    second_call_messages = mock_create.call_args_list[1][1]["messages"]
    assistant_turn = next(m for m in second_call_messages if m["role"] == "assistant")
    block_types = [b.type for b in assistant_turn["content"]]
    assert "thinking" in block_types
```

---

## Exercise 4 — Token budget arithmetic

```python
"""solutions/lesson_2_ex4.py"""


def max_tokens_needed(budget: int, max_answer: int) -> int:
    """Return the minimum safe max_tokens for a thinking call."""
    return budget + max_answer
```

```python
"""tests/test_lesson_2_ex4.py"""
import pytest
from solutions.lesson_2_ex4 import max_tokens_needed


@pytest.mark.parametrize("budget,max_answer,expected", [
    (2000, 500, 2500),
    (10000, 1000, 11000),
    (1024, 256, 1280),
])
def test_max_tokens_needed(budget: int, max_answer: int, expected: int):
    assert max_tokens_needed(budget, max_answer) == expected
```

---

## Exercise 5 — Cost estimator

```python
"""solutions/lesson_2_ex5.py"""
from __future__ import annotations


def estimate_cost(
    budget_tokens: int,
    output_tokens: int,
    thinking_rate_per_m: float,
    output_rate_per_m: float,
) -> float:
    """
    Estimate cost in USD.

    Args:
        budget_tokens: thinking tokens used
        output_tokens: text output tokens
        thinking_rate_per_m: USD per 1M thinking tokens
        output_rate_per_m: USD per 1M output tokens
    """
    thinking_cost = (budget_tokens / 1_000_000) * thinking_rate_per_m
    output_cost = (output_tokens / 1_000_000) * output_rate_per_m
    return round(thinking_cost + output_cost, 8)
```

```python
"""tests/test_lesson_2_ex5.py"""
import pytest
from solutions.lesson_2_ex5 import estimate_cost

SONNET_THINKING = 3.0   # $/1M
SONNET_OUTPUT = 15.0    # $/1M


def test_zero_tokens():
    assert estimate_cost(0, 0, SONNET_THINKING, SONNET_OUTPUT) == 0.0


def test_thinking_only():
    # 1M thinking tokens at $3/1M = $3.00
    result = estimate_cost(1_000_000, 0, SONNET_THINKING, SONNET_OUTPUT)
    assert abs(result - 3.0) < 1e-6


def test_combined():
    # 100k thinking + 10k output
    result = estimate_cost(100_000, 10_000, SONNET_THINKING, SONNET_OUTPUT)
    expected = (100_000 / 1e6) * 3.0 + (10_000 / 1e6) * 15.0
    assert abs(result - expected) < 1e-8


def test_rate_injection():
    """Rates must be injected, not hardcoded — test different rates."""
    r1 = estimate_cost(1000, 1000, 1.0, 1.0)
    r2 = estimate_cost(1000, 1000, 2.0, 2.0)
    assert r2 == pytest.approx(r1 * 2, rel=1e-6)
```
