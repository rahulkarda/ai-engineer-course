# Solutions · Lesson 1 — What Are Reasoning Models?

## Exercise 1 — Block inspector

```python
"""solutions/lesson_1_ex1.py"""
from __future__ import annotations
from typing import Any


def parse_thinking_response(response: Any) -> dict[str, str | None]:
    """
    Extract thinking trace and answer from a Messages API response.

    Returns:
        {"thinking": str | None, "answer": str}
    """
    thinking: str | None = None
    answer = ""

    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text

    return {"thinking": thinking, "answer": answer}
```

```python
"""tests/test_lesson_1_ex1.py"""
from unittest.mock import MagicMock
import pytest
from solutions.lesson_1_ex1 import parse_thinking_response


def _make_response(blocks: list[dict]) -> MagicMock:
    content = []
    for b in blocks:
        block = MagicMock()
        block.type = b["type"]
        if b["type"] == "thinking":
            block.thinking = b["thinking"]
        elif b["type"] == "text":
            block.text = b["text"]
        content.append(block)
    response = MagicMock()
    response.content = content
    return response


def test_with_thinking():
    response = _make_response([
        {"type": "thinking", "thinking": "Let me reason..."},
        {"type": "text", "text": "The answer is 42."},
    ])
    result = parse_thinking_response(response)
    assert result["thinking"] == "Let me reason..."
    assert result["answer"] == "The answer is 42."


def test_without_thinking():
    response = _make_response([
        {"type": "text", "text": "Paris."},
    ])
    result = parse_thinking_response(response)
    assert result["thinking"] is None
    assert result["answer"] == "Paris."
```

---

## Exercise 2 — Budget effect (mock)

```python
"""solutions/lesson_1_ex2.py"""
from __future__ import annotations
import logging
from unittest.mock import MagicMock, patch

log = logging.getLogger(__name__)


def make_stub_response(budget: int) -> MagicMock:
    """Return a stub response with thinking_chars proportional to budget."""
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "x" * budget  # proportional length

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Answer"

    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50

    response = MagicMock()
    response.content = [thinking_block, text_block]
    response.usage = usage
    return response


def test_budget_scaling_mock():
    """Mock-based test: thinking_chars should scale with budget."""
    budgets = [1024, 3000, 8000]
    chars = []

    for budget in budgets:
        stub = make_stub_response(budget)
        thinking_chars = sum(
            len(b.thinking) for b in stub.content if b.type == "thinking"
        )
        chars.append(thinking_chars)
        log.info("budget=%d  thinking_chars=%d", budget, thinking_chars)

    assert chars[0] < chars[1] < chars[2], "Thinking chars should grow with budget"
```

---

## Exercise 3 — max_tokens guard

```python
"""solutions/lesson_1_ex3.py"""
import anthropic
import os


def safe_thinking_call(
    problem: str,
    budget: int,
    max_combined: int = 16000,
    answer_headroom: int = 500,
) -> tuple[str, str]:
    if budget < 1024:
        raise ValueError(f"budget_tokens must be >= 1024, got {budget}")
    if budget + answer_headroom > max_combined:
        raise ValueError(
            f"budget ({budget}) + headroom ({answer_headroom}) = "
            f"{budget + answer_headroom} exceeds max_combined ({max_combined})"
        )

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + answer_headroom,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": problem}],
    )

    thinking = ""
    answer = ""
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
        elif block.type == "text":
            answer = block.text

    return thinking, answer
```

```python
"""tests/test_lesson_1_ex3.py"""
import pytest
from solutions.lesson_1_ex3 import safe_thinking_call


def test_budget_too_small():
    with pytest.raises(ValueError, match="1024"):
        safe_thinking_call("problem", budget=500)


def test_budget_plus_headroom_exceeds_max():
    # 15600 + 500 = 16100 > 16000
    with pytest.raises(ValueError, match="exceeds max_combined"):
        safe_thinking_call("problem", budget=15600, answer_headroom=500)


def test_valid_budget_does_not_raise(monkeypatch):
    # patch the Anthropic client so no real call is made
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    thinking_block.thinking = "ok"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "answer"
    mock_response.content = [thinking_block, text_block]

    with patch("solutions.lesson_1_ex3.anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        trace, answer = safe_thinking_call("What is 2+2?", budget=2000)
    assert answer == "answer"
```

---

## Exercise 4 — Thinking disabled

```python
"""tests/test_lesson_1_ex4.py"""
from unittest.mock import MagicMock, patch


def call_without_thinking(problem: str) -> str:
    """Call Claude with thinking disabled."""
    import anthropic, os
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": problem}],
    )
    return next(b.text for b in response.content if b.type == "text")


def test_no_thinking_block(monkeypatch):
    mock_response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Paris."
    mock_response.content = [text_block]

    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_response
        # No thinking block present
        types = [b.type for b in mock_response.content]
        assert "thinking" not in types
```

---

## Exercise 5 — Logging with caplog

```python
"""solutions/lesson_1_ex5.py"""
import logging
import os
import anthropic

log = logging.getLogger(__name__)


def solve_with_thinking_logged(problem: str, budget: int = 5000) -> tuple[str, str]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": problem}],
    )

    thinking_trace = ""
    answer = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_trace = block.thinking
        elif block.type == "text":
            answer = block.text

    log.debug(
        "thinking_stats budget=%d thinking_tokens=%d answer_tokens=%d",
        budget,
        len(thinking_trace),
        len(answer),
    )
    return thinking_trace, answer
```

```python
"""tests/test_lesson_1_ex5.py"""
import logging
from unittest.mock import MagicMock, patch
from solutions.lesson_1_ex5 import solve_with_thinking_logged


def _make_response(thinking_text: str, answer_text: str) -> MagicMock:
    t = MagicMock(); t.type = "thinking"; t.thinking = thinking_text
    a = MagicMock(); a.type = "text"; a.text = answer_text
    r = MagicMock(); r.content = [t, a]
    return r


def test_debug_log_appears(caplog):
    stub = _make_response("I think...", "42")
    with patch("solutions.lesson_1_ex5.anthropic.Anthropic") as M:
        M.return_value.messages.create.return_value = stub
        with caplog.at_level(logging.DEBUG, logger="solutions.lesson_1_ex5"):
            solve_with_thinking_logged("2+2?", budget=2000)
    assert "thinking_stats" in caplog.text
    assert "budget=2000" in caplog.text
```
