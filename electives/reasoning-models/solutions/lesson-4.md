# Solutions · Lesson 4 — Prompting Reasoning Models

## Exercise 1 — Prompt comparison (structural test)

```python
"""solutions/lesson_4_ex1.py"""
from __future__ import annotations
import re

OVERCONSTRAINED_PROMPT = """\
Solve this problem step by step.
Step 1: Identify the variables.
Step 2: Write the equations.
Step 3: Solve for each variable.
Step 4: Verify your answer.

Problem: A train leaves station A at 60 mph heading toward station B 120 miles away.
Another train leaves B toward A at 40 mph. They depart simultaneously.
After how many hours do they meet?
"""

CLEAN_PROMPT = """\
A train leaves station A at 60 mph heading toward station B 120 miles away.
Another train leaves B toward A at 40 mph. They depart simultaneously.
After how many hours do they meet? Return only the number.
"""


def extract_number(text: str) -> str | None:
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return matches[-1] if matches else None
```

```python
"""tests/test_lesson_4_ex1.py"""
from unittest.mock import MagicMock, patch
from solutions.lesson_4_ex1 import (
    OVERCONSTRAINED_PROMPT, CLEAN_PROMPT, extract_number
)


def _make_response(text: str) -> MagicMock:
    block = MagicMock(); block.type = "text"; block.text = text
    r = MagicMock(); r.content = [block]
    return r


def test_both_prompts_produce_result():
    """Both prompts should produce a parseable numeric answer (structural test)."""
    with patch("anthropic.Anthropic") as MockClient:
        mock_create = MockClient.return_value.messages.create
        mock_create.side_effect = [
            _make_response("The answer is 1.2 hours."),
            _make_response("1.2"),
        ]

        client = MockClient()
        for prompt in [OVERCONSTRAINED_PROMPT, CLEAN_PROMPT]:
            response = client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = next(b.text for b in response.content if b.type == "text")
            assert extract_number(answer_text) is not None, f"No number in: {answer_text}"
```

---

## Exercise 2 — Majority vote edge cases

```python
"""solutions/lesson_4_ex2.py"""
from __future__ import annotations
import re
from collections import Counter


def extract_numeric(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


def majority_vote(answers: list[str]) -> str:
    """Return the most common numeric answer; fallback to first on tie/no-number."""
    extracted = [extract_numeric(a) for a in answers]
    valid = [x for x in extracted if x is not None]
    if not valid:
        return answers[0] if answers else ""
    counts = Counter(valid)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    top_answers = [ans for ans, cnt in most_common if cnt == top_count]
    return top_answers[0]  # deterministic: pick the first among ties
```

```python
"""tests/test_lesson_4_ex2.py"""
import pytest
from solutions.lesson_4_ex2 import majority_vote


def test_all_identical():
    assert majority_vote(["The answer is 5.", "5.", "Answer: 5"]) == "5"


def test_no_numeric_answer():
    answers = ["Paris.", "The city is Paris.", "Paris is correct."]
    result = majority_vote(answers)
    assert result == answers[0]  # fallback to first


def test_three_way_tie():
    # "1", "2", "3" each appear once — any is valid
    answers = ["1", "2", "3"]
    result = majority_vote(answers)
    assert result in {"1", "2", "3"}


def test_single_answer():
    assert majority_vote(["42"]) == "42"


def test_clear_majority():
    answers = ["3", "3", "4", "3", "5"]
    assert majority_vote(answers) == "3"
```

---

## Exercise 3 — Custom verifier

```python
"""solutions/lesson_4_ex3.py"""
from __future__ import annotations
import json


def json_verifier(answer: str) -> float:
    """
    Return 1.0 if answer is valid JSON with 'x' and 'y' keys, else 0.0.
    """
    try:
        data = json.loads(answer)
    except (json.JSONDecodeError, ValueError):
        return 0.0
    if isinstance(data, dict) and "x" in data and "y" in data:
        return 1.0
    return 0.0
```

```python
"""tests/test_lesson_4_ex3.py"""
from solutions.lesson_4_ex3 import json_verifier


def test_valid_json_with_xy():
    assert json_verifier('{"x": 3, "y": -1}') == 1.0


def test_valid_json_missing_y():
    assert json_verifier('{"x": 3}') == 0.0


def test_invalid_json():
    assert json_verifier("The answer is x=3, y=-1.") == 0.0


def test_empty_object():
    assert json_verifier("{}") == 0.0


def test_extra_keys_still_valid():
    assert json_verifier('{"x": 1, "y": 2, "note": "checked"}') == 1.0
```

---

## Exercise 4 — Async sampling test

```python
"""tests/test_lesson_4_ex4.py"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

# Import from lesson file
import sys
import importlib


def _make_async_response(text: str) -> MagicMock:
    block = MagicMock(); block.type = "text"; block.text = text
    r = MagicMock(); r.content = [block]
    return r


@pytest.mark.asyncio
async def test_sample_n_returns_n_results():
    """sample_n_async should issue exactly n calls and return n results."""
    from lesson_4_self_consistency import sample_n_async  # adjust import as needed

    stub = _make_async_response("1.2")

    with patch("anthropic.AsyncAnthropic") as MockAsync:
        mock_create = AsyncMock(return_value=stub)
        MockAsync.return_value.messages.create = mock_create

        results = await sample_n_async("Q?", n=3, budget=2000)

    assert len(results) == 3
    assert mock_create.call_count == 3
```

---

## Exercise 5 — SelfConsistency class

```python
"""solutions/lesson_4_ex5.py"""
from __future__ import annotations
import asyncio
import os
import re
from collections import Counter

import anthropic


class SelfConsistency:
    def __init__(self, n: int = 5, budget: int = 4000) -> None:
        self.n = n
        self.budget = budget

    async def solve(self, prompt: str) -> str:
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        tasks = [self._call_once(client, prompt) for _ in range(self.n)]
        answers = await asyncio.gather(*tasks)
        return self._majority(list(answers))

    async def _call_once(self, client: anthropic.AsyncAnthropic, prompt: str) -> str:
        response = await client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=self.budget + 2000,
            thinking={"type": "enabled", "budget_tokens": self.budget},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    @staticmethod
    def _majority(answers: list[str]) -> str:
        nums = [m[-1] if (m := re.findall(r"-?\d+(?:\.\d+)?", a)) else None for a in answers]
        valid = [x for x in nums if x is not None]
        if not valid:
            return answers[0] if answers else ""
        return Counter(valid).most_common(1)[0][0]
```

```python
"""tests/test_lesson_4_ex5.py"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from solutions.lesson_4_ex5 import SelfConsistency


def _make_response(text: str) -> MagicMock:
    block = MagicMock(); block.type = "text"; block.text = text
    r = MagicMock(); r.content = [block]
    return r


@pytest.mark.asyncio
async def test_self_consistency_calls_n_times():
    sc = SelfConsistency(n=3, budget=2000)

    stub = _make_response("42")
    mock_create = AsyncMock(return_value=stub)

    with patch("anthropic.AsyncAnthropic") as MockAsync:
        MockAsync.return_value.messages.create = mock_create
        result = await sc.solve("What is 6x7?")

    assert mock_create.call_count == 3
    assert result == "42"
```
