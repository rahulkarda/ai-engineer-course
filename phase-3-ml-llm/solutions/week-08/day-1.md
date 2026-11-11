# Day 1 Solutions — Prompt Engineering

## Exercise 1: System Prompt for Code-Review Assistant

A complete, specific system prompt that enforces Python style rules:

```
You are a senior Python code reviewer. Your role is to provide thorough, actionable feedback on Python code submitted for review. You enforce the following standards:

**Style and Formatting**
- PEP 8 compliance: 4-space indentation, 79-character line limit, two blank lines between top-level definitions
- Type hints required on all function signatures (parameters and return types)
- Docstrings required for all public functions, classes, and modules (Google style)
- Use f-strings for string interpolation; avoid % formatting and .format()

**Code Quality**
- No mutable default arguments (use None and set inside function body)
- Prefer list/dict/set comprehensions over imperative loops when readable
- Avoid bare `except`; catch specific exceptions
- No unused imports or variables
- Functions should do one thing; suggest splitting if a function exceeds ~30 lines

**Security**
- Flag any use of eval(), exec(), subprocess with shell=True, or pickle.load()
- Flag hardcoded secrets, credentials, or API keys
- Validate all external inputs

**Output Format**
For each issue found, respond with:
1. Severity: [CRITICAL | WARNING | SUGGESTION]
2. Location: line number or function name
3. Issue: one-sentence description
4. Fix: corrected code snippet

End your review with a short summary (2-3 sentences) of overall code quality.
If the code is clean, say so explicitly.
```

---

## Exercise 2: Zero-Shot to 3-Shot Chain-of-Thought Conversion

### Zero-Shot Version

```python
messages = [
    {
        "role": "user",
        "content": "A train travels 120 km in 1.5 hours. What is its average speed in km/h?"
    }
]
```

### 3-Shot Chain-of-Thought Version

```python
messages = [
    {
        "role": "user",
        "content": """Solve the following math problems step by step.

Example 1:
Problem: A car drives 200 km in 4 hours. What is its average speed?
Solution:
Step 1: Identify the formula. Speed = Distance / Time
Step 2: Plug in values. Speed = 200 km / 4 hours
Step 3: Calculate. Speed = 50 km/h
Answer: 50 km/h

Example 2:
Problem: If a recipe needs 2.5 cups of flour for 12 cookies, how many cups are needed for 30 cookies?
Solution:
Step 1: Find the ratio. 30 cookies / 12 cookies = 2.5
Step 2: Scale the ingredient. 2.5 cups * 2.5 = 6.25 cups
Answer: 6.25 cups

Example 3:
Problem: A store sells 3 apples for $1.20. What is the price of 7 apples?
Solution:
Step 1: Find the unit price. $1.20 / 3 = $0.40 per apple
Step 2: Multiply by desired quantity. $0.40 * 7 = $2.80
Answer: $2.80

Now solve this problem:
Problem: A train travels 120 km in 1.5 hours. What is its average speed in km/h?
Solution:"""
    }
]
```

**Why CoT works better:** The step-by-step examples teach the model to reason incrementally, reducing arithmetic errors and making intermediate logic auditable.

---

## Exercise 3: XML Delimiters

XML delimiters create unambiguous structure that the model can parse reliably, avoiding prompt injection and boundary confusion.

```python
def build_code_review_prompt(code: str, language: str, context: str) -> str:
    """Build a structured code review prompt using XML delimiters."""
    return f"""Review the following code for bugs, style issues, and security problems.

<context>
{context}
</context>

<language>{language}</language>

<code>
{code}
</code>

<instructions>
1. List each issue with severity (CRITICAL / WARNING / SUGGESTION).
2. For each issue, provide the line number and a corrected code snippet.
3. End with a one-paragraph overall assessment.
</instructions>"""
```

**Why XML delimiters:**
- The model sees `<code>...</code>` as a clearly bounded unit; injected text inside the code cannot escape to modify `<instructions>`.
- Nested structures (e.g., code that contains curly braces or quotes) do not confuse the prompt parser.
- Claude is specifically trained to respect XML-style tags as structural delimiters.

---

## Exercise 4: `build_few_shot_prompt` Implementation

```python
# few_shot.py
"""Utility for building few-shot prompt message lists for Anthropic API calls."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_few_shot_prompt(
    examples: list[dict[str, str]],
    query: str,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build a few-shot message list for the Anthropic Messages API.

    Args:
        examples: List of dicts with 'input' and 'output' keys representing
                  question/answer demonstration pairs.
        query: The actual question or task for the model to answer.
        system_prompt: Optional system-level instruction string.

    Returns:
        A list of message dicts ready to pass as the `messages` argument
        to ``anthropic.Anthropic().messages.create()``.

    Raises:
        ValueError: If examples contains dicts missing 'input' or 'output' keys,
                    or if query is empty.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    for i, ex in enumerate(examples):
        if "input" not in ex or "output" not in ex:
            raise ValueError(
                f"Example at index {i} must have 'input' and 'output' keys; "
                f"got keys: {list(ex.keys())}"
            )

    messages: list[dict[str, str]] = []

    # Interleave user/assistant turns for each demonstration
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "assistant", "content": ex["output"]})
        logger.debug("Added few-shot pair: input=%r", ex["input"][:60])

    # Final user turn with the real query
    messages.append({"role": "user", "content": query})
    logger.debug("Built few-shot prompt with %d examples", len(examples))

    return messages


def build_few_shot_prompt_with_system(
    examples: list[dict[str, str]],
    query: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Build a complete payload dict including a system prompt.

    Returns:
        Dict with 'system' and 'messages' keys, spread directly into
        ``messages.create(**payload)``.
    """
    return {
        "system": system_prompt,
        "messages": build_few_shot_prompt(examples, query),
    }
```

### Tests

```python
# tests/test_few_shot.py
"""Unit tests for build_few_shot_prompt."""

from __future__ import annotations

import pytest

from few_shot import build_few_shot_prompt, build_few_shot_prompt_with_system


EXAMPLES = [
    {"input": "What is 2 + 2?", "output": "4"},
    {"input": "What is 3 * 3?", "output": "9"},
]


class TestBuildFewShotPrompt:
    def test_empty_examples_returns_single_user_message(self) -> None:
        result = build_few_shot_prompt([], "What is 5 + 5?")
        assert result == [{"role": "user", "content": "What is 5 + 5?"}]

    def test_alternating_roles(self) -> None:
        result = build_few_shot_prompt(EXAMPLES, "What is 4 + 4?")
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "user", "assistant", "user"]

    def test_last_message_is_query(self) -> None:
        result = build_few_shot_prompt(EXAMPLES, "What is 4 + 4?")
        assert result[-1] == {"role": "user", "content": "What is 4 + 4?"}

    def test_example_content_preserved(self) -> None:
        result = build_few_shot_prompt(EXAMPLES, "query")
        assert result[0]["content"] == "What is 2 + 2?"
        assert result[1]["content"] == "4"
        assert result[2]["content"] == "What is 3 * 3?"
        assert result[3]["content"] == "9"

    def test_single_example(self) -> None:
        result = build_few_shot_prompt([EXAMPLES[0]], "q")
        assert len(result) == 3

    def test_raises_on_empty_query(self) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            build_few_shot_prompt(EXAMPLES, "   ")

    def test_raises_on_missing_input_key(self) -> None:
        bad = [{"input": "q"}]  # missing 'output'
        with pytest.raises(ValueError, match="output"):
            build_few_shot_prompt(bad, "query")

    def test_raises_on_missing_output_key(self) -> None:
        bad = [{"output": "a"}]  # missing 'input'
        with pytest.raises(ValueError, match="input"):
            build_few_shot_prompt(bad, "query")


class TestBuildFewShotPromptWithSystem:
    def test_returns_system_and_messages(self) -> None:
        payload = build_few_shot_prompt_with_system(
            EXAMPLES, "What is 4 + 4?", system_prompt="You are a math tutor."
        )
        assert "system" in payload
        assert "messages" in payload
        assert payload["system"] == "You are a math tutor."

    def test_messages_structure_matches_standalone(self) -> None:
        standalone = build_few_shot_prompt(EXAMPLES, "q")
        with_system = build_few_shot_prompt_with_system(EXAMPLES, "q", "sys")
        assert with_system["messages"] == standalone
```
