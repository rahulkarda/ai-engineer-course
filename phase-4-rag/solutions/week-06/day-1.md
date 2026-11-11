# Day 1 Solutions — Query Rewriting & Decomposition

## Exercise 1: Add `temperature` parameter to QueryRewriter

```python
import logging
import os
from dataclasses import dataclass, field

import anthropic

logger = logging.getLogger(__name__)


@dataclass
class QueryRewriter:
    model: str = "claude-sonnet-5-5"
    temperature: float = 0.7

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def rewrite(self, query: str) -> str:
        logger.debug("Rewriting query: %s (temp=%.2f)", query, self.temperature)
        message = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rewrite the following search query to be clearer and more effective "
                        "for a vector database retrieval system. Return only the rewritten query.\n\n"
                        f"Query: {query}"
                    ),
                }
            ],
        )
        rewritten = message.content[0].text.strip()
        logger.info("Query rewritten: '%s' → '%s'", query, rewritten)
        return rewritten
```

Key point: `temperature` is stored as an instance attribute and forwarded to `messages.create`.
The Anthropic SDK accepts `temperature` directly in `messages.create`.

---

## Exercise 2: Implement `multi_query_expand`

```python
import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)


def multi_query_expand(query: str, n: int = 3) -> list[str]:
    """Return N distinct phrasings of `query` for multi-query retrieval."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = (
        f"Generate exactly {n} distinct phrasings of the following search query. "
        "Each phrasing should capture the same intent but use different vocabulary or framing. "
        "Return a JSON array of strings only — no explanation.\n\n"
        f"Original query: {query}"
    )

    message = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    phrasings: list[str] = json.loads(raw)
    logger.info("Expanded '%s' into %d phrasings", query, len(phrasings))
    return phrasings[:n]
```

Usage:
```python
queries = multi_query_expand("What causes memory leaks in Python?", n=3)
# → ["What leads to memory leaks in Python applications?",
#    "How do Python programs accumulate unreleased memory?",
#    "Common sources of memory retention bugs in Python"]
```

---

## Exercise 3: Implement `decompose_query`

```python
import json
import logging
import os

import anthropic

logger = logging.getLogger(__name__)


def decompose_query(query: str) -> list[str]:
    """Break a complex query into atomic sub-questions."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = (
        "Decompose the following complex question into a list of simpler, atomic sub-questions "
        "that together answer the original. Return a JSON array of strings only.\n\n"
        f"Question: {query}"
    )

    message = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    sub_questions: list[str] = json.loads(raw)
    logger.info("Decomposed query into %d sub-questions", len(sub_questions))
    return sub_questions
```

Example:
```python
parts = decompose_query(
    "How did the 2008 financial crisis affect housing prices in major US cities?"
)
# → [
#     "What caused the 2008 financial crisis?",
#     "Which major US cities were most affected economically in 2008?",
#     "How did housing prices change nationally during 2008–2010?",
#     "What was the recovery timeline for housing prices post-2008?",
# ]
```

---

## Exercise 4: Implement `step_back`

```python
import logging
import os

import anthropic

logger = logging.getLogger(__name__)


def step_back(query: str) -> str:
    """Generalize a specific question to its broader concept (Step-Back prompting)."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = (
        "Apply Step-Back prompting: take the specific question below and rephrase it as a "
        "broader, more general question about the underlying concept or principle. "
        "Return only the generalized question.\n\n"
        f"Specific question: {query}"
    )

    message = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )

    generalized = message.content[0].text.strip()
    logger.info("Step-back: '%s' → '%s'", query, generalized)
    return generalized
```

Example:
```python
broad = step_back("Why did Apple's stock drop on 2024-03-15?")
# → "What factors generally cause a tech company's stock price to decline?"
```

---

## Exercise 5: Unit Tests for QueryRewriter

```python
# tests/unit/test_query_rewriter.py
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from day1_solutions import QueryRewriter  # adjust import to your module layout


def _make_message(text: str) -> MagicMock:
    """Build a mock anthropic.Message with a single text content block."""
    content_block = MagicMock()
    content_block.text = text
    msg = MagicMock()
    msg.content = [content_block]
    return msg


class TestQueryRewriterInit(unittest.TestCase):
    def test_default_temperature(self) -> None:
        rewriter = QueryRewriter()
        self.assertAlmostEqual(rewriter.temperature, 0.7)

    def test_custom_temperature(self) -> None:
        rewriter = QueryRewriter(temperature=0.2)
        self.assertAlmostEqual(rewriter.temperature, 0.2)

    def test_default_model(self) -> None:
        rewriter = QueryRewriter()
        self.assertEqual(rewriter.model, "claude-sonnet-5-5")


class TestQueryRewriterRewrite(unittest.TestCase):
    @patch("anthropic.Anthropic")
    def test_rewrite_returns_string(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message(
            "How do Python programs leak memory?"
        )

        rewriter = QueryRewriter()
        result = rewriter.rewrite("Python memory leak")

        self.assertEqual(result, "How do Python programs leak memory?")

    @patch("anthropic.Anthropic")
    def test_temperature_forwarded_to_api(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("rewritten")

        rewriter = QueryRewriter(temperature=0.3)
        rewriter.rewrite("some query")

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.3)

    @patch("anthropic.Anthropic")
    def test_model_forwarded_to_api(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("rewritten")

        rewriter = QueryRewriter(model="claude-opus-5-5")
        rewriter.rewrite("query")

        call_kwargs = mock_client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-opus-5-5")

    @patch("anthropic.Anthropic")
    def test_original_query_in_prompt(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("rewritten")

        rewriter = QueryRewriter()
        rewriter.rewrite("unique_test_query_xyz")

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt_content: str = call_kwargs["messages"][0]["content"]
        self.assertIn("unique_test_query_xyz", prompt_content)

    @patch("anthropic.Anthropic")
    def test_whitespace_stripped(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_message("  padded result  ")

        rewriter = QueryRewriter()
        result = rewriter.rewrite("query")
        self.assertEqual(result, "padded result")


if __name__ == "__main__":
    unittest.main()
```

Run with:
```bash
python -m pytest tests/unit/test_query_rewriter.py -v
```
