# Phase 3 · Week 08 · Day 2 — Structured Outputs

> Objective: Reliably extract typed, validated data from LLM responses using Pydantic v2 and the Anthropic SDK.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

---

## Concept

LLMs produce free-form text. Most production systems need structured data — a dict, a typed object, a database row. The gap between "a string that looks like JSON" and "a validated Python object" is where bugs live.

Structured output is the practice of instructing the model to emit a specific data format (usually JSON), then parsing and validating that output against a schema before any downstream code touches it.

Three reasons this matters:

1. **Downstream reliability.** Code that processes LLM output shouldn't need `try/except KeyError` on every field access. A validated Pydantic model guarantees fields exist and have the right types.
2. **Prompt as contract.** Embedding the JSON schema in the prompt makes the expected shape explicit — both for the model and for future maintainers.
3. **Failure surface is narrow.** Validation errors happen at the boundary (the `parse_llm_json` function), not scattered through business logic.

Pydantic v2 is the right tool here: `model_json_schema()` generates a JSON Schema dict you can embed directly in the prompt, and `model_validate_json()` parses and validates a JSON string in one call.

---

## How it works internally

**Why LLMs can emit valid JSON.** JSON syntax is heavily represented in training data (APIs, config files, data dumps). The model has learned strong statistical associations between a JSON-instructing prompt and well-formed JSON output. It is not guaranteed — the model can still hallucinate extra fields, use wrong types, or close brackets incorrectly — which is exactly why validation is mandatory.

**Schema-in-prompt pattern.** Embedding the JSON Schema in the system prompt works because:
- It constrains the token distribution: field names, type strings, and example values from the schema become high-probability next tokens.
- It removes ambiguity: the model doesn't have to infer what `confidence` means if the schema says `"type": "number", "minimum": 0, "maximum": 1`.

**`model_json_schema()` vs manual schema writing.** Pydantic v2's `model_json_schema()` produces a standards-compliant JSON Schema from your model class. Using it as the source of truth means the prompt schema and the validation schema are always in sync — one change to the Pydantic model updates both.

**Parsing pipeline.**
```
LLM response (str)
  → strip markdown fences if present
  → json.loads()           ← raises json.JSONDecodeError on malformed JSON
  → Model.model_validate() ← raises pydantic.ValidationError on schema mismatch
  → typed Python object    ← safe to use
```

**Retry vs repair.** Two strategies for validation failures:
- **Retry**: re-send the original prompt (simple, works if the failure was stochastic).
- **Repair**: send the invalid JSON back to the model with the validation error message and ask it to fix the specific problem. Repair is more token-efficient for deterministic errors (wrong type, missing required field).

---

## Worked example

```python
"""
week-08/day-2-structured-outputs.py

Extract structured article summaries from free text using Claude
and validate the output with Pydantic v2.

Requirements:
    pip install anthropic pydantic
    export ANTHROPIC_API_KEY="..."
"""

import json
import logging
import os
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
MAX_TOKENS = 1024
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ArticleSummary(BaseModel):
    """Structured summary of a news or blog article."""

    title: str = Field(description="The article's headline or inferred title.")
    key_points: list[str] = Field(
        description="3–5 bullet-point key takeaways from the article.",
        min_length=1,
        max_length=5,
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the article."
    )
    confidence: float = Field(
        description="Model's confidence in the sentiment classification, 0.0–1.0.",
        ge=0.0,
        le=1.0,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_extraction_system_prompt(schema: dict[str, Any]) -> str:
    """
    Build a system prompt that embeds the JSON Schema so the model
    knows the exact structure required.
    """
    schema_str = json.dumps(schema, indent=2)
    return (
        "You are a precise information extraction assistant. "
        "Given an article, extract a structured summary.\n\n"
        "You MUST respond with a single JSON object that strictly conforms "
        "to the following JSON Schema. Output only the JSON — no markdown "
        "fences, no explanation, no additional text.\n\n"
        f"Schema:\n{schema_str}"
    )


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_llm_json(response_text: str, model: type[BaseModel]) -> BaseModel:
    """
    Parse and validate a JSON string from an LLM response.

    Strips markdown code fences before parsing in case the model
    wraps the JSON in a ```json ... ``` block.

    Raises:
        json.JSONDecodeError: response is not valid JSON.
        pydantic.ValidationError: JSON does not match the model schema.
    """
    text = response_text.strip()

    # Strip optional markdown fence
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line and the closing ``` line
        inner = [
            line for line in lines[1:]
            if line.strip() != "```"
        ]
        text = "\n".join(inner).strip()

    return model.model_validate_json(text)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_article_summary(
    text: str,
    *,
    client: anthropic.Anthropic | None = None,
) -> ArticleSummary:
    """
    Extract a structured ArticleSummary from article text.

    Retries up to MAX_RETRIES times on validation failure before raising.

    Args:
        text: The raw article text to summarise.
        client: Optional pre-constructed Anthropic client (useful in tests).

    Returns:
        A validated ArticleSummary instance.

    Raises:
        EnvironmentError: ANTHROPIC_API_KEY not set.
        ValueError: LLM response failed validation after all retries.
    """
    if client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set.")
        client = anthropic.Anthropic(api_key=api_key)

    schema = ArticleSummary.model_json_schema()
    system_prompt = build_extraction_system_prompt(schema)

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("Extraction attempt %d/%d.", attempt, MAX_RETRIES)

        message = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<article>\n{text}\n</article>\n\n"
                        "Extract the structured summary as JSON."
                    ),
                }
            ],
        )

        raw = message.content[0].text
        logger.debug("Raw LLM response:\n%s", raw)

        try:
            result = parse_llm_json(raw, ArticleSummary)
            logger.info("Extraction succeeded on attempt %d.", attempt)
            return result  # type: ignore[return-value]
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Attempt %d failed validation: %s", attempt, exc)
            last_error = exc

    raise ValueError(
        f"Failed to extract a valid ArticleSummary after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE = """
Renewable Energy Investments Hit Record High in 2024

Global investment in renewable energy reached $1.7 trillion in 2024,
surpassing fossil fuel investment for the first time in history, according
to a report released Tuesday by the International Energy Agency.

Solar power accounted for the largest share at $680 billion, driven by
falling panel costs and favorable government incentives in China, the EU,
and the United States. Wind energy attracted $320 billion in new capital,
while battery storage investment doubled year-over-year to $150 billion.

Analysts described the shift as a structural turning point. "We are past
the tipping point," said IEA Director Fatih Birol. "Clean energy is now
the default choice for new power generation in most of the world."

The report also noted that emerging economies contributed 40 percent of
total renewable investment, up from 25 percent in 2020, suggesting the
transition is no longer confined to wealthy nations.
"""


def main() -> None:
    summary = extract_article_summary(SAMPLE_ARTICLE)
    print(f"Title:      {summary.title}")
    print(f"Sentiment:  {summary.sentiment}  (confidence: {summary.confidence:.2f})")
    print("Key points:")
    for point in summary.key_points:
        print(f"  - {point}")


if __name__ == "__main__":
    main()
```

---

### Unit test (mocked — no real API calls)

```python
"""
tests/unit/test_day2_structured_outputs.py

All Anthropic API calls are mocked. No network required.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from week_08.day2_structured_outputs import (
    ArticleSummary,
    extract_article_summary,
    parse_llm_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_client(response_text: str) -> MagicMock:
    """Return a mock anthropic.Anthropic client whose messages.create()
    returns a response with a single text content block."""
    content_block = MagicMock()
    content_block.text = response_text

    message = MagicMock()
    message.content = [content_block]

    client = MagicMock()
    client.messages.create.return_value = message
    return client


VALID_JSON = json.dumps(
    {
        "title": "Renewable Energy Investments Hit Record High",
        "key_points": [
            "Global renewables investment reached $1.7T in 2024.",
            "Solar accounted for $680B, the largest share.",
            "Emerging economies contributed 40% of investment.",
        ],
        "sentiment": "positive",
        "confidence": 0.92,
    }
)


# ---------------------------------------------------------------------------
# parse_llm_json tests
# ---------------------------------------------------------------------------

class TestParseLlmJson:
    def test_valid_json_parses_correctly(self) -> None:
        result = parse_llm_json(VALID_JSON, ArticleSummary)
        assert isinstance(result, ArticleSummary)
        assert result.sentiment == "positive"
        assert result.confidence == pytest.approx(0.92)

    def test_strips_markdown_fence(self) -> None:
        fenced = f"```json\n{VALID_JSON}\n```"
        result = parse_llm_json(fenced, ArticleSummary)
        assert isinstance(result, ArticleSummary)

    def test_invalid_json_raises(self) -> None:
        import json
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("not json at all", ArticleSummary)

    def test_schema_mismatch_raises(self) -> None:
        from pydantic import ValidationError
        bad = json.dumps({"title": "ok", "sentiment": "unknown_value", "confidence": 0.5})
        with pytest.raises(ValidationError):
            parse_llm_json(bad, ArticleSummary)


# ---------------------------------------------------------------------------
# extract_article_summary tests
# ---------------------------------------------------------------------------

class TestExtractArticleSummary:
    def test_successful_extraction(self) -> None:
        client = make_mock_client(VALID_JSON)
        result = extract_article_summary("Some article text.", client=client)

        assert isinstance(result, ArticleSummary)
        assert result.title == "Renewable Energy Investments Hit Record High"
        client.messages.create.assert_called_once()

    def test_retries_on_validation_error_then_succeeds(self) -> None:
        """First response is malformed JSON; second is valid."""
        content_bad = MagicMock()
        content_bad.text = "{ broken json"

        content_good = MagicMock()
        content_good.text = VALID_JSON

        msg_bad = MagicMock()
        msg_bad.content = [content_bad]

        msg_good = MagicMock()
        msg_good.content = [content_good]

        client = MagicMock()
        client.messages.create.side_effect = [msg_bad, msg_good]

        result = extract_article_summary("Article.", client=client)
        assert isinstance(result, ArticleSummary)
        assert client.messages.create.call_count == 2

    def test_raises_after_max_retries(self) -> None:
        client = make_mock_client("{ always broken")
        with pytest.raises(ValueError, match="Failed to extract"):
            extract_article_summary("Article.", client=client)
```

---

## Exercises

1. Define a Pydantic v2 model `JobPosting` for extracting structured data from a job advertisement. Include at minimum: `title: str`, `company: str`, `required_skills: list[str]`, `salary_range: tuple[int, int] | None`, `remote_allowed: bool`. Add `Field(description=...)` annotations to every field so the generated JSON Schema is self-documenting.

2. Write a system prompt function `build_job_extraction_prompt(schema: dict) -> str` that embeds the `JobPosting` JSON Schema and instructs the model to return only valid JSON. The prompt should include an explicit instruction to use `null` for optional fields when the information is not present in the source text.

3. Implement `parse_llm_json(response: str, model: type[BaseModel]) -> BaseModel` as a standalone utility function. It must:
   - Strip markdown fences before parsing
   - Raise `json.JSONDecodeError` for malformed JSON (do not convert to a different exception)
   - Raise `pydantic.ValidationError` for schema mismatches (do not convert)
   - Have a docstring and full type hints
   - Be importable from a `utils.py` module

4. Write a unit test file for your `extract_article_summary`-equivalent function that:
   - Uses `unittest.mock.MagicMock` to mock `anthropic.Anthropic` — no real API calls
   - Covers the happy path (valid JSON returned on first attempt)
   - Covers the retry path (first attempt fails, second succeeds)
   - Covers the exhausted-retry path (all attempts fail, `ValueError` raised)
   - Uses `pytest.raises` for exception assertions

   (Solutions: ../solutions/week-08/day-2.md)

---

## Self-check

- [ ] I can explain why embedding the JSON Schema in the prompt improves output reliability compared to natural-language format instructions
- [ ] I can implement `parse_llm_json` without reference, including markdown fence stripping
- [ ] I can use `model_json_schema()` to generate a JSON Schema from a Pydantic v2 model and embed it in a prompt string
- [ ] I can describe the difference between the retry and repair strategies for validation failures and when to use each
- [ ] I can write a unit test for an LLM extraction function that mocks the Anthropic client and asserts on the parsed output type
