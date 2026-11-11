# Elective: Guardrails · Lesson 3 — Guardrails AI: Pydantic Output Validation & Reask

> Objective: Replace RAIL XML with Pydantic models as the Guard spec; add field-level validators and reask logic.  
> Time: ~60 min  
> Prereqs: [lesson-2.md](lesson-2.md), Pydantic v2

---

## Concept

Guardrails AI v0.5+ treats a **Pydantic model as a first-class spec**. You annotate
fields with `Annotated[..., validator]` and the Guard builds the same validation/reask
loop automatically — but with full type safety and IDE support.

Key advantages over RAIL XML:
- Standard Python typing; no XML parsing
- Validators are just Python callables or `guardrails.validators.*` instances
- Composable: validators chain with `Annotated[T, v1, v2]`
- Works with `BaseModel` subclasses directly — no string-based spec

---

## How it works internally

### Pydantic Guard construction

```python
from guardrails import Guard
from pydantic import BaseModel

class MyOutput(BaseModel):
    summary: str
    score: int

guard = Guard.from_pydantic(output_class=MyOutput)
```

Internally:
1. Guardrails introspects `MyOutput`'s fields and annotations
2. Builds a JSON schema from the model
3. Appends schema-aware formatting instructions to the prompt
4. After LLM response: `model_validate(parsed_json)` — Pydantic validates types
5. Custom Guardrails validators (from `Annotated`) run after Pydantic validation
6. Failures trigger the reask loop, injecting error context into the next prompt

### Reask mechanics

On a field failure Guardrails constructs a **reask prompt** that includes:
- The original prompt
- The LLM's previous (invalid) response
- A description of what failed and why

```
Previous response had errors:
- summary: length must be between 20 and 200 characters; got 8.
Please regenerate, fixing only the flagged fields.
```

The model sees its own mistake and (usually) corrects it. `num_reasks=N` caps the loop.

### Field-level validators with `Annotated`

```python
from typing import Annotated
from guardrails.validators import ValidLength, ValidRange

class Output(BaseModel):
    summary: Annotated[str, ValidLength(min=20, max=200, on_fail="reask")]
    score:   Annotated[int, ValidRange(min=-5, max=5, on_fail="fix")]
```

---

## Worked example

```python
"""
guardrails/pydantic_guard.py

Guardrails AI with Pydantic models. No real LLM calls.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Literal

import anthropic
from guardrails import Guard
from guardrails.validators import ValidLength, ValidRange
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── 1. Pydantic output model with inline validators ──────────────────────────
class ReviewAnalysis(BaseModel):
    """Structured output from review analysis."""

    summary: Annotated[
        str,
        ValidLength(min=20, max=300, on_fail="reask"),
        Field(description="One-to-two sentence summary of the review"),
    ]
    sentiment_score: Annotated[
        int,
        ValidRange(min=-5, max=5, on_fail="fix"),
        Field(description="Sentiment score: -5 (very negative) to 5 (very positive)"),
    ]
    category: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment category"
    )
    confidence: Annotated[
        float,
        ValidRange(min=0.0, max=1.0, on_fail="fix"),
        Field(description="Model confidence 0–1"),
    ]


# ── 2. LLM caller (real Anthropic, injectable for tests) ────────────────────
def anthropic_llm(prompt: str, *args, **kwargs) -> str:
    """
    Real LLM call via Anthropic.
    api_key read from ANTHROPIC_API_KEY env var.
    Replace with mock in tests — do NOT call in CI.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── 3. Guard construction ────────────────────────────────────────────────────
def build_guard() -> Guard:
    return Guard.from_pydantic(
        output_class=ReviewAnalysis,
        prompt="""
Analyze the following product review. Return a JSON object with these fields:
- summary: a one-to-two sentence summary
- sentiment_score: integer from -5 to 5
- category: "positive", "neutral", or "negative"
- confidence: float from 0.0 to 1.0 indicating your certainty

Review: ${review}

${gr.complete_json_suffix_v2}
""",
    )


# ── 4. Analysis function ─────────────────────────────────────────────────────
def analyze_review(
    review: str,
    llm_fn=anthropic_llm,
    num_reasks: int = 2,
) -> ReviewAnalysis:
    """
    Analyze a product review using Guardrails AI + Pydantic validation.

    Args:
        review: Raw review text.
        llm_fn: LLM callable (defaults to real Anthropic; override in tests).
        num_reasks: Max correction attempts.

    Returns:
        Validated ReviewAnalysis instance.

    Raises:
        ValueError: If validation fails after all reasks.
    """
    guard = build_guard()
    result = guard(
        llm_api=llm_fn,
        prompt_params={"review": review},
        num_reasks=num_reasks,
    )

    if not result.validation_passed or result.validated_output is None:
        raise ValueError(f"Validation failed: {result.error}")

    validated: ReviewAnalysis = result.validated_output
    logger.info(
        "Review analyzed: score=%d category=%s",
        validated.sentiment_score,
        validated.category,
    )
    return validated


# ── 5. Smoke test (no real API) ───────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def _mock_llm(prompt: str, *args, **kwargs) -> str:
        return json.dumps({
            "summary": "The item shipped fast and matched the product description exactly.",
            "sentiment_score": 4,
            "category": "positive",
            "confidence": 0.92,
        })

    result = analyze_review("Fast shipping, exactly as described!", llm_fn=_mock_llm)
    print(f"Result: {result.model_dump_json(indent=2)}")
```

### Pydantic guard tests

```python
"""tests/test_pydantic_guard.py"""
from __future__ import annotations

import json
import pytest
from guardrails_example.pydantic_guard import ReviewAnalysis, analyze_review


def _llm(response: dict):
    payload = json.dumps(response)
    def _fn(prompt: str, *a, **kw) -> str:
        return payload
    return _fn


def test_valid_response_returns_model():
    llm = _llm({
        "summary": "Product arrived on time and quality is excellent.",
        "sentiment_score": 5,
        "category": "positive",
        "confidence": 0.95,
    })
    result = analyze_review("Best purchase ever!", llm_fn=llm)
    assert isinstance(result, ReviewAnalysis)
    assert result.category == "positive"
    assert 0.0 <= result.confidence <= 1.0


def test_score_clamped_by_fix():
    llm = _llm({
        "summary": "Very surprising product, totally exceeded expectations indeed.",
        "sentiment_score": 100,  # way out of range
        "category": "positive",
        "confidence": 0.8,
    })
    result = analyze_review("Surprised!", llm_fn=llm)
    assert result.sentiment_score <= 5


def test_invalid_confidence_clamped():
    llm = _llm({
        "summary": "Decent product, nothing special but meets the basic needs.",
        "sentiment_score": 1,
        "category": "neutral",
        "confidence": 2.5,  # > 1.0
    })
    result = analyze_review("It's okay.", llm_fn=llm)
    assert result.confidence <= 1.0


def test_short_summary_triggers_reask_then_fails():
    """Summary < 20 chars triggers reask; if still short after reasks, raise."""
    call_count = 0

    def _always_short(prompt: str, *a, **kw) -> str:
        nonlocal call_count
        call_count += 1
        return json.dumps({
            "summary": "Too short.",    # 10 chars — always fails
            "sentiment_score": 0,
            "category": "neutral",
            "confidence": 0.5,
        })

    with pytest.raises(ValueError, match="Validation failed"):
        analyze_review("Test.", llm_fn=_always_short, num_reasks=1)

    # Should have called LLM at least twice (original + reask)
    assert call_count >= 2
```

---

## Exercises

1. **Nested model.** Add a `tags: list[str]` field to `ReviewAnalysis` with
   `ValidLength(min=1, max=5)` on the list. Write a test with a mock returning 7 tags
   and verify the fix/filter behavior.

2. **Custom Pydantic validator.** Use `@field_validator` (Pydantic v2 syntax) to reject
   summaries that contain the literal word "perfect" (too hyperbolic for your brand).
   Combine it with the Guardrails reask mechanism. How does a Pydantic ValidationError
   interact with the Guard reask loop?

3. **Multiple models.** Create a second Pydantic model `QuickAnalysis` with only
   `sentiment_score` and `category`. Wrap both in separate Guards. Compare the
   approximate token overhead each adds to the prompt (hint: inspect the JSON schema
   appended to the prompt via `guard.raw_prompt`).

4. **Reask prompt inspection.** Trigger a reask deliberately, then print
   `result.reask_prompts` (if available in your version). What extra context does
   Guardrails inject? How could a malicious user exploit knowledge of this format?

(Solutions: [../solutions/lesson-3.md](../solutions/lesson-3.md))

---

## Self-check

- [ ] I can build a `Guard.from_pydantic` with field-level validators
- [ ] I can write `Annotated[T, Validator(..., on_fail="...")]` for any validator
- [ ] I can implement a mock LLM to test validation logic without API calls
- [ ] I understand how reask prompts are constructed and their security implications
