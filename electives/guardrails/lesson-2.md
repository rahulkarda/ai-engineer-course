# Elective: Guardrails · Lesson 2 — Guardrails AI: RAIL Specs & Validators

> Objective: Use Guardrails AI to wrap an LLM call with built-in and custom validators via RAIL specs.  
> Time: ~60 min  
> Prereqs: [lesson-1.md](lesson-1.md), `uv add guardrails-ai`

---

## Concept

**Guardrails AI** is an open-source framework that wraps any LLM call with a
validation/correction loop. Its core primitives are:

- **RAIL spec** (Reliable AI Language) — an XML schema that describes the expected
  output structure and the validators applied to each field.
- **Guard** — a Python object built from a RAIL spec (or Pydantic model). Calling
  `guard(llm_api, prompt=...)` runs the LLM, parses the output, validates each field,
  and optionally reasks until constraints are met.
- **Validators** — Python callables that accept a value and return it unchanged (pass)
  or raise `ValidationError` / return a fixed value (fail → reask or filter).

As of Guardrails AI v0.5+, RAIL XML is largely replaced by Pydantic-based specs
(covered in Lesson 3), but understanding RAIL XML is still important for reading
existing codebases and understanding the reask mechanism.

---

## How it works internally

### The Guard loop

```
guard(llm_api, prompt=...)
    │
    ├─► 1. Call llm_api(prompt)       ← your LLM function
    │
    ├─► 2. Parse output               ← JSON extract, schema coerce
    │
    ├─► 3. Validate each field        ← run validators
    │       ├─ all pass  →  return validated output
    │       └─ any fail  →  build reask prompt, loop (up to max_reasks)
    │
    └─► 4. Return GuardOutput
            .validated_output   ← dict of validated values (or None on failure)
            .raw_llm_output     ← original LLM string
            .validation_passed  ← bool
            .error              ← str | None
```

Validators declare their failure action:
- `"reask"` — add the failure reason to context, call the LLM again
- `"filter"` — remove the offending field from output (set to `None`)
- `"fix"` — apply a corrective transform (e.g., truncate to max length)
- `"exception"` — raise immediately

### RAIL spec anatomy

```xml
<rail version="0.1">
  <output>
    <string
      name="summary"
      description="A one-sentence summary"
      validators="length: 10 200"
      on-fail-length="reask"
    />
    <integer
      name="sentiment_score"
      description="Sentiment -5 to 5"
      validators="valid-range: -5 5"
      on-fail-valid-range="fix"
    />
  </output>

  <prompt>
    Summarize the following review and give a sentiment score from -5 to 5.

    Review: {{review}}

    @complete_json_suffix_v2
  </prompt>
</rail>
```

`@complete_json_suffix_v2` is a built-in Guardrails AI template variable that appends
JSON-formatting instructions to the prompt automatically.

### Built-in validators (selection)

| Validator | Class | What it checks |
|-----------|-------|---------------|
| `length` | `guardrails.validators.ValidLength` | min/max character count |
| `valid-range` | `guardrails.validators.ValidRange` | numeric min/max |
| `valid-choices` | `guardrails.validators.ValidChoices` | allowlist |
| `no-banned-words` | from hub | deny-list filter |
| `toxic-language` | from hub | ML toxicity classifier |
| `detect-pii` | from hub | PII categories via Presidio |
| `provenance-llm` | from hub | hallucination / grounding check |

Hub validators: `guardrails hub install validator/<name>` — not covered here since we
mock LLM calls. The pattern is identical to built-ins.

---

## Worked example

```python
"""
guardrails/rail_example.py

Demonstrates Guardrails AI RAIL spec validation with a mocked LLM.
No real API calls.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import guardrails as gd
from guardrails import Guard
from guardrails.validators import ValidLength, ValidRange

logger = logging.getLogger(__name__)


# ── 1. Define the RAIL spec as a Python string ──────────────────────────────
RAIL_SPEC = """
<rail version="0.1">
  <output>
    <string
      name="summary"
      description="One-sentence summary of the product review"
      validators="length: 20 200"
      on-fail-length="reask"
    />
    <integer
      name="sentiment_score"
      description="Sentiment score from -5 (very negative) to 5 (very positive)"
      validators="valid-range: -5 5"
      on-fail-valid-range="fix"
    />
    <string
      name="category"
      description="One of: positive, neutral, negative"
      validators="valid-choices: positive neutral negative"
      on-fail-valid-choices="reask"
    />
  </output>

  <prompt>
Analyze this product review. Return a JSON with keys: summary, sentiment_score, category.

Review: {{review}}

@complete_json_suffix_v2
  </prompt>
</rail>
"""


# ── 2. A mock LLM function ───────────────────────────────────────────────────
def mock_llm(prompt: str, *args, **kwargs) -> str:
    """
    Returns a canned valid JSON response.
    In real usage, replace with:
        import anthropic, os
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        ...
    """
    logger.debug("mock_llm called (prompt length=%d)", len(prompt))
    return json.dumps({
        "summary": "The product arrived quickly and works exactly as described.",
        "sentiment_score": 4,
        "category": "positive",
    })


def mock_llm_bad_range(prompt: str, *args, **kwargs) -> str:
    """Returns a response with an out-of-range sentiment score to trigger fix."""
    return json.dumps({
        "summary": "Great product, highly recommend.",
        "sentiment_score": 99,     # out of range → fix action
        "category": "positive",
    })


# ── 3. Build and run the Guard ───────────────────────────────────────────────
def analyze_review(review: str, llm_fn=mock_llm) -> dict:
    """
    Wrap an LLM call with RAIL-based validation.

    Args:
        review: The product review text.
        llm_fn: The LLM callable (injectable for testing).

    Returns:
        Validated output dict, or raises if validation cannot be resolved.
    """
    guard = Guard.from_rail_string(RAIL_SPEC)

    result = guard(
        llm_api=llm_fn,
        prompt_params={"review": review},
        num_reasks=2,
    )

    if not result.validation_passed:
        logger.error("Validation failed after reasks: %s", result.error)
        raise ValueError(f"Guard validation failed: {result.error}")

    logger.info("Validation passed: %s", result.validated_output)
    return result.validated_output


# ── 4. Demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Case 1: valid response
    output = analyze_review(
        "Fast shipping, product works great, exactly as described!",
        llm_fn=mock_llm,
    )
    print("Valid case:", output)

    # Case 2: out-of-range score → fix applied
    output = analyze_review(
        "Amazing!",
        llm_fn=mock_llm_bad_range,
    )
    print("Fixed case:", output)
    # sentiment_score will be clamped to 5 by the fix action
```

### Corresponding pytest

```python
"""tests/test_rail_example.py"""
import json
import pytest
from guardrails import Guard
from guardrails_example.rail_example import RAIL_SPEC, analyze_review


def _make_llm(response: dict):
    """Factory: returns a mock LLM function returning a fixed JSON payload."""
    payload = json.dumps(response)
    def _llm(prompt: str, *args, **kwargs) -> str:
        return payload
    return _llm


def test_valid_response_passes():
    llm = _make_llm({
        "summary": "The product arrived quickly and works great.",
        "sentiment_score": 4,
        "category": "positive",
    })
    result = analyze_review("Great product!", llm_fn=llm)
    assert result["sentiment_score"] == 4
    assert result["category"] == "positive"
    assert len(result["summary"]) >= 20


def test_out_of_range_score_is_fixed():
    llm = _make_llm({
        "summary": "Absolutely the best thing ever made.",
        "sentiment_score": 99,  # out of [-5, 5]
        "category": "positive",
    })
    result = analyze_review("Best product!", llm_fn=llm)
    assert -5 <= result["sentiment_score"] <= 5


def test_invalid_category_triggers_reask_then_fails(monkeypatch):
    """If reasks never produce a valid category, Guard raises."""
    # Always returns invalid category even after reask
    llm = _make_llm({
        "summary": "A decent but unremarkable experience overall.",
        "sentiment_score": 1,
        "category": "INVALID_CATEGORY",
    })
    with pytest.raises(ValueError, match="Guard validation failed"):
        analyze_review("Meh product.", llm_fn=llm)
```

---

## Exercises

1. **Add a validator.** Extend `RAIL_SPEC` with a `list` field `"keywords"` (3–5 strings).
   Apply `length: 3 5` to the list. Write a mock LLM returning 2 keywords and verify the
   reask is triggered.

2. **Explore failure actions.** Change `on-fail-valid-range` from `"fix"` to `"exception"`.
   Run the bad-range test; catch the exception and log it. When would `"exception"` be
   preferable over `"fix"`?

3. **Guard introspection.** Call `guard.json_fc_import_schema` on your guard and pretty-print
   it. Explain what Guardrails AI sends to the model to instruct JSON formatting.

4. **Validator composition.** Apply two validators to the `summary` field: `length` and
   `no-banned-words` (use the `guardrails-ai` hub validator or write a simple regex stub).
   Test that a summary containing a banned word triggers the correct action.

(Solutions: [../solutions/lesson-2.md](../solutions/lesson-2.md))

---

## Self-check

- [ ] I can read a RAIL spec and describe what each validator does
- [ ] I can implement the Guard loop with a mock LLM
- [ ] I can distinguish the four failure actions and choose the right one per use-case
- [ ] I can write a pytest that injects a mock LLM to test validation logic without API calls
