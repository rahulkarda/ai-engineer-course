# Solutions: Lesson 2 — Guardrails AI: RAIL Specs & Validators

## Exercise 1: Add a `keywords` list field with length validation

```python
RAIL_SPEC_WITH_KEYWORDS = """
<rail version="0.1">
  <output>
    <string
      name="summary"
      description="One-sentence summary"
      validators="length: 20 200"
      on-fail-length="reask"
    />
    <integer
      name="sentiment_score"
      validators="valid-range: -5 5"
      on-fail-valid-range="fix"
    />
    <string
      name="category"
      validators="valid-choices: positive neutral negative"
      on-fail-valid-choices="reask"
    />
    <list
      name="keywords"
      description="3 to 5 keywords from the review"
      validators="length: 3 5"
      on-fail-length="reask"
    />
  </output>

  <prompt>
Analyze this review. Return a JSON with keys: summary, sentiment_score, category, keywords.
Review: {{review}}
@complete_json_suffix_v2
  </prompt>
</rail>
"""

import json, pytest
from guardrails import Guard


def test_too_few_keywords_triggers_reask():
    call_count = 0
    responses = [
        # First call: only 2 keywords (fails length min=3)
        json.dumps({
            "summary": "Good product that works as expected every time.",
            "sentiment_score": 3,
            "category": "positive",
            "keywords": ["fast", "good"],            # only 2 → reask
        }),
        # Second call (reask): 4 keywords (passes)
        json.dumps({
            "summary": "Fast shipping and product matched description perfectly.",
            "sentiment_score": 4,
            "category": "positive",
            "keywords": ["fast", "good", "reliable", "shipping"],
        }),
    ]

    def sequential_llm(prompt: str, *a, **kw) -> str:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    guard = Guard.from_rail_string(RAIL_SPEC_WITH_KEYWORDS)
    result = guard(llm_api=sequential_llm, prompt_params={"review": "test"}, num_reasks=1)

    assert result.validation_passed
    assert 3 <= len(result.validated_output["keywords"]) <= 5
    assert call_count == 2  # original + one reask
```

---

## Exercise 2: `on-fail-valid-range="exception"` vs `"fix"`

```python
RAIL_EXCEPTION = """
<rail version="0.1">
  <output>
    <integer
      name="sentiment_score"
      validators="valid-range: -5 5"
      on-fail-valid-range="exception"
    />
  </output>
  <prompt>Rate sentiment of: {{review}} @complete_json_suffix_v2</prompt>
</rail>
"""

import logging
import json
import pytest
from guardrails import Guard

logger = logging.getLogger(__name__)


def test_exception_on_out_of_range():
    def bad_llm(p, *a, **kw):
        return json.dumps({"sentiment_score": 100})

    guard = Guard.from_rail_string(RAIL_EXCEPTION)
    with pytest.raises(Exception) as exc_info:
        guard(llm_api=bad_llm, prompt_params={"review": "test"}, num_reasks=0)
    logger.warning("Validation exception caught: %s", exc_info.value)
```

**When to prefer `"exception"` over `"fix"`:**
- When out-of-range values indicate a fundamental model failure (not just noise)
- When the fix would be semantically meaningless (e.g., clamping a score to 5 when the
  model returned 100 means it didn't understand the task)
- In batch/pipeline processing where you want to fail fast and alert rather than silently
  correct potentially misleading data

---

## Exercise 3: Guard schema introspection

```python
from guardrails import Guard
from guardrails_example.rail_example import RAIL_SPEC
import json

guard = Guard.from_rail_string(RAIL_SPEC)

# The prompt template with JSON suffix appended
print(guard.raw_prompt)

# The JSON schema that Guardrails AI instructs the model to follow
# (exact attribute name may vary by version)
try:
    schema = guard.output_schema.to_dict()
    print(json.dumps(schema, indent=2))
except AttributeError:
    # Fallback: inspect the guard's internal RAIL element
    print(guard.rail.output_schema)
```

The JSON schema shows every field name, type, and description. Guardrails AI appends
this to the prompt so the model knows what structure to produce. This is why model-output
instructions add tokens to every call — budget 100–300 extra tokens per Guard call.

---

## Exercise 4: Composed validators (length + custom deny-list)

```python
"""A simple stub no-banned-words validator for environments without hub access."""
import re
from guardrails.validators import EventDetail, Validator, register_validator


@register_validator(name="no-banned-words-stub", data_type="string")
class NoBannedWordsStub(Validator):
    def __init__(self, banned: list[str], on_fail="filter"):
        super().__init__(on_fail=on_fail)
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in banned) + r")\b", re.I
        )

    def validate(self, value: str, metadata: dict) -> str:
        if m := self._pattern.search(value):
            raise EventDetail(
                key="no-banned-words-stub",
                value=value,
                error_message=f"Banned word found: {m.group()}",
                fix_value=self._pattern.sub("[REMOVED]", value),
                metadata=metadata,
            )
        return value


# RAIL spec using both validators on summary
RAIL_DUAL = """
<rail version="0.1">
  <output>
    <string
      name="summary"
      validators="length: 20 200 no-banned-words-stub: ['awful', 'terrible', 'horrible']"
      on-fail-length="reask"
      on-fail-no-banned-words-stub="filter"
    />
  </output>
  <prompt>Summarize: {{review}} @complete_json_suffix_v2</prompt>
</rail>
"""

# Test
import json
from guardrails import Guard

def test_banned_word_filtered():
    def bad_llm(p, *a, **kw):
        return json.dumps({"summary": "This is an awful product with no redeeming qualities."})

    guard = Guard.from_rail_string(RAIL_DUAL)
    result = guard(llm_api=bad_llm, prompt_params={"review": "test"}, num_reasks=0)
    # filter action → field set to None
    assert result.validated_output.get("summary") is None or \
           "awful" not in (result.validated_output.get("summary") or "")
```
