# Solutions: Lesson 3 — Guardrails AI: Pydantic Output Validation & Reask

## Exercise 1: `tags: list[str]` with list length validation

```python
from typing import Annotated
from guardrails.validators import ValidLength, ValidRange
from pydantic import BaseModel, Field
from guardrails import Guard
import json, pytest


class ReviewAnalysisV2(BaseModel):
    summary: Annotated[str, ValidLength(min=20, max=300, on_fail="reask")]
    sentiment_score: Annotated[int, ValidRange(min=-5, max=5, on_fail="fix")]
    category: str
    confidence: Annotated[float, ValidRange(min=0.0, max=1.0, on_fail="fix")]
    tags: Annotated[
        list[str],
        ValidLength(min=1, max=5, on_fail="filter"),  # filter → set to None if violated
        Field(description="1–5 descriptive tags for the review"),
    ]


def test_too_many_tags_filtered():
    def llm(p, *a, **kw):
        return json.dumps({
            "summary": "Great product that arrived fast and in perfect condition.",
            "sentiment_score": 5,
            "category": "positive",
            "confidence": 0.9,
            "tags": ["fast", "great", "reliable", "cheap", "awesome", "perfect", "best"],
        })

    guard = Guard.from_pydantic(output_class=ReviewAnalysisV2)
    result = guard(llm_api=llm, num_reasks=0)
    # filter action sets field to None when list is too long
    assert result.validated_output is not None
    tags = result.validated_output.get("tags") if isinstance(result.validated_output, dict) else \
           result.validated_output.tags if result.validated_output else None
    # Either None (filtered) or within bounds
    assert tags is None or len(tags) <= 5
```

---

## Exercise 2: Pydantic `@field_validator` combined with Guardrails reask

```python
from pydantic import BaseModel, field_validator, Field
from typing import Annotated
from guardrails.validators import ValidLength, ValidRange
from guardrails import Guard
import json, pytest


class StrictReviewAnalysis(BaseModel):
    summary: Annotated[
        str,
        ValidLength(min=20, max=300, on_fail="reask"),
        Field(description="Summary — must not use the word 'perfect'"),
    ]
    sentiment_score: Annotated[int, ValidRange(min=-5, max=5, on_fail="fix")]
    category: str
    confidence: Annotated[float, ValidRange(min=0.0, max=1.0, on_fail="fix")]

    @field_validator("summary")
    @classmethod
    def no_hyperbole(cls, v: str) -> str:
        if "perfect" in v.lower():
            raise ValueError("Summary must not use the word 'perfect' (brand policy).")
        return v
```

**How Pydantic `ValidationError` interacts with Guard reask:**

Pydantic runs its validators (including `@field_validator`) during `model_validate()`.
If a `ValueError` is raised, Pydantic raises a `ValidationError`. Guardrails AI catches
`ValidationError` and treats it the same as a Guardrails `EventDetail` — it adds the
error message to the reask prompt and retries up to `num_reasks` times.

This means you can mix Pydantic field validators (for business logic) and Guardrails
validators (for format/length/range) freely. Both contribute to the reask context.

```python
def test_hyperbole_triggers_reask():
    call_count = 0
    responses = [
        json.dumps({
            "summary": "This is a perfect product with no issues whatsoever.",
            "sentiment_score": 5,
            "category": "positive",
            "confidence": 0.95,
        }),
        json.dumps({
            "summary": "An excellent product that works exactly as advertised.",
            "sentiment_score": 5,
            "category": "positive",
            "confidence": 0.95,
        }),
    ]

    def sequential_llm(prompt: str, *a, **kw) -> str:
        nonlocal call_count
        r = responses[min(call_count, 1)]
        call_count += 1
        return r

    guard = Guard.from_pydantic(output_class=StrictReviewAnalysis)
    result = guard(llm_api=sequential_llm, num_reasks=1)
    assert result.validation_passed
    assert "perfect" not in (result.validated_output.get("summary") or "").lower()
    assert call_count == 2
```

---

## Exercise 3: Token overhead comparison

```python
from guardrails import Guard
from pydantic import BaseModel, Field
from typing import Literal

class QuickAnalysis(BaseModel):
    sentiment_score: int = Field(description="Score -5 to 5")
    category: Literal["positive", "neutral", "negative"]

class FullAnalysis(BaseModel):
    summary: str = Field(description="One-sentence summary")
    sentiment_score: int = Field(description="Score -5 to 5")
    category: Literal["positive", "neutral", "negative"]
    confidence: float = Field(description="Confidence 0–1")
    tags: list[str] = Field(description="1–5 tags")

quick_guard = Guard.from_pydantic(output_class=QuickAnalysis)
full_guard  = Guard.from_pydantic(output_class=FullAnalysis)

# Inspect the JSON schema appended to the prompt
# (exact API may vary — adjust for your installed version)
print("--- Quick guard prompt suffix ---")
print(quick_guard.raw_prompt[-500:] if hasattr(quick_guard, "raw_prompt") else "N/A")
print("--- Full guard prompt suffix ---")
print(full_guard.raw_prompt[-500:] if hasattr(full_guard, "raw_prompt") else "N/A")
```

Typical overhead:
- `QuickAnalysis`: ~50–80 tokens added to prompt
- `FullAnalysis`: ~120–180 tokens added

At $0.00025/1K input tokens (Haiku): 100 tokens ≈ $0.000025 extra per call. Negligible
per call but adds up at 1M calls/month (~$25). More importantly, long schemas push short
context models closer to their limits.

---

## Exercise 4: Reask prompt security analysis

```python
# Trigger a reask and inspect the prompt sent to the LLM
import json, logging
from guardrails import Guard
from pydantic import BaseModel, Field
from typing import Annotated
from guardrails.validators import ValidLength

logging.basicConfig(level=logging.DEBUG)

class Simple(BaseModel):
    answer: Annotated[str, ValidLength(min=50, max=200, on_fail="reask")]

call_count = 0
def short_llm(prompt: str, *a, **kw) -> str:
    global call_count
    call_count += 1
    print(f"\n=== PROMPT (call {call_count}) ===\n{prompt}\n")
    return json.dumps({"answer": "too short"})  # always fails

guard = Guard.from_pydantic(output_class=Simple)
try:
    guard(llm_api=short_llm, num_reasks=1)
except Exception:
    pass
```

**Reask prompt structure:**
```
Previous response:
{"answer": "too short"}

Errors:
- answer: length must be between 50 and 200 characters; got 9 characters.

Please re-generate the JSON, correcting only the fields with errors.
```

**Security implication:** A sophisticated attacker who knows Guardrails AI is in use
could craft inputs that deliberately fail validation to force reasks, inspect the reask
prompt format via error messages, and use that structure to craft a second-stage injection.

Mitigation: sanitize error messages before including in reask prompts; avoid including
the user's original input verbatim in the reask context; rate-limit reasks per session.
