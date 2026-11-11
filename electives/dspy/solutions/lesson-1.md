# Solutions — Lesson 1: DSPy Philosophy & Signatures

---

## Exercise 1 — Inline signature shorthand

```python
# String shorthand: "inputs -> outputs"
summarizer = dspy.Predict("text -> summary")
result = summarizer(text="DSPy is a framework for ...")
print(result.summary)
```

DSPy parses the string and creates the same Signature structure as the class form.
No `desc=` is possible with the shorthand — the class form is preferred for production
code.

---

## Exercise 2 — Multi-output signature

```python
import dspy

class AnalyzeReview(dspy.Signature):
    """Analyze a product review for sentiment and confidence."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="exactly one of: positive, negative, neutral")
    confidence: str = dspy.OutputField(desc="exactly one of: high, medium, low")

analyzer = dspy.Predict(AnalyzeReview)
result = analyzer(review="This product is decent but nothing special.")
print(result.sentiment, result.confidence)
```

---

## Exercise 3 — Field descriptions matter

Without `desc=`, the model is free to return any format. With a strict description,
the output conforms to the expected structure.

```python
# Inspect raw prompts sent to the LM
dspy.inspect_history(n=2)
```

`inspect_history` prints the last N completed LM interactions: the full prompt sent
and the raw completion returned. Use it to verify that the description was injected
into the prompt template.

---

## Exercise 4 — Error handling

```python
import os
import dspy

def run_demo(text: str) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — cannot initialize "
            "anthropic/claude-haiku-5-5"
        )
    lm = dspy.LM(model="anthropic/claude-haiku-5-5", api_key=api_key, max_tokens=512)
    dspy.configure(lm=lm)
    # ... rest of the demo
```

Checking before `dspy.LM()` gives a clear message rather than a cryptic SDK error.

---

## Exercise 5 — Unit test with mocked Predict

```python
"""test_lesson1.py"""
from unittest.mock import MagicMock, patch

import dspy
import pytest


class AnalyzeReview(dspy.Signature):
    """Analyze a product review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField()
    confidence: str = dspy.OutputField()


def test_analyze_review_fields() -> None:
    """AnalyzeReview returns both output fields without real LM call."""
    mock_prediction = MagicMock()
    mock_prediction.sentiment = "positive"
    mock_prediction.confidence = "high"

    with patch.object(dspy.Predict, "__call__", return_value=mock_prediction):
        predictor = dspy.Predict(AnalyzeReview)
        result = predictor(review="Great product!")

    assert result.sentiment == "positive"
    assert result.confidence == "high"
```

Key points:
- `patch.object(dspy.Predict, "__call__", ...)` intercepts the LM call at the module
  boundary — no API key needed.
- Always return a `MagicMock` with the expected field names set.
