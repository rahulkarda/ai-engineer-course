# Solutions — Lesson 3: Metrics & Evaluation

---

## Exercise 1 — Token F1 metric and extraction dataset

```python
import dspy
from typing import Any


def token_f1_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float:
    gold = set(example.answer.lower().split())
    predicted = set(pred.answer.lower().split())
    if not gold or not predicted:
        return 0.0
    precision = len(gold & predicted) / len(predicted)
    recall = len(gold & predicted) / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


extraction_dataset = [
    dspy.Example(
        text="The United Nations was founded in 1945 after World War II.",
        answer="United Nations 1945",
    ).with_inputs("text"),
    dspy.Example(
        text="Python was created by Guido van Rossum.",
        answer="Python Guido van Rossum",
    ).with_inputs("text"),
    dspy.Example(
        text="Mount Everest is the highest peak at 8849 metres.",
        answer="Mount Everest 8849 metres",
    ).with_inputs("text"),
    dspy.Example(
        text="The speed of light is approximately 299792 km/s.",
        answer="speed light 299792",
    ).with_inputs("text"),
    dspy.Example(
        text="Shakespeare wrote Hamlet around 1600.",
        answer="Shakespeare Hamlet 1600",
    ).with_inputs("text"),
]
```

Run evaluation:

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=extraction_dataset, metric=token_f1_metric, num_threads=1)
# score = evaluator(program)
```

---

## Exercise 2 — Length-penalized metric

```python
def exact_match_with_length_penalty(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> bool:
    if len(pred.sentiment.strip().split()) > 1:
        return False
    return pred.sentiment.strip().lower() == example.sentiment.strip().lower()
```

Effect: multi-word outputs like "very positive" score 0 even if semantically correct.
This forces the model to produce single-word labels — but reduces overall score since
the model occasionally adds qualifiers.

---

## Exercise 3 — LLM-as-judge unit test

```python
"""test_llm_judge.py"""
from unittest.mock import MagicMock, patch

import dspy
import pytest


def llm_judge(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    class _JudgeSig(dspy.Signature):
        """Rate answer quality 0–1."""
        question: str = dspy.InputField()
        answer: str = dspy.InputField()
        score: float = dspy.OutputField(desc="float between 0 and 1")

    judge = dspy.Predict(_JudgeSig)
    result = judge(question=example.question, answer=pred.answer)
    try:
        return float(result.score)
    except (ValueError, TypeError):
        return 0.0


def test_llm_judge_returns_float() -> None:
    mock_result = MagicMock()
    mock_result.score = "0.9"

    example = dspy.Example(question="What is 2+2?", answer="4").with_inputs("question")
    pred = MagicMock()
    pred.answer = "4"

    with patch.object(dspy.Predict, "__call__", return_value=mock_result):
        score = llm_judge(example, pred)

    assert score == pytest.approx(0.9)


def test_llm_judge_handles_invalid_score() -> None:
    mock_result = MagicMock()
    mock_result.score = "not_a_number"

    example = dspy.Example(question="Q?", answer="A").with_inputs("question")
    pred = MagicMock()
    pred.answer = "A"

    with patch.object(dspy.Predict, "__call__", return_value=mock_result):
        score = llm_judge(example, pred)

    assert score == 0.0
```

---

## Exercise 4 — Composite metric

```python
def composite_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float:
    correct = pred.sentiment.strip().lower() == example.sentiment.strip().lower()
    brief = len(pred.sentiment.strip().split()) == 1
    return 0.8 * float(correct) + 0.2 * float(brief)
```

---

## Exercise 5 — Train/dev split explanation

```python
def build_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    # ... (same as lesson)
    return examples[:6], examples[6:]   # trainset, devset
```

**Why disjoint?** The optimizer sees training examples during compilation; it may
directly encode them as demonstrations. Evaluating on those same examples measures
memorization, not generalization. A disjoint dev set gives an unbiased estimate of how
the compiled program will perform on new, unseen inputs.
