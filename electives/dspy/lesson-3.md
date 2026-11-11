# Lesson 3 — Metrics & Evaluation

> Objective: Write a metric function and run a DSPy evaluation loop.
> Time: ~45 min · Prereqs: lesson-2.md

---

## Concept

An optimizer is only as good as the signal it optimizes toward. In DSPy that signal is
a **metric** — a Python function `(example, prediction, trace) -> float | bool` that
scores one prediction.

Getting the metric right is the hardest part of DSPy. A loose metric produces
over-fitted programs; a too-strict metric makes the optimizer give up.

### What a metric does

```
metric(example, prediction, trace) → score ∈ [0, 1]
```

- `example` — one item from your dataset (`dspy.Example`)
- `prediction` — the module's output for that example
- `trace` — internal execution trace (useful for multi-hop programs); optional to use
- Returns a scalar (float/bool) — higher is better

### dspy.Example

`dspy.Example` is a thin dict-like container:

```python
ex = dspy.Example(question="What year was Python created?", answer="1991")
ex = ex.with_inputs("question")   # mark which fields are inputs
```

Calling `.with_inputs()` is important: it tells the optimizer which fields to feed
to the program vs. which to use only for scoring.

### dspy.Evaluate

`dspy.Evaluate` runs your metric over a dataset and returns aggregate statistics:

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=examples,
    metric=my_metric,
    num_threads=4,          # parallelism
    display_progress=True,
)
score = evaluator(program)
```

---

## How it works internally

### Metric function contract

DSPy calls your metric like this for every example in the eval set:

```python
score = metric(example, prediction, trace=None)
```

`trace` is populated during optimization (contains the full LM call history for that
forward pass) and is `None` during plain evaluation. You can use `trace` to penalize
programs that take too many LM calls.

### Scoring patterns

**Exact match (classification)**:

```python
def exact_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    return pred.label.strip().lower() == example.label.strip().lower()
```

**Token F1 (extraction)**:

```python
def token_f1(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    gold = set(example.answer.lower().split())
    pred_tokens = set(pred.answer.lower().split())
    if not gold or not pred_tokens:
        return 0.0
    precision = len(gold & pred_tokens) / len(pred_tokens)
    recall = len(gold & pred_tokens) / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

**LLM-as-judge** — when ground truth is unavailable, use a second LM call to score:

```python
class _JudgeSignature(dspy.Signature):
    """Rate the answer quality on a scale of 0–1."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="float between 0 and 1")

_judge = dspy.Predict(_JudgeSignature)

def llm_judge(example, pred, trace=None) -> float:
    result = _judge(question=example.question, answer=pred.answer)
    try:
        return float(result.score)
    except (ValueError, TypeError):
        return 0.0
```

### Composite metrics

Combine multiple signals:

```python
def composite(example, pred, trace=None) -> float:
    correctness = exact_match(example, pred)
    length_ok = len(pred.answer.split()) <= 20
    return 0.8 * correctness + 0.2 * length_ok
```

---

## Worked example

```python
"""lesson3_metrics.py — Metrics and evaluation."""
import logging
import os
from typing import Any

import dspy
from dspy.evaluate import Evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature and program
# ---------------------------------------------------------------------------

class SentimentClassify(dspy.Signature):
    """Classify the sentiment of a product review."""
    review: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="exactly one of: positive, negative, neutral")


class SentimentProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(SentimentClassify)

    def forward(self, review: str) -> dspy.Prediction:
        return self.classify(review=review)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset() -> list[dspy.Example]:
    raw = [
        ("This product is amazing, love it!", "positive"),
        ("Terrible quality, broke after one day.", "negative"),
        ("It works, nothing special.", "neutral"),
        ("Best purchase I've made this year!", "positive"),
        ("Completely disappointed, total waste of money.", "negative"),
        ("Average, does what it says.", "neutral"),
        ("Exceeded my expectations in every way.", "positive"),
        ("Would not recommend to anyone.", "negative"),
    ]
    return [
        dspy.Example(review=review, sentiment=label).with_inputs("review")
        for review, label in raw
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> bool:
    """Return True if predicted sentiment matches ground truth (case-insensitive)."""
    return pred.sentiment.strip().lower() == example.sentiment.strip().lower()


def lenient_match_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Any = None,
) -> float:
    """Partial credit: full match = 1.0, near-miss = 0.5, wrong = 0.0."""
    pred_clean = pred.sentiment.strip().lower()
    gold_clean = example.sentiment.strip().lower()
    if pred_clean == gold_clean:
        return 1.0
    # partial credit: positive/negative near-miss
    near_miss_pairs = {("positive", "neutral"), ("negative", "neutral")}
    if (pred_clean, gold_clean) in near_miss_pairs or (gold_clean, pred_clean) in near_miss_pairs:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(program: dspy.Module, dataset: list[dspy.Example]) -> dict[str, float]:
    evaluator_exact = Evaluate(
        devset=dataset,
        metric=exact_match_metric,
        num_threads=1,
        display_progress=True,
        display_table=False,
    )
    evaluator_lenient = Evaluate(
        devset=dataset,
        metric=lenient_match_metric,
        num_threads=1,
        display_progress=True,
        display_table=False,
    )
    exact_score = evaluator_exact(program)
    lenient_score = evaluator_lenient(program)
    logger.info("Exact match: %.1f%%", exact_score)
    logger.info("Lenient match: %.1f%%", lenient_score)
    return {"exact": exact_score, "lenient": lenient_score}


if __name__ == "__main__":
    lm = dspy.LM(
        model="anthropic/claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=64,
    )
    dspy.configure(lm=lm)

    dataset = build_dataset()
    program = SentimentProgram()

    # Quick manual check before full eval
    sample = dataset[0]
    pred = program(review=sample.review)
    logger.info("Sample: %s → %s (gold: %s)", sample.review[:40], pred.sentiment, sample.sentiment)

    scores = run_evaluation(program, dataset)
    logger.info("Scores: %s", scores)
```

---

## Exercises

1. **Token F1** — Implement a `token_f1_metric` function (as shown in the concept
   section). Create a short extraction dataset: 5 examples where the input is a
   sentence and the gold output is a key phrase. Run `Evaluate` with your metric.

2. **Length penalty** — Modify `exact_match_metric` to penalize responses longer than
   one word: if the prediction contains more than one word, return `False` regardless
   of correctness. Observe how this affects the score.

3. **LLM-as-judge skeleton** — Write the `llm_judge` function from the concept section.
   Do not run it against a real LM. Instead write a unit test that mocks `dspy.Predict`
   to return `score="0.9"` and asserts the metric returns `0.9`.

4. **Composite metric** — Create a composite metric that weights exact match at 80% and
   a brevity bonus (answer ≤ 1 word) at 20%. Run it on the sentiment dataset.

5. **Dataset split** — Split `build_dataset()` into a `trainset` (first 6) and
   `devset` (last 2). Run `Evaluate` only on `devset`. Why does DSPy need the train/dev
   split to be separate from the metric evaluation set?

(Solutions: [solutions/lesson-3.md](solutions/lesson-3.md))

---

## Self-check

- [ ] I can write a metric function with the correct `(example, prediction, trace)` signature
- [ ] I know what `dspy.Example.with_inputs()` does and why it's required
- [ ] I can run `dspy.Evaluate` and interpret the result
- [ ] I can implement exact match, token F1, and LLM-as-judge metrics
- [ ] I understand why metric quality directly limits optimizer quality
