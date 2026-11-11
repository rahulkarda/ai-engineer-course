# Lesson 5 — Teleprompters & Advanced Patterns

> Objective: Survey the full optimizer family; apply advanced patterns (typed outputs, assertions, ensembling).
> Time: ~45 min · Prereqs: lesson-4.md

---

## Concept

"Teleprompter" is DSPy's name for any optimizer. The name reflects the idea that the
optimizer suggests what the LM should say — it doesn't change your code, only the
prompts fed to the model.

Beyond `BootstrapFewShot` and `MIPRO`, DSPy ships additional teleprompters for
specific scenarios:

| Teleprompter | Best for |
|---|---|
| `BootstrapFewShot` | Small trainsets; fast iteration |
| `BootstrapFewShotWithRandomSearch` | Slightly larger budget; more robust demo selection |
| `MIPROv2` | Instruction + demo co-optimization; larger compute budget |
| `BootstrapFinetune` | Fine-tuning open-weight models (not applicable to Anthropic API) |
| `KNNFewShot` | Retrieve similar examples at runtime instead of fixed demos |
| `Ensemble` | Run multiple compiled variants, aggregate answers |

---

## How it works internally

### Typed outputs with `dspy.Literal` and Pydantic

DSPy can enforce structured outputs beyond plain strings. Two approaches:

**1. Constrained string literals via Python `Literal`:**

```python
from typing import Literal

class ClassifyTopic(dspy.Signature):
    """Assign a support ticket to exactly one category."""
    ticket: str = dspy.InputField()
    category: Literal["billing", "technical", "account", "other"] = dspy.OutputField()
```

DSPy serializes the valid values into the prompt and validates the response.

**2. Pydantic model outputs:**

```python
from pydantic import BaseModel

class ExtractionResult(BaseModel):
    company: str
    ticker: str
    sentiment: Literal["bullish", "bearish", "neutral"]

class ExtractFinancial(dspy.Signature):
    """Extract financial entities from a news headline."""
    headline: str = dspy.InputField()
    result: ExtractionResult = dspy.OutputField()
```

DSPy uses `dspy.TypedPredictor` (or `dspy.TypedChainOfThought`) to wrap structured
output signatures.

### dspy.Assert and dspy.Suggest

These primitives let you add runtime constraints that the optimizer can learn to satisfy:

```python
import dspy

class SafeQA(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        pred = self.predict(question=question)
        # Hard constraint — fails and retries if violated
        dspy.Assert(
            len(pred.answer.split()) <= 30,
            "Answer must be 30 words or fewer",
        )
        # Soft constraint — hints optimizer, does not raise
        dspy.Suggest(
            pred.answer.endswith("."),
            "Answer should end with a period",
        )
        return pred
```

`dspy.Assert` raises `dspy.AssertionError` on failure; DSPy's backtracking mechanism
retries the LM call with the error message appended to the prompt (up to
`max_retries` times). `dspy.Suggest` only fires during optimization — it penalizes
but does not block.

### KNNFewShot

Instead of fixed demos, retrieve the `k` most similar training examples at inference
time using an embedding index:

```python
from dspy.teleprompt import KNNFewShot

optimizer = KNNFewShot(k=3, trainset=trainset)
compiled = optimizer.compile(program)
```

This is useful when the input space is large and diverse — fixed demos may not
represent rare sub-distributions.

### Ensemble

Run `n` compiled variants (e.g., from different random seeds) and aggregate:

```python
from dspy.teleprompt import Ensemble

variants = [compile_bootstrap(program, trainset, seed=i) for i in range(3)]
ensemble = Ensemble(reduce_fn=dspy.majority)
compiled_ensemble = ensemble.compile(variants)
```

`dspy.majority` picks the most common prediction across variants — effective for
classification tasks.

---

## Worked example

```python
"""lesson5_advanced.py — Typed outputs, Assertions, Ensemble sketch."""
import logging
import os
from typing import Literal

import dspy
from dspy.teleprompt import BootstrapFewShot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed output example
# ---------------------------------------------------------------------------

class TicketRoute(dspy.Signature):
    """Route a support ticket to the correct team."""
    ticket: str = dspy.InputField(desc="raw support ticket text")
    category: Literal["billing", "technical", "account", "other"] = dspy.OutputField()
    priority: Literal["low", "medium", "high"] = dspy.OutputField()


class TicketRouter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.route = dspy.ChainOfThought(TicketRoute)

    def forward(self, ticket: str) -> dspy.Prediction:
        return self.route(ticket=ticket)


# ---------------------------------------------------------------------------
# Assertion-guarded module
# ---------------------------------------------------------------------------

class BriefAnswer(dspy.Signature):
    """Answer concisely in at most 15 words."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class BriefQA(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.ChainOfThought(BriefAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        pred = self.predict(question=question)
        dspy.Assert(
            len(pred.answer.split()) <= 15,
            f"Answer is {len(pred.answer.split())} words; must be ≤ 15",
        )
        dspy.Suggest(
            not pred.answer.startswith("I "),
            "Prefer third-person or impersonal phrasing",
        )
        return pred


# ---------------------------------------------------------------------------
# Demo: compile TicketRouter
# ---------------------------------------------------------------------------

def build_ticket_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    raw = [
        ("My invoice is wrong, I was charged twice.", "billing", "high"),
        ("I can't log into my account.", "account", "medium"),
        ("The API is returning 500 errors.", "technical", "high"),
        ("How do I change my password?", "account", "low"),
        ("I need a refund for my subscription.", "billing", "medium"),
        ("The dashboard is loading slowly.", "technical", "low"),
        ("What are your office hours?", "other", "low"),
        ("My payment failed three times.", "billing", "high"),
    ]
    examples = [
        dspy.Example(ticket=t, category=c, priority=p).with_inputs("ticket")
        for t, c, p in raw
    ]
    return examples[:6], examples[6:]


def category_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    return (
        pred.category.strip().lower() == example.category.strip().lower()
        and pred.priority.strip().lower() == example.priority.strip().lower()
    )


if __name__ == "__main__":
    lm = dspy.LM(
        model="anthropic/claude-haiku-5-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=128,
    )
    dspy.configure(lm=lm)

    # Test typed routing
    router = TicketRouter()
    result = router(ticket="My API key stopped working after I upgraded my plan.")
    logger.info("Category: %s | Priority: %s", result.category, result.priority)

    # Test assertion-guarded QA
    brief_qa = BriefQA()
    answer = brief_qa(question="What is the speed of light in a vacuum?")
    logger.info("Brief answer (%d words): %s", len(answer.answer.split()), answer.answer)

    # Compile ticket router
    trainset, devset = build_ticket_dataset()
    optimizer = BootstrapFewShot(metric=category_match, max_bootstrapped_demos=3)
    compiled_router = optimizer.compile(router, trainset=trainset)
    logger.info("Compiled router ready. Demos: %d", len(compiled_router.route.demos))
```

---

## Exercises

1. **Pydantic structured output** — Create a Pydantic model `InvoiceFields` with
   fields `vendor: str`, `amount: float`, `date: str`. Write a Signature that extracts
   it from invoice text. Use `dspy.TypedPredictor` to wrap it. Write a unit test that
   mocks the predictor and asserts the Pydantic model is instantiated correctly.

2. **Assert retry behavior** — Modify `BriefQA` to use `max_retries=3` in
   `dspy.settings.configure`. Add a `logger.warning` inside the `forward` method that
   fires when the assertion would fail (before DSPy retries). Run it on a question
   that typically produces a long answer and confirm retries are attempted.

3. **Suggest vs Assert** — Convert the `dspy.Assert` in `BriefQA` to `dspy.Suggest`.
   Observe: does the program still raise an error on a long answer? What changes?

4. **KNNFewShot** — Write a `compile_knn(program, trainset)` function using
   `KNNFewShot(k=2, trainset=trainset)`. Do not run it. Write a unit test that mocks
   `KNNFewShot.compile` and asserts it receives `program` as an argument.

5. **Ensemble sketch** — Write a function `build_ensemble(trainset)` that:
   - Creates three `SentimentProgram()` instances (from lesson 3)
   - Compiles each with `BootstrapFewShot` using a different random seed
   - Combines them with `Ensemble(reduce_fn=dspy.majority)`
   - Returns the compiled ensemble
   Mock the optimizer in tests.

(Solutions: [solutions/lesson-5.md](solutions/lesson-5.md))

---

## Self-check

- [ ] I can use `Literal` types in a Signature to constrain model output to valid values
- [ ] I can write a `dspy.Assert` constraint and understand the retry mechanism
- [ ] I know the difference between `dspy.Assert` (hard) and `dspy.Suggest` (soft)
- [ ] I can explain when KNNFewShot outperforms fixed-demo BootstrapFewShot
- [ ] I can sketch an Ensemble over multiple compiled variants
