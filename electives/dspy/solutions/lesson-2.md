# Solutions — Lesson 2: Modules

---

## Exercise 1 — Predict vs CoT on arithmetic

```python
import dspy, os, logging

lm = dspy.LM("anthropic/claude-haiku-5-5", api_key=os.getenv("ANTHROPIC_API_KEY"))
dspy.configure(lm=lm)

logger = logging.getLogger(__name__)

class QA(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

q = "If a train travels 120 km in 1.5 hours, what is its speed in km/h?"

predict = dspy.Predict(QA)
cot = dspy.ChainOfThought(QA)

r_predict = predict(question=q)
r_cot = cot(question=q)

logger.info("Predict: %s", r_predict.answer)
logger.info("CoT reasoning: %s", r_cot.reasoning)
logger.info("CoT answer: %s", r_cot.answer)
```

CoT typically shows "120 / 1.5 = 80 km/h" in the reasoning scratchpad before the
answer. Predict gives the answer directly — may be correct but is more brittle on
multi-step arithmetic.

---

## Exercise 2 — Custom lookup_capital tool

```python
def lookup_capital(country: str) -> str:
    """Return the capital city of the given country (stub)."""
    data = {"germany": "Berlin", "france": "Paris", "japan": "Tokyo"}
    return data.get(country.lower(), "unknown")

class AgentTask(dspy.Signature):
    """Use available tools to answer the question. Tools: lookup_capital, lookup_population, calculate."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

tools = [
    dspy.Tool(lookup_capital, name="lookup_capital", desc="Get capital of a country"),
    dspy.Tool(lookup_population, name="lookup_population", desc="Get population"),
    dspy.Tool(calculate, name="calculate", desc="Arithmetic"),
]
react = dspy.ReAct(AgentTask, tools=tools, max_iters=4)
result = react(question="What is the capital of the country with a population of 84 million?")
# Expected: Berlin (Germany ≈ 84 million)
```

---

## Exercise 3 — Extended FactCheckPipeline

```python
class FactCheckPipeline(dspy.Module):
    class _ExtractClaim(dspy.Signature):
        """Extract the main factual claim."""
        statement: str = dspy.InputField()
        claim: str = dspy.OutputField()

    class _VerifyClaim(dspy.Signature):
        """Assess whether the claim is true or false."""
        claim: str = dspy.InputField()
        verdict: str = dspy.OutputField(desc="'true', 'false', or 'uncertain'")
        explanation: str = dspy.OutputField()

    class _SummarizeCorrection(dspy.Signature):
        """If the claim is false, produce a one-sentence correction."""
        statement: str = dspy.InputField()
        explanation: str = dspy.InputField()
        correction: str = dspy.OutputField(desc="one sentence; empty string if not applicable")

    def __init__(self) -> None:
        super().__init__()
        self.extract = dspy.Predict(self._ExtractClaim)
        self.verify = dspy.ChainOfThought(self._VerifyClaim)
        self.summarize = dspy.Predict(self._SummarizeCorrection)

    def forward(self, statement: str) -> dspy.Prediction:
        extracted = self.extract(statement=statement)
        verified = self.verify(claim=extracted.claim)
        if verified.verdict == "false":
            correction = self.summarize(
                statement=statement,
                explanation=verified.explanation,
            )
            return dspy.Prediction(
                verdict=verified.verdict,
                explanation=verified.explanation,
                correction=correction.correction,
            )
        return dspy.Prediction(
            verdict=verified.verdict,
            explanation=verified.explanation,
            correction="",
        )
```

---

## Exercise 4 — Module introspection

```python
pipe = FactCheckPipeline()
predictors = pipe.named_predictors()
# Returns a list of (name, Predict instance) tuples, e.g.:
# [("extract", Predict(…)), ("verify", ChainOfThought(…)), ("summarize", Predict(…))]
```

`named_predictors()` walks the module tree and collects every `dspy.Predict`
(including `ChainOfThought`, which subclasses `Predict`). The optimizer iterates this
list to inject demos and swap instructions — one entry per optimizable node.

---

## Exercise 5 — Unit test for FactCheckPipeline

```python
"""test_lesson2.py"""
from unittest.mock import MagicMock, patch

import dspy
import pytest

from lesson2_modules import FactCheckPipeline


def test_pipeline_returns_verdict_and_explanation() -> None:
    mock_extract = MagicMock(return_value=MagicMock(claim="The Eiffel Tower is in Berlin."))
    mock_verify = MagicMock(return_value=MagicMock(verdict="false", explanation="It is in Paris."))

    with (
        patch.object(dspy.Predict, "__call__", mock_extract),
        patch.object(dspy.ChainOfThought, "__call__", mock_verify),
    ):
        pipe = FactCheckPipeline()
        result = pipe(statement="The Eiffel Tower is in Berlin.")

    assert result.verdict == "false"
    assert "Paris" in result.explanation
```

Note: patching at the class level intercepts all instances. If your pipeline has
multiple `Predict` nodes, use `side_effect` to return different values per call:

```python
mock_predict = MagicMock(side_effect=[
    MagicMock(claim="..."),        # first call (extract)
    MagicMock(correction="..."),   # second call (summarize)
])
```
