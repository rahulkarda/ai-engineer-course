# Solutions — Week 05 · Day 3: Verifier / Critic Agents

## Exercise 1: Per-dimension suggestions

Extended model:

```python
from pydantic import BaseModel, Field
from enum import Enum


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class DimensionScore(BaseModel):
    dimension: str
    score: int = Field(ge=0, le=10)
    reasoning: str
    suggestion: str | None = None  # populated on fail


class CriticVerdict(BaseModel):
    verdict: Verdict
    overall_score: int = Field(ge=0, le=10)
    dimensions: list[DimensionScore]
    summary: str
    revision_request: str | None = None
    per_dimension_suggestions: dict[str, str] = {}
```

Updated prompt suffix:

```python
"For every dimension that scores below its threshold, add it to "
"\"per_dimension_suggestions\": {dimension_name: specific_suggestion}. "
"Leave the dict empty if all pass."
```

Test:

```python
import json
from unittest.mock import MagicMock
from day3_critic_agent import CriticAgent, RubricAxis


FAIL_VERDICT = {
    "verdict": "FAIL",
    "overall_score": 5,
    "dimensions": [
        {"dimension": "accuracy", "score": 4, "reasoning": "wrong claim"},
        {"dimension": "clarity", "score": 8, "reasoning": "clear"},
    ],
    "summary": "Inaccurate content",
    "revision_request": "Fix the factual error",
    "per_dimension_suggestions": {"accuracy": "Verify the claim about X"},
}


def test_per_dimension_suggestions_on_fail():
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=json.dumps(FAIL_VERDICT))]
    )
    rubric = [RubricAxis(name="accuracy", description="factual", pass_threshold=7),
              RubricAxis(name="clarity", description="clear", pass_threshold=6)]
    critic = CriticAgent(rubric=rubric)
    critic.client = client

    verdict = critic.evaluate("Explain X", "X is caused by Y.")
    assert verdict.verdict.value == "FAIL"
    assert "accuracy" in verdict.per_dimension_suggestions
    assert len(verdict.per_dimension_suggestions["accuracy"]) > 0
```

---

## Exercise 2: Priority filter — only pass feedback for failing dimensions

```python
def filter_feedback_for_failing(
    verdict: CriticVerdict, rubric: list[RubricAxis]
) -> str:
    """Return feedback only for dimensions below their threshold."""
    threshold_map = {ax.name: ax.pass_threshold for ax in rubric}
    failing = [
        f"{dim.dimension}: {dim.reasoning}"
        for dim in verdict.dimensions
        if dim.score < threshold_map.get(dim.dimension, 7)
    ]
    if not failing:
        return ""
    return "Address only these issues:\n" + "\n".join(f"- {f}" for f in failing)
```

In `run_with_critic`:

```python
feedback = filter_feedback_for_failing(verdict, critic.rubric) or verdict.revision_request
```

---

## Exercise 3: `compare` method

```python
def compare(self, task: str, output_a: str, output_b: str) -> str:
    """Returns 'A', 'B', or 'TIE'."""
    verdict_a = self.evaluate(task, output_a)
    verdict_b = self.evaluate(task, output_b)
    if verdict_a.overall_score > verdict_b.overall_score:
        return "A"
    if verdict_b.overall_score > verdict_a.overall_score:
        return "B"
    return "TIE"
```

Test:

```python
def test_compare_returns_higher_score(monkeypatch):
    rubric = [RubricAxis(name="accuracy", description="factual")]
    critic = CriticAgent(rubric=rubric)

    verdict_a_data = {"verdict": "PASS", "overall_score": 9, "dimensions": [], "summary": "great"}
    verdict_b_data = {"verdict": "PASS", "overall_score": 6, "dimensions": [], "summary": "ok"}

    call_count = [0]
    def fake_evaluate(task, output):
        result = verdict_a_data if call_count[0] == 0 else verdict_b_data
        call_count[0] += 1
        return CriticVerdict(**result)

    monkeypatch.setattr(critic, "evaluate", fake_evaluate)
    assert critic.compare("task", "output A", "output B") == "A"
```

---

## Exercise 4: Rotating critic model

```python
from dataclasses import dataclass, field
from itertools import cycle


@dataclass
class RotatingCriticAgent:
    rubric: list[RubricAxis]
    models: list[str] = field(default_factory=lambda: ["claude-haiku-5-5", "claude-opus-5-5"])

    def __post_init__(self):
        self._model_cycle = cycle(self.models)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def evaluate(self, task: str, output: str) -> CriticVerdict:
        model = next(self._model_cycle)
        log.info("Using critic model: %s", model)
        # ... same as CriticAgent.evaluate but uses `model` variable
        ...
```

The cycle automatically advances through models on each call. On retry 2 it uses the second model in the list.
