# Solutions — Week 05 · Day 1: Reflection Pattern

## Exercise 1: Dynamic axes in critique prompt

```python
from dataclasses import dataclass, field
import anthropic, os, json, logging
from pydantic import BaseModel

log = logging.getLogger(__name__)


class CritiqueResult(BaseModel):
    passed: bool
    feedback: str
    score: int


@dataclass
class ReflectionLoop:
    client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    )
    model: str = "claude-opus-5-5"
    max_iterations: int = 3
    axes: list[str] = field(default_factory=lambda: ["correctness", "completeness", "clarity"])

    def _critique(self, task: str, draft: str) -> CritiqueResult:
        axis_text = ", ".join(f"({i+1}) {a}" for i, a in enumerate(self.axes))
        prompt = (
            f"Original task:\n{task}\n\nDraft answer:\n{draft}\n\n"
            f"Evaluate on: {axis_text}.\n"
            "Respond as JSON: {\"passed\": bool, \"feedback\": str, \"score\": 0-10}.\n"
            "passed=true only if score >= 8."
        )
        resp = self.client.messages.create(
            model=self.model, max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return CritiqueResult(**json.loads(raw.strip()))
```

Test:

```python
# tests/test_day1_axes.py
import json
from unittest.mock import MagicMock
import pytest
from day1_reflection import ReflectionLoop, CritiqueResult


def make_mock_client(text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    client.messages.create.return_value = msg
    return client


def test_axes_appear_in_critique_prompt():
    axes = ["grammar", "tone", "length"]
    mock_response = json.dumps({"passed": True, "feedback": "good", "score": 9})
    client = make_mock_client(mock_response)

    loop = ReflectionLoop(client=client, axes=axes)
    loop._critique("Write a greeting.", "Hello!")

    call_args = client.messages.create.call_args
    prompt_content = call_args.kwargs["messages"][0]["content"]
    for axis in axes:
        assert axis in prompt_content, f"Axis '{axis}' missing from critique prompt"
```

---

## Exercise 2: `min_score` parameter

```python
@dataclass
class ReflectionLoop:
    min_score: int = 8

    def _critique(self, task: str, draft: str) -> CritiqueResult:
        # Remove hardcoded threshold from prompt; just return score
        ...

    def run(self, task: str) -> tuple[str, int]:
        draft = self._generate(task)
        for iteration in range(1, self.max_iterations + 1):
            critique = self._critique(task, draft)
            log.info("Iteration %d — score=%d", iteration, critique.score)
            # Use self.min_score instead of hardcoded 8
            if critique.score >= self.min_score:
                return draft, iteration
            if iteration < self.max_iterations:
                draft = self._revise(task, draft, critique.feedback)
        return draft, self.max_iterations
```

---

## Exercise 3: Structured feedback with axis scores

```python
class AxisFeedback(BaseModel):
    axis: str
    score: int
    suggestion: str


class CritiqueResult(BaseModel):
    passed: bool
    feedback: str
    score: int
    axis_scores: dict[str, int] = {}
    axis_feedback: list[AxisFeedback] = []
```

Revise prompt addition:

```
"Also return \"axis_feedback\": [{\"axis\": str, \"score\": 0-10, \"suggestion\": str}, ...]"
```

Revise step — pass structured feedback:

```python
def _revise(self, task: str, draft: str, critique: CritiqueResult) -> str:
    structured = "\n".join(
        f"- {af.axis} (score {af.score}): {af.suggestion}"
        for af in critique.axis_feedback
    )
    messages = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": draft},
        {"role": "user", "content": f"Issues:\n{structured}\nPlease rewrite addressing all."},
    ]
    resp = self.client.messages.create(
        model=self.model, max_tokens=512, messages=messages
    )
    return resp.content[0].text
```

---

## Exercise 4: `ReflectionHistory`

```python
from dataclasses import dataclass, field
from pydantic import BaseModel


class HistoryEntry(BaseModel):
    iteration: int
    draft: str
    critique: CritiqueResult


@dataclass
class ReflectionHistory:
    entries: list[HistoryEntry] = field(default_factory=list)

    def add(self, iteration: int, draft: str, critique: CritiqueResult) -> None:
        self.entries.append(HistoryEntry(iteration=iteration, draft=draft, critique=critique))

    def __len__(self) -> int:
        return len(self.entries)
```

Test:

```python
def test_history_length_matches_iterations():
    # Mock: fail twice, pass third
    responses = [
        "Draft 1",
        json.dumps({"passed": False, "feedback": "needs work", "score": 5}),
        "Draft 2",
        json.dumps({"passed": False, "feedback": "almost there", "score": 6}),
        "Draft 3",
        json.dumps({"passed": True, "feedback": "great", "score": 9}),
    ]
    # Wire mock client to cycle through responses
    # Verify history length == 3 (one entry per iteration)
    ...
```
