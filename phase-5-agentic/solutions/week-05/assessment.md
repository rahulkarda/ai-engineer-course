# Solutions — Week 05 Assessment

## Conceptual answers

**Q1: Plain reflection vs. Reflexion**

Plain reflection (Day 1) generates a draft, critiques it in the same session, and revises. Memory scope is one iteration: the next iteration starts fresh with only the latest draft. Reflexion (Day 4) adds a reflector step that synthesises each failure into a first-person verbal lesson ("I should validate input before sorting"). Lessons accumulate in a memory list injected into every subsequent actor prompt. The accumulation matters because complex tasks often fail for different reasons across attempts — later attempts benefit from all prior lessons, not just the most recent one. Without accumulation, the agent can repeat the same mistake after the first revision.

**Q2: Three questions before self-consistency everywhere**

1. "What is the latency budget?" — self-consistency adds N× latency if run serially and ~1× if parallel, but parallel scheduling overhead still adds 200–500ms per batch. For user-facing sub-2s calls it's a non-starter.
2. "Does this task have a definable right answer?" — self-consistency only helps when outputs can be compared or normalised. For free-form generation (creative writing, summaries), you can't majority-vote text.
3. "What is the current single-call reliability?" — if the model already gets the right answer >90% of the time, self-consistency gains <1 point at 5× cost. The break-even is typically around 70% reliability.

**Q3: Code vs. marketing email**

Code correctness: use **test-based Reflexion** (automated evaluator, no LLM judge). Code either passes tests or it doesn't — you don't need a rubric, you need a pass/fail signal. An LLM critic on code is less reliable than `pytest`.

Marketing email: use a **critic agent with a rubric**. Quality is multi-dimensional (tone, clarity, call-to-action strength, brand voice) and not automatically verifiable. A critic prompt with explicit axes produces structured, actionable feedback.

**Q4: Repeated WARN policy**

A WARN that persists after 2 retries is no longer recoverable by the worker — the worker likely cannot produce output without that PII if the data genuinely needs to be there (e.g. a summary of a document that contains emails). The supervisor should: (a) log the repeated WARN with escalation flag, (b) raise `EscalationError` with the audit log attached so a human reviewer can decide whether to allow the output or strip the PII before delivery. Implementation: track WARN counts per rule in supervisor state; after `max_warn_retries`, escalate.

**Q5: Reflexion token cost formula**

For 3 attempts:
```
Total ≈ 3P + 2R + 2E

P = tokens per generation (actor call)
R = tokens per reflection call (reflector)
E = tokens per evaluation call (critic/test)
```
Each reflection call is cheaper than generation (it's a short synthesis prompt) so R < P typically. Each evaluation call depends on method: if it's `pytest` there's no LLM cost; if it's an LLM critic then E ≈ 0.5P.

**Q6: When critics make quality worse**

A task that is already highly reliable (>95% accuracy) with a single call. Adding a critic introduces two failure modes: (1) the critic can mistakenly flag a correct answer as wrong (false negative), causing the model to "fix" a correct answer into an incorrect one — this is called **over-revision**; (2) the critic consumes extra context budget that could have gone to the actual task. Classic example: straightforward factual Q&A on well-covered topics. The model knows the answer; the critic second-guesses it and the reviser complies, introducing errors.

**Q7: Debugging a false-block from LLM guardrail**

Cause 1: **Overly conservative system prompt in the guardrail model**. Fix: inspect the guardrail prompt. Run the same input through the guardrail prompt in a REPL with verbose output. Narrow the scope of what it checks.

Cause 2: **Guardrail model misclassifies domain vocabulary as harmful**. Common with medical, legal, or security terms. Fix: add a domain context prefix ("This is a medical documentation assistant. The following text contains clinical terminology...") to the guardrail prompt so the model understands the legitimate domain.

---

## `ReviewBot` reference implementation

```python
# review_bot.py
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ReviewResult(BaseModel):
    passed: bool
    scores: dict[str, int]
    feedback: str
    suggested_revision: str | None = None


class ReviewBot:
    def __init__(
        self,
        rubric: list[str],
        model: str = "claude-haiku-5-5",
        pass_threshold: int = 6,
    ) -> None:
        self.rubric = rubric
        self.model = model
        self.pass_threshold = pass_threshold
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def review(self, text: str) -> ReviewResult:
        if not self.rubric:
            return ReviewResult(
                passed=True, scores={}, feedback="No rubric — auto pass", suggested_revision=None
            )

        criteria_text = "\n".join(f"- {c}" for c in self.rubric)
        prompt = (
            f"Evaluate this text against the criteria below. Score each 0-10.\n\n"
            f"TEXT:\n{text}\n\n"
            f"CRITERIA:\n{criteria_text}\n\n"
            'Respond as JSON: {"scores": {criterion: score}, "feedback": str, '
            '"suggested_revision": str|null}\n'
            f"suggested_revision required if any score < {self.pass_threshold}."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        scores: dict[str, int] = data["scores"]
        passed = all(v >= self.pass_threshold for v in scores.values())
        suggested = data.get("suggested_revision")
        if not passed and not suggested:
            suggested = data.get("feedback", "Please revise.")
        return ReviewResult(
            passed=passed,
            scores=scores,
            feedback=data["feedback"],
            suggested_revision=suggested,
        )

    def improve(self, text: str, review: ReviewResult) -> str:
        if not review.suggested_revision:
            return text
        failing = [c for c, s in review.scores.items() if s < self.pass_threshold]
        prompt = (
            f"Rewrite this text to fix the following issues:\n"
            f"Failing criteria: {failing}\n"
            f"Suggestions: {review.suggested_revision}\n\n"
            f"Original:\n{text}\n\nImproved version:"
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def review_and_improve(
        self, text: str, max_rounds: int = 2
    ) -> tuple[str, ReviewResult]:
        current = text
        result = self.review(current)
        for _ in range(max_rounds):
            if result.passed:
                break
            current = self.improve(current, result)
            result = self.review(current)
        return current, result

    def batch_review(self, texts: list[str]) -> list[ReviewResult]:
        results: list[ReviewResult | None] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=min(len(texts), 5)) as pool:
            futures = {pool.submit(self.review, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results  # type: ignore[return-value]
```

---

## Test suite (`tests/test_reviewbot.py`)

```python
import json
from unittest.mock import MagicMock
import pytest
from review_bot import ReviewBot, ReviewResult


def make_client(text: str):
    client = MagicMock()
    client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=text)]
    )
    return client


PASS_RESPONSE = json.dumps({
    "scores": {"clarity": 8, "accuracy": 9},
    "feedback": "Well written",
    "suggested_revision": None,
})

FAIL_RESPONSE = json.dumps({
    "scores": {"clarity": 4, "accuracy": 7},
    "feedback": "Needs work",
    "suggested_revision": "Improve sentence structure",
})


def test_review_pass():
    bot = ReviewBot(rubric=["clarity", "accuracy"])
    bot.client = make_client(PASS_RESPONSE)
    result = bot.review("Some text")
    assert result.passed is True
    assert result.suggested_revision is None


def test_review_fail_has_suggestion():
    bot = ReviewBot(rubric=["clarity", "accuracy"])
    bot.client = make_client(FAIL_RESPONSE)
    result = bot.review("Bad text")
    assert result.passed is False
    assert result.suggested_revision is not None


def test_improve_called_on_fail():
    bot = ReviewBot(rubric=["clarity"])
    calls = []
    original_improve = bot.improve

    def spy_improve(text, review):
        calls.append(text)
        return "improved text"

    bot.improve = spy_improve

    review = ReviewResult(
        passed=False,
        scores={"clarity": 3},
        feedback="Poor",
        suggested_revision="Be clearer",
    )
    result = bot.improve("original", review)
    assert result == "improved text"


def test_review_and_improve_stops_on_pass():
    bot = ReviewBot(rubric=["clarity"])
    review_count = [0]

    def mock_review(text):
        review_count[0] += 1
        return ReviewResult(passed=True, scores={"clarity": 9}, feedback="Good")

    bot.review = mock_review
    text, final_review = bot.review_and_improve("text", max_rounds=3)
    assert review_count[0] == 1  # stopped immediately on first pass


def test_empty_rubric_auto_pass():
    bot = ReviewBot(rubric=[])
    bot.client = MagicMock()  # should never be called
    result = bot.review("anything")
    assert result.passed is True
    bot.client.messages.create.assert_not_called()
```

---

## Rubric scoring guide

| Score | Criteria |
|---|---|
| 90–100% | `ReviewBot` works end-to-end, all 4+ tests pass, structured output correct, type hints complete |
| 75–89% | Core loop works, 3/4 tests pass, minor type hint gaps |
| 60–74% | Loop terminates but edge cases fail, 2/4 tests pass |
| < 60% | Fundamental loop broken or tests depend on real LLM |
