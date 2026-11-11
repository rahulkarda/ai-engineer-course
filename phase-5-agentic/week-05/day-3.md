# Phase 5 · Week 05 · Day 3 — Verifier / Critic Agents

> Objective: Build a dedicated critic agent that evaluates another agent's output against explicit rubric axes and returns structured verdicts.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]]

## Concept

A **verifier** or **critic agent** is a separate agent whose sole job is to evaluate output — not produce it. Unlike the self-critique in Day 1 (same model, different prompt), a critic agent can be:

- A **different model** (cheaper/faster for checking vs. generating)
- A **separate service** (independent codebase, separate latency budget)
- A **specialised prompt** optimised purely for evaluation

Critic agents are inspired by how humans work: a writer's editor is a separate person with different skills. The critic's prompt space is dominated by "what makes output good" rather than "how to produce output."

### Roles in the pipeline

```
Worker Agent ──→ output ──→ Critic Agent ──→ verdict
                                 │
                          ┌──────▼──────┐
                          │  PASS/FAIL  │
                          │  + feedback │
                          └─────────────┘
```

If FAIL, the worker agent gets the feedback and retries. The critic has no knowledge of internal worker state — it only sees input + output. This decoupling is the architectural benefit.

### Rubric-based evaluation

A rubric anchors the critic. Without it, the critic drifts toward subjective preferences. A rubric specifies:
- **Dimensions**: what axes to check (accuracy, safety, format, completeness)
- **Pass criteria**: a threshold per dimension
- **Weight**: how dimensions combine into a verdict

## How it works internally

The critic prompt is a scoring contract. It should:
1. Receive: task description + produced output
2. Evaluate: each dimension independently
3. Return: structured verdict (Pydantic-parseable JSON)

Keeping critic prompts under ~200 tokens (excluding the evaluated content) keeps latency manageable. The heavy content (the output being evaluated) is what drives token cost.

**Critic vs. judge**: a critic evaluates one output against a rubric. A judge compares multiple outputs. Both are critic agents; the name shifts with use-case.

## Worked example

```python
"""
day3_critic_agent.py — Dedicated critic agent with rubric-based evaluation.

Run:  python day3_critic_agent.py
Env:  ANTHROPIC_API_KEY
"""
import logging
import os
from enum import Enum

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class DimensionScore(BaseModel):
    dimension: str
    score: int = Field(ge=0, le=10)
    reasoning: str


class CriticVerdict(BaseModel):
    verdict: Verdict
    overall_score: int = Field(ge=0, le=10)
    dimensions: list[DimensionScore]
    summary: str
    revision_request: str | None = None  # populated only on FAIL


class RubricAxis(BaseModel):
    name: str
    description: str
    pass_threshold: int = 7  # score >= this → passes this dimension


class CriticAgent:
    def __init__(
        self,
        rubric: list[RubricAxis],
        model: str = "claude-haiku-5-5",  # faster/cheaper for evaluation
        pass_threshold: int = 7,
    ) -> None:
        self.rubric = rubric
        self.model = model
        self.pass_threshold = pass_threshold
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def _build_prompt(self, task: str, output: str) -> str:
        axes = "\n".join(
            f"- {ax.name}: {ax.description} (pass if score >= {ax.pass_threshold})"
            for ax in self.rubric
        )
        return (
            f"You are an expert evaluator. Score this output strictly.\n\n"
            f"TASK GIVEN TO WORKER:\n{task}\n\n"
            f"WORKER OUTPUT:\n{output}\n\n"
            f"RUBRIC AXES:\n{axes}\n\n"
            "Respond as JSON matching exactly:\n"
            '{"verdict": "PASS"|"FAIL", "overall_score": 0-10, '
            '"dimensions": [{"dimension": str, "score": 0-10, "reasoning": str}, ...], '
            '"summary": str, "revision_request": str|null}\n'
            "verdict=PASS only if overall_score >= 7 AND every dimension >= its threshold."
        )

    def evaluate(self, task: str, output: str) -> CriticVerdict:
        log.info("Critic evaluating output (task length=%d)...", len(task))
        prompt = self._build_prompt(task, output)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        import json

        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw.strip())
        verdict = CriticVerdict(**data)
        log.info("Verdict=%s overall=%d", verdict.verdict, verdict.overall_score)
        return verdict


class WorkerAgent:
    """Simple single-call worker whose output the critic evaluates."""

    def __init__(self, model: str = "claude-opus-5-5") -> None:
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def run(self, task: str, feedback: str | None = None) -> str:
        messages: list[dict] = []
        if feedback:
            messages.append({
                "role": "user",
                "content": (
                    f"Previous attempt was rejected with this feedback:\n{feedback}\n\n"
                    "Now complete the original task with these issues fixed."
                ),
            })
        messages.append({"role": "user", "content": task})
        resp = self.client.messages.create(
            model=self.model, max_tokens=512, messages=messages
        )
        return resp.content[0].text


def run_with_critic(
    task: str, worker: WorkerAgent, critic: CriticAgent, max_retries: int = 2
) -> tuple[str, CriticVerdict]:
    feedback: str | None = None
    for attempt in range(1, max_retries + 2):
        log.info("Attempt %d/%d", attempt, max_retries + 1)
        output = worker.run(task, feedback=feedback)
        verdict = critic.evaluate(task, output)
        if verdict.verdict == Verdict.PASS:
            return output, verdict
        feedback = verdict.revision_request or verdict.summary
        log.warning("Attempt %d failed: %s", attempt, feedback)
    log.error("All attempts failed. Returning best effort.")
    return output, verdict


if __name__ == "__main__":
    RUBRIC = [
        RubricAxis(name="accuracy", description="Factually correct", pass_threshold=7),
        RubricAxis(name="completeness", description="Covers all required points", pass_threshold=7),
        RubricAxis(name="clarity", description="Clear and well-structured prose", pass_threshold=6),
        RubricAxis(name="safety", description="No harmful or misleading content", pass_threshold=8),
    ]

    worker = WorkerAgent()
    critic = CriticAgent(rubric=RUBRIC)

    task = (
        "Write a 3-sentence explanation of how transformers use self-attention, "
        "suitable for a software engineer with no ML background."
    )

    final_output, final_verdict = run_with_critic(task, worker, critic, max_retries=2)
    print(f"\n=== Final Output ===\n{final_output}")
    print(f"\n=== Verdict ===\n{final_verdict.model_dump_json(indent=2)}")
```

## Exercises

1. The critic returns a single `revision_request` string. Extend `CriticVerdict` to include `per_dimension_suggestions: dict[str, str]` — one suggestion per failing dimension. Update the prompt to produce this field. Write a test verifying the field is populated on FAIL verdicts.

2. Currently the worker receives all feedback at once. Implement a **priority filter**: only pass feedback for dimensions that scored below their threshold. Dimensions that passed should not appear in the revision request.

3. Add a `CriticAgent.compare(task, output_a, output_b) -> str` method that returns `"A"`, `"B"`, or `"TIE"` by evaluating both against the rubric and comparing overall scores. Write a test mocking both evaluate calls.

4. The current setup uses the same `CriticAgent` for all retries. Implement a **rotating critic**: on retry 2, use a different `model` (e.g. a larger model) for a "second opinion". The constructor should accept `models: list[str]` and cycle through them.

(Solutions: ../solutions/week-05/day-3.md)

## Self-check

- [ ] I can articulate why a separate critic agent outperforms self-critique for some tasks
- [ ] I can write a rubric-based critic prompt that returns parseable JSON
- [ ] I can implement a Pydantic model for a structured critic verdict
- [ ] I understand when to use haiku vs. opus as the critic model
- [ ] I can wire worker + critic into a retry loop

---

## Supplementary links

- "Constitutional AI: Harmlessness from AI Feedback" — https://arxiv.org/abs/2212.08073
- "Judging LLM-as-a-Judge" — https://arxiv.org/abs/2306.05685
- Pydantic field validators — https://docs.pydantic.dev/latest/concepts/validators/
