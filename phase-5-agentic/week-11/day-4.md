# Phase 5 · Week 11 · Day 4 — LLM-as-Judge for Agent Evaluation

> Objective: Build an LLM judge that scores agent outputs on quality dimensions; understand when and how to use it safely.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

Exact-match and keyword metrics fail for open-ended agent tasks. When the expected output is "a coherent summary with correct facts", you need semantic evaluation. An **LLM judge** is a second LLM call that rates the agent's output on explicit dimensions.

### The judge contract

A good LLM judge must:
1. Have an explicit rubric — not "is this good" but "rate factual accuracy 1–5".
2. Return structured output — so you can aggregate and threshold.
3. Be prompted to reason before scoring (chain-of-thought reduces bias).
4. Use a capable model — judges need to understand the domain.
5. Be tested for consistency — same input should produce ±1 score across runs.

### When to use an LLM judge vs. deterministic metrics

| Situation | Use |
|-----------|-----|
| Fixed schema output, known correct value | Exact/field match |
| Text contains required keywords | Contains metric |
| Semantic correctness, nuanced quality | LLM judge |
| Safety / policy adherence | LLM judge + rule filter |

## How it works internally

```
JudgeRequest(query, agent_response, reference_answer?)
         │
         └─ judge_agent.run(formatted_prompt)
                   │
                   └─ JudgeVerdict(
                         scores: dict[str, int],  # dimension → 1-5
                         reasoning: str,
                         overall: float,
                         passed: bool
                      )
```

The judge agent is itself a Pydantic AI agent — typed output, no manual JSON parsing. This means your judge output is validated: if the model tries to return `score: "excellent"` instead of `score: 5`, it fails validation and retries automatically.

## Worked example

```python
# src/eval/llm_judge.py
"""LLM-as-judge for agent output evaluation."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# ── Judge output schema ───────────────────────────────────────────────────────
Score = Annotated[int, Field(ge=1, le=5)]


class JudgeVerdict(BaseModel):
    """Structured verdict from the LLM judge."""
    factual_accuracy: Score
    completeness: Score
    conciseness: Score
    reasoning: str = Field(description="Judge's chain-of-thought before scoring")
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool

    @classmethod
    def from_scores(
        cls,
        factual: int,
        completeness: int,
        conciseness: int,
        reasoning: str,
        pass_threshold: float = 0.6,
    ) -> "JudgeVerdict":
        overall = (factual + completeness + conciseness) / 15.0  # normalise 3-15 → 0-1
        return cls(
            factual_accuracy=factual,
            completeness=completeness,
            conciseness=conciseness,
            reasoning=reasoning,
            overall_score=round(overall, 3),
            passed=overall >= pass_threshold,
        )


# ── Judge agent ───────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = """\
You are an impartial evaluator of AI agent responses.
Given a user query and an agent response (and optionally a reference answer),
score the response on three dimensions, each 1–5:

- factual_accuracy: Are all stated facts correct? (5 = no errors)
- completeness: Does the response address all aspects of the query? (5 = fully complete)
- conciseness: Is the response free of unnecessary repetition/filler? (5 = perfectly concise)

Think step-by-step in `reasoning`, then fill in the integer scores.
Set `overall_score` = (factual + completeness + conciseness) / 15.0 (rounded to 3dp).
Set `passed` = true if overall_score >= 0.6.
"""

judge_agent: Agent[None, JudgeVerdict] = Agent(
    model="anthropic:claude-opus-5-5",  # use capable model for judging
    result_type=JudgeVerdict,
    system_prompt=_JUDGE_SYSTEM,
)


# ── Judge interface ───────────────────────────────────────────────────────────
@dataclass
class JudgeRequest:
    query: str
    agent_response: str
    reference_answer: str | None = None


async def judge(request: JudgeRequest) -> JudgeVerdict:
    """Run the LLM judge on a single agent response."""
    prompt_parts = [
        f"User query: {request.query}",
        f"\nAgent response:\n{request.agent_response}",
    ]
    if request.reference_answer:
        prompt_parts.append(f"\nReference answer:\n{request.reference_answer}")

    prompt = "\n".join(prompt_parts)
    logger.info("Running LLM judge")
    result = await judge_agent.run(prompt)
    logger.info(
        "Judge verdict: overall=%.3f passed=%s",
        result.data.overall_score,
        result.data.passed,
    )
    return result.data


# ── Batch judging ─────────────────────────────────────────────────────────────
async def batch_judge(requests: list[JudgeRequest]) -> list[JudgeVerdict]:
    """Judge a list of agent responses and return verdicts."""
    import asyncio
    verdicts = await asyncio.gather(*(judge(r) for r in requests))
    passed = sum(v.passed for v in verdicts)
    logger.info("Batch judge: %d/%d passed", passed, len(verdicts))
    return list(verdicts)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    req = JudgeRequest(
        query="What is retrieval-augmented generation?",
        agent_response=(
            "RAG combines a retrieval system with a language model. "
            "The retriever fetches relevant documents from a vector store, "
            "and the LLM uses them as context to generate a grounded answer."
        ),
        reference_answer=(
            "Retrieval-Augmented Generation (RAG) augments LLM generation "
            "by retrieving relevant documents at inference time and including "
            "them in the prompt context, reducing hallucination."
        ),
    )
    verdict = asyncio.run(judge(req))
    print(verdict.model_dump_json(indent=2))
```

### Testing the judge without real API calls

```python
# tests/unit/test_llm_judge.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.eval.llm_judge import JudgeRequest, JudgeVerdict, judge


@pytest.fixture
def sample_verdict() -> JudgeVerdict:
    return JudgeVerdict(
        factual_accuracy=4,
        completeness=4,
        conciseness=5,
        reasoning="The response is mostly accurate and complete.",
        overall_score=0.867,
        passed=True,
    )


@pytest.mark.asyncio
async def test_judge_passed(sample_verdict):
    """Judge returns passed=True for a good response."""
    mock_result = MagicMock()
    mock_result.data = sample_verdict

    with patch("src.eval.llm_judge.judge_agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        req = JudgeRequest(query="test query", agent_response="test response")
        verdict = await judge(req)

    assert verdict.passed is True
    assert verdict.overall_score >= 0.6
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_judge_failed():
    """Judge returns passed=False for a poor response."""
    bad_verdict = JudgeVerdict(
        factual_accuracy=2,
        completeness=2,
        conciseness=2,
        reasoning="Response has multiple inaccuracies.",
        overall_score=0.4,
        passed=False,
    )
    mock_result = MagicMock()
    mock_result.data = bad_verdict

    with patch("src.eval.llm_judge.judge_agent.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_result
        req = JudgeRequest(query="q", agent_response="bad response")
        verdict = await judge(req)

    assert verdict.passed is False


def test_judge_verdict_validation():
    """Scores outside 1-5 raise ValidationError."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        JudgeVerdict(
            factual_accuracy=6,  # invalid: > 5
            completeness=3,
            conciseness=3,
            reasoning="",
            overall_score=0.5,
            passed=False,
        )
```

### Consistency testing for judges

Run the same request 5 times and check the variance:

```python
async def test_judge_consistency(n_runs: int = 5) -> float:
    """Return std dev of overall_score across n_runs. Should be < 0.1."""
    import statistics
    req = JudgeRequest(query="What is 2+2?", agent_response="4")
    verdicts = await asyncio.gather(*(judge(req) for _ in range(n_runs)))
    scores = [v.overall_score for v in verdicts]
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    logger.info("Judge consistency std_dev=%.4f", std)
    return std
```

## Exercises

1. Add a `safety` dimension (1–5) to `JudgeVerdict` that rates whether the response avoids harmful content. Update `overall_score` to include it (now 4 dimensions, normalise over 20).

2. Implement `judge_trajectory(trajectory: Trajectory, task_description: str) -> JudgeVerdict` — a judge that rates whether the tool calls made were appropriate and efficient.

3. Create a `JudgeCalibrationTest` dataclass with `request: JudgeRequest` and `expected_pass: bool`. Write a `calibrate(tests: list[JudgeCalibrationTest]) -> float` function that returns accuracy of the judge against known ground truths.

4. The judge agent above uses `anthropic:claude-opus-5-5`. Write a test fixture that swaps it for `TestModel` and returns a fixed `JudgeVerdict`. Explain why using `TestModel` here doesn't test the judge's quality — only its plumbing.

(Solutions: ../solutions/week-11/day-4.md)

## Self-check

- [ ] I can explain when to use an LLM judge vs. deterministic metrics.
- [ ] I can build a typed judge agent that returns a structured `JudgeVerdict`.
- [ ] I understand why using `Annotated[int, Field(ge=1, le=5)]` is important for judge outputs.
- [ ] I can mock the judge agent in tests to avoid real API calls.
- [ ] I can measure judge consistency and know what a reasonable std dev looks like.

## Supplementary links

- [Pydantic AI — Result types](https://ai.pydantic.dev/results/)
- [LMSYS Chatbot Arena paper — LLM judge methodology](https://arxiv.org/abs/2306.05685)
