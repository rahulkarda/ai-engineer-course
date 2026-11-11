# Reasoning Models · Lesson 5 — Verifiers, Evals, and Agents

> Objective: Build verifier models, evaluate reasoning quality, and combine reasoning with tool use.
> Time: ~1 hr · Prereqs: [lesson-4.md](lesson-4.md)

---

## Concept

### Verifier models

A **verifier** is a model (or function) that scores a candidate answer without solving
the problem from scratch. Verifiers are cheaper than generators and enable:

- **Best-of-N**: generate N candidates, pick the highest-scoring one.
- **Reranking**: sort a beam of partial solutions by predicted correctness.
- **Step-level verification (process reward models)**: score each reasoning step, not
  just the final answer.

Types of verifiers:

| Type | How it works | Best for |
|------|-------------|---------|
| **Outcome reward model (ORM)** | Scores final answer only | Math, code output |
| **Process reward model (PRM)** | Scores each reasoning step | Long proofs, debugging |
| **LLM-as-judge** | Asks an LLM to rate the answer | Open-ended text |
| **Deterministic verifier** | Unit tests, constraint checking | Code, logic puzzles |

For most production use cases, **deterministic verifiers** are cheapest and most
reliable. Use an LLM-as-judge only when there is no ground truth signal.

### LLM-as-judge pattern

An LLM verifier call looks like:

```python
JUDGE_PROMPT = """
You are a mathematical reasoning judge.
Given a problem and a candidate answer, rate the answer on a scale 0–10.
Return only a JSON object: {"score": <int>, "reason": "<one sentence>"}

Problem: {problem}
Answer: {answer}
"""
```

Pitfalls:
- Judges are biased toward longer, more confident-sounding answers.
- Judges agree with themselves (position bias: ask twice, same answer each time).
- Mitigate with: multiple judges, swapped order, reference answer as anchor.

### Evaluating reasoning quality

Beyond answer correctness, you may want to evaluate:

| Metric | How to measure |
|--------|---------------|
| **Accuracy** | `correct / total` on a labelled eval set |
| **Calibration** | Do high-confidence answers actually have higher accuracy? |
| **Reasoning step validity** | Sample-check thinking traces manually |
| **Cost-per-correct-answer** | `total_cost / correct_answers` |
| **Latency-at-accuracy** | Accuracy achieved under a latency SLA |

Build an eval harness, not one-off spot checks. Even 50 labelled examples reveals
calibration patterns a manual review misses.

### Reasoning models + tool use

Extended thinking is **compatible with tool use**. When both are enabled, the model can:
1. Think about what tool to call.
2. Call the tool.
3. Think about the tool's result.
4. Produce a final answer.

This unlocks agentic patterns where the model plans multi-step tool chains before
committing to actions:

```python
response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=12000,
    thinking={"type": "enabled", "budget_tokens": 6000},
    tools=[calculator_tool, search_tool],
    messages=[{"role": "user", "content": "..."}],
)
```

The model's thinking trace will contain reasoning about which tools to call and why —
this makes the agent's plan inspectable before execution.

---

## How it works internally

### Tool use with thinking: the conversation loop

When the model wants to call a tool, `stop_reason == "tool_use"` and `response.content`
contains a `tool_use` block. **You must preserve all blocks** (including `thinking`)
when appending the assistant turn to history before executing the tool:

```python
# Correct: keep ALL content blocks in history
messages.append({"role": "assistant", "content": response.content})

# Execute the tool
tool_result = execute_tool(response.content)

# Append tool result
messages.append({"role": "user", "content": [{"type": "tool_result", ...}]})

# Continue with thinking still enabled
response2 = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=12000,
    thinking={"type": "enabled", "budget_tokens": 6000},
    tools=[...],
    messages=messages,
)
```

Stripping thinking blocks from history causes an API error.

### Eval harness structure

```
eval/
  problems.jsonl       # {"id": "...", "problem": "...", "answer": "..."}
  run_eval.py          # loads problems, calls model, compares, reports
  results/
    2025-01-15.json    # {"accuracy": 0.82, "avg_cost": 0.023, "n": 50}
```

Store eval results by date so you can track accuracy vs. cost over model versions.

---

## Worked example

```python
"""lesson_5_verifier_eval.py — Verifier and eval harness."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@dataclass
class EvalProblem:
    id: str
    problem: str
    answer: str  # ground-truth answer


@dataclass
class EvalResult:
    id: str
    model_answer: str
    correct: bool
    input_tokens: int
    output_tokens: int


# --- Deterministic verifier ---

def exact_match_verifier(model_answer: str, ground_truth: str) -> bool:
    """Exact string match after normalisation."""
    return model_answer.strip().lower() == ground_truth.strip().lower()


def numeric_match_verifier(model_answer: str, ground_truth: str, tol: float = 0.01) -> bool:
    """Float comparison with tolerance."""
    import re
    nums_model = re.findall(r"-?\d+(?:\.\d+)?", model_answer)
    nums_gt = re.findall(r"-?\d+(?:\.\d+)?", ground_truth)
    if not nums_model or not nums_gt:
        return False
    return abs(float(nums_model[-1]) - float(nums_gt[-1])) <= tol


# --- LLM-as-judge verifier ---

JUDGE_PROMPT = """\
You are a reasoning quality judge.
Given a maths problem, a ground-truth answer, and a candidate answer, return:
{{"score": <int 0-10>, "reason": "<one sentence>"}}

Problem: {problem}
Ground truth: {ground_truth}
Candidate answer: {candidate}

Return only valid JSON."""


def llm_judge(
    client: anthropic.Anthropic,
    problem: str,
    ground_truth: str,
    candidate: str,
) -> dict:
    """Call an LLM judge and return the score dict."""
    prompt = JUDGE_PROMPT.format(
        problem=problem,
        ground_truth=ground_truth,
        candidate=candidate,
    )
    response = client.messages.create(
        model="claude-haiku-5-5",  # cheap judge
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# --- Eval harness ---

def run_eval(
    problems: list[EvalProblem],
    solver: Callable[[str], str],
    verifier: Callable[[str, str], bool],
) -> dict:
    """
    Run all problems through solver, score with verifier, report summary.

    solver: callable(problem_text) -> model_answer_text
    verifier: callable(model_answer, ground_truth) -> bool
    """
    results: list[EvalResult] = []

    for prob in problems:
        answer = solver(prob.problem)
        correct = verifier(answer, prob.answer)
        results.append(EvalResult(
            id=prob.id,
            model_answer=answer,
            correct=correct,
            input_tokens=0,   # populate from usage in real impl
            output_tokens=0,
        ))
        log.info("id=%s correct=%s", prob.id, correct)

    n = len(results)
    n_correct = sum(r.correct for r in results)
    accuracy = n_correct / n if n else 0.0

    summary = {
        "n": n,
        "correct": n_correct,
        "accuracy": round(accuracy, 4),
        "results": [asdict(r) for r in results],
    }
    log.info("accuracy=%.2f%%  (%d/%d)", accuracy * 100, n_correct, n)
    return summary


# --- Tool use + thinking agent ---

CALCULATOR_TOOL = {
    "name": "calculate",
    "description": "Evaluate a Python arithmetic expression and return the result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "A safe arithmetic expression"},
        },
        "required": ["expression"],
    },
}


def safe_eval(expression: str) -> str:
    """Evaluate arithmetic only — no arbitrary code."""
    import ast
    import operator as op
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsafe expression: {ast.dump(node)}")

    return str(_eval(ast.parse(expression, mode="eval").body))


def agentic_solver(client: anthropic.Anthropic, problem: str, budget: int = 5000) -> str:
    """Solve a problem with thinking + tool use in a loop."""
    messages = [{"role": "user", "content": problem}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=budget + 2000,
            thinking={"type": "enabled", "budget_tokens": budget},
            tools=[CALCULATOR_TOOL],
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = safe_eval(block.input["expression"])
                    log.info("tool: calculate(%s) = %s", block.input["expression"], result)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            log.warning("unexpected stop_reason: %s", response.stop_reason)
            break

    return ""


if __name__ == "__main__":
    # Demo: eval harness with mocked solver
    problems = [
        EvalProblem(id="p1", problem="What is 2+2?", answer="4"),
        EvalProblem(id="p2", problem="What is 10/2?", answer="5"),
    ]

    def mock_solver(problem: str) -> str:
        return "4" if "2+2" in problem else "5"

    summary = run_eval(problems, mock_solver, exact_match_verifier)
    log.info("summary: %s", summary)
```

---

## Exercises

1. **Process reward model** — Write a `step_verifier(steps: list[str], problem: str) -> list[float]`
   function that calls an LLM judge once per step to score it. Mock the client in tests
   and assert the returned list has the same length as `steps`.

2. **Calibration curve** — Given a list of `(confidence: float, correct: bool)` pairs,
   write `calibration_error(pairs) -> float` that computes the mean absolute difference
   between binned confidence and bin accuracy. Write 3 pytest cases.

3. **Eval harness with cost tracking** — Extend `run_eval` to accept a `usage_tracker`
   callable that receives `(input_tokens, output_tokens)` per call. Return a `total_cost`
   field in the summary using an injected cost table. Test with mocked inputs.

4. **Tool use + thinking test** — Write a pytest test for `agentic_solver` that mocks
   `client.messages.create` to: (1) return `stop_reason="tool_use"` with a `calculate`
   block on the first call, (2) return `stop_reason="end_turn"` with the text answer on
   the second call. Assert the solver returns the text answer and that the second call's
   `messages` contains the tool result.

5. **Multi-judge aggregation** — Write `aggregate_judges(scores: list[dict]) -> float`
   that averages `score` values from multiple LLM judge responses. Handle missing keys
   gracefully. Test with 3 judges returning `8`, `7`, and missing `score`.

(Solutions: [solutions/lesson-5.md](solutions/lesson-5.md))

---

## Self-check

- [ ] I can describe the difference between ORM, PRM, and LLM-as-judge verifiers
- [ ] I can write an eval harness that reports accuracy and cost-per-correct-answer
- [ ] I know to preserve thinking blocks in multi-turn tool-use conversations
- [ ] I can implement a safe arithmetic tool that rejects arbitrary code execution
- [ ] I can build a best-of-N pipeline combining a generator and a verifier

## Supplementary links

- [Let's Verify Step by Step (Lightman et al. 2023)](https://arxiv.org/abs/2305.20050)
- [LLM-as-judge (Zheng et al. 2023)](https://arxiv.org/abs/2306.05685)
- [Tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Extended thinking with tools](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#tool-use-with-extended-thinking)
