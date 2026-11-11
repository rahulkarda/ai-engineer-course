# Reasoning Models · Lesson 4 — Prompting Reasoning Models

> Objective: Apply "less is more" prompting, self-consistency sampling, and best-of-N selection.
> Time: ~1 hr · Prereqs: [lesson-3.md](lesson-3.md)

---

## Concept

### Less is more

Standard prompt engineering wisdom says: give the model step-by-step instructions,
examples of the reasoning process, and explicit hints about how to think. For reasoning
models this advice **inverts**.

When extended thinking is enabled, the model has its own internal scratchpad. If you
pre-specify the reasoning path in the prompt ("First do X, then Y, then Z"), you:

1. **Constrain the search space** — the model follows your prescribed path even when a
   better path exists.
2. **Waste output tokens** — the model echoes your steps back rather than exploring.
3. **Reduce surprise generalisation** — hand-specified CoT prompts work well on examples
   resembling the ones you wrote them for, and worse on novel problems.

For reasoning models, the correct prompting style is:

- State the **problem clearly and completely**.
- State the **output format required** (structured JSON, a number, a decision).
- State any **hard constraints** (time limit, forbidden operations).
- Do **not** tell the model how to think.

### Over-specifying CoT: what goes wrong

```python
# Bad: over-specified CoT
prompt = """
Solve this maths problem.
Step 1: Identify the variables.
Step 2: Write the equations.
Step 3: Solve the system.
Step 4: Verify your answer.
The problem is: ...
"""

# Good: clean problem statement
prompt = """
Find the values of x and y that satisfy both equations simultaneously.
Return your answer as JSON: {"x": <number>, "y": <number>}

Equations:
2x + 3y = 12
x - y = 1
"""
```

The second prompt lets the model choose its own solution strategy (substitution,
elimination, matrix) and produces a verifiable structured output.

### Self-consistency

A single model call can produce a wrong answer even with extended thinking. Running
the same prompt N times and taking the majority answer (**self-consistency**) improves
reliability significantly on problems with discrete, verifiable answers.

Algorithm:
1. Sample N responses with temperature > 0.
2. Extract the final answer from each.
3. Return the answer that appears most often.

Self-consistency works best when:
- The answer space is discrete (numbers, classifications, yes/no).
- N ≥ 5 (fewer samples → majority is unreliable).
- Temperature is 0.6–1.0 (at temperature 0 all samples are identical).

### Best-of-N

A variant of self-consistency where instead of voting, you score each candidate with a
separate verifier and return the highest-scoring answer:

```
N completions → [verifier scores each] → return argmax(scores)
```

Best-of-N is stronger than majority voting when:
- The answer space is too large for counting (e.g., open-ended code).
- A verifier can score answers cheaper than generating them.
- You have a ground-truth signal (unit tests, known constraints).

---

## How it works internally

### Temperature with extended thinking

When `thinking` is enabled, Claude's API requires `temperature=1` (the default). You
cannot lower temperature to make thinking deterministic — the model's internal search
benefits from stochastic exploration. Do not try to set temperature to 0 for consistency;
use majority voting or verifiers instead.

### Extracting structured answers from thinking responses

The thinking trace may repeat candidate answers and then revise them. Never parse the
thinking trace for the final answer — always read the `text` block:

```python
def extract_answer(response) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("No text block in response")
```

### Sampling N responses

The Anthropic API does not support a `n=5` parameter (unlike OpenAI). You must issue N
separate calls. Use `asyncio` to run them in parallel:

```python
import asyncio
import anthropic

async def sample_once(client: anthropic.AsyncAnthropic, prompt: str, budget: int) -> str:
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_answer(response)

async def sample_n(prompt: str, n: int, budget: int) -> list[str]:
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tasks = [sample_once(client, prompt, budget) for _ in range(n)]
    return await asyncio.gather(*tasks)
```

---

## Worked example

```python
"""lesson_4_self_consistency.py — Self-consistency + best-of-N."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import Counter

import anthropic

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_answer(response: anthropic.types.Message) -> str:
    """Extract the text block from a response."""
    for block in response.content:
        if block.type == "text":
            return block.text
    raise ValueError("No text block found")


def extract_numeric(text: str) -> str | None:
    """Pull the last number from an answer string."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return matches[-1] if matches else None


async def _call_once(
    client: anthropic.AsyncAnthropic,
    prompt: str,
    budget: int,
) -> str:
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=budget + 2000,
        thinking={"type": "enabled", "budget_tokens": budget},
        messages=[{"role": "user", "content": prompt}],
    )
    return extract_answer(response)


async def sample_n_async(prompt: str, n: int = 5, budget: int = 4000) -> list[str]:
    """Issue N parallel calls and return raw answer strings."""
    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    tasks = [_call_once(client, prompt, budget) for _ in range(n)]
    results = await asyncio.gather(*tasks)
    log.info("sampled %d responses", len(results))
    return list(results)


def majority_vote(answers: list[str]) -> str:
    """Extract numeric answers and return the majority."""
    extracted = [extract_numeric(a) for a in answers]
    valid = [x for x in extracted if x is not None]
    if not valid:
        return answers[0]  # fallback: return first
    counts = Counter(valid)
    winner, count = counts.most_common(1)[0]
    log.info("vote: %s (count=%d/%d)", winner, count, len(valid))
    return winner


def best_of_n(answers: list[str], verifier) -> str:
    """
    Score each answer with `verifier(answer) -> float` and return the best.

    verifier: callable that returns a score in [0, 1].
    """
    scored = [(verifier(a), a) for a in answers]
    scored.sort(reverse=True)
    log.info("best score=%.2f", scored[0][0])
    return scored[0][1]


# Example verifier: prefers answers that contain a number
def numeric_verifier(answer: str) -> float:
    return 1.0 if extract_numeric(answer) is not None else 0.0


async def main() -> None:
    problem = (
        "A train leaves station A at 60 mph. Another leaves station B (120 miles away) "
        "heading toward A at 40 mph. They depart simultaneously. "
        "After how many hours do they meet? Return only the number."
    )

    answers = await sample_n_async(problem, n=5, budget=3000)
    for i, a in enumerate(answers):
        log.info("sample %d: %s", i, a[:100])

    majority = majority_vote(answers)
    log.info("majority vote answer: %s", majority)

    best = best_of_n(answers, numeric_verifier)
    log.info("best-of-N answer: %s", best[:100])


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Exercises

1. **Prompt comparison** — Write two prompts for the same logic puzzle: one with
   explicit step-by-step instructions and one clean problem statement. Mock the API to
   return distinct answers and write a test that asserts both prompts produce a valid
   result (without testing which is "better" — this is a structural test).

2. **Majority vote edge cases** — Write tests for `majority_vote` covering: all answers
   identical, no numeric answer found, three-way tie (assert it returns one of the
   top-frequency answers), single answer.

3. **Custom verifier** — Write a `json_verifier(answer: str) -> float` that returns 1.0
   if the answer is valid JSON with keys `"x"` and `"y"`, else 0.0. Test it with
   valid and invalid inputs.

4. **Async sampling test** — Mock `anthropic.AsyncAnthropic.messages.create` to return a
   stub response. Write a pytest-asyncio test that calls `sample_n_async` with `n=3`
   and asserts exactly 3 results are returned.

5. **Aggregator class** — Refactor the sampling + voting into a `SelfConsistency` class
   with `__init__(self, n: int, budget: int)` and `async solve(self, prompt: str) -> str`.
   Write a unit test using `AsyncMock`.

(Solutions: [solutions/lesson-4.md](solutions/lesson-4.md))

---

## Self-check

- [ ] I can explain why over-specifying CoT hurts reasoning model performance
- [ ] I know the correct temperature setting for extended thinking calls
- [ ] I can implement self-consistency sampling with asyncio.gather
- [ ] I can implement majority-vote and best-of-N selection strategies
- [ ] I know when best-of-N outperforms majority voting

## Supplementary links

- [Self-Consistency improves CoT reasoning (Wang et al. 2022)](https://arxiv.org/abs/2203.11171)
- [Let's Verify Step by Step (process reward, Lightman 2023)](https://arxiv.org/abs/2305.20050)
- [Prompting extended thinking guide](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#tips-for-prompting-extended-thinking)
