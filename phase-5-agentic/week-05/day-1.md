# Phase 5 · Week 05 · Day 1 — Reflection Pattern (Self-Critique)

> Objective: Implement a self-critique loop where an agent evaluates its own output and revises until quality passes.  ·  Time: ~1 hr  ·  Prereqs: [[week-04/day-7]]

## Concept

**Reflection** is the pattern where a model generates an output, then critiques that output (as a separate reasoning step), then revises based on the critique. The cycle repeats until either the critique finds no significant flaws or a maximum iteration cap is reached.

This is not magic — it works because:
1. **Generation vs. evaluation are different cognitive tasks.** A model may write a flawed answer but recognize the flaw when asked "what's wrong with this?"
2. **Separating the roles reduces anchoring.** A fresh prompt for critique resets the context window enough that the model applies a different lens.
3. **Iteration compounds.** Each cycle tends toward better output, with diminishing returns after 2–3 passes.

The pattern was formalized in the *Reflexion* paper (Shinn et al., 2023) and is foundational to most agent self-improvement loops.

### Anatomy of a reflection loop

```
generate(task) → draft
critique(draft) → feedback
if feedback == "PASS": return draft
revise(draft, feedback) → new_draft
repeat (max N times)
```

The **critic prompt** is as important as the generator prompt. A vague critic ("is this good?") yields vague feedback. A structured critic that checks specific axes (correctness, completeness, tone, safety) produces actionable revisions.

## How it works internally

At the LLM level, each step is an independent API call. The model has no persistent memory across calls — you're manually threading context. The typical context packet passed to the reviser contains:

```
[system: role of reviser]
[user: original task]
[assistant: draft]
[user: critique]
[user: "Now write an improved version."]
```

This multi-turn construction ensures the model sees: what was asked, what it said, what was wrong, and what to do next.

**Token budget**: reflection triples (at minimum) your token cost per task. Budget accordingly. Use it only when output quality is critical and the task has objectively evaluable dimensions.

**Termination**: always cap iterations. An uncapped reflection loop can spiral when the model hallucinates critique issues that don't exist. Max 3 iterations is a sensible default.

## Worked example

```python
"""
day1_reflection.py — Self-critique reflection loop using Claude.

Run:  python day1_reflection.py
Env:  ANTHROPIC_API_KEY
"""
import logging
import os
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class CritiqueResult(BaseModel):
    passed: bool
    feedback: str
    score: int  # 0-10


@dataclass
class ReflectionLoop:
    client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    )
    model: str = "claude-opus-5-5"
    max_iterations: int = 3

    def _generate(self, task: str) -> str:
        log.info("Generating initial draft...")
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text

    def _critique(self, task: str, draft: str) -> CritiqueResult:
        log.info("Critiquing draft...")
        prompt = (
            f"Original task:\n{task}\n\n"
            f"Draft answer:\n{draft}\n\n"
            "Evaluate this draft on: (1) correctness, (2) completeness, (3) clarity.\n"
            "Respond as JSON: {\"passed\": bool, \"feedback\": str, \"score\": 0-10}.\n"
            "passed=true only if score >= 8 and no critical flaws exist."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        import json
        raw = resp.content[0].text.strip()
        # strip code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return CritiqueResult(**data)

    def _revise(self, task: str, draft: str, feedback: str) -> str:
        log.info("Revising draft based on feedback...")
        messages = [
            {"role": "user", "content": task},
            {"role": "assistant", "content": draft},
            {
                "role": "user",
                "content": (
                    f"Your draft has issues:\n{feedback}\n\n"
                    "Please write an improved version addressing all points."
                ),
            },
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=messages,
        )
        return resp.content[0].text

    def run(self, task: str) -> tuple[str, int]:
        """Returns (final_answer, iterations_used)."""
        draft = self._generate(task)
        for iteration in range(1, self.max_iterations + 1):
            critique = self._critique(task, draft)
            log.info("Iteration %d — score=%d passed=%s", iteration, critique.score, critique.passed)
            if critique.passed:
                log.info("Reflection passed after %d iteration(s).", iteration)
                return draft, iteration
            if iteration < self.max_iterations:
                draft = self._revise(task, draft, critique.feedback)
        log.warning("Max iterations reached without passing critique.")
        return draft, self.max_iterations


if __name__ == "__main__":
    loop = ReflectionLoop(max_iterations=3)
    task = (
        "Explain, in 3 bullet points, why gradient descent can get stuck "
        "in local minima and two practical techniques to escape them."
    )
    answer, iters = loop.run(task)
    print(f"\n=== Final answer (after {iters} iteration(s)) ===\n{answer}")
```

Key design decisions:
- Each step (`_generate`, `_critique`, `_revise`) is a pure function — easy to swap individual pieces.
- `CritiqueResult` is a Pydantic model — structured, validated.
- `max_iterations` is a hard cap to prevent runaway loops.
- Logging at each stage makes debugging visible without polluting stdout.

## Exercises

1. The current critique checks three axes uniformly. Modify `_critique` to accept a list of `axes: list[str]` and include them in the prompt dynamically. Write a test that mocks the LLM and verifies the axes appear in the constructed prompt.

2. The loop stops on the first passing critique. Add a `min_score: int = 8` parameter to `ReflectionLoop`. Revise the termination logic so it uses `min_score` rather than the hardcoded value inside `_critique`.

3. The revise step only sends feedback as plain text. Extend it to send structured feedback: each axis + its score + a specific suggestion. Update `CritiqueResult` to carry `axis_scores: dict[str, int]` and pass them to the reviser.

4. Implement a `ReflectionHistory` dataclass that stores each `(draft, critique)` pair per iteration and returns it alongside the final answer. Write a test asserting the history length equals iterations used.

(Solutions: ../solutions/week-05/day-1.md)

## Self-check

- [ ] I can explain why generating and critiquing are separate prompts (not combined)
- [ ] I can implement a 3-step reflection loop (generate → critique → revise) without reference
- [ ] I can write a structured critic prompt with specific evaluation axes
- [ ] I understand why max_iterations is mandatory
- [ ] I can mock the LLM calls in a pytest test for this loop

---

## Supplementary links

- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) — https://arxiv.org/abs/2303.11366
- Anthropic multi-turn messages docs — https://docs.anthropic.com/en/api/messages
- "Constitutional AI" (self-critique at training scale) — https://arxiv.org/abs/2212.08073
