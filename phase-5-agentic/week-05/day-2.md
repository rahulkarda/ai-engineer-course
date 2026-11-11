# Phase 5 · Week 05 · Day 2 — Self-Consistency & Voting

> Objective: Generate multiple independent responses and aggregate them via majority vote or scoring to improve reliability.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

**Self-consistency** (Wang et al., 2022) answers a question by sampling the model multiple times and taking the majority answer. The insight: LLMs are stochastic. Different samples explore different reasoning paths. The most common answer across paths is more likely correct than any single answer.

Contrast with reflection:
- **Reflection** iterates: generate → critique → improve (serial, one answer evolves).
- **Self-consistency** samples: generate N times → vote (parallel, pick the winner).

Self-consistency shines on tasks with a definite right answer: math, code, factual Q&A, classification. Reflection shines on open-ended tasks where "quality" is about structure, completeness, or tone.

### Voting strategies

| Strategy | When to use |
|---|---|
| **Majority vote** | Discrete outputs (classification, multiple-choice) |
| **Weighted majority** | When you can score confidence |
| **LLM judge** | Open-ended outputs where you can't count tokens |
| **Ensemble mean** | Numerical outputs |

### Temperature matters

Self-consistency requires diversity in samples. Use `temperature ≥ 0.7` (or the model's default, which is already >0). At `temperature=0` all samples are identical — no benefit.

## How it works internally

Each sample is an independent API call. The model state is not shared between them. You're running N separate completions and post-processing the results.

```
samples = [generate(task, temperature=0.9) for _ in range(N)]
answer = majority_vote(extract_answer(s) for s in samples)
```

**Cost**: N × single-call cost. Use sparingly. N=5 is a practical sweet spot — meaningful diversity without 10× cost.

**Extraction**: raw responses differ in formatting. A dedicated extraction step (regex, another LLM call, or structured output) normalises before voting. This is often the fragile part.

## Worked example

```python
"""
day2_self_consistency.py — Self-consistency with majority voting.

Run:  python day2_self_consistency.py
Env:  ANTHROPIC_API_KEY
"""
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class SelfConsistency:
    client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    )
    model: str = "claude-opus-5-5"
    n_samples: int = 5
    temperature: float = 0.9

    def _sample(self, task: str, sample_id: int) -> str:
        log.info("Sampling %d/%d...", sample_id + 1, self.n_samples)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{task}\n\n"
                        "Think step by step, then end with: ANSWER: <your final answer>"
                    ),
                }
            ],
        )
        return resp.content[0].text

    def _extract_answer(self, response: str) -> str | None:
        """Extract text after 'ANSWER:' marker."""
        match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        log.warning("Could not extract answer from: %s...", response[:80])
        return None

    def run(self, task: str) -> dict:
        """
        Returns {
            "winner": str,
            "votes": Counter,
            "samples": list[str],
            "confidence": float
        }
        """
        samples: list[str] = []

        # Parallel sampling for speed
        with ThreadPoolExecutor(max_workers=self.n_samples) as pool:
            futures = {
                pool.submit(self._sample, task, i): i
                for i in range(self.n_samples)
            }
            for fut in as_completed(futures):
                samples.append(fut.result())

        answers = [self._extract_answer(s) for s in samples]
        valid = [a for a in answers if a is not None]
        log.info("Extracted %d/%d valid answers", len(valid), self.n_samples)

        if not valid:
            raise ValueError("No valid answers extracted from any sample.")

        votes: Counter[str] = Counter(valid)
        winner, top_count = votes.most_common(1)[0]
        confidence = top_count / len(valid)

        log.info("Winner=%r confidence=%.0f%%", winner, confidence * 100)
        return {
            "winner": winner,
            "votes": votes,
            "samples": samples,
            "confidence": confidence,
        }


if __name__ == "__main__":
    sc = SelfConsistency(n_samples=5)
    task = (
        "A store sells apples at $0.50 each and oranges at $0.75 each. "
        "Alice buys 4 apples and 3 oranges. How much does she spend in total?"
    )
    result = sc.run(task)
    print(f"\nWinner: {result['winner']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"All votes: {dict(result['votes'])}")
```

### Voting with an LLM judge (open-ended tasks)

When answers cannot be normalised to a single token, a judge aggregates:

```python
def llm_vote(client: anthropic.Anthropic, task: str, samples: list[str]) -> str:
    """Ask a judge to pick the best answer from multiple samples."""
    numbered = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(samples))
    prompt = (
        f"Task: {task}\n\nHere are {len(samples)} candidate answers:\n{numbered}\n\n"
        "Which answer is most correct and complete? Reply with only the number."
    )
    resp = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    idx = int(resp.content[0].text.strip()) - 1
    return samples[idx]
```

This adds a second LLM call but handles free-form responses where regex extraction fails.

## Exercises

1. The current implementation runs all samples concurrently. Add a `timeout_seconds: float = 30` parameter; if any sample exceeds the timeout, skip it and log a warning (don't raise). Write a test using `unittest.mock` to simulate a timed-out thread.

2. Implement a `weighted_vote` function that takes `answers: list[str]` and `scores: list[float]` and returns the weighted-majority winner. Cover edge cases: tie, empty inputs.

3. Self-consistency is expensive for open-ended tasks. Add a `confidence_threshold: float = 0.6` early-exit: if after any sample the current leader's confidence exceeds the threshold, stop sampling early and return. Write a test that asserts sampling stops early when the threshold is hit.

4. The extraction regex is fragile. Replace it with a structured output approach: ask the model to respond as JSON `{"reasoning": str, "answer": str}`, parse with Pydantic. Verify the new extractor in a unit test.

(Solutions: ../solutions/week-05/day-2.md)

## Self-check

- [ ] I can explain when self-consistency beats reflection and vice versa
- [ ] I can implement parallel sampling with `ThreadPoolExecutor`
- [ ] I understand why temperature must be > 0 for self-consistency
- [ ] I can implement majority vote from a `Counter`
- [ ] I can write an LLM judge fallback for open-ended tasks

---

## Supplementary links

- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning" — https://arxiv.org/abs/2203.11171
- Python `concurrent.futures` docs — https://docs.python.org/3/library/concurrent.futures.html
- Anthropic temperature guidance — https://docs.anthropic.com/en/api/messages
