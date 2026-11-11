# Phase 5 · Week 05 · Day 4 — Reflexion (Learning from Failures In-Context)

> Objective: Implement the Reflexion pattern where an agent accumulates verbal reflections on past failures and uses them to guide future attempts.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-3]]

## Concept

**Reflexion** (Shinn et al., 2023) extends simple reflection by maintaining a **persistent verbal memory** of past failures. The agent doesn't just revise the current draft — it synthesises *what went wrong and why* into a reflection note, then carries that note into the next attempt as context.

The key idea: the model simulates the role of a researcher annotating their own failed experiments. Each failure produces a lesson. Lessons accumulate. Future attempts inherit all lessons.

### Reflexion vs. plain reflection

| Feature | Plain reflection (Day 1) | Reflexion |
|---|---|---|
| Memory scope | Single iteration | Accumulates across retries |
| What's stored | Raw feedback | Synthesised lesson ("I should ...") |
| Context growth | Fixed | Grows with failures |
| Best for | Polish | Complex tasks requiring strategy change |

### The three-component loop

```
1. Actor:     generate output for task
2. Evaluator: score output (pass/fail)
3. Reflector: if fail → synthesise a verbal lesson
              add lesson to episodic memory
              go to step 1 with memory prepended
```

The **reflector** is a distinct LLM call. Its prompt: "given this task, this attempt, and this failure signal — what should the actor do differently?"

The resulting text is short, first-person, actionable: "I incorrectly assumed the input is always sorted. Next time I should validate input ordering first."

## How it works internally

Reflexion works entirely within the context window. "Memory" is just a list of strings prepended to the actor prompt. There is no external storage.

```python
# Context construction for attempt N
system = (
    "You are a coding agent.\n"
    + (f"Past reflections:\n{chr(10).join(reflections)}\n" if reflections else "")
)
```

This means:
- **Context grows** with each failure. For long tasks, this can exceed context limits. Cap reflections at 3–5 entries and/or truncate older ones.
- **Quality of reflections matters** as much as quality of the actor. A vague reflection ("I made a mistake") adds noise. A specific reflection ("I missed the edge case where n=0") adds signal.

## Worked example

```python
"""
day4_reflexion.py — Reflexion: learning from failures in-context.

Run:  python day4_reflexion.py
Env:  ANTHROPIC_API_KEY
"""
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


@dataclass
class ReflexionAgent:
    """
    A code-writing agent that uses Reflexion: it accumulates verbal lessons
    from failed test runs and injects them into subsequent attempts.
    """

    model: str = "claude-opus-5-5"
    max_attempts: int = 4
    max_reflections: int = 3  # cap to avoid context overflow

    client: anthropic.Anthropic = field(
        default_factory=lambda: anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    )
    _reflections: list[str] = field(default_factory=list, init=False)

    def _build_actor_system(self) -> str:
        base = "You are an expert Python programmer. Write only the function body — no test code, no markdown."
        if self._reflections:
            lessons = "\n".join(f"- {r}" for r in self._reflections[-self.max_reflections :])
            base += f"\n\nLessons from your past failures:\n{lessons}\nApply these lessons now."
        return base

    def _act(self, task: str) -> str:
        log.info("Actor generating code (reflections=%d)...", len(self._reflections))
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self._build_actor_system(),
            messages=[{"role": "user", "content": task}],
        )
        return resp.content[0].text

    def _evaluate(self, code: str, test_code: str) -> tuple[bool, str]:
        """Run tests; return (passed, output)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "solution.py"
            tests = Path(tmpdir) / "test_solution.py"
            src.write_text(code)
            tests.write_text(test_code)
            result = subprocess.run(
                ["python", "-m", "pytest", str(tests), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        return passed, output

    def _reflect(self, task: str, code: str, test_output: str) -> str:
        log.info("Reflector synthesising lesson...")
        prompt = (
            f"Task: {task}\n\n"
            f"Your code attempt:\n```python\n{code}\n```\n\n"
            f"Test failure output:\n{test_output}\n\n"
            "Write ONE concise sentence (15-25 words) starting with 'I should' that captures "
            "the key lesson from this failure. Be specific about the bug, not generic."
        )
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        lesson = resp.content[0].text.strip()
        log.info("Lesson: %s", lesson)
        return lesson

    def run(self, task: str, test_code: str) -> tuple[str, bool, list[str]]:
        """
        Returns (final_code, passed, reflections_accumulated).
        """
        code = ""
        for attempt in range(1, self.max_attempts + 1):
            log.info("=== Attempt %d/%d ===", attempt, self.max_attempts)
            code = self._act(task)
            passed, test_output = self._evaluate(code, test_code)

            if passed:
                log.info("Tests passed on attempt %d!", attempt)
                return code, True, list(self._reflections)

            log.warning("Tests failed on attempt %d.", attempt)
            lesson = self._reflect(task, code, test_output)
            self._reflections.append(lesson)

        log.error("All %d attempts failed.", self.max_attempts)
        return code, False, list(self._reflections)


if __name__ == "__main__":
    # Task: implement a function with a tricky edge case
    TASK = """
    Write a Python function:
        def count_valid_pairs(nums: list[int], target: int) -> int
    
    Return the count of pairs (i, j) where i < j and nums[i] + nums[j] == target.
    Do NOT use nested loops — use a hash map approach for O(n) time.
    Handle edge cases: empty list, single element, duplicate values.
    """

    TEST_CODE = """
from solution import count_valid_pairs

def test_basic():
    assert count_valid_pairs([1, 2, 3, 4], 5) == 2  # (1,4), (2,3)

def test_empty():
    assert count_valid_pairs([], 5) == 0

def test_single():
    assert count_valid_pairs([5], 5) == 0

def test_duplicates():
    assert count_valid_pairs([2, 2, 2], 4) == 3  # (0,1),(0,2),(1,2)

def test_no_pairs():
    assert count_valid_pairs([1, 1, 1], 10) == 0
"""

    agent = ReflexionAgent(max_attempts=4)
    code, passed, reflections = agent.run(TASK, TEST_CODE)
    print(f"\nPassed: {passed}")
    print(f"Reflections accumulated: {reflections}")
    print(f"\nFinal code:\n{code}")
```

## Exercises

1. The current reflector always writes a single sentence. Extend it to produce a structured reflection: `{"lesson": str, "root_cause": str, "fix_strategy": str}`. Store `ReflectionEntry` Pydantic objects instead of plain strings. Update the actor context builder to format these fields clearly.

2. Add a **reflection deduplication** step: before appending a new lesson, call the LLM to check if it's substantially similar to an existing one. If so, merge or discard. Write a unit test with mocked LLM that verifies the dedup fires when reflections are semantically similar.

3. The current evaluator runs `pytest` for code tasks. Write an alternative evaluator `_evaluate_factual(answer: str, gold: str) -> tuple[bool, str]` that uses fuzzy string matching (`difflib.SequenceMatcher`) to check factual answers. When score >= 0.85, return `passed=True`.

4. Implement a `ReflexionMemoryStore` class that persists reflections to a JSON file (keyed by task hash) and loads them at startup. This simulates cross-session memory. Cover it with tests using a `tmp_path` fixture.

(Solutions: ../solutions/week-05/day-4.md)

## Self-check

- [ ] I can explain the difference between Reflexion and simple reflection
- [ ] I can implement the three-component loop (actor, evaluator, reflector)
- [ ] I understand why reflection quality matters as much as acting quality
- [ ] I can cap and trim the reflection memory to avoid context overflow
- [ ] I can mock subprocess test execution in unit tests

---

## Supplementary links

- Shinn et al., "Reflexion" — https://arxiv.org/abs/2303.11366
- Python `subprocess` docs — https://docs.python.org/3/library/subprocess.html
- LangGraph Reflexion implementation reference — https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/
