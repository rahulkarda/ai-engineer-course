# Week 05 Assessment — Reflection, Self-Critique & Supervision

> Time: ~1 hr  ·  Pass gate: 75%

## Conceptual (interview-style)

Answer each in 3–5 sentences. Precision matters more than length.

1. Explain the difference between **plain reflection** (Day 1) and **Reflexion** (Day 4). Why does verbal memory accumulation matter for multi-attempt tasks?

2. A colleague proposes using self-consistency (5 samples) on every LLM call in their app to "improve quality." What three questions would you ask before agreeing or disagreeing?

3. You have a critic agent with a rubric and a self-reflection loop. Which should you use for evaluating code correctness and why? Which for evaluating a product marketing email and why?

4. A guardrail's PII check fires a WARN on an output. The supervisor has `max_retries=2` and both retries produce the same WARN. What should the supervisor do, and how would you implement that policy in code?

5. What is the token cost model for a Reflexion loop with 3 attempts? Write the formula in terms of `P` (tokens per generation), `R` (tokens per reflection call), and `E` (tokens per evaluation call).

6. Describe a task where adding a critic agent would actively make output quality **worse**. Explain the mechanism.

7. A supervisor's LLM-based guardrail says BLOCK on an output that is clearly safe. Name two likely causes and how you would debug each.

---

## Coding challenge

### Spec: `ReviewBot`

Build a `ReviewBot` class that takes a piece of text (e.g. a code snippet or a paragraph) and returns a structured review.

**Requirements:**

1. `ReviewBot` accepts a `rubric: list[str]` (list of evaluation criteria as plain-English strings) at construction time.

2. Method `review(text: str) -> ReviewResult` where `ReviewResult` is a Pydantic model with:
   - `passed: bool`
   - `scores: dict[str, int]`  (criterion → score 0–10)
   - `feedback: str`
   - `suggested_revision: str | None`

3. If any score is below 6, `passed = False` and `suggested_revision` must be non-None.

4. If `passed = False`, `ReviewBot.improve(text: str, review: ReviewResult) -> str` calls the LLM to produce a revised text incorporating the feedback.

5. `ReviewBot.review_and_improve(text: str, max_rounds: int = 2) -> tuple[str, ReviewResult]` loops `review → improve` until passed or rounds exhausted.

**Acceptance criteria:**
- All LLM calls use `os.getenv("ANTHROPIC_API_KEY")` — no hardcoded keys
- Model ID: `claude-opus-5-5` (or `claude-haiku-5-5` for the reviewer to save cost)
- Every public method has a type annotation
- A `tests/test_reviewbot.py` file with at least 4 tests, all LLM calls mocked
- Tests pass with `pytest -v` and zero failures

**Bonus (+5%):** Add a `ReviewBot.batch_review(texts: list[str]) -> list[ReviewResult]` method that runs reviews concurrently using `ThreadPoolExecutor`.

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | `ReviewBot` works end-to-end; `review_and_improve` loop terminates correctly; Pydantic models used |
| Conceptual understanding | 25% | Written answers are precise, show awareness of trade-offs, not surface-level |
| Debugging & problem-solving | 20% | Tests actually mock the LLM; edge cases handled (empty rubric, max_rounds=1, all pass first try) |
| Code quality & testing | 15% | Type hints throughout; `logging` not `print`; at least 4 passing tests |

**Total: 100% — pass gate: 75%**

Answer key: ../solutions/week-05/assessment.md
