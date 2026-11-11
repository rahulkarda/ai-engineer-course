# Week 11 Assessment — Pydantic AI, Agent Evaluation & Observability

> Pass gate: 75% · Time: ~1 hr · Answer key: ../solutions/week-11/assessment.md

## Conceptual (interview-style)

1. What does the generic `Agent[DepsType, ResultType]` signature enforce, and why is it preferable to an untyped `Agent`?

2. Explain the difference between `@agent.tool` and `@agent.tool_plain`. In what situation must you use `@agent.tool`?

3. A tool function returns a `list[str]`. The LLM sends tool call arguments `{"count": "five"}` where `count` is typed as `int`. What happens in Pydantic AI, and what does the model see?

4. You have two agents that both answer a question correctly, but Agent A takes 2 tool calls and Agent B takes 6. Why does this matter from an evaluation perspective? Name two metrics that would catch this difference.

5. What is a golden trajectory? Describe the full lifecycle from capture to CI gate failure.

6. When is an LLM judge more appropriate than an exact-match metric? Give a concrete example from agent evaluation.

7. Explain the relationship between traces, spans, and a single `agent.run()` call. What does parent-child span structure tell you?

8. What does `result.usage()` return in Pydantic AI, and what are the two main fields you use to compute cost?

9. What is the role of `ModelRetry` in a tool? How is it different from raising a plain `ValueError`?

10. Describe two ways Pydantic AI output validation (on `result_type`) improves reliability compared to manually parsing `json.loads(result.text)`.

---

## Coding challenge

**Spec:** Build a typed `SummaryAgent` with the following requirements:

### Agent contract

```python
@dataclass
class SummaryDeps:
    max_words: int
    language: str  # e.g. "English", "German"

class SummaryResult(BaseModel):
    summary: str
    word_count: int
    language_detected: str
```

### Tools required

1. `count_words(text: str) -> int` — plain tool, counts words.
2. `detect_language(ctx: RunContext[SummaryDeps], text: str) -> str` — returns `ctx.deps.language` (simulates language detection from deps).
3. `truncate(text: str, max_words: int) -> str` — plain tool, returns first `max_words` words joined by space.

### Evaluation harness requirements

- Write `EvalCase` records for at least 2 inputs.
- Implement `score_summary(result: SummaryResult, expected_word_count: int) -> bool` that passes if `result.word_count <= expected_word_count`.
- Create a `GoldenStore` (in-memory / tmp_path) with one approved golden.
- Run `RegressionGate.evaluate()` on a passing case and assert no regression.

### Observability requirements

- Use `CostTracker` to record at least one run (mocked usage: 500 input, 100 output tokens) with `claude-haiku-5-5`.
- Assert `summary()["total_tokens"] == 600`.

### Testing requirements

- All LLM calls must use `TestModel` or `unittest.mock` — **no real API keys**.
- Minimum 5 pytest tests covering: tool correctness, `score_summary`, regression gate, cost tracker.
- Tests run with `pytest -x` and pass.

### Acceptance criteria

- [ ] `SummaryAgent` is `Agent[SummaryDeps, SummaryResult]`
- [ ] All three tools are registered with type-safe signatures
- [ ] `SummaryResult.word_count` matches actual word count of `summary`
- [ ] Regression gate correctly returns `None` for a passing case
- [ ] `CostTracker.summary()["total_tokens"]` is correct
- [ ] 5+ tests pass with no real LLM calls

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Independent implementation | 40% | Agent, tools, eval harness, tracker — all wired correctly without reference |
| Conceptual understanding | 25% | Conceptual questions answered with precision; no vague generalities |
| Debugging & problem-solving | 20% | Tests cover edge cases; `ModelRetry` used appropriately |
| Code quality & testing | 15% | Type hints throughout; `logging` not `print`; pytest structure; no hardcoded secrets |

Answer key: ../solutions/week-11/assessment.md
