# Reasoning Models · Capstone Project — Adaptive Problem Router

> Time: ~4 hr · Prereqs: All 5 lessons

---

## Overview

Build a production-quality **adaptive routing layer** that:

1. Classifies each incoming problem as easy, medium, or hard.
2. Routes easy/medium problems to `claude-haiku-5-5` (no thinking).
3. Routes hard problems to `claude-sonnet-5-5` with extended thinking + best-of-N sampling + a verifier.
4. Tracks cost and accuracy across a labelled eval set.
5. Reports a cost/accuracy comparison between the routing strategy and always-thinking and never-thinking baselines.

All LLM calls are **mocked in tests**. The live path requires `ANTHROPIC_API_KEY`.

---

## Specification

### Module structure

```
reasoning_router/
    __init__.py
    classifier.py      # rule-based complexity classifier
    router.py          # routing logic + model caller
    sampler.py         # async best-of-N sampler
    verifier.py        # deterministic + LLM-as-judge verifier
    eval.py            # eval harness + cost tracker
    config.py          # Pydantic settings
    main.py            # CLI entry point
tests/
    test_classifier.py
    test_router.py
    test_sampler.py
    test_verifier.py
    test_eval.py
pyproject.toml
.env.example
```

### config.py

Use `pydantic-settings` to load from environment:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    fast_model: str = "claude-haiku-5-5"
    reasoning_model: str = "claude-sonnet-5-5"
    thinking_budget: int = 8000
    best_of_n: int = 3
    hard_threshold: int = 5

    model_config = {"env_file": ".env"}
```

### classifier.py

`classify(problem: str) -> Complexity`

- Pure function, no LLM call.
- Returns `Complexity.EASY | MEDIUM | HARD`.
- Based on keyword hits, word count, and constraint count (see lesson 3).
- Accept an optional `hard_threshold: int` parameter.

### router.py

`route(problem: str, settings: Settings) -> RouteDecision`

- Calls `classify` and maps to model + thinking config.
- `RouteDecision` is a Pydantic model with: `complexity`, `model`, `thinking_budget | None`, `estimated_cost_usd`.

`async solve(problem: str, settings: Settings) -> SolveResult`

- For EASY/MEDIUM: single call, return answer.
- For HARD: call `sample_n` to get N candidates, call `select_best` to pick the winner.
- `SolveResult`: `answer`, `complexity`, `model`, `thinking_budget`, `input_tokens`, `output_tokens`, `thinking_tokens`, `cost_usd`.

### sampler.py

`async sample_n(problem: str, n: int, settings: Settings) -> list[str]`

- Issues N concurrent `asyncio` calls to the reasoning model.
- Returns list of raw answer strings.

### verifier.py

Two verifiers:

1. `numeric_match(answer: str, ground_truth: str, tol: float = 0.01) -> bool`
2. `llm_judge(client, problem: str, ground_truth: str, candidate: str) -> JudgeResult`
   - Returns `JudgeResult(score: int, reason: str)`.
   - Uses `claude-haiku-5-5` (cheap judge).

`select_best(candidates: list[str], verifier_fn) -> str`

- Applies `verifier_fn` to each candidate and returns the one with the highest score.
- Falls back to `candidates[0]` if all scores are equal.

### eval.py

`async run_eval(problems: list[EvalProblem], settings: Settings) -> EvalReport`

- Solves each problem with `solve()`.
- Compares against ground truth with `numeric_match`.
- Returns `EvalReport(accuracy, total_cost_usd, avg_latency_s, per_problem: list[EvalResult])`.

`compare_strategies(problems, settings) -> StrategyComparison`

- Runs three strategies: routing, always-thinking, never-thinking.
- Returns accuracy and cost for each.

### main.py

CLI entry point:

```bash
# Solve a single problem
python -m reasoning_router.main solve "What is 3^10?"

# Run eval on a JSONL file
python -m reasoning_router.main eval problems.jsonl

# Compare strategies
python -m reasoning_router.main compare problems.jsonl
```

---

## Acceptance criteria

### Functional

- [ ] `classify("What is the capital of France?")` → `EASY`
- [ ] `classify("Prove that the halting problem is undecidable using a diagonalisation argument")` → `HARD`
- [ ] HARD problems go through best-of-N (N=3 by default)
- [ ] EASY problems skip sampling and thinking
- [ ] `run_eval` returns an `EvalReport` with `accuracy` in `[0, 1]`
- [ ] `compare_strategies` returns costs for all three strategies

### Code quality

- [ ] All data crossing module boundaries is typed (Pydantic models or dataclasses)
- [ ] `logging` used throughout, no `print()`
- [ ] `ANTHROPIC_API_KEY` only from `os.getenv` / pydantic-settings
- [ ] No `eval()`, `exec()`, or `subprocess(shell=True)`
- [ ] Ruff passes on all source files

### Tests (all LLM calls mocked)

- [ ] `test_classifier.py` — ≥ 8 cases covering easy/medium/hard with edge cases
- [ ] `test_router.py` — asserts correct model and thinking_budget per complexity level
- [ ] `test_sampler.py` — mock returns N results; asserts N tasks issued
- [ ] `test_verifier.py` — numeric_match with ints, floats, tolerances, no-match
- [ ] `test_eval.py` — full eval with 3 problems, asserts accuracy + cost in summary
- [ ] All tests pass: `pytest -v`

---

## Setup

```bash
uv init reasoning_router
cd reasoning_router
uv add anthropic pydantic-settings pytest pytest-asyncio
```

`.env.example`:
```
ANTHROPIC_API_KEY=your-key-here
```

`.gitignore` must include `.env`.

---

## Deliverables

1. All source files under `reasoning_router/`
2. All test files under `tests/`
3. `pyproject.toml` with dependencies
4. `.env.example`
5. A `results/` directory with a sample `eval_report.json` (can be generated from mocked run)

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Pass criteria |
|-----------|--------|--------------|
| Routing logic correct | 30% | EASY → haiku, HARD → sonnet + thinking + best-of-N |
| Test coverage (mocked) | 25% | All 5 test files present and passing |
| Cost/accuracy tracking | 20% | `compare_strategies` returns correct structure |
| Code quality | 15% | Typed, logged, ruff-clean, no secrets |
| CLI functional | 10% | `main.py` runs all three subcommands |

Answer key: [solutions/project.md](solutions/project.md)
