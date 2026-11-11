# Week 11 Project — Agent Evaluation & Observability Harness

> Time: ~4 hr (Saturday) + ~3 hr (Sunday)  ·  Prereqs: week-11 days 1–6

## Overview

Take the code review agent from Day 2 and build a production-ready harness around it:
- **Pydantic AI typing** — `Agent[ReviewDeps, ReviewResult]`, all tools typed
- **Evaluation** — golden trajectories, task-success scoring, LLM judge
- **Regression gate** — CI-ready gate that fails on trajectory drift
- **Observability** — spans, token tracking, cost reporting

All LLM calls mocked in tests. No real API keys required.

---

## Deliverables

### 1. Typed agent (`src/agents/code_review_agent.py`)

Requirements:
- `Agent[ReviewDeps, ReviewResult]` using `anthropic:claude-sonnet-5-5`
- `ReviewDeps(allowed_imports: list[str], max_complexity: int, base_dir: pathlib.Path)`
- `ReviewResult(issues: list[str], severity: Annotated[str, Field(pattern="^(low|medium|high)$")], recommendation: str)`
- Tools: `count_functions`, `check_imports`, `estimate_complexity`, `detect_security_issues`
- `detect_security_issues` checks for `eval(`, `exec(`, `subprocess(shell=True`, `pickle.load`
- Dynamic system prompt that appends `max_complexity` to the prompt

### 2. Evaluation harness (`src/eval/`)

Files: `trajectory_eval.py`, `llm_judge.py`, `golden_store.py`, `regression_gate.py`

Requirements:
- `Trajectory` + `extract_trajectory(messages)` correctly parses all tool calls
- `score_trajectory(actual, golden)` returns `TrajectoryScore` with `tool_name_match`, `step_count_delta`, `exact_match`
- `JudgeVerdict` with `factual_accuracy`, `completeness`, `conciseness` scores (1–5) and `passed`
- `GoldenStore` saves/loads/approves `GoldenRecord` JSON files
- `RegressionGate` with configurable `max_step_delta` and `min_name_match`

### 3. Observability (`src/observability/`)

Files: `cost_tracker.py`, `tracer.py`

Requirements:
- `CostTracker` with pricing for all three models: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`
- `cost_usd(model, usage)` computes correct USD from `Usage.request_tokens` and `Usage.response_tokens`
- `Tracer` with async context-manager `span(name, trace_id, ...)` that records start/end time
- `Tracer.export_json()` returns list of span dicts

### 4. Golden capture script (`scripts/capture_golden.py`)

Requirements:
- Accepts a list of `(case_id, input)` pairs from a JSONL file (passed as CLI arg)
- Runs the agent with `TestModel` to produce deterministic trajectories
- Saves and auto-approves golden records to `tests/golden/`
- Logs a summary table: case_id, tool calls, output snippet

### 5. Tests (`tests/`)

Minimum 10 tests:

| Test | File | What it verifies |
|------|------|-----------------|
| `test_count_functions_basic` | unit/test_tools.py | Returns correct count for known code |
| `test_detect_security_issues` | unit/test_tools.py | Detects `eval(` and `exec(` |
| `test_extract_trajectory` | unit/test_trajectory.py | Parses tool call messages |
| `test_score_trajectory_exact` | unit/test_trajectory.py | Exact match = name_match 1.0, delta 0 |
| `test_score_trajectory_drift` | unit/test_trajectory.py | Extra steps detected |
| `test_gate_passes` | unit/test_gate.py | No regression on matching run |
| `test_gate_task_failure` | unit/test_gate.py | Returns regression on task failure |
| `test_cost_usd_sonnet` | unit/test_cost.py | Correct cost for known token count |
| `test_judge_verdict_validation` | unit/test_judge.py | Pydantic rejects score > 5 |
| `test_judge_mocked` | unit/test_judge.py | Returns correct verdict with mocked model |
| `test_eval_suite_integration` | integration/test_suite.py | Full suite run passes regression gate |

All tests must pass with `pytest -x`. No real API calls allowed.

---

## Stretch goals

- Implement `Tracer.waterfall(trace_id)` ASCII diagram.
- Add `JudgeCalibrationTest` with known ground truths; assert judge accuracy ≥ 80%.
- Expose `get_summary()` via a FastAPI `/metrics` endpoint.
- Add `budget_alert(max_cost_usd)` to `CostTracker`; write a test that triggers it.

---

## Submission checklist

- [ ] All deliverables present under `week-11-project/`
- [ ] `pytest -x` passes with 10+ tests, zero failures
- [ ] No hardcoded API keys anywhere; `os.getenv("ANTHROPIC_API_KEY")` pattern used
- [ ] `logging` used throughout — no `print()` in production code
- [ ] All public functions and classes have type hints
- [ ] `pyproject.toml` lists `pydantic-ai` as dependency (not installed, just declared)
- [ ] `README.md` (one paragraph) describes what the project does and how to run tests

## Solution

Full reference implementation: ../solutions/week-11/project.md
