# Week 05 Project — Reflection + Supervision Wrapper

> Time: ~3 hr (Day 6 build + Day 7 polish)  ·  Prerequisite: Days 1–5 complete

## Project: `AgentGuard`

Build `AgentGuard` — a reusable Python package that wraps any callable "worker" with reflection, critic evaluation, and supervision guardrails.

The design goal: a developer should be able to drop `AgentGuard` in front of any existing LLM call and get critic-loop retry + policy guardrails with minimal changes to their code.

---

## Specification

### Package structure

```
agent_guard/
├── __init__.py           # exports: AgentGuard, Rubric, GuardrailSet, RunResult
├── models.py             # all Pydantic models
├── guardrails.py         # composable guardrail functions
├── critic.py             # CriticAgent
├── worker.py             # WorkerAgent (or any callable wrapper)
├── reflection.py         # ReflectionLoop with Reflexion memory
├── supervisor.py         # SupervisorAgent
└── guard.py             # AgentGuard façade class
tests/
├── conftest.py           # mock_anthropic fixture
├── test_guardrails.py
├── test_critic.py
├── test_reflection.py
├── test_supervisor.py
└── test_guard.py         # integration tests via façade
pyproject.toml
```

### `AgentGuard` façade

```python
from agent_guard import AgentGuard, Rubric, GuardrailSet

guard = AgentGuard(
    rubric=Rubric(
        axes=[
            {"name": "accuracy", "description": "Factually correct", "pass_threshold": 7},
            {"name": "clarity", "description": "Clear prose", "pass_threshold": 6},
        ]
    ),
    guardrails=GuardrailSet(
        input_checks=["prompt_injection"],
        output_checks=["pii", "length"],
        max_output_chars=2000,
    ),
    max_reflection_attempts=3,
    reflexion_enabled=True,
)

result = guard.run("Explain transformer self-attention in 3 sentences.")
print(result.final_output)
print(f"Passed: {result.passed}, Attempts: {result.attempts}")
```

### Required features

**F1 — Configurable rubric**
- `Rubric` accepts a list of axes (name, description, pass_threshold)
- `CriticAgent` uses the rubric; PASS only if all dimensions >= threshold AND overall >= 7

**F2 — Composable guardrail set**
- `GuardrailSet` lists which checks to apply by name
- Built-in checks: `prompt_injection`, `pii`, `length`, `topic_block`
- Custom checks: `GuardrailSet(custom_checks=[my_fn])` where `my_fn: str -> GuardrailResult`

**F3 — Reflexion memory**
- When `reflexion_enabled=True`, failed attempts produce a verbal lesson
- Lessons are injected into subsequent worker prompts
- Lessons capped at `max_reflections=3`

**F4 — Supervisor escalation**
- If output guardrails BLOCK after all retries: raise `EscalationError` with `audit_log`
- Input BLOCK always raises immediately, no retries

**F5 — RunResult**
```python
class RunResult(BaseModel):
    final_output: str
    passed: bool
    attempts: int
    reflections: list[str]
    audit_log: list[str]
    total_input_tokens: int
    total_output_tokens: int
```
Token counts should be tracked across all LLM calls made during the run.

**F6 — Async variant (bonus)**
- `AgentGuard.arun(task: str) -> RunResult` — async version using `asyncio`
- Worker and critic calls run concurrently where possible

---

## Test requirements

Minimum 12 tests across all modules. Required test cases:

| Test | Covers |
|---|---|
| `test_happy_path` | Worker passes critic on first attempt; result.passed=True; attempts=1 |
| `test_reflection_retry` | Worker fails, then passes; attempts=2; 1 reflection recorded |
| `test_max_attempts_exhausted` | Worker always fails; passed=False; attempts=max |
| `test_reflexion_lessons_injected` | Lessons appear in worker prompt context after failure |
| `test_input_block` | Prompt injection → EscalationError before worker called |
| `test_output_pii_warn` | PII in output → WARN logged, not blocked |
| `test_output_block_escalates` | Output fails guardrail → EscalationError with audit_log |
| `test_custom_guardrail` | Custom guardrail function integrated via GuardrailSet |
| `test_rubric_partial_fail` | One dimension below threshold → FAIL even if overall >= 7 |
| `test_token_tracking` | total_input_tokens and total_output_tokens populated correctly |
| `test_empty_rubric` | Empty rubric → all outputs pass (no dimension thresholds to fail) |
| `test_audit_log_sequence` | Audit log entries appear in correct order |

All tests must mock the Anthropic client. No real API calls in tests.

---

## Deliverables

1. **Working package** — `python -m pytest tests/ -v` passes with 0 failures
2. **`pyproject.toml`** — correct dependencies: `anthropic`, `pydantic>=2`, `pytest`
3. **`README.md`** — usage example (the `AgentGuard` façade snippet above), install instructions, and a table of built-in guardrails
4. **`DESIGN.md`** — one page: architecture diagram (ASCII), token cost analysis per run mode, and one tradeoff you made with rationale

---

## Evaluation rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | All F1–F5 working; loop terminates correctly; no stubs |
| Conceptual understanding | 25% | DESIGN.md shows understanding of when each pattern fires; tradeoff is real |
| Debugging & problem-solving | 20% | All 12 required tests pass; edge cases covered; no test depends on real LLM |
| Code quality & testing | 15% | Type hints; logging; ruff clean; no god functions (>50 lines); Pydantic everywhere |

Solutions: ../solutions/week-05/project.md
