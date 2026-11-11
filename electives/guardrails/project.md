# Elective: Guardrails · Project — Composable Guardrail Layer for an Agent

> Time: ~4 hours  
> Prereqs: All 6 lessons complete

---

## Overview

Build a production-quality, composable guardrail layer that wraps a LangGraph agent.
The layer must be independently testable (no real LLM calls), configurable per
deployment, and validated against an adversarial test suite.

This project integrates every lesson:
- Lesson 1: threat classification
- Lessons 2–3: Guardrails AI validators (RAIL + Pydantic)
- Lesson 4: NeMo Guardrails dialogue rails
- Lesson 5: custom validators + GuardrailLayer pattern
- Lesson 6: adversarial testing + latency benchmarking

---

## Deliverables

```
guardrail_agent/
├── pyproject.toml
├── .env.example
├── src/
│   └── guardrail_agent/
│       ├── __init__.py
│       ├── config.py            # pydantic-settings config
│       ├── validators/
│       │   ├── __init__.py
│       │   ├── input.py         # input validator functions
│       │   └── output.py        # custom Guardrails AI validators
│       ├── models.py            # Pydantic output models with validators
│       ├── layer.py             # GuardrailLayer orchestrator
│       ├── agent.py             # stub LangGraph agent (mocked LLM)
│       └── rails/
│           ├── config.yml       # NeMo Guardrails config
│           └── flows.co         # Colang dialogue flows
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_input_validators.py
    │   ├── test_output_validators.py
    │   └── test_models.py
    ├── integration/
    │   └── test_layer.py
    └── adversarial/
        └── test_adversarial.py
```

---

## Spec

### 1. Config (`config.py`)

Use `pydantic-settings`. Fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `anthropic_api_key` | `str` | `os.getenv("ANTHROPIC_API_KEY")` | Never hardcode |
| `model_name` | `str` | `"claude-haiku-5-5"` | Main model |
| `max_input_chars` | `int` | `2000` | Token bomb limit |
| `num_reasks` | `int` | `2` | Max output reasks |
| `competitors` | `list[str]` | `["AcmeCorp", "RivalCo"]` | Banned brands |
| `allowed_topics` | `list[str]` | `["billing", "account", "shipping", "returns"]` | Topic allow-list |

### 2. Input validators (`validators/input.py`)

Implement as standalone functions (no class needed):

- `check_length(text, max_chars) -> None` — raises `ValueError` if exceeded
- `check_injection(text) -> None` — regex-based injection detection
- `check_jailbreak(text) -> None` — regex-based jailbreak detection
- `check_topic(text, allowed_topics) -> None` — keyword-based topic check
- `check_pii_input(text) -> None` — detect SSN, credit card, email in input

Each function must:
- Accept `text: str` as first argument
- Raise `ValueError` with a clear human-readable message on failure
- Be independently unit-testable
- Use `logging`, not `print`

### 3. Output validators (`validators/output.py`)

Implement as Guardrails AI `Validator` subclasses:

- `NoCompetitorMention(competitors: list[str], on_fail="fix")` — seen in Lesson 5
- `NoLegalAdvice(on_fail="fix")` — seen in Lesson 5
- `NoPIIOutput(on_fail="filter")` — new: detect email/phone/SSN in output

### 4. Output model (`models.py`)

```python
class AgentResponse(BaseModel):
    answer: Annotated[str, ValidLength(min=10, max=600, on_fail="reask"),
                           NoCompetitorMention(...), NoLegalAdvice(...), NoPIIOutput(...)]
    confidence: Annotated[float, ValidRange(min=0.0, max=1.0, on_fail="fix")]
    sources: list[str] = Field(default_factory=list)
    requires_escalation: bool = False
```

### 5. Guardrail layer (`layer.py`)

`GuardrailLayer` class:

```python
class GuardrailLayer:
    def __init__(self, config: GuardrailConfig, agent_fn: Callable[[str], str]) -> None: ...

    def __call__(self, user_input: str, llm_fn: Callable | None = None) -> LayerResult: ...
```

`LayerResult` is a Pydantic model:

```python
class LayerResult(BaseModel):
    user_input: str
    response: str
    blocked: bool
    block_stage: Literal["input", "output", None]
    block_reason: str | None
    latency_ms: float
    reask_count: int
```

The layer must:
- Run all input validators in order; stop at first failure
- Call the agent (or injected `llm_fn`) only if input passes
- Run the output Guard; track reask count
- Return a `LayerResult` in all cases (never raise)
- Log every block at `WARNING` level with the reason

### 6. Stub agent (`agent.py`)

A minimal LangGraph graph with one node:

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    response: str

def support_node(state: AgentState) -> AgentState:
    # In production: calls real LLM
    # In tests: override with mock
    ...

def build_agent() -> Compiled:
    ...
```

The agent must accept a query string and return a response string. Dependency on the
real LLM must be injectable (default to `os.getenv("ANTHROPIC_API_KEY")` client).

### 7. NeMo rails (`rails/`)

`config.yml` must configure:
- `main` model: `claude-haiku-5-5`
- Input rail: `self check input`

`flows.co` must define:
- At least 4 user intents (greeting, account, billing, off-topic)
- Corresponding bot responses
- A jailbreak blocking flow

---

## Tests specification

### Unit tests (mocked LLM — no API calls)

Each module has a dedicated test file. Required coverage:

| File | Required tests |
|------|---------------|
| `test_input_validators.py` | One pass + one fail per validator (10 tests min) |
| `test_output_validators.py` | Pass + fail per custom validator (6 tests min) |
| `test_models.py` | Valid response, out-of-range confidence, short answer trigger reask |

All tests mock the LLM — never call `anthropic.Anthropic` in tests.

### Integration tests (`test_layer.py`)

Test `GuardrailLayer.__call__` end-to-end with a mock agent function:

- Clean input → not blocked, returns `LayerResult(blocked=False)`
- Injection input → blocked at input stage, agent never called (verify with `call_count`)
- Competitor mention in output → fixed by validator, not blocked
- Short output (< 10 chars) → reask triggered, then fails → blocked at output stage
- PII in output → filtered by `NoPIIOutput`

### Adversarial tests (`test_adversarial.py`)

Implement `run_adversarial_suite(guardrail_fn, cases)` (from Lesson 6).

Test cases must cover all six bypass categories:

| Category | Min cases |
|----------|-----------|
| `direct_bypass` | 3 |
| `encoding_bypass` | 2 |
| `semantic_bypass` | 2 |
| `topic_violation` | 2 |
| `indirect_injection` | 1 |
| `clean` | 3 |

Acceptance criterion: **overall pass rate ≥ 80%**; **all `direct_bypass` cases blocked**.

---

## Acceptance criteria

- [ ] `pyproject.toml` lists `guardrails-ai`, `nemoguardrails`, `anthropic`, `langchain-anthropic`, `pydantic`, `pydantic-settings` as dependencies (not installed — listed for `uv add`)
- [ ] `.env.example` has `ANTHROPIC_API_KEY=` placeholder; no actual key anywhere
- [ ] All LLM calls mocked in tests; `pytest` passes with no `ANTHROPIC_API_KEY` set
- [ ] `GuardrailLayer.__call__` never raises — always returns `LayerResult`
- [ ] Adversarial suite: all `direct_bypass` blocked; overall ≥ 80% pass rate
- [ ] Ruff clean: `ruff check src/ tests/` reports no errors
- [ ] `LayerResult` is a Pydantic model, not a plain dict
- [ ] Every module uses `logging`, not `print`
- [ ] `config.py` reads secrets from env vars only

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Criteria |
|-----------|--------|----------|
| Independent implementation | 40% | Layer, validators, model all implemented correctly without copying lesson solutions |
| Adversarial coverage | 25% | ≥ 80% pass rate, all direct bypass cases blocked |
| Testing quality | 20% | LLM mocked, edge cases covered, assertions are meaningful (not just "not None") |
| Code quality | 15% | Type hints, logging, Pydantic models, ruff clean, no hardcoded secrets |

---

## Hints

- For `check_topic`, a keyword approach will have false negatives. Document this
  limitation and explain what a better approach (embedding similarity) would look like.

- The `NoPIIOutput` validator with `on_fail="filter"` sets the field to `None`.
  Make sure `AgentResponse.answer` can handle `None` (use `Optional[str]`) and
  substitute a safe fallback string in `GuardrailLayer`.

- For the NeMo rails tests, monkeypatching `LLMRails.generate_async` is the
  recommended approach (Lesson 4). Do not try to mock the Colang interpreter directly.

- Reask count is available on the `Guard` result object as `result.reask_attempts`
  (check your installed version; field name may vary).

- The adversarial test suite should be deterministic (no randomness, no LLM calls)
  so it runs fast in CI.

---

## Extension challenges (optional)

1. **Latency benchmark**: add `benchmark_guardrail` to the test suite as a separate
   `pytest` mark (`@pytest.mark.benchmark`). Record p50/p95/p99 for 1000 calls.

2. **NeMo integration test**: write one integration test that actually loads the NeMo
   config and sends a message through it (using a fully mocked main LLM). Verify the
   canonical form flow fires correctly.

3. **Async layer**: make `GuardrailLayer.__call__` async (`async def __call__`) and
   use `asyncio.gather` to run input validators concurrently.

4. **Metrics endpoint**: add a `GuardrailMetrics` collector that tracks block rates,
   avg latency, and reask rates. Expose as a `dict` via `layer.metrics`. Add a test
   that verifies metrics update correctly after several calls.

---

Solution: [solutions/project.md](solutions/project.md)
