# Week 03 Project — Plan-and-Execute Research Agent

## Goal

Build a complete multi-step research agent that accepts a natural-language research goal, decomposes it into typed execution steps via a Planner, executes each step through a tool registry, recovers from failures via re-planning with a circuit breaker, and delivers a validated `ResearchReport` Pydantic model.

---

## Requirements

### Functional

- Accept a `goal: str` as the entry point (e.g., `"Research recent advances in retrieval-augmented generation and write a structured report"`).
- Planner decomposes the goal into a `Plan` containing an ordered list of typed steps. Supported step types: `SearchStep`, `SummarizeStep`, `WriteReportStep`. Each step type has distinct fields validated via a Pydantic v2 discriminated union.
- Executor runs steps in order using a `ToolRegistry` that maps each step type to a callable tool (`search`, `summarize`, `write_report`). Tool outputs are passed forward as context to subsequent steps.
- On any step failure, classify the error:
  - `TransientError` — retry the step up to 2 times before triggering re-plan.
  - `DataError` — trigger re-plan immediately with error context injected into the re-plan prompt.
  - `FatalError` — abort execution and surface in `ResearchReport.error`.
- Re-planning is guarded by a circuit breaker: maximum 3 re-plans per execution. If the limit is reached, abort and mark the report as failed.
- Return a `ResearchReport` Pydantic v2 model containing: `goal`, `plan` (final plan used), `step_results` (list of step name + status + output), `replan_count`, `success`, and optional `error`.

### Technical

- Python 3.12+.
- Pydantic v2 for all data models (`Plan`, `SearchStep`, `SummarizeStep`, `WriteReportStep`, `StepResult`, `ResearchReport`). No untyped dicts crossing module boundaries.
- Full type hints on all functions, methods, and class attributes.
- `logging` (not `print`) throughout. Re-plan events logged at `WARNING`. Step completions at `INFO`. Errors at `ERROR`.
- API key consumed via `os.getenv("ANTHROPIC_API_KEY")` — never hardcoded.
- Real planner implementation (when not mocked) uses the Anthropic SDK with model `claude-sonnet-5-5`. Structured output via tool-use or JSON mode.
- All external calls (LLM, search API) mockable via dependency injection — no monkeypatching required in tests.
- Separation of concerns: `models.py`, `planner.py`, `executor.py`, `tools.py`, `agent.py` as separate modules. No module exceeds 300 lines.

### Testing

- pytest test suite under `tests/` mirroring `src/` structure.
- LLM calls mocked in all tests — no real API calls in the test suite.
- Tests cover: successful 3-step execution, `TransientError` retry then success, `TransientError` retry exhaustion triggering re-plan, `DataError` triggering immediate re-plan, circuit-breaker limit reached, `FatalError` aborting execution, `ResearchReport` serializes to valid JSON.
- At minimum one integration-style test that runs the full agent with all tools mocked and asserts on the final `ResearchReport`.

---

## Deliverables

- [ ] `src/research_agent/models.py` — all Pydantic v2 models including discriminated union for step types
- [ ] `src/research_agent/planner.py` — `Planner` class with real Anthropic SDK path and injectable mock interface
- [ ] `src/research_agent/executor.py` — `Executor` class with retry logic, error classification, and circuit breaker
- [ ] `src/research_agent/tools.py` — `ToolRegistry` and stub implementations of `search`, `summarize`, `write_report`
- [ ] `src/research_agent/agent.py` — top-level `ResearchAgent` that wires planner, executor, and tools; exposes `run(goal: str) -> ResearchReport`
- [ ] `tests/unit/test_models.py` — model validation and discriminated union tests
- [ ] `tests/unit/test_planner.py` — planner output parsing with mocked LLM response
- [ ] `tests/unit/test_executor.py` — retry, re-plan, and circuit-breaker logic
- [ ] `tests/integration/test_agent.py` — full agent run with all tools mocked
- [ ] `README.md` — setup instructions, how to run tests, how to run with a real API key, architecture diagram (text-based ASCII is fine)

---

## Stretch goals

- Parallel step execution: identify steps with no inter-dependencies in the plan DAG and execute them concurrently using `asyncio.gather`. Update `Executor` to use `async def` throughout.
- Plan caching: cache the `Plan` for a given goal hash so re-runs of the same goal skip the planner call. Use a simple in-memory dict with a TTL. Make it injectable so tests can inspect cache hits.
- Streaming output: stream partial `step_results` to the caller as each step completes rather than returning the full `ResearchReport` at the end. Use an `AsyncGenerator[StepResult, None]` return type on the executor and surface it in `agent.py`.

---

## Evaluation criteria

Graded against the course rubric (pass >= 75%):

**Independent implementation (40%)** — Does the submitted code solve the problem without scaffolding left in place? Is the re-plan/circuit-breaker logic the learner's own design, not a direct copy of course examples? Does the agent handle all three error classes distinctly?

**Conceptual understanding (25%)** — Are Pydantic discriminated unions used correctly (not a plain `Union` without a discriminator field)? Does the circuit breaker have explicit state (closed/open/half-open)? Is error classification clean and justified?

**Debugging and problem-solving (20%)** — Do tests demonstrate that the learner ran the agent, hit failure cases, and fixed them? Are edge cases (empty plan, all steps fail, re-plan produces same failing plan) addressed or at minimum documented in the README?

**Code quality and testing (15%)** — No untyped functions. No secrets in code. `logging` used correctly. Test coverage includes at minimum the happy path, one retry scenario, and the circuit-breaker limit. Module sizes within the 300-line guideline.
