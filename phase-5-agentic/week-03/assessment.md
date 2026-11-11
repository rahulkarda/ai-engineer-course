# Week 03 Assessment

## Conceptual (interview-style)

1. **ReAct vs. chain-of-thought vs. pure tool-use.** Explain the key architectural difference between ReAct (Reason + Act), chain-of-thought prompting, and pure tool-use loops. When does each pattern break down, and what failure mode does ReAct specifically address that the other two do not?

2. **Plan-and-execute architecture tradeoffs.** A plan-and-execute agent separates planning from execution into distinct phases. What are two concrete advantages this gives over a single ReAct loop for long-horizon tasks? What is the primary disadvantage, and under what task characteristics does it hurt most?

3. **Task decomposition strategies.** Describe two different strategies for decomposing a high-level goal into sub-steps (e.g., sequential vs. dependency-graph). For each strategy, give one task type where it is the better choice and briefly explain why.

4. **Re-plan vs. retry.** A step in your execution pipeline fails. Under what conditions should the agent retry the same step, and under what conditions should it trigger a full re-plan? Give one concrete signal that distinguishes the two cases.

5. **Pydantic discriminated unions for step types.** You have three step types — `SearchStep`, `SummarizeStep`, and `WriteReportStep` — each with different fields. Show the Pydantic v2 model structure (class signatures only, no implementation) that uses a discriminated union so a `Plan` model can hold a `list` of mixed step types and be validated correctly.

6. **Topological sort for dependency execution.** Your plan produces a DAG of steps where some steps depend on outputs of earlier steps. Explain why topological sort is the correct ordering algorithm here. What happens if the dependency graph contains a cycle, and how should the agent handle it at plan-validation time rather than at runtime?

7. **Circuit-breaker pattern.** Define the circuit-breaker pattern in the context of an agent that calls external tools (search API, LLM summarizer). Describe the three states (closed, open, half-open) and explain what transitions the agent between them. Why is this preferable to a simple retry-with-backoff for a re-planning agent?

8. **Failure recovery design.** You are designing a plan-and-execute agent where any step can fail with one of three error classes: `TransientError` (network timeout), `DataError` (bad input to the tool), and `FatalError` (upstream service down). Describe a recovery policy for each class — what action the agent takes and why. How does your design prevent infinite re-plan loops?

---

## Coding challenge

### Spec

Implement a minimal plan-and-execute agent that:

1. Accepts a goal string (`"Research the history of the Turing Award and write a summary"`).
2. Calls a `Planner` (LLM-backed, but mockable via dependency injection) that returns a `Plan` — a validated Pydantic v2 model containing exactly three typed steps in sequence: `SearchStep`, `SummarizeStep`, `WriteReportStep`.
3. Executes each step in order through a `ToolRegistry` that maps step type to a callable. All tools must be mockable.
4. Simulates one failure: the `SummarizeStep` raises `TransientError` on the first call.
5. Detects the failure, triggers a re-plan (calls `Planner` again with updated context), and completes execution.
6. Returns a final `ExecutionReport` Pydantic v2 model containing: the original goal, list of completed steps with status, total re-plan count, and a `success: bool` field.

### Constraints

- Python 3.12+, Pydantic v2 models for all data (`Plan`, `Step` discriminated union, `ExecutionReport`).
- Full type hints on all functions and classes.
- `logging` (not `print`) for re-plan events.
- `os.getenv("ANTHROPIC_API_KEY")` for the LLM key — no hardcoded values.
- If using the Anthropic SDK in the real planner, use model `claude-sonnet-5-5`.

### Acceptance criteria

- All pytest tests pass with a mocked `Planner` and mocked tools.
- Re-planning is triggered exactly once and logged at `WARNING` level with the failed step name.
- The returned `ExecutionReport.success` is `True` and `replan_count` is `1`.
- `ExecutionReport` serializes to valid JSON via `.model_dump_json()`.
- No hardcoded secrets. `ANTHROPIC_API_KEY` consumed via `os.getenv`.

---

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-03/assessment.md
