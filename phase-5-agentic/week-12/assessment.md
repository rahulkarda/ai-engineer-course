# Week 12 Assessment — Enterprise Agent Orchestration Framework Capstone

## Conceptual (interview-style)

1. Explain the role of `ToolSchema.risk_level` in the full agent workflow. How does it connect `ToolRegistry`, `HumanApprovalGate`, and `AuditLogger`?

2. What is the difference between session memory and long-term memory in the context of a ReAct planner? When would you use each?

3. Describe the request lifecycle from `POST /agent/run` to final answer, naming every subsystem touched and in what order.

4. Why should Pydantic models be used at module seams instead of raw `dict`s? Give a concrete example from the capstone framework.

5. What is the difference between MCP and A2A? When would you use one versus the other?

6. How does the evaluation harness differ from a standard unit test? Why is exact string match insufficient for agent outputs?

7. Describe two security controls in the framework and explain what threat each mitigates.

8. How does the FastAPI lifespan pattern avoid global state? Why does this matter for testing?

## Coding challenge

Build a `ReflectionAgent` that post-processes a completed planner run and generates a structured improvement report.

**Specification:**

- Input: a `list[PlanStep]` (from `ReActPlanner`) and a `task: str`
- Output: a `ReflectionReport` Pydantic v2 model with fields:
  - `task: str`
  - `total_steps: int`
  - `tool_calls: list[str]` — names of all tools called across all steps
  - `success: bool` — `True` if any step has `is_final=True`
  - `issues: list[str]` — detected problems (e.g., `"No final answer reached"`, `"Tool 'foo' called but not in registry"`)
  - `suggestions: list[str]` — at least 1 suggestion if `issues` is non-empty, empty list otherwise
  - `confidence: float` — `0.0`–`1.0`, computed as `sum(1 for s in steps if s.observation) / max(1, len(steps))`
- The agent must **not** call any LLM — it is a pure Python analysis function
- The function must be fully type-annotated and use `logging`, not `print`
- Write at least 4 unit tests covering:
  1. All-success path: all steps have observations, one step is final — verify `success=True`, `confidence=1.0`, `issues=[]`
  2. No final answer: steps have observations but none is final — verify `success=False`, `"No final answer reached"` in `issues`
  3. Empty steps list — verify `total_steps=0`, `success=False`, `confidence=0.0`
  4. Steps with tool calls but missing observations — verify `confidence < 1.0`

**Acceptance criteria:**

- [ ] `ReflectionReport` is a valid Pydantic v2 model with all fields typed
- [ ] `success` is correctly derived from `is_final` on steps
- [ ] `confidence` is always in `[0.0, 1.0]`
- [ ] All 4 unit tests pass
- [ ] Function has full type hints, uses `logging`, no `print` statements

## Rubric (pass threshold: ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging and problem-solving | 20% |
| Code quality and testing | 15% |

Answer key: `../solutions/week-12/assessment.md`
