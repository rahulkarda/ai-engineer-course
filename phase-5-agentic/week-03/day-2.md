# Phase 5 · Week 03 · Day 2 — Task Decomposition & Planning

> Objective: Break a complex goal into validated, dependency-ordered sub-tasks using structured LLM output and Pydantic v2 models.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-1]]

## Concept

A single flat prompt fails on multi-step goals because:

- The LLM must hold the entire execution strategy in working memory simultaneously.
- There is no checkpoint to catch an incorrect sub-task before it poisons later steps.
- The prompt grows unbounded as context accumulates.

**Task decomposition** is the act of mapping one high-level goal into an ordered list of concrete, individually verifiable sub-tasks before any execution begins.

### Hierarchical vs. sequential decomposition

| Style | When to use |
|---|---|
| Sequential | Steps form a strict pipeline — each step consumes the prior step's output. |
| Hierarchical | Some steps can be parallelised or have complex branching; a tree structure expresses this naturally. |
| Dependency graph | General form: a directed acyclic graph (DAG) where each step declares its inputs by step ID. |

This course uses the DAG representation because it is the most general and serialises cleanly to JSON.

### Why structured output matters

Asking an LLM for a plan in free text produces inconsistent schemas across runs. Asking it to emit JSON that validates against a Pydantic model gives you:

- Guaranteed field presence — your executor won't KeyError at runtime.
- Type coercion — string `"3"` becomes `int 3` automatically.
- Validation errors that surface during planning, not mid-execution.

### Validating plan completeness

A plan is complete when:

1. Every `depends_on` reference points to a step ID that exists in the plan.
2. The dependency graph has no cycles (a cycle would deadlock the executor).
3. There is at least one step with no dependencies (a root) and at least one step with no dependents (a leaf).

Condition 2 can be checked with a topological sort — if the sort fails, a cycle exists.

## How it works internally

The planner is a thin LLM wrapper. It:

1. Serialises the user's goal into a prompt containing the `Step` and `Plan` JSON schemas.
2. Asks the LLM to produce a JSON object conforming to `Plan`.
3. Parses the JSON with `Plan.model_validate(json.loads(raw))` — Pydantic raises `ValidationError` on schema violations.
4. Passes the validated plan to `validate_plan()` for semantic checks (cycles, missing deps).

The LLM does not execute any step. It only produces a structured description of what should be executed. This separation is the first principle of Plan-and-Execute architectures.

### Why mock the LLM in tests?

The planner's correctness does not depend on which plan the LLM generates — it depends on whether the planner correctly validates and rejects bad plans. Tests should inject synthetic plans directly rather than relying on LLM output, which is non-deterministic.

## Worked example

```python
"""
day_2_planner.py — Task decomposition with Pydantic v2 structured plans.

Run:
    python day_2_planner.py

Set ANTHROPIC_API_KEY to use a real LLM; otherwise a hardcoded mock plan is used.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from typing import Any

from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------

class Step(BaseModel):
    """A single executable unit within a plan."""

    id: str = Field(
        description="Short unique identifier, e.g. 'step_1'.",
        pattern=r"^[a-z_][a-z0-9_]*$",
    )
    description: str = Field(description="Human-readable description of what this step does.")
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of steps that must complete before this one runs.",
    )
    tool: str | None = Field(
        default=None,
        description="Tool name to invoke, or None if this is an LLM-only step.",
    )
    expected_output: str = Field(
        description="What a successful result looks like (used for validation prompts)."
    )

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Step id must not be empty.")
        return v


class Plan(BaseModel):
    """An ordered, dependency-annotated decomposition of a goal."""

    goal: str = Field(description="The original user goal this plan addresses.")
    steps: list[Step] = Field(description="Ordered list of steps.", min_length=1)

    @property
    def step_ids(self) -> set[str]:
        return {s.id for s in self.steps}

    def get_step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(plan: Plan) -> list[str]:
    """
    Semantic validation of a plan.

    Returns a list of human-readable error messages.
    An empty list means the plan is valid.
    """
    errors: list[str] = []
    ids = plan.step_ids

    # 1. All depends_on references must point to existing step IDs.
    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in ids:
                errors.append(
                    f"Step '{step.id}' depends on '{dep}' which does not exist in the plan."
                )

    # 2. No cycles — attempt a topological sort (Kahn's algorithm).
    in_degree: dict[str, int] = {s.id: 0 for s in plan.steps}
    adjacency: dict[str, list[str]] = {s.id: [] for s in plan.steps}

    for step in plan.steps:
        for dep in step.depends_on:
            if dep in adjacency:
                adjacency[dep].append(step.id)
                in_degree[step.id] += 1

    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if visited_count != len(plan.steps):
        errors.append("Plan contains a dependency cycle — topological sort failed.")

    # 3. At least one root (no deps) and one leaf (nothing depends on it).
    depended_upon = {dep for s in plan.steps for dep in s.depends_on}
    roots = [s.id for s in plan.steps if not s.depends_on]
    leaves = [s.id for s in plan.steps if s.id not in depended_upon]

    if not roots:
        errors.append("Plan has no root step (every step has at least one dependency).")
    if not leaves:
        errors.append("Plan has no leaf step (every step is depended upon by another).")

    return errors


def topological_order(plan: Plan) -> list[str]:
    """
    Return step IDs in an order that respects all depends_on constraints.

    Raises ValueError if a cycle is detected (validate_plan should be run first).
    """
    in_degree: dict[str, int] = {s.id: 0 for s in plan.steps}
    adjacency: dict[str, list[str]] = {s.id: [] for s in plan.steps}

    for step in plan.steps:
        for dep in step.depends_on:
            adjacency[dep].append(step.id)
            in_degree[step.id] += 1

    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(plan.steps):
        raise ValueError("Cycle detected — cannot produce topological order.")

    return order


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are a planning agent. Given a goal, produce a JSON object that conforms exactly to this schema:

{
  "goal": "<original goal>",
  "steps": [
    {
      "id": "<snake_case_id>",
      "description": "<what this step does>",
      "depends_on": ["<step_id>", ...],
      "tool": "<tool_name or null>",
      "expected_output": "<description of a successful result>"
    }
  ]
}

Rules:
- Use only the tools: search, calculate, summarise, write_file.
- Each step must be independently verifiable.
- depends_on must reference IDs defined earlier in the same list.
- Return valid JSON only — no prose, no markdown fences.
"""


class Planner:
    def __init__(self, model: str = "claude-sonnet-5-5") -> None:
        self.model = model
        self._client: object | None = None

    def plan(self, goal: str) -> Plan:
        """Produce a validated Plan for the given goal."""
        log.info("Planning goal: %r", goal)
        raw_json = self._call_llm(goal)
        log.debug("Raw LLM output: %s", raw_json)

        data: dict[str, Any] = json.loads(raw_json)
        plan = Plan.model_validate(data)
        log.info("Plan parsed: %d steps", len(plan.steps))

        errors = validate_plan(plan)
        if errors:
            log.error("Plan validation failed: %s", errors)
            raise ValueError(f"Invalid plan: {errors}")

        order = topological_order(plan)
        log.info("Execution order: %s", order)
        return plan

    # ------------------------------------------------------------------
    # LLM layer
    # ------------------------------------------------------------------

    def _call_llm(self, goal: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return self._call_anthropic(goal)
        return self._mock_llm(goal)

    def _call_anthropic(self, goal: str) -> str:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        message = self._client.messages.create(  # type: ignore[union-attr]
            model=self.model,
            max_tokens=1024,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
        )
        return message.content[0].text

    @staticmethod
    def _mock_llm(goal: str) -> str:
        """Returns a fixed plan for offline tests."""
        _ = goal  # ignored in mock
        return json.dumps(
            {
                "goal": goal,
                "steps": [
                    {
                        "id": "step_search",
                        "description": "Search for relevant background information.",
                        "depends_on": [],
                        "tool": "search",
                        "expected_output": "A paragraph of factual background.",
                    },
                    {
                        "id": "step_calculate",
                        "description": "Perform any required numerical computation.",
                        "depends_on": ["step_search"],
                        "tool": "calculate",
                        "expected_output": "A numeric result.",
                    },
                    {
                        "id": "step_summarise",
                        "description": "Summarise findings into a final answer.",
                        "depends_on": ["step_search", "step_calculate"],
                        "tool": "summarise",
                        "expected_output": "A concise answer to the goal.",
                    },
                ],
            }
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    planner = Planner()

    goal = "Research the history of the Eiffel Tower and calculate its height in feet."
    plan = planner.plan(goal)

    print("\n=== Plan ===")
    print(plan.model_dump_json(indent=2))

    print("\n=== Execution order ===")
    for sid in topological_order(plan):
        step = plan.get_step(sid)
        assert step is not None
        deps = step.depends_on or ["(none)"]
        print(f"  {sid}  deps={deps}  tool={step.tool}")

    # Demonstrate validation catching a bad plan
    bad_plan_data = {
        "goal": "test",
        "steps": [
            {
                "id": "step_a",
                "description": "A",
                "depends_on": ["step_b"],  # forward ref — will cause cycle
                "tool": None,
                "expected_output": "x",
            },
            {
                "id": "step_b",
                "description": "B",
                "depends_on": ["step_a"],  # cycle: a→b→a
                "tool": None,
                "expected_output": "y",
            },
        ],
    }
    bad_plan = Plan.model_validate(bad_plan_data)
    errors = validate_plan(bad_plan)
    print(f"\n=== Cycle detection ===\nErrors: {errors}")
```

## Exercises

1. The mock LLM always returns the same plan regardless of `goal`. Extend `_mock_llm` to produce different plans for at least two distinct goal patterns (detect the goal with `in`). Write a pytest test that checks each branch.

2. `validate_plan` checks for missing deps and cycles. Add a third check: no step's `description` may be an empty string after stripping. Return an appropriate error message.

3. Add a `Plan.summary() -> str` method that returns a single-line human-readable summary: `"Plan '<goal>': N steps, tools used: [tool_a, tool_b]"`.

4. Implement `Plan.to_json_file(path: str) -> None` and `Plan.from_json_file(path: str) -> Plan` using `model_dump_json` / `model_validate_json`. Verify the round-trip preserves all fields.

5. The `topological_order` function raises `ValueError` on cycles. Modify `validate_plan` to call it internally and consolidate the cycle-detection logic (removing the duplicate Kahn's algorithm in `validate_plan`).

(Solutions: ../solutions/week-03/day-2.md)

## Self-check

- [ ] I can explain why structured output from the LLM is preferable to free-text plan descriptions.
- [ ] I can implement Kahn's algorithm for cycle detection on a DAG from memory.
- [ ] I can write Pydantic v2 models with `Field`, `field_validator`, and `model_validate`.
- [ ] I can articulate the three semantic validation checks: missing deps, cycles, root/leaf existence.
- [ ] I can describe the boundary between the planner and the executor and why they are kept separate.

---

## Supplementary links

- "Large Language Models as Optimizers" (Yang et al., 2023) — https://arxiv.org/abs/2309.03409
- Pydantic v2 validators documentation — https://docs.pydantic.dev/latest/concepts/validators/
- Wikipedia — Kahn's algorithm for topological sorting — https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
- "Plan-and-Solve Prompting" (Wang et al., 2023) — https://arxiv.org/abs/2305.04091
