# Solutions — Phase 5 · Week 03 · Day 2 — Task Decomposition & Planning

## Exercise 1

`validate_plan` with Pydantic v2 — checks duplicate IDs, missing `depends_on` refs, and cycles via DFS.

```python
"""
planning/models.py — Plan and Step Pydantic v2 models.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Step(BaseModel):
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)


class Plan(BaseModel):
    goal: str
    steps: list[Step]
```

```python
"""
planning/validator.py — validate_plan implementation.
"""
from __future__ import annotations

import logging

from .models import Plan, Step

logger = logging.getLogger(__name__)


def validate_plan(plan: Plan) -> list[str]:
    """Validate a Plan and return a list of error strings.

    Checks:
    - No duplicate step IDs.
    - All depends_on references point to existing step IDs.
    - No dependency cycles (via iterative DFS).

    Returns an empty list when the plan is valid.
    """
    errors: list[str] = []
    step_ids: set[str] = set()

    # --- (a) Duplicate IDs ---
    seen: set[str] = set()
    for step in plan.steps:
        if step.id in seen:
            msg = f"Duplicate step ID: '{step.id}'"
            errors.append(msg)
            logger.warning(msg)
        seen.add(step.id)
    step_ids = seen  # all known IDs (including duplicates — kept for ref checks)

    # --- (b) Missing depends_on references ---
    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                msg = f"Step '{step.id}' depends on unknown step '{dep}'"
                errors.append(msg)
                logger.warning(msg)

    # --- (c) Cycle detection via iterative DFS ---
    adjacency: dict[str, list[str]] = {s.id: s.depends_on for s in plan.steps}
    cycles = _find_cycles(adjacency)
    for cycle in cycles:
        msg = f"Dependency cycle detected: {' -> '.join(cycle)}"
        errors.append(msg)
        logger.warning(msg)

    return errors


def _find_cycles(graph: dict[str, list[str]]) -> list[list[str]]:
    """Return a list of cycles found in *graph* using iterative DFS.

    Each cycle is represented as the sequence of node IDs that forms it.
    Returns an empty list when the graph is acyclic.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {node: WHITE for node in graph}
    parent: dict[str, str | None] = {node: None for node in graph}
    cycles: list[list[str]] = []

    for start in graph:
        if color[start] != WHITE:
            continue

        # Iterative DFS using an explicit stack
        # Stack entries: (node, iterator_over_neighbours, entered_as_grey)
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        path: list[str] = [start]

        while stack:
            node, idx = stack[-1]
            neighbours = graph.get(node, [])

            if idx < len(neighbours):
                stack[-1] = (node, idx + 1)
                neighbour = neighbours[idx]

                if neighbour not in color:
                    continue  # unknown node — skip (already reported as missing ref)

                if color[neighbour] == GRAY:
                    # Found a back-edge — reconstruct cycle
                    cycle_start = path.index(neighbour)
                    cycles.append(path[cycle_start:] + [neighbour])
                elif color[neighbour] == WHITE:
                    color[neighbour] = GRAY
                    parent[neighbour] = node
                    path.append(neighbour)
                    stack.append((neighbour, 0))
            else:
                color[node] = BLACK
                stack.pop()
                if path and path[-1] == node:
                    path.pop()

    return cycles
```

**Explanation:**
- `validate_plan` gathers all IDs first so ref-checks work independently of declaration order.
- Cycle detection uses iterative DFS with three-colour marking (WHITE/GRAY/BLACK) — the standard approach. A GRAY node on the stack means we are currently traversing its subtree; finding a GRAY neighbour is a back-edge.
- Errors are accumulated into a list and returned — callers decide what to do with them (raise, log, display).

---

## Exercise 2

Add `priority: int` to `Step` with a Pydantic v2 validator ensuring `1 <= priority <= 5`, default 3.

```python
"""
planning/models.py — Step with priority field.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Step(BaseModel):
    id: str
    description: str
    depends_on: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)

    # Explicit field_validator for clear error messages (ge/le on Field also works)
    @field_validator("priority")
    @classmethod
    def priority_in_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"priority must be between 1 and 5, got {v}")
        return v


class Plan(BaseModel):
    goal: str
    steps: list[Step]
```

**Alternative — using `Field(ge=1, le=5)` alone (more concise):**

```python
priority: int = Field(default=3, ge=1, le=5)
```

Pydantic v2 generates the same `ValidationError` from `ge`/`le` constraints without needing an explicit `@field_validator`. The explicit validator is shown above for readability and to demonstrate where custom logic (e.g., logging or business rules) would go.

**Test:**

```python
import pytest
from pydantic import ValidationError
from planning.models import Step

def test_priority_default():
    s = Step(id="s1", description="d")
    assert s.priority == 3

def test_priority_valid_range():
    for v in (1, 3, 5):
        s = Step(id="s1", description="d", priority=v)
        assert s.priority == v

def test_priority_out_of_range():
    with pytest.raises(ValidationError):
        Step(id="s1", description="d", priority=6)

    with pytest.raises(ValidationError):
        Step(id="s1", description="d", priority=0)
```

---

## Exercise 3

`topological_sort(plan: Plan) -> list[Step]` — Kahn's algorithm.

```python
"""
planning/sort.py — topological sort of a Plan.
"""
from __future__ import annotations

import logging
from collections import deque

from .models import Plan, Step

logger = logging.getLogger(__name__)


def topological_sort(plan: Plan) -> list[Step]:
    """Return plan steps in an execution order that respects all dependencies.

    Uses Kahn's algorithm (BFS-based topological sort).

    Raises:
        ValueError: if the plan contains a cycle (not a DAG).
    """
    step_map: dict[str, Step] = {s.id: s for s in plan.steps}

    # Build in-degree count and reverse adjacency
    in_degree: dict[str, int] = {s.id: 0 for s in plan.steps}
    dependents: dict[str, list[str]] = {s.id: [] for s in plan.steps}

    for step in plan.steps:
        for dep in step.depends_on:
            in_degree[step.id] += 1
            dependents[dep].append(step.id)

    # Start with all nodes that have no dependencies
    # Sort by priority (ascending = higher priority first) for determinism
    queue: deque[str] = deque(
        sorted(
            (sid for sid, deg in in_degree.items() if deg == 0),
            key=lambda sid: step_map[sid].priority,
        )
    )

    sorted_steps: list[Step] = []

    while queue:
        node = queue.popleft()
        sorted_steps.append(step_map[node])
        logger.debug("Scheduled step '%s' (priority %d)", node, step_map[node].priority)

        # Reduce in-degree for dependents
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_steps) != len(plan.steps):
        raise ValueError(
            "Plan contains a cycle — topological sort not possible. "
            f"Unresolved steps: {[sid for sid, d in in_degree.items() if d > 0]}"
        )

    return sorted_steps
```

**Explanation:**
- Kahn's algorithm is iterative and avoids recursion-depth limits on large plans.
- Steps with equal in-degree are ordered by `priority` (1 = highest priority runs first).
- If the sorted list is shorter than the plan, at least one cycle exists — a `ValueError` is raised with the offending step IDs.

---

## Exercise 4

Pytest test: Plan with a cycle → `validate_plan` returns non-empty errors.

```python
"""
tests/unit/test_validate_plan.py
"""
from __future__ import annotations

import pytest

from planning.models import Plan, Step
from planning.validator import validate_plan
from planning.sort import topological_sort


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_plan() -> Plan:
    return Plan(
        goal="Deploy service",
        steps=[
            Step(id="build", description="Compile code"),
            Step(id="test", description="Run tests", depends_on=["build"]),
            Step(id="deploy", description="Deploy", depends_on=["test"]),
        ],
    )


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestValidatePlanCycles:
    def test_cycle_returns_errors(self) -> None:
        """A plan with a cycle must produce at least one error string."""
        plan = Plan(
            goal="Cyclic plan",
            steps=[
                Step(id="a", description="Step A", depends_on=["c"]),
                Step(id="b", description="Step B", depends_on=["a"]),
                Step(id="c", description="Step C", depends_on=["b"]),
            ],
        )
        errors = validate_plan(plan)
        assert len(errors) > 0, "Expected errors for cyclic plan, got none"
        assert any("cycle" in e.lower() for e in errors), (
            f"Expected a cycle error, got: {errors}"
        )

    def test_two_step_cycle(self) -> None:
        """Minimal two-node cycle is detected."""
        plan = Plan(
            goal="Two-step cycle",
            steps=[
                Step(id="x", description="X", depends_on=["y"]),
                Step(id="y", description="Y", depends_on=["x"]),
            ],
        )
        errors = validate_plan(plan)
        assert len(errors) > 0

    def test_self_loop(self) -> None:
        """A step that depends on itself is a cycle."""
        plan = Plan(
            goal="Self-loop",
            steps=[
                Step(id="s1", description="Self-referencing", depends_on=["s1"]),
            ],
        )
        errors = validate_plan(plan)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Valid plan passes
# ---------------------------------------------------------------------------

class TestValidatePlanValid:
    def test_valid_plan_no_errors(self) -> None:
        plan = make_valid_plan()
        errors = validate_plan(plan)
        assert errors == [], f"Unexpected errors: {errors}"


# ---------------------------------------------------------------------------
# Duplicate IDs
# ---------------------------------------------------------------------------

class TestValidatePlanDuplicates:
    def test_duplicate_id_error(self) -> None:
        plan = Plan(
            goal="Duplicate IDs",
            steps=[
                Step(id="a", description="First A"),
                Step(id="a", description="Second A"),
            ],
        )
        errors = validate_plan(plan)
        assert any("duplicate" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Missing references
# ---------------------------------------------------------------------------

class TestValidatePlanMissingRefs:
    def test_missing_dependency_error(self) -> None:
        plan = Plan(
            goal="Missing dep",
            steps=[
                Step(id="a", description="A", depends_on=["ghost"]),
            ],
        )
        errors = validate_plan(plan)
        assert any("ghost" in e for e in errors)


# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

class TestTopologicalSort:
    def test_sort_respects_dependencies(self) -> None:
        plan = make_valid_plan()
        order = topological_sort(plan)
        ids = [s.id for s in order]
        assert ids.index("build") < ids.index("test")
        assert ids.index("test") < ids.index("deploy")

    def test_sort_raises_on_cycle(self) -> None:
        plan = Plan(
            goal="Cycle",
            steps=[
                Step(id="a", depends_on=["b"], description="A"),
                Step(id="b", depends_on=["a"], description="B"),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            topological_sort(plan)
```

**Explanation:**
- Three independent test classes cover cycles, valid plans, duplicates, and missing refs.
- `assert any("cycle" in e.lower() for e in errors)` avoids coupling the test to exact wording.
- The `topological_sort` tests verify both the happy path (dependency order) and the failure path (raises `ValueError` on cycle).
