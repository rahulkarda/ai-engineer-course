# Phase 5 · Week 11 · Day 5 — Golden Trajectories & Regression Gates

> Objective: Build a golden-trajectory store, implement regression detection, and gate CI on trajectory drift.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

A **golden trajectory** is the approved, known-correct sequence of tool calls for a given input. You record it once from a working agent run, store it, and then compare every subsequent run against it. This is the agent equivalent of a snapshot test.

A **regression gate** fails the build if:
- Task success drops below a threshold (e.g., 80% pass rate).
- Average trajectory drift exceeds a threshold (e.g., step-count delta > 1).
- Any previously-passing case now fails.

### Why snapshot tests for agents?

LLM output is non-deterministic, but tool call patterns are more stable because:
- Tool selection is driven by the prompt + system design, not by token sampling.
- A meaningful change in tool calling usually signals a prompt regression or model behaviour change.
- Catching "agent now calls `search` 4× instead of 1×" is actionable; exact token comparison is not.

### Golden trajectory lifecycle

```
1. Capture  — run agent in "capture mode", record trajectory + output
2. Review   — human approves the golden (is this the right path?)
3. Store    — serialise to JSON/YAML in tests/golden/
4. Gate     — CI runs eval suite against stored goldens, fails on drift
5. Update   — when intentionally changing behaviour, re-capture and review
```

## How it works internally

```
GoldenStore.load(path)  → dict[case_id, GoldenRecord]
                                │
           EvalRunner.run_all(cases, goldens)
                                │
                    for each case:
                        run agent → actual_traj
                        compare actual_traj vs golden_traj
                        compare actual_output vs golden_output
                                │
                    RegressionReport(
                        regressions: list[Regression],
                        new_passes: list[str],
                        gate_passed: bool
                    )
```

## Worked example

```python
# src/eval/golden_store.py
"""Golden trajectory store — capture, persist, and compare."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.eval.trajectory_eval import Trajectory, ToolCall, score_trajectory

logger = logging.getLogger(__name__)


# ── Golden record ─────────────────────────────────────────────────────────────
class GoldenRecord(BaseModel):
    case_id: str
    user_input: str
    golden_trajectory: Trajectory
    expected_output: str
    captured_at: str  # ISO datetime string
    approved: bool = False


# ── Store ─────────────────────────────────────────────────────────────────────
class GoldenStore:
    def __init__(self, store_dir: Path) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        return self._dir / f"{case_id}.json"

    def save(self, record: GoldenRecord) -> None:
        path = self._path(record.case_id)
        path.write_text(record.model_dump_json(indent=2))
        logger.info("Saved golden: %s → %s", record.case_id, path)

    def load(self, case_id: str) -> GoldenRecord:
        path = self._path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"No golden found for case: {case_id}")
        return GoldenRecord.model_validate_json(path.read_text())

    def load_all(self) -> dict[str, GoldenRecord]:
        records: dict[str, GoldenRecord] = {}
        for p in self._dir.glob("*.json"):
            rec = GoldenRecord.model_validate_json(p.read_text())
            records[rec.case_id] = rec
        return records

    def approve(self, case_id: str) -> None:
        rec = self.load(case_id)
        rec = rec.model_copy(update={"approved": True})
        self.save(rec)
        logger.info("Approved golden: %s", case_id)


# ── Capture helper ────────────────────────────────────────────────────────────
def capture_golden(
    case_id: str,
    user_input: str,
    actual_trajectory: Trajectory,
    actual_output: str,
    store: GoldenStore,
) -> GoldenRecord:
    """Record an agent run as a candidate golden (not yet approved)."""
    from datetime import datetime, timezone
    record = GoldenRecord(
        case_id=case_id,
        user_input=user_input,
        golden_trajectory=actual_trajectory,
        expected_output=actual_output,
        captured_at=datetime.now(timezone.utc).isoformat(),
        approved=False,
    )
    store.save(record)
    logger.info("Captured golden candidate: %s (pending approval)", case_id)
    return record
```

```python
# src/eval/regression_gate.py
"""Regression gate — compare current run against approved goldens."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.golden_store import GoldenRecord, GoldenStore
from src.eval.trajectory_eval import (
    Trajectory,
    TrajectoryScore,
    extract_trajectory,
    score_trajectory,
)

logger = logging.getLogger(__name__)


@dataclass
class Regression:
    case_id: str
    kind: str  # "task_failure" | "trajectory_drift" | "new_failure"
    detail: str
    trajectory_score: TrajectoryScore


@dataclass
class RegressionReport:
    regressions: list[Regression] = field(default_factory=list)
    new_passes: list[str] = field(default_factory=list)
    total_cases: int = 0

    @property
    def gate_passed(self) -> bool:
        return len(self.regressions) == 0

    @property
    def pass_rate(self) -> float:
        if self.total_cases == 0:
            return 1.0
        return 1.0 - len(self.regressions) / self.total_cases

    def summary(self) -> str:
        lines = [
            f"Regression gate: {'PASSED' if self.gate_passed else 'FAILED'}",
            f"  Cases: {self.total_cases}  Regressions: {len(self.regressions)}",
            f"  Pass rate: {self.pass_rate:.1%}",
        ]
        for r in self.regressions:
            lines.append(f"  ✗ {r.case_id}: [{r.kind}] {r.detail}")
        return "\n".join(lines)


class RegressionGate:
    """Compare agent runs against approved goldens; report regressions."""

    def __init__(
        self,
        store: GoldenStore,
        *,
        max_step_delta: int = 2,
        min_name_match: float = 0.8,
        task_pass_threshold: float = 0.75,
    ) -> None:
        self._store = store
        self._max_step_delta = max_step_delta
        self._min_name_match = min_name_match
        self._task_threshold = task_pass_threshold

    def evaluate(
        self,
        case_id: str,
        actual_trajectory: Trajectory,
        actual_output: str,
        task_passed: bool,
    ) -> Regression | None:
        """Return a Regression if the run drifts from the golden, else None."""
        try:
            golden = self._store.load(case_id)
        except FileNotFoundError:
            logger.warning("No golden for case %s — skipping gate", case_id)
            return None

        if not golden.approved:
            logger.debug("Golden for %s not yet approved — skipping", case_id)
            return None

        traj_score = score_trajectory(actual_trajectory, golden.golden_trajectory)

        # Task failure
        if not task_passed:
            return Regression(
                case_id=case_id,
                kind="task_failure",
                detail="Task did not pass acceptance criteria",
                trajectory_score=traj_score,
            )

        # Trajectory drift
        if abs(traj_score.step_count_delta) > self._max_step_delta:
            return Regression(
                case_id=case_id,
                kind="trajectory_drift",
                detail=(
                    f"Step delta {traj_score.step_count_delta} exceeds "
                    f"threshold {self._max_step_delta}"
                ),
                trajectory_score=traj_score,
            )

        if traj_score.tool_name_match < self._min_name_match:
            return Regression(
                case_id=case_id,
                kind="trajectory_drift",
                detail=(
                    f"Tool name match {traj_score.tool_name_match:.2f} below "
                    f"threshold {self._min_name_match:.2f}"
                ),
                trajectory_score=traj_score,
            )

        return None

    def run_report(
        self,
        results: list[dict[str, Any]],  # list of {case_id, trajectory, output, passed}
    ) -> RegressionReport:
        report = RegressionReport(total_cases=len(results))
        for item in results:
            regression = self.evaluate(
                item["case_id"],
                item["trajectory"],
                item["output"],
                item["passed"],
            )
            if regression:
                report.regressions.append(regression)
            else:
                report.new_passes.append(item["case_id"])

        logger.info(report.summary())
        return report
```

### Integrating with pytest as a CI gate

```python
# tests/integration/test_regression_gate.py
"""Regression gate integration test — fails CI on trajectory drift."""
import pytest
from pathlib import Path
from src.eval.golden_store import GoldenRecord, GoldenStore
from src.eval.regression_gate import RegressionGate
from src.eval.trajectory_eval import Trajectory, ToolCall, TrajectoryScore


GOLDEN_DIR = Path(__file__).parent / "goldens"


@pytest.fixture
def store(tmp_path) -> GoldenStore:
    s = GoldenStore(tmp_path)
    # Seed an approved golden
    golden = GoldenRecord(
        case_id="search-001",
        user_input="search for Python agents",
        golden_trajectory=Trajectory(steps=[
            ToolCall(tool_name="search", args={"query": "Python agents"}),
            ToolCall(tool_name="summarise", args={"max_length": 200}),
        ]),
        expected_output="Python agents are ...",
        captured_at="2025-01-01T00:00:00+00:00",
        approved=True,
    )
    s.save(golden)
    return s


def test_gate_passes_on_matching_trajectory(store):
    gate = RegressionGate(store)
    actual_traj = Trajectory(steps=[
        ToolCall(tool_name="search", args={"query": "Python agents"}),
        ToolCall(tool_name="summarise", args={"max_length": 200}),
    ])
    regression = gate.evaluate("search-001", actual_traj, "Python agents are ...", True)
    assert regression is None


def test_gate_fails_on_extra_steps(store):
    gate = RegressionGate(store, max_step_delta=0)
    actual_traj = Trajectory(steps=[
        ToolCall(tool_name="search", args={}),
        ToolCall(tool_name="search", args={}),  # extra step
        ToolCall(tool_name="summarise", args={}),
    ])
    regression = gate.evaluate("search-001", actual_traj, "...", True)
    assert regression is not None
    assert regression.kind == "trajectory_drift"


def test_gate_fails_on_task_failure(store):
    gate = RegressionGate(store)
    actual_traj = Trajectory(steps=[ToolCall(tool_name="search", args={})])
    regression = gate.evaluate("search-001", actual_traj, "", False)
    assert regression is not None
    assert regression.kind == "task_failure"
```

## Exercises

1. Add a `--capture` CLI flag to a script `scripts/capture_golden.py` that runs the agent on a list of inputs (JSONL file), captures goldens, and saves them to `tests/golden/`. Outputs a table of case IDs and their trajectories.

2. Implement `GoldenStore.diff(case_id, actual_traj)` that returns a human-readable diff of tool names between golden and actual (like `difflib.SequenceMatcher`).

3. Add a `min_pass_rate: float = 0.8` parameter to `RegressionGate`. After `run_report`, raise `AssertionError` if `pass_rate < min_pass_rate`. Wire this into a pytest plugin fixture.

4. Design a YAML schema for golden trajectory files. Implement `GoldenStore.load_from_yaml(path)` and `save_to_yaml(record, path)`. Discuss when YAML is preferable to JSON for this use case.

(Solutions: ../solutions/week-11/day-5.md)

## Self-check

- [ ] I can explain the golden trajectory lifecycle (capture → review → store → gate → update).
- [ ] I can implement a `GoldenStore` with CRUD operations on `GoldenRecord` models.
- [ ] I can build a `RegressionGate` that detects task failure and trajectory drift.
- [ ] I understand why trajectory snapshot tests are more useful than exact-output snapshot tests.
- [ ] I can integrate the regression gate into a pytest suite that fails CI on drift.

## Supplementary links

- [Pydantic AI — Testing & Evals](https://ai.pydantic.dev/testing-evals/)
- [Python difflib — SequenceMatcher](https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher)
