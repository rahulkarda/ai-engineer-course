# Solutions — Week 11 · Day 5: Golden Trajectories & Regression Gates

## Exercise 1 — capture_golden CLI script

```python
# scripts/capture_golden.py
"""Capture golden trajectories from a JSONL input file."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pydantic_ai.models.test import TestModel

# Add project src to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.code_review_agent import ReviewDeps, review_agent
from src.eval.golden_store import GoldenStore, capture_golden
from src.eval.trajectory_eval import extract_trajectory

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def main(input_file: Path, golden_dir: Path) -> None:
    store = GoldenStore(golden_dir)
    cases = [json.loads(line) for line in input_file.read_text().splitlines() if line.strip()]

    deps = ReviewDeps(allowed_imports=["os", "sys", "pathlib"], max_complexity=10)

    print(f"{'case_id':<20} {'tool_calls':>10}  output_snippet")
    print("-" * 60)

    for case in cases:
        case_id: str = case["case_id"]
        user_input: str = case["input"]

        with review_agent.override(model=TestModel()):
            result = await review_agent.run(user_input, deps=deps)

        traj = extract_trajectory(result.all_messages())
        record = capture_golden(
            case_id=case_id,
            user_input=user_input,
            actual_trajectory=traj,
            actual_output=str(result.data),
            store=store,
        )
        store.approve(case_id)

        snippet = str(result.data)[:40].replace("\n", " ")
        print(f"{case_id:<20} {len(traj.steps):>10}  {snippet}…")

    logger.info("Captured %d goldens → %s", len(cases), golden_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture golden trajectories")
    parser.add_argument("input_file", type=Path, help="JSONL file with {case_id, input} records")
    parser.add_argument("--golden-dir", type=Path, default=Path("tests/golden"))
    args = parser.parse_args()
    asyncio.run(main(args.input_file, args.golden_dir))
```

**Example JSONL input (`cases.jsonl`):**
```jsonl
{"case_id": "review-001", "input": "Review this code:\n\ndef f(x): return eval(x)\n"}
{"case_id": "review-002", "input": "Review this code:\n\nimport os\ndef g(): pass\n"}
```

---

## Exercise 2 — GoldenStore.diff

```python
import difflib

def diff(self, case_id: str, actual_traj: "Trajectory") -> str:
    """Return human-readable diff of tool names vs golden."""
    try:
        golden = self.load(case_id)
    except FileNotFoundError:
        return f"No golden for {case_id}"

    golden_names = golden.golden_trajectory.tool_names
    actual_names = actual_traj.tool_names

    diff_lines = list(
        difflib.unified_diff(
            golden_names,
            actual_names,
            fromfile="golden",
            tofile="actual",
            lineterm="",
        )
    )
    if not diff_lines:
        return "No diff — trajectories match."
    return "\n".join(diff_lines)
```

---

## Exercise 3 — min_pass_rate gate + pytest plugin fixture

```python
# In regression_gate.py
class RegressionGate:
    def __init__(
        self,
        store: GoldenStore,
        *,
        max_step_delta: int = 2,
        min_name_match: float = 0.8,
        min_pass_rate: float = 0.8,
    ) -> None:
        self._min_pass_rate = min_pass_rate
        # ... rest unchanged

    def run_report(self, results: list[dict]) -> RegressionReport:
        report = RegressionReport(total_cases=len(results))
        for item in results:
            regression = self.evaluate(...)
            if regression:
                report.regressions.append(regression)
            else:
                report.new_passes.append(item["case_id"])

        if report.pass_rate < self._min_pass_rate:
            raise AssertionError(
                f"Regression gate failed: pass_rate={report.pass_rate:.1%} "
                f"< threshold={self._min_pass_rate:.1%}\n{report.summary()}"
            )
        return report
```

**Pytest fixture:**
```python
# conftest.py
import pytest
from pathlib import Path
from src.eval.regression_gate import RegressionGate
from src.eval.golden_store import GoldenStore


@pytest.fixture
def regression_gate(tmp_path) -> RegressionGate:
    """CI regression gate with approved goldens loaded from tests/golden/."""
    golden_dir = Path("tests/golden")
    store = GoldenStore(golden_dir if golden_dir.exists() else tmp_path)
    return RegressionGate(store, min_pass_rate=0.8)
```

---

## Exercise 4 — YAML golden schema

```yaml
# tests/golden/review-001.yaml
case_id: review-001
user_input: "Review this code:\n\ndef f(x): return eval(x)\n"
golden_trajectory:
  steps:
    - tool_name: count_functions
      args: {}
    - tool_name: detect_security_issues
      args: {}
expected_output: "high"
captured_at: "2025-01-01T00:00:00+00:00"
approved: true
```

```python
import yaml  # requires: uv add pyyaml

def load_from_yaml(self, path: Path) -> GoldenRecord:
    data = yaml.safe_load(path.read_text())
    return GoldenRecord.model_validate(data)

def save_to_yaml(self, record: GoldenRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record.model_dump(), sort_keys=False))
```

**When YAML is preferable to JSON:**
- Human reviewers edit golden trajectories manually — YAML is more readable (no quotes on strings, multi-line strings natural).
- YAML supports comments (`# this golden was updated after the summarise tool was added`).
- For CI review: YAML diffs are cleaner in PRs.
- JSON is preferable for machine-to-machine exchange or when strict schema validation is paramount.
