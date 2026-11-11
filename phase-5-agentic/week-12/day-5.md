# Phase 5 · Week 12 · Day 5 — Evaluation Suite + Observability

> Objective: Build a dataset-driven eval harness and instrument agent traces with metrics and spans.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]], [[day-3]], [[day-4]]

## Concept

Agent evaluation differs fundamentally from unit tests. Unit tests check deterministic function outputs with exact assertions. Agent evals check *behavioral properties* of non-deterministic, multi-step systems:

- Did the agent call the right tools in roughly the right order?
- Does the final answer contain the expected information?
- Was the approval gate triggered when it should have been?
- How many steps did it take, and how long did it run?

**Eval dataset design:** Structure each case with an input, expected tool calls, expected answer phrases, and tags for filtering. Store as JSONL — one case per line — for easy streaming and incremental addition.

**Failure categorization:**
- *Wrong tool*: agent used a different tool than expected
- *Missing phrase*: answer lacks required information
- *Hallucination*: agent fabricated an invoice ID or amount not in the retrieval result
- *Timeout*: exceeded step budget
- *Gate miss*: critical action ran without approval

**Observability triad:**
- *Metrics*: counters and histograms — `tool_calls_total`, `api_requests_total`, latency
- *Traces*: spans with parent/child relationships — one span per agent step, nested under a root span per request
- *Logs*: structured log lines tying a session ID to each action

## How it works internally

The eval harness wraps the agent as a callable `(input: str) -> (tool_calls: list[str], answer: str)`. For each JSONL case it:

1. Calls the agent function and times it
2. Compares actual tool calls against `expected_tool_calls` — reports missing calls as errors
3. Checks that `expected_answer_contains` phrases appear (case-insensitive) in the answer
4. Computes a `score` proportional to how many checks passed
5. Aggregates into a report with `pass_rate`, `avg_latency_ms`, `avg_score`

This avoids exact string match, which is brittle across model versions and prompt tweaks. Structural property checks survive rephrasing.

For observability, `MetricsCollector` holds in-memory counters and a list of `Span` objects. Each `Span` records a name, start/end time, optional parent name, and free-form attributes. The trace is a flat list — reconstructing the tree is done by following `parent_name` links. This is intentionally simple: a production system would export to OpenTelemetry.

The planner integrates observability by accepting an optional `MetricsCollector`. This keeps the planner testable without an observability dependency.

## Worked example

```python
# src/finops_agent/evaluation/harness.py
import json
import logging
import time
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class EvalCase(BaseModel):
    case_id: str
    input: str
    expected_tool_calls: list[str] = Field(default_factory=list)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_approval_required: bool = False
    tags: list[str] = Field(default_factory=list)

class EvalResult(BaseModel):
    case_id: str
    passed: bool
    latency_ms: float
    tool_calls_made: list[str] = Field(default_factory=list)
    final_answer: str = ""
    errors: list[str] = Field(default_factory=list)
    score: float = 0.0

class EvalHarness:
    def __init__(self, agent_fn, dataset_path: str = "eval/dataset.jsonl"):
        self.agent_fn = agent_fn  # callable(input: str) -> (tool_calls: list[str], answer: str)
        self.dataset_path = Path(dataset_path)

    def load_dataset(self) -> list[EvalCase]:
        cases = []
        with self.dataset_path.open() as f:
            for line in f:
                cases.append(EvalCase.model_validate_json(line.strip()))
        logger.info("Loaded %d eval cases from %s", len(cases), self.dataset_path)
        return cases

    def run_case(self, case: EvalCase) -> EvalResult:
        start = time.perf_counter()
        errors: list[str] = []
        try:
            tool_calls, answer = self.agent_fn(case.input)
        except Exception as e:
            logger.error("Eval case %s failed: %s", case.case_id, e)
            return EvalResult(case_id=case.case_id, passed=False, latency_ms=0, errors=[str(e)])

        latency_ms = (time.perf_counter() - start) * 1000

        # Check expected tool calls
        missing_tools = [t for t in case.expected_tool_calls if t not in tool_calls]
        if missing_tools:
            errors.append(f"Missing tool calls: {missing_tools}")

        # Check answer contains expected strings
        for phrase in case.expected_answer_contains:
            if phrase.lower() not in answer.lower():
                errors.append(f"Answer missing expected phrase: {phrase!r}")

        passed = len(errors) == 0
        score = 1.0 - (len(errors) / max(1, len(case.expected_tool_calls) + len(case.expected_answer_contains)))
        score = max(0.0, score)

        result = EvalResult(
            case_id=case.case_id,
            passed=passed,
            latency_ms=latency_ms,
            tool_calls_made=tool_calls,
            final_answer=answer,
            errors=errors,
            score=score,
        )
        logger.info("Eval %s: passed=%s score=%.2f latency=%.0fms", case.case_id, passed, score, latency_ms)
        return result

    def run_all(self) -> list[EvalResult]:
        cases = self.load_dataset()
        return [self.run_case(c) for c in cases]

    def report(self, results: list[EvalResult]) -> dict[str, Any]:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_latency = sum(r.latency_ms for r in results) / max(1, total)
        avg_score = sum(r.score for r in results) / max(1, total)
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / max(1, total),
            "avg_latency_ms": avg_latency,
            "avg_score": avg_score,
        }
```

Sample eval dataset (`eval/dataset.jsonl`):

```json
{"case_id": "fin-001", "input": "Retrieve invoice INV-001 and validate payment of $15000", "expected_tool_calls": ["retrieve_invoice", "validate_payment"], "expected_answer_contains": ["INV-001", "valid"], "expected_approval_required": false, "tags": ["finance", "retrieval"]}
{"case_id": "fin-002", "input": "Approve payment for INV-002 of $4200", "expected_tool_calls": ["approve_payment"], "expected_answer_contains": ["approval"], "expected_approval_required": true, "tags": ["finance", "approval", "hitl"]}
{"case_id": "ops-001", "input": "Check budget for engineering department for $20000", "expected_tool_calls": ["check_budget"], "expected_answer_contains": ["engineering", "approved"], "expected_approval_required": false, "tags": ["ops", "budget"]}
```

Observability metrics:

```python
# src/finops_agent/observability/metrics.py
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_name: str | None = None

    def finish(self) -> None:
        self.end_time = time.perf_counter()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000

@dataclass
class MetricsCollector:
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _spans: list[Span] = field(default_factory=list)

    def increment(self, metric: str, amount: int = 1) -> None:
        self._counters[metric] += amount
        logger.debug("Metric %s=%d", metric, self._counters[metric])

    def start_span(self, name: str, parent_name: str | None = None, **attrs) -> Span:
        span = Span(name=name, attributes=attrs, parent_name=parent_name)
        self._spans.append(span)
        return span

    def get_counters(self) -> dict[str, int]:
        return dict(self._counters)

    def get_trace(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "duration_ms": s.duration_ms,
                "parent": s.parent_name,
                "attributes": s.attributes,
            }
            for s in self._spans
        ]
```

Instrumenting the planner:

```python
# Updated ReActPlanner snippet showing observability integration
def plan(self, session_id: str, task: str, metrics: "MetricsCollector | None" = None) -> list:
    span = metrics.start_span("planner.run", session_id=session_id) if metrics else None
    steps = []
    # ... existing plan loop ...
    for step_num in range(self.max_steps):
        tool_span = metrics.start_span("tool.call", parent_name="planner.run") if metrics else None
        # ... tool execution ...
        if metrics:
            metrics.increment("tool_calls_total")
            if tool_span:
                tool_span.finish()
    if span:
        span.finish()
    if metrics:
        metrics.increment("planning_sessions_total")
    return steps
```

## Exercises

1. Add a `tags` filter to `EvalHarness.run_all` so you can run only cases with a specific tag (e.g., `"hitl"`). The filter should be optional — when not provided, all cases run.
2. Extend `MetricsCollector` with a `histogram(metric: str, value: float)` method that stores a list of float values per metric name. Add a `p95(metric: str) -> float` method that returns the 95th percentile value.
3. Write an integration test that runs the eval harness against a mock `agent_fn` that always calls `retrieve_invoice` and returns `"INV-001 retrieved"`. Verify `pass_rate > 0` and that `avg_latency_ms` is a positive float.
4. The eval dataset only has 3 cases. Add 3 more cases covering: (a) a failed budget check that should return an error message, (b) a rejected payment that should contain "rejected", (c) an unknown invoice lookup (`INV-999`) that should return an error phrase.

## Self-check

- [ ] I can build an eval harness that checks structural properties rather than exact string match
- [ ] I can write JSONL eval datasets with expected tool calls and answer phrases
- [ ] I understand spans, traces, and how to instrument a ReAct loop
- [ ] I can compute pass rate, average latency, and average score from eval results
