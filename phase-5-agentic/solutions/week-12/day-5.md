# Week 12 · Day 5 Solutions — Evaluation Suite + Observability

## Exercise 1 — Tags filter in EvalHarness.run_all

```python
# src/finops_agent/evaluation/harness.py — extended run_all
def run_all(self, tags: list[str] | None = None) -> list[EvalResult]:
    cases = self.load_dataset()
    if tags:
        cases = [c for c in cases if any(t in c.tags for t in tags)]
        logger.info("Filtered to %d cases with tags %s", len(cases), tags)
    return [self.run_case(c) for c in cases]
```

Usage:
```python
harness = EvalHarness(agent_fn=my_agent)
hitl_results = harness.run_all(tags=["hitl"])
finance_results = harness.run_all(tags=["finance"])
```

## Exercise 2 — Histogram and p95

```python
# src/finops_agent/observability/metrics.py — extended MetricsCollector
import statistics

@dataclass
class MetricsCollector:
    _counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _spans: list[Span] = field(default_factory=list)
    _histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, metric: str, amount: int = 1) -> None:
        self._counters[metric] += amount

    def histogram(self, metric: str, value: float) -> None:
        self._histograms[metric].append(value)
        logger.debug("Histogram %s += %.3f (n=%d)", metric, value, len(self._histograms[metric]))

    def p95(self, metric: str) -> float | None:
        values = self._histograms.get(metric)
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * 0.95)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]

    def start_span(self, name: str, parent_name: str | None = None, **attrs) -> Span:
        span = Span(name=name, attributes=attrs, parent_name=parent_name)
        self._spans.append(span)
        return span

    def get_counters(self) -> dict[str, int]:
        return dict(self._counters)

    def get_trace(self) -> list[dict]:
        return [
            {"name": s.name, "duration_ms": s.duration_ms, "parent": s.parent_name, "attributes": s.attributes}
            for s in self._spans
        ]
```

Test:
```python
def test_histogram_p95():
    from finops_agent.observability.metrics import MetricsCollector
    m = MetricsCollector()
    for i in range(100):
        m.histogram("latency_ms", float(i))
    p95 = m.p95("latency_ms")
    assert p95 is not None
    assert 93.0 <= p95 <= 96.0  # p95 of 0..99 is ~94

def test_histogram_empty_returns_none():
    from finops_agent.observability.metrics import MetricsCollector
    m = MetricsCollector()
    assert m.p95("nonexistent") is None
```

## Exercise 3 — Integration test with mock agent

```python
# tests/integration/test_eval_harness.py
import json
import os
import tempfile
import pytest

def test_eval_harness_mock_agent():
    from finops_agent.evaluation.harness import EvalHarness

    def mock_agent(task: str):
        # Always calls retrieve_invoice and returns INV-001
        return (["retrieve_invoice"], "INV-001 retrieved successfully")

    # Write 3-case dataset to a temp file
    cases = [
        {"case_id": "t1", "input": "retrieve INV-001", "expected_tool_calls": ["retrieve_invoice"], "expected_answer_contains": ["INV-001"], "tags": ["finance"]},
        {"case_id": "t2", "input": "get invoice", "expected_tool_calls": ["retrieve_invoice"], "expected_answer_contains": ["retrieved"], "tags": ["finance"]},
        {"case_id": "t3", "input": "fetch INV-001 data", "expected_tool_calls": ["retrieve_invoice"], "expected_answer_contains": ["INV-001"], "tags": ["finance"]},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
        tmp_path = f.name

    try:
        harness = EvalHarness(agent_fn=mock_agent, dataset_path=tmp_path)
        results = harness.run_all()
        report = harness.report(results)
        assert report["pass_rate"] > 0
        assert report["total"] == 3
        assert report["passed"] == 3
    finally:
        os.unlink(tmp_path)
```

## Exercise 4 — 3 additional eval cases

Additional entries for `eval/dataset.jsonl`:
```json
{"case_id": "fin-003", "input": "Check engineering budget for $60000 - should be rejected", "expected_tool_calls": ["check_budget"], "expected_answer_contains": ["engineering"], "expected_approval_required": false, "tags": ["finance", "budget", "rejection"]}
{"case_id": "fin-004", "input": "Reject payment for INV-002", "expected_tool_calls": ["approve_payment"], "expected_answer_contains": ["rejected"], "expected_approval_required": true, "tags": ["finance", "approval", "hitl"]}
{"case_id": "fin-005", "input": "Retrieve invoice INV-999 that does not exist", "expected_tool_calls": ["retrieve_invoice"], "expected_answer_contains": ["not found"], "expected_approval_required": false, "tags": ["finance", "error"]}
```

Note on the "not found" case: the `retrieve_invoice` tool raises `ValueError` for unknown IDs. The eval harness catches exceptions and marks the case as failed. Your mock agent (or real agent) should either propagate the error message to the answer string, or the harness should be extended to treat certain errors as expected. For the error case, add an `expected_error_contains` field to `EvalCase` and handle it in `run_case`.
