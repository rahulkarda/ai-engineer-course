# Solutions — Week 07 · Day 7: Project Day & Reflection

## Exercise 1: retry_with_backoff decorator

```python
import functools
import time
import logging
import anthropic

logger = logging.getLogger(__name__)


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator: retry on anthropic.APIError with exponential backoff."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except anthropic.APIError as exc:
                    if attempt == max_retries - 1:
                        logger.error("max retries (%d) exceeded: %s", max_retries, exc)
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "APIError on attempt %d/%d, retrying in %.1fs: %s",
                        attempt + 1, max_retries, delay, exc,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# Usage on SpecialistRunner
class SpecialistRunner:
    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    @retry_with_backoff(max_retries=3)
    def execute(self, spec: "AgentSpec", task: "Task") -> "TaskResult":
        # ... existing implementation
        pass
```

## Exercise 2: PipelineMetrics

```python
import time
from dataclasses import dataclass, field

@dataclass
class StageMetric:
    name: str
    start: float = field(default_factory=time.perf_counter)
    end: float = 0.0

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000


@dataclass
class PipelineMetrics:
    stages: list[StageMetric] = field(default_factory=list)
    llm_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def start_stage(self, name: str) -> StageMetric:
        m = StageMetric(name=name)
        self.stages.append(m)
        return m

    def end_stage(self, metric: StageMetric) -> None:
        metric.end = time.perf_counter()

    def record_llm_call(self, usage: "anthropic.Usage | None" = None) -> None:
        self.llm_call_count += 1
        if usage:
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens

    def report(self) -> str:
        lines = [f"LLM calls: {self.llm_call_count}",
                 f"Tokens in: {self.total_input_tokens} | out: {self.total_output_tokens}"]
        for s in self.stages:
            lines.append(f"  {s.name}: {s.duration_ms:.0f}ms")
        return "\n".join(lines)


# In PipelineSupervisor.run():
# metrics = PipelineMetrics()
# m = metrics.start_stage("query_analysis")
# sub_queries = analyser.analyse(question)
# metrics.end_stage(m)
```

## Exercise 3: --dry-run flag

```python
# main.py
import argparse
import logging
import os
from unittest.mock import MagicMock
import anthropic
from src.supervisor import PipelineSupervisor

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

MOCK_RESPONSES = [
    '{"sub_queries": ["sub-query 1", "sub-query 2"]}',
    "Finding for sub-query 1.",
    "Finding for sub-query 2.",
    "DRAFT: Combined findings.\nCLAIMS:\n- claim 1\n- claim 2",
    "CLAIM: claim 1\nVERDICT: VERIFIED\nREASON: OK.\nCLAIM: claim 2\nVERDICT: VERIFIED\nREASON: OK.",
]


def make_dry_run_client() -> MagicMock:
    client = MagicMock()
    state = {"n": 0}

    def _create(**kwargs):
        resp = MagicMock()
        resp.content = [MagicMock(text=MOCK_RESPONSES[state["n"] % len(MOCK_RESPONSES)])]
        state["n"] += 1
        return resp

    client.messages.create.side_effect = _create
    return client


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="What is Python?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logging.getLogger().setLevel(logging.DEBUG)
        client = make_dry_run_client()
        print("[DRY RUN] Using mock LLM responses")
    else:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    supervisor = PipelineSupervisor(client)
    report = supervisor.run(args.question)
    print(f"\nSUMMARY: {report.summary}")
    print(f"VERIFIED CLAIMS: {len(report.verified_claims)}")


if __name__ == "__main__":
    main()
```

## Exercise 4: FactChecker with batched claims

```python
def check(self, claims: list[str], batch_size: int = 3) -> list[FactCheckResult]:
    """Process claims in batches to reduce LLM calls."""
    all_results: list[FactCheckResult] = []
    for i in range(0, len(claims), batch_size):
        batch = claims[i:i + batch_size]
        content = "\n".join(f"- {c}" for c in batch)
        resp = self._client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=512,
            system=self._system,
            messages=[{"role": "user", "content": f"Check these claims:\n{content}"}],
        )
        # parse same as before
        all_results.extend(self._parse(resp.content[0].text))
    return all_results

# Token saving: one API call for 3 claims instead of 3 calls.
# Prompt tokens: fixed overhead amortised over batch.
# Roughly 30–40% token reduction for prompt overhead.
```

## Exercise 5: test_supervisor_resilience

```python
# tests/test_pipeline.py (addition)

def test_supervisor_resilience_partial_researcher_failure() -> None:
    """Pipeline should complete even if one researcher raises."""
    from src.supervisor import PipelineSupervisor
    from unittest.mock import MagicMock, patch

    client = MagicMock()
    call_count = {"n": 0}
    # First response: analyser gives 2 sub-queries
    # Second response: researcher 0 will fail
    # Third response: researcher 1 succeeds
    # Fourth: synthesiser
    # Fifth: fact-checker
    responses = [
        '{"sub_queries": ["q1", "q2"]}',
        Exception("API timeout"),  # researcher 0 fails
        "Finding for q2.",          # researcher 1 succeeds
        "DRAFT: Partial findings.\nCLAIMS:\n- one claim",
        "CLAIM: one claim\nVERDICT: VERIFIED\nREASON: OK.",
    ]

    def _create(**kwargs):
        val = responses[call_count["n"] % len(responses)]
        call_count["n"] += 1
        if isinstance(val, Exception):
            raise val
        resp = MagicMock()
        resp.content = [MagicMock(text=val)]
        return resp

    client.messages.create.side_effect = _create

    supervisor = PipelineSupervisor(client)
    # Should NOT raise; should produce a partial report
    report = supervisor.run("Test resilience question")
    assert report is not None
    # At least one finding (from researcher 1)
    assert len(report.findings) >= 1
```

### Implementation note for resilience

In `PipelineSupervisor.run()`, wrap the `ThreadPoolExecutor` result collection:

```python
for fut in as_completed(futures):
    idx = futures[fut]
    try:
        finding = fut.result()
        findings.append(finding)
    except Exception as exc:
        logger.exception("researcher-%d failed: %s", idx, exc)
        # continue; other researchers still deliver their findings
```

This ensures partial failures don't abort the pipeline.
