# Solutions — Phase 5 · Week 03 · Project — Research Assistant Plan-and-Execute Agent

---

## Project overview

A production-quality Research Assistant agent that uses the plan-and-execute pattern to
decompose a research goal into parallel search + summarize tasks, then synthesizes the
results into a structured `ResearchReport`. Includes automatic re-planning on transient
failures and a full pytest suite with all LLM calls mocked.

---

## Directory layout

```
research_assistant/
    __init__.py
    models.py
    planner.py
    executor.py
    tools.py
    agent.py
tests/
    __init__.py
    test_models.py
    test_planner.py
    test_executor.py
    test_agent.py
.env.example
pyproject.toml
```

---

## How to run

```bash
# 1. Install dependencies
pip install pydantic anthropic pytest

# 2. Run the full test suite (no network required — LLM is mocked)
pytest tests/ -v

# 3. Run against the real Anthropic API
export ANTHROPIC_API_KEY="sk-ant-..."   # your key — never committed
python -m research_assistant.agent
```

---

## `.env.example`

```
ANTHROPIC_API_KEY=your-key-here
```

---

## `research_assistant/models.py`

```python
"""
Pydantic v2 models for the Research Assistant.

Design decisions:
- Discriminated union on `step_type` for type-safe dispatch without isinstance chains.
- ResearchReport separates raw sources from the synthesized body so consumers can
  re-rank, re-summarize, or cite individual sources without re-running the agent.
- ExecutionReport wraps the research report so callers get both run metadata
  (replan_count, step statuses) and the final deliverable in one object.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Step types
# ---------------------------------------------------------------------------

class SearchStep(BaseModel):
    step_type: Literal["search"] = "search"
    step_id: str
    query: str
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = 2


class SummarizeStep(BaseModel):
    step_type: Literal["summarize"] = "summarize"
    step_id: str
    source_step_id: str
    max_words: int = 150
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = 2


class WriteReportStep(BaseModel):
    step_type: Literal["write_report"] = "write_report"
    step_id: str
    title: str
    source_step_ids: list[str]
    depends_on: list[str] = Field(default_factory=list)
    max_retries: int = 1


ResearchStep = Annotated[
    Union[SearchStep, SummarizeStep, WriteReportStep],
    Field(discriminator="step_type"),
]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class ResearchPlan(BaseModel):
    goal: str
    steps: list[ResearchStep]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    step_id: str
    step_type: str
    status: Literal["success", "failed", "skipped"]
    output: str | None = None
    error: str | None = None
    attempt: int = 1


# ---------------------------------------------------------------------------
# Final deliverables
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """A single search-summarize pair contributing to the report."""
    search_query: str
    raw_results: str
    summary: str


class ResearchReport(BaseModel):
    """
    The structured deliverable produced by the Research Assistant.

    Fields:
    - title: human-readable report title
    - goal: the original user goal (for provenance)
    - sources: list of (query, raw, summary) triples
    - body: the final synthesized markdown report
    - generated_at: UTC timestamp
    - replan_count: how many times execution was replanned
    - success: False if the report is incomplete
    """
    title: str
    goal: str
    sources: list[Source]
    body: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    replan_count: int = 0
    success: bool


class ExecutionReport(BaseModel):
    """Execution metadata + final ResearchReport."""
    goal: str
    steps: list[StepResult]
    replan_count: int
    success: bool
    research_report: ResearchReport | None = None

    def summary(self) -> str:
        succeeded = sum(1 for s in self.steps if s.status == "success")
        failed = sum(1 for s in self.steps if s.status == "failed")
        return (
            f"Goal       : {self.goal}\n"
            f"Steps      : {len(self.steps)} total | {succeeded} success | {failed} failed\n"
            f"Re-plans   : {self.replan_count}\n"
            f"Success    : {self.success}\n"
        )
```

---

## `research_assistant/planner.py`

```python
"""
ResearchPlanner: LLM-backed planner with dependency injection for testing.

Design decision: LLMInterface is a Protocol (not an ABC) so any object with
a .generate(str) -> str method satisfies it — no inheritance required.
This makes mocking trivial and avoids coupling to the Anthropic SDK in tests.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, runtime_checkable

from research_assistant.models import ResearchPlan

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMInterface(Protocol):
    def generate(self, prompt: str) -> str: ...


class AnthropicLLM:
    """
    Production LLM using the Anthropic SDK.

    Model: claude-sonnet-5-5 for planning (good instruction-following).
          Swap to claude-haiku-5-5 for cost reduction if plan quality is sufficient.
    API key: os.getenv("ANTHROPIC_API_KEY") — never hard-coded.
    """

    def __init__(self, model: str = "claude-sonnet-5-5") -> None:
        self._model = model
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set — real calls will fail.")

    def generate(self, prompt: str) -> str:
        import anthropic  # deferred — not imported in test environments
        client = anthropic.Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


_PLAN_TEMPLATE = """
You are a research planning assistant.

GOAL: {goal}

Decompose the goal into a research plan. Use ONLY these step types:
- search      : web/database search
- summarize   : summarize a previous search step's output
- write_report: synthesize summaries into a final report

Typical pattern:
  search(topic A) -> summarize(search A) -> search(topic B) -> summarize(search B)
  -> write_report(summarize A, summarize B)

Output a single JSON object:
{{
  "goal": "<goal>",
  "steps": [
    {{"step_type": "search",      "step_id": "s1", "query": "...", "depends_on": []}},
    {{"step_type": "summarize",   "step_id": "s2", "source_step_id": "s1", "depends_on": ["s1"]}},
    {{"step_type": "write_report","step_id": "s3", "title": "...",
      "source_step_ids": ["s2"], "depends_on": ["s2"]}}
  ]
}}

Output ONLY valid JSON — no prose, no markdown fences.
""".strip()

_REPLAN_TEMPLATE = """
You are a research planning assistant performing a recovery replan.

GOAL: {goal}
COMPLETED STEPS (succeeded): {completed_ids}
FAILED STEP: {failed_step_id}
ERROR: {error}

Produce a revised JSON plan tail — only the steps that still need to run.
Do NOT repeat completed steps. Use the same step types and schema as before.
Output ONLY valid JSON.
""".strip()


class ResearchPlanner:
    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    def decompose(self, goal: str) -> ResearchPlan:
        logger.info("ResearchPlanner.decompose: goal=%r", goal)
        raw = self._llm.generate(_PLAN_TEMPLATE.format(goal=goal))
        return self._parse(raw)

    def replan(
        self,
        goal: str,
        completed_ids: list[str],
        failed_step_id: str,
        error: str,
    ) -> ResearchPlan:
        logger.warning(
            "ResearchPlanner.replan: failed=%r error=%s", failed_step_id, error
        )
        prompt = _REPLAN_TEMPLATE.format(
            goal=goal,
            completed_ids=completed_ids,
            failed_step_id=failed_step_id,
            error=error,
        )
        raw = self._llm.generate(prompt)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> ResearchPlan:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ResearchPlanner: LLM returned non-JSON: {exc}") from exc
        return ResearchPlan.model_validate(data)
```

---

## `research_assistant/tools.py`

```python
"""
Research tools.

All functions are deterministic stubs so tests require no network access.
To switch to real implementations:
  - search: call SerpAPI, Bing Search, or Tavily.
  - summarize: call claude-haiku-5-5 via the Anthropic SDK.
  - write_report: call claude-sonnet-5-5 or claude-opus-5-5.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5) -> str:
    """
    Deterministic search stub.

    Production swap: replace body with a real search API call.
    Keep the signature (query: str, **kwargs) -> str stable so the executor
    does not need to change.
    """
    logger.info("tool:search query=%r max_results=%d", query, max_results)
    return (
        f"[SEARCH: '{query}']\n"
        f"Result 1: Comprehensive overview of {query}.\n"
        f"Result 2: Historical background of {query}.\n"
        f"Result 3: Recent developments in {query}.\n"
    )


def summarize(text: str, max_words: int = 150) -> str:
    """
    Deterministic summarizer stub.

    Production swap: call claude-haiku-5-5 with a summarization prompt.
    """
    logger.info("tool:summarize len=%d max_words=%d", len(text), max_words)
    words = text.split()
    truncated = " ".join(words[:max_words])
    suffix = " [truncated]" if len(words) > max_words else ""
    return f"[SUMMARY]: {truncated}{suffix}"


def write_report(title: str, sections: list[str]) -> str:
    """
    Assemble section summaries into a markdown report.

    Production swap: call claude-opus-5-5 or claude-sonnet-5-5 to generate
    a coherent narrative from the sections rather than just joining them.
    """
    logger.info("tool:write_report title=%r sections=%d", title, len(sections))
    body = f"# {title}\n\n"
    for i, section in enumerate(sections, 1):
        body += f"## Section {i}\n\n{section}\n\n"
    return body.rstrip()
```

---

## `research_assistant/executor.py`

```python
"""
ResearchExecutor: runs a ResearchPlan, handles retries and replanning.

Design decisions:
- Topological sort: guarantees dependency-respecting execution order and catches
  cycles at plan-validation time rather than at runtime.
- Retry on each step: max_retries per step (not per plan); reduces spurious replans
  for transient errors.
- ReplanHook: typed callable injected at construction — keeps the executor
  decoupled from the Planner.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Callable

from research_assistant.models import (
    ExecutionReport, ResearchPlan, ResearchReport,
    SearchStep, Source, StepResult, SummarizeStep, WriteReportStep,
)

logger = logging.getLogger(__name__)

ReplanHook = Callable[
    [str, list[StepResult], StepResult, str],
    ResearchPlan,
]


def _topological_sort(plan: ResearchPlan) -> list[str]:
    in_deg: dict[str, int] = {s.step_id: 0 for s in plan.steps}
    children: dict[str, list[str]] = defaultdict(list)
    for step in plan.steps:
        for dep in step.depends_on:
            children[dep].append(step.step_id)
            in_deg[step.step_id] += 1
    queue = deque(sid for sid, d in in_deg.items() if d == 0)
    order: list[str] = []
    while queue:
        sid = queue.popleft()
        order.append(sid)
        for child in children[sid]:
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)
    if len(order) != len(plan.steps):
        cycle_ids = set(in_deg) - set(order)
        raise ValueError(f"Dependency cycle detected in plan. Cyclic steps: {cycle_ids}")
    return order


class ResearchExecutor:
    def __init__(
        self,
        search_fn: Callable[[str], str],
        summarize_fn: Callable[[str, int], str],
        write_report_fn: Callable[[str, list[str]], str],
        replan_hook: ReplanHook | None = None,
        max_replans: int = 3,
    ) -> None:
        self._search = search_fn
        self._summarize = summarize_fn
        self._write_report = write_report_fn
        self._replan_hook = replan_hook
        self._max_replans = max_replans

    def run(self, plan: ResearchPlan) -> ExecutionReport:
        completed: list[StepResult] = []
        outputs: dict[str, str] = {}
        sources: list[Source] = []
        replan_count = 0
        current_plan = plan
        final_body: str | None = None

        while True:
            try:
                order = _topological_sort(current_plan)
            except ValueError as exc:
                logger.error("Invalid plan: %s", exc)
                return self._abort(plan.goal, completed, replan_count)

            step_map = {s.step_id: s for s in current_plan.steps}
            failed: StepResult | None = None

            for sid in order:
                step = step_map[sid]
                result = self._run_step(step, outputs, sources)
                completed.append(result)

                if result.status == "success" and result.output:
                    outputs[sid] = result.output
                    if isinstance(step, WriteReportStep):
                        final_body = result.output

                if result.status == "failed":
                    failed = result
                    break  # exit inner loop; decide replan or abort

            if failed is None:
                break  # all steps succeeded

            if self._replan_hook is None or replan_count >= self._max_replans:
                logger.error(
                    "Aborting: no replan hook or max_replans (%d) reached.", self._max_replans
                )
                return self._abort(plan.goal, completed, replan_count)

            replan_count += 1
            logger.warning(
                "ResearchExecutor: replan %d/%d triggered by step '%s': %s",
                replan_count, self._max_replans, failed.step_id, failed.error,
            )
            current_plan = self._replan_hook(
                plan.goal, completed, failed, failed.error or "unknown"
            )

        research_report = ResearchReport(
            title=plan.goal,
            goal=plan.goal,
            sources=sources,
            body=final_body or "(no write_report step produced output)",
            replan_count=replan_count,
            success=True,
        )
        return ExecutionReport(
            goal=plan.goal,
            steps=completed,
            replan_count=replan_count,
            success=True,
            research_report=research_report,
        )

    def _run_step(
        self,
        step: SearchStep | SummarizeStep | WriteReportStep,
        outputs: dict[str, str],
        sources: list[Source],
    ) -> StepResult:
        max_retries = getattr(step, "max_retries", 1)
        step_type = step.step_type
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                if isinstance(step, SearchStep):
                    output = self._search(step.query)
                    return StepResult(
                        step_id=step.step_id, step_type=step_type,
                        status="success", output=output, attempt=attempt,
                    )

                if isinstance(step, SummarizeStep):
                    source_text = outputs.get(step.source_step_id, "")
                    output = self._summarize(source_text, step.max_words)
                    sources.append(Source(
                        search_query=step.source_step_id,
                        raw_results=source_text,
                        summary=output,
                    ))
                    return StepResult(
                        step_id=step.step_id, step_type=step_type,
                        status="success", output=output, attempt=attempt,
                    )

                if isinstance(step, WriteReportStep):
                    sections = [
                        outputs[sid]
                        for sid in step.source_step_ids
                        if sid in outputs
                    ]
                    output = self._write_report(step.title, sections)
                    return StepResult(
                        step_id=step.step_id, step_type=step_type,
                        status="success", output=output, attempt=attempt,
                    )

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Step '%s' attempt %d/%d failed: %s",
                    step.step_id, attempt, max_retries, exc,
                )

        return StepResult(
            step_id=step.step_id, step_type=step_type,
            status="failed", error=str(last_exc),
            attempt=max_retries,
        )

    @staticmethod
    def _abort(
        goal: str, completed: list[StepResult], replan_count: int
    ) -> ExecutionReport:
        report = ResearchReport(
            title="Incomplete",
            goal=goal,
            sources=[],
            body="Execution aborted before report could be written.",
            replan_count=replan_count,
            success=False,
        )
        return ExecutionReport(
            goal=goal,
            steps=completed,
            replan_count=replan_count,
            success=False,
            research_report=report,
        )
```

---

## `research_assistant/agent.py`

```python
"""
ResearchAssistantAgent: top-level entry point.

Wires together ResearchPlanner + ResearchExecutor. Thin by design —
all business logic lives in executor.py and planner.py.
"""
from __future__ import annotations

import logging

from research_assistant.executor import ResearchExecutor
from research_assistant.models import ExecutionReport, ResearchPlan, StepResult
from research_assistant.planner import AnthropicLLM, LLMInterface, ResearchPlanner
from research_assistant.tools import search, summarize, write_report

logger = logging.getLogger(__name__)


class ResearchAssistantAgent:
    """
    Orchestrates ResearchPlanner + ResearchExecutor for a research goal.

    Parameters
    ----------
    llm : LLMInterface
        Any object with a .generate(str) -> str method.
        Use AnthropicLLM() in production; a MagicMock in tests.
    max_replans : int
        Hard cap on replan cycles. Prevents infinite loops.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        max_replans: int = 3,
    ) -> None:
        self._llm = llm or AnthropicLLM()
        self._planner = ResearchPlanner(self._llm)
        self._max_replans = max_replans

    def run(self, goal: str) -> ExecutionReport:
        logger.info("ResearchAssistantAgent.run: goal=%r", goal)

        plan: ResearchPlan = self._planner.decompose(goal)
        logger.info("Plan: %d steps generated.", len(plan.steps))

        def replan_hook(
            g: str,
            completed: list[StepResult],
            failed: StepResult,
            error: str,
        ) -> ResearchPlan:
            done_ids = [r.step_id for r in completed if r.status == "success"]
            return self._planner.replan(g, done_ids, failed.step_id, error)

        executor = ResearchExecutor(
            search_fn=search,
            summarize_fn=summarize,
            write_report_fn=write_report,
            replan_hook=replan_hook,
            max_replans=self._max_replans,
        )
        report = executor.run(plan)
        logger.info(
            "ResearchAssistantAgent: done. success=%s replan_count=%d",
            report.success, report.replan_count,
        )
        return report


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    agent = ResearchAssistantAgent()
    result = agent.run("Research the history of the Turing Award and write a summary")
    if result.research_report:
        print(result.research_report.body)
    print(result.summary())
```

---

## Tests

### `tests/test_models.py`

```python
from __future__ import annotations

import json
import pytest

from research_assistant.models import (
    ExecutionReport, ResearchPlan, ResearchReport,
    SearchStep, Source, StepResult, SummarizeStep, WriteReportStep,
)


def test_discriminated_union_mixed_list() -> None:
    data = {
        "goal": "test",
        "steps": [
            {"step_type": "search",       "step_id": "s1", "query": "q"},
            {"step_type": "summarize",    "step_id": "s2", "source_step_id": "s1"},
            {"step_type": "write_report", "step_id": "s3", "title": "T",
             "source_step_ids": ["s2"]},
        ],
    }
    plan = ResearchPlan.model_validate(data)
    assert isinstance(plan.steps[0], SearchStep)
    assert isinstance(plan.steps[1], SummarizeStep)
    assert isinstance(plan.steps[2], WriteReportStep)


def test_invalid_step_type_raises() -> None:
    with pytest.raises(Exception):
        ResearchPlan.model_validate({
            "goal": "g",
            "steps": [{"step_type": "unknown", "step_id": "x"}],
        })


def test_execution_report_json_round_trip() -> None:
    report = ExecutionReport(
        goal="test",
        steps=[StepResult(step_id="s1", step_type="search", status="success")],
        replan_count=0,
        success=True,
    )
    data = json.loads(report.model_dump_json())
    assert data["success"] is True
    assert data["replan_count"] == 0


def test_research_report_has_generated_at() -> None:
    r = ResearchReport(
        title="T", goal="g", sources=[], body="body", success=True
    )
    assert r.generated_at is not None
```

### `tests/test_planner.py`

```python
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from research_assistant.models import ResearchPlan, SearchStep
from research_assistant.planner import ResearchPlanner


def _plan_json() -> str:
    return json.dumps({
        "goal": "test",
        "steps": [
            {"step_type": "search", "step_id": "s1", "query": "q", "depends_on": []},
        ],
    })


@pytest.fixture()
def planner() -> ResearchPlanner:
    llm = MagicMock()
    llm.generate.return_value = _plan_json()
    return ResearchPlanner(llm=llm)


def test_decompose_returns_plan(planner: ResearchPlanner) -> None:
    plan = planner.decompose("test goal")
    assert isinstance(plan, ResearchPlan)


def test_decompose_step_types(planner: ResearchPlanner) -> None:
    plan = planner.decompose("test goal")
    assert all(isinstance(s, SearchStep) for s in plan.steps)


def test_invalid_json_raises(planner: ResearchPlanner) -> None:
    planner._llm.generate.return_value = "not json"  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        planner.decompose("goal")
```

### `tests/test_executor.py`

```python
from __future__ import annotations

import pytest

from research_assistant.executor import ResearchExecutor, _topological_sort
from research_assistant.models import (
    ExecutionReport, ResearchPlan, SearchStep, SummarizeStep, WriteReportStep,
)


def _make_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="test",
        steps=[
            SearchStep(step_id="s1", query="q", depends_on=[]),
            SummarizeStep(step_id="s2", source_step_id="s1",
                          depends_on=["s1"], max_retries=1),
            WriteReportStep(step_id="s3", title="T",
                            source_step_ids=["s2"], depends_on=["s2"]),
        ],
    )


def _make_executor(**kwargs) -> ResearchExecutor:
    return ResearchExecutor(
        search_fn=lambda q: f"results:{q}",
        summarize_fn=lambda t, n: f"summary:{t[:20]}",
        write_report_fn=lambda title, secs: f"# {title}\n" + "\n".join(secs),
        **kwargs,
    )


class TestTopologicalSort:
    def test_linear_order(self) -> None:
        plan = _make_plan()
        order = _topological_sort(plan)
        assert order.index("s1") < order.index("s2") < order.index("s3")

    def test_cycle_raises(self) -> None:
        plan = ResearchPlan(
            goal="g",
            steps=[
                SearchStep(step_id="a", query="q", depends_on=["b"]),
                SearchStep(step_id="b", query="q", depends_on=["a"]),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            _topological_sort(plan)


class TestExecutorHappyPath:
    def test_success_true(self) -> None:
        report = _make_executor().run(_make_plan())
        assert report.success is True

    def test_all_steps_recorded(self) -> None:
        report = _make_executor().run(_make_plan())
        assert len(report.steps) == 3

    def test_report_type(self) -> None:
        report = _make_executor().run(_make_plan())
        assert isinstance(report, ExecutionReport)

    def test_research_report_attached(self) -> None:
        report = _make_executor().run(_make_plan())
        assert report.research_report is not None
        assert "# T" in report.research_report.body


class TestExecutorReplan:
    def test_replan_triggered_on_failure(self) -> None:
        call_n = {"n": 0}

        def flaky_summarize(text: str, max_words: int) -> str:
            call_n["n"] += 1
            if call_n["n"] == 1:
                raise ValueError("bad text")
            return "ok summary"

        replan_plan = ResearchPlan(
            goal="test",
            steps=[
                SummarizeStep(step_id="s2_retry", source_step_id="s1", depends_on=[]),
                WriteReportStep(step_id="s3", title="T",
                                source_step_ids=["s2_retry"], depends_on=["s2_retry"]),
            ],
        )

        executor = ResearchExecutor(
            search_fn=lambda q: "results",
            summarize_fn=flaky_summarize,
            write_report_fn=lambda title, secs: f"# {title}",
            replan_hook=lambda *_: replan_plan,
            max_replans=3,
        )
        report = executor.run(_make_plan())
        assert report.replan_count == 1
        assert report.success is True
```

### `tests/test_agent.py`

```python
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from research_assistant.agent import ResearchAssistantAgent
from research_assistant.models import ExecutionReport, ResearchReport


def _three_step_plan(goal: str = "test") -> str:
    return json.dumps({
        "goal": goal,
        "steps": [
            {"step_type": "search",       "step_id": "s1",
             "query": "q",               "depends_on": []},
            {"step_type": "summarize",    "step_id": "s2",
             "source_step_id": "s1",     "depends_on": ["s1"]},
            {"step_type": "write_report", "step_id": "s3",
             "title": "Report",          "source_step_ids": ["s2"],
             "depends_on": ["s2"]},
        ],
    })


@pytest.fixture()
def mock_llm() -> MagicMock:
    m = MagicMock()
    m.generate.return_value = _three_step_plan()
    return m


@pytest.fixture()
def agent(mock_llm: MagicMock) -> ResearchAssistantAgent:
    return ResearchAssistantAgent(llm=mock_llm)


def test_returns_execution_report(agent: ResearchAssistantAgent) -> None:
    assert isinstance(agent.run("History of AI"), ExecutionReport)


def test_success_true(agent: ResearchAssistantAgent) -> None:
    assert agent.run("History of AI").success is True


def test_research_report_is_research_report(agent: ResearchAssistantAgent) -> None:
    r = agent.run("History of AI").research_report
    assert isinstance(r, ResearchReport)


def test_body_non_empty(agent: ResearchAssistantAgent) -> None:
    r = agent.run("History of AI").research_report
    assert r and len(r.body) > 0


def test_serialises_to_json(agent: ResearchAssistantAgent) -> None:
    data = json.loads(agent.run("History of AI").model_dump_json())
    assert "success" in data
    assert "research_report" in data


def test_no_replans_in_happy_path(agent: ResearchAssistantAgent) -> None:
    assert agent.run("History of AI").replan_count == 0
```

---

## Design decisions

**Why a discriminated union for step types?**
Each step type has a different set of required fields (e.g., `WriteReportStep` needs
`source_step_ids`; `SearchStep` needs `query`). A discriminated union validates each
step against its own model and gives exhaustive type-narrowing in the executor's dispatch
logic without `isinstance` chains.

**Why topological sort at plan-validation time?**
Detecting cycles at runtime — when some steps have already executed — wastes tool calls
and produces a partially-completed state that is hard to reason about. Failing fast at
validation keeps the executor simple: if it enters the execution loop, the plan is known
to be acyclically ordered.

**Why inject tool functions rather than calling module-level functions?**
Dependency injection makes tests trivial: pass a `lambda` that returns a known string,
a counter that raises on the Nth call, or any other deterministic stub. The executor
has no knowledge of network or LLM APIs.

**Why separate `ResearchReport` from `ExecutionReport`?**
`ExecutionReport` records run metadata (step statuses, replan_count). `ResearchReport`
is the domain deliverable. Keeping them separate means callers that only care about the
final report do not have to unpack execution internals, and callers that need to audit
the run (for eval or debugging) can inspect step statuses without parsing the report body.

**Why cap replans with `max_replans`?**
Without a cap, a plan that consistently triggers re-planning (e.g., a malformed goal that
always produces failing steps) will loop indefinitely. The cap converts an infinite loop
into a bounded failure with a clear error message.

---

## Extending to real LLM calls

1. Swap the mock in tests with `AnthropicLLM()`:
   ```python
   agent = ResearchAssistantAgent(llm=AnthropicLLM(model="claude-sonnet-5-5"))
   ```

2. Replace the stub tool functions with real implementations:
   - `search`: call Tavily or SerpAPI; return formatted result snippets.
   - `summarize`: call `claude-haiku-5-5` with a summarization system prompt.
   - `write_report`: call `claude-sonnet-5-5` or `claude-opus-5-5` for synthesis.

3. Add `ANTHROPIC_API_KEY` to `.env` (never commit it; `.env` is in `.gitignore`).

4. Run:
   ```bash
   export ANTHROPIC_API_KEY="$(cat .env | grep ANTHROPIC_API_KEY | cut -d= -f2)"
   python -m research_assistant.agent
   ```

5. For production: add structured logging, request tracing (correlation IDs on each LLM
   call), and a persistent store for `ExecutionReport` so partial runs can be resumed
   after crashes.
