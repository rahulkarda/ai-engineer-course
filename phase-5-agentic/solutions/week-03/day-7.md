# Solutions — Phase 5 · Week 03 · Day 7 — Week Project + Reflection

---

## Research Assistant — Complete Reference Implementation

### `research_assistant/models.py`

```python
"""Pydantic v2 models for the Research Assistant agent."""
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


class SummarizeStep(BaseModel):
    step_type: Literal["summarize"] = "summarize"
    step_id: str
    source_step_id: str          # step whose output to summarize
    max_words: int = 200
    depends_on: list[str] = Field(default_factory=list)


class WriteReportStep(BaseModel):
    step_type: Literal["write_report"] = "write_report"
    step_id: str
    title: str
    source_step_ids: list[str]   # steps whose summaries to combine
    depends_on: list[str] = Field(default_factory=list)


ResearchStep = Annotated[
    Union[SearchStep, SummarizeStep, WriteReportStep],
    Field(discriminator="step_type"),
]


# ---------------------------------------------------------------------------
# Plan & Results
# ---------------------------------------------------------------------------

class ResearchPlan(BaseModel):
    goal: str
    steps: list[ResearchStep]


class StepResult(BaseModel):
    step_id: str
    status: Literal["success", "failed", "skipped"]
    output: str | None = None
    error: str | None = None
    attempt: int = 1


# ---------------------------------------------------------------------------
# ResearchReport — the final deliverable
# ---------------------------------------------------------------------------

class Source(BaseModel):
    query: str
    raw_results: str
    summary: str


class ResearchReport(BaseModel):
    """
    The final output of the Research Assistant agent.

    Design decision: keeping sources as a list of Source objects (rather than
    a single merged string) lets downstream consumers re-rank, filter, or
    re-summarize individual sources without re-running the agent.
    """
    title: str
    goal: str
    sources: list[Source]
    body: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    replan_count: int = 0
    success: bool


class ExecutionReport(BaseModel):
    goal: str
    steps: list[StepResult]
    replan_count: int
    success: bool
    research_report: ResearchReport | None = None
```

---

### `research_assistant/planner.py`

```python
"""Planner: converts a research goal into a ResearchPlan."""
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
    Production LLM backend.
    Model: claude-sonnet-5-5 (or claude-haiku-5-5 for cheaper/faster planning).
    API key: read from environment — never hard-coded.
    """

    def __init__(self, model: str = "claude-sonnet-5-5") -> None:
        self._model = model
        self._api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set.")

    def generate(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        msg = client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


_PLAN_PROMPT = """
You are a research planning assistant.

GOAL: {goal}

Decompose this into steps using ONLY these step types: search, summarize, write_report.
Typical pattern: one search + one summarize per sub-topic, then a final write_report.

Output a JSON object:
{{
  "goal": "<goal>",
  "steps": [
    {{"step_type": "search",     "step_id": "s1", "query": "...",     "depends_on": []}},
    {{"step_type": "summarize",  "step_id": "s2", "source_step_id": "s1", "depends_on": ["s1"]}},
    {{"step_type": "write_report","step_id": "s3", "title": "...",
      "source_step_ids": ["s2"], "depends_on": ["s2"]}}
  ]
}}

Output ONLY valid JSON.
""".strip()


class ResearchPlanner:
    def __init__(self, llm: LLMInterface) -> None:
        self._llm = llm

    def decompose(self, goal: str) -> ResearchPlan:
        raw = self._llm.generate(_PLAN_PROMPT.format(goal=goal))
        return self._parse(raw)

    def replan(
        self,
        goal: str,
        completed_ids: list[str],
        failed_step_id: str,
        error: str,
    ) -> ResearchPlan:
        prompt = (
            f"Goal: {goal}\n"
            f"Completed: {completed_ids}\n"
            f"Failed: {failed_step_id} — {error}\n"
            "Produce a revised JSON research plan tail (same schema)."
        )
        logger.warning("ResearchPlanner.replan triggered by %r: %s", failed_step_id, error)
        raw = self._llm.generate(prompt)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> ResearchPlan:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner returned non-JSON: {exc}") from exc
        return ResearchPlan.model_validate(data)
```

---

### `research_assistant/tools.py`

```python
"""Research tools — deterministic stubs, swap for real APIs in production."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5) -> str:
    """Stub: returns a formatted fake result list."""
    logger.info("tool:search query=%r", query)
    return (
        f"Search results for '{query}':\n"
        f"1. Article A — covers {query} in depth.\n"
        f"2. Article B — overview of {query}.\n"
        f"3. Article C — recent developments in {query}.\n"
    )


def summarize(text: str, max_words: int = 200) -> str:
    """Stub: returns a truncated version of the text."""
    logger.info("tool:summarize len=%d", len(text))
    words = text.split()[:max_words]
    return " ".join(words) + ("..." if len(text.split()) > max_words else "")


def write_report(title: str, sections: list[str]) -> str:
    """Assemble sections into a markdown report."""
    logger.info("tool:write_report title=%r sections=%d", title, len(sections))
    body = f"# {title}\n\n" + "\n\n".join(sections)
    return body
```

---

### `research_assistant/executor.py`

```python
"""Executor for the research assistant — mirrors plan_execute_agent.executor."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Callable

from research_assistant.models import (
    ExecutionReport, ResearchPlan, ResearchReport,
    SearchStep, Source, StepResult, SummarizeStep, WriteReportStep,
)

logger = logging.getLogger(__name__)


def _topo_sort(plan: ResearchPlan) -> list[str]:
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
        raise ValueError("Dependency cycle detected in plan.")
    return order


ReplanHook = Callable[[str, list[StepResult], StepResult, str], ResearchPlan]


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
        final_report_body: str | None = None

        while True:
            order = _topo_sort(current_plan)
            step_map = {s.step_id: s for s in current_plan.steps}
            failed: StepResult | None = None

            for sid in order:
                step = step_map[sid]
                result = self._run_step(step, outputs, sources)
                completed.append(result)
                if result.status == "success" and result.output:
                    outputs[sid] = result.output
                    if isinstance(step, WriteReportStep):
                        final_report_body = result.output
                if result.status == "failed":
                    failed = result
                    break

            if failed is None:
                break

            if self._replan_hook is None or replan_count >= self._max_replans:
                report = ResearchReport(
                    title="Incomplete",
                    goal=plan.goal,
                    sources=sources,
                    body="Execution failed before report could be written.",
                    replan_count=replan_count,
                    success=False,
                )
                return ExecutionReport(
                    goal=plan.goal,
                    steps=completed,
                    replan_count=replan_count,
                    success=False,
                    research_report=report,
                )
            replan_count += 1
            logger.warning("Replan %d triggered by %r", replan_count, failed.step_id)
            current_plan = self._replan_hook(
                plan.goal, completed, failed, failed.error or "unknown"
            )

        research_report = ResearchReport(
            title=plan.goal,
            goal=plan.goal,
            sources=sources,
            body=final_report_body or "(no report step produced output)",
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
        try:
            if isinstance(step, SearchStep):
                output = self._search(step.query)
                return StepResult(step_id=step.step_id, status="success", output=output)

            if isinstance(step, SummarizeStep):
                source_text = outputs.get(step.source_step_id, "")
                output = self._summarize(source_text, step.max_words)
                # Record source for the final report.
                raw = outputs.get(step.source_step_id, "")
                sources.append(Source(
                    query=step.source_step_id,
                    raw_results=raw,
                    summary=output,
                ))
                return StepResult(step_id=step.step_id, status="success", output=output)

            if isinstance(step, WriteReportStep):
                sections = [outputs[sid] for sid in step.source_step_ids if sid in outputs]
                output = self._write_report(step.title, sections)
                return StepResult(step_id=step.step_id, status="success", output=output)

        except Exception as exc:
            logger.error("Step %r failed: %s", step.step_id, exc)
            return StepResult(step_id=step.step_id, status="failed", error=str(exc))

        return StepResult(step_id=step.step_id, status="skipped")
```

---

### `research_assistant/agent.py`

```python
"""ResearchAssistantAgent: orchestrates planner + executor."""
from __future__ import annotations

import logging

from research_assistant.executor import ResearchExecutor
from research_assistant.models import ExecutionReport, ResearchPlan, StepResult
from research_assistant.planner import LLMInterface, ResearchPlanner
from research_assistant.tools import search, summarize, write_report

logger = logging.getLogger(__name__)


class ResearchAssistantAgent:
    """
    High-level agent for the Research Assistant project.

    Usage:
        llm = AnthropicLLM()  # or a mock in tests
        agent = ResearchAssistantAgent(llm)
        report = agent.run("History of the Turing Award")
        print(report.research_report.body)
    """

    def __init__(
        self,
        llm: LLMInterface,
        max_replans: int = 3,
    ) -> None:
        self._planner = ResearchPlanner(llm)
        self._max_replans = max_replans

    def run(self, goal: str) -> ExecutionReport:
        logger.info("ResearchAssistantAgent: goal=%r", goal)
        plan: ResearchPlan = self._planner.decompose(goal)
        logger.info("Plan: %d steps", len(plan.steps))

        def replan_hook(
            g: str,
            completed: list[StepResult],
            failed: StepResult,
            error: str,
        ) -> ResearchPlan:
            completed_ids = [r.step_id for r in completed if r.status == "success"]
            return self._planner.replan(g, completed_ids, failed.step_id, error)

        executor = ResearchExecutor(
            search_fn=search,
            summarize_fn=summarize,
            write_report_fn=write_report,
            replan_hook=replan_hook,
            max_replans=self._max_replans,
        )
        report = executor.run(plan)
        logger.info(
            "ResearchAssistantAgent: done success=%s replan_count=%d",
            report.success, report.replan_count,
        )
        return report
```

---

### Sample run trace

```python
"""
Sample run trace for ResearchAssistantAgent.

Goal: "Research the history of the Turing Award and write a summary"

--- TRACE START ---

[INFO] ResearchAssistantAgent: goal='Research the history of the Turing Award...'
[INFO] ResearchPlanner: LLM generates plan with 5 steps:
        s1 SearchStep  query="Turing Award history founding"
        s2 SearchStep  query="Turing Award notable recipients"
        s3 SummarizeStep  source_step_id="s1" depends_on=["s1"]
        s4 SummarizeStep  source_step_id="s2" depends_on=["s2"]
        s5 WriteReportStep title="History of the Turing Award"
                           source_step_ids=["s3","s4"] depends_on=["s3","s4"]

[INFO] Topological order: [s1, s2, s3, s4, s5]

[INFO] tool:search query='Turing Award history founding'
       -> "Search results for 'Turing Award history founding': ..."
[INFO] Step s1: SUCCESS

[INFO] tool:search query='Turing Award notable recipients'
       -> "Search results for 'Turing Award notable recipients': ..."
[INFO] Step s2: SUCCESS

[INFO] tool:summarize len=142
       -> "Search results for 'Turing Award history founding': 1. Article A ..."
[INFO] Step s3: SUCCESS

[INFO] tool:summarize len=148
       -> "Search results for 'Turing Award notable recipients': 1. Article A ..."
[INFO] Step s4: SUCCESS

[INFO] tool:write_report title='History of the Turing Award' sections=2
       -> "# History of the Turing Award\n\n<s3 summary>\n\n<s4 summary>"
[INFO] Step s5: SUCCESS

[INFO] ResearchAssistantAgent: done success=True replan_count=0

--- ExecutionReport ---
goal         : "Research the history of the Turing Award and write a summary"
steps        : 5 (all success)
replan_count : 0
success      : True

--- ResearchReport ---
title   : "Research the history of the Turing Award..."
sources : [Source(query="s1", ...), Source(query="s2", ...)]
body    : "# History of the Turing Award\n\n..."
--- TRACE END ---
"""
```

---

### Tests

```python
# tests/test_research_agent.py
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from research_assistant.agent import ResearchAssistantAgent
from research_assistant.models import ExecutionReport, ResearchReport


def _plan_json(goal: str = "test") -> str:
    return json.dumps({
        "goal": goal,
        "steps": [
            {"step_type": "search",      "step_id": "s1",
             "query": "topic A",         "depends_on": []},
            {"step_type": "summarize",   "step_id": "s2",
             "source_step_id": "s1",     "depends_on": ["s1"]},
            {"step_type": "write_report","step_id": "s3",
             "title": "Report",          "source_step_ids": ["s2"],
             "depends_on": ["s2"]},
        ],
    })


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = _plan_json()
    return llm


@pytest.fixture()
def agent(mock_llm: MagicMock) -> ResearchAssistantAgent:
    return ResearchAssistantAgent(llm=mock_llm)


def test_agent_returns_execution_report(agent: ResearchAssistantAgent) -> None:
    report = agent.run("History of the Turing Award")
    assert isinstance(report, ExecutionReport)


def test_success_true(agent: ResearchAssistantAgent) -> None:
    report = agent.run("History of the Turing Award")
    assert report.success is True


def test_research_report_attached(agent: ResearchAssistantAgent) -> None:
    report = agent.run("History of the Turing Award")
    assert isinstance(report.research_report, ResearchReport)


def test_report_has_body(agent: ResearchAssistantAgent) -> None:
    report = agent.run("History of the Turing Award")
    assert report.research_report is not None
    assert len(report.research_report.body) > 0


def test_sources_populated(agent: ResearchAssistantAgent) -> None:
    report = agent.run("History of the Turing Award")
    assert report.research_report is not None
    assert len(report.research_report.sources) >= 1


def test_report_serialises(agent: ResearchAssistantAgent) -> None:
    report = agent.run("History")
    data = json.loads(report.model_dump_json())
    assert "success" in data
    assert "research_report" in data


def test_replan_on_summarize_failure(mock_llm: MagicMock) -> None:
    call_count = {"n": 0}

    def flaky_summarize(text: str, max_words: int = 200) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("summarizer unavailable")
        return f"summary of: {text[:50]}"

    replan_json = json.dumps({
        "goal": "test",
        "steps": [
            {"step_type": "summarize",   "step_id": "s2_retry",
             "source_step_id": "s1",     "depends_on": []},
            {"step_type": "write_report","step_id": "s3",
             "title": "Report",          "source_step_ids": ["s2_retry"],
             "depends_on": ["s2_retry"]},
        ],
    })
    mock_llm.generate.side_effect = [_plan_json(), replan_json]

    from research_assistant.executor import ResearchExecutor
    from research_assistant.planner import ResearchPlanner
    from research_assistant.tools import search, write_report

    planner = ResearchPlanner(llm=mock_llm)
    plan = planner.decompose("test")

    def replan_hook(goal, completed, failed, error):
        return planner.replan(goal, [r.step_id for r in completed], failed.step_id, error)

    executor = ResearchExecutor(
        search_fn=search,
        summarize_fn=flaky_summarize,
        write_report_fn=write_report,
        replan_hook=replan_hook,
        max_replans=3,
    )
    report = executor.run(plan)
    assert report.replan_count == 1
    assert report.success is True
```

---

## Reflection — Answers to the 5 Questions

### Q1: What did you find hardest about implementing the plan-and-execute pattern?

The hardest part is maintaining consistency between the plan and the execution context.
When re-planning fires mid-execution, the new plan tail must reference already-completed
step outputs by stable IDs, but those IDs now live in a completed-results list rather
than in the plan's step objects. The coupling between "what the executor knows" and
"what the planner needs to know" is subtle: passing too little context produces a naive
replan; passing too much bloats the prompt and risks hallucinated step references.
The cleanest resolution is a typed `ExecutionContext` model that the planner consumes,
rather than threading raw strings.

### Q2: How would you add parallelism to the executor without changing the planner?

Topological sort already produces the partial order; steps at the same depth level have
no mutual dependencies and can run concurrently. Swap the sequential `for sid in order`
loop for a level-by-level fan-out using `asyncio.gather` (or `concurrent.futures.ThreadPoolExecutor`
for blocking I/O tools). The planner output is unchanged; only the executor's scheduling
loop changes. Key invariant: a step's input is not consumed until all its `depends_on`
steps have written their outputs to the shared context dict — this is safe under a lock
or by using asyncio's cooperative model.

### Q3: What would break if the LLM returns a plan with a dependency cycle?

At runtime, `_topological_sort` would not include all step IDs in its output (Kahn's
algorithm stalls when in-degrees never reach zero). Without an explicit cycle check, the
executor would silently skip the cyclic steps, producing an incomplete report with no
error. The correct fix — already in the reference implementation — is to compare
`len(order)` against `len(plan.steps)` immediately after sorting and raise `ValueError`
at plan-validation time, before any tool is called.

### Q4: How does the circuit breaker complement re-planning, and when should you use one vs. the other?

Re-planning changes *what* the agent does; the circuit breaker changes *whether* the agent
attempts a call at all. Use the circuit breaker to protect a single external resource (search
API, summarizer endpoint) from repeated hammering when that resource is known-down. Use
re-planning when the failure suggests the *strategy* was wrong — wrong query, wrong tool,
bad decomposition. In practice: wrap each tool call in a circuit breaker; on `CircuitOpenError`
feed that signal to the error classifier, which routes it to "replan" or "abort" depending
on whether an alternative tool or strategy exists.

### Q5: If you were extending this to a multi-agent system, what interface contract would you define between agents?

Define a `SubAgentRequest` / `SubAgentResponse` Pydantic pair:

```python
class SubAgentRequest(BaseModel):
    request_id: str
    goal: str
    context: dict[str, str]   # relevant outputs from the parent's completed steps
    timeout_s: int = 60

class SubAgentResponse(BaseModel):
    request_id: str
    output: str
    success: bool
    replan_count: int
    error: str | None = None
```

The parent agent treats a `SubAgentStep` as a black box that accepts a `SubAgentRequest`
and returns a `SubAgentResponse`. Communication can be synchronous HTTP, async message
queue, or in-process depending on deployment topology — the interface stays the same.
This mirrors the A2A (Agent-to-Agent) protocol pattern: typed request/response envelopes
with stable IDs for correlation and a `success` flag the parent executor can route on.
