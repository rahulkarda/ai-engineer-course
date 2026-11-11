# Phase 5 · Week 03 · Day 7 — Week Project + Reflection

> Objective: Extend the plan-and-execute agent into a Research Assistant with structured output, simulated failure, and re-planning; then reflect on the full week.  ·  Time: ~3 hr project + 1 hr reflection  ·  Prereqs: [[phase-5-agentic/week-03/day-6]]

## Concept

Today closes the week by applying everything: ReAct loop (days 1–2), plan-and-execute pattern (days 3–4), failure recovery (day 5), and the full agent build (day 6). The project is intentionally realistic enough to surface design trade-offs without requiring live API calls.

### Project: Research Assistant

The Research Assistant accepts a research question, decomposes it into subtopics, searches each subtopic, gathers facts, synthesizes findings, and writes a structured `ResearchReport`. At least one simulated failure must be introduced and recovered via re-planning.

### What distinguishes this from day-6

| Day 6 agent | Research Assistant |
|-------------|-------------------|
| Generic goal | Research-specific domain model |
| Plain `ExecutionReport` | Typed `ResearchReport` with sections |
| Failure is incidental | Failure is explicitly simulated and recovered |
| Single pass | Two-phase: gather then synthesize |

### ResearchReport model

```
ResearchReport
├── question: str
├── subtopics: list[str]
├── findings: list[Finding]
│       └── Finding: subtopic, facts, source_steps
├── synthesis: str
├── execution_report: ExecutionReport
└── generated_at: datetime
```

The `ExecutionReport` is embedded so the caller can inspect replan count, step results, and abort status.

## How it works internally

**Phase 1 — gather**: The planner generates search steps for each subtopic (typically `len(subtopics) * 1` search steps). Results are collected in a `findings` accumulator keyed by subtopic.

**Phase 2 — synthesize**: A final summarize step receives all gathered facts concatenated. Output becomes `ResearchReport.synthesis`.

**Failure injection**: One search step raises a `ValueError` (structural error) on its first call. This triggers re-planning. The re-planner replaces the failed step with a simpler search. After recovery the gather phase completes normally.

**Report assembly**: After execution, the agent inspects `ExecutionReport.step_results`, extracts outputs keyed by step ID, and builds the `ResearchReport`.

## Worked example

```python
"""
day7_research_assistant.py — Research Assistant built on plan_execute_agent.

Requires plan_execute_agent package from day-6 on sys.path.
Run: python day7_research_assistant.py
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import anthropic
from pydantic import BaseModel, Field

# Import from the day-6 package (adjust path as needed)
from plan_execute_agent.executor import Executor, LLMClient
from plan_execute_agent.models import (
    ExecutionReport,
    Plan,
    Step,
    StepResult,
    StepStatus,
)
from plan_execute_agent.tools import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    subtopic: str
    facts: list[str]
    source_step_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    question: str
    subtopics: list[str]
    findings: list[Finding]
    synthesis: str
    execution_report: ExecutionReport
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        lines = [
            f"Question:  {self.question}",
            f"Subtopics: {', '.join(self.subtopics)}",
            f"Findings:  {len(self.findings)} subtopic(s) covered",
            f"Replan count: {self.execution_report.replan_count}",
            f"Aborted:   {self.execution_report.aborted}",
            "",
            "Synthesis:",
            self.synthesis,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Research planner
# ---------------------------------------------------------------------------

RESEARCH_PLAN_PROMPT = """\
You are a research planning agent.

Research question: {question}
Subtopics to investigate: {subtopics}

Create a research plan with these phases:
1. For each subtopic, one "search" step that queries facts about it.
2. One final "summarize" step that synthesizes all results.

Return a raw JSON array of step objects.
Each object must have: id, description, tool, inputs.
Tool parameter names:
  search:    query (str)
  summarize: text (str)

For the summarize step use inputs: {{"text": "{{gathered_facts}}"}}
(The executor will substitute actual gathered text at runtime.)

Output raw JSON array only.
"""


class ResearchPlanner:
    """Generates a research Plan from a question and subtopics list."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def plan(self, question: str, subtopics: list[str]) -> Plan:
        subtopics_str = ", ".join(subtopics)
        prompt = RESEARCH_PLAN_PROMPT.format(
            question=question,
            subtopics=subtopics_str,
        )
        raw = self._llm.complete(prompt)
        log.debug("research planner raw response: %s", raw[:400])
        steps_data: list[dict] = json.loads(raw)
        steps = [Step(**d) for d in steps_data]
        return Plan(goal=question, steps=steps)


# ---------------------------------------------------------------------------
# Research tool registry
# ---------------------------------------------------------------------------

# Global call counter for failure injection
_search_call_count: dict[str, int] = {}


def _injected_search(query: str) -> str:
    """Search tool that fails once for queries containing 'concurrency'."""
    _search_call_count[query] = _search_call_count.get(query, 0) + 1
    if "concurrency" in query.lower() and _search_call_count[query] == 1:
        raise ValueError(
            f"schema mismatch: search service returned null for query {query!r}"
        )
    return (
        f"Facts about '{query}': "
        "[Fact A: relevant finding] "
        "[Fact B: supporting evidence] "
        "[Fact C: counterpoint]"
    )


def _mock_summarize(text: str) -> str:
    word_count = len(text.split())
    return (
        f"Synthesis of {word_count} words: "
        "The research reveals three key themes — "
        "performance improvements, simplified syntax, and improved error handling. "
        "Async patterns in Python have matured significantly. "
        "Concurrency models show measurable benefits in I/O-bound workloads."
    )


def build_research_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("search", _injected_search)
    registry.register("summarize", _mock_summarize)
    return registry


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------

SUBTOPIC_DECOMPOSITION_PROMPT = """\
You are a research assistant. Break the following question into 3–4 specific subtopics.

Question: {question}

Return a raw JSON array of strings (subtopic names only). No markdown.
"""


class ResearchAssistant:
    """
    Full Research Assistant.

    1. Decomposes question into subtopics.
    2. Plans search + synthesize steps.
    3. Executes with failure injection and re-planning.
    4. Assembles ResearchReport.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: ToolRegistry | None = None,
        max_replans: int = 3,
    ) -> None:
        self._llm = llm_client
        self._registry = registry or build_research_registry()
        self._executor = Executor(
            llm_client=llm_client,
            max_replans=max_replans,
            max_retries=0,
        )
        self._planner = ResearchPlanner(llm_client)

    def research(self, question: str) -> ResearchReport:
        log.info("research question: %r", question)

        # Step 1: decompose into subtopics
        subtopics = self._decompose(question)
        log.info("subtopics: %s", subtopics)

        # Step 2: generate plan
        plan = self._planner.plan(question, subtopics)
        log.info("plan generated with %d steps", len(plan.steps))

        # Step 3: execute
        execution_report = self._executor.run(plan, self._registry)

        # Step 4: assemble report
        report = self._assemble_report(question, subtopics, plan, execution_report)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decompose(self, question: str) -> list[str]:
        prompt = SUBTOPIC_DECOMPOSITION_PROMPT.format(question=question)
        raw = self._llm.complete(prompt)
        subtopics: list[str] = json.loads(raw)
        return subtopics

    def _assemble_report(
        self,
        question: str,
        subtopics: list[str],
        plan: Plan,
        execution_report: ExecutionReport,
    ) -> ResearchReport:
        # Map step_id → output for successful steps
        output_by_id: dict[str, str] = {
            r.step_id: (r.output or "")
            for r in execution_report.step_results
            if r.status == StepStatus.SUCCESS
        }

        # Build findings by matching search step descriptions to subtopics
        findings: list[Finding] = []
        synthesis = ""

        for step in plan.steps:
            if step.tool == "summarize":
                synthesis = output_by_id.get(step.id, "[synthesis not completed]")
                continue
            if step.tool == "search":
                output = output_by_id.get(step.id, "")
                # Match subtopic heuristically by checking description
                matched = next(
                    (s for s in subtopics if s.lower() in step.description.lower()),
                    step.description,
                )
                facts = [
                    line.strip()
                    for line in output.split("[")
                    if line.strip() and "Fact" in line
                ]
                findings.append(
                    Finding(
                        subtopic=matched,
                        facts=facts or [output[:120]] if output else [],
                        source_step_ids=[step.id],
                    )
                )

        # Also capture replanned steps
        for result in execution_report.step_results:
            if result.status == StepStatus.SUCCESS and result.step_id not in [
                s.id for s in plan.steps
            ]:
                findings.append(
                    Finding(
                        subtopic=f"recovered (step {result.step_id})",
                        facts=[result.output[:120]] if result.output else [],
                        source_step_ids=[result.step_id],
                    )
                )

        return ResearchReport(
            question=question,
            subtopics=subtopics,
            findings=findings,
            synthesis=synthesis,
            execution_report=execution_report,
        )


# ---------------------------------------------------------------------------
# Stub LLM for demo (no real API key needed)
# ---------------------------------------------------------------------------


class StubResearchLLM:
    """Returns deterministic stub responses for each prompt type."""

    def complete(self, prompt: str) -> str:
        if "Break the following question into" in prompt:
            return json.dumps(
                [
                    "async/await syntax",
                    "concurrency models",
                    "error handling in async code",
                ]
            )

        if "Create a research plan" in prompt:
            return json.dumps(
                [
                    {
                        "id": "search-async-syntax",
                        "description": "search for async/await syntax fundamentals",
                        "tool": "search",
                        "inputs": {"query": "Python async await syntax"},
                    },
                    {
                        "id": "search-concurrency",
                        "description": "search for concurrency models in Python",
                        "tool": "search",
                        "inputs": {"query": "Python concurrency models asyncio"},
                    },
                    {
                        "id": "search-error-handling",
                        "description": "search for error handling in async code",
                        "tool": "search",
                        "inputs": {"query": "Python async error handling patterns"},
                    },
                    {
                        "id": "summarize-findings",
                        "description": "synthesize all gathered facts",
                        "tool": "summarize",
                        "inputs": {"text": "gathered_facts"},
                    },
                ]
            )

        # Re-plan prompt: return a simplified replacement search step
        return json.dumps(
            [
                {
                    "id": "search-concurrency-retry",
                    "description": "search for concurrency models simplified",
                    "tool": "search",
                    "inputs": {"query": "Python async concurrency"},
                }
            ]
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    llm = StubResearchLLM()
    assistant = ResearchAssistant(llm_client=llm, max_replans=3)

    question = "What are the best practices for async programming in Python?"
    report = assistant.research(question)

    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(report.summary())
    print()
    print(f"Findings detail:")
    for finding in report.findings:
        print(f"  [{finding.subtopic}]")
        for fact in finding.facts:
            print(f"    - {fact}")
    print()
    print(f"Execution stats:")
    print(f"  Total steps attempted: {report.execution_report.total_steps_attempted}")
    print(f"  Replan count:          {report.execution_report.replan_count}")
    print(f"  Successful steps:      {len(report.execution_report.successful_results())}")
    print(f"  Aborted:               {report.execution_report.aborted}")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python day7_research_assistant.py
```

Expected output (approximate):
```
============================================================
RESEARCH REPORT
============================================================
Question:  What are the best practices for async programming in Python?
Subtopics: async/await syntax, concurrency models, error handling in async code
Findings:  3 subtopic(s) covered
Replan count: 1
Aborted:   False

Synthesis:
Synthesis of N words: The research reveals three key themes ...

Findings detail:
  [async/await syntax]
    - Fact A: relevant finding]
    ...
  [recovered (step search-concurrency-retry)]
    - Facts about 'Python async concurrency': ...
```

### Test suite for the Research Assistant

```python
"""
test_research_assistant.py — pytest tests for ResearchAssistant.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from plan_execute_agent.models import StepStatus
from plan_execute_agent.tools import ToolRegistry


# Import the research assistant from day7 module
# (adjust import path to match your directory layout)
# from day7_research_assistant import ResearchAssistant, StubResearchLLM, Finding


def _stub_llm_with_responses(responses: list[str]) -> MagicMock:
    llm = MagicMock()
    llm.complete.side_effect = responses
    return llm


def _make_registry(fail_query: str | None = None) -> ToolRegistry:
    call_counts: dict[str, int] = {}

    def search(query: str) -> str:
        call_counts[query] = call_counts.get(query, 0) + 1
        if fail_query and fail_query in query and call_counts[query] == 1:
            raise ValueError("simulated schema mismatch")
        return f"Results for {query!r}: [Fact A] [Fact B]"

    def summarize(text: str) -> str:
        return f"Synthesis: {text[:60]}"

    registry = ToolRegistry()
    registry.register("search", search)
    registry.register("summarize", summarize)
    return registry


# Note: these tests require day7_research_assistant.py on sys.path.
# Activate by importing at top of file when testing locally.

def test_stub_llm_returns_subtopics() -> None:
    """Verify StubResearchLLM returns parseable subtopics JSON."""
    # from day7_research_assistant import StubResearchLLM
    # llm = StubResearchLLM()
    # raw = llm.complete("Break the following question into subtopics: Question: X")
    # subtopics = json.loads(raw)
    # assert isinstance(subtopics, list)
    # assert len(subtopics) >= 2
    pass  # Remove pass and uncomment above when module is available


def test_research_report_contains_question() -> None:
    """ResearchReport.question must match the input question."""
    # from day7_research_assistant import ResearchAssistant, StubResearchLLM
    # assistant = ResearchAssistant(llm_client=StubResearchLLM())
    # report = assistant.research("What is Python asyncio?")
    # assert report.question == "What is Python asyncio?"
    pass


def test_replan_fires_on_first_concurrency_search() -> None:
    """The injected failure on 'concurrency' queries triggers exactly one replan."""
    # from day7_research_assistant import ResearchAssistant, StubResearchLLM
    # assistant = ResearchAssistant(llm_client=StubResearchLLM())
    # report = assistant.research("What are async best practices?")
    # assert report.execution_report.replan_count == 1
    pass


def test_synthesis_present_in_report() -> None:
    """ResearchReport.synthesis must be a non-empty string."""
    # from day7_research_assistant import ResearchAssistant, StubResearchLLM
    # assistant = ResearchAssistant(llm_client=StubResearchLLM())
    # report = assistant.research("What is Python asyncio?")
    # assert report.synthesis
    pass
```

## Exercises

1. Add a `confidence: float` field (0.0–1.0) to `Finding`. Populate it based on the number of distinct facts found: `len(facts) / 5.0` capped at 1.0. Write a test that verifies `confidence` is 0.0 when facts is empty.

2. Add a `ResearchCache` class that stores `ResearchReport` by question (case-insensitive). If a question was already researched in the same session, return the cached report without re-executing. Write a test that verifies the LLM is called only once for two identical questions.

3. Extend `StubResearchLLM` to log each prompt category (subtopic decomposition, planning, replanning) using the `logging` module. Write a test that uses `caplog` to verify all three log categories appear after a full research run with a simulated failure.

(Solutions: ../solutions/week-03/day-7.md)

## Reflection

Answer these five questions in writing (use a journal, comment block, or separate notes file):

1. **Planning vs. execution separation**: What would break if the planner and executor were merged into a single function? Give a specific testing or maintenance scenario.

2. **Failure granularity**: The current classifier treats all non-transient, non-fatal errors as "structural" and triggers re-planning. When would this over-trigger re-planning? Sketch a more precise classifier.

3. **Token cost of re-planning**: Each re-plan call sends the full completed-steps context to the LLM. At what plan size does this become expensive? What would you do differently for a 50-step plan?

4. **Stateful tools**: All tools in this week's examples were stateless (same input always produces same output). What changes in your executor logic when a tool is stateful — e.g., a file tool that appends rather than overwrites?

5. **Testing LLM behaviour**: The week's tests mock all LLM calls. What would you need to add to run integration tests against a real Claude endpoint without making your CI slow or expensive? (Think: test categories, caching, cost budgets.)

## Week self-assessment

Rate yourself 1–5 on each item:

- [ ] I can explain the ReAct loop (Reason + Act cycles, observation feedback)
- [ ] I can implement a `ReActAgent` with a tool registry from scratch
- [ ] I can describe plan-and-execute and explain when it is better than ReAct
- [ ] I can implement `LLMPlanner.plan()` including JSON parsing and Pydantic validation
- [ ] I can implement `Executor.run()` with retry, re-plan, and circuit-breaker logic
- [ ] I can write the re-planning prompt with all four required context pieces
- [ ] I can classify errors as transient, structural, or fatal with concrete examples
- [ ] I can build a `ToolRegistry` and inject it for both real and test environments
- [ ] I can mock an LLM client in pytest using `unittest.mock.patch`
- [ ] I can design a domain-specific Pydantic report model and assemble it from step results

Score 40–50: ready for week 4 (multi-agent systems).
Score 30–39: review days 3–5 before moving on.
Score below 30: revisit the full week; the next week builds directly on these patterns.

## Week 3 links

- Week assessment: [[phase-5-agentic/week-03/week-assessment]]
- Next week preview: [[phase-5-agentic/week-04/day-1]] — Multi-Agent Systems

## Supplementary links

- Anthropic model IDs: https://docs.anthropic.com/en/docs/about-claude/models
- Pydantic v2 validators: https://docs.pydantic.dev/latest/concepts/validators/
- Python asyncio docs: https://docs.python.org/3/library/asyncio.html
- Plan-and-execute paper (ReWOO): https://arxiv.org/abs/2305.18323
- LangGraph plan-and-execute example: https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/
- pytest `caplog` fixture: https://docs.pytest.org/en/stable/how-to/logging.html
