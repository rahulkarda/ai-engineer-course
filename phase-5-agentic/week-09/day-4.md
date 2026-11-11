# Phase 5 · Week 09 · Day 4 — Task Delegation Across Agents

> Objective: Implement a complete task delegation pipeline with routing, execution, error handling, and result aggregation.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

**Task delegation** is the act of an orchestrator agent decomposing a goal and assigning sub-tasks to specialist agents. It's more than forwarding a message — it involves:

1. **Decomposition** — splitting a complex goal into sub-tasks each specialist can handle
2. **Routing** — selecting the right agent per sub-task (from day-3's registry)
3. **Execution** — sending tasks, handling the lifecycle states
4. **Aggregation** — merging results from multiple agents into a coherent output
5. **Error recovery** — retrying, falling back, or gracefully failing when a delegate fails

**Delegation patterns:**

| Pattern | When to use |
|---|---|
| Sequential | Each sub-task depends on the previous result |
| Parallel | Sub-tasks are independent; aggregate when all done |
| Fan-out / fan-in | Broadcast same task to multiple agents; pick best/merge |
| Hierarchical | Orchestrator delegates to sub-orchestrators that delegate further |

**Trust in delegation.** When Agent A delegates to Agent B, it must decide: does B receive the user's full context, or only what's needed for the sub-task? Over-sharing leaks PII; under-sharing causes B to ask follow-up questions. A good delegation layer passes a scoped context envelope.

## How it works internally

The orchestrator's delegation loop:

```
for each sub_task in decomposed_tasks:
    agent = registry.route(sub_task.required_skill)
    task_handle = agent.submit(sub_task)
    while task_handle.status not in TERMINAL_STATES:
        if task_handle.status == INPUT_REQUIRED:
            clarification = self.generate_clarification(task_handle)
            task_handle.send(clarification)
        else:
            task_handle = agent.poll(task_handle.task_id)
    results.append(task_handle.result)
return self.aggregate(results)
```

**Error handling strategy:**
- `failed` status → check if retryable (transient error) or permanent
- Transient: exponential backoff, max 3 retries
- Permanent: fall back to alternative agent from registry, or surface error to user
- Timeout: treat as `failed` after deadline

**Parallel delegation** uses `asyncio.gather` or a thread pool. Results arrive out of order; you need a correlation map (`task_id → sub_task_index`) to reassemble.

## Worked example

```python
"""
day_4_task_delegation.py
Complete delegation pipeline: decompose, route, execute (parallel + sequential),
aggregate, and recover from errors.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Domain types
# ──────────────────────────────────────────────

class TaskStatus(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class SubTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    required_skill: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)   # task_ids of prerequisites


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: str = ""
    error: str = ""


# ──────────────────────────────────────────────
# Mock specialist agents
# ──────────────────────────────────────────────

class SpecialistAgent:
    """Simulates a remote A2A agent. Handler determines behaviour."""

    def __init__(
        self,
        name: str,
        skill: str,
        handler: Callable[[dict[str, Any]], str],
        fail_once: bool = False,
    ) -> None:
        self.name = name
        self.skill = skill
        self._handler = handler
        self._fail_once = fail_once
        self._call_count = 0

    async def execute(self, payload: dict[str, Any]) -> TaskResult:
        task_id = str(uuid.uuid4())
        self._call_count += 1
        await asyncio.sleep(0.05)   # Simulate network latency

        if self._fail_once and self._call_count == 1:
            return TaskResult(task_id=task_id, status=TaskStatus.FAILED, error="Transient error")

        try:
            output = self._handler(payload)
            return TaskResult(task_id=task_id, status=TaskStatus.COMPLETED, output=output)
        except Exception as exc:
            return TaskResult(task_id=task_id, status=TaskStatus.FAILED, error=str(exc))


# ──────────────────────────────────────────────
# Simple registry
# ──────────────────────────────────────────────

class SimpleRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, list[SpecialistAgent]] = {}

    def register(self, agent: SpecialistAgent) -> None:
        self._agents.setdefault(agent.skill, []).append(agent)

    def get(self, skill: str) -> SpecialistAgent | None:
        candidates = self._agents.get(skill, [])
        return candidates[0] if candidates else None

    def get_fallback(self, skill: str, exclude: str) -> SpecialistAgent | None:
        candidates = [a for a in self._agents.get(skill, []) if a.name != exclude]
        return candidates[0] if candidates else None


# ──────────────────────────────────────────────
# Delegation engine
# ──────────────────────────────────────────────

class DelegationEngine:
    MAX_RETRIES = 3
    RETRY_DELAY = 0.1   # seconds (tiny for demo; use exponential in prod)

    def __init__(self, registry: SimpleRegistry) -> None:
        self._registry = registry

    async def execute_with_retry(self, sub_task: SubTask) -> TaskResult:
        """Execute a single sub-task with retry on transient failure."""
        agent = self._registry.get(sub_task.required_skill)
        if agent is None:
            return TaskResult(
                task_id=sub_task.task_id,
                status=TaskStatus.FAILED,
                error=f"No agent for skill '{sub_task.required_skill}'",
            )

        for attempt in range(1, self.MAX_RETRIES + 1):
            result = await agent.execute(sub_task.payload)
            if result.status == TaskStatus.COMPLETED:
                logger.info("[Delegate] '%s' completed on attempt %d", agent.name, attempt)
                return result

            if attempt < self.MAX_RETRIES:
                logger.warning(
                    "[Delegate] '%s' failed (attempt %d/%d): %s — retrying in %.1fs",
                    agent.name, attempt, self.MAX_RETRIES, result.error, self.RETRY_DELAY,
                )
                await asyncio.sleep(self.RETRY_DELAY)
            else:
                # Try fallback agent
                fallback = self._registry.get_fallback(sub_task.required_skill, agent.name)
                if fallback:
                    logger.info("[Delegate] Falling back to '%s'", fallback.name)
                    return await fallback.execute(sub_task.payload)
                logger.error("[Delegate] All agents failed for skill '%s'", sub_task.required_skill)
                return result

        return result   # unreachable, satisfies type checker

    async def execute_parallel(self, sub_tasks: list[SubTask]) -> list[TaskResult]:
        """Execute independent sub-tasks concurrently."""
        logger.info("[Engine] Executing %d sub-tasks in parallel", len(sub_tasks))
        return list(await asyncio.gather(*[self.execute_with_retry(t) for t in sub_tasks]))

    async def execute_sequential(
        self,
        sub_tasks: list[SubTask],
        results_so_far: dict[str, TaskResult] | None = None,
    ) -> list[TaskResult]:
        """Execute sub-tasks sequentially; inject prior results into payload."""
        context = results_so_far or {}
        ordered_results: list[TaskResult] = []

        for sub_task in sub_tasks:
            # Inject dependency outputs into payload
            for dep_id in sub_task.depends_on:
                if dep_id in context and context[dep_id].status == TaskStatus.COMPLETED:
                    sub_task.payload[f"dep_{dep_id[:8]}"] = context[dep_id].output

            result = await self.execute_with_retry(sub_task)
            ordered_results.append(result)
            context[sub_task.task_id] = result

            if result.status != TaskStatus.COMPLETED:
                logger.error("[Engine] Sequential chain broken at task %s", sub_task.task_id[:8])
                break   # Don't continue if a dependency fails

        return ordered_results


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

class ResearchOrchestrator:
    """
    Goal: given a company name, produce a structured research brief.
    Delegation plan:
      Phase 1 (parallel): news-search, financials-fetch
      Phase 2 (sequential, depends on phase 1): sentiment-analysis → brief-generation
    """

    def __init__(self, registry: SimpleRegistry) -> None:
        self._engine = DelegationEngine(registry)

    async def run(self, company: str) -> str:
        logger.info("[Orchestrator] Starting research for '%s'", company)

        # Phase 1: parallel information gathering
        phase1_tasks = [
            SubTask(required_skill="news-search", payload={"query": f"{company} latest news"}),
            SubTask(required_skill="financials-fetch", payload={"ticker": company}),
        ]
        phase1_results = await self._engine.execute_parallel(phase1_tasks)

        # Check failures
        failed = [r for r in phase1_results if r.status != TaskStatus.COMPLETED]
        if failed:
            logger.warning("[Orchestrator] %d phase-1 tasks failed", len(failed))

        # Collect successful outputs
        context = {r.task_id: r for r in phase1_results}
        combined = " | ".join(r.output for r in phase1_results if r.status == TaskStatus.COMPLETED)

        # Phase 2: sequential — sentiment needs raw data; brief needs sentiment
        sentiment_task = SubTask(
            required_skill="sentiment-analysis",
            payload={"text": combined},
        )
        brief_task = SubTask(
            required_skill="brief-generation",
            payload={"company": company},
            depends_on=[sentiment_task.task_id],
        )
        phase2_results = await self._engine.execute_sequential(
            [sentiment_task, brief_task], context
        )

        final = next(
            (r.output for r in phase2_results if r.status == TaskStatus.COMPLETED and "brief" in r.task_id or True),
            "Research incomplete",
        )
        # Grab the last completed result as the final brief
        completed = [r for r in phase2_results if r.status == TaskStatus.COMPLETED]
        return completed[-1].output if completed else "Research incomplete"


# ──────────────────────────────────────────────
# Wire up and run
# ──────────────────────────────────────────────

def build_registry() -> SimpleRegistry:
    registry = SimpleRegistry()

    registry.register(SpecialistAgent(
        name="news-agent",
        skill="news-search",
        handler=lambda p: f"Top news for '{p['query']}': Market expansion, new product launch.",
        fail_once=True,   # First call fails to demo retry
    ))
    registry.register(SpecialistAgent(
        name="news-agent-backup",
        skill="news-search",
        handler=lambda p: f"Backup news for '{p['query']}': Quarterly earnings beat expectations.",
    ))
    registry.register(SpecialistAgent(
        name="financials-agent",
        skill="financials-fetch",
        handler=lambda p: f"Financials for {p['ticker']}: Revenue +12%, Profit margin 18%.",
    ))
    registry.register(SpecialistAgent(
        name="sentiment-agent",
        skill="sentiment-analysis",
        handler=lambda p: f"Sentiment: POSITIVE (confidence 0.87). Key signals: growth, expansion.",
    ))
    registry.register(SpecialistAgent(
        name="brief-agent",
        skill="brief-generation",
        handler=lambda p: (
            f"Research Brief — {p['company']}: "
            f"Strong fundamentals, positive market sentiment. "
            f"Recommendation: Monitor for entry opportunity."
        ),
    ))
    return registry


async def main() -> None:
    registry = build_registry()
    orchestrator = ResearchOrchestrator(registry)
    result = await orchestrator.run("ACME Corp")
    logger.info("[Orchestrator] Final brief: %s", result)


if __name__ == "__main__":
    asyncio.run(main())
```

## Exercises

1. Add a `timeout_seconds` parameter to `execute_with_retry`. Use `asyncio.wait_for` to cancel tasks that exceed the timeout. Return a `FAILED` result with error `"timeout"`. Write a test that confirms timeout is triggered.

2. Implement a **fan-out** strategy: send the same `news-search` task to all registered agents for that skill, run them in parallel, and return the result from whichever completes first (cancel the others). Use `asyncio.wait` with `FIRST_COMPLETED`.

3. Add `priority: int` (1=low, 5=high) to `SubTask`. Modify `execute_parallel` to process high-priority tasks first using a priority queue (`asyncio.PriorityQueue`).

4. Implement result aggregation for parallel phase output: instead of concatenating strings, produce a `ResearchContext` Pydantic model with typed fields (`news: str`, `financials: str`, `sentiment: str`). Validate that all required fields are populated before proceeding to phase 2.

(Solutions: ../solutions/week-09/day-4.md)

## Self-check
- [ ] I can explain four delegation patterns and when each applies
- [ ] I can implement retry-with-fallback for async agent calls
- [ ] I understand how `depends_on` enables sequential dependency chains
- [ ] I can build a two-phase (parallel then sequential) orchestration pipeline

## Supplementary links
- [asyncio.gather and asyncio.wait docs](https://docs.python.org/3/library/asyncio-task.html)
- [A2A task lifecycle specification](https://a2aproject.github.io/A2A/specification/#task)
- [Circuit breaker pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
