# Phase 5 · Week 06 · Day 2 — Delegation Patterns: Orchestrator-Worker

> Objective: Build an orchestrator that decomposes tasks, delegates to specialist workers in parallel, and aggregates results  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

The **orchestrator-worker pattern** separates *planning* from *execution*. The orchestrator owns the goal and knows how to break it into pieces. Workers own a single specialty and know nothing about the wider goal.

```
User query
    │
    ▼
Orchestrator ── decompose ──► [Task, Task, Task]
    │                              │
    │         ┌────────────────────┤
    │         ▼         ▼         ▼
    │      Worker    Worker    Worker
    │     (search) (summarize)(calculate)
    │         │         │         │
    └──── aggregate ◄───┴─────────┘
    │
    ▼
Final response
```

**When to use parallel vs sequential dispatch:**

| Mode | Use when |
|---|---|
| Parallel | Sub-tasks are independent — order doesn't matter, no task reads another's output |
| Sequential | Task N depends on Task N-1's result (pipeline); or rate limits prevent concurrency |
| Mixed | Some tasks independent, some form a chain — fan out what you can, chain what you must |

## How it works internally

**Decomposition via LLM tool_use:**  
The orchestrator sends the original query to the LLM with a `decompose_query` tool. The model returns a list of tasks, each with a `specialty` field that maps to a registered worker.

**`asyncio.gather()` for parallel dispatch:**  
Each worker's `execute()` method is a coroutine. `asyncio.gather(*coroutines)` runs them concurrently on the event loop, giving wall-clock time close to the slowest worker rather than the sum of all workers. Order of results matches order of input.

**Worker isolation:**  
Workers receive only their own task — they never see the full decomposed plan. This keeps workers simple, testable, and replaceable. The orchestrator is the only component that holds the full picture.

**Aggregation:**  
After all workers finish, the orchestrator calls the LLM a second time (or uses a simple template) to synthesise worker outputs into a coherent response. In production you may add a "reducer" step that checks for contradictions between workers.

**Error handling:**  
`asyncio.gather()` with `return_exceptions=True` prevents a single failed worker from cancelling the rest. The aggregation step must check for `Exception` instances in the results and either retry, skip, or escalate.

## Worked example

```python
"""
Orchestrator-worker pattern with async parallel dispatch.

Workers: search | summarize | calculate
Run: python day-2.py
Requires: ANTHROPIC_API_KEY in environment
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

VALID_SPECIALTIES = {"search", "summarize", "calculate"}


class Task(BaseModel):
    """A single sub-task produced by the orchestrator's decomposition step."""
    task_id: str = Field(..., description="Unique identifier, e.g. 'task_1'")
    specialty: str = Field(..., description="Worker specialty to handle this task")
    description: str = Field(..., description="Human-readable description of what to do")
    input_data: str = Field(..., description="The specific input the worker should act on")


class DecomposedPlan(BaseModel):
    tasks: list[Task]
    reasoning: str = Field(..., description="Why the query was split this way")


class WorkerResult(BaseModel):
    task_id: str
    specialty: str
    output: str
    duration_ms: float
    success: bool


class OrchestratorResult(BaseModel):
    original_query: str
    plan: DecomposedPlan
    worker_results: list[WorkerResult]
    final_response: str
    total_duration_ms: float


# ---------------------------------------------------------------------------
# LLM tool schemas
# ---------------------------------------------------------------------------

DECOMPOSE_TOOL: dict[str, Any] = {
    "name": "decompose_query",
    "description": (
        "Break a user query into sub-tasks. Each task must map to exactly one specialty: "
        "'search' (retrieve/look up information), "
        "'summarize' (condense or reformat text), "
        "'calculate' (numeric computation or data analysis)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "specialty": {
                            "type": "string",
                            "enum": list(VALID_SPECIALTIES),
                        },
                        "description": {"type": "string"},
                        "input_data": {"type": "string"},
                    },
                    "required": ["task_id", "specialty", "description", "input_data"],
                },
            },
            "reasoning": {"type": "string"},
        },
        "required": ["tasks", "reasoning"],
    },
}

DECOMPOSE_SYSTEM = """\
You are a task planner. Decompose user queries into the minimum number of sub-tasks needed.
Map each sub-task to one of: search, summarize, calculate.
Use only the specialties that are genuinely needed — do not create unnecessary tasks.\
"""


# ---------------------------------------------------------------------------
# Worker agents
# ---------------------------------------------------------------------------

class Worker:
    """
    A specialist agent that handles one category of task.

    In production each worker would call a different tool (web search,
    document store, calculator service). Here we simulate with the LLM
    so the example is self-contained and runnable.
    """

    SYSTEM_PROMPTS: dict[str, str] = {
        "search": (
            "You are a search specialist. Given a query, return a concise factual answer "
            "as if retrieved from a knowledge base. Be specific and cite plausible sources."
        ),
        "summarize": (
            "You are a summarization specialist. Condense the provided text into the key points. "
            "Use bullet points. Be concise."
        ),
        "calculate": (
            "You are a calculation specialist. Perform the requested numeric analysis. "
            "Show your working and give a clear numeric result."
        ),
    }

    def __init__(self, specialty: str, client: anthropic.AsyncAnthropic) -> None:
        if specialty not in VALID_SPECIALTIES:
            raise ValueError(f"Unknown specialty '{specialty}'")
        self.specialty = specialty
        self._client = client

    async def execute(self, task: Task) -> WorkerResult:
        """Run the task and return a WorkerResult. Never raises — errors captured in result."""
        start = time.monotonic()
        logger.info("Worker[%s] starting task=%s", self.specialty, task.task_id)
        try:
            response = await self._client.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=512,
                system=self.SYSTEM_PROMPTS[self.specialty],
                messages=[{"role": "user", "content": task.input_data}],
            )
            output = response.content[0].text
            success = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Worker[%s] task=%s failed: %s", self.specialty, task.task_id, exc)
            output = f"[error] {exc}"
            success = False

        duration_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Worker[%s] task=%s done in %.0fms success=%s",
            self.specialty, task.task_id, duration_ms, success,
        )
        return WorkerResult(
            task_id=task.task_id,
            specialty=self.specialty,
            output=output,
            duration_ms=duration_ms,
            success=success,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Decomposes a query, dispatches to workers in parallel, and aggregates results.

    Workers are registered by specialty. Unknown specialties in the decomposed plan
    are skipped with a warning.
    """

    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client
        self._workers: dict[str, Worker] = {}

    def register_worker(self, worker: Worker) -> "Orchestrator":
        """Register a worker. Chainable."""
        self._workers[worker.specialty] = worker
        logger.debug("Registered worker specialty=%s", worker.specialty)
        return self

    async def _decompose(self, query: str) -> DecomposedPlan:
        """Call LLM to break the query into sub-tasks."""
        logger.info("Decomposing query: %r", query)
        response = await self._client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=512,
            system=DECOMPOSE_SYSTEM,
            tools=[DECOMPOSE_TOOL],  # type: ignore[list-item]
            tool_choice={"type": "tool", "name": "decompose_query"},
            messages=[{"role": "user", "content": query}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        plan = DecomposedPlan.model_validate(tool_block.input)
        logger.info("Plan: %d tasks — %s", len(plan.tasks), plan.reasoning)
        return plan

    async def _dispatch_parallel(self, tasks: list[Task]) -> list[WorkerResult]:
        """
        Dispatch all tasks to their registered workers concurrently.

        return_exceptions=True ensures one failing worker does not cancel others.
        """
        coroutines = []
        task_list: list[Task] = []

        for task in tasks:
            worker = self._workers.get(task.specialty)
            if worker is None:
                logger.warning(
                    "No worker for specialty=%s (task=%s) — skipping",
                    task.specialty, task.task_id,
                )
                continue
            coroutines.append(worker.execute(task))
            task_list.append(task)

        raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

        results: list[WorkerResult] = []
        for task, raw in zip(task_list, raw_results):
            if isinstance(raw, Exception):
                results.append(
                    WorkerResult(
                        task_id=task.task_id,
                        specialty=task.specialty,
                        output=f"[unhandled error] {raw}",
                        duration_ms=0.0,
                        success=False,
                    )
                )
            else:
                results.append(raw)
        return results

    async def _aggregate(self, query: str, results: list[WorkerResult]) -> str:
        """Synthesise worker outputs into a final response."""
        worker_outputs = "\n\n".join(
            f"[{r.specialty} / {r.task_id}]\n{r.output}" for r in results
        )
        prompt = (
            f"Original query: {query}\n\n"
            f"Worker outputs:\n{worker_outputs}\n\n"
            "Synthesise these into a single, coherent response for the user. "
            "Be concise and directly address the original query."
        )
        response = await self._client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    async def run(self, query: str) -> OrchestratorResult:
        """Full orchestration pipeline: decompose → dispatch → aggregate."""
        start = time.monotonic()

        plan = await self._decompose(query)
        worker_results = await self._dispatch_parallel(plan.tasks)
        final_response = await self._aggregate(query, worker_results)

        total_ms = (time.monotonic() - start) * 1000
        logger.info("Orchestration complete in %.0fms", total_ms)

        return OrchestratorResult(
            original_query=query,
            plan=plan,
            worker_results=worker_results,
            final_response=final_response,
            total_duration_ms=total_ms,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    orchestrator = (
        Orchestrator(client)
        .register_worker(Worker("search", client))
        .register_worker(Worker("summarize", client))
        .register_worker(Worker("calculate", client))
    )

    query = (
        "What is the population of Tokyo, summarize the key facts about the city, "
        "and calculate how many times larger it is than London (population 9 million)."
    )

    result = await orchestrator.run(query)

    print(f"\nQuery   : {result.original_query}")
    print(f"Plan    : {result.plan.reasoning}")
    for wr in result.worker_results:
        status = "OK" if wr.success else "FAIL"
        print(f"  [{status}] {wr.task_id} ({wr.specialty}) — {wr.duration_ms:.0f}ms")
    print(f"\nResponse: {result.final_response}")
    print(f"\nTotal   : {result.total_duration_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tests (pytest-style) — run: pytest day-2.py -v
# ---------------------------------------------------------------------------

import unittest.mock  # noqa: E402


def _async_return(value: Any) -> Any:
    """Wrap a value in a coroutine so AsyncMock.return_value works."""
    async def _coro(*_args: Any, **_kwargs: Any) -> Any:
        return value
    return _coro()


def _make_async_mock_client(
    decompose_payload: dict[str, Any] | None = None,
    worker_text: str = "worker output",
) -> unittest.mock.AsyncMock:
    """
    Build an AsyncAnthropic mock that:
    - Returns a tool_use block for the decompose call
    - Returns a text block for every worker / aggregation call
    """
    client = unittest.mock.AsyncMock(spec=anthropic.AsyncAnthropic)

    # Decompose response — tool_use block
    tool_block = unittest.mock.MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = decompose_payload or {
        "tasks": [
            {
                "task_id": "task_1",
                "specialty": "search",
                "description": "Look up X",
                "input_data": "X",
            }
        ],
        "reasoning": "Single search task needed",
    }
    decompose_response = unittest.mock.MagicMock()
    decompose_response.content = [tool_block]

    # Worker / aggregation response — text block
    text_block = unittest.mock.MagicMock()
    text_block.type = "text"
    text_block.text = worker_text
    text_response = unittest.mock.MagicMock()
    text_response.content = [text_block]

    # First call = decompose (tool_use), all subsequent = text
    client.messages.create.side_effect = [
        decompose_response,  # decompose
        text_response,       # worker execute
        text_response,       # aggregation
    ]
    return client


def test_decomposed_plan_validation() -> None:
    """DecomposedPlan parses correctly from raw dict."""
    raw = {
        "tasks": [
            {
                "task_id": "t1",
                "specialty": "search",
                "description": "find info",
                "input_data": "query",
            }
        ],
        "reasoning": "needs search",
    }
    plan = DecomposedPlan.model_validate(raw)
    assert len(plan.tasks) == 1
    assert plan.tasks[0].specialty == "search"


def test_worker_result_model() -> None:
    """WorkerResult round-trips through Pydantic without loss."""
    wr = WorkerResult(
        task_id="t1",
        specialty="calculate",
        output="42",
        duration_ms=123.4,
        success=True,
    )
    assert wr.model_dump()["task_id"] == "t1"


def test_worker_execute_success() -> None:
    """Worker.execute returns WorkerResult with success=True on normal response."""
    text_block = unittest.mock.MagicMock()
    text_block.text = "search result"

    mock_response = unittest.mock.MagicMock()
    mock_response.content = [text_block]

    async_client = unittest.mock.AsyncMock(spec=anthropic.AsyncAnthropic)
    async_client.messages.create.return_value = mock_response

    worker = Worker("search", async_client)
    task = Task(
        task_id="t1",
        specialty="search",
        description="find X",
        input_data="X",
    )
    result = asyncio.run(worker.execute(task))

    assert result.success is True
    assert result.output == "search result"
    assert result.task_id == "t1"


def test_worker_execute_captures_exception() -> None:
    """Worker.execute returns success=False when the LLM call raises."""
    async_client = unittest.mock.AsyncMock(spec=anthropic.AsyncAnthropic)
    async_client.messages.create.side_effect = RuntimeError("timeout")

    worker = Worker("summarize", async_client)
    task = Task(
        task_id="t2",
        specialty="summarize",
        description="summarize Y",
        input_data="Y",
    )
    result = asyncio.run(worker.execute(task))

    assert result.success is False
    assert "timeout" in result.output


def test_orchestrator_run_full_pipeline() -> None:
    """
    Full pipeline: decompose → dispatch → aggregate.
    Verifies OrchestratorResult structure and that all three LLM calls were made.
    """
    mock_client = _make_async_mock_client(worker_text="aggregated answer")

    orchestrator = Orchestrator(mock_client).register_worker(
        Worker("search", mock_client)
    )

    result = asyncio.run(orchestrator.run("find something"))

    assert isinstance(result, OrchestratorResult)
    assert len(result.worker_results) == 1
    assert result.worker_results[0].specialty == "search"
    assert result.final_response == "aggregated answer"
    # decompose + worker + aggregate = 3 calls
    assert mock_client.messages.create.call_count == 3


def test_orchestrator_skips_unregistered_specialty() -> None:
    """Tasks whose specialty has no registered worker are silently skipped."""
    mock_client = _make_async_mock_client(
        decompose_payload={
            "tasks": [
                {
                    "task_id": "t1",
                    "specialty": "calculate",   # no worker registered
                    "description": "calc",
                    "input_data": "1+1",
                }
            ],
            "reasoning": "needs calc",
        }
    )

    # Register only search worker — no calculate worker
    orchestrator = Orchestrator(mock_client).register_worker(
        Worker("search", mock_client)
    )

    result = asyncio.run(orchestrator.run("compute something"))

    # No worker ran — worker_results should be empty
    assert result.worker_results == []
```

## Exercises

1. **Sequential pipeline**: Modify the orchestrator to support a `"sequential"` flag on tasks. When set, the task waits for the previous task's `WorkerResult` and appends it to its own `input_data` before execution. Implement this without breaking the parallel path for tasks that don't need it.

2. **Retry logic**: Wrap `Worker.execute()` with an `async_retry(max_attempts=3, backoff_ms=200)` decorator that retries on failure. Write a test using `AsyncMock.side_effect` to simulate two failures followed by a success, and assert the final result is `success=True`.

3. **Aggregation without a second LLM call**: Replace the `_aggregate()` LLM call with a deterministic template renderer that formats worker results into a structured Markdown report. Compare the output quality and cost trade-offs with the LLM aggregation approach.

(Solutions: ../solutions/week-06/day-2.md)

## Self-check

- [ ] I can explain why `asyncio.gather(return_exceptions=True)` is safer than the default for multi-worker dispatch
- [ ] I can describe the two LLM calls the orchestrator makes and what each one does
- [ ] I can implement `Orchestrator.run()` end-to-end without reference, including `return_exceptions` handling
- [ ] I can write an `AsyncMock`-based test for a coroutine that raises on first call and succeeds on second
- [ ] I can articulate the trade-off between parallel and sequential dispatch and choose the right mode for a given dependency graph
