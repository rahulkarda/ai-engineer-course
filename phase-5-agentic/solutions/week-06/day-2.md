# Solutions — Week 06 · Day 2 — Delegation & Orchestrator-Worker

> These are reference solutions. Your implementation may differ and still be correct.

## Exercise 1

**Add a `"validate"` worker type that checks the plausibility of results from other workers before aggregation.**

Design: the orchestrator runs all non-validate tasks first, then runs each `"validate"` task. A validate task receives prior `WorkerResult` objects in its prompt and emits a `WorkerResult` with `success=False` when a result is implausible. The aggregator skips results flagged by a corresponding validator.

```python
"""
day2_ex1.py — Orchestrator-worker with validate worker type.

Run: python day2_ex1.py
Env: ANTHROPIC_API_KEY
"""
import asyncio
import json
import logging
import os
import unittest.mock
from typing import Any

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: str
    description: str
    worker_type: str  # "research" | "summarize" | "validate" | "code" | ...
    depends_on: list[str] | None = None


class WorkerResult(BaseModel):
    task_id: str
    result: str
    success: bool
    attempts: int = 1
    invalidated_by: str | None = None  # set when a validator rejects this result


# ---------------------------------------------------------------------------
# Worker implementations
# ---------------------------------------------------------------------------

async def research_worker(
    task: Task, client: anthropic.AsyncAnthropic
) -> WorkerResult:
    logger.info("[research] task=%s", task.id)
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Research: {task.description}"}],
    )
    return WorkerResult(
        task_id=task.id,
        result=response.content[0].text,
        success=True,
    )


async def summarize_worker(
    task: Task, client: anthropic.AsyncAnthropic
) -> WorkerResult:
    logger.info("[summarize] task=%s", task.id)
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": f"Summarise in 2 sentences: {task.description}",
            }
        ],
    )
    return WorkerResult(
        task_id=task.id,
        result=response.content[0].text,
        success=True,
    )


async def validate_worker(
    task: Task,
    client: anthropic.AsyncAnthropic,
    prior_results: dict[str, WorkerResult],
) -> WorkerResult:
    """
    Checks plausibility of prior_results.
    Returns a WorkerResult whose result is a JSON object:
      {"valid": bool, "reason": str, "flagged_task_ids": list[str]}
    """
    logger.info("[validate] task=%s checking %d prior results", task.id, len(prior_results))
    summary = "\n".join(
        f"[{tid}] {wr.result[:200]}" for tid, wr in prior_results.items()
    )
    prompt = (
        f"You are a fact-checker. Here are results from worker tasks:\n\n"
        f"{summary}\n\n"
        "Validation criteria: Are these results plausible, coherent, and non-contradictory? "
        "Respond as JSON: "
        '{"valid": true/false, "reason": "...", "flagged_task_ids": ["id1", ...]}\n'
        "flagged_task_ids should list any task IDs whose results seem implausible."
    )
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw.strip())
    return WorkerResult(
        task_id=task.id,
        result=response.content[0].text,
        success=data.get("valid", True),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

    async def run(self, tasks: list[Task]) -> dict[str, WorkerResult]:
        results: dict[str, WorkerResult] = {}

        # Separate validate tasks from work tasks (validate runs after work)
        work_tasks = [t for t in tasks if t.worker_type != "validate"]
        validate_tasks = [t for t in tasks if t.worker_type == "validate"]

        # Run all work tasks concurrently
        coros = [self._run_task(t, results) for t in work_tasks]
        work_results = await asyncio.gather(*coros)
        for wr in work_results:
            results[wr.task_id] = wr

        # Run validate tasks sequentially (each sees all prior work)
        for vtask in validate_tasks:
            vresult = await validate_worker(vtask, self._client, results)
            results[vtask.task_id] = vresult

            # Mark invalidated results
            if not vresult.success:
                try:
                    data = json.loads(vresult.result)
                    for flagged_id in data.get("flagged_task_ids", []):
                        if flagged_id in results:
                            results[flagged_id] = results[flagged_id].model_copy(
                                update={"invalidated_by": vtask.id}
                            )
                            logger.warning(
                                "Task %s invalidated by validator %s",
                                flagged_id,
                                vtask.id,
                            )
                except (json.JSONDecodeError, AttributeError):
                    pass

        return results

    async def _run_task(
        self, task: Task, _results: dict[str, WorkerResult]
    ) -> WorkerResult:
        if task.worker_type == "research":
            return await research_worker(task, self._client)
        if task.worker_type == "summarize":
            return await summarize_worker(task, self._client)
        return WorkerResult(
            task_id=task.id,
            result=f"Unknown worker type: {task.worker_type}",
            success=False,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    orchestrator = Orchestrator(client)

    tasks = [
        Task(id="t1", description="History of the Python language", worker_type="research"),
        Task(id="t2", description="Benefits of async programming", worker_type="summarize"),
        Task(
            id="v1",
            description="Validate prior research results",
            worker_type="validate",
        ),
    ]

    results = await orchestrator.run(tasks)
    for tid, wr in results.items():
        status = "INVALIDATED" if wr.invalidated_by else ("OK" if wr.success else "FAILED")
        print(f"\n[{tid}] {status}: {wr.result[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_validate_worker_marks_success_on_plausible_results() -> None:
    """Validate worker returns success=True when LLM reports valid=true."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    json_response = '{"valid": true, "reason": "Results are coherent", "flagged_task_ids": []}'
    mock_msg = unittest.mock.MagicMock()
    mock_msg.content = [unittest.mock.MagicMock(text=json_response)]
    mock_client.messages.create = unittest.mock.AsyncMock(return_value=mock_msg)

    task = Task(id="v1", description="Check results", worker_type="validate")
    prior = {
        "t1": WorkerResult(task_id="t1", result="Python was created in 1991", success=True)
    }

    result = asyncio.run(validate_worker(task, mock_client, prior))
    assert result.success is True
    assert result.task_id == "v1"


def test_validate_worker_marks_failure_on_implausible_results() -> None:
    """Validate worker returns success=False when LLM reports valid=false."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    json_response = (
        '{"valid": false, "reason": "Contradictory claims", "flagged_task_ids": ["t1"]}'
    )
    mock_msg = unittest.mock.MagicMock()
    mock_msg.content = [unittest.mock.MagicMock(text=json_response)]
    mock_client.messages.create = unittest.mock.AsyncMock(return_value=mock_msg)

    task = Task(id="v1", description="Check", worker_type="validate")
    prior = {
        "t1": WorkerResult(task_id="t1", result="Contradictory claim", success=True)
    }

    result = asyncio.run(validate_worker(task, mock_client, prior))
    assert result.success is False


def test_orchestrator_separates_validate_tasks() -> None:
    """Validate tasks are detected and separated from work tasks."""
    tasks = [
        Task(id="t1", description="Research X", worker_type="research"),
        Task(id="v1", description="Validate", worker_type="validate"),
    ]
    work = [t for t in tasks if t.worker_type != "validate"]
    validate = [t for t in tasks if t.worker_type == "validate"]
    assert len(work) == 1
    assert len(validate) == 1
    assert work[0].id == "t1"
    assert validate[0].id == "v1"


def test_orchestrator_invalidates_flagged_results() -> None:
    """Results flagged by a validator are marked with invalidated_by."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    # Work task response
    work_msg = unittest.mock.MagicMock()
    work_msg.content = [unittest.mock.MagicMock(text="Some research result")]
    # Validate task response — flags t1
    validate_msg = unittest.mock.MagicMock()
    validate_msg.content = [
        unittest.mock.MagicMock(
            text='{"valid": false, "reason": "Implausible", "flagged_task_ids": ["t1"]}'
        )
    ]
    mock_client.messages.create = unittest.mock.AsyncMock(
        side_effect=[work_msg, validate_msg]
    )

    orchestrator = Orchestrator(mock_client)
    tasks = [
        Task(id="t1", description="Research", worker_type="research"),
        Task(id="v1", description="Validate", worker_type="validate"),
    ]
    results = asyncio.run(orchestrator.run(tasks))

    assert results["t1"].invalidated_by == "v1"
```


## Exercise 2

**Implement sequential task dependency: Task B can only run after Task A completes, by checking a `depends_on` field in the Task model.**

Design: the orchestrator topologically sorts tasks. Tasks without `depends_on` (or with all dependencies satisfied) form the runnable frontier. Each completed result is added to the resolved set before the next wave is dispatched. Tasks whose dependencies are never resolved stay unscheduled and produce a `WorkerResult(success=False)`.

```python
"""
day2_ex2.py — Orchestrator with sequential task dependencies.

Run: python day2_ex2.py
Env: ANTHROPIC_API_KEY
"""
import asyncio
import logging
import os
import unittest.mock

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: str
    description: str
    worker_type: str
    depends_on: list[str] | None = None


class WorkerResult(BaseModel):
    task_id: str
    result: str
    success: bool
    attempts: int = 1


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

async def run_worker(task: Task, client: anthropic.AsyncAnthropic) -> WorkerResult:
    logger.info("Running worker type=%s id=%s", task.worker_type, task.id)
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        messages=[{"role": "user", "content": task.description}],
    )
    return WorkerResult(
        task_id=task.id,
        result=response.content[0].text,
        success=True,
    )


# ---------------------------------------------------------------------------
# Dependency-aware orchestrator
# ---------------------------------------------------------------------------

def _resolve_execution_waves(tasks: list[Task]) -> list[list[Task]]:
    """
    Return tasks grouped into dependency waves.

    Wave 0 has no dependencies (or all None).
    Wave N+1 depends only on tasks resolved in waves 0..N.

    Raises ValueError for circular dependencies.
    """
    resolved: set[str] = set()
    remaining = list(tasks)
    waves: list[list[Task]] = []

    while remaining:
        # Tasks whose dependencies are all resolved (or have no deps)
        wave = [
            t
            for t in remaining
            if not t.depends_on or all(d in resolved for d in t.depends_on)
        ]
        if not wave:
            unresolved = [t.id for t in remaining]
            raise ValueError(
                f"Circular or unresolvable dependencies for tasks: {unresolved}"
            )
        waves.append(wave)
        resolved.update(t.id for t in wave)
        remaining = [t for t in remaining if t.id not in resolved]

    return waves


class Orchestrator:
    def __init__(self, client: anthropic.AsyncAnthropic) -> None:
        self._client = client

    async def run(self, tasks: list[Task]) -> dict[str, WorkerResult]:
        results: dict[str, WorkerResult] = {}

        try:
            waves = _resolve_execution_waves(tasks)
        except ValueError as exc:
            logger.error("Dependency resolution failed: %s", exc)
            # Mark all tasks as failed
            for task in tasks:
                results[task.id] = WorkerResult(
                    task_id=task.id,
                    result=f"Dependency error: {exc}",
                    success=False,
                )
            return results

        for wave_idx, wave in enumerate(waves):
            logger.info(
                "Wave %d: running %d task(s) — %s",
                wave_idx,
                len(wave),
                [t.id for t in wave],
            )
            wave_results = await asyncio.gather(
                *[run_worker(t, self._client) for t in wave]
            )
            for wr in wave_results:
                results[wr.task_id] = wr
                logger.info("Completed task=%s success=%s", wr.task_id, wr.success)

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    orchestrator = Orchestrator(client)

    # t2 depends on t1; t3 depends on t2 — strictly sequential
    tasks = [
        Task(id="t1", description="List 3 famous scientists", worker_type="research"),
        Task(
            id="t2",
            description="Summarise the work of each scientist from t1",
            worker_type="summarize",
            depends_on=["t1"],
        ),
        Task(
            id="t3",
            description="Identify common themes across the summaries from t2",
            worker_type="analyze",
            depends_on=["t2"],
        ),
    ]

    results = await orchestrator.run(tasks)
    for tid, wr in results.items():
        print(f"\n[{tid}] success={wr.success}: {wr.result[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resolve_waves_no_deps() -> None:
    """Tasks with no dependencies go into a single wave."""
    tasks = [
        Task(id="a", description="A", worker_type="research"),
        Task(id="b", description="B", worker_type="research"),
    ]
    waves = _resolve_execution_waves(tasks)
    assert len(waves) == 1
    assert {t.id for t in waves[0]} == {"a", "b"}


def test_resolve_waves_sequential_chain() -> None:
    """A → B → C produces three waves of one task each."""
    tasks = [
        Task(id="a", description="A", worker_type="w"),
        Task(id="b", description="B", worker_type="w", depends_on=["a"]),
        Task(id="c", description="C", worker_type="w", depends_on=["b"]),
    ]
    waves = _resolve_execution_waves(tasks)
    assert len(waves) == 3
    assert waves[0][0].id == "a"
    assert waves[1][0].id == "b"
    assert waves[2][0].id == "c"


def test_resolve_waves_diamond() -> None:
    """Diamond: A → (B, C) → D — three waves."""
    tasks = [
        Task(id="a", description="A", worker_type="w"),
        Task(id="b", description="B", worker_type="w", depends_on=["a"]),
        Task(id="c", description="C", worker_type="w", depends_on=["a"]),
        Task(id="d", description="D", worker_type="w", depends_on=["b", "c"]),
    ]
    waves = _resolve_execution_waves(tasks)
    assert len(waves) == 3
    assert waves[0][0].id == "a"
    assert {t.id for t in waves[1]} == {"b", "c"}
    assert waves[2][0].id == "d"


def test_resolve_waves_circular_raises() -> None:
    tasks = [
        Task(id="a", description="A", worker_type="w", depends_on=["b"]),
        Task(id="b", description="B", worker_type="w", depends_on=["a"]),
    ]
    try:
        _resolve_execution_waves(tasks)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_orchestrator_respects_dependency_order() -> None:
    """Tasks run in dependency order — verified by call order on mock."""
    call_order: list[str] = []

    async def mock_create(**kwargs: object) -> object:
        # Extract task description from messages
        msgs = kwargs.get("messages", [])
        content = msgs[0]["content"] if msgs else ""
        # Map description prefix to task id
        task_id = "t1" if "first" in str(content).lower() else "t2"
        call_order.append(task_id)
        mock_msg = unittest.mock.MagicMock()
        mock_msg.content = [unittest.mock.MagicMock(text=f"Result for {task_id}")]
        return mock_msg

    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    mock_client.messages.create = unittest.mock.AsyncMock(side_effect=mock_create)

    tasks = [
        Task(id="t1", description="First task", worker_type="research"),
        Task(id="t2", description="Second task", worker_type="summarize", depends_on=["t1"]),
    ]

    orchestrator = Orchestrator(mock_client)
    results = asyncio.run(orchestrator.run(tasks))

    assert results["t1"].success is True
    assert results["t2"].success is True
    # t1 must complete before t2 starts
    assert call_order.index("t1") < call_order.index("t2")


def test_orchestrator_reports_all_failed_on_circular_dep() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    tasks = [
        Task(id="a", description="A", worker_type="w", depends_on=["b"]),
        Task(id="b", description="B", worker_type="w", depends_on=["a"]),
    ]
    orchestrator = Orchestrator(mock_client)
    results = asyncio.run(orchestrator.run(tasks))

    assert all(not r.success for r in results.values())
```


## Exercise 3

**Add retry logic: if a worker fails, retry up to 2 times before marking the task as failed.**

Design: wrap the worker call in a loop with `max_retries=2` extra attempts (3 total). Track `attempts` on `WorkerResult`. Use exponential back-off between retries. Any non-`Exception` or a successful result breaks the loop early.

```python
"""
day2_ex3.py — Orchestrator-worker with retry logic (max 2 retries).

Run: python day2_ex3.py
Env: ANTHROPIC_API_KEY
"""
import asyncio
import logging
import os
import unittest.mock

import anthropic
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 2  # extra attempts after initial failure; total = MAX_RETRIES + 1
RETRY_BASE_DELAY = 0.5  # seconds; doubles each retry (exponential back-off)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: str
    description: str
    worker_type: str
    depends_on: list[str] | None = None


class WorkerResult(BaseModel):
    task_id: str
    result: str
    success: bool
    attempts: int = Field(default=1, ge=1)  # total attempts made


# ---------------------------------------------------------------------------
# Worker with retry wrapper
# ---------------------------------------------------------------------------

async def _call_llm(task: Task, client: anthropic.AsyncAnthropic) -> str:
    """Single LLM call — can raise on failure."""
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=256,
        messages=[{"role": "user", "content": task.description}],
    )
    return response.content[0].text


async def run_worker_with_retry(
    task: Task,
    client: anthropic.AsyncAnthropic,
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BASE_DELAY,
) -> WorkerResult:
    """
    Run the worker, retrying up to max_retries times on exception.

    Exponential back-off: delay = base_delay * 2^(attempt - 1).
    Returns WorkerResult with success=False after exhausting retries.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 2):  # 1..max_retries+1
        try:
            logger.info(
                "Task %s attempt %d/%d", task.id, attempt, max_retries + 1
            )
            result_text = await _call_llm(task, client)
            logger.info("Task %s succeeded on attempt %d", task.id, attempt)
            return WorkerResult(
                task_id=task.id,
                result=result_text,
                success=True,
                attempts=attempt,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Task %s attempt %d failed: %s", task.id, attempt, exc
            )
            if attempt <= max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.info("Retrying task %s in %.1fs...", task.id, delay)
                await asyncio.sleep(delay)

    return WorkerResult(
        task_id=task.id,
        result=f"Failed after {max_retries + 1} attempts: {last_error}",
        success=False,
        attempts=max_retries + 1,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._client = client
        self._max_retries = max_retries

    async def run(self, tasks: list[Task]) -> dict[str, WorkerResult]:
        coros = [
            run_worker_with_retry(
                t, self._client, max_retries=self._max_retries, base_delay=0.0
            )
            for t in tasks
        ]
        results = await asyncio.gather(*coros)
        return {r.task_id: r for r in results}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    orchestrator = Orchestrator(client)

    tasks = [
        Task(id="t1", description="Explain recursion in one paragraph", worker_type="research"),
        Task(id="t2", description="Give 3 examples of sorting algorithms", worker_type="research"),
    ]

    results = await orchestrator.run(tasks)
    for tid, wr in results.items():
        print(f"\n[{tid}] success={wr.success} attempts={wr.attempts}")
        print(f"Result: {wr.result[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_worker_succeeds_on_first_attempt() -> None:
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    mock_msg = unittest.mock.MagicMock()
    mock_msg.content = [unittest.mock.MagicMock(text="Success")]
    mock_client.messages.create = unittest.mock.AsyncMock(return_value=mock_msg)

    task = Task(id="t1", description="Test", worker_type="research")
    result = asyncio.run(
        run_worker_with_retry(task, mock_client, max_retries=2, base_delay=0.0)
    )

    assert result.success is True
    assert result.attempts == 1
    mock_client.messages.create.assert_called_once()


def test_worker_retries_on_failure_and_eventually_succeeds() -> None:
    """Fails twice, succeeds on third attempt."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    mock_msg = unittest.mock.MagicMock()
    mock_msg.content = [unittest.mock.MagicMock(text="Success on attempt 3")]
    mock_client.messages.create = unittest.mock.AsyncMock(
        side_effect=[
            RuntimeError("network error"),
            RuntimeError("timeout"),
            mock_msg,
        ]
    )

    task = Task(id="t1", description="Test", worker_type="research")
    result = asyncio.run(
        run_worker_with_retry(task, mock_client, max_retries=2, base_delay=0.0)
    )

    assert result.success is True
    assert result.attempts == 3
    assert mock_client.messages.create.call_count == 3


def test_worker_marks_failed_after_exhausting_retries() -> None:
    """Fails all 3 attempts — WorkerResult success=False, attempts=3."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    mock_client.messages.create = unittest.mock.AsyncMock(
        side_effect=RuntimeError("persistent error")
    )

    task = Task(id="t1", description="Test", worker_type="research")
    result = asyncio.run(
        run_worker_with_retry(task, mock_client, max_retries=2, base_delay=0.0)
    )

    assert result.success is False
    assert result.attempts == 3
    assert "persistent error" in result.result
    assert mock_client.messages.create.call_count == 3


def test_worker_no_retries_means_single_attempt() -> None:
    """max_retries=0 — exactly one attempt, no retry on failure."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    mock_client.messages.create = unittest.mock.AsyncMock(
        side_effect=RuntimeError("fail")
    )

    task = Task(id="t1", description="Test", worker_type="research")
    result = asyncio.run(
        run_worker_with_retry(task, mock_client, max_retries=0, base_delay=0.0)
    )

    assert result.success is False
    assert result.attempts == 1
    mock_client.messages.create.assert_called_once()


def test_orchestrator_collects_all_results_with_retry() -> None:
    """Orchestrator returns a result dict keyed by task_id."""
    mock_client = unittest.mock.MagicMock(spec=anthropic.AsyncAnthropic)
    mock_msg = unittest.mock.MagicMock()
    mock_msg.content = [unittest.mock.MagicMock(text="OK")]
    mock_client.messages.create = unittest.mock.AsyncMock(return_value=mock_msg)

    tasks = [
        Task(id="a", description="Task A", worker_type="research"),
        Task(id="b", description="Task B", worker_type="research"),
    ]
    orchestrator = Orchestrator(mock_client)
    results = asyncio.run(orchestrator.run(tasks))

    assert set(results.keys()) == {"a", "b"}
    assert all(r.success for r in results.values())


def test_worker_result_attempts_field_minimum() -> None:
    """WorkerResult rejects attempts < 1 via Pydantic validation."""
    from pydantic import ValidationError
    try:
        WorkerResult(task_id="x", result="ok", success=True, attempts=0)
        assert False, "Should raise"
    except ValidationError:
        pass
```
