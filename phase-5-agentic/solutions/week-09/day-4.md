# Solutions — Week 09 · Day 4 — Task Delegation Across Agents

## Exercise 1: timeout_seconds with asyncio.wait_for

```python
import asyncio
import pytest
from dataclasses import dataclass, field
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.COMPLETED
    output: str = ""
    error: str = ""


async def _slow_handler(payload: dict) -> TaskResult:
    await asyncio.sleep(5.0)  # Simulates a slow agent
    return TaskResult(output="done")


async def execute_with_timeout(
    handler,
    payload: dict,
    timeout_seconds: float,
) -> TaskResult:
    task_id = str(uuid.uuid4())
    try:
        result = await asyncio.wait_for(handler(payload), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error="timeout",
        )


@pytest.mark.asyncio
async def test_timeout_triggered():
    result = await execute_with_timeout(_slow_handler, {}, timeout_seconds=0.1)
    assert result.status == TaskStatus.FAILED
    assert result.error == "timeout"


@pytest.mark.asyncio
async def test_no_timeout_on_fast_handler():
    async def fast_handler(payload: dict) -> TaskResult:
        return TaskResult(output="fast")

    result = await execute_with_timeout(fast_handler, {}, timeout_seconds=1.0)
    assert result.status == TaskStatus.COMPLETED
```

## Exercise 2: Fan-out — first completed wins

```python
import asyncio
from typing import Callable, Awaitable


async def fan_out_first(
    handlers: list[Callable[[dict], Awaitable[TaskResult]]],
    payload: dict,
) -> TaskResult:
    """Run all handlers; return the first one that completes. Cancel the rest."""
    tasks = [asyncio.create_task(h(payload)) for h in handlers]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for p in pending:
        p.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    return done.pop().result()


@pytest.mark.asyncio
async def test_fan_out_first():
    async def slow(p: dict) -> TaskResult:
        await asyncio.sleep(0.5)
        return TaskResult(output="slow")

    async def fast(p: dict) -> TaskResult:
        await asyncio.sleep(0.01)
        return TaskResult(output="fast")

    result = await fan_out_first([slow, fast], {})
    assert result.output == "fast"
```

## Exercise 3: Priority queue delegation

```python
import asyncio
from dataclasses import dataclass, field
import uuid


@dataclass(order=True)
class PrioritizedTask:
    priority: int           # lower number = higher priority in asyncio.PriorityQueue
    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    payload: dict = field(compare=False, default_factory=dict)


async def execute_with_priority(
    tasks: list,  # list of (priority, payload) tuples
    handler,
) -> list[TaskResult]:
    queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    for priority, payload in tasks:
        # Negate priority so higher priority (5) comes first
        await queue.put(PrioritizedTask(priority=-priority, payload=payload))

    results: list[TaskResult] = []
    while not queue.empty():
        item = await queue.get()
        result = await handler(item.payload)
        results.append(result)
    return results


@pytest.mark.asyncio
async def test_priority_ordering():
    execution_order = []

    async def tracking_handler(payload: dict) -> TaskResult:
        execution_order.append(payload["label"])
        return TaskResult(output=payload["label"])

    tasks = [(1, {"label": "low"}), (5, {"label": "high"}), (3, {"label": "mid"})]
    await execute_with_priority(tasks, tracking_handler)
    assert execution_order[0] == "high"   # Priority 5 first
    assert execution_order[-1] == "low"   # Priority 1 last
```

## Exercise 4: ResearchContext aggregation

```python
from pydantic import BaseModel


class ResearchContext(BaseModel):
    news: str = ""
    financials: str = ""
    sentiment: str = ""

    def is_complete(self) -> bool:
        return bool(self.news and self.financials and self.sentiment)


def aggregate_phase1(results: list[TaskResult], skill_map: dict[str, str]) -> ResearchContext:
    """
    Build a ResearchContext from phase-1 results.
    skill_map: {task_id -> skill_name}
    """
    ctx = ResearchContext()
    for result in results:
        skill = skill_map.get(result.task_id, "")
        if skill == "news-search":
            ctx.news = result.output
        elif skill == "financials-fetch":
            ctx.financials = result.output
        elif skill == "sentiment-analysis":
            ctx.sentiment = result.output
    return ctx


def test_research_context_complete():
    t1 = TaskResult(task_id="t1", output="News data here", status=TaskStatus.COMPLETED)
    t2 = TaskResult(task_id="t2", output="Revenue +12%", status=TaskStatus.COMPLETED)
    skill_map = {"t1": "news-search", "t2": "financials-fetch"}
    ctx = aggregate_phase1([t1, t2], skill_map)
    assert ctx.news == "News data here"
    assert ctx.financials == "Revenue +12%"
    assert not ctx.is_complete()   # Missing sentiment


def test_research_context_fully_populated():
    tasks = [
        TaskResult(task_id="t1", output="news", status=TaskStatus.COMPLETED),
        TaskResult(task_id="t2", output="financials", status=TaskStatus.COMPLETED),
        TaskResult(task_id="t3", output="positive", status=TaskStatus.COMPLETED),
    ]
    skill_map = {"t1": "news-search", "t2": "financials-fetch", "t3": "sentiment-analysis"}
    ctx = aggregate_phase1(tasks, skill_map)
    assert ctx.is_complete()
```
