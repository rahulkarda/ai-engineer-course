# Day 1 Solutions: Async Messaging vs Sync Communication

## Exercise 1: Priority Queue

Add priority support to `InMemoryBroker` so high-priority messages are dequeued first.

```python
import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedMessage:
    """Wrapper so heapq can order by priority (higher = more urgent)."""
    priority: int
    sequence: int  # tie-breaker: earlier messages first within same priority
    message: Any = field(compare=False)

    def __lt__(self, other: "PrioritizedMessage") -> bool:
        # invert priority so heapq (min-heap) serves highest priority first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.sequence < other.sequence


class PriorityInMemoryBroker:
    def __init__(self) -> None:
        self._heap: list[PrioritizedMessage] = []
        self._sequence: int = 0
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()

    async def publish(self, message: Any, priority: int = 0) -> None:
        async with self._lock:
            entry = PrioritizedMessage(
                priority=priority,
                sequence=self._sequence,
                message=message,
            )
            heapq.heappush(self._heap, entry)
            self._sequence += 1
            self._not_empty.set()
            logger.debug("published priority=%d seq=%d", priority, self._sequence - 1)

    async def consume(self) -> Any:
        while True:
            await self._not_empty.wait()
            async with self._lock:
                if self._heap:
                    entry = heapq.heappop(self._heap)
                    if not self._heap:
                        self._not_empty.clear()
                    logger.debug("consumed priority=%d", entry.priority)
                    return entry.message


async def demo_priority() -> None:
    broker = PriorityInMemoryBroker()

    # publish out of order
    await broker.publish({"body": "low"}, priority=1)
    await broker.publish({"body": "critical"}, priority=10)
    await broker.publish({"body": "medium"}, priority=5)

    results = [await broker.consume() for _ in range(3)]
    assert [r["body"] for r in results] == ["critical", "medium", "low"], results
    logger.info("priority ordering verified: %s", [r["body"] for r in results])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_priority())
```

**Key points:**
- `heapq` is a min-heap; invert priority comparison so highest value wins.
- Sequence number breaks ties so messages with equal priority preserve FIFO order.
- `asyncio.Lock` guards both push and pop; `asyncio.Event` avoids busy-waiting.

---

## Exercise 2: Message Latency Tracking

Attach a `published_at` timestamp at publish time and measure end-to-end latency at consume time.

```python
import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TimestampedMessage:
    payload: Any
    published_at: float = field(default_factory=time.monotonic)


class LatencyTrackingBroker:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[TimestampedMessage] = asyncio.Queue()
        self._latencies: list[float] = []

    async def publish(self, payload: Any) -> None:
        msg = TimestampedMessage(payload=payload)
        await self._queue.put(msg)

    async def consume(self) -> tuple[Any, float]:
        """Returns (payload, latency_seconds)."""
        msg = await self._queue.get()
        latency = time.monotonic() - msg.published_at
        self._latencies.append(latency)
        self._queue.task_done()
        return msg.payload, latency

    def latency_stats(self) -> dict[str, float]:
        if not self._latencies:
            return {}
        return {
            "count": len(self._latencies),
            "mean_ms": statistics.mean(self._latencies) * 1000,
            "p50_ms": statistics.median(self._latencies) * 1000,
            "p99_ms": sorted(self._latencies)[int(len(self._latencies) * 0.99)] * 1000,
            "max_ms": max(self._latencies) * 1000,
        }


async def producer(broker: LatencyTrackingBroker, n: int) -> None:
    for i in range(n):
        await broker.publish({"id": i})
        await asyncio.sleep(0.001)  # simulate 1ms inter-message gap


async def consumer(broker: LatencyTrackingBroker, n: int) -> None:
    for _ in range(n):
        payload, latency = await broker.consume()
        logger.debug("id=%d latency=%.3fms", payload["id"], latency * 1000)


async def demo_latency() -> None:
    broker = LatencyTrackingBroker()
    n = 100
    await asyncio.gather(producer(broker, n), consumer(broker, n))
    stats = broker.latency_stats()
    logger.info("latency stats: %s", stats)
    assert stats["count"] == n


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_latency())
```

**Key points:**
- `time.monotonic()` is correct here — it never goes backward, unlike `time.time()`.
- Latency is stored per-message; aggregation happens after all messages are consumed.
- p99 uses a simple sort-and-index; for production use `numpy.percentile` or `scipy.stats`.

---

## Exercise 3: Circuit Breaker on Queue Depth

Pause publishing when queue depth exceeds a threshold; auto-recover when depth drops.

```python
import asyncio
import logging
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = auto()    # normal: publishing allowed
    OPEN = auto()      # tripped: publishing blocked
    HALF_OPEN = auto() # probe: one message allowed to test recovery


class CircuitBreakerBroker:
    def __init__(
        self,
        depth_threshold: int = 100,
        recovery_probe_interval: float = 1.0,
    ) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._threshold = depth_threshold
        self._probe_interval = recovery_probe_interval
        self._state = CircuitState.CLOSED
        self._last_open_at: float | None = None

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _evaluate(self) -> None:
        if self._state == CircuitState.CLOSED and self.depth >= self._threshold:
            self._state = CircuitState.OPEN
            logger.warning("circuit OPEN — depth=%d >= threshold=%d", self.depth, self._threshold)
        elif self._state == CircuitState.OPEN and self.depth < self._threshold:
            self._state = CircuitState.HALF_OPEN
            logger.info("circuit HALF_OPEN — depth=%d, allowing probe", self.depth)
        elif self._state == CircuitState.HALF_OPEN and self.depth < self._threshold // 2:
            self._state = CircuitState.CLOSED
            logger.info("circuit CLOSED — depth back to normal")

    async def publish(self, message: Any, *, timeout: float = 5.0) -> bool:
        """
        Returns True if published, False if circuit is open and timeout exceeded.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            self._evaluate()
            if self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN):
                await self._queue.put(message)
                logger.debug("published depth=%d", self.depth)
                return True
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                logger.error("publish rejected — circuit OPEN, timeout exceeded")
                return False
            await asyncio.sleep(min(self._probe_interval, remaining))

    async def consume(self) -> Any:
        return await self._queue.get()


async def demo_circuit_breaker() -> None:
    broker = CircuitBreakerBroker(depth_threshold=5, recovery_probe_interval=0.1)

    # fill queue past threshold
    for i in range(5):
        ok = await broker.publish({"id": i})
        assert ok

    assert broker.state == CircuitState.OPEN

    # drain
    drain_task = asyncio.create_task(_drain(broker, 5))

    # this publish should block, then succeed after drain
    ok = await broker.publish({"id": 999}, timeout=2.0)
    await drain_task
    assert ok, "publish should have succeeded after circuit recovered"
    logger.info("circuit breaker demo complete, final state=%s", broker.state)


async def _drain(broker: CircuitBreakerBroker, n: int) -> None:
    await asyncio.sleep(0.05)
    for _ in range(n):
        await broker.consume()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_circuit_breaker())
```

**Key points:**
- Three states mirror the standard circuit-breaker pattern (Closed → Open → Half-Open → Closed).
- `_evaluate()` is called at every publish attempt, not on a timer, to keep the logic simple.
- `timeout` parameter prevents infinite blocking if the circuit never recovers.

---

## Exercise 4: Throughput Comparison — Sync vs Async

Measure messages/sec for synchronous (serial) processing vs async (concurrent) for 1 000 messages.

```python
import asyncio
import logging
import queue
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

N = 1_000


# --- synchronous baseline ---

def sync_producer(q: queue.Queue[dict], n: int) -> None:
    for i in range(n):
        q.put({"id": i})


def sync_consumer(q: queue.Queue[dict], n: int) -> None:
    received = 0
    while received < n:
        msg = q.get(timeout=5)
        received += 1


def measure_sync() -> float:
    q: queue.Queue[dict] = queue.Queue()
    start = time.perf_counter()
    p = threading.Thread(target=sync_producer, args=(q, N))
    c = threading.Thread(target=sync_consumer, args=(q, N))
    p.start(); c.start()
    p.join(); c.join()
    elapsed = time.perf_counter() - start
    return N / elapsed


# --- async baseline ---

async def async_producer(q: asyncio.Queue[dict], n: int) -> None:
    for i in range(n):
        await q.put({"id": i})


async def async_consumer(q: asyncio.Queue[dict], n: int) -> None:
    for _ in range(n):
        await q.get()
        q.task_done()


async def measure_async() -> float:
    q: asyncio.Queue[dict] = asyncio.Queue()
    start = time.perf_counter()
    await asyncio.gather(async_producer(q, N), async_consumer(q, N))
    elapsed = time.perf_counter() - start
    return N / elapsed


def run_comparison() -> None:
    sync_tps = measure_sync()
    async_tps = asyncio.run(measure_async())
    logger.info("sync  throughput: %8.0f msg/sec", sync_tps)
    logger.info("async throughput: %8.0f msg/sec", async_tps)
    logger.info(
        "async is %.1fx %s than sync",
        max(async_tps, sync_tps) / min(async_tps, sync_tps),
        "faster" if async_tps > sync_tps else "slower",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_comparison()
```

**Expected results (typical on a modern laptop):**

| Mode | ~msg/sec |
|------|----------|
| sync (threading + stdlib queue) | 50 000–200 000 |
| async (asyncio) | 200 000–800 000 |

Async wins because there is no OS thread-switch overhead; all concurrency lives in a single thread.

---

## Exercise 5: Consumer Timeout

Stop waiting after N seconds if no messages arrive.

```python
import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TimeoutConsumer:
    def __init__(self, idle_timeout: float = 5.0) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._idle_timeout = idle_timeout

    async def publish(self, message: Any) -> None:
        await self._queue.put(message)

    async def consume_with_timeout(self) -> Any | None:
        """
        Returns next message, or None if idle_timeout elapses with no message.
        """
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=self._idle_timeout)
        except asyncio.TimeoutError:
            logger.info("consumer timed out after %.1fs with no messages", self._idle_timeout)
            return None

    async def consume_all(self) -> list[Any]:
        """Drain queue until idle_timeout elapses with nothing to consume."""
        results: list[Any] = []
        while True:
            msg = await self.consume_with_timeout()
            if msg is None:
                break
            results.append(msg)
            logger.debug("consumed: %s", msg)
        return results


async def demo_timeout() -> None:
    broker = TimeoutConsumer(idle_timeout=0.2)

    # publish 3 messages with short gaps
    for i in range(3):
        await broker.publish({"id": i})
        await asyncio.sleep(0.05)

    # consumer should get all 3 then timeout
    messages = await broker.consume_all()
    assert len(messages) == 3, messages
    logger.info("consumed %d messages, then stopped on timeout", len(messages))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_timeout())
```

**Key points:**
- `asyncio.wait_for` is the idiomatic way to add a timeout to any awaitable.
- `consume_all` uses the `None` sentinel return to naturally terminate the loop — no `StopIteration` or exception propagation needed.
- Adjust `idle_timeout` per use-case: short for real-time processing, longer for batch jobs.
