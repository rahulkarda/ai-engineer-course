# Day 2 Solutions: Queues vs Pub/Sub

## Exercise 1: Routing Keys

Extend `InMemoryBroker` so messages are only delivered to subscribers whose routing key matches.

```python
import asyncio
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class RoutingBroker:
    """
    Publish/Subscribe broker with routing keys.
    Subscribers declare the routing key they want; publishers attach one.
    Exact-match only (pattern matching is Exercise 5).
    """

    def __init__(self) -> None:
        # routing_key -> list of queues
        self._subscribers: dict[str, list[asyncio.Queue[Any]]] = defaultdict(list)

    def subscribe(self, routing_key: str) -> asyncio.Queue[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue()
        self._subscribers[routing_key].append(q)
        logger.debug("new subscriber for key=%r (total=%d)", routing_key, len(self._subscribers[routing_key]))
        return q

    async def publish(self, routing_key: str, message: Any) -> int:
        """Returns number of subscribers that received the message."""
        targets = self._subscribers.get(routing_key, [])
        for q in targets:
            await q.put(message)
        logger.debug("published key=%r to %d subscriber(s)", routing_key, len(targets))
        return len(targets)


async def demo_routing() -> None:
    broker = RoutingBroker()

    orders_q = broker.subscribe("orders")
    payments_q = broker.subscribe("payments")
    orders_q2 = broker.subscribe("orders")  # two subscribers to same key

    await broker.publish("orders", {"event": "order_created", "id": 1})
    await broker.publish("payments", {"event": "payment_received", "id": 42})
    await broker.publish("inventory", {"event": "stock_updated"})  # no subscribers

    msg = await orders_q.get()
    assert msg["event"] == "order_created"

    msg2 = await orders_q2.get()
    assert msg2["event"] == "order_created"

    pay_msg = await payments_q.get()
    assert pay_msg["event"] == "payment_received"

    assert orders_q.empty() and orders_q2.empty() and payments_q.empty()
    logger.info("routing key demo passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_routing())
```

**Key points:**
- Each `subscribe()` call returns a private `asyncio.Queue`; the broker fans out by iterating over all queues for a given routing key.
- Unmatched routing keys are silently dropped (log at DEBUG). Production systems may want a dead-letter queue here.

---

## Exercise 2: Competing Consumers (Exactly-Once Delivery per Message)

Three consumers share one queue; each message is processed by exactly one of them.

```python
import asyncio
import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


class SharedQueue:
    """Single queue consumed competitively by multiple workers."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def put(self, item: Any) -> None:
        await self._queue.put(item)

    async def get(self) -> Any:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()


async def worker(worker_id: int, q: SharedQueue, results: Counter) -> None:
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=0.5)
        except asyncio.TimeoutError:
            break  # no more messages, exit
        results[msg["id"]] += 1
        logger.debug("worker %d processed id=%d", worker_id, msg["id"])
        q.task_done()


async def demo_competing_consumers() -> None:
    q = SharedQueue()
    n = 30

    for i in range(n):
        await q.put({"id": i})

    results: Counter = Counter()
    workers = [asyncio.create_task(worker(i, q, results)) for i in range(3)]
    await asyncio.gather(*workers)

    # every message id should appear exactly once
    assert len(results) == n, f"expected {n} unique ids, got {len(results)}"
    assert all(v == 1 for v in results.values()), f"duplicate processing: {results.most_common(3)}"
    logger.info("competing consumer demo passed — %d messages, each processed exactly once", n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_competing_consumers())
```

**Key points:**
- `asyncio.Queue` is the natural competing-consumer primitive: `get()` atomically removes one item, so two coroutines can never receive the same message.
- Workers use a timeout to self-terminate instead of needing a sentinel value.

---

## Exercise 3: Subscription Filters

Consumers receive only messages matching a predicate.

```python
import asyncio
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

Predicate = Callable[[Any], bool]


class FilteredBroker:
    def __init__(self) -> None:
        self._subscriptions: list[tuple[Predicate, asyncio.Queue[Any]]] = []

    def subscribe(self, predicate: Predicate) -> asyncio.Queue[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue()
        self._subscriptions.append((predicate, q))
        return q

    async def publish(self, message: Any) -> None:
        for predicate, q in self._subscriptions:
            if predicate(message):
                await q.put(message)


async def demo_filters() -> None:
    broker = FilteredBroker()

    high_priority_q = broker.subscribe(lambda m: m.get("priority", 0) > 5)
    low_priority_q = broker.subscribe(lambda m: m.get("priority", 0) <= 5)
    errors_q = broker.subscribe(lambda m: m.get("level") == "error")

    messages = [
        {"id": 1, "priority": 8, "level": "info"},
        {"id": 2, "priority": 3, "level": "info"},
        {"id": 3, "priority": 9, "level": "error"},
        {"id": 4, "priority": 1, "level": "error"},
    ]

    for m in messages:
        await broker.publish(m)

    # high priority: ids 1, 3
    assert high_priority_q.qsize() == 2
    # low priority: ids 2, 4
    assert low_priority_q.qsize() == 2
    # errors: ids 3, 4
    assert errors_q.qsize() == 2

    # a message can match multiple subscriptions (fan-out)
    high_ids = {(await high_priority_q.get())["id"] for _ in range(2)}
    assert high_ids == {1, 3}
    logger.info("subscription filter demo passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_filters())
```

**Key points:**
- Predicates are plain callables — no DSL, easy to unit-test.
- A message can satisfy multiple predicates and will be delivered to all matching subscribers (fan-out semantics).
- For performance-sensitive paths, compile predicates to rule engines (e.g., `rule-engine` lib) rather than re-evaluating lambdas per message.

---

## Exercise 4: Message TTL (Time-to-Live)

Messages expire if not consumed within N seconds.

```python
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TTLMessage:
    payload: Any
    expires_at: float  # monotonic timestamp


class TTLQueue:
    """Queue that silently drops expired messages on consume."""

    def __init__(self, ttl_seconds: float = 5.0) -> None:
        self._ttl = ttl_seconds
        self._items: asyncio.Queue[TTLMessage] = asyncio.Queue()
        self._expired_count: int = 0

    async def put(self, payload: Any) -> None:
        msg = TTLMessage(payload=payload, expires_at=time.monotonic() + self._ttl)
        await self._items.put(msg)

    async def get(self) -> Any | None:
        """
        Returns next live message payload, or None if all pending messages are expired.
        Expired messages are discarded.
        """
        while not self._items.empty():
            msg = await self._items.get()
            self._items.task_done()
            if time.monotonic() < msg.expires_at:
                return msg.payload
            self._expired_count += 1
            logger.debug("dropped expired message (payload=%s)", msg.payload)
        return None

    async def get_blocking(self, poll_interval: float = 0.05) -> Any | None:
        """
        Blocks until a live message arrives or idle_timeout is passed.
        Returns None only if no live messages remain.
        """
        while True:
            result = await self.get()
            if result is not None:
                return result
            if self._items.empty():
                return None
            await asyncio.sleep(poll_interval)

    @property
    def expired_count(self) -> int:
        return self._expired_count


async def demo_ttl() -> None:
    q = TTLQueue(ttl_seconds=0.1)

    await q.put({"id": 1})
    await q.put({"id": 2})

    # consume immediately — both should arrive
    m1 = await q.get()
    assert m1 is not None and m1["id"] == 1

    # wait for TTL to expire
    await asyncio.sleep(0.2)

    await q.put({"id": 3})  # fresh message
    m2 = await q.get()      # id=2 is expired, id=3 is live
    assert m2 is not None and m2["id"] == 3, f"got {m2}"
    assert q.expired_count == 1
    logger.info("TTL demo passed — expired_count=%d", q.expired_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_ttl())
```

**Key points:**
- Expiry is checked at *consume* time, not via a background reaper. This is sufficient for correctness and avoids the complexity of a timer per message.
- `expired_count` allows callers to monitor how many messages are being dropped (feed this to your metrics system).

---

## Exercise 5: Topic Pattern Exchange Router

Route messages based on topic patterns — `"orders.*"` matches `"orders.created"` and `"orders.cancelled"`.

```python
import asyncio
import fnmatch
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class TopicExchange:
    """
    Pub/Sub exchange supporting wildcard topic patterns.

    Pattern syntax (fnmatch):
      *  — matches any sequence of characters within one segment
      ** — not supported here; use multiple subscriptions instead

    Examples:
      "orders.*"   matches "orders.created", "orders.cancelled"
      "*.failed"   matches "payment.failed", "order.failed"
      "events"     exact match only
    """

    def __init__(self) -> None:
        # pattern -> list of subscriber queues
        self._bindings: dict[str, list[asyncio.Queue[Any]]] = defaultdict(list)

    def bind(self, pattern: str) -> asyncio.Queue[Any]:
        """Subscribe to all topics matching pattern."""
        q: asyncio.Queue[Any] = asyncio.Queue()
        self._bindings[pattern].append(q)
        logger.debug("bound pattern=%r", pattern)
        return q

    async def publish(self, topic: str, message: Any) -> int:
        """Deliver message to all subscribers whose pattern matches topic."""
        delivered = 0
        for pattern, queues in self._bindings.items():
            if fnmatch.fnmatch(topic, pattern):
                for q in queues:
                    await q.put({"topic": topic, "message": message})
                    delivered += 1
        logger.debug("topic=%r delivered to %d subscriber(s)", topic, delivered)
        return delivered


async def demo_exchange() -> None:
    exchange = TopicExchange()

    orders_q = exchange.bind("orders.*")
    payments_q = exchange.bind("payments.*")
    all_q = exchange.bind("*.*")           # matches any two-segment topic
    exact_q = exchange.bind("audit")       # exact match

    await exchange.publish("orders.created", {"order_id": 1})
    await exchange.publish("orders.cancelled", {"order_id": 2})
    await exchange.publish("payments.received", {"amount": 99.0})
    await exchange.publish("audit", {"action": "login"})
    await exchange.publish("unrouted.event", {})  # matches only "*.*"

    assert orders_q.qsize() == 2          # orders.created, orders.cancelled
    assert payments_q.qsize() == 1        # payments.received
    assert all_q.qsize() == 4             # all two-segment topics
    assert exact_q.qsize() == 1           # audit only

    msg = await orders_q.get()
    assert msg["topic"] == "orders.created"
    logger.info("topic exchange demo passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_exchange())
```

**Key points:**
- `fnmatch.fnmatch` provides shell-style glob matching with zero external dependencies.
- The exchange wraps each delivered payload with `{"topic": ..., "message": ...}` so consumers know what topic triggered the delivery.
- For hierarchical topics (AMQP `#` wildcard), extend by splitting on `.` and matching segment-by-segment.
