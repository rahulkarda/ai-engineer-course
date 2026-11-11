# Phase 6 · Week 02 · Day 1 — Async Messaging vs Sync Communication

> Objective: Understand why async messaging exists, when to use it, and how decoupling changes system behavior.  ·  Time: ~1 hr  ·  Prereqs: [[../week-01/day-7]]

---

## Concept

Two services need to talk. There are exactly two models for how they can do it.

**Synchronous**: Caller sends a request and blocks — it waits for the response before doing anything else. HTTP is the canonical example. The caller and callee must both be alive and responsive at the same moment.

**Asynchronous**: Caller fires a message into a broker and moves on. The callee picks it up when it is ready. The two sides never need to be alive at the same time.

That single difference has cascading consequences for throughput, resilience, and how tightly the two services are joined at the hip.

### Temporal coupling

Temporal coupling is the dependency that two components must be available simultaneously.

With HTTP between Service A and Service B:
- B is down → A's request fails → A may fail too
- B is slow → A blocks for the duration → A's own latency grows
- B is overwhelmed → A's retries add load → cascade

With a message queue:
- B is down → the message sits in the queue → no immediate failure in A
- B is slow → messages accumulate → A keeps running at full speed
- B restarts → it drains the backlog at its own pace

The queue absorbs the temporal mismatch. A and B are decoupled in time.

### Throughput and backpressure

A sync caller can only have as many in-flight requests as it has threads or connections. Each request ties up a slot until the response arrives.

An async producer writes to a queue at memory/network speed. The queue accepts the burst. A pool of consumers drains it at processing speed. The two rates can differ — the queue buffers the difference.

This is called **backpressure**: the system accumulates work rather than shedding it.

### Resilience tradeoffs

Async messaging gives you:
- **Durability**: a persistent broker survives restarts; messages are not lost
- **Retry decoupling**: consumers retry without the producer knowing
- **Independent scaling**: add consumers without changing the producer

It costs you:
- **Observability complexity**: a request no longer has a single call stack
- **Eventual consistency**: the consumer processes later, not now
- **Ordering challenges**: concurrent consumers may process out of order

### When async is the wrong choice

Async messaging is not universally better. Avoid it when:

- **The caller needs the result immediately** — user-facing reads, login checks, payment confirmation. You cannot render a page using "I'll tell you later."
- **The operation is simple CRUD** — a REST endpoint that reads one row and returns it has no burst, no slow consumer, nothing to decouple. A queue adds latency and complexity with no benefit.
- **End-to-end latency matters more than throughput** — a message queue adds at least one extra hop and serialization round-trip.
- **You need strong consistency** — queues give you eventual delivery, not transactions.

A rule of thumb: if the caller needs to act on the result before it can return to the user, use HTTP. If the caller just needs to record that something happened, use a queue.

---

## How it works internally

### Under the hood of a blocking HTTP call

When a Python service calls `httpx.get(url)`:
1. A socket is opened and the request bytes are written to the kernel buffer.
2. The calling thread is suspended (or in asyncio, the coroutine is suspended at `await`).
3. The OS waits for bytes to arrive on the socket.
4. The response is read, deserialized, and the thread/coroutine resumes.

The thread (or event loop slot) is occupied the entire time the wire is silent.

### Under the hood of asyncio queue

Python's `asyncio.Queue` is a coroutine-safe FIFO backed by a `collections.deque`. It lives entirely in the process — no network, no serialization. Producers call `await queue.put(item)`, which:
1. Appends the item to the deque.
2. If a consumer is blocked on `await queue.get()`, it wakes that consumer's coroutine and schedules it to resume on the event loop.

This is an in-memory analogue of a real broker like Kafka or RabbitMQ. Real brokers add:
- **Persistence**: write-ahead log on disk so messages survive restarts
- **Replication**: copies on multiple brokers for fault tolerance
- **Consumer groups**: coordinated offset tracking so multiple instances share the load
- **Acknowledgements**: explicit ack/nack so the broker knows whether to redeliver

The asyncio queue in the worked example below captures the essential decoupling without requiring external infrastructure.

### Event loop and concurrency

`asyncio` uses a single-threaded cooperative scheduler. Coroutines yield control at `await` points. When a producer calls `await queue.put()` the event loop can switch to a consumer that is waiting on `await queue.get()`. This interleaving is deterministic and visible in the execution trace — which makes it ideal for teaching async behavior.

---

## Worked example

```python
"""
day_1_async_queue.py

Demonstrates producer/consumer decoupling with Python asyncio.
No external broker needed — runs entirely in memory.

Usage:
    python day_1_async_queue.py
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
)
log = logging.getLogger("day1")

# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

@dataclass
class OrderEvent:
    order_id: str
    item: str
    quantity: int
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Simulated sync caller (blocks while waiting for a slow "payment service")
# ---------------------------------------------------------------------------

async def sync_place_order(order: OrderEvent, payment_latency_s: float = 0.5) -> str:
    """Mimic a synchronous HTTP call to a payment service."""
    log.info("[SYNC ] Calling payment service for order %s — blocking...", order.order_id)
    await asyncio.sleep(payment_latency_s)          # simulates network round-trip
    log.info("[SYNC ] Payment confirmed for order %s", order.order_id)
    return "confirmed"


# ---------------------------------------------------------------------------
# Async producer: fires message and moves on immediately
# ---------------------------------------------------------------------------

async def async_producer(queue: asyncio.Queue[OrderEvent], orders: list[OrderEvent]) -> None:
    for order in orders:
        log.info(
            "[PROD ] Enqueuing order %s (queue depth: %d)",
            order.order_id,
            queue.qsize(),
        )
        await queue.put(order)
        # Producer continues immediately — no wait for consumer
        await asyncio.sleep(0.05)   # simulate time between orders being placed
    log.info("[PROD ] All orders enqueued. Producer done.")


# ---------------------------------------------------------------------------
# Async consumer: drains queue independently of producer
# ---------------------------------------------------------------------------

async def async_consumer(
    consumer_id: int,
    queue: asyncio.Queue[OrderEvent],
    processing_latency_s: float = 0.3,
) -> None:
    while True:
        try:
            # Wait for an item, but yield control while the queue is empty
            order = await asyncio.wait_for(queue.get(), timeout=2.0)
        except TimeoutError:
            log.info("[CON-%d] No more orders after timeout. Exiting.", consumer_id)
            break

        log.info("[CON-%d] Processing order %s", consumer_id, order.order_id)
        await asyncio.sleep(processing_latency_s)   # simulate DB write / fulfillment
        log.info("[CON-%d] Done with order %s", consumer_id, order.order_id)
        queue.task_done()


# ---------------------------------------------------------------------------
# Comparison: sync vs async throughput
# ---------------------------------------------------------------------------

async def demo_sync(orders: list[OrderEvent]) -> float:
    """Process orders one at a time, each blocking until payment returns."""
    log.info("=== SYNC DEMO ===")
    start = time.perf_counter()
    for order in orders:
        await sync_place_order(order, payment_latency_s=0.3)
    elapsed = time.perf_counter() - start
    log.info("=== SYNC done in %.2f s ===\n", elapsed)
    return elapsed


async def demo_async(orders: list[OrderEvent], num_consumers: int = 2) -> float:
    """Enqueue all orders; consumers process them concurrently."""
    log.info("=== ASYNC DEMO (consumers=%d) ===", num_consumers)
    start = time.perf_counter()

    queue: asyncio.Queue[OrderEvent] = asyncio.Queue()

    # Launch producer and consumers as concurrent tasks
    producer_task = asyncio.create_task(async_producer(queue, orders))
    consumer_tasks = [
        asyncio.create_task(async_consumer(i, queue, processing_latency_s=0.3))
        for i in range(num_consumers)
    ]

    await producer_task
    await queue.join()          # block until all items are task_done()

    for t in consumer_tasks:
        t.cancel()

    elapsed = time.perf_counter() - start
    log.info("=== ASYNC done in %.2f s ===\n", elapsed)
    return elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    orders = [
        OrderEvent(order_id=f"ORD-{i:03d}", item="widget", quantity=i)
        for i in range(1, 7)
    ]

    sync_time = await demo_sync(orders[:])
    async_time = await demo_async(orders[:], num_consumers=2)

    log.info(
        "Throughput gain: sync=%.2f s, async=%.2f s, speedup=%.1fx",
        sync_time,
        async_time,
        sync_time / async_time,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

**What to observe when you run this:**

1. In the sync demo, orders process strictly sequentially. Total time ≈ N × latency.
2. In the async demo, the producer enqueues all orders quickly. Consumers run concurrently. Total time ≈ ceil(N / consumers) × latency.
3. The producer log lines and consumer log lines interleave — the producer does not wait for the consumer to finish before moving to the next order.

---

## Exercises

1. Modify `demo_async` to use 3 consumers instead of 2. Measure the speedup. What happens to speedup as you add more consumers relative to the number of orders?

2. Add a `max_queue_size` to the `asyncio.Queue` constructor (e.g., `asyncio.Queue(maxsize=2)`). Now run the async demo with a slow consumer (`processing_latency_s=1.0`) and a fast producer (`asyncio.sleep(0.05)`). Observe: what happens when the queue is full? This is **backpressure** in action.

3. Simulate a "consumer crash": raise an exception inside `async_consumer` after processing 3 orders. What happens to the remaining messages? How would you recover them? (Hint: `queue.task_done()` is never called for the crashed item — what does `queue.join()` do?)

4. Add a `total_processing_time` metric: record the time between `OrderEvent.created_at` and when the consumer finishes. Compare the P50 and P99 latency between sync and async modes. When is async latency actually *worse* for a single request?

5. Rewrite `async_producer` to batch-enqueue orders: produce 3 at once, then sleep 0.2 s, then produce the next 3. Does this change consumer behavior? Why or why not?

(Solutions: ../solutions/week-02/day-1.md)

---

## Self-check

- [ ] I can explain temporal coupling and why it makes synchronous systems fragile under load.
- [ ] I can describe what a message queue does during a consumer outage.
- [ ] I can list three scenarios where async messaging is the wrong choice.
- [ ] I can implement a basic producer/consumer with `asyncio.Queue` without reference.
- [ ] I can measure and explain the throughput difference between sync and async processing.

---

## Supplementary links

- Python `asyncio.Queue` docs: https://docs.python.org/3/library/asyncio-queue.html
- Martin Fowler — Enterprise Integration Patterns (messaging patterns): https://www.enterpriseintegrationpatterns.com/patterns/messaging/
- AWS — Synchronous vs Asynchronous Microservices: https://aws.amazon.com/compare/the-difference-between-synchronous-asynchronous/
- Cloudflare — What is a message queue?: https://www.cloudflare.com/learning/performance/what-is-a-message-queue/
- Python `asyncio` — Tasks and coroutines: https://docs.python.org/3/library/asyncio-task.html
