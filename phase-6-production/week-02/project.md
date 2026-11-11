# Phase 6 · Week 02 — Project: Event-Driven Order Processing Pipeline

> Objective: Build a production-ready (fully mockable) event-driven order processing pipeline that exercises every concept from Week 2 — async messaging, pub/sub fan-out, at-least-once delivery with idempotency, DLQ, backpressure, and a pluggable broker abstraction.  ·  Time: ~3 hr  ·  Prereqs: all week days

---

## Spec

### 1. Broker abstraction layer

```
src/messaging/broker.py
```

Define an `AbstractBroker` ABC with the following interface:

```python
class AbstractBroker(ABC):
    @abstractmethod
    def publish(self, topic: str, event: BaseModel) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, handler: Callable[[BaseModel], None]) -> None: ...

    @abstractmethod
    def queue_depth(self, topic: str) -> int: ...
```

Implement two concrete classes:

- **`InMemoryBroker`** — synchronous, in-process; used in all tests. Stores events in `collections.deque` per topic. Delivers to all subscribers on `publish`. Retries failed handlers up to `max_retries` (default 3) before routing to `"dlq.<topic>"`.
- **`KafkaBroker` (stub)** — skeleton only; constructor accepts `bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")`. Methods raise `NotImplementedError` with a docstring describing what the real implementation would do. No real Kafka connection required.

Both inherit from `AbstractBroker`. All code under `src/messaging/`.

### 2. Pydantic v2 models

```
src/models/events.py
```

```python
class OrderItem(BaseModel):
    sku: str
    quantity: int
    unit_price_cents: int

class OrderEvent(BaseModel):
    event_id: UUID
    order_id: str
    customer_id: str
    items: list[OrderItem]
    total_cents: int
    timestamp: datetime

class InventoryEvent(BaseModel):
    event_id: UUID
    order_id: str          # links back to OrderEvent
    sku: str
    quantity_reserved: int
    timestamp: datetime

class NotificationEvent(BaseModel):
    event_id: UUID
    customer_id: str
    message: str
    channel: Literal["email", "sms", "push"]
    timestamp: datetime
```

All fields required. Use `model_validator` or `field_validator` where appropriate (e.g. `total_cents > 0`, `quantity >= 1`).

### 3. Producers

```
src/producers/order_producer.py
```

`OrderProducer(broker: AbstractBroker)`:
- `place_order(order: OrderEvent) -> None` — publishes to topic `"orders.placed"`
- Logs at `INFO` level: `"Published order {order.order_id} to orders.placed"`

### 4. Consumers

#### InventoryConsumer — at-least-once with idempotency

```
src/consumers/inventory_consumer.py
```

`InventoryConsumer(broker: AbstractBroker)`:
- Subscribes to `"orders.placed"` on construction
- Handler processes each `OrderEvent`: for each `OrderItem`, publishes an `InventoryEvent` to `"inventory.reserved"`
- **Idempotent by `order_id`**: tracks processed order IDs in a set; if already seen, logs `WARN "Duplicate order_id {order_id} — skipping"` and returns without reprocessing
- Raises `ValueError` for orders with `total_cents <= 0` (simulates a validation failure to test DLQ)

#### NotificationConsumer — fan-out subscriber

```
src/consumers/notification_consumer.py
```

`NotificationConsumer(broker: AbstractBroker)`:
- Subscribes **independently** to `"orders.placed"` (separate subscription from `InventoryConsumer`)
- For each `OrderEvent`, publishes a `NotificationEvent` to `"notifications.send"` with `channel="email"` and a message summarising the order
- Stores all sent notifications in `self.sent: list[NotificationEvent]` for test assertions

### 5. DLQ handler

```
src/consumers/dlq_consumer.py
```

`DLQConsumer(broker: AbstractBroker)`:
- Subscribes to `"dlq.orders.placed"`
- Logs each failed event at `ERROR` level with the original payload
- Stores failed events in `self.failed: list[BaseModel]`

### 6. Backpressure

`InMemoryBroker` must enforce a configurable `backpressure_threshold: int = 100`.

When `queue_depth(topic) >= backpressure_threshold`:
- `publish()` raises `BackpressureError(topic, depth)` (custom exception, subclass of `RuntimeError`)
- Log at `WARNING`: `"Backpressure triggered on {topic}: depth={depth}"`

Consumers should catch `BackpressureError` from downstream `publish` calls and pause by re-queuing the original event (or in tests, simply propagating the error to verify it's raised).

### 7. Test suite

```
tests/unit/test_broker.py
tests/unit/test_inventory_consumer.py
tests/unit/test_notification_consumer.py
tests/unit/test_dlq.py
tests/unit/test_backpressure.py
```

All tests use `InMemoryBroker`. No real Kafka, no network calls.

---

## Acceptance criteria

- [ ] `AbstractBroker`, `InMemoryBroker`, and `KafkaBroker` (stub) are in `src/messaging/broker.py`; consumers and producers accept any `AbstractBroker` instance
- [ ] Publishing one `OrderEvent` triggers both `InventoryConsumer` and `NotificationConsumer` (fan-out verified in test)
- [ ] `InventoryConsumer` processes a duplicate `order_id` only once — second call emits a WARN and does nothing
- [ ] A consumer handler that raises is retried exactly `max_retries` times before routing to the DLQ
- [ ] Failed event appears on `"dlq.orders.placed"` and is captured by `DLQConsumer`
- [ ] `BackpressureError` is raised when `queue_depth >= backpressure_threshold`; threshold is configurable at broker construction
- [ ] All Pydantic models have field validators; invalid data raises `ValidationError` before reaching the broker
- [ ] `os.getenv` used for all configuration (Kafka bootstrap, thresholds); no hardcoded values in source
- [ ] `logging` used throughout — no `print` statements anywhere in `src/`
- [ ] `pytest -q` passes with zero failures; at least one test per acceptance criterion above

---

## Deliverables

- [ ] `src/messaging/broker.py` — `AbstractBroker`, `InMemoryBroker`, `KafkaBroker` stub, `BackpressureError`
- [ ] `src/models/events.py` — all four Pydantic models with validators
- [ ] `src/producers/order_producer.py` — `OrderProducer`
- [ ] `src/consumers/inventory_consumer.py` — `InventoryConsumer` (idempotent)
- [ ] `src/consumers/notification_consumer.py` — `NotificationConsumer`
- [ ] `src/consumers/dlq_consumer.py` — `DLQConsumer`
- [ ] `tests/unit/` — full test suite, all passing
- [ ] `.env.example` with `KAFKA_BOOTSTRAP_SERVERS`, `BACKPRESSURE_THRESHOLD`
- [ ] `pyproject.toml` with `pydantic>=2.0`, `pytest` as dev dependency

---

## Hints

**Broker fan-out pattern**
`InMemoryBroker` should store `subscribers: dict[str, list[Callable]]`. On `publish`, iterate all subscribers for that topic. This is the core of pub/sub.

**Retry + DLQ in InMemoryBroker**
```python
for attempt in range(self.max_retries):
    try:
        handler(event)
        break
    except Exception as exc:
        logger.warning("Handler failed attempt %d/%d: %s", attempt + 1, self.max_retries, exc)
else:
    logger.error("Routing to DLQ after %d failures", self.max_retries)
    self.publish(f"dlq.{topic}", event)
```

**Idempotency pattern**
```python
def _handle(self, event: OrderEvent) -> None:
    if event.order_id in self._processed_ids:
        logger.warning("Duplicate order_id %s — skipping", event.order_id)
        return
    self._processed_ids.add(event.order_id)
    # ... process
```

**KafkaBroker stub structure**
Accept `bootstrap_servers` from `os.getenv`. In each method, raise `NotImplementedError` with a comment describing the real implementation (e.g. `# real impl: confluent_kafka.Producer.produce(topic, value.model_dump_json())`). This keeps the abstraction honest without needing a running broker.

**Testing backpressure**
Prepopulate the broker's internal queue deque with dummy items to hit the threshold, then assert `BackpressureError` is raised on the next `publish`.

**Type hints**
Use `Protocol` or `ABC` for `AbstractBroker`. Inject it everywhere — never instantiate a concrete broker inside a consumer or producer. This is what makes unit testing with `InMemoryBroker` effortless.

---

## Extension challenges (fast finishers)

1. **Async variant** — rewrite `InMemoryBroker` as `AsyncInMemoryBroker` using `asyncio.Queue` and `async def publish/subscribe`. Update producers and consumers to be `async`. Test with `pytest-asyncio`.
2. **Schema registry mock** — add a `SchemaRegistry` class that validates event JSON against a stored schema before `publish`. Raise `SchemaValidationError` on mismatch.
3. **Metrics** — add a `BrokerMetrics` dataclass that tracks `published_count`, `consumed_count`, `dlq_count`, `retry_count` per topic. Expose via a `metrics(topic: str) -> BrokerMetrics` method on `AbstractBroker`.
4. **Replay** — add a `replay(topic: str, from_offset: int) -> None` method to `InMemoryBroker` that re-delivers events from a stored log starting at `from_offset`. This simulates Kafka's offset-based replay.
5. **Partition routing** — extend `InMemoryBroker` to accept a `partition_key_fn: Callable[[BaseModel], str]`. Hash the key to N virtual partitions and ensure events with the same key always go to the same partition's consumer (ordered delivery per key).
