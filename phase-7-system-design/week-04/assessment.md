# Week 04 Assessment

## Instructions

Time limit: 90 minutes. All code must be Python 3.12+ with full type hints. No external dependencies beyond `pydantic` and `pytest`. You may reference your notes but not your day-6 build files.

---

## Conceptual (interview-style)

Answer each question in 3–5 sentences. Precision matters more than length.

1. **Ordering vs throughput tradeoff**: A Kafka topic has 1 partition. A team wants to increase throughput by adding partitions. The topic stores user session events and consumers need all events for a single user in order. Describe the strategy and its limitation.

2. **Idempotency failure scenario**: A consumer processes a payment event, inserts a row into a `payments` table, then stores the `event_id` in a `processed_events` table — as two separate DB commits. Describe the failure window and what goes wrong. Propose a fix in one sentence.

3. **Saga compensation ordering**: A saga has 5 steps: `A → B → C (pivot) → D → E`. Step E fails. List the compensations that must fire and the order they must fire in. Why does the pivot change the compensation scope?

4. **Choreography vs orchestration — choose one**: You are building a mortgage application workflow with 12 steps, some running in parallel, some conditional on previous results, and SLA tracking per step required. Which coordination pattern do you choose and why?

5. **Snapshot strategy**: An aggregate has 50,000 events in its history. A snapshot was taken at event 45,000. On reconstitution, how many events does the system replay? What is the risk of the snapshot being stale or corrupted, and how do you mitigate it?

6. **Windowed processing — late arrivals**: A tumbling 60-second window closes at t=60s. A record with timestamp t=58s arrives at wall-clock time t=75s (15s late, due to a slow producer). Describe two strategies for handling this and the tradeoff between them.

7. **Outbox pattern**: Explain in one paragraph why publishing a Kafka event inside a DB transaction (via a Kafka producer inside the transaction boundary) is NOT equivalent to the outbox pattern, and what failure mode the outbox pattern prevents that the direct approach does not.

---

## Coding challenge

### Part A: Extend InventoryItem (30 min)

Starting from the `InventoryItem` aggregate built in Day 6, add the following without modifying existing event handlers:

1. Add a command `adjust(delta: int)` that emits `StockAdjusted` with `{"delta": delta, "reason": str}`. Delta can be positive or negative. Guard: `on_hand + delta` must not go below `reserved`.

2. Add a command `mark_damaged(qty: int)` that emits `StockDamagedAndRemoved` with `{"qty": qty}`. Guard: `qty` must not exceed `available`.

3. Add a `snapshot_state` property that returns `{"sku": ..., "on_hand": ..., "reserved": ..., "version": self.sequence}`.

Write **6 pytest tests** covering:
- `adjust` increases and decreases `on_hand`
- `adjust` raises when result would breach `reserved`
- `mark_damaged` correctly reduces `on_hand` and `available`
- `mark_damaged` raises when `qty > available`
- Full reconstitution round-trip including the new event types
- Snapshot includes the `version` field

### Part B: Multi-topic stream processor (30 min)

Extend the `TumblingWindowProcessor` (or create a new class `MultiTopicWindowProcessor`) to:

1. Accept records from **any topic** (no filter).
2. Maintain **a separate window per topic**.
3. `flush_all()` flushes all open topic windows.
4. `closed_windows(topic: str | None = None)` returns all windows when `topic=None`, or windows for a specific topic when provided.

Write **5 pytest tests** covering:
- Two topics produce two separate window sets
- Each topic's windows have correct `record_count` and `total_value`
- `flush_all` closes all open windows across all topics
- Filtering by topic in `closed_windows` returns only that topic's results
- A topic with a single record produces one window on `flush_all`

### Part C: Saga compensation chain (30 min)

Implement a `SagaOrchestrator` class (orchestration-based, not choreography) for a 3-step payment saga:

```
Step 1: authorize_payment(order_id, amount) -> bool
Step 2: reserve_inventory(order_id, sku, qty) -> bool
Step 3: schedule_shipping(order_id) -> bool
```

Requirements:
- Constructor accepts three callable arguments: `payment_fn`, `inventory_fn`, `shipping_fn` (each takes the relevant args and returns `bool`)
- `run(order_id, amount, sku, qty) -> SagaResult`
- `SagaResult` is a Pydantic model with fields: `order_id`, `status` (`"completed"` or `"compensated"`), `completed_steps: list[str]`, `failed_step: str | None`, `compensations_run: list[str]`
- If step 2 fails: compensate step 1 by calling `compensate_payment_fn`
- If step 3 fails: compensate steps 2 and 1 in reverse order
- Constructor also accepts `compensate_payment_fn`, `compensate_inventory_fn`

Write **6 pytest tests** covering:
- Happy path: all steps succeed
- Step 1 failure: no compensations run, status = compensated
- Step 2 failure: payment compensated, status = compensated
- Step 3 failure: both payment and inventory compensated, in reverse order
- Compensation functions receive correct arguments
- `SagaResult.completed_steps` accurately lists which steps ran successfully

---

## Rubric (pass >= 75%)

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | Code runs without modification; no copy-paste from day-6 files |
| Conceptual understanding | 25% | Conceptual answers are precise, not vague; tradeoffs named |
| Debugging & problem-solving | 20% | Tests pass; edge cases handled; guards raise correct exception types |
| Code quality & testing | 15% | Type hints everywhere; `logging` not `print`; Pydantic models used; no hardcoded values |

### Scoring breakdown

| Score | Grade |
|---|---|
| 90–100% | Distinction — ready to lead design reviews |
| 75–89%  | Pass — solid understanding, minor gaps |
| 60–74%  | Near-pass — review day 5 (sagas) and day 6 (event store) |
| < 60%   | Repeat week — schedule 1:1 with course mentor |

Answer key: ../solutions/week-04/assessment.md
