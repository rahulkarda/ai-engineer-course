# Phase 7 · Week 09 · Day 6 (Saturday) — Deep Practice: Diagnose, Fix & Post-Mortem Three Incident Scenarios

> Objective: Work through three realistic production incident scenarios end-to-end: read symptoms, diagnose root cause, implement a fix with tests, and write a post-mortem.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Overview

This session is scenario-driven. For each of the three incidents you will:

1. Read the provided symptoms, logs, and metrics snapshot
2. Diagnose using the five-step method from Day 1
3. Locate the seeded bug in the provided code
4. Fix the bug — minimal, surgical change
5. Write tests that would have caught the bug
6. Draft a two-paragraph post-mortem

Work through all three. Estimated time: 80 min each.

---

## Scenario A — "The Leaking Agent" (~80 min)

### Symptoms

```
Alert: process RSS memory grew from 512 MB to 6.1 GB over 4 hours
Service: order-summarization agent (LangGraph pipeline)
Deployment: no recent deploys — same code for 3 weeks
Metrics: request rate steady at 120 RPM; no latency change
```

### Log excerpt

```json
{"ts":"2026-11-10T14:01:00Z","level":"info","msg":"summarize_order","order_id":"ORD-1001","tokens":312}
{"ts":"2026-11-10T14:01:01Z","level":"info","msg":"summarize_order","order_id":"ORD-1002","tokens":318}
{"ts":"2026-11-10T18:00:00Z","level":"warning","msg":"gc_collection","gen2_count":841,"collected":0}
{"ts":"2026-11-10T18:00:01Z","level":"error","msg":"OOMKilled"}
```

### Buggy code to diagnose

```python
"""scenario_a_buggy.py — find and fix the memory leak."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent")


# BUG IS SOMEWHERE IN THIS FILE — diagnose it.

@dataclass
class OrderSummaryAgent:
    model: str = "claude-opus-5-5"  # never used — mock only
    _history: list[dict[str, Any]] = field(default_factory=list)

    def summarize(self, order_id: str, order_data: dict[str, Any]) -> str:
        """Summarize an order for customer support."""
        # Build prompt with context
        prompt = {
            "role": "user",
            "content": f"Summarize order {order_id}: {order_data}",
        }
        self._history.append(prompt)  # <<< stores every prompt forever

        # Simulate LLM response (mocked)
        response = f"Order {order_id}: {len(order_data)} items, total ${order_data.get('total', 0):.2f}"

        response_msg = {"role": "assistant", "content": response}
        self._history.append(response_msg)  # <<< appends response too

        return response

    def get_context_window(self) -> list[dict]:
        """Returns current conversation context."""
        return self._history  # <<< returns unbounded list


def simulate_load(n_orders: int = 10_000) -> None:
    agent = OrderSummaryAgent()
    for i in range(n_orders):
        order_data = {
            "items": [{"sku": f"SKU-{j}", "qty": 1, "price": 9.99} for j in range(10)],
            "total": 99.90,
            "customer_id": f"CUST-{i % 500}",  # 500 unique customers
        }
        agent.summarize(f"ORD-{i:06d}", order_data)

    history_mb = len(str(agent._history).encode()) / (1024 ** 2)
    log.info("History size after %d orders: %.1f MB", n_orders, history_mb)
    log.info("History entries: %d", len(agent._history))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    simulate_load()
```

### Your tasks

1. **Identify** the bug: which lines cause the leak, and why the gc_collection log shows `collected=0`.
2. **Fix** `OrderSummaryAgent` so history does not grow unbounded. The agent is stateless across orders — conversation history should not persist between `summarize()` calls.
3. **Add tests**:
   - `test_memory_does_not_grow`: call `summarize()` 1 000 times, assert `len(agent._history) == 0` after each call
   - `test_summarize_returns_string`: assert the return value is a non-empty string

### Post-mortem prompt

Write two paragraphs: (1) what happened and root cause; (2) action items.

---

## Scenario B — "Pool Apocalypse" (~80 min)

### Symptoms

```
Alert: /api/checkout p99 latency > 10 s (baseline 120 ms)
Error messages: "pool timeout after 5.0s waiting for connection"
Metrics: DB active connections = 100/100 (maxed), idle_in_transaction = 87
Deploy log: checkout-service v1.8.2 deployed 22 min before alert
```

### Log excerpt

```json
{"ts":"2026-11-10T09:00:00Z","level":"info","msg":"checkout","order_id":"CH-001","status":"ok","db_ms":45}
{"ts":"2026-11-10T09:22:10Z","level":"error","msg":"checkout","order_id":"CH-500","error":"pool timeout"}
{"ts":"2026-11-10T09:22:11Z","level":"error","msg":"checkout","order_id":"CH-501","error":"pool timeout"}
{"ts":"2026-11-10T09:22:12Z","level":"info","msg":"db_stats","active":100,"idle_in_tx":87,"waiting":34}
```

### Buggy code to diagnose

```python
"""scenario_b_buggy.py — find and fix the connection pool exhaustion."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("checkout")


@dataclass
class FakeConnection:
    conn_id: int
    in_transaction: bool = False


@dataclass
class SimplePool:
    max_size: int = 5
    timeout_s: float = 5.0
    _pool: queue.Queue = field(default_factory=queue.Queue)

    def __post_init__(self) -> None:
        for i in range(self.max_size):
            self._pool.put(FakeConnection(i))

    def acquire(self, timeout: float | None = None) -> FakeConnection:
        t = timeout if timeout is not None else self.timeout_s
        try:
            return self._pool.get(timeout=t)
        except queue.Empty:
            raise TimeoutError(f"Pool timeout after {t}s waiting for connection")

    def release(self, conn: FakeConnection) -> None:
        conn.in_transaction = False
        self._pool.put(conn)


# BUG IS SOMEWHERE IN THIS FILE

class CheckoutService:
    def __init__(self, pool: SimplePool) -> None:
        self._pool = pool

    def process_checkout(self, order_id: str, items: list[dict]) -> dict[str, Any]:
        conn = self._pool.acquire()
        conn.in_transaction = True
        log.debug("Acquired connection %d for %s", conn.conn_id, order_id)

        try:
            # Step 1: validate inventory
            result = self._validate_inventory(conn, items)
            if not result["ok"]:
                return {"status": "failed", "reason": "inventory"}

            # Step 2: charge payment
            payment = self._charge_payment(order_id)  # can raise PaymentError
            if not payment["ok"]:
                return {"status": "failed", "reason": "payment"}  # BUG: returns without releasing conn

            # Step 3: commit
            time.sleep(0.01)
            self._pool.release(conn)
            return {"status": "ok", "order_id": order_id}

        except Exception as exc:
            log.error("Checkout failed for %s: %s", order_id, exc)
            # BUG: no release on exception path either
            raise

    def _validate_inventory(self, conn: FakeConnection, items: list[dict]) -> dict:
        time.sleep(0.005)
        return {"ok": True}

    def _charge_payment(self, order_id: str) -> dict:
        # Simulate 30% payment decline
        if hash(order_id) % 10 < 3:
            return {"ok": False}
        return {"ok": True}


def simulate_checkouts(n: int = 20) -> None:
    pool = SimplePool(max_size=5, timeout_s=2.0)
    service = CheckoutService(pool)
    errors = 0

    def run(i: int) -> None:
        nonlocal errors
        try:
            result = service.process_checkout(
                f"CH-{i:04d}",
                [{"sku": "SKU-1", "qty": 1}],
            )
            log.info("CH-%04d: %s", i, result["status"])
        except TimeoutError as exc:
            log.error("CH-%04d: %s", i, exc)
            errors += 1

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    remaining = pool._pool.qsize()
    log.info("Pool remaining: %d/%d | Timeout errors: %d", remaining, pool.max_size, errors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    simulate_checkouts()
```

### Your tasks

1. **Identify** all code paths where the connection is not released.
2. **Fix** using a context manager (`__enter__`/`__exit__`) on `SimplePool` so every acquire is guaranteed to release.
3. **Add tests**:
   - `test_connection_released_on_success`
   - `test_connection_released_on_payment_decline`
   - `test_connection_released_on_exception`

### Post-mortem prompt

Write two paragraphs: (1) what happened and root cause; (2) action items including the deploy review that should have caught this.

---

## Scenario C — "The Hallucinating Analyst" (~80 min)

### Symptoms

```
Alert: customer support tickets spike 3× — "wrong order amounts quoted by agent"
Service: AI order analyst (LLM-powered chat)
Deploy: prompt updated yesterday "to be more helpful and conversational"
Metrics: avg_response_length increased from 180 to 340 tokens
```

### Log excerpt

```json
{"ts":"2026-11-10T10:00:00Z","session":"S-1","input":"What is the total for order ORD-001?","output":"Your order ORD-001 totals $1,240.00 including shipping and tax.","actual_total":103.50}
{"ts":"2026-11-10T10:01:00Z","session":"S-2","input":"List items in order ORD-002","output":"Order ORD-002 contains: 3x Widget A ($45 each), 2x Widget B ($30 each) — estimated total $195.","actual_item_count":4}
```

### Buggy prompt to diagnose

```python
"""scenario_c_buggy.py — diagnose and fix the prompt regression."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

log = logging.getLogger("analyst")

# BUG: the v2 prompt was changed to be "more helpful"
SYSTEM_PROMPT_V1 = """You are an order management assistant.
Answer questions ONLY using the order data provided in the context.
Do not infer, estimate, or add information not present in the data.
Format amounts exactly as they appear in the data."""

SYSTEM_PROMPT_V2_BUGGY = """You are a friendly and helpful order management assistant!
Be conversational and proactive. If you don't have exact data, provide a helpful estimate
based on similar orders. Include relevant context like shipping and taxes when discussing totals.
Always be encouraging and positive!"""  # BUG: allows estimation and fabrication


@dataclass
class Order:
    order_id: str
    items: list[dict[str, Any]]
    subtotal: float

    def to_context(self) -> str:
        items_str = "\n".join(
            f"  - {it['qty']}x {it['name']} @ ${it['price']:.2f}"
            for it in self.items
        )
        return f"Order {self.order_id}:\n{items_str}\nSubtotal: ${self.subtotal:.2f}"


def mock_llm(system_prompt: str, user_query: str, context: str) -> str:
    """Mock LLM — v1 prompt produces grounded output; v2 produces estimates."""
    if "ONLY using the order data" in system_prompt:
        # Grounded — extract from context
        if "Subtotal" in context:
            subtotal_line = [l for l in context.split("\n") if "Subtotal" in l][0]
            return f"The order total is {subtotal_line.split('Subtotal: ')[1]}."
        return "I don't have that information in the provided order data."
    else:
        # Creative/estimating — hallucinates
        return "Your order totals approximately $1,240.00 including estimated shipping and tax."


@dataclass
class OrderAnalyst:
    system_prompt: str

    def answer(self, query: str, order: Order) -> str:
        context = order.to_context()
        return mock_llm(self.system_prompt, query, context)


# ── Eval suite ────────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    query: str
    order: Order
    must_contain: str
    must_not_contain: str = ""


EVAL_CASES = [
    EvalCase(
        query="What is the total for my order?",
        order=Order(
            "ORD-001",
            [{"qty": 2, "name": "Widget A", "price": 45.00}],
            subtotal=90.00,
        ),
        must_contain="90.00",
        must_not_contain="1,240",
    ),
    EvalCase(
        query="What is the total for my order?",
        order=Order(
            "ORD-002",
            [{"qty": 1, "name": "Gadget X", "price": 103.50}],
            subtotal=103.50,
        ),
        must_contain="103.50",
        must_not_contain="estimated",
    ),
]


def run_eval(analyst: OrderAnalyst, label: str) -> float:
    passed = 0
    for case in EVAL_CASES:
        answer = analyst.answer(case.query, case.order)
        ok_pos = case.must_contain in answer
        ok_neg = case.must_not_contain not in answer if case.must_not_contain else True
        passed += 1 if (ok_pos and ok_neg) else 0
        log.info("[%s] Q=%s | pass=%s | answer=%s", label, case.query[:30], ok_pos and ok_neg, answer[:60])
    return passed / len(EVAL_CASES)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    v1_analyst = OrderAnalyst(SYSTEM_PROMPT_V1)
    v2_analyst = OrderAnalyst(SYSTEM_PROMPT_V2_BUGGY)

    v1_score = run_eval(v1_analyst, "v1")
    v2_score = run_eval(v2_analyst, "v2-buggy")

    log.info("V1 score: %.0f%%  V2 score: %.0f%%", v1_score * 100, v2_score * 100)
    if v2_score < v1_score:
        log.error("REGRESSION DETECTED — v2 should not be shipped")
```

### Your tasks

1. **Identify** exactly which lines in `SYSTEM_PROMPT_V2_BUGGY` caused the regression and why.
2. **Write `SYSTEM_PROMPT_V2_FIXED`** that adds conversational tone without allowing estimation or fabrication.
3. **Add tests** that run the eval suite against both v1 and v2-fixed and assert score ≥ 1.0 (100%).
4. **Write `prompt_regression_gate(v1_prompt, v2_prompt, eval_cases, threshold=0.0) -> bool`** that returns False if v2 regresses more than `threshold` vs v1. This should be wired into CI.

### Post-mortem prompt

Write two paragraphs: (1) what happened and why the prompt change caused hallucination; (2) action items including an eval gate in CI.

---

## Wrap-up

After completing all three scenarios, answer these reflection questions:

1. What single structural pattern did all three bugs share?
2. Which of the five debugging steps do you find hardest to execute under pressure, and why?
3. What CI gate would have caught each bug before it reached production?

(Solutions: ../solutions/week-09/day-6.md)
