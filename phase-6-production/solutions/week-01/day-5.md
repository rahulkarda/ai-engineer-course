# Solutions — Day 5: Failure Modes, Time & Ordering, Idempotency

## Exercise 1 — Lamport clock trace

After B receives A's send at ts=2:
- B's clock = max(B.clock, 2) + 1 = max(1, 2) + 1 = 3 (B was at clock=1 from its own previous event, if any; if B's clock was 0: max(0,2)+1=3)

B does a local event → B.clock = 4
B sends to C → B.clock = 5, message carries ts=5
C receives → C.clock = max(C.clock, 5) + 1 = max(0, 5) + 1 = 6

```python
a = LamportNode("A")
b = LamportNode("B")
c = LamportNode("C")

a.local_event("write_embedding")   # A.clock = 1
msg, ts = a.send("notify")          # A.clock = 2, ts = 2
b.receive("notify", ts)             # B.clock = max(0,2)+1 = 3
b.local_event("start_query")        # B.clock = 4
msg2, ts2 = b.send("query_result")  # B.clock = 5, ts2 = 5
recv_ts = c.receive("query_result", ts2)  # C.clock = max(0,5)+1 = 6

assert recv_ts == 6
print(f"C's clock after receive: {recv_ts}")  # 6
```

## Exercise 2 — Concurrent writes with vector clocks

```python
num_nodes = 3
a = VectorClockNode("A", 0, num_nodes)
b = VectorClockNode("B", 1, num_nodes)
c = VectorClockNode("C", 2, num_nodes)  # not used in this scenario

# Node A and B each independently write — no causal relationship
a.local_event("write_x=1")   # A.vc = (1,0,0)
b.local_event("write_x=2")   # B.vc = (0,1,0)

print(f"A.vc = {list(a.vc.values)}")  # (1,0,0)
print(f"B.vc = {list(b.vc.values)}")  # (0,1,0)

assert a.vc.concurrent_with(b.vc), "A and B wrote independently — they are concurrent"
print("Concurrent: True")
```

Why this requires application-level conflict resolution:
- `(1,0,0)` does not dominate `(0,1,0)` and vice versa
- A conflict-detection system sees a true conflict: two writes happened without knowledge of each other
- Neither is "newer" — both are equally valid
- The application must decide: LWW (pick higher timestamp), merge (union of values), or surface conflict to user

## Exercise 3 — IdempotentToolExecutor with test

```python
# idempotent_executor.py
from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger(__name__)


class IdempotentToolExecutor:
    """
    Wraps a tool function so it is only ever called once per idempotency_key.
    Subsequent calls with the same key return the cached result.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self._cache: dict[str, Any] = {}

    def execute(self, idempotency_key: str, **kwargs: Any) -> Any:
        if idempotency_key in self._cache:
            log.debug("Cache hit: key=%s", idempotency_key[:8])
            return self._cache[idempotency_key]
        result = self._fn(**kwargs)
        self._cache[idempotency_key] = result
        log.info("Executed and cached: key=%s", idempotency_key[:8])
        return result
```

```python
# test_idempotent_executor.py
from unittest.mock import MagicMock
from idempotent_executor import IdempotentToolExecutor


def test_underlying_function_called_once_for_duplicate_key() -> None:
    mock_fn = MagicMock(return_value={"status": "charged", "amount": 100})
    executor = IdempotentToolExecutor(fn=mock_fn)

    key = "payment-abc123"
    result1 = executor.execute(key, amount=100, account_id="acc-1")
    result2 = executor.execute(key, amount=100, account_id="acc-1")

    assert result1 == result2
    mock_fn.assert_called_once()  # Critical: only called once despite two execute() calls


def test_different_keys_call_function_separately() -> None:
    mock_fn = MagicMock(side_effect=lambda **kw: {"amount": kw["amount"]})
    executor = IdempotentToolExecutor(fn=mock_fn)

    executor.execute("key-1", amount=50)
    executor.execute("key-2", amount=75)

    assert mock_fn.call_count == 2
```

## Exercise 4 — charge_card without idempotency keys

**Scenario 1 — Double charge**: Agent calls `charge_card(100, "card-42")`. Card is charged. HTTP response is lost (network timeout). Agent retries. Card is charged again. Customer sees two charges of $100. No way to detect the duplicate because the tool has no record of the first call's intent.

**Scenario 2 — Partial retry with different amount**: Agent is mid-workflow. First call for $100 succeeds server-side; response is lost. A different agent step recalculates and retries with $95 (order discount applied). Now two charges exist: $100 and $95. Reconciliation is impossible without idempotency tracking.

**Scenario 3 — Cascading retry storm**: A queue-based agent framework retries all "failed" (no-response) tool calls at startup. 50 orders placed yesterday — all responses were lost during an outage. On restart, all 50 are retried. 50 double-charges occur before anyone notices.

**Mitigation**: Client generates `idempotency_key = uuid4()` before calling. Server stores `(idempotency_key, result)`. On duplicate: return stored result. Key must be generated *before* the call (not inside the tool), stored durably (not in memory), and scoped to the logical business operation (not the HTTP request).
