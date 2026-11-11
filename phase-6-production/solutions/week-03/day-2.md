# Solutions — Week 03 · Day 2: Idempotency Keys & Outbox Pattern

---

## Exercise 1 — TTL boundary validation

```python
import pytest
from day2_idempotency import IdempotencyStore


def test_max_ttl_raises() -> None:
    store = IdempotencyStore()
    with pytest.raises(ValueError, match="ttl_seconds"):
        store.set("key", "value", ttl_seconds=8 * 86_400)  # 8 days — over limit


def test_max_ttl_boundary_passes() -> None:
    store = IdempotencyStore()
    store.set("key", "value", ttl_seconds=7 * 86_400)  # exactly 7 days — OK
    assert store.get("key") == "value"
```

Add to `IdempotencyStore.set`:

```python
def set(self, key: str, result: Any, ttl_seconds: float = 86_400) -> None:
    MAX_TTL = 7 * 86_400
    if ttl_seconds > MAX_TTL:
        raise ValueError(
            f"ttl_seconds {ttl_seconds} exceeds maximum {MAX_TTL} ({MAX_TTL // 86_400} days)"
        )
    self._store[key] = IdempotencyRecord(key=key, result=result, ttl_seconds=ttl_seconds)
```

---

## Exercise 2 — Batched relay

```python
import pytest
from day2_idempotency import InMemoryOutbox


def test_batch_size_sweep() -> None:
    outbox = InMemoryOutbox()
    for i in range(250):
        outbox.append("test.event", {"i": i})

    published_total = 0
    sweeps = 0

    def mock_pub(event):
        pass  # no-op

    while outbox.pending():
        n = outbox.relay(mock_pub, batch_size=100)
        published_total += n
        sweeps += 1

    assert sweeps == 3
    assert published_total == 250
    assert len(outbox.pending()) == 0
```

Modified `relay`:

```python
def relay(self, publisher: Any, batch_size: int = 100) -> int:
    published = 0
    for event in self._events:
        if event.published:
            continue
        if published >= batch_size:
            break
        try:
            publisher(event)
            event.published = True
            published += 1
        except Exception:
            logger.exception("Outbox: relay failed for id=%s", event.event_id)
    return published
```

---

## Exercise 3 — Content-addressed key (order-independent)

```python
import hashlib
import json
from typing import Any


def content_addressed_key(tool_name: str, args: dict[str, Any]) -> str:
    """SHA-256 of tool name + args with keys sorted at all nesting levels."""
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_key_order_independence() -> None:
    args_a = {"prompt": "hello", "model": "claude-opus-5-5", "max_tokens": 100}
    args_b = {"model": "claude-opus-5-5", "max_tokens": 100, "prompt": "hello"}
    assert content_addressed_key("llm", args_a) == content_addressed_key("llm", args_b)

    args_c = {"prompt": "different", "model": "claude-opus-5-5", "max_tokens": 100}
    assert content_addressed_key("llm", args_a) != content_addressed_key("llm", args_c)
```

---

## Exercise 4 — Outbox failure mode analysis

**Scenario A: DB crashes between `INSERT outbox_event` and `COMMIT`**

The `INSERT outbox_event` was never committed — it is rolled back atomically with the rest of the transaction. Neither the business record nor the outbox event persists. The relay has nothing to forward. From the system's perspective, the operation never happened. The client will retry, potentially issuing the same operation again (which is why the caller should use an idempotency key for the business operation itself).

**Scenario B: DB crashes between `COMMIT` and relay's `UPDATE published=true`**

The business record and the outbox event are both durably committed. The relay has not yet marked the event as published. When the relay restarts, it finds `published=false` and forwards the event again — **at-least-once delivery**. This is by design: the downstream consumer must be idempotent (e.g. use the `event_id` as an idempotency key). The outbox pattern guarantees **at-least-once**, not exactly-once, unless the consumer deduplicates.
