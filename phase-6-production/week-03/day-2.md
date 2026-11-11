# Phase 6 · Week 03 · Day 2 — Idempotency Keys & the Outbox Pattern

> Objective: Use idempotency keys to make LLM/tool calls safe to retry, and implement the outbox pattern for reliable event emission.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]] (retries)

---

## Concept

Retries introduce a new hazard: **duplicate execution**. If a tool call succeeds on the server but the response is lost in transit, a retry charges the user twice, sends two emails, or creates two records.

**Idempotency** means "applying the same operation multiple times produces the same outcome as applying it once." Two tools:

1. **Idempotency keys** — the client generates a unique key per logical operation and sends it with every attempt. The server (or your wrapper) deduplicates on that key.
2. **Outbox pattern** — instead of emitting events/side-effects directly from a transaction, write them to an `outbox` table in the same DB transaction, then a background process forwards them. If the forward fails, it retries from the outbox — exactly-once delivery without distributed transactions.

### Where this matters for agents

- Calling `send_email(...)` — should not send twice on retry
- Calling `charge_payment(...)` — critical
- LLM calls — usually safe to duplicate (stateless), but tool side-effects are not
- Saga steps — each compensatable action must be idempotent

---

## How it works internally

**Idempotency key flow:**

```
Client                              Server/Wrapper
  │                                      │
  ├─ generate key = uuid4() ────────────►│
  │                                      ├─ check store: key exists? ──► return cached result
  │                                      ├─ execute operation
  │                                      ├─ store {key: result, ttl: 24h}
  │◄─ return result ─────────────────────┤
```

**Outbox flow:**

```
Business TX                  Outbox Relay
  ├─ INSERT record               │
  ├─ INSERT outbox event         │
  └─ COMMIT ─────────────────────►├─ SELECT unpublished events
                                  ├─ publish to downstream
                                  └─ UPDATE outbox SET published=true
```

Key insight: the outbox and the business record commit in the **same transaction** — atomicity guaranteed even if the relay crashes mid-publish.

---

## Worked example

```python
"""
day2_idempotency.py — Idempotency key store + outbox pattern.

Install: pip install pydantic
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Idempotency store (in-memory; swap for Redis/Postgres in prod) ─────────────

@dataclass
class IdempotencyRecord:
    key: str
    result: Any
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 86_400  # 24 hours


class IdempotencyStore:
    """Thread-safe in-memory idempotency cache."""

    def __init__(self) -> None:
        self._store: dict[str, IdempotencyRecord] = {}

    def get(self, key: str) -> Any | None:
        record = self._store.get(key)
        if record is None:
            return None
        age = time.monotonic() - record.created_at
        if age > record.ttl_seconds:
            del self._store[key]
            return None
        logger.info("Idempotency hit for key=%s", key)
        return record.result

    def set(self, key: str, result: Any, ttl_seconds: float = 86_400) -> None:
        self._store[key] = IdempotencyRecord(key=key, result=result, ttl_seconds=ttl_seconds)

    def __len__(self) -> int:
        return len(self._store)


# ── Tool wrapper with idempotency ─────────────────────────────────────────────

_idempotency_store = IdempotencyStore()


def _make_key(tool_name: str, args: dict[str, Any]) -> str:
    """Deterministic key from tool name + sorted args."""
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def idempotent_tool_call(
    tool_name: str,
    args: dict[str, Any],
    execute_fn: Any,
    idempotency_key: str | None = None,
) -> Any:
    """
    Wrap any tool call with idempotency.

    Args:
        tool_name: logical name for the operation
        args: arguments (used to build a deterministic key if none provided)
        execute_fn: zero-arg callable that performs the actual side-effect
        idempotency_key: explicit key; auto-generated from args if None
    """
    key = idempotency_key or _make_key(tool_name, args)
    cached = _idempotency_store.get(key)
    if cached is not None:
        return cached

    result = execute_fn()
    _idempotency_store.set(key, result)
    logger.info("Executed %s, stored result under key=%s", tool_name, key[:12])
    return result


# ── Outbox pattern ─────────────────────────────────────────────────────────────

class OutboxEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    payload: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    published: bool = False


class InMemoryOutbox:
    """
    Simulates the outbox table + relay loop.
    In production: events and business records share a DB transaction.
    """

    def __init__(self) -> None:
        self._events: list[OutboxEvent] = []

    def append(self, event_type: str, payload: dict[str, Any]) -> OutboxEvent:
        """Append event in the same 'transaction' as business logic."""
        event = OutboxEvent(event_type=event_type, payload=payload)
        self._events.append(event)
        logger.debug("Outbox: appended %s id=%s", event_type, event.event_id)
        return event

    def relay(self, publisher: Any) -> int:
        """
        Forward unpublished events to the downstream publisher.
        Returns count of events published in this sweep.
        """
        published = 0
        for event in self._events:
            if event.published:
                continue
            try:
                publisher(event)
                event.published = True
                published += 1
                logger.info("Outbox: published %s id=%s", event.event_type, event.event_id)
            except Exception:
                logger.exception("Outbox: relay failed for id=%s — will retry", event.event_id)
        return published

    def pending(self) -> list[OutboxEvent]:
        return [e for e in self._events if not e.published]


# ── Demo ──────────────────────────────────────────────────────────────────────

def mock_send_email(to: str, subject: str) -> dict[str, str]:
    logger.info("EMAIL SENT to=%s subject=%s", to, subject)
    return {"status": "sent", "to": to}


if __name__ == "__main__":
    # Idempotency: calling twice with same args only executes once
    args = {"to": "user@example.com", "subject": "Your report"}
    for i in range(3):
        result = idempotent_tool_call(
            tool_name="send_email",
            args=args,
            execute_fn=lambda: mock_send_email(**args),
        )
        logger.info("Call %d result: %s", i + 1, result)

    # Outbox: events written atomically, published separately
    outbox = InMemoryOutbox()
    outbox.append("user.registered", {"user_id": "u123", "email": "user@example.com"})
    outbox.append("welcome.email.queued", {"user_id": "u123"})
    logger.info("Pending events: %d", len(outbox.pending()))

    def mock_publisher(event: OutboxEvent) -> None:
        logger.info("Published: %s", event.event_type)

    published = outbox.relay(mock_publisher)
    logger.info("Relay published %d events, pending now: %d", published, len(outbox.pending()))
```

---

## Exercises

1. Add a `max_ttl_seconds` check to `IdempotencyStore.set` that raises `ValueError` if `ttl_seconds > 7 * 86_400` (7 days). Write a pytest test for this boundary.

2. Extend `InMemoryOutbox.relay` to accept a `batch_size: int = 100` argument so it publishes at most `batch_size` events per sweep. Write a test with 250 events that verifies three sweeps are needed.

3. Implement a `content_addressed_key(tool_name, args)` that produces the same SHA-256 key regardless of dict key insertion order. Verify with a pytest test using two dicts with the same keys in different order.

4. In a real system, what happens to the outbox if the DB crashes between `INSERT outbox_event` and `COMMIT`? What happens if it crashes between `COMMIT` and the relay `UPDATE published=true`? Write a one-paragraph explanation of each scenario.

(Solutions: ../solutions/week-03/day-2.md)

---

## Self-check

- [ ] I can explain why idempotency keys are needed even when retries are correct
- [ ] I can build an idempotency store with deterministic key generation
- [ ] I can explain the outbox pattern and why it guarantees at-least-once delivery
- [ ] I can describe the failure modes the outbox pattern protects against
- [ ] I can wrap an arbitrary tool call with idempotency without modifying its implementation

---

## Supplementary links

- [Stripe idempotency keys guide](https://stripe.com/docs/idempotency)
- [The Outbox Pattern — Microservices.io](https://microservices.io/patterns/data/transactional-outbox.html)
- [Debezium change data capture (outbox relay in production)](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)
