# Phase 6 · Week 03 · Day 6 (Saturday) — Build the Reliability Toolkit

> Objective: Assemble all week's concepts into a cohesive, production-ready reliability toolkit wrapping an unreliable LLM/tool client.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

---

## Overview

Today you build **`reliability_toolkit`**: a Python package that wraps any LLM or tool client with a full reliability stack. This is a deeper practice session — you design the architecture, write the code, and write the tests.

### What you're building

```
reliability_toolkit/
├── __init__.py
├── retry.py          # RetryPolicy, retry decorator
├── idempotency.py    # IdempotencyStore, idempotent_call
├── circuit_breaker.py# CircuitBreaker, CircuitBreakerConfig
├── timeout.py        # with_timeout, TimeoutConfig
├── bulkhead.py       # Bulkhead
├── rate_limiter.py   # TokenBucket, RateLimiter
├── fallback.py       # FallbackChain
├── client.py         # ResilientLLMClient (composes all of the above)
└── chaos.py          # ChaosMiddleware (for tests)
tests/
├── conftest.py
├── test_retry.py
├── test_idempotency.py
├── test_circuit_breaker.py
├── test_rate_limiter.py
├── test_client.py     # integration test of ResilientLLMClient
└── test_chaos.py
```

---

## Part 1 — Architecture design (30 min)

Before writing code, answer these design questions in comments at the top of `client.py`:

1. **Order of operations**: when a call arrives at `ResilientLLMClient`, in what order should bulkhead, circuit breaker, rate limiter, timeout, and retry apply? Draw the call stack.

2. **Idempotency placement**: should the idempotency check happen before or after the retry wrapper? Why?

3. **Fallback placement**: should the fallback chain wrap the retry+circuit-breaker stack, or should each fallback option have its own retry? Justify with a failure scenario.

---

## Part 2 — Core modules (90 min)

Implement each module listed above. Constraints:
- All modules use Python 3.12+ type hints
- No global mutable state (pass state via constructor)
- Each module is independently importable and testable
- `logging` throughout, no `print`
- Pydantic v2 models for all config objects

### `client.py` interface (implement this)

```python
from pydantic import BaseModel
from typing import Any

class LLMRequest(BaseModel):
    prompt: str
    model: str = "claude-opus-5-5"     # mock — never actually called
    max_tokens: int = 1024
    idempotency_key: str | None = None

class LLMResponse(BaseModel):
    content: str
    model: str
    attempt_count: int
    served_from_cache: bool = False
    degraded: bool = False

class ResilientLLMClient:
    """
    Wraps a mock LLM transport with:
    - Token bucket rate limiting
    - Circuit breaker
    - Timeout per call
    - Exponential backoff retry
    - Idempotency key deduplication
    - Fallback chain (primary → backup → static)
    """

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...
```

---

## Part 3 — Tests (60 min)

Write tests covering every reliability dimension. Use `pytest-asyncio` for async tests.

### Required test cases

**`test_retry.py`**
- Succeeds on 3rd attempt; verify `attempt_count == 3`
- Non-retryable error not retried
- Deadline exceeded before max attempts

**`test_idempotency.py`**
- Same idempotency key returns cached result without re-executing
- Different keys execute independently
- TTL expiry causes re-execution

**`test_circuit_breaker.py`**
- CLOSED → OPEN after `failure_threshold` failures
- OPEN rejects immediately (no call to underlying fn)
- OPEN → HALF_OPEN after `reset_timeout`; success closes it

**`test_rate_limiter.py`**
- Burst up to capacity without waiting
- Request beyond capacity waits or is rejected (test both modes)
- Token refill correct after sleep

**`test_client.py`** (integration)
- Happy path: returns `LLMResponse` with `degraded=False`
- Primary fails, backup used: `degraded=True`
- All providers fail, static fallback: `degraded=True`, `content` contains "unavailable"
- Idempotency: two calls with same key return same `content`
- Circuit open: client returns fallback, not a timeout

**`test_chaos.py`**
- Inject 100% errors: all calls fail
- Inject 0% errors: all calls succeed
- With `seed=42`, error sequence is deterministic across runs

---

## Part 4 — Chaos scenario (30 min)

Write a script `scenarios/chaos_scenario.py` that:

1. Creates a `ResilientLLMClient`
2. Wraps the primary transport with `ChaosMiddleware(error_rate=0.7, latency_ms=200, seed=123)`
3. Sends 20 requests concurrently
4. Collects results and prints a summary: `{success, degraded, failed, avg_latency_ms}`
5. Asserts that at least 80% of requests return *some* result (success or degraded fallback)

This is your chaos hypothesis: "even with 70% primary failures, 80%+ of requests are served."

---

## Part 5 — Documentation (20 min)

Add a module docstring to `client.py` that:
- States the call-stack order with a one-line justification for each layer position
- Documents all config parameters with their defaults and recommended values for LLM workloads
- Lists the failure modes that each layer handles and which it doesn't

---

## Deliverable checklist

- [ ] All 8 source modules exist with no import errors
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] `chaos_scenario.py` assertion passes with 70% error rate
- [ ] No hardcoded credentials; mock LLM transport used throughout
- [ ] Zero `print()` calls — `logging` only
- [ ] All config via Pydantic models

(Solutions: ../solutions/week-03/day-6.md)

---

## Self-check

- [ ] My `ResilientLLMClient` composes all reliability layers correctly
- [ ] I can articulate why each layer is in its position
- [ ] My tests simulate failures without network access
- [ ] My chaos experiment has a clear hypothesis and measurable outcome
- [ ] I can swap the mock LLM transport for a real one with zero code changes to the toolkit

---

## Supplementary links

- [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io/en/latest/)
- [Resilience4j design patterns (Java, but architecture transfers)](https://resilience4j.readme.io/docs/getting-started)
- [Reliability patterns for ML systems — Google SRE book ch. 26](https://sre.google/sre-book/addressing-cascading-failures/)
