# Week 03 Project — Reliability Toolkit for LLM/Tool Clients

> Phase 6 · Production AI  ·  Build time: ~4 hr (Sat) + polish on Sun

---

## Overview

Build **`reliability_toolkit`**: a standalone Python package that wraps any async LLM or tool client with production-grade fault tolerance. No real API calls. All failure scenarios driven by a chaos layer in tests.

---

## Requirements

### Functional

| # | Requirement |
|---|---|
| F1 | `RetryPolicy` with configurable max attempts, exponential backoff, full jitter, retryable error predicate |
| F2 | `IdempotencyStore` (in-memory, TTL-expiring); `idempotent_call(key, fn)` helper |
| F3 | `CircuitBreaker` with CLOSED/OPEN/HALF-OPEN states, configurable threshold and reset timeout |
| F4 | `with_timeout(coro, seconds)` coroutine wrapper raising `asyncio.TimeoutError` |
| F5 | `Bulkhead` with configurable max concurrency and fast-fail on overflow |
| F6 | `TokenBucket` rate limiter with configurable rate, burst capacity, blocking and non-blocking modes |
| F7 | `FallbackChain` trying a list of async callables in order |
| F8 | `ResilientLLMClient` composing F1–F7; accepts a mock transport; returns `LLMResponse` |
| F9 | `ChaosMiddleware` for test injection: error rate, latency injection, seeded RNG |
| F10 | `ReliabilityConfig` loaded from environment via `pydantic-settings` |

### Non-functional

| # | Requirement |
|---|---|
| NF1 | Python 3.12+; all public APIs fully type-hinted |
| NF2 | Pydantic v2 for all config and data models |
| NF3 | `logging` not `print`; structured log fields (component, attempt, state) |
| NF4 | Zero real network calls; mock transport injected via constructor |
| NF5 | Test coverage ≥ 80% on `src/` measured by `pytest --cov` |
| NF6 | All tests pass with `pytest tests/ -v` |
| NF7 | Package is importable as `from reliability_toolkit import ResilientLLMClient` |

---

## Project structure

```
reliability_toolkit/
├── pyproject.toml
├── .env.example
├── src/
│   └── reliability_toolkit/
│       ├── __init__.py
│       ├── config.py         # ReliabilityConfig (pydantic-settings)
│       ├── retry.py          # RetryPolicy
│       ├── idempotency.py    # IdempotencyStore, idempotent_call
│       ├── circuit_breaker.py
│       ├── timeout.py
│       ├── bulkhead.py
│       ├── rate_limiter.py   # TokenBucket
│       ├── fallback.py       # FallbackChain
│       ├── client.py         # ResilientLLMClient, LLMRequest, LLMResponse
│       └── chaos.py          # ChaosMiddleware
├── tests/
│   ├── conftest.py
│   ├── test_retry.py
│   ├── test_idempotency.py
│   ├── test_circuit_breaker.py
│   ├── test_timeout.py
│   ├── test_bulkhead.py
│   ├── test_rate_limiter.py
│   ├── test_fallback.py
│   ├── test_client.py
│   └── test_chaos.py
└── scenarios/
    ├── chaos_scenario.py     # 70% failure rate, verify 80%+ served
    └── saga_scenario.py      # distributed transaction with rollback
```

---

## Deliverables

### 1. `pyproject.toml`

```toml
[project]
name = "reliability-toolkit"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "tenacity>=8.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### 2. Mock transport interface

```python
from typing import Protocol

class LLMTransport(Protocol):
    async def complete(self, prompt: str, model: str) -> str:
        ...
```

Your `ResilientLLMClient` takes `primary: LLMTransport` and `fallbacks: list[LLMTransport]`.

### 3. `ResilientLLMClient` call contract

```
acquire bulkhead slot
  → check idempotency key (return cached if hit)
  → acquire rate limiter tokens
  → check circuit breaker (raise CircuitOpenError if OPEN)
  → start timeout
  → retry loop:
      try: call transport.complete()
      on TransientError: wait backoff, retry
      on PermanentError: break immediately
  → if all retries fail: try next fallback
  → if all fallbacks fail: return static degraded response
  → store idempotency result
  → release bulkhead slot
```

Document this order in `client.py` with a comment explaining each position.

### 4. Chaos scenario

`scenarios/chaos_scenario.py` must:
- Run 20 concurrent `complete()` calls
- Primary transport wrapped with 70% error rate, 200 ms latency injection
- Print summary: `{total: 20, success: N, degraded: N, failed: N, avg_latency_ms: N}`
- Assert `success + degraded >= 16` (80% served)

### 5. Saga scenario

`scenarios/saga_scenario.py` must demonstrate:
- A 3-step saga: `reserve_slot → deduct_credits → send_confirmation`
- Step 2 fails → step 1 compensated (slot released)
- Logged output shows compensation
- All compensation calls are idempotent (safe to retry)

---

## Assessment integration

This project feeds directly into the week-03 assessment coding challenge. Your `ResilientLLMClient` is the foundation for the `ReliableAgentNode` you build in the assessment.

---

## Grading

| Dimension | Weight | Indicators |
|---|---|---|
| Correctness | 40% | All F1–F10 requirements met; tests pass |
| Architecture | 25% | Clean separation; call-stack order documented and justified |
| Test quality | 20% | ≥80% coverage; chaos tests deterministic with seed; edge cases covered |
| Code quality | 15% | Type hints, Pydantic v2, logging, no global state |

Pass gate: 75%.  
Solution reference: ../solutions/week-03/project.md
