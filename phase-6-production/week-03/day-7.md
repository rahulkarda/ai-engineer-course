# Phase 6 · Week 03 · Day 7 (Sunday) — Project Integration + Review

> Objective: Finalize and harden the reliability toolkit, run the full assessment, reflect on the week.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[day-6]] (toolkit build)

---

## Morning: Project hardening (3 hr)

### 1. Code review pass (45 min)

Read every file in `reliability_toolkit/` with this checklist:

- [ ] Every public function/method has a docstring with Args, Returns, Raises
- [ ] All config models use `model_config = ConfigDict(frozen=True)` — configs are immutable
- [ ] No bare `except Exception:` without logging the exception
- [ ] Every `await` that can raise has explicit exception handling or is inside the retry wrapper
- [ ] `CircuitBreaker` and `TokenBucket` state is not shared between unrelated callers

### 2. Missing edge cases (45 min)

Add tests for these scenarios you probably didn't cover:

**Concurrent idempotency**: two coroutines call `complete()` with the same idempotency key at the exact same time. Only one should execute; both should receive the same result. (Hint: use `asyncio.Barrier` or `asyncio.gather` to fire them simultaneously.)

**Rate limiter fairness**: with `max_concurrent=1` in `Bulkhead`, 10 queued requests should complete in arrival order (FIFO). Verify ordering.

**Saga partial compensation failure**: step 3 of 4 fails. Steps 1 and 2 are compensated. Compensation for step 1 raises. Verify the orchestrator logs the manual intervention warning and does not raise again.

**Circuit breaker thread safety**: use `asyncio.gather` to fire 20 concurrent calls when the circuit is OPEN. All 20 should receive `CircuitOpenError` — none should slip through.

### 3. Observability (30 min)

Add `metrics()` methods to your key components:

```python
# RetryPolicy
def metrics(self) -> dict[str, int]:
    return {"total_attempts": ..., "total_successes": ..., "total_failures": ...}

# CircuitBreaker
def metrics(self) -> dict[str, Any]:
    return {"state": ..., "failure_count": ..., "opened_at": ...}

# TokenBucket
def metrics(self) -> dict[str, float]:
    return {"available_tokens": ..., "total_acquired": ..., "total_rejected": ...}

# ResilientLLMClient
def metrics(self) -> dict[str, Any]:
    return {
        "retry": self._retry_policy.metrics(),
        "circuit_breaker": self._circuit_breaker.metrics(),
        "rate_limiter": self._rate_limiter.metrics(),
        "cache_hits": ...,
    }
```

Write a test that sends 10 requests and verifies `metrics()` counters match expected values.

### 4. Config from environment (30 min)

Create `reliability_toolkit/config.py` using `pydantic-settings`:

```python
from pydantic_settings import BaseSettings

class ReliabilityConfig(BaseSettings):
    retry_max_attempts: int = 5
    retry_max_delay_s: float = 30.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_s: float = 30.0
    rate_limit_rpm: float = 60.0
    rate_limit_burst: int = 10
    bulkhead_max_concurrent: int = 20
    call_timeout_s: float = 30.0
    idempotency_ttl_s: float = 86_400.0

    model_config = {"env_prefix": "RELIABILITY_"}
```

Wire `ResilientLLMClient` to accept a `ReliabilityConfig` instance. Write a test that reads overrides from environment variables using `monkeypatch.setenv`.

---

## Afternoon: Assessment (1 hr)

See `assessment.md` in this week's folder.

---

## Week 3 retrospective

Answer these in a personal log (not graded):

1. Which reliability pattern was hardest to get right? Why?
2. What failure mode did you discover during chaos testing that you hadn't anticipated?
3. If you were adding this toolkit to a real LangGraph agent, which three config values would you tune first and why?
4. What's one scenario where adding a retry would make things worse (hint: think about non-idempotent operations)?

---

## Self-check

- [ ] My toolkit has no known correctness bugs after the edge-case test pass
- [ ] `metrics()` on `ResilientLLMClient` gives accurate runtime counters
- [ ] Config loads from environment variables correctly
- [ ] I can demo the chaos scenario live: 70% failure rate, 80%+ requests served
- [ ] I can explain every design decision in the toolkit in an interview

---

## Supplementary links

- [pydantic-settings docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [asyncio synchronization primitives](https://docs.python.org/3/library/asyncio-sync.html)
- [Production Readiness Review checklist — Google](https://sre.google/sre-book/production-readiness-review/)
