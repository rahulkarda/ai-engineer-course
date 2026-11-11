# Phase 6 · Week 03 · Day 1 — Retries, Backoff, and Jitter

> Objective: Implement production-grade retry logic with exponential backoff and jitter for flaky LLM/tool APIs.  ·  Time: ~1 hr  ·  Prereqs: [[day-7 week-02]] (async Python, httpx basics)

---

## Concept

Remote calls fail. LLM APIs return 429 (rate-limited), 503 (overloaded), or simply drop connections. A **retry** re-issues a failed request in hope of success. Done naively, retries amplify load and cause **retry storms** — every client hammers the server simultaneously after a shared outage.

Three controls prevent storms:

| Control | Purpose |
|---|---|
| **Exponential backoff** | Each attempt waits `base * 2^attempt` seconds — waits grow, load spreads |
| **Jitter** | Random noise added to wait — prevents synchronized bursts across clients |
| **Max retries + deadline** | Caps total time spent; avoids infinite loops |

### Retryable vs non-retryable errors

Only retry *transient* failures:

- **Retry**: 429, 503, 502, 504, connection timeout, `httpx.TimeoutException`
- **Do not retry**: 400 (bad request), 401 (auth), 404 (not found), 422 (validation) — these will never succeed without a code change

---

## How it works internally

**Full jitter algorithm** (AWS recommended):

```
wait = random(0, min(cap, base * 2^attempt))
```

- `base`: starting wait (e.g. 0.5 s)
- `cap`: maximum wait (e.g. 60 s)
- `random(0, x)`: uniform sample — spreads clients across the full window

**Decorrelated jitter** (slightly less bursty, often preferred):

```
wait = random(base, min(cap, last_wait * 3))
```

When wrapping LLM clients, retry should live at the **transport layer**, not inside business logic. This keeps agent nodes clean.

**tenacity** is the standard Python library for retry policies. It provides:
- `@retry` decorator
- `wait_exponential_jitter`, `wait_random_exponential`
- `retry_if_exception_type`, `retry_if_result`
- `stop_after_attempt`, `stop_after_delay`
- `before_sleep` hooks for logging

---

## Worked example

```python
"""
day1_retry.py — Retry with backoff + jitter for a mock LLM client.

Install: pip install tenacity httpx
"""

import logging
import os
import random
import time
from typing import Any
from unittest.mock import MagicMock, call

import httpx
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Retryable exception types ────────────────────────────────────────────────

class TransientLLMError(Exception):
    """Rate-limit, overload, or transient network error."""
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class PermanentLLMError(Exception):
    """Auth failure, bad request — retrying won't help."""


# ── Mock unreliable LLM transport ────────────────────────────────────────────

class FlakyLLMTransport:
    """Simulates an LLM API that fails transiently before succeeding."""

    def __init__(self, fail_count: int = 2) -> None:
        self._attempts = 0
        self._fail_count = fail_count

    def complete(self, prompt: str) -> str:
        self._attempts += 1
        if self._attempts <= self._fail_count:
            raise TransientLLMError(429, "Rate limit exceeded")
        return f"Response to: {prompt}"


# ── Retry-wrapped client ──────────────────────────────────────────────────────

RETRYABLE = (TransientLLMError, httpx.TimeoutException, httpx.ConnectError)


def is_retryable(exc: BaseException) -> bool:
    """True for transient errors worth retrying."""
    if isinstance(exc, TransientLLMError):
        return exc.status_code in {429, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_random_exponential(multiplier=0.5, max=30),
    stop=stop_after_attempt(5) | stop_after_delay(60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_llm_with_retry(transport: FlakyLLMTransport, prompt: str) -> str:
    """Call the LLM transport with automatic retry on transient failures."""
    return transport.complete(prompt)


# ── Manual backoff implementation (no tenacity) ──────────────────────────────

def full_jitter_wait(attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
    """Return wait seconds using full-jitter exponential backoff."""
    ceiling = min(cap, base * (2 ** attempt))
    return random.uniform(0, ceiling)


def call_with_manual_retry(
    transport: FlakyLLMTransport,
    prompt: str,
    max_attempts: int = 5,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return transport.complete(prompt)
        except TransientLLMError as exc:
            last_exc = exc
            wait = full_jitter_wait(attempt)
            logger.warning(
                "Attempt %d/%d failed (%s) — retrying in %.2fs",
                attempt + 1,
                max_attempts,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Exhausted {max_attempts} attempts") from last_exc


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = FlakyLLMTransport(fail_count=2)
    result = call_llm_with_retry(transport, "What is backoff?")
    logger.info("Result: %s", result)
```

Run it — you'll see two WARNING log lines (attempts 1 and 2) then the success result.

---

## Exercises

1. Modify `FlakyLLMTransport` so it raises `PermanentLLMError` on attempt 3. Confirm the retry decorator does **not** retry it and the error propagates immediately.

2. Implement a `call_with_deadline` function that wraps `call_with_manual_retry` and raises `TimeoutError` if the total elapsed time exceeds a configurable deadline (e.g. 10 s), regardless of remaining attempts. Use `time.monotonic()`.

3. Add a `retry_stats` counter to `call_with_manual_retry` that returns `(result, attempts_used)` as a named tuple. Write a pytest test that asserts `attempts_used == 3` when `fail_count=2`.

4. The `wait_random_exponential` in the worked example uses `multiplier=0.5, max=30`. Explain in comments why `max=30` is preferable to `max=3600` for an interactive agent chat system.

(Solutions: ../solutions/week-03/day-1.md)

---

## Self-check

- [ ] I can explain why naive retry causes retry storms and how jitter prevents it
- [ ] I can distinguish retryable from non-retryable HTTP errors
- [ ] I can implement exponential backoff with full jitter from scratch
- [ ] I can configure `tenacity` with type-based retry predicates and sleep logging
- [ ] I can write a pytest test that verifies retry count without real network calls

---

## Supplementary links

- [AWS Architecture Blog — Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- [tenacity docs](https://tenacity.readthedocs.io/en/latest/)
- [httpx exception hierarchy](https://www.python-httpx.org/exceptions/)
