# Solutions — Week 03 · Day 1: Retries, Backoff, and Jitter

---

## Exercise 1 — Permanent errors not retried

```python
import pytest
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_none
from day1_retry import FlakyLLMTransport, PermanentLLMError, TransientLLMError


class FlakyThenPermanent:
    """Fails transiently twice, then raises PermanentLLMError."""

    def __init__(self) -> None:
        self._count = 0

    def complete(self, prompt: str) -> str:
        self._count += 1
        if self._count <= 2:
            raise TransientLLMError(429, "rate limit")
        raise PermanentLLMError("invalid API key")


def test_permanent_error_not_retried() -> None:
    transport = FlakyThenPermanent()

    # Only TransientLLMError is retryable; PermanentLLMError should propagate immediately
    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        wait=wait_none(),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def call() -> str:
        return transport.complete("test")

    with pytest.raises(PermanentLLMError):
        call()

    # Only 3 calls made: 2 transient + 1 permanent (not retried)
    assert transport._count == 3
```

---

## Exercise 2 — `call_with_deadline`

```python
import time
from typing import NamedTuple


def call_with_deadline(
    transport: FlakyLLMTransport,
    prompt: str,
    deadline_s: float = 10.0,
    max_attempts: int = 5,
) -> str:
    """Raise TimeoutError if total elapsed time exceeds deadline_s."""
    start = time.monotonic()
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        if time.monotonic() - start >= deadline_s:
            raise TimeoutError(
                f"Deadline of {deadline_s}s exceeded after {attempt} attempts"
            )
        try:
            return transport.complete(prompt)
        except TransientLLMError as exc:
            last_exc = exc
            wait = full_jitter_wait(attempt)
            elapsed = time.monotonic() - start
            if elapsed + wait >= deadline_s:
                raise TimeoutError(f"Would exceed deadline; aborting after {attempt+1} attempts") from exc
            time.sleep(wait)

    raise RuntimeError(f"Exhausted {max_attempts} attempts") from last_exc
```

---

## Exercise 3 — `retry_stats` with named tuple

```python
from typing import NamedTuple


class RetryResult(NamedTuple):
    result: str
    attempts_used: int


def call_with_manual_retry_stats(
    transport: FlakyLLMTransport,
    prompt: str,
    max_attempts: int = 5,
) -> RetryResult:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = transport.complete(prompt)
            return RetryResult(result=result, attempts_used=attempt)
        except TransientLLMError as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(full_jitter_wait(attempt - 1))
    raise RuntimeError(f"Exhausted {max_attempts} attempts") from last_exc


def test_attempt_count() -> None:
    transport = FlakyLLMTransport(fail_count=2)
    result = call_with_manual_retry_stats(transport, "Q")
    assert result.attempts_used == 3
    assert "Response" in result.result
```

---

## Exercise 4 — Why `max=30` vs `max=3600`

```python
"""
Why max=30 is preferable for interactive chat:

- max=3600 means a single failed request could stall the user for up to 1 hour
  before the retry fires. Exponential backoff at multiplier=0.5 hits 3600s only
  after many failures, but in theory it can happen.

- For interactive chat, user tolerance for waiting is ~10-30 seconds.
  max=30 guarantees the user sees a response (or an error) within 30 seconds
  per attempt, which enables the UI to show meaningful progress.

- For background batch jobs with no user waiting, max=3600 may be fine —
  it allows time for the API to recover from extended outages.

Rule: max = max_acceptable_user_wait * safety_factor
      For interactive: max ≤ 30s
      For batch: max ≤ SLA / total_attempts
"""
```
