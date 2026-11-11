# Solutions — Week 09 · Day 3

## Exercise 1 — half_open_probe_count

```python
@dataclass
class CircuitBreakerV2:
    name: str
    failure_threshold: int = 5
    recovery_timeout_s: float = 10.0
    half_open_probe_count: int = 3  # NEW: require N successes before closing
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _half_open_successes: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def call(self, fn):
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time > self.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                else:
                    raise CircuitOpenError(f"Circuit {self.name} is OPEN")

        try:
            result = fn()
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._half_open_successes = 0
            if (self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED) and
                    self._failure_count >= self.failure_threshold):
                self._state = CircuitState.OPEN

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_probe_count:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
```

## Exercise 2 — retry_with_backoff

```python
import time
import random
from typing import Callable, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay_s: float = 0.1,
    jitter: bool = True,
) -> T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay_s * (2 ** attempt)
                if jitter:
                    delay *= (0.5 + random.random())
                time.sleep(delay)
    raise last_exc  # type: ignore


def test_retry_call_count():
    call_count = 0

    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("transient")
        return "ok"

    result = retry_with_backoff(flaky, max_retries=3, base_delay_s=0.001, jitter=False)
    assert result == "ok"
    assert call_count == 3  # failed twice, succeeded on third


def test_retry_exhausted():
    import pytest

    def always_fails():
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        retry_with_backoff(always_fails, max_retries=2, base_delay_s=0.001)
```

## Exercise 3 — trace_log_correlator

```python
import json
from datetime import datetime


def trace_log_correlator(log_lines: list[str]) -> dict[str, list[dict]]:
    """Group and sort log lines by trace_id."""
    groups: dict[str, list[dict]] = {}
    for line in log_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = record.get("trace_id", "unknown")
        groups.setdefault(tid, []).append(record)

    for tid in groups:
        groups[tid].sort(key=lambda r: r.get("ts", ""))

    return groups


def test_correlator():
    import json
    from datetime import datetime, timezone

    lines = []
    for i in range(10):
        trace = f"T-{i % 3}"  # 3 distinct traces
        ts = f"2026-01-01T00:00:{i:02d}Z"
        lines.append(json.dumps({"ts": ts, "trace_id": trace, "msg": f"event-{i}"}))

    result = trace_log_correlator(lines)

    assert len(result) == 3
    for tid, events in result.items():
        # verify sorted by ts
        tss = [e["ts"] for e in events]
        assert tss == sorted(tss), f"Trace {tid} not sorted"
```
