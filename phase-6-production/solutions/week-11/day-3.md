# Week 11 · Day 3 Solutions

## Exercise 1

LLM token histogram with model/token_type labels and custom buckets.

```python
"""
llm_metrics.py — Prometheus metrics for LLM token usage.
"""
import logging
import os
from typing import Literal

import anthropic
from prometheus_client import Histogram, start_http_server

logger = logging.getLogger(__name__)

ModelName = Literal["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
TokenType = Literal["input", "output"]

# Custom buckets: designed for LLM tokens — granular at the low end,
# wide at the high end where Haiku vs Opus usage diverges significantly.
_TOKEN_BUCKETS: tuple[float, ...] = (
    10, 50, 100, 500, 1000, 2000, 4000, 8000
)

llm_tokens_per_request = Histogram(
    name="llm_tokens_per_request",
    documentation="Number of tokens used per LLM request",
    labelnames=["model", "token_type"],
    buckets=_TOKEN_BUCKETS,
)


def record_token_usage(
    model: ModelName,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record input and output token counts for a completed LLM call."""
    llm_tokens_per_request.labels(model=model, token_type="input").observe(
        input_tokens
    )
    llm_tokens_per_request.labels(model=model, token_type="output").observe(
        output_tokens
    )
    logger.debug(
        "token usage recorded",
        extra={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )


def call_llm_with_metrics(prompt: str, model: ModelName = "claude-haiku-5-5") -> str:
    """Call the Anthropic API and record token metrics."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    record_token_usage(
        model=model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
    return message.content[0].text  # type: ignore[index]


# --- unit tests (LLM mocked) ---
import pytest
from unittest.mock import MagicMock, patch
from prometheus_client import REGISTRY


def test_record_token_usage_increments_histogram() -> None:
    # Observe and verify counts via the histogram's sample method
    before = llm_tokens_per_request.labels(
        model="claude-haiku-5-5", token_type="input"
    )._sum.get()
    record_token_usage(
        model="claude-haiku-5-5", input_tokens=100, output_tokens=50
    )
    after = llm_tokens_per_request.labels(
        model="claude-haiku-5-5", token_type="input"
    )._sum.get()
    assert after == before + 100


def test_call_llm_with_metrics_mocked() -> None:
    mock_usage = MagicMock()
    mock_usage.input_tokens = 80
    mock_usage.output_tokens = 40

    mock_content = MagicMock()
    mock_content.text = "Hello, world!"

    mock_message = MagicMock()
    mock_message.usage = mock_usage
    mock_message.content = [mock_content]

    with patch("anthropic.Anthropic") as MockClient:
        MockClient.return_value.messages.create.return_value = mock_message
        result = call_llm_with_metrics("Say hello", model="claude-haiku-5-5")

    assert result == "Hello, world!"
    MockClient.return_value.messages.create.assert_called_once()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    start_http_server(9090)
    logger.info("metrics server running on :9090")
    # Call with real API key to populate metrics
    # call_llm_with_metrics("What is 2+2?")
```

---

## Exercise 2

PromQL query for 5xx error rate as a percentage over the last 5 minutes.

```python
"""
promql_queries.py — Reference PromQL queries for request error rate.

Assumes metrics from a standard HTTP middleware that exposes:
    http_requests_total{method, path, status_code}

The histogram variant (http_request_duration_seconds_count) also works.
"""

# --- PromQL: 5xx error rate as a percentage of all requests (last 5 min) ---
#
# Numerator:   rate of all 5xx responses over 5 min
# Denominator: rate of all responses over 5 min
# Result:      percentage (0–100), per (method, path) label pair
#
# PromQL:
#
#   100 * (
#     sum by (method, path) (
#       rate(http_requests_total{status_code=~"5.."}[5m])
#     )
#   ) / (
#     sum by (method, path) (
#       rate(http_requests_total[5m])
#     )
#   )
#
# Notes:
#   - status_code=~"5.." uses a regex to match 500–599.
#   - Use `or vector(0)` on the denominator if you want 0% instead of no-data
#     on paths with zero traffic.
#   - For an alert: wrap in `> 1` to fire when error rate exceeds 1%.

ERROR_RATE_QUERY = """
100 * (
  sum by (method, path) (
    rate(http_requests_total{status_code=~"5.."}[5m])
  )
) / (
  sum by (method, path) (
    rate(http_requests_total[5m])
  )
)
""".strip()

# Variant: overall error rate (no per-path breakdown)
OVERALL_ERROR_RATE_QUERY = """
100 * sum(rate(http_requests_total{status_code=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))
""".strip()

# Alert rule (Alertmanager-style YAML representation, as a Python string)
ALERT_RULE = """
groups:
  - name: http_errors
    rules:
      - alert: HighErrorRate
        expr: |
          100 * (
            sum by (method, path) (
              rate(http_requests_total{status_code=~"5.."}[5m])
            )
          ) / (
            sum by (method, path) (
              rate(http_requests_total[5m])
            )
          ) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High 5xx error rate on {{ $labels.method }} {{ $labels.path }}"
          description: "Error rate is {{ $value | humanize }}% over the last 5 minutes."
"""

if __name__ == "__main__":
    print("5xx error rate query:")
    print(ERROR_RATE_QUERY)
    print("\nOverall error rate query:")
    print(OVERALL_ERROR_RATE_QUERY)
```

---

## Exercise 3

USE method metrics for a connection pool: utilisation, saturation, errors.

```python
"""
connection_pool_metrics.py — Prometheus USE-method metrics for a connection pool.

USE method:
  Utilisation  → connections currently in use (gauge)
  Saturation   → time spent waiting for a connection (histogram)
  Errors        → failed acquisition attempts (counter)
"""
import contextlib
import logging
import time
from collections.abc import Generator
from threading import Semaphore

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

# --- Metrics ---

connections_total = Gauge(
    name="connections_total",
    documentation="Current number of connections in use (utilisation)",
)

connections_max = Gauge(
    name="connections_max",
    documentation="Maximum pool size (capacity)",
)

connection_wait_seconds = Histogram(
    name="connection_wait_seconds",
    documentation="Time (seconds) spent waiting to acquire a connection (saturation)",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

connection_errors_total = Counter(
    name="connection_errors_total",
    documentation="Total failed connection acquisition attempts (errors)",
)


# --- Pool implementation ---

class InstrumentedConnectionPool:
    """A semaphore-backed pool that exports USE-method Prometheus metrics."""

    def __init__(self, pool_size: int = 10) -> None:
        self._pool_size = pool_size
        self._semaphore = Semaphore(pool_size)
        connections_max.set(pool_size)
        logger.info("connection pool initialised", extra={"pool_size": pool_size})

    @contextlib.contextmanager
    def acquire(self, timeout: float = 5.0) -> Generator[None, None, None]:
        """Acquire a connection from the pool, yielding control to the caller.

        Metrics updated:
          - connection_wait_seconds: time blocked on acquire
          - connections_total: incremented/decremented around the critical section
          - connection_errors_total: incremented on timeout
        """
        start = time.perf_counter()
        acquired = self._semaphore.acquire(timeout=timeout)
        wait_secs = time.perf_counter() - start

        connection_wait_seconds.observe(wait_secs)

        if not acquired:
            connection_errors_total.inc()
            logger.error(
                "connection pool exhausted",
                extra={"timeout": timeout, "pool_size": self._pool_size},
            )
            raise TimeoutError(
                f"Could not acquire connection within {timeout}s"
            )

        connections_total.inc()
        logger.debug("connection acquired", extra={"wait_secs": round(wait_secs, 4)})
        try:
            yield
        finally:
            connections_total.dec()
            self._semaphore.release()
            logger.debug("connection released")


# --- unit tests ---
import pytest
from unittest.mock import patch


def test_acquire_increments_and_decrements_gauge() -> None:
    pool = InstrumentedConnectionPool(pool_size=2)
    initial = connections_total._value.get()
    with pool.acquire():
        assert connections_total._value.get() == initial + 1
    assert connections_total._value.get() == initial


def test_acquire_timeout_increments_error_counter() -> None:
    pool = InstrumentedConnectionPool(pool_size=1)
    error_before = connection_errors_total._value.get()

    # Drain the pool
    pool._semaphore.acquire()
    try:
        with pytest.raises(TimeoutError):
            with pool.acquire(timeout=0.01):
                pass
    finally:
        pool._semaphore.release()

    assert connection_errors_total._value.get() == error_before + 1


def test_pool_size_sets_max_gauge() -> None:
    InstrumentedConnectionPool(pool_size=5)
    assert connections_max._value.get() == 5


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    start_http_server(9091)
    pool = InstrumentedConnectionPool(pool_size=3)
    logger.info("pool ready, metrics at :9091/metrics")
```

---

## Exercise 4

Suggest histogram bucket boundaries from observed values using percentile-based analysis.

```python
"""
bucket_advisor.py — Suggest Prometheus histogram bucket boundaries
from a list of observed values using percentile-based analysis.

The output bucket list can be pasted directly into a Histogram(buckets=...) call.
"""
import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0)


def suggest_buckets(
    observations: Sequence[float],
    percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES,
    round_to_sigfigs: int = 2,
) -> list[float]:
    """Suggest histogram bucket boundaries from observed values.

    Strategy:
      1. Compute the specified percentiles from the data.
      2. Round each to `round_to_sigfigs` significant figures for clean labels.
      3. Deduplicate and sort.
      4. Append +Inf is handled automatically by prometheus_client.

    Args:
        observations: Non-empty sequence of observed metric values.
        percentiles:  Percentile points to use as bucket boundaries (0–100).
        round_to_sigfigs: Significant figures to round each boundary to.

    Returns:
        Sorted, deduplicated list of bucket boundaries.

    Raises:
        ValueError: If `observations` is empty.
    """
    if not observations:
        raise ValueError("observations must not be empty")

    arr = np.array(observations, dtype=float)
    raw_boundaries = np.percentile(arr, list(percentiles)).tolist()

    def _round_sigfigs(value: float, sigfigs: int) -> float:
        if value == 0:
            return 0.0
        magnitude = math.floor(math.log10(abs(value)))
        factor = 10 ** (sigfigs - 1 - magnitude)
        return round(value * factor) / factor

    rounded = [_round_sigfigs(b, round_to_sigfigs) for b in raw_boundaries]
    # Remove duplicates (can happen when data clusters) and sort
    buckets = sorted(set(rounded))

    logger.info(
        "bucket suggestion",
        extra={
            "n_observations": len(observations),
            "percentiles": percentiles,
            "suggested_buckets": buckets,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        },
    )
    return buckets


def print_bucket_report(
    observations: Sequence[float],
    metric_name: str = "my_metric",
) -> None:
    """Print a human-readable bucket suggestion with coverage stats."""
    buckets = suggest_buckets(observations)
    arr = np.array(observations, dtype=float)

    print(f"\n=== Bucket suggestion for '{metric_name}' ===")
    print(f"Observations: {len(observations)} values")
    print(f"  min={arr.min():.3g}  mean={arr.mean():.3g}  max={arr.max():.3g}")
    print(f"\nSuggested buckets: {buckets}")
    print("\nCoverage per bucket:")
    for b in buckets:
        pct_below = float(np.mean(arr <= b) * 100)
        print(f"  <= {b:>10.3g}  :  {pct_below:5.1f}% of observations")
    print(f"\nPrometheus usage:")
    print(f"  {metric_name} = Histogram(")
    print(f'      name="{metric_name}",')
    print(f'      documentation="...",')
    print(f"      buckets={buckets},")
    print(f"  )")


# --- unit tests ---
import pytest


def test_suggest_buckets_sorted_and_unique() -> None:
    obs = list(range(1, 101))  # 1–100
    buckets = suggest_buckets(obs)
    assert buckets == sorted(set(buckets))
    assert len(buckets) >= 1


def test_suggest_buckets_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        suggest_buckets([])


def test_suggest_buckets_single_value() -> None:
    buckets = suggest_buckets([42.0])
    assert len(buckets) >= 1
    assert all(b > 0 for b in buckets)


def test_suggest_buckets_all_same_value() -> None:
    buckets = suggest_buckets([10.0] * 100)
    # All percentiles equal → deduplication yields a single bucket
    assert len(buckets) == 1
    assert buckets[0] == 10.0


def test_suggest_buckets_custom_percentiles() -> None:
    obs = list(range(1, 1001))
    buckets = suggest_buckets(obs, percentiles=(50.0, 99.0))
    assert len(buckets) <= 2  # may collapse if rounded to same value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import random
    # Simulate LLM response-time observations (ms), log-normal distribution
    sample = [random.lognormvariate(5, 1) for _ in range(2000)]
    print_bucket_report(sample, metric_name="llm_response_time_ms")
```
