# Week 10 · Day 1 Solutions — Latency vs Throughput & TTFT/TBT

---

## Exercise 1 — Measure TTFT on a mock stream

### Approach

TTFT is the elapsed time from when the client sends a request to when it receives the first token. In a real system this includes network round-trip plus prefill time. Here we simulate it with a configurable delay before the first `yield`.

The 100-iteration loop surfaces the distribution shape: p95 TTFT vs median TTFT reveals queuing effects. With a deterministic prefill delay and no jitter you expect a tight distribution; jitter widens it.

### Solution

```python
"""
solutions/week_10/ex1_ttft_measurement.py

Mock streaming TTFT measurement over N iterations.
Reports mean, p50, p95.

Run:
    python ex1_ttft_measurement.py
"""

from __future__ import annotations

import logging
import random
import statistics
import time
from typing import Generator

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class TTFTStats(BaseModel):
    """Aggregated TTFT statistics across multiple measurement runs."""

    n_iterations: int
    mean_ms: float = Field(description="Mean TTFT across all runs")
    p50_ms: float = Field(description="Median TTFT")
    p95_ms: float = Field(description="95th-percentile TTFT")
    min_ms: float
    max_ms: float

    def summary(self) -> str:
        return (
            f"n={self.n_iterations}  "
            f"mean={self.mean_ms:.1f}ms  "
            f"p50={self.p50_ms:.1f}ms  "
            f"p95={self.p95_ms:.1f}ms  "
            f"[{self.min_ms:.1f}, {self.max_ms:.1f}]ms"
        )


# ---------------------------------------------------------------------------
# Mock streaming generator
# ---------------------------------------------------------------------------

def mock_stream(
    n_tokens: int = 20,
    ttft_delay_ms: float = 150.0,
    inter_token_delay_ms: float = 30.0,
    ttft_jitter_ms: float = 20.0,
) -> Generator[str, None, None]:
    """
    Yield `n_tokens` fake tokens, simulating realistic streaming timing.

    Parameters
    ----------
    n_tokens:
        Number of tokens to emit.
    ttft_delay_ms:
        Base delay before the first token (simulates prefill).
    inter_token_delay_ms:
        Delay between consecutive tokens (simulates decode TBT).
    ttft_jitter_ms:
        Uniform random jitter on TTFT (simulates load variation).
    """
    actual_ttft = ttft_delay_ms + random.uniform(-ttft_jitter_ms, ttft_jitter_ms)
    actual_ttft = max(0.0, actual_ttft)
    time.sleep(actual_ttft / 1000.0)

    for i in range(n_tokens):
        yield f"tok{i}"
        if i < n_tokens - 1:
            time.sleep(inter_token_delay_ms / 1000.0)


# ---------------------------------------------------------------------------
# TTFT measurer
# ---------------------------------------------------------------------------

def measure_ttft(stream_gen: Generator[str, None, None]) -> float:
    """
    Consume a streaming generator and return TTFT in milliseconds.

    Measures wall-clock time from the moment iteration begins (request start)
    to the moment the first token is received.
    """
    start = time.perf_counter()
    first_token_time: float | None = None

    for _token in stream_gen:
        if first_token_time is None:
            first_token_time = time.perf_counter()
        # In production we would forward the token to the client here.
        # For measurement we just consume it.

    if first_token_time is None:
        raise RuntimeError("Stream yielded no tokens")

    return (first_token_time - start) * 1000.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0.0–1.0) via linear interpolation."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = p * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac


def run_ttft_benchmark(
    n_iterations: int = 100,
    ttft_delay_ms: float = 150.0,
    inter_token_delay_ms: float = 30.0,
    ttft_jitter_ms: float = 20.0,
) -> TTFTStats:
    """
    Run `n_iterations` mock streaming requests and aggregate TTFT statistics.
    """
    ttft_samples: list[float] = []

    for _ in range(n_iterations):
        gen = mock_stream(
            n_tokens=5,  # short stream: we only care about TTFT, not total latency
            ttft_delay_ms=ttft_delay_ms,
            inter_token_delay_ms=inter_token_delay_ms,
            ttft_jitter_ms=ttft_jitter_ms,
        )
        ttft_ms = measure_ttft(gen)
        ttft_samples.append(ttft_ms)

    return TTFTStats(
        n_iterations=n_iterations,
        mean_ms=statistics.mean(ttft_samples),
        p50_ms=statistics.median(ttft_samples),
        p95_ms=_percentile(ttft_samples, 0.95),
        min_ms=min(ttft_samples),
        max_ms=max(ttft_samples),
    )


def main() -> None:
    logger.info("=== TTFT Benchmark (100 iterations) ===")

    # Baseline: 150ms prefill, 20ms jitter
    stats = run_ttft_benchmark(
        n_iterations=100,
        ttft_delay_ms=150.0,
        inter_token_delay_ms=30.0,
        ttft_jitter_ms=20.0,
    )
    logger.info("Baseline  %s", stats.summary())

    # With higher jitter (heavy load)
    stats_loaded = run_ttft_benchmark(
        n_iterations=100,
        ttft_delay_ms=150.0,
        inter_token_delay_ms=30.0,
        ttft_jitter_ms=80.0,
    )
    logger.info("High jitter %s", stats_loaded.summary())

    logger.info(
        "p95/p50 ratio baseline=%.2f  high-jitter=%.2f  "
        "(higher ratio = tail latency diverges more)",
        stats.p95_ms / stats.p50_ms,
        stats_loaded.p95_ms / stats_loaded.p50_ms,
    )


if __name__ == "__main__":
    main()
```

### Pytest test

```python
"""tests/test_ex1_ttft_measurement.py"""

from __future__ import annotations

import pytest

from solutions.week_10.ex1_ttft_measurement import (
    TTFTStats,
    _percentile,
    measure_ttft,
    mock_stream,
    run_ttft_benchmark,
)


# ---------------------------------------------------------------------------
# Unit tests: _percentile
# ---------------------------------------------------------------------------

def test_percentile_median() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(data, 0.5) == pytest.approx(3.0)


def test_percentile_p95_list_of_100() -> None:
    data = list(range(1, 101))  # 1..100
    # p95 of 1..100 should be ~95.05
    result = _percentile([float(x) for x in data], 0.95)
    assert 94.0 <= result <= 96.0


def test_percentile_empty() -> None:
    assert _percentile([], 0.95) == 0.0


def test_percentile_single_element() -> None:
    assert _percentile([42.0], 0.95) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Unit tests: mock_stream
# ---------------------------------------------------------------------------

def test_mock_stream_yields_correct_count() -> None:
    tokens = list(mock_stream(n_tokens=10, ttft_delay_ms=0.0, inter_token_delay_ms=0.0))
    assert len(tokens) == 10


def test_mock_stream_yields_strings() -> None:
    tokens = list(mock_stream(n_tokens=3, ttft_delay_ms=0.0, inter_token_delay_ms=0.0))
    assert all(isinstance(t, str) for t in tokens)


# ---------------------------------------------------------------------------
# Unit tests: measure_ttft
# ---------------------------------------------------------------------------

def test_measure_ttft_returns_positive_ms() -> None:
    gen = mock_stream(n_tokens=5, ttft_delay_ms=10.0, inter_token_delay_ms=0.0, ttft_jitter_ms=0.0)
    ttft = measure_ttft(gen)
    assert ttft > 0.0


def test_measure_ttft_respects_delay() -> None:
    """TTFT should be at least the configured prefill delay."""
    delay_ms = 30.0
    gen = mock_stream(n_tokens=3, ttft_delay_ms=delay_ms, inter_token_delay_ms=0.0, ttft_jitter_ms=0.0)
    ttft = measure_ttft(gen)
    # Allow 10ms tolerance for system scheduling
    assert ttft >= delay_ms - 10.0


def test_measure_ttft_empty_stream_raises() -> None:
    def empty_gen():
        return
        yield  # noqa: unreachable

    with pytest.raises(RuntimeError, match="no tokens"):
        measure_ttft(empty_gen())


# ---------------------------------------------------------------------------
# Integration test: benchmark
# ---------------------------------------------------------------------------

def test_run_ttft_benchmark_statistics_shape() -> None:
    """Run a small benchmark and verify statistical invariants."""
    stats = run_ttft_benchmark(
        n_iterations=10,
        ttft_delay_ms=20.0,
        inter_token_delay_ms=0.0,
        ttft_jitter_ms=5.0,
    )
    assert isinstance(stats, TTFTStats)
    assert stats.p50_ms <= stats.p95_ms
    assert stats.min_ms <= stats.p50_ms
    assert stats.p95_ms <= stats.max_ms
    assert stats.mean_ms > 0.0


def test_higher_jitter_increases_p95_spread() -> None:
    """p95/p50 ratio should be higher with larger jitter."""
    low_jitter = run_ttft_benchmark(
        n_iterations=30, ttft_delay_ms=50.0, inter_token_delay_ms=0.0, ttft_jitter_ms=2.0
    )
    high_jitter = run_ttft_benchmark(
        n_iterations=30, ttft_delay_ms=50.0, inter_token_delay_ms=0.0, ttft_jitter_ms=40.0
    )
    # High jitter should produce a wider spread (p95 further from p50)
    low_spread = low_jitter.p95_ms - low_jitter.p50_ms
    high_spread = high_jitter.p95_ms - high_jitter.p50_ms
    assert high_spread > low_spread
```

### Key observation

With zero jitter, p95 ≈ p50 because every request takes exactly the same prefill time. As jitter grows (representing variable server load, queue wait, or network), p95 diverges from p50. In production, p95 TTFT under load is typically 3–5x the median — this is why SLAs are written against p95, not mean.

---

## Exercise 2 — Compare latency profiles for two token lengths

### Approach

Two competing forces determine total response latency:

- **TTFT** (dominated by prefill, proportional to prompt length) — same for both short and long responses if the prompt is identical.
- **Decode time** (`output_tokens × TBT`) — scales linearly with output length.

For short outputs, TTFT is a large fraction of total latency. For long outputs, accumulated TBT dominates. The per-token latency metric (total latency / output tokens) falls sharply as output length increases because TTFT is amortized.

### Solution

```python
"""
solutions/week_10/ex2_latency_profiles.py

Compare TTFT vs total-latency contribution for short vs long outputs.
Pure math — no real delays or GPU required.

Run:
    python ex2_latency_profiles.py
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, computed_field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class LatencyProfile(BaseModel):
    """
    Deterministic latency breakdown for one generation scenario.

    All times assume:
      - TTFT is fixed and independent of output length (it depends on *input* length).
      - Each decode step costs exactly TBT ms.
    """

    label: str
    output_tokens: int = Field(gt=0)
    ttft_ms: float = Field(gt=0.0, description="Time to first token in milliseconds")
    tbt_ms: float = Field(gt=0.0, description="Time between tokens in milliseconds")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def decode_time_ms(self) -> float:
        """Total decode time = number of additional tokens × TBT.
        The first token is covered by TTFT, so decode contributes (output_tokens - 1) steps."""
        return max(0, self.output_tokens - 1) * self.tbt_ms

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_latency_ms(self) -> float:
        return self.ttft_ms + self.decode_time_ms

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ttft_fraction(self) -> float:
        """Fraction of total latency attributable to TTFT."""
        return self.ttft_ms / self.total_latency_ms if self.total_latency_ms > 0 else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def per_token_latency_ms(self) -> float:
        """Average wall-clock time per output token (total / output_tokens)."""
        return self.total_latency_ms / self.output_tokens

    @computed_field  # type: ignore[prop-decorator]
    @property
    def throughput_tokens_per_s(self) -> float:
        return (self.output_tokens / self.total_latency_ms) * 1000.0

    def summary(self) -> str:
        return (
            f"{self.label:30s}  "
            f"out={self.output_tokens:5d} tok  "
            f"TTFT={self.ttft_ms:.0f}ms  "
            f"decode={self.decode_time_ms:.0f}ms  "
            f"total={self.total_latency_ms:.0f}ms  "
            f"TTFT%={self.ttft_fraction * 100:.0f}%  "
            f"per-tok={self.per_token_latency_ms:.1f}ms  "
            f"tput={self.throughput_tokens_per_s:.1f} tok/s"
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

TTFT_MS = 200.0   # fixed prefill latency
TBT_MS = 30.0     # decode step latency

SHORT_TOKENS = 100
LONG_TOKENS = 1000


def run_comparison() -> None:
    short = LatencyProfile(
        label="short (100 tokens)",
        output_tokens=SHORT_TOKENS,
        ttft_ms=TTFT_MS,
        tbt_ms=TBT_MS,
    )
    long = LatencyProfile(
        label="long (1000 tokens)",
        output_tokens=LONG_TOKENS,
        ttft_ms=TTFT_MS,
        tbt_ms=TBT_MS,
    )

    logger.info("=== Latency Profile Comparison ===")
    logger.info("TTFT=%.0f ms (fixed), TBT=%.0f ms per token", TTFT_MS, TBT_MS)
    logger.info("")
    logger.info(short.summary())
    logger.info(long.summary())
    logger.info("")

    logger.info(
        "TTFT dominates short output: %.0f%% of total latency",
        short.ttft_fraction * 100,
    )
    logger.info(
        "TBT dominates long output: %.0f%% of total latency is decode",
        (1.0 - long.ttft_fraction) * 100,
    )
    logger.info(
        "Per-token latency improves %.1fx for long vs short output "
        "(TTFT amortized over more tokens)",
        short.per_token_latency_ms / long.per_token_latency_ms,
    )
    logger.info(
        "Throughput improves %.1fx for long output (same TTFT overhead, more tokens per request)",
        long.throughput_tokens_per_s / short.throughput_tokens_per_s,
    )

    # Show the crossover point: at what output length does TTFT < 50% of total?
    logger.info("\n=== TTFT dominance crossover ===")
    logger.info("(output tokens at which TTFT fraction drops below 50%%)")
    for n_tok in [10, 20, 50, 100, 200, 500, 1000]:
        p = LatencyProfile(label="", output_tokens=n_tok, ttft_ms=TTFT_MS, tbt_ms=TBT_MS)
        marker = " <-- crossover" if 40.0 < p.ttft_fraction * 100 < 55.0 else ""
        logger.info(
            "  %5d tokens: TTFT=%4.0f%%  total=%5.0f ms%s",
            n_tok,
            p.ttft_fraction * 100,
            p.total_latency_ms,
            marker,
        )


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
```

### Expected output (approximate)

```
INFO === Latency Profile Comparison ===
INFO TTFT=200 ms (fixed), TBT=30 ms per token

INFO short (100 tokens)                out=  100 tok  TTFT=200ms  decode=2970ms  total=3170ms  TTFT%=6%   per-tok=31.7ms  tput=31.5 tok/s
INFO long (1000 tokens)                out= 1000 tok  TTFT=200ms  decode=29970ms total=30170ms TTFT%=1%   per-tok=30.2ms  tput=33.1 tok/s

INFO TTFT dominates short output: 6% of total latency
INFO TBT dominates long output: 99% of total latency is decode
INFO Per-token latency improves 1.1x for long vs short output
INFO Throughput improves 1.1x for long output

INFO === TTFT dominance crossover ===
     10 tokens: TTFT= 60%  total=  470 ms
     20 tokens: TTFT= 40%  total=  770 ms <-- crossover (~7 tokens)
```

### Explanation

The crossover where TTFT = 50% of total latency occurs at `output_tokens = TTFT_MS / TBT_MS + 1`. With TTFT=200ms and TBT=30ms that is approximately 8 tokens. Beyond 8 output tokens, TBT accumulation dominates. At 100 tokens, TTFT is only 6% of total time. At 1000 tokens, it is under 1%.

**Implication for optimization:**
- For short-output use cases (classification, slot filling, yes/no answers): reduce TTFT. Prompt caching and speculative decoding help most.
- For long-output use cases (generation, summarization, code): reduce TBT. Quantization (less data to load from VRAM per decode step) and larger batch sizes help most.

### Pytest tests

```python
"""tests/test_ex2_latency_profiles.py"""

from __future__ import annotations

import pytest

from solutions.week_10.ex2_latency_profiles import LatencyProfile


def make_profile(output_tokens: int, ttft_ms: float = 200.0, tbt_ms: float = 30.0) -> LatencyProfile:
    return LatencyProfile(label="test", output_tokens=output_tokens, ttft_ms=ttft_ms, tbt_ms=tbt_ms)


class TestLatencyProfile:
    def test_total_latency_single_token(self) -> None:
        """With 1 output token there are 0 decode steps after TTFT."""
        p = make_profile(output_tokens=1, ttft_ms=100.0, tbt_ms=50.0)
        assert p.total_latency_ms == pytest.approx(100.0)

    def test_total_latency_two_tokens(self) -> None:
        """TTFT + 1 TBT step for 2 tokens."""
        p = make_profile(output_tokens=2, ttft_ms=100.0, tbt_ms=50.0)
        assert p.total_latency_ms == pytest.approx(150.0)

    def test_total_latency_100_tokens(self) -> None:
        p = make_profile(output_tokens=100, ttft_ms=200.0, tbt_ms=30.0)
        expected = 200.0 + 99 * 30.0  # TTFT + 99 decode steps
        assert p.total_latency_ms == pytest.approx(expected)

    def test_ttft_fraction_long_output_is_small(self) -> None:
        """For 1000 output tokens with TTFT=200ms, TTFT fraction < 2%."""
        p = make_profile(output_tokens=1000, ttft_ms=200.0, tbt_ms=30.0)
        assert p.ttft_fraction < 0.02

    def test_ttft_fraction_short_output_is_large(self) -> None:
        """For 5 output tokens with TTFT=200ms and TBT=30ms, TTFT > 50%."""
        p = make_profile(output_tokens=5, ttft_ms=200.0, tbt_ms=30.0)
        assert p.ttft_fraction > 0.5

    def test_per_token_latency_approaches_tbt_for_long_output(self) -> None:
        """As output grows, per-token latency converges toward TBT (TTFT amortized)."""
        p_short = make_profile(output_tokens=10, ttft_ms=200.0, tbt_ms=30.0)
        p_long = make_profile(output_tokens=10000, ttft_ms=200.0, tbt_ms=30.0)
        assert p_long.per_token_latency_ms < p_short.per_token_latency_ms
        assert p_long.per_token_latency_ms == pytest.approx(30.0, abs=0.1)

    def test_throughput_tokens_per_s_formula(self) -> None:
        p = make_profile(output_tokens=100, ttft_ms=200.0, tbt_ms=30.0)
        expected_tput = (100 / p.total_latency_ms) * 1000.0
        assert p.throughput_tokens_per_s == pytest.approx(expected_tput)

    def test_identical_ttft_across_output_lengths(self) -> None:
        """TTFT is independent of output length — only prefill (input) determines it."""
        p_short = make_profile(output_tokens=10, ttft_ms=200.0, tbt_ms=30.0)
        p_long = make_profile(output_tokens=1000, ttft_ms=200.0, tbt_ms=30.0)
        assert p_short.ttft_ms == p_long.ttft_ms
```

---

## Exercise 3 — Compute throughput from a batch of requests

### Approach

System throughput is not the average of per-request throughputs. It must be computed as total tokens generated divided by wall-clock time from first request start to last request end. For concurrent requests this is shorter than sequential time; for sequential requests it equals the sum of individual durations.

### Solution

```python
"""
solutions/week_10/ex3_batch_throughput.py

Compute system-level throughput from a batch of request records.
Pydantic models for input and output.

Run:
    python ex3_batch_throughput.py
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, model_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

class RequestRecord(BaseModel):
    """Timing record for a single completed inference request."""

    request_id: str
    input_tokens: int = Field(gt=0)
    output_tokens: int = Field(gt=0)
    start_time_ms: float = Field(ge=0.0, description="Wall-clock start time in ms (epoch or relative)")
    duration_ms: float = Field(gt=0.0, description="Total duration from request start to last token")

    @property
    def end_time_ms(self) -> float:
        return self.start_time_ms + self.duration_ms

    @property
    def per_request_throughput(self) -> float:
        """Output tokens per second for this individual request."""
        return (self.output_tokens / self.duration_ms) * 1000.0


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class BatchStats(BaseModel):
    """
    System-level throughput and latency statistics for a batch of requests.

    System throughput = total_output_tokens / window_duration_s
    where window_duration_s spans from earliest start to latest end.
    """

    n_requests: int
    total_input_tokens: int
    total_output_tokens: int
    window_start_ms: float = Field(description="Earliest request start time")
    window_end_ms: float = Field(description="Latest request end time")
    window_duration_ms: float = Field(description="window_end - window_start")
    system_throughput_tokens_per_s: float = Field(
        description="Total output tokens / window duration"
    )
    system_throughput_requests_per_s: float
    p50_latency_ms: float = Field(description="Median per-request duration")
    p95_latency_ms: float = Field(description="p95 per-request duration")
    mean_per_request_throughput: float = Field(
        description="Average of individual request token/s rates"
    )

    def summary(self) -> str:
        return (
            f"n={self.n_requests}  "
            f"total_out={self.total_output_tokens} tok  "
            f"window={self.window_duration_ms:.0f}ms  "
            f"sys_tput={self.system_throughput_tokens_per_s:.1f} tok/s  "
            f"sys_rps={self.system_throughput_requests_per_s:.2f} req/s  "
            f"p50_lat={self.p50_latency_ms:.0f}ms  "
            f"p95_lat={self.p95_latency_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sd = sorted(data)
    idx = p * (len(sd) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sd) - 1)
    frac = idx - lo
    return sd[lo] * (1.0 - frac) + sd[hi] * frac


def compute_batch_stats(records: list[RequestRecord]) -> BatchStats:
    """
    Compute system-level throughput and latency statistics from request records.

    Parameters
    ----------
    records:
        List of completed request records. start_time_ms can be relative
        (e.g., offset from experiment start) or absolute.

    Returns
    -------
    BatchStats:
        Aggregated metrics.

    Notes
    -----
    System throughput is measured over the entire observation window
    (earliest request start → latest request end), not just active time.
    This is the metric that matters for capacity planning.
    """
    if not records:
        raise ValueError("records list is empty")

    window_start = min(r.start_time_ms for r in records)
    window_end = max(r.end_time_ms for r in records)
    window_ms = window_end - window_start

    if window_ms <= 0:
        raise ValueError("window_duration_ms must be positive")

    total_out = sum(r.output_tokens for r in records)
    total_in = sum(r.input_tokens for r in records)
    durations = [r.duration_ms for r in records]

    sys_tput_tps = (total_out / window_ms) * 1000.0
    sys_rps = (len(records) / window_ms) * 1000.0
    mean_per_req = sum(r.per_request_throughput for r in records) / len(records)

    return BatchStats(
        n_requests=len(records),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        window_start_ms=window_start,
        window_end_ms=window_end,
        window_duration_ms=window_ms,
        system_throughput_tokens_per_s=sys_tput_tps,
        system_throughput_requests_per_s=sys_rps,
        p50_latency_ms=_percentile(durations, 0.50),
        p95_latency_ms=_percentile(durations, 0.95),
        mean_per_request_throughput=mean_per_req,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # Simulate 5 overlapping requests (concurrent serving)
    # start_time_ms is relative to experiment start
    records: list[RequestRecord] = [
        RequestRecord(request_id="r1", input_tokens=128, output_tokens=150, start_time_ms=0.0,   duration_ms=4800.0),
        RequestRecord(request_id="r2", input_tokens=256, output_tokens=300, start_time_ms=50.0,  duration_ms=9200.0),
        RequestRecord(request_id="r3", input_tokens=64,  output_tokens=80,  start_time_ms=100.0, duration_ms=2600.0),
        RequestRecord(request_id="r4", input_tokens=512, output_tokens=400, start_time_ms=200.0, duration_ms=12200.0),
        RequestRecord(request_id="r5", input_tokens=128, output_tokens=200, start_time_ms=300.0, duration_ms=6300.0),
    ]

    stats = compute_batch_stats(records)

    logger.info("=== Batch Throughput Analysis ===")
    logger.info(stats.summary())
    logger.info("")
    logger.info("Per-request throughputs:")
    for r in records:
        logger.info(
            "  %s: %d out-tokens in %.0f ms = %.1f tok/s",
            r.request_id,
            r.output_tokens,
            r.duration_ms,
            r.per_request_throughput,
        )

    logger.info("")
    logger.info(
        "System throughput (%.1f tok/s) vs mean per-request (%.1f tok/s):",
        stats.system_throughput_tokens_per_s,
        stats.mean_per_request_throughput,
    )
    logger.info(
        "System throughput is higher because requests overlapped — "
        "the window is shorter than the sum of individual durations."
    )


if __name__ == "__main__":
    main()
```

### Key insight: system throughput vs per-request throughput

System throughput is computed over the observation window (first start → last end), not the sum of individual request durations. When requests overlap (concurrent serving), the window is shorter than the sum — so system throughput is higher than the mean per-request rate. This is the metric that determines actual capacity and cost per token.

If all 5 requests ran sequentially, the window would be 4800+9200+2600+12200+6300 = 35,100 ms. Because they overlap, the window is ~12,400 ms (first start=0, last end=200+12200=12400). System throughput improves by the overlap factor.

### Pytest tests

```python
"""tests/test_ex3_batch_throughput.py"""

from __future__ import annotations

import pytest

from solutions.week_10.ex3_batch_throughput import (
    BatchStats,
    RequestRecord,
    _percentile,
    compute_batch_stats,
)


def make_record(
    rid: str,
    output_tokens: int,
    start_ms: float,
    duration_ms: float,
    input_tokens: int = 64,
) -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        start_time_ms=start_ms,
        duration_ms=duration_ms,
    )


class TestRequestRecord:
    def test_end_time(self) -> None:
        r = make_record("r1", output_tokens=100, start_ms=100.0, duration_ms=500.0)
        assert r.end_time_ms == pytest.approx(600.0)

    def test_per_request_throughput(self) -> None:
        r = make_record("r1", output_tokens=100, start_ms=0.0, duration_ms=1000.0)
        assert r.per_request_throughput == pytest.approx(100.0)  # 100 tokens/s


class TestComputeBatchStats:
    def test_single_request(self) -> None:
        records = [make_record("r1", output_tokens=100, start_ms=0.0, duration_ms=1000.0)]
        stats = compute_batch_stats(records)
        assert stats.n_requests == 1
        assert stats.total_output_tokens == 100
        assert stats.window_duration_ms == pytest.approx(1000.0)
        assert stats.system_throughput_tokens_per_s == pytest.approx(100.0)

    def test_sequential_requests_window(self) -> None:
        """Sequential requests: window = sum of durations."""
        records = [
            make_record("r1", output_tokens=100, start_ms=0.0,    duration_ms=1000.0),
            make_record("r2", output_tokens=100, start_ms=1000.0, duration_ms=1000.0),
        ]
        stats = compute_batch_stats(records)
        assert stats.window_duration_ms == pytest.approx(2000.0)
        assert stats.system_throughput_tokens_per_s == pytest.approx(100.0)

    def test_concurrent_requests_shorter_window(self) -> None:
        """Overlapping requests: window < sum of durations, higher throughput."""
        records = [
            make_record("r1", output_tokens=100, start_ms=0.0,  duration_ms=1000.0),
            make_record("r2", output_tokens=100, start_ms=100.0, duration_ms=1000.0),
        ]
        stats = compute_batch_stats(records)
        # Window: 0 → 1100ms (not 2000ms)
        assert stats.window_duration_ms == pytest.approx(1100.0)
        # System throughput higher than per-request rates
        assert stats.system_throughput_tokens_per_s > 100.0

    def test_p95_latency_with_straggler(self) -> None:
        """p95 should capture the straggler request."""
        records = [make_record(f"r{i}", 100, float(i * 100), 500.0) for i in range(9)]
        records.append(make_record("r9", 100, 0.0, 5000.0))  # straggler
        stats = compute_batch_stats(records)
        assert stats.p95_latency_ms > 500.0

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_batch_stats([])

    def test_total_tokens_summed_correctly(self) -> None:
        records = [
            make_record("r1", output_tokens=100, start_ms=0.0, duration_ms=1000.0, input_tokens=50),
            make_record("r2", output_tokens=200, start_ms=0.0, duration_ms=2000.0, input_tokens=80),
        ]
        stats = compute_batch_stats(records)
        assert stats.total_output_tokens == 300
        assert stats.total_input_tokens == 130

    def test_returns_batch_stats_type(self) -> None:
        records = [make_record("r1", 50, 0.0, 500.0)]
        stats = compute_batch_stats(records)
        assert isinstance(stats, BatchStats)
```
