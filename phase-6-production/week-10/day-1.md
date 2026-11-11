# Phase 6 · Week 10 · Day 1 — Latency vs Throughput + TTFT/TBT Metrics

> Objective: Measure and reason about inference latency (TTFT, TBT) and throughput as competing pressures in production LLM serving.  ·  Time: ~1 hr  ·  Prereqs: [[../week-09/day-5.md]]

---

## Concept

### The two forces in inference serving

Every production LLM deployment must satisfy two distinct — and often opposing — goals:

**Latency** is how long a single user waits. It has two sub-metrics:
- **TTFT (Time to First Token)**: milliseconds from sending a request to receiving the first token. Dominates perceived responsiveness. A user typing in a chat UI feels this directly.
- **TBT (Time Between Tokens)**: milliseconds between successive tokens during generation. If TBT is too high, the stream stutters or feels slow even if TTFT was fast.

**Throughput** is how many tokens (or requests) the system delivers per second across all users. A high-throughput server processes many requests cheaply; a low-latency server answers each one fast.

### Why they conflict

Batching is the primary mechanism for improving throughput: process multiple requests together so GPU computation is amortized. But batching *increases latency* — a request may wait in queue for the batch to fill, and TTFT rises. Serving requests one-by-one (low batch size) minimizes latency but wastes GPU capacity.

This is not a flaw — it is a fundamental tradeoff governed by use case:

| Use case | Priority | Typical constraint |
|---|---|---|
| Chat / copilot | Low TTFT, smooth TBT | User waiting at keyboard |
| Batch summarization | High throughput | Cost per token matters |
| RAG pipeline (automated) | Throughput | No human in the loop |
| Code completion | Very low TTFT | Sub-100 ms feel instant |

Understanding which regime you are in determines every optimization decision downstream.

---

## How it works internally

### Autoregressive decoding — two phases

Modern transformer inference has two structurally different phases:

**Prefill phase**
The model processes the entire input prompt in one forward pass. This is a large matrix multiplication (prompt tokens × model dimensions) that is compute-bound: the GPU is doing arithmetic, not waiting for memory. Output: the KV cache — a stored representation of every token's key and value vectors across all attention layers. TTFT is dominated by prefill time.

- Prefill time scales roughly with prompt length.
- Long prompts mean high TTFT.
- A 4096-token prompt takes ~4× longer to prefill than a 1024-token prompt.

**Decode phase**
After prefill, the model generates tokens one at a time. Each step performs a forward pass for exactly one new token, but must read the entire KV cache from GPU memory to compute attention over past context. This is memory-bandwidth bound: the GPU is waiting for data, not doing arithmetic. TBT is dominated by decode time.

- Decode time per token is roughly constant regardless of generation length (because the KV cache already exists).
- It scales with model size (larger model = larger KV cache = more bandwidth required).
- This is why TBT does not accelerate mid-generation: each new token still needs the full cache.

### Why this matters for optimization

- Reduce TTFT: make prefill faster — smaller prompts, speculative decoding, model distillation, prompt caching (reuse KV cache across requests with shared prefixes).
- Reduce TBT: make decode faster — quantization (smaller weights = less bandwidth), batching (amortize memory reads across multiple decode steps), speculative decoding.
- These interventions interact. Quantization helps TBT more than TTFT. Prompt caching helps TTFT more than TBT.

---

## Worked example

```python
"""
day_1_inference_metrics.py

Simulate a streaming LLM response and compute TTFT, per-token TBT,
and p50/p95 latency statistics — no GPU required.

Run:
    python day_1_inference_metrics.py
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class LatencyBenchmark(BaseModel):
    """Aggregated latency statistics from a single inference run."""

    request_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float = Field(..., description="Time to first token in milliseconds")
    tbt_p50_ms: float = Field(..., description="Median time-between-tokens in ms")
    tbt_p95_ms: float = Field(..., description="p95 time-between-tokens in ms")
    total_latency_ms: float = Field(..., description="Wall time from request to last token")
    tokens_per_second: float = Field(..., description="Generated tokens / total latency in seconds")

    def summary(self) -> str:
        return (
            f"[{self.request_id}] "
            f"TTFT={self.ttft_ms:.1f}ms  "
            f"TBT p50={self.tbt_p50_ms:.1f}ms p95={self.tbt_p95_ms:.1f}ms  "
            f"total={self.total_latency_ms:.0f}ms  "
            f"throughput={self.tokens_per_second:.1f} tok/s"
        )


# ---------------------------------------------------------------------------
# Streaming simulator
# ---------------------------------------------------------------------------

@dataclass
class StreamSimulator:
    """
    Mock a streaming token response.

    Parameters
    ----------
    prefill_ms:
        Simulated prefill latency (= TTFT when no queuing).
    tbt_ms:
        Base time between tokens during decode.
    tbt_jitter_ms:
        Random jitter added to each TBT (simulates real-world variance).
    """

    prefill_ms: float = 120.0
    tbt_ms: float = 30.0
    tbt_jitter_ms: float = 5.0

    def stream(self, tokens: list[str]) -> Generator[str, None, None]:
        """Yield tokens with realistic timing delays."""
        import random

        # Simulate prefill: block for prefill_ms before first token
        time.sleep(self.prefill_ms / 1000.0)
        for token in tokens:
            yield token
            jitter = random.uniform(-self.tbt_jitter_ms, self.tbt_jitter_ms)
            delay = max(0.0, self.tbt_ms + jitter) / 1000.0
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class InferenceMetrics:
    """
    Wrap a token stream and record TTFT and per-token TBT timestamps.

    Usage
    -----
        simulator = StreamSimulator(prefill_ms=100, tbt_ms=25)
        tokens = ["The", " quick", " brown", " fox"]
        metrics = InferenceMetrics(request_id="req-001", prompt_tokens=32)

        for token in metrics.observe(simulator.stream(tokens)):
            pass  # in a real app, send token to client here

        benchmark = metrics.finalize(generated_tokens=len(tokens))
        print(benchmark.summary())
    """

    def __init__(self, request_id: str, prompt_tokens: int) -> None:
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self._start: float | None = None
        self._first_token_time: float | None = None
        self._token_timestamps: list[float] = []

    def observe(self, stream: Iterator[str]) -> Generator[str, None, None]:
        """Wrap a token iterator, recording wall-clock timestamps."""
        self._start = time.perf_counter()
        for token in stream:
            now = time.perf_counter()
            if self._first_token_time is None:
                self._first_token_time = now
            self._token_timestamps.append(now)
            yield token

    def finalize(self, generated_tokens: int) -> LatencyBenchmark:
        """Compute aggregated statistics after the stream has ended."""
        if self._start is None or self._first_token_time is None:
            raise RuntimeError("observe() must be called before finalize()")

        end = self._token_timestamps[-1] if self._token_timestamps else self._first_token_time
        ttft_ms = (self._first_token_time - self._start) * 1000.0
        total_ms = (end - self._start) * 1000.0

        # TBT: differences between consecutive token timestamps
        tbts: list[float] = []
        for i in range(1, len(self._token_timestamps)):
            tbts.append((self._token_timestamps[i] - self._token_timestamps[i - 1]) * 1000.0)

        tbt_p50 = statistics.median(tbts) if tbts else 0.0
        tbt_p95 = _percentile(tbts, 0.95) if tbts else 0.0
        tps = generated_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

        return LatencyBenchmark(
            request_id=self.request_id,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            tbt_p50_ms=tbt_p50,
            tbt_p95_ms=tbt_p95,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
        )


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0.0–1.0) of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = p * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


# ---------------------------------------------------------------------------
# Demo: compare short vs long generation
# ---------------------------------------------------------------------------

def run_benchmark(
    label: str,
    prefill_ms: float,
    tbt_ms: float,
    num_tokens: int,
    prompt_tokens: int,
) -> LatencyBenchmark:
    simulator = StreamSimulator(prefill_ms=prefill_ms, tbt_ms=tbt_ms, tbt_jitter_ms=4.0)
    fake_tokens = [f"tok{i}" for i in range(num_tokens)]
    metrics = InferenceMetrics(request_id=label, prompt_tokens=prompt_tokens)

    # Consume the stream (in production this would stream to the client)
    for _ in metrics.observe(simulator.stream(fake_tokens)):
        pass

    return metrics.finalize(generated_tokens=num_tokens)


def main() -> None:
    logger.info("=== Inference Latency Benchmark ===")

    # Scenario A: short response (e.g. classification, short answer)
    bench_a = run_benchmark(
        label="short-response",
        prefill_ms=110.0,
        tbt_ms=28.0,
        num_tokens=20,
        prompt_tokens=64,
    )
    logger.info(bench_a.summary())

    # Scenario B: long response (e.g. document summary, code generation)
    bench_b = run_benchmark(
        label="long-response",
        prefill_ms=110.0,
        tbt_ms=28.0,
        num_tokens=200,
        prompt_tokens=64,
    )
    logger.info(bench_b.summary())

    # Observation: TTFT is identical (same prefill), but total latency and
    # throughput differ sharply — TBT dominates long generations.
    logger.info(
        "TTFT ratio long/short: %.2f  (expect ~1.0 — prefill is independent of output length)",
        bench_b.ttft_ms / bench_a.ttft_ms,
    )
    logger.info(
        "Total latency ratio long/short: %.1fx",
        bench_b.total_latency_ms / bench_a.total_latency_ms,
    )


if __name__ == "__main__":
    main()
```

**Expected output (approximate — values vary with system load):**

```
INFO === Inference Latency Benchmark ===
INFO [short-response] TTFT=110.3ms  TBT p50=27.8ms p95=32.1ms  total=666ms  throughput=30.0 tok/s
INFO [long-response]  TTFT=111.0ms  TBT p50=28.0ms p95=32.6ms  total=5721ms throughput=35.0 tok/s
INFO TTFT ratio long/short: 1.01  (expect ~1.0 — prefill is independent of output length)
INFO Total latency ratio long/short: 8.6x
```

Key observation: TTFT is nearly identical across both scenarios because it depends on prompt length (the input), not output length. Total latency scales with output tokens because each decode step adds one TBT unit.

---

## Exercises

1. **Measure TTFT under load.** Modify `StreamSimulator` to add a `queue_wait_ms` parameter representing time the request waits in queue before prefill begins. Run 10 simulated requests with queue waits drawn from a uniform distribution `[0, 200]` ms. Plot or print the TTFT distribution. What happens to p95 TTFT vs median TTFT?

2. **Compare latency profiles for two prompt lengths.** Prefill time scales with prompt token count. Add a `prefill_ms_per_token` parameter to `StreamSimulator` (e.g., 0.5 ms/token) and compute prefill time as `base_prefill_ms + prompt_tokens * prefill_ms_per_token`. Run benchmarks for prompts of 128, 512, and 2048 tokens with 100 output tokens each. How does TTFT scale vs total latency?

3. **Compute throughput from a batch.** Write a function `batch_throughput(benchmarks: list[LatencyBenchmark]) -> float` that computes system-level throughput: total tokens generated across all requests divided by the elapsed wall-clock time from first request start to last request end. Simulate 5 requests running concurrently (overlapping start times) and report the aggregate tokens/second. Compare it to the per-request throughput values.

(Solutions: ../solutions/week-10/day-1.md)

---

## Self-check

- [ ] I can explain why TTFT and TBT are measured separately and what each indicates about system behavior.
- [ ] I can describe the prefill and decode phases of autoregressive decoding and explain which hardware resource each is bound by.
- [ ] I can implement a token stream observer that records timestamps and computes TTFT and TBT statistics.
- [ ] I can explain why batching improves throughput but increases TTFT.
- [ ] I can identify which latency metric matters most for a given use case (interactive chat vs batch pipeline).

---

## Further reading

- [Continuous Batching: The Secret to 23x LLM Throughput Gains — Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)
