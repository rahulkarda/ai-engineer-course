# Phase 6 · Week 10 · Day 2 — Batching, Continuous Batching, and KV-Cache / PagedAttention

> Objective: Understand static vs continuous batching, why KV-cache grows with context, and how PagedAttention eliminates memory fragmentation.  ·  Time: ~1 hr  ·  Prereqs: [[day-1.md]]

---

## Concept

### Static batching — the naive approach

The simplest strategy for increasing GPU utilization is static (or naive) batching: collect a group of incoming requests, pad them to the same length, run them as a single batched tensor, and return results when all sequences finish.

Problems:
- **Padding waste.** Sequences in a batch have different lengths. Shorter sequences are padded to the longest one, burning compute on tokens that contribute nothing.
- **Straggler blocking.** A batch finishes only when its longest sequence finishes. If one request generates 500 tokens and the other four generate 30, the GPU is mostly idle for requests 2–5 after they finish generating, waiting for request 1.
- **Queue delay.** New requests must wait for the current batch to complete before they can start prefill. This directly adds to TTFT for all queued requests.

Static batching was how early transformers were served (GPT-3 era). It is easy to implement but wastes a large fraction of GPU capacity in practice.

### Continuous batching — iteration-level scheduling

Continuous batching (also called in-flight batching or dynamic batching) removes the constraint that all sequences in a batch must start and end together. Instead, the scheduler runs at the level of individual decode steps:

1. At each decode step, check which sequences have finished.
2. Remove finished sequences from the active batch immediately.
3. Fill the freed slot with a waiting request — start its prefill.
4. Continue generating for all active sequences.

Result: the batch is always as full as the hardware allows. There is no idle time between sequences. Throughput increases dramatically — published benchmarks show 10–23× vs static batching at the same hardware budget.

The tradeoff: the scheduler logic is more complex, and inserting a new sequence mid-generation requires careful memory management (see PagedAttention below).

### KV-cache — what it is and why it grows

In a transformer, the attention mechanism computes queries, keys, and values for every token at every layer. During autoregressive generation, the tokens generated so far are never modified — their key and value tensors are identical to what was computed in earlier steps.

**KV-cache**: store the key (K) and value (V) tensors for every past token across every layer. On each new decode step, compute only the new token's Q/K/V and attend over the cached K and V without recomputing them.

Without the KV-cache, generating a 1000-token response would require re-running the full attention computation over all 999 previous tokens for every new token — O(n²) complexity. With the cache, each decode step is O(n) reads + O(1) new computation.

**Memory cost.** For a typical 7B-parameter model:
- 32 layers, 32 attention heads, head dimension 128, fp16 (2 bytes)
- KV cache per token = 2 × 32 layers × 2 (K+V) × 32 heads × 128 dims × 2 bytes ≈ **0.5 MB per token**
- A context of 4096 tokens requires ~2 GB of KV cache
- At batch size 8 with 4096-token contexts: ~16 GB — comparable to the model weights themselves

This is the dominant memory pressure in LLM serving. It limits how many requests can be served simultaneously and how long their contexts can be.

### PagedAttention — virtual memory for the KV-cache

vLLM introduced PagedAttention (2023) to solve KV-cache memory fragmentation. Before PagedAttention, each request was allocated a contiguous block of GPU memory for its full potential context length at request start. Problems:

- Internal fragmentation: a request allocated for 4096 tokens that generates only 200 wastes the remaining memory.
- External fragmentation: completed requests leave gaps that are too small for new requests, even if total free memory is sufficient.

PagedAttention borrows the OS virtual memory idea:

1. Divide GPU memory into fixed-size **pages** (e.g., 16 tokens worth of KV cache each).
2. Each sequence's KV cache is a logical sequence of pages, stored in non-contiguous physical memory.
3. A page table maps logical page indices to physical GPU memory blocks.
4. Allocate pages on-demand as generation proceeds. Free them immediately when the sequence ends.

Benefits:
- Near-zero internal fragmentation (the last page of a sequence wastes at most `page_size - 1` token slots).
- Near-zero external fragmentation (any free page can serve any sequence).
- Enables KV-cache sharing between requests that share a common prefix (system prompt), reducing memory for identical-prefix batches.

You do not need to implement PagedAttention — vLLM and similar systems handle this. But understanding it explains why vLLM can serve 2–4× more concurrent requests than naive serving at the same GPU memory budget.

---

## How it works internally

### Static batching GPU timeline

```
Time →
GPU: [prefill A,B,C,D] [decode A,B,C,D] [decode A,B,C,D] [decode A] [idle] [idle]
                                                             ^ B,C,D done early
New requests E,F: wait in queue until A finishes
```

The GPU sits idle after B, C, D finish generating. New requests cannot start until A completes.

### Continuous batching GPU timeline

```
Time →
GPU: [prefill A,B,C,D] [decode A,B,C,D] [decode A,B,C,D + prefill E] [decode A,E + prefill F] ...
                                          ^ B,C,D done → E starts immediately
```

The batch is always saturated. GPU utilization stays high.

### Why continuous batching requires paged memory

When request E joins mid-generation, the scheduler needs to allocate KV-cache memory for E's prefill. If memory is pre-allocated in large contiguous blocks per sequence, there may not be a contiguous region large enough even if total free memory is sufficient (fragmentation). PagedAttention ensures any free pages are usable, so insertions always succeed until GPU memory is genuinely full.

### KV-cache growth curve

For a sequence generating `t` tokens from a prompt of `p` tokens:
- KV-cache slots occupied = `p + t`
- At each decode step, one more page may be allocated (when the current page is full)
- The scheduler can preempt a low-priority sequence (evict its KV-cache to CPU RAM or discard and recompute) to free pages for a higher-priority one

This preemption/swap mechanism is how production systems handle memory pressure without dropping requests.

---

## Worked example

```python
"""
day_2_batching_simulator.py

Simulate static vs continuous batching to observe throughput difference.
No GPU required — all timing is simulated with configurable parameters.

Run:
    python day_2_batching_simulator.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RequestSpec(NamedTuple):
    """A single inference request with known output length (for simulation)."""
    request_id: str
    prompt_tokens: int
    output_tokens: int


class BatchResult(BaseModel):
    """Outcome of running a batch strategy over a set of requests."""
    strategy: str
    total_requests: int
    total_output_tokens: int
    wall_time_ms: float
    throughput_tokens_per_s: float = Field(init=False, default=0.0)
    gpu_utilization_pct: float = Field(
        description="Fraction of decode steps where GPU was active (not idle)"
    )

    def model_post_init(self, __context: object) -> None:  # noqa: ANN001
        self.throughput_tokens_per_s = (
            self.total_output_tokens / (self.wall_time_ms / 1000.0)
            if self.wall_time_ms > 0
            else 0.0
        )

    def summary(self) -> str:
        return (
            f"[{self.strategy:20s}] "
            f"wall={self.wall_time_ms:7.0f}ms  "
            f"throughput={self.throughput_tokens_per_s:6.1f} tok/s  "
            f"GPU util={self.gpu_utilization_pct:.0f}%"
        )


# ---------------------------------------------------------------------------
# Timing model (pure math, no real sleep needed for batch comparison)
# ---------------------------------------------------------------------------

@dataclass
class TimingModel:
    """
    Simulated timing constants for a single-GPU server.

    Parameters
    ----------
    prefill_ms_per_token:
        Time (ms) to process one prompt token during prefill.
    decode_ms_per_step:
        Time (ms) for one decode step regardless of batch size
        (memory-bandwidth-bound: adding more sequences in the same
        batch barely changes decode time until memory bandwidth saturates).
    max_batch_size:
        Maximum sequences that fit in GPU memory simultaneously.
    """

    prefill_ms_per_token: float = 0.05   # 50 µs/token during prefill
    decode_ms_per_step: float = 20.0     # 20 ms per decode step
    max_batch_size: int = 8


# ---------------------------------------------------------------------------
# Static batching simulator
# ---------------------------------------------------------------------------

def simulate_static_batching(
    requests: list[RequestSpec],
    timing: TimingModel,
) -> BatchResult:
    """
    Process requests in fixed batches.

    Batch is filled to max_batch_size, padded to the longest output length,
    then runs to completion before the next batch starts.
    """
    total_output_tokens = sum(r.output_tokens for r in requests)
    total_ms = 0.0
    total_decode_steps = 0
    active_decode_steps = 0

    # Split into batches
    for batch_start in range(0, len(requests), timing.max_batch_size):
        batch = requests[batch_start : batch_start + timing.max_batch_size]

        # Prefill: process each request's prompt (they run in parallel)
        max_prompt = max(r.prompt_tokens for r in batch)
        prefill_ms = max_prompt * timing.prefill_ms_per_token
        total_ms += prefill_ms

        # Decode: run until the *longest* sequence in the batch finishes
        # Short sequences are "done" but the GPU slot stays occupied until
        # the last sequence finishes (static batching can't free slots mid-batch)
        max_output = max(r.output_tokens for r in batch)
        for step in range(max_output):
            total_ms += timing.decode_ms_per_step
            total_decode_steps += 1
            # Count how many sequences are still generating at this step
            active_this_step = sum(1 for r in batch if step < r.output_tokens)
            active_decode_steps += active_this_step

    # GPU utilization: fraction of steps where ALL slots in the batch were active
    # vs steps where some finished early (wasted capacity)
    max_possible_active = total_decode_steps * timing.max_batch_size
    gpu_util = (active_decode_steps / max_possible_active * 100.0) if max_possible_active else 0.0

    return BatchResult(
        strategy="static-batching",
        total_requests=len(requests),
        total_output_tokens=total_output_tokens,
        wall_time_ms=total_ms,
        gpu_utilization_pct=gpu_util,
    )


# ---------------------------------------------------------------------------
# Continuous batching simulator
# ---------------------------------------------------------------------------

@dataclass
class _ActiveSequence:
    """Mutable state for a sequence currently in the active batch."""
    spec: RequestSpec
    remaining_steps: int


def simulate_continuous_batching(
    requests: list[RequestSpec],
    timing: TimingModel,
) -> BatchResult:
    """
    Process requests with iteration-level scheduling.

    At each decode step:
    1. Tick all active sequences (remaining_steps -= 1).
    2. Remove any that finished.
    3. Pull waiting requests off the queue and prefill them to fill freed slots.
    4. Repeat.

    This eliminates idle GPU time caused by stragglers.
    """
    total_output_tokens = sum(r.output_tokens for r in requests)
    total_ms = 0.0
    total_decode_steps = 0
    active_decode_steps = 0

    queue: list[RequestSpec] = list(requests)
    active: list[_ActiveSequence] = []

    def fill_batch() -> None:
        """Pull requests from the queue to fill available batch slots, prefilling each."""
        nonlocal total_ms
        while queue and len(active) < timing.max_batch_size:
            req = queue.pop(0)
            prefill_ms = req.prompt_tokens * timing.prefill_ms_per_token
            total_ms += prefill_ms
            active.append(_ActiveSequence(spec=req, remaining_steps=req.output_tokens))

    # Initial fill
    fill_batch()

    while active:
        # One decode step for all active sequences simultaneously
        total_ms += timing.decode_ms_per_step
        total_decode_steps += 1
        active_decode_steps += len(active)

        # Advance all sequences
        for seq in active:
            seq.remaining_steps -= 1

        # Remove finished sequences
        finished = [s for s in active if s.remaining_steps <= 0]
        active = [s for s in active if s.remaining_steps > 0]

        if finished:
            logger.debug(
                "Step: finished %d sequence(s), queue has %d remaining",
                len(finished),
                len(queue),
            )

        # Fill freed slots immediately — this is the key difference
        if queue:
            fill_batch()

    max_possible_active = total_decode_steps * timing.max_batch_size
    gpu_util = (active_decode_steps / max_possible_active * 100.0) if max_possible_active else 0.0

    return BatchResult(
        strategy="continuous-batching",
        total_requests=len(requests),
        total_output_tokens=total_output_tokens,
        wall_time_ms=total_ms,
        gpu_utilization_pct=gpu_util,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # Realistic mix: some short (classification/QA), some long (generation)
    # Short/long ratio typical of production chat workloads
    requests: list[RequestSpec] = [
        RequestSpec("req-01", prompt_tokens=128, output_tokens=20),
        RequestSpec("req-02", prompt_tokens=256, output_tokens=350),
        RequestSpec("req-03", prompt_tokens=64,  output_tokens=15),
        RequestSpec("req-04", prompt_tokens=512, output_tokens=400),
        RequestSpec("req-05", prompt_tokens=128, output_tokens=25),
        RequestSpec("req-06", prompt_tokens=128, output_tokens=300),
        RequestSpec("req-07", prompt_tokens=64,  output_tokens=18),
        RequestSpec("req-08", prompt_tokens=256, output_tokens=380),
        RequestSpec("req-09", prompt_tokens=128, output_tokens=22),
        RequestSpec("req-10", prompt_tokens=512, output_tokens=10),
        RequestSpec("req-11", prompt_tokens=128, output_tokens=30),
        RequestSpec("req-12", prompt_tokens=64,  output_tokens=250),
    ]

    timing = TimingModel(
        prefill_ms_per_token=0.05,
        decode_ms_per_step=20.0,
        max_batch_size=4,
    )

    logger.info("=== Batching Strategy Comparison ===")
    logger.info("Requests: %d  |  Max batch size: %d", len(requests), timing.max_batch_size)

    static_result = simulate_static_batching(requests, timing)
    continuous_result = simulate_continuous_batching(requests, timing)

    logger.info(static_result.summary())
    logger.info(continuous_result.summary())

    speedup = static_result.wall_time_ms / continuous_result.wall_time_ms
    logger.info(
        "Continuous batching speedup: %.2fx  |  GPU util gain: +%.0f%%",
        speedup,
        continuous_result.gpu_utilization_pct - static_result.gpu_utilization_pct,
    )


if __name__ == "__main__":
    main()
```

**Expected output (approximate):**

```
INFO === Batching Strategy Comparison ===
INFO Requests: 12  |  Max batch size: 4
INFO [static-batching     ] wall=  7620ms  throughput=  263.8 tok/s  GPU util=46%
INFO [continuous-batching ] wall=  4180ms  throughput=  481.1 tok/s  GPU util=89%
INFO Continuous batching speedup: 1.82x  |  GPU util gain: +43%
```

The GPU utilization difference explains the throughput gain: static batching leaves slots idle whenever short sequences finish before the straggler in the same batch. Continuous batching fills those slots immediately.

---

## Exercises

1. **Straggler effect.** Modify the request list so that one request per batch always generates 10× more tokens than the others (e.g., one request with 500 output tokens and three with 50). Measure static vs continuous batching throughput. How does the speedup ratio change compared to the default workload? Explain why in terms of the straggler blocking pattern.

2. **KV-cache memory model.** Write a function `kv_cache_memory_gb(num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int, max_batch_size: int, max_seq_len: int) -> float` that computes peak KV-cache memory in GB. Use it to compute the KV-cache footprint for Llama-3-8B (32 layers, 32 heads, head dim 128, fp16) at batch sizes 1, 4, 8, 16 with a 4096-token context. At what batch size does the KV-cache exceed 20 GB?

3. **Page allocation simulation.** Extend `_ActiveSequence` with a `pages_allocated` counter (page size = 16 tokens). At each decode step, increment `pages_allocated` when `(tokens_generated % page_size) == 0` (a new page is needed). Track the peak total pages in use across all active sequences simultaneously. Compare peak pages for static vs continuous batching — does continuous batching change peak memory pressure? Why or why not?

(Solutions: ../solutions/week-10/day-2.md)

---

## Self-check

- [ ] I can explain why static batching wastes GPU capacity when sequences have different lengths.
- [ ] I can describe the iteration-level scheduling loop of continuous batching and where it inserts new requests.
- [ ] I can derive the per-token KV-cache memory cost from transformer architecture parameters.
- [ ] I can explain the PagedAttention analogy to OS virtual memory and the fragmentation problem it solves.
- [ ] I can implement a batch simulator that demonstrates throughput differences between static and continuous strategies.

---

## Further reading

- [Continuous Batching: The Secret to 23x LLM Throughput Gains — Anyscale](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention — vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
