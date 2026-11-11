# Week 10 · Day 2 Solutions — Batching & KV-Cache

---

## Exercise 1 — Straggler effect on static batching

### Approach

Static batching forces every request in a batch to wait for the slowest (straggler) request. A batch of [50, 50, 50, 800] tokens runs for 800 decode steps — the 50-token requests are idle for 750 steps. The GPU time "wasted" on idle slots is the key metric.

Continuous batching eliminates this: once a 50-token request finishes, its slot is immediately reused for a new request.

### Solution

```python
"""
solutions/week_10/ex1_straggler_effect.py

Simulate static vs continuous batching for a batch with one straggler.
Demonstrates GPU time wasted due to the straggler effect.

Run:
    python ex1_straggler_effect.py
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
# Data types
# ---------------------------------------------------------------------------

class RequestSpec(NamedTuple):
    """Inference request with known output length (for simulation)."""
    request_id: str
    prompt_tokens: int
    output_tokens: int


class BatchingResult(BaseModel):
    """Outcome of running a batching strategy over a workload."""

    strategy: str
    total_requests: int
    total_output_tokens: int
    total_decode_steps: int = Field(description="Total decode iterations executed")
    active_slot_steps: int = Field(description="Sum of active slots across all decode steps")
    max_batch_size: int
    wall_time_ms: float

    @property
    def wasted_slot_steps(self) -> int:
        """Decode steps × slots where a slot was idle (could have served a request but didn't)."""
        return self.total_decode_steps * self.max_batch_size - self.active_slot_steps

    @property
    def gpu_utilization_pct(self) -> float:
        max_possible = self.total_decode_steps * self.max_batch_size
        return (self.active_slot_steps / max_possible * 100.0) if max_possible else 0.0

    @property
    def wasted_pct(self) -> float:
        return 100.0 - self.gpu_utilization_pct

    @property
    def throughput_tokens_per_s(self) -> float:
        return (self.total_output_tokens / self.wall_time_ms * 1000.0) if self.wall_time_ms > 0 else 0.0

    def summary(self) -> str:
        return (
            f"[{self.strategy:20s}]  "
            f"decode_steps={self.total_decode_steps:4d}  "
            f"GPU_util={self.gpu_utilization_pct:5.1f}%  "
            f"wasted={self.wasted_pct:5.1f}%  "
            f"wall={self.wall_time_ms:.0f}ms  "
            f"tput={self.throughput_tokens_per_s:.1f} tok/s"
        )


# ---------------------------------------------------------------------------
# Timing model
# ---------------------------------------------------------------------------

@dataclass
class TimingModel:
    prefill_ms_per_token: float = 0.05  # 50 µs/token
    decode_ms_per_step: float = 20.0    # 20 ms per decode iteration
    max_batch_size: int = 4


# ---------------------------------------------------------------------------
# Static batching simulation
# ---------------------------------------------------------------------------

def simulate_static_batching(
    requests: list[RequestSpec],
    timing: TimingModel,
) -> BatchingResult:
    """
    All requests in a batch wait for the longest one to finish.
    Idle slots accumulate after short requests complete.
    """
    total_ms = 0.0
    total_steps = 0
    active_slot_steps = 0
    total_out = sum(r.output_tokens for r in requests)

    for batch_start in range(0, len(requests), timing.max_batch_size):
        batch = requests[batch_start : batch_start + timing.max_batch_size]
        max_out = max(r.output_tokens for r in batch)

        # Prefill all requests in batch (run concurrently, cost = max prompt length)
        max_prompt = max(r.prompt_tokens for r in batch)
        total_ms += max_prompt * timing.prefill_ms_per_token

        # Decode: run max_out steps; short requests become idle after their step count
        for step in range(max_out):
            total_ms += timing.decode_ms_per_step
            total_steps += 1
            still_active = sum(1 for r in batch if step < r.output_tokens)
            active_slot_steps += still_active

    return BatchingResult(
        strategy="static",
        total_requests=len(requests),
        total_output_tokens=total_out,
        total_decode_steps=total_steps,
        active_slot_steps=active_slot_steps,
        max_batch_size=timing.max_batch_size,
        wall_time_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# Continuous batching simulation
# ---------------------------------------------------------------------------

@dataclass
class _ActiveSeq:
    spec: RequestSpec
    remaining: int


def simulate_continuous_batching(
    requests: list[RequestSpec],
    timing: TimingModel,
) -> BatchingResult:
    """
    At each decode step, finished requests are removed and new ones start immediately.
    Slots are never idle as long as the queue is non-empty.
    """
    total_ms = 0.0
    total_steps = 0
    active_slot_steps = 0
    total_out = sum(r.output_tokens for r in requests)

    queue: list[RequestSpec] = list(requests)
    active: list[_ActiveSeq] = []

    def fill_slots() -> None:
        nonlocal total_ms
        while queue and len(active) < timing.max_batch_size:
            req = queue.pop(0)
            total_ms += req.prompt_tokens * timing.prefill_ms_per_token
            active.append(_ActiveSeq(spec=req, remaining=req.output_tokens))

    fill_slots()

    while active:
        total_ms += timing.decode_ms_per_step
        total_steps += 1
        active_slot_steps += len(active)

        for seq in active:
            seq.remaining -= 1

        active = [s for s in active if s.remaining > 0]

        if queue:
            fill_slots()

    return BatchingResult(
        strategy="continuous",
        total_requests=len(requests),
        total_output_tokens=total_out,
        total_decode_steps=total_steps,
        active_slot_steps=active_slot_steps,
        max_batch_size=timing.max_batch_size,
        wall_time_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# Demo with straggler pattern
# ---------------------------------------------------------------------------

def main() -> None:
    timing = TimingModel(max_batch_size=4, decode_ms_per_step=20.0)

    # Straggler pattern: one request generates 16x more tokens than the rest
    requests_straggler: list[RequestSpec] = [
        RequestSpec("r1", prompt_tokens=128, output_tokens=50),
        RequestSpec("r2", prompt_tokens=128, output_tokens=50),
        RequestSpec("r3", prompt_tokens=128, output_tokens=50),
        RequestSpec("r4", prompt_tokens=128, output_tokens=800),  # straggler
        RequestSpec("r5", prompt_tokens=128, output_tokens=50),
        RequestSpec("r6", prompt_tokens=128, output_tokens=50),
        RequestSpec("r7", prompt_tokens=128, output_tokens=50),
        RequestSpec("r8", prompt_tokens=128, output_tokens=800),  # straggler
    ]

    # Uniform baseline for comparison
    requests_uniform: list[RequestSpec] = [
        RequestSpec(f"u{i}", prompt_tokens=128, output_tokens=200)
        for i in range(8)
    ]

    logger.info("=== Straggler Effect: Static vs Continuous Batching ===")
    logger.info("")

    for label, reqs in [("Straggler workload", requests_straggler), ("Uniform workload", requests_uniform)]:
        static = simulate_static_batching(reqs, timing)
        cont = simulate_continuous_batching(reqs, timing)
        speedup = static.wall_time_ms / cont.wall_time_ms

        logger.info("--- %s ---", label)
        logger.info(static.summary())
        logger.info(cont.summary())
        logger.info(
            "Wasted GPU slot-steps (static): %d / %d = %.1f%%",
            static.wasted_slot_steps,
            static.total_decode_steps * static.max_batch_size,
            static.wasted_pct,
        )
        logger.info("Continuous batching speedup: %.2fx", speedup)
        logger.info("")

    logger.info("Key insight: with stragglers, static batching wastes ~75%% of GPU slots")
    logger.info("in the batch where 3 requests finish at step 50 but 1 runs to step 800.")
    logger.info("Continuous batching fills those slots immediately with new requests.")


if __name__ == "__main__":
    main()
```

### Percentage GPU time wasted — derivation

For batch [50, 50, 50, 800]:
- Total decode steps = 800 (determined by straggler)
- Active slot-steps = 50+50+50 + 800 = 950 (three requests contribute only 50 steps each)
- Maximum possible slot-steps = 800 × 4 = 3200
- Wasted fraction = (3200 - 950) / 3200 = 70.3%

Continuous batching refills the 3 finished slots at step 51 with new requests, so utilization stays near 100% as long as the queue is non-empty.

### Pytest tests

```python
"""tests/test_ex1_straggler_effect.py"""

from __future__ import annotations

import pytest

from solutions.week_10.ex1_straggler_effect import (
    RequestSpec,
    TimingModel,
    simulate_continuous_batching,
    simulate_static_batching,
)


def make_timing(max_batch_size: int = 4) -> TimingModel:
    return TimingModel(prefill_ms_per_token=0.0, decode_ms_per_step=1.0, max_batch_size=max_batch_size)


class TestStaticBatching:
    def test_uniform_batch_full_utilization(self) -> None:
        """All requests same length: no idle slots."""
        reqs = [RequestSpec(f"r{i}", 0, 100) for i in range(4)]
        result = simulate_static_batching(reqs, make_timing(4))
        assert result.gpu_utilization_pct == pytest.approx(100.0)

    def test_straggler_reduces_utilization(self) -> None:
        """One 10x straggler should drop utilization well below 100%."""
        reqs = [
            RequestSpec("r1", 0, 50),
            RequestSpec("r2", 0, 50),
            RequestSpec("r3", 0, 50),
            RequestSpec("r4", 0, 500),  # straggler
        ]
        result = simulate_static_batching(reqs, make_timing(4))
        assert result.gpu_utilization_pct < 70.0

    def test_straggler_wasted_pct_formula(self) -> None:
        """Manually verify wasted calculation for [50, 50, 50, 800]."""
        reqs = [
            RequestSpec("r1", 0, 50),
            RequestSpec("r2", 0, 50),
            RequestSpec("r3", 0, 50),
            RequestSpec("r4", 0, 800),
        ]
        result = simulate_static_batching(reqs, make_timing(4))
        expected_active = 50 * 3 + 800  # 3 short + 1 full
        expected_total = 800 * 4
        expected_util = expected_active / expected_total * 100.0
        assert result.gpu_utilization_pct == pytest.approx(expected_util, abs=0.1)

    def test_wall_time_single_request(self) -> None:
        reqs = [RequestSpec("r1", 0, 10)]
        result = simulate_static_batching(reqs, make_timing(4))
        assert result.wall_time_ms == pytest.approx(10.0)


class TestContinuousBatching:
    def test_continuous_higher_util_than_static_with_straggler(self) -> None:
        reqs = [
            RequestSpec("r1", 0, 50),
            RequestSpec("r2", 0, 50),
            RequestSpec("r3", 0, 50),
            RequestSpec("r4", 0, 800),
            RequestSpec("r5", 0, 50),
            RequestSpec("r6", 0, 50),
        ]
        timing = make_timing(4)
        static = simulate_static_batching(reqs, timing)
        cont = simulate_continuous_batching(reqs, timing)
        assert cont.gpu_utilization_pct > static.gpu_utilization_pct

    def test_continuous_faster_wall_time(self) -> None:
        reqs = [RequestSpec(f"r{i}", 0, 50 if i % 4 != 3 else 800) for i in range(8)]
        timing = make_timing(4)
        static = simulate_static_batching(reqs, timing)
        cont = simulate_continuous_batching(reqs, timing)
        assert cont.wall_time_ms < static.wall_time_ms

    def test_both_process_same_total_tokens(self) -> None:
        reqs = [RequestSpec(f"r{i}", 0, 100) for i in range(8)]
        timing = make_timing(4)
        static = simulate_static_batching(reqs, timing)
        cont = simulate_continuous_batching(reqs, timing)
        assert static.total_output_tokens == cont.total_output_tokens
```

---

## Exercise 2 — KV-cache memory model

### Approach

Each transformer layer stores K and V tensors for every cached token. The total memory is:

```
KV_bytes = 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes
```

The factor of 2 accounts for both K and V. For Llama-3-8B (32 layers, 32 heads, 128 head_dim) at 4096 context and batch 8 in FP16 (2 bytes), this is approximately 16 GB — comparable to the model weights themselves.

### Solution

```python
"""
solutions/week_10/ex2_kv_cache_memory.py

KV-cache memory model function and Llama-3-8B analysis.

Run:
    python ex2_kv_cache_memory.py
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def compute_kv_memory(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> int:
    """
    Compute KV-cache memory in bytes.

    Formula:
        2 (K + V) * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes

    Parameters
    ----------
    n_layers:
        Number of transformer layers (each has its own K and V cache).
    n_heads:
        Number of attention heads per layer.
    head_dim:
        Dimension of each attention head (hidden_dim // n_heads typically).
    seq_len:
        Number of tokens in the context (prompt + generated so far).
    batch_size:
        Number of concurrent sequences.
    dtype_bytes:
        Bytes per value. 2 = FP16/BF16, 4 = FP32, 1 = INT8.

    Returns
    -------
    int:
        Total KV-cache memory in bytes.

    Notes
    -----
    This is the theoretical minimum for a standard multi-head attention model.
    Grouped-query attention (GQA) models like Llama-3 reduce n_heads for K/V
    (n_kv_heads < n_heads), reducing cache proportionally. This function
    assumes full MHA (n_kv_heads == n_heads).
    """
    return 2 * n_layers * n_heads * head_dim * seq_len * batch_size * dtype_bytes


def compute_kv_memory_gb(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,
) -> float:
    """Convenience wrapper returning gigabytes (1 GB = 1024^3 bytes)."""
    return compute_kv_memory(n_layers, n_heads, head_dim, seq_len, batch_size, dtype_bytes) / (1024 ** 3)


# ---------------------------------------------------------------------------
# Analysis model
# ---------------------------------------------------------------------------

class KVCacheAnalysis(BaseModel):
    """Memory breakdown for a specific model/config combination."""

    model_name: str
    n_layers: int
    n_heads: int
    head_dim: int
    seq_len: int
    batch_size: int
    dtype_bytes: int
    kv_cache_bytes: int = Field(description="KV cache in bytes")

    @property
    def kv_cache_gb(self) -> float:
        return self.kv_cache_bytes / (1024 ** 3)

    @property
    def dtype_label(self) -> str:
        return {1: "INT8", 2: "FP16", 4: "FP32"}.get(self.dtype_bytes, f"{self.dtype_bytes*8}bit")

    def summary(self) -> str:
        return (
            f"{self.model_name:15s}  "
            f"batch={self.batch_size:3d}  "
            f"ctx={self.seq_len:5d}  "
            f"dtype={self.dtype_label:5s}  "
            f"KV={self.kv_cache_gb:6.2f} GB"
        )


# ---------------------------------------------------------------------------
# Demo: Llama-3-8B
# ---------------------------------------------------------------------------

# Llama-3-8B architecture
# Note: Llama-3-8B uses GQA with 8 KV heads, but this exercise uses full MHA
# (n_kv_heads=32) to match the standard formula taught in the lesson.
LLAMA_3_8B = dict(n_layers=32, n_heads=32, head_dim=128, dtype_bytes=2)


def main() -> None:
    logger.info("=== KV-Cache Memory: Llama-3-8B (MHA, FP16) ===")
    logger.info("Architecture: 32 layers, 32 heads, head_dim=128, context=4096")
    logger.info("")
    logger.info("%-15s  %-5s  %-6s  %-6s  %-9s  Exceeds 20 GB?", "model", "batch", "ctx", "dtype", "KV (GB)")

    for batch_size in [1, 2, 4, 8, 16, 32]:
        kv_gb = compute_kv_memory_gb(seq_len=4096, batch_size=batch_size, **LLAMA_3_8B)
        exceeds = "YES" if kv_gb > 20.0 else "no"
        logger.info(
            "Llama-3-8B      batch=%3d  ctx=4096  FP16   KV=%6.2f GB  %s",
            batch_size,
            kv_gb,
            exceeds,
        )

    # Find crossover
    logger.info("")
    logger.info("=== Crossover: KV cache > 20 GB ===")
    for batch_size in range(1, 33):
        kv_gb = compute_kv_memory_gb(seq_len=4096, batch_size=batch_size, **LLAMA_3_8B)
        if kv_gb > 20.0:
            logger.info("KV cache exceeds 20 GB at batch_size=%d (%.2f GB)", batch_size, kv_gb)
            break

    # Per-token cost
    per_token_bytes = compute_kv_memory(seq_len=1, batch_size=1, **LLAMA_3_8B)
    logger.info("")
    logger.info("Per-token KV cost: %d bytes = %.3f KB", per_token_bytes, per_token_bytes / 1024)
    logger.info(
        "This means: 4096-token context = %.1f MB per request in the batch",
        compute_kv_memory(seq_len=4096, batch_size=1, **LLAMA_3_8B) / (1024 ** 2),
    )


if __name__ == "__main__":
    main()
```

### Expected output

```
=== KV-Cache Memory: Llama-3-8B (MHA, FP16) ===
Architecture: 32 layers, 32 heads, head_dim=128, context=4096

Llama-3-8B      batch=  1  ctx=4096  FP16   KV=  2.00 GB  no
Llama-3-8B      batch=  2  ctx=4096  FP16   KV=  4.00 GB  no
Llama-3-8B      batch=  4  ctx=4096  FP16   KV=  8.00 GB  no
Llama-3-8B      batch=  8  ctx=4096  FP16   KV= 16.00 GB  no
Llama-3-8B      batch= 16  ctx=4096  FP16   KV= 32.00 GB  YES
Llama-3-8B      batch= 32  ctx=4096  FP16   KV= 64.00 GB  YES

KV cache exceeds 20 GB at batch_size=11 (22.00 GB)
Per-token KV cost: 524288 bytes = 512.000 KB
This means: 4096-token context = 2048.0 MB per request
```

Manual verification: 2 × 32 × 32 × 128 × 4096 × 1 × 2 = 2,147,483,648 bytes = 2.0 GB per request. At batch 8: 16 GB.

### Pytest tests

```python
"""tests/test_ex2_kv_cache_memory.py"""

from __future__ import annotations

import pytest

from solutions.week_10.ex2_kv_cache_memory import compute_kv_memory, compute_kv_memory_gb


class TestComputeKVMemory:
    def test_known_value_single_layer_single_head(self) -> None:
        """1 layer, 1 head, head_dim=4, seq_len=1, batch=1, FP16 = 2*1*1*4*1*1*2 = 16 bytes."""
        result = compute_kv_memory(
            n_layers=1, n_heads=1, head_dim=4, seq_len=1, batch_size=1, dtype_bytes=2
        )
        assert result == 16

    def test_scales_linearly_with_batch_size(self) -> None:
        base = compute_kv_memory(n_layers=4, n_heads=4, head_dim=64, seq_len=512, batch_size=1)
        doubled = compute_kv_memory(n_layers=4, n_heads=4, head_dim=64, seq_len=512, batch_size=2)
        assert doubled == 2 * base

    def test_scales_linearly_with_seq_len(self) -> None:
        base = compute_kv_memory(n_layers=4, n_heads=4, head_dim=64, seq_len=512, batch_size=1)
        doubled = compute_kv_memory(n_layers=4, n_heads=4, head_dim=64, seq_len=1024, batch_size=1)
        assert doubled == 2 * base

    def test_scales_linearly_with_layers(self) -> None:
        base = compute_kv_memory(n_layers=16, n_heads=16, head_dim=64, seq_len=1024, batch_size=1)
        doubled = compute_kv_memory(n_layers=32, n_heads=16, head_dim=64, seq_len=1024, batch_size=1)
        assert doubled == 2 * base

    def test_fp32_doubles_fp16(self) -> None:
        fp16 = compute_kv_memory(n_layers=8, n_heads=8, head_dim=64, seq_len=2048, batch_size=4, dtype_bytes=2)
        fp32 = compute_kv_memory(n_layers=8, n_heads=8, head_dim=64, seq_len=2048, batch_size=4, dtype_bytes=4)
        assert fp32 == 2 * fp16

    def test_llama_3_8b_at_batch1(self) -> None:
        """Llama-3-8B, 4096 context, batch 1, FP16 should be 2.0 GB."""
        kv_gb = compute_kv_memory_gb(
            n_layers=32, n_heads=32, head_dim=128, seq_len=4096, batch_size=1, dtype_bytes=2
        )
        assert kv_gb == pytest.approx(2.0, abs=0.01)

    def test_llama_3_8b_at_batch8(self) -> None:
        """Batch 8 should be 8× batch 1."""
        kv_b1 = compute_kv_memory_gb(n_layers=32, n_heads=32, head_dim=128, seq_len=4096, batch_size=1)
        kv_b8 = compute_kv_memory_gb(n_layers=32, n_heads=32, head_dim=128, seq_len=4096, batch_size=8)
        assert kv_b8 == pytest.approx(8 * kv_b1)

    def test_returns_integer(self) -> None:
        result = compute_kv_memory(n_layers=4, n_heads=4, head_dim=64, seq_len=128, batch_size=2)
        assert isinstance(result, int)
```

---

## Exercise 3 — Page allocation simulation (PagedKVCache)

### Approach

PagedAttention divides GPU memory into fixed-size pages (e.g., 16 token slots each). Sequences do not own contiguous memory — they hold a logical list of physical page IDs. When a page is freed, it goes back to a free pool and can immediately serve any new sequence, eliminating fragmentation.

This contrasts with contiguous allocation where a sequence reserves its full potential context length upfront. If the sequence ends early, the tail is wasted (internal fragmentation); if sequences complete at different times, they leave gaps that may be too small for new requests (external fragmentation).

### Solution

```python
"""
solutions/week_10/ex3_paged_kv_cache.py

PagedKVCache: virtual memory for KV cache with page-level allocation.
Demonstrates page allocate/free/reuse mechanics.

Run:
    python ex3_paged_kv_cache.py
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class AllocationResult(BaseModel):
    """Result of a page allocation request."""

    seq_id: str
    pages_allocated: list[int] = Field(description="Physical page IDs allocated to this request")
    tokens_covered: int = Field(description="Total token slots covered by allocated pages")

    def summary(self) -> str:
        return (
            f"seq={self.seq_id}  pages={self.pages_allocated}  "
            f"token_slots={self.tokens_covered}"
        )


class PoolStats(BaseModel):
    """Snapshot of page pool state."""

    total_pages: int
    free_pages: int
    allocated_pages: int
    sequences_active: int
    fragmentation_pct: float = Field(
        description="Internal fragmentation: unused slots in last page of each sequence"
    )

    def summary(self) -> str:
        return (
            f"total={self.total_pages}  free={self.free_pages}  "
            f"allocated={self.allocated_pages}  seqs={self.sequences_active}  "
            f"internal_frag={self.fragmentation_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# PagedKVCache
# ---------------------------------------------------------------------------

class PagedKVCache:
    """
    Simulates PagedAttention-style KV cache memory management.

    The GPU's KV cache memory is divided into `total_pages` fixed-size pages.
    Each page holds `page_size` token slots. Sequences hold a logical list of
    physical page IDs — pages are non-contiguous in physical memory.

    Parameters
    ----------
    total_pages:
        Total number of physical pages in the pool.
    page_size:
        Number of token slots per page.
    """

    def __init__(self, total_pages: int, page_size: int = 16) -> None:
        if total_pages <= 0:
            raise ValueError("total_pages must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        self.total_pages = total_pages
        self.page_size = page_size

        # Free pool: all pages start unallocated
        # Use a deque for O(1) pop from front (FIFO allocation order for predictability in tests)
        self._free_pool: deque[int] = deque(range(total_pages))

        # Mapping from seq_id to list of physical page IDs
        self._seq_pages: dict[str, list[int]] = {}

        # Track tokens_used in the last page of each sequence (for fragmentation)
        self._seq_last_page_tokens: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, seq_id: str, tokens_needed: int) -> AllocationResult:
        """
        Allocate pages for `seq_id` to cover `tokens_needed` token slots.

        Pages are allocated on-demand: only enough pages are allocated to cover
        the tokens requested. If the sequence already has pages, new pages are
        appended as needed (simulating incremental generation).

        Parameters
        ----------
        seq_id:
            Unique identifier for the sequence.
        tokens_needed:
            Total token slots required (prompt + generated tokens so far).

        Returns
        -------
        AllocationResult:
            The page IDs allocated in this call.

        Raises
        ------
        RuntimeError:
            If there are not enough free pages to satisfy the request.
        """
        existing_pages = self._seq_pages.get(seq_id, [])
        already_covered = len(existing_pages) * self.page_size

        new_pages_needed = max(0, -(-tokens_needed // self.page_size) - len(existing_pages))
        # -(-a // b) is ceiling division

        if new_pages_needed > len(self._free_pool):
            raise RuntimeError(
                f"Out of pages: need {new_pages_needed} free, have {len(self._free_pool)}"
            )

        newly_allocated: list[int] = []
        for _ in range(new_pages_needed):
            page_id = self._free_pool.popleft()
            newly_allocated.append(page_id)

        self._seq_pages.setdefault(seq_id, []).extend(newly_allocated)
        tokens_covered = len(self._seq_pages[seq_id]) * self.page_size

        # Track how many tokens of the last page are actually used
        last_page_used = tokens_needed % self.page_size or self.page_size
        self._seq_last_page_tokens[seq_id] = last_page_used

        logger.debug(
            "allocate seq=%s tokens=%d new_pages=%s total_pages=%s",
            seq_id,
            tokens_needed,
            newly_allocated,
            self._seq_pages[seq_id],
        )

        return AllocationResult(
            seq_id=seq_id,
            pages_allocated=newly_allocated,
            tokens_covered=tokens_covered,
        )

    def free(self, seq_id: str) -> list[int]:
        """
        Free all pages held by `seq_id`.

        Freed pages return to the pool immediately and are available for reuse
        by any subsequent allocation — regardless of which sequence originally
        used them. This is the key property that eliminates external fragmentation.

        Parameters
        ----------
        seq_id:
            Sequence to free.

        Returns
        -------
        list[int]:
            The page IDs that were freed.

        Raises
        ------
        KeyError:
            If seq_id is not currently allocated.
        """
        if seq_id not in self._seq_pages:
            raise KeyError(f"seq_id '{seq_id}' not found in allocated sequences")

        freed_pages = self._seq_pages.pop(seq_id)
        self._seq_last_page_tokens.pop(seq_id, None)

        # Return pages to free pool — LIFO order mirrors typical allocator behavior
        self._free_pool.extendleft(freed_pages)

        logger.debug("free seq=%s released pages=%s", seq_id, freed_pages)
        return freed_pages

    def stats(self) -> PoolStats:
        """Return a snapshot of current pool utilization."""
        allocated_count = sum(len(pages) for pages in self._seq_pages.values())
        free_count = len(self._free_pool)

        # Internal fragmentation: slots in the last page of each seq that are unused
        total_last_page_waste = sum(
            self.page_size - used
            for used in self._seq_last_page_tokens.values()
        )
        total_allocated_slots = allocated_count * self.page_size
        frag_pct = (total_last_page_waste / total_allocated_slots * 100.0) if total_allocated_slots > 0 else 0.0

        return PoolStats(
            total_pages=self.total_pages,
            free_pages=free_count,
            allocated_pages=allocated_count,
            sequences_active=len(self._seq_pages),
            fragmentation_pct=frag_pct,
        )

    @property
    def free_page_count(self) -> int:
        return len(self._free_pool)

    @property
    def allocated_sequences(self) -> list[str]:
        return list(self._seq_pages.keys())

    def pages_for(self, seq_id: str) -> list[int]:
        return list(self._seq_pages.get(seq_id, []))


# ---------------------------------------------------------------------------
# Fragmentation comparison: paged vs contiguous
# ---------------------------------------------------------------------------

def compare_fragmentation(total_memory_slots: int, page_size: int = 16) -> None:
    """
    Compare peak contiguous-allocation waste vs paged allocation waste.

    Scenario: 8 sequences, each requesting between 10 and 150 tokens.
    """
    import random
    random.seed(42)

    total_pages = total_memory_slots // page_size
    cache = PagedKVCache(total_pages=total_pages, page_size=page_size)

    requests = [(f"seq{i}", random.randint(10, 150)) for i in range(8)]

    logger.info("=== Fragmentation Comparison (paged vs contiguous) ===")
    logger.info("page_size=%d  total_pages=%d  total_slots=%d", page_size, total_pages, total_memory_slots)
    logger.info("")

    # Paged allocation
    for seq_id, tokens in requests:
        try:
            result = cache.allocate(seq_id, tokens)
            logger.info("Paged: %s", result.summary())
        except RuntimeError as exc:
            logger.warning("Paged allocation failed: %s", exc)

    paged_stats = cache.stats()
    logger.info("Paged pool stats: %s", paged_stats.summary())

    # Contiguous allocation: each sequence reserves ceil(tokens/page_size)*page_size slots
    logger.info("")
    contiguous_waste = 0
    contiguous_total_reserved = 0
    for _seq_id, tokens in requests:
        reserved = -(-tokens // page_size) * page_size  # ceiling * page_size
        waste = reserved - tokens
        contiguous_waste += waste
        contiguous_total_reserved += reserved

    cont_frag_pct = contiguous_waste / contiguous_total_reserved * 100.0 if contiguous_total_reserved else 0.0
    logger.info(
        "Contiguous: total_reserved=%d  wasted=%d  internal_frag=%.1f%%",
        contiguous_total_reserved,
        contiguous_waste,
        cont_frag_pct,
    )
    logger.info(
        "Note: contiguous also suffers external fragmentation when requests end and gaps form."
    )


def main() -> None:
    # Demonstrate page allocation and reuse
    cache = PagedKVCache(total_pages=10, page_size=16)

    logger.info("=== PagedKVCache Demo ===")
    logger.info("total_pages=10, page_size=16 (160 total token slots)")
    logger.info("")

    # Allocate two sequences
    r1 = cache.allocate("seq-A", tokens_needed=40)  # needs ceil(40/16)=3 pages
    r2 = cache.allocate("seq-B", tokens_needed=20)  # needs ceil(20/16)=2 pages
    logger.info("After allocating seq-A (40 tok) and seq-B (20 tok):")
    logger.info("  %s", r1.summary())
    logger.info("  %s", r2.summary())
    logger.info("  Pool: %s", cache.stats().summary())

    # seq-B finishes — free its pages
    freed = cache.free("seq-B")
    logger.info("")
    logger.info("seq-B finished, freed pages: %s", freed)
    logger.info("  Pool: %s", cache.stats().summary())

    # New sequence seq-C reuses seq-B's pages immediately
    r3 = cache.allocate("seq-C", tokens_needed=25)  # needs 2 pages
    logger.info("New seq-C (25 tok) allocated immediately to freed pages:")
    logger.info("  %s", r3.summary())
    logger.info("  Pool: %s", cache.stats().summary())

    # Demonstrate incremental allocation (growth during generation)
    logger.info("")
    logger.info("=== Incremental allocation (simulating token-by-token growth) ===")
    gen_cache = PagedKVCache(total_pages=20, page_size=16)
    gen_cache.allocate("gen-seq", tokens_needed=16)  # 1 page for prompt
    logger.info("After prompt (16 tok): pages=%s", gen_cache.pages_for("gen-seq"))
    gen_cache.allocate("gen-seq", tokens_needed=32)  # grows to 2 pages
    logger.info("After 32 tok: pages=%s", gen_cache.pages_for("gen-seq"))
    gen_cache.allocate("gen-seq", tokens_needed=48)  # grows to 3 pages
    logger.info("After 48 tok: pages=%s", gen_cache.pages_for("gen-seq"))

    logger.info("")
    compare_fragmentation(total_memory_slots=256, page_size=16)


if __name__ == "__main__":
    main()
```

### Fragmentation: paged vs contiguous — explanation

| Property | Contiguous | Paged |
|---|---|---|
| Internal fragmentation | Up to `page_size - 1` per sequence | Up to `page_size - 1` per sequence (same) |
| External fragmentation | Yes — completed sequences leave gaps | None — any free page serves any sequence |
| Peak memory at fixed batch | Must pre-allocate max context per sequence | Allocate only actual tokens used |
| Reuse latency | Must find contiguous region | Instant — any free page works |

Both approaches have identical internal fragmentation (up to `page_size-1` wasted slots in the last page). PagedAttention eliminates external fragmentation entirely, which is the dominant form in LLM serving where requests have wildly different lengths.

### Pytest tests

```python
"""tests/test_ex3_paged_kv_cache.py"""

from __future__ import annotations

import pytest

from solutions.week_10.ex3_paged_kv_cache import PagedKVCache


class TestPagedKVCacheAllocate:
    def test_allocate_single_sequence_exact_page(self) -> None:
        cache = PagedKVCache(total_pages=4, page_size=16)
        result = cache.allocate("s1", tokens_needed=16)
        assert len(result.pages_allocated) == 1
        assert result.tokens_covered == 16

    def test_allocate_ceiling_pages(self) -> None:
        """17 tokens on page_size=16 needs 2 pages."""
        cache = PagedKVCache(total_pages=4, page_size=16)
        result = cache.allocate("s1", tokens_needed=17)
        assert len(result.pages_allocated) == 2
        assert result.tokens_covered == 32

    def test_allocate_reduces_free_pool(self) -> None:
        cache = PagedKVCache(total_pages=5, page_size=16)
        cache.allocate("s1", tokens_needed=32)  # 2 pages
        assert cache.free_page_count == 3

    def test_allocate_two_sequences_disjoint_pages(self) -> None:
        """Two sequences must hold disjoint page IDs."""
        cache = PagedKVCache(total_pages=8, page_size=16)
        r1 = cache.allocate("s1", tokens_needed=32)
        r2 = cache.allocate("s2", tokens_needed=32)
        assert not set(r1.pages_allocated) & set(r2.pages_allocated)

    def test_allocate_out_of_memory_raises(self) -> None:
        cache = PagedKVCache(total_pages=2, page_size=16)
        cache.allocate("s1", tokens_needed=32)  # uses all 2 pages
        with pytest.raises(RuntimeError, match="Out of pages"):
            cache.allocate("s2", tokens_needed=16)

    def test_incremental_allocation_adds_pages(self) -> None:
        """Calling allocate again for the same seq with more tokens adds pages."""
        cache = PagedKVCache(total_pages=10, page_size=16)
        cache.allocate("s1", tokens_needed=16)  # 1 page
        cache.allocate("s1", tokens_needed=32)  # grows to 2 pages
        assert len(cache.pages_for("s1")) == 2

    def test_incremental_allocation_no_duplicate_pages(self) -> None:
        cache = PagedKVCache(total_pages=10, page_size=16)
        cache.allocate("s1", tokens_needed=16)
        pages_before = set(cache.pages_for("s1"))
        cache.allocate("s1", tokens_needed=32)
        pages_after = set(cache.pages_for("s1"))
        # Original page still present, one new page added
        assert pages_before.issubset(pages_after)
        assert len(pages_after) == 2


class TestPagedKVCacheFree:
    def test_free_returns_pages_to_pool(self) -> None:
        cache = PagedKVCache(total_pages=4, page_size=16)
        cache.allocate("s1", tokens_needed=32)  # 2 pages
        assert cache.free_page_count == 2
        cache.free("s1")
        assert cache.free_page_count == 4

    def test_freed_pages_reusable_immediately(self) -> None:
        cache = PagedKVCache(total_pages=2, page_size=16)
        cache.allocate("s1", tokens_needed=32)
        freed = cache.free("s1")
        # Immediately allocate same slots for a different sequence
        result = cache.allocate("s2", tokens_needed=32)
        assert set(result.pages_allocated) == set(freed)

    def test_free_unknown_seq_raises(self) -> None:
        cache = PagedKVCache(total_pages=4, page_size=16)
        with pytest.raises(KeyError, match="not found"):
            cache.free("nonexistent")

    def test_double_free_raises(self) -> None:
        cache = PagedKVCache(total_pages=4, page_size=16)
        cache.allocate("s1", tokens_needed=16)
        cache.free("s1")
        with pytest.raises(KeyError):
            cache.free("s1")


class TestPagedKVCacheStats:
    def test_stats_after_allocation(self) -> None:
        cache = PagedKVCache(total_pages=8, page_size=16)
        cache.allocate("s1", tokens_needed=32)
        cache.allocate("s2", tokens_needed=16)
        stats = cache.stats()
        assert stats.total_pages == 8
        assert stats.allocated_pages == 3
        assert stats.free_pages == 5
        assert stats.sequences_active == 2

    def test_stats_empty_cache(self) -> None:
        cache = PagedKVCache(total_pages=10, page_size=16)
        stats = cache.stats()
        assert stats.free_pages == 10
        assert stats.allocated_pages == 0
        assert stats.sequences_active == 0
        assert stats.fragmentation_pct == 0.0

    def test_fragmentation_exact_page_boundary(self) -> None:
        """Allocating exactly page_size tokens: no internal fragmentation."""
        cache = PagedKVCache(total_pages=4, page_size=16)
        cache.allocate("s1", tokens_needed=16)
        stats = cache.stats()
        assert stats.fragmentation_pct == pytest.approx(0.0)

    def test_fragmentation_partial_last_page(self) -> None:
        """Allocating 17 tokens on page_size=16: 15/32 slots wasted = 46.9%."""
        cache = PagedKVCache(total_pages=4, page_size=16)
        cache.allocate("s1", tokens_needed=17)
        stats = cache.stats()
        # 1 slot used of 16 in last page → 15/32 = 46.875% internal frag
        assert stats.fragmentation_pct == pytest.approx(15.0 / 32.0 * 100.0, abs=0.1)


class TestPagedKVCacheEdgeCases:
    def test_invalid_total_pages_raises(self) -> None:
        with pytest.raises(ValueError):
            PagedKVCache(total_pages=0, page_size=16)

    def test_invalid_page_size_raises(self) -> None:
        with pytest.raises(ValueError):
            PagedKVCache(total_pages=4, page_size=0)

    def test_allocated_sequences_list(self) -> None:
        cache = PagedKVCache(total_pages=8, page_size=16)
        cache.allocate("alpha", tokens_needed=16)
        cache.allocate("beta", tokens_needed=16)
        seqs = cache.allocated_sequences
        assert "alpha" in seqs
        assert "beta" in seqs
```
