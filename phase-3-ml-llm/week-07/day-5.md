# Phase 3 · Week 07 · Day 5 — Latency, Throughput, Quantization & Scaling Laws Intuition

> Objective: Understand the performance metrics and efficiency techniques that govern LLM serving and training costs.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

---

## Concept

LLMs in production face two competing constraints: **latency** (how fast does one user get a response) and **throughput** (how many tokens per second can the system generate across all users). These are not the same thing and optimizing one often hurts the other.

**Latency** is decomposed into two phases:

- **TTFT (Time to First Token)**: the delay from request receipt to the first output token. Driven by the *prefill* phase — processing the full input prompt.
- **TBT (Time Between Tokens)**: the delay between consecutive output tokens. Driven by the *decode* phase — one forward pass per output token.

**Throughput** is the total tokens generated per second across all concurrent requests. It improves with batching because fixed hardware costs (loading weights) are amortized across many sequences.

**Quantization** reduces model weight precision from FP32/FP16 to INT8 or INT4. Smaller weights mean:
1. Less GPU memory required.
2. Faster memory-bandwidth-bound operations (matmul with INT8 is faster on modern accelerators).
3. Slight accuracy loss that depends on the quantization scheme.

**Scaling laws** (Hoffmann et al. 2022, "Chinchilla") describe how LLM loss depends on model size N (parameters) and training data size D (tokens):

```
L(N, D) ≈ A/N^α + B/D^β + L_∞
```

The key practical result: to minimize loss given a fixed compute budget, you should train a *smaller* model on *more data* than previously assumed. The Chinchilla-optimal ratio is approximately **20 tokens per parameter**.

---

## How it works internally

### Prefill vs Decode

During **prefill**, the full input sequence of length T is processed in one batched matrix multiply. All T token positions are computed simultaneously. Computation scales as O(T^2) due to full self-attention. This is compute-bound and benefits from large batch sizes.

During **decode**, only *one new token* is generated per forward pass. The KV cache stores past keys/values, so the attention computation per step is O(T) not O(T^2). But one step must complete before the next begins — this is *memory-bandwidth-bound* because the full model weights must be streamed through the GPU's memory bus.

### Why batching helps throughput

A GPU spends a fixed cost loading model weights regardless of how many sequences it processes simultaneously. If weights occupy W bytes and the GPU's memory bandwidth is B bytes/sec, the minimum time per decode step is W/B — whether you process 1 sequence or 100. Batching 100 sequences through the same weight load gives 100x more tokens for the same cost. This is **hardware amortization**.

### Quantization internals

FP16 stores each weight in 16 bits (2 bytes). INT8 uses 1 byte. INT4 uses 0.5 bytes.

The quantization mapping:
```
x_int8 = round(x_fp16 / scale) + zero_point
x_dequant = (x_int8 - zero_point) * scale
```

Where `scale = (x_max - x_min) / (2^bits - 1)` and `zero_point` shifts the range.

The **error** is the rounding error introduced at each weight. For most transformer weights, the distribution is approximately Gaussian, and INT8 quantization introduces errors of ~0.1% relative to FP16 values. INT4 is coarser and requires more sophisticated schemes (GPTQ, AWQ, GGUF) to maintain accuracy.

**PTQ (Post-Training Quantization)**: quantize after training completes. Fast but can lose accuracy for aggressive compression.

**QAT (Quantization-Aware Training)**: simulate quantization noise during training so the model learns to be robust to it. Better accuracy, but requires retraining.

Common production formats:
- **GGUF**: portable format used by llama.cpp, supports mixed precision per layer.
- **GPTQ**: layer-wise second-order PTQ, minimizes reconstruction error.
- **AWQ (Activation-aware Weight Quantization)**: protects salient channels based on activation magnitudes.

### Compute budget and FLOPs

For a transformer with N parameters trained on D tokens:

```
Total FLOPs ≈ 6 * N * D
```

The factor 6 comes from: 2 (forward pass matmuls) × 3 (forward + backward + activation checkpointing overhead).

The Chinchilla insight: for a fixed compute budget C = 6ND, the optimal allocation is N ≈ C^0.5 and D ≈ C^0.5, meaning scale both equally. GPT-3 (175B parameters, 300B tokens = 1.7 tokens/param) was undertrained relative to this optimum.

---

## Worked example

This example simulates INT8 quantization of a weight matrix and measures error and memory savings.

```python
# quantization_demo.py
# Run: uv run python quantization_demo.py
# Install: uv add numpy

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def quantize_to_int8(
    weights: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Symmetric per-tensor INT8 quantization.

    Returns (quantized_int8, scale, zero_point).
    For symmetric quantization, zero_point = 0.
    """
    w_max = float(np.abs(weights).max())
    # Scale maps [-w_max, w_max] -> [-127, 127]
    scale = w_max / 127.0
    if scale == 0.0:
        scale = 1e-8  # guard against all-zero tensor
    quantized = np.round(weights / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale, 0.0


def dequantize(
    quantized: np.ndarray,
    scale: float,
    zero_point: float,
) -> np.ndarray:
    """Reconstruct FP32 approximation from INT8."""
    return (quantized.astype(np.float32) - zero_point) * scale


def quantization_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float]:
    """Compute absolute and relative error metrics."""
    diff = original.astype(np.float32) - reconstructed
    abs_err = float(np.abs(diff).mean())
    rel_err = abs_err / (float(np.abs(original).mean()) + 1e-12)
    max_err = float(np.abs(diff).max())
    return {"mean_abs_error": abs_err, "mean_rel_error": rel_err, "max_abs_error": max_err}


def memory_bytes(arr: np.ndarray) -> int:
    """Return memory footprint in bytes."""
    return arr.nbytes


def scaling_law_tokens(n_params: int) -> int:
    """Chinchilla-optimal token count: 20 tokens per parameter."""
    return n_params * 20


def compute_flops(n_params: int, n_tokens: int) -> float:
    """Approximate training FLOPs: 6 * N * D."""
    return 6.0 * n_params * n_tokens


def main() -> None:
    rng = np.random.default_rng(seed=42)

    # Simulate a weight matrix from a linear layer (e.g., 512 x 512)
    rows, cols = 512, 512
    # Typical transformer weights ~ N(0, 0.02)
    weights_fp32 = rng.normal(loc=0.0, scale=0.02, size=(rows, cols)).astype(np.float32)

    log.info("Original weights | dtype=fp32 | shape=%s", weights_fp32.shape)
    log.info("Memory (FP32): %d bytes = %.2f KB", memory_bytes(weights_fp32), memory_bytes(weights_fp32) / 1024)

    # Quantize to INT8
    q_int8, scale, zero_point = quantize_to_int8(weights_fp32)
    log.info("Quantized weights | dtype=int8 | scale=%.6f", scale)
    log.info("Memory (INT8): %d bytes = %.2f KB", memory_bytes(q_int8), memory_bytes(q_int8) / 1024)

    # Dequantize and measure error
    reconstructed = dequantize(q_int8, scale, zero_point)
    errors = quantization_error(weights_fp32, reconstructed)

    log.info("Quantization errors:")
    for k, v in errors.items():
        log.info("  %s = %.6f", k, v)

    # Memory savings
    fp32_bytes = memory_bytes(weights_fp32)
    int8_bytes = memory_bytes(q_int8)
    compression_ratio = fp32_bytes / int8_bytes
    log.info("Compression ratio (FP32 -> INT8): %.1fx", compression_ratio)

    # Simulate INT4: pack two 4-bit values per byte
    # Symmetric INT4 range: [-7, 7]
    scale_int4 = float(np.abs(weights_fp32).max()) / 7.0
    q_int4_values = np.round(weights_fp32 / scale_int4).clip(-7, 7).astype(np.int8)
    # INT4 packing: 2 values per byte
    int4_bytes = q_int4_values.size // 2
    log.info("Memory (INT4, packed): %d bytes = %.2f KB", int4_bytes, int4_bytes / 1024)
    log.info("Compression ratio (FP32 -> INT4): %.1fx", fp32_bytes / int4_bytes)

    # Scaling laws demo
    log.info("\n--- Chinchilla Scaling Laws ---")
    for n_params in [7_000_000_000, 13_000_000_000, 70_000_000_000]:
        d_optimal = scaling_law_tokens(n_params)
        flops = compute_flops(n_params, d_optimal)
        log.info(
            "Model: %dB params | Optimal tokens: %.1fB | FLOPs: %.2e",
            n_params // 1_000_000_000,
            d_optimal / 1e9,
            flops,
        )

    # Fixed compute budget: what's optimal?
    log.info("\n--- Fixed Compute Budget: 1e23 FLOPs ---")
    compute_budget = 1e23
    # From 6ND = C and N ~= D (Chinchilla balanced)
    # 6 * N^2 = C  ->  N = sqrt(C/6)
    n_optimal = int((compute_budget / 6) ** 0.5)
    d_optimal_budget = int(compute_budget / (6 * n_optimal))
    log.info("Optimal N: %.2fB params", n_optimal / 1e9)
    log.info("Optimal D: %.2fB tokens", d_optimal_budget / 1e9)
    log.info("Tokens/param ratio: %.1f", d_optimal_budget / n_optimal)


if __name__ == "__main__":
    main()
```

Expected output (approximate):
```
INFO: Original weights | dtype=fp32 | shape=(512, 512)
INFO: Memory (FP32): 1048576 bytes = 1024.00 KB
INFO: Quantized weights | dtype=int8 | scale=0.000615
INFO: Memory (INT8): 262144 bytes = 256.00 KB
INFO: Quantization errors:
INFO:   mean_abs_error = 0.000176
INFO:   mean_rel_error = 0.014...
INFO:   max_abs_error = 0.000308
INFO: Compression ratio (FP32 -> INT8): 4.0x
INFO: Memory (INT4, packed): 131072 bytes = 128.00 KB
INFO: Compression ratio (FP32 -> INT4): 8.0x
INFO: --- Chinchilla Scaling Laws ---
INFO: Model: 7B params | Optimal tokens: 140.0B | FLOPs: 5.88e+21
INFO: Model: 13B params | Optimal tokens: 260.0B | FLOPs: 2.03e+22
INFO: Model: 70B params | Optimal tokens: 1400.0B | FLOPs: 5.88e+23
```

---

## Exercises

1. **Quantization sensitivity**: modify the worked example to quantize only 90% of weights (leave 10% as FP16). Measure: how much does error drop compared to full INT8? What does this suggest about mixed-precision schemes like GPTQ that keep outlier channels in FP16?

2. **Scaling law calculation**: a startup has a compute budget of 3.2e22 FLOPs. Using the Chinchilla formula (6ND = C, scale N and D equally), compute: (a) the optimal N and D, (b) the optimal dataset size in tokens, (c) the training tokens-per-parameter ratio. Compare your answer to GPT-3 (175B params, 300B tokens = 1.7 tokens/param) — how undertrained was GPT-3?

3. **Latency analysis**: a model has 7B FP16 parameters (14 GB). The GPU memory bandwidth is 2 TB/s. (a) What is the theoretical minimum time per decode step (1 sequence)? (b) What is the minimum time if you batch 32 sequences simultaneously, assuming weights still fit in memory? (c) At what batch size does prefill latency (assume 512-token prompt, ~2e12 FLOPs/prefill) equal decode latency? Use compute throughput of 312 TFLOP/s.

(Solutions: ../solutions/week-07/day-5.md)

---

## Self-check

- [ ] I can explain the difference between TTFT and TBT, and which phase dominates each.
- [ ] I can derive the memory savings formula for INT8 vs FP32 quantization and compute the compression ratio.
- [ ] I can state the Chinchilla scaling law and explain why GPT-3 was undertrained relative to compute budget.
- [ ] I can calculate approximate training FLOPs given model size and dataset size using the 6ND formula.

---

## Supplementary links

- Hoffmann et al. 2022 — "Training Compute-Optimal Large Language Models" (Chinchilla paper): https://arxiv.org/abs/2203.15556
- Dettmers et al. 2022 — "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale": https://arxiv.org/abs/2208.07339
- Frantar et al. 2022 — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers": https://arxiv.org/abs/2210.17323
- Lin et al. 2023 — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration": https://arxiv.org/abs/2306.00978
- "Roofline model" explainer for memory-bound vs compute-bound: https://en.wikipedia.org/wiki/Roofline_model
