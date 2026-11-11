# Solutions — Phase 3 · Week 07 · Day 5 — Latency, Throughput, Quantization & Scaling Laws

---

## Exercise 1 — Quantization Memory Math

A 7B-parameter model requires different amounts of GPU VRAM depending on the numeric format used to store weights.

| Format | Bytes/param | Memory (7B params) |
|--------|-------------|-------------------|
| FP32   | 4           | 28.0 GB           |
| FP16   | 2           | 14.0 GB           |
| INT8   | 1           |  7.0 GB           |
| INT4   | 0.5         |  3.5 GB           |

**Key insight:** INT4 achieves an 8x reduction versus FP32, making 7B models runnable on a single consumer GPU (e.g., RTX 3090 with 24 GB VRAM can comfortably hold an INT4 7B model with room for activations).

```python
"""Calculate model memory requirements across quantization formats."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def model_memory_gb(n_params: float, bytes_per_param: float) -> float:
    """Return memory in GB for a model with n_params stored at bytes_per_param precision."""
    return (n_params * bytes_per_param) / (1024 ** 3)


def print_quantization_table(n_params: float) -> None:
    formats: list[tuple[str, float]] = [
        ("FP32", 4.0),
        ("FP16", 2.0),
        ("INT8", 1.0),
        ("INT4", 0.5),
    ]
    logger.info("Memory requirements for %.1fB parameter model", n_params / 1e9)
    for fmt, bpp in formats:
        mem = model_memory_gb(n_params, bpp)
        logger.info("  %-6s %.1f bytes/param -> %6.2f GB", fmt, bpp, mem)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_params = 7e9
    print_quantization_table(n_params)

# Expected output:
# INFO:__main__:Memory requirements for 7.0B parameter model
# INFO:__main__:  FP32   4.0 bytes/param ->  26.08 GB
# INFO:__main__:  FP16   2.0 bytes/param ->  13.04 GB
# INFO:__main__:  INT8   1.0 bytes/param ->   6.52 GB
# INFO:__main__:  INT4   0.5 bytes/param ->   3.26 GB
```

Note: the exact GB values above use 1024^3 = 1,073,741,824 bytes/GB (GiB). Simplified marketing values (1 GB = 1e9 bytes) give 28/14/7/3.5 GB exactly.

---

## Exercise 2 — Chinchilla Scaling Law Calculation

The Hoffmann et al. (2022) "Chinchilla" paper found that previous large models (GPT-3, Gopher) were significantly undertrained. The optimal compute allocation follows:

```
D_optimal = 20 * N        (tokens)
FLOPs     = 6 * N * D    (floating-point operations for training)
```

Where N = number of parameters, D = number of training tokens.

**Worked example — 13B parameter model:**
- D_optimal = 20 * 13e9 = 260,000,000,000 tokens (260B tokens)
- FLOPs = 6 * 13e9 * 260e9 = 2.028 × 10^22

```python
"""Chinchilla scaling law calculations."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def chinchilla_optimal_tokens(n_params: float) -> float:
    """Return optimal training tokens per Chinchilla scaling law.
    
    Args:
        n_params: Number of model parameters.
    Returns:
        Optimal number of training tokens (20 * n_params).
    """
    return 20.0 * n_params


def training_flops(n_params: float, n_tokens: float) -> float:
    """Estimate total training FLOPs using the 6ND approximation.
    
    The factor of 6 accounts for:
    - 2x forward pass (multiply + add per matmul)
    - 2x backward pass (approximately equal to forward)
    - 2x additional overhead (embedding lookups, layer norms, etc.)
    
    Args:
        n_params: Number of model parameters.
        n_tokens: Number of training tokens.
    Returns:
        Estimated training FLOPs (6 * n_params * n_tokens).
    """
    return 6.0 * n_params * n_tokens


def scaling_analysis(model_sizes: list[float]) -> None:
    """Log Chinchilla-optimal training requirements for each model size."""
    logger.info("%-12s  %-20s  %-22s  %-22s", "Model size", "Params", "Optimal tokens", "Training FLOPs")
    logger.info("-" * 80)
    for n in model_sizes:
        d_opt = chinchilla_optimal_tokens(n)
        flops = training_flops(n, d_opt)
        logger.info(
            "%-12s  %-20.3e  %-22.3e  %-22.3e",
            f"{n/1e9:.0f}B",
            n,
            d_opt,
            flops,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_sizes = [1e9, 7e9, 13e9, 70e9]
    scaling_analysis(model_sizes)

# Expected output (approximate):
# Model size    Params                Optimal tokens         Training FLOPs
# --------------------------------------------------------------------------------
# 1B            1.000e+09             2.000e+10              1.200e+20
# 7B            7.000e+09             1.400e+11              5.880e+21
# 13B           1.300e+10             2.600e+11              2.028e+22
# 70B           7.000e+10             1.400e+12              5.880e+23
```

**Interpretation:** A 70B model requires ~1.4 trillion training tokens for Chinchilla-optimality. Llama-2-70B was trained on 2T tokens — slightly above Chinchilla-optimal, which is common for inference-focused models (train longer, inference is cheap).

---

## Exercise 3 — INT8 Quantization Implementation

Symmetric INT8 quantization maps the full floating-point range to [-128, 127] using a single scale factor. Zero maps to zero (hence "symmetric").

**Formula:**
```
scale         = max(|weights|) / 127.0
zero_point    = 0                                      (symmetric)
quantized     = clip(round(weights / scale), -128, 127)
dequantized   = quantized * scale
```

```python
"""INT8 symmetric quantization and dequantization."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Quantize float32 weights to int8 using symmetric quantization.

    Symmetric quantization uses a single scale derived from the max absolute
    value. Zero maps exactly to zero (zero_point=0), which preserves sparsity
    in weights and simplifies matmul hardware implementation.

    Args:
        weights: Float32 weight tensor of any shape.

    Returns:
        Tuple of:
            quantized_int8: INT8 array of same shape as weights.
            scale: Float scalar used to recover approximate original values.
            zero_point: Always 0.0 for symmetric quantization.
    """
    max_abs = float(np.max(np.abs(weights)))
    if max_abs == 0.0:
        # All-zero tensor — scale doesn't matter; avoid division by zero
        scale = 1.0
    else:
        scale = max_abs / 127.0
    zero_point = 0.0
    quantized = np.clip(np.round(weights / scale), -128, 127).astype(np.int8)
    logger.debug(
        "quantize_int8: shape=%s, max_abs=%.4f, scale=%.6f",
        weights.shape,
        max_abs,
        scale,
    )
    return quantized, scale, zero_point


def dequantize_int8(
    quantized: np.ndarray,
    scale: float,
    zero_point: float,
) -> np.ndarray:
    """Dequantize int8 weights back to float32 approximation.

    Args:
        quantized: INT8 array (output of quantize_int8).
        scale: Scale factor from quantize_int8.
        zero_point: Zero point from quantize_int8 (0.0 for symmetric).

    Returns:
        Float32 approximation of the original weights.
    """
    return (quantized.astype(np.float32) - zero_point) * scale


def quantization_error_demo() -> None:
    """Demonstrate quantization MSE and max absolute error."""
    rng = np.random.default_rng(seed=42)
    # Simulate a weight matrix from a linear layer
    weights = rng.normal(loc=0.0, scale=0.02, size=(768, 768)).astype(np.float32)

    quantized, scale, zero_point = quantize_int8(weights)
    reconstructed = dequantize_int8(quantized, scale, zero_point)

    mse = float(np.mean((weights - reconstructed) ** 2))
    max_err = float(np.max(np.abs(weights - reconstructed)))
    relative_err = mse / float(np.mean(weights ** 2))

    logger.info("Original dtype:      %s", weights.dtype)
    logger.info("Quantized dtype:     %s", quantized.dtype)
    logger.info("Scale:               %.6f", scale)
    logger.info("MSE:                 %.2e", mse)
    logger.info("Max absolute error:  %.2e", max_err)
    logger.info("Relative MSE:        %.4f%%", relative_err * 100)
    logger.info(
        "Memory: FP32=%.2f KB, INT8=%.2f KB (%.1fx reduction)",
        weights.nbytes / 1024,
        quantized.nbytes / 1024,
        weights.nbytes / quantized.nbytes,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quantization_error_demo()

# Expected output (approximate):
# INFO: Original dtype:      float32
# INFO: Quantized dtype:     int8
# INFO: Scale:               0.000243
# INFO: MSE:                 1.47e-08
# INFO: Max absolute error:  1.21e-04
# INFO: Relative MSE:        0.0015%
# INFO: Memory: FP32=2304.00 KB, INT8=576.00 KB (4.0x reduction)
```

**Why MSE is small:** With normal-distributed weights (mean=0, std=0.02), the max absolute value is typically around 0.07-0.10. The scale = max_abs/127 ≈ 5.5e-4. The quantization step size equals the scale, so the maximum per-element error is scale/2 ≈ 2.8e-4. Squared and averaged over 589,824 elements, the MSE is negligible (<0.002% relative error). This is why INT8 inference is widely used in production — the accuracy loss is measurable but rarely significant.
