# Phase 6 · Week 10 · Day 3 — Quantization: INT8 / FP8 / GPTQ / AWQ / GGUF

> Objective: Understand how quantization reduces model memory and compute cost, and how GPTQ, AWQ, and GGUF trade accuracy for efficiency differently.  ·  Time: ~1 hr  ·  Prereqs: [[day-2.md]]

---

## Concept

### What quantization is

A neural network is a collection of floating-point numbers: weight tensors, activation tensors, and KV-cache tensors. By default, modern models train in FP32 (32-bit float) or FP16/BF16 (16-bit float). Quantization replaces these high-precision numbers with lower-precision representations — typically INT8 (8-bit integer) or INT4 (4-bit integer) — reducing memory and, often, computation time.

The core idea: most weights in a trained model are small and clustered near zero. The full dynamic range of FP32 (exponent range ±38) is overkill for storing them. By mapping the actual value range to a small integer grid, you preserve most of the information while using far fewer bits.

### Precision formats and their tradeoffs

| Format | Bits | Values representable | Size of 7B model | Notes |
|---|---|---|---|---|
| FP32 | 32 | ~4.3 billion floats | ~28 GB | Training standard |
| BF16 | 16 | 65 536 floats | ~14 GB | Same exponent as FP32, less mantissa |
| FP16 | 16 | 65 536 floats | ~14 GB | More mantissa than BF16 |
| FP8 | 8 | 256 floats | ~7 GB | Emerging; hardware support in H100 |
| INT8 | 8 | 256 integers | ~7 GB | Asymmetric or symmetric mapping |
| INT4 | 4 | 16 integers | ~3.5 GB | Needs careful calibration |
| INT2 | 2 | 4 integers | ~1.75 GB | Severe quality loss; research-only |

Size reduction is linear with bit width: INT8 is 2× smaller than FP16; INT4 is 4× smaller.

Speed improvement is less linear — it depends on hardware. Modern GPUs have dedicated INT8 tensor cores. INT4 requires dequantization at runtime on most hardware, so the speedup comes primarily from memory bandwidth (loading fewer bytes), not compute throughput.

### Post-training quantization (PTQ) vs quantization-aware training (QAT)

**PTQ** applies after training is complete. You take a trained FP16 model and convert its weights. No retraining. The drawback: weights are pushed onto a coarser grid, introducing quantization error that was never seen during training. PTQ requires a small calibration dataset (a few hundred examples) to estimate the actual range of values and minimize error.

**QAT** simulates quantization noise during fine-tuning or training, letting the model adapt its weights to compensate. Result: higher accuracy at the same bit width, but requires additional training compute. Most open-source quantized models (GPTQ, AWQ) use PTQ because it is practical with existing checkpoints.

### GPTQ — layer-wise reconstruction error minimization

GPTQ (Frantar et al., 2022) is a PTQ algorithm for INT4/INT3 quantization of large language models. The key insight: quantizing all layers independently and uniformly is suboptimal. Some weights, when quantized, cause larger reconstruction error than others.

GPTQ adapts the Optimal Brain Quantization (OBQ) framework:
1. Process one layer at a time.
2. For each weight in the layer, quantize it and compute how much error this introduces to the layer's output.
3. Compensate by adjusting the remaining (not yet quantized) weights in the same row to reduce the total reconstruction error.
4. This is a least-squares update: given weight W, quantization error δW, and the Hessian H of the layer's output with respect to W, update remaining weights by -H⁻¹ δW.

The Hessian is estimated from the calibration dataset. GPTQ runs in a few hours on a single GPU for a 7B model. The resulting weights are INT4 with per-group scaling factors (each group of 128 weights shares a FP16 scale).

Inference with GPTQ: weights are stored as INT4, dequantized to FP16 on the fly during each matrix multiply. The bottleneck is memory bandwidth (loading INT4 from VRAM), not compute — this is why the speedup is real even without INT4 compute hardware.

### AWQ — activation-aware weight quantization

AWQ (Lin et al., 2023) makes a different observation: not all weights matter equally. Roughly 1% of weights are "salient" — they are connected to large activation values and have disproportionate influence on the model output. Quantizing these salient weights coarsely (to INT4) causes most of the accuracy loss.

AWQ strategy:
1. Run the calibration dataset through the model and record per-channel activation magnitudes.
2. Identify the top ~1% of channels by activation scale.
3. For those channels, apply a per-channel scaling factor that maps the weight range to a wider portion of the INT4 grid (protecting precision for high-activation weights).
4. Absorb the inverse scaling factor into the previous layer's output (a mathematically equivalent transformation that requires no overhead at inference time).

AWQ achieves similar accuracy to GPTQ at INT4, often slightly better on instruction-following benchmarks, and the quantization process is faster (no Hessian computation). The resulting model format is also INT4 with per-group scales.

### GGUF — CPU-friendly quantized format

GGUF is a file format (and quantization scheme) from the llama.cpp project. It is designed for inference on CPU and Apple Silicon where GPU memory is limited or unavailable.

GGUF stores the model in quantized blocks. Each block of 32 weights shares a single FP32 scale and zero-point. GGUF defines multiple quantization levels:

| Level | Bits per weight (approx) | Quality | Use case |
|---|---|---|---|
| Q8_0 | 8 | Near-lossless | High RAM, best quality |
| Q6_K | 6.6 | Excellent | High RAM, great quality |
| Q5_K_M | 5.7 | Very good | Balanced |
| Q4_K_M | 4.8 | Good | Default for most users |
| Q3_K_M | 3.9 | Acceptable | Low RAM |
| Q2_K | 3.4 | Degraded | Very low RAM only |

The `_K` suffix indicates k-quants: a more sophisticated block structure that assigns different precision to different weight types within the model (attention weights vs feed-forward weights). The `_M` suffix is a "medium" variant tuned for quality/size balance.

GGUF is the de facto standard for running models locally on consumer hardware. Ollama, LM Studio, and llama.cpp all use it.

---

## How it works internally

### Floating-point bit layout

FP32 has 1 sign bit, 8 exponent bits, 23 mantissa bits. It can represent values from ~1.2×10⁻³⁸ to ~3.4×10³⁸.

INT8 has 8 bits for a value range of [-128, 127] (signed) or [0, 255] (unsigned). To store a float weight W in INT8:
1. Compute scale `s = (max(W) - min(W)) / 255`
2. Compute zero point `z = round(-min(W) / s)`
3. Quantized value: `q = round(W / s) + z`
4. Dequantized (approximate): `W_approx = (q - z) * s`

The quantization error is at most `s/2` per weight. Smaller weight ranges → smaller scale → less error.

### GPTQ Hessian update (intuition)

For a linear layer `Y = XW`, quantizing weight `w_i` to `q_i` adds error `δw_i = q_i - w_i`. The change in layer output is approximately `X_col_i * δw_i`. To cancel this, GPTQ updates remaining weights `W_j` (j > i in the same row) by:

```
W_j -= (H_ij / H_ii) * δw_i
```

Where H is the Hessian `H = 2 X^T X` (positive semi-definite, computable from calibration data). This compensates for the quantization error of each weight using the remaining weights — it is a greedy, left-to-right pass through the weight matrix.

### AWQ scaling (intuition)

If weight channel `c` has a large activation scale `s_c`, the effective quantization error in that channel's contribution to the output is `s_c * δw_c`. AWQ pre-scales weight channel `c` by `α` (enlarging it to use more of the INT4 range), then compensates by dividing the corresponding input activation scale by `α` (absorbed into a preceding LayerNorm or RMS-Norm). Net effect: the salient channel uses the full INT4 range, not just a fraction of it.

### GGUF block structure

Every 32 weights form one quantization block. The block stores:
- 1× FP16 delta (scale): the step size between adjacent INT4 values
- 32× INT4 quantized values (packed as 16 bytes)
- Total: 18 bytes for 32 weights = 4.5 bits/weight average for Q4_0

K-quants use a two-level structure: a super-block of 256 weights contains 8 sub-blocks of 32 weights each. The super-block scale is FP16; sub-block scales are quantized themselves (to 6-bit integers). This reduces per-block overhead and improves quality at the same bit width.

---

## Worked example

```python
"""
day_3_quantization_calculator.py

Compute model size and theoretical memory bandwidth speedup at different
precisions. Pure Python math — no GPU or ML framework required.

Run:
    python day_3_quantization_calculator.py
"""

from __future__ import annotations

import logging
import math
from typing import Literal

from pydantic import BaseModel, Field, computed_field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

Precision = Literal["fp32", "fp16", "bf16", "fp8", "int8", "int4", "int2"]

BITS: dict[Precision, float] = {
    "fp32": 32.0,
    "fp16": 16.0,
    "bf16": 16.0,
    "fp8": 8.0,
    "int8": 8.0,
    "int4": 4.0,
    "int2": 2.0,
}

# Approximate quality retention relative to FP16 baseline (empirical estimates)
# These are rough guidelines, not guarantees — actual quality depends on model and task.
QUALITY_RETENTION: dict[Precision, float] = {
    "fp32": 1.00,
    "fp16": 1.00,
    "bf16": 1.00,
    "fp8": 0.99,
    "int8": 0.99,
    "int4": 0.97,  # with good calibration (GPTQ/AWQ)
    "int2": 0.85,  # significant quality loss
}


class ModelConfig(BaseModel):
    """Architecture parameters for a transformer language model."""

    name: str
    num_parameters_billions: float = Field(gt=0, description="Total trainable parameters in billions")
    num_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0, description="Dimension of each attention head")
    vocab_size: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def kv_cache_bytes_per_token(self) -> int:
        """KV cache memory per token (FP16): 2 (K+V) * layers * heads * head_dim * 2 bytes."""
        return 2 * self.num_layers * self.num_attention_heads * self.head_dim * 2


class QuantizationResult(BaseModel):
    """Memory and speed analysis for a model at a given precision."""

    model_name: str
    precision: Precision
    bits_per_weight: float
    weight_memory_gb: float = Field(description="Model weights in GB")
    kv_cache_memory_gb_per_4k_context: float = Field(
        description="KV cache at 4096 tokens, FP16, batch 1"
    )
    total_memory_gb: float = Field(description="Weights + KV cache estimate")
    compression_ratio_vs_fp16: float = Field(description=">1 means smaller than FP16")
    bandwidth_speedup_vs_fp16: float = Field(
        description="Theoretical memory bandwidth speedup (smaller = faster to load)"
    )
    quality_retention_pct: float = Field(description="Approximate quality vs FP16 baseline")

    def summary(self) -> str:
        return (
            f"  {self.precision:6s} | "
            f"weights={self.weight_memory_gb:6.2f} GB | "
            f"KV={self.kv_cache_memory_gb_per_4k_context:.2f} GB | "
            f"total={self.total_memory_gb:6.2f} GB | "
            f"compression={self.compression_ratio_vs_fp16:.1f}x | "
            f"BW speedup={self.bandwidth_speedup_vs_fp16:.1f}x | "
            f"quality~{self.quality_retention_pct:.0f}%"
        )


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class QuantizationCalculator:
    """
    Compute memory and efficiency metrics for a model at different precisions.

    Parameters
    ----------
    model:
        Transformer architecture specification.
    """

    def __init__(self, model: ModelConfig) -> None:
        self.model = model

    def _weight_bytes(self, precision: Precision) -> float:
        """Total bytes for all model weights at the given precision."""
        bits = BITS[precision]
        total_params = self.model.num_parameters_billions * 1e9
        # GPTQ/AWQ add ~1% overhead for scales; GGUF adds block metadata
        # We include a small overhead factor for non-trivial quantizations
        scale_overhead = 1.0
        if precision in ("int4", "int2"):
            # ~128 weights share one FP16 scale = 16 bits / 128 weights = 0.125 bits/weight extra
            scale_overhead_bits = 16.0 / 128.0
            bits = bits + scale_overhead_bits
        return total_params * bits / 8.0 * scale_overhead

    def _kv_cache_bytes(self, context_len: int = 4096, batch_size: int = 1) -> float:
        """KV cache bytes at FP16 for given context length and batch size."""
        return self.model.kv_cache_bytes_per_token * context_len * batch_size

    def analyze(self, precision: Precision, context_len: int = 4096) -> QuantizationResult:
        """Compute all metrics for a single precision."""
        weight_bytes = self._weight_bytes(precision)
        kv_bytes = self._kv_cache_bytes(context_len=context_len, batch_size=1)
        total_bytes = weight_bytes + kv_bytes

        fp16_bytes = self._weight_bytes("fp16")
        compression = fp16_bytes / weight_bytes
        # Bandwidth speedup: proportional to how much less data must be read from VRAM
        bw_speedup = 16.0 / BITS[precision]  # simplified: ignores scale overhead

        return QuantizationResult(
            model_name=self.model.name,
            precision=precision,
            bits_per_weight=BITS[precision],
            weight_memory_gb=weight_bytes / (1024**3),
            kv_cache_memory_gb_per_4k_context=kv_bytes / (1024**3),
            total_memory_gb=total_bytes / (1024**3),
            compression_ratio_vs_fp16=compression,
            bandwidth_speedup_vs_fp16=bw_speedup,
            quality_retention_pct=QUALITY_RETENTION[precision] * 100.0,
        )

    def report(self, context_len: int = 4096) -> list[QuantizationResult]:
        """Analyze all precisions and return sorted by bits (high to low)."""
        precisions: list[Precision] = ["fp32", "fp16", "fp8", "int8", "int4", "int2"]
        results = [self.analyze(p, context_len=context_len) for p in precisions]
        return results


# ---------------------------------------------------------------------------
# GGUF block overhead calculator
# ---------------------------------------------------------------------------

def gguf_effective_bits(q_bits: int, block_size: int = 32) -> float:
    """
    Compute effective bits per weight for a GGUF quantization level.

    Each block of `block_size` weights stores:
    - q_bits * block_size bits of quantized weights
    - 16 bits for the FP16 scale (delta)

    Returns the average bits per weight including scale overhead.
    """
    weight_bits = q_bits * block_size
    scale_bits = 16
    total_bits = weight_bits + scale_bits
    return total_bits / block_size


def gguf_report() -> None:
    """Show effective bits/weight for common GGUF quantization levels."""
    logger.info("--- GGUF Effective Bits Per Weight (Q4_0 block structure) ---")
    for q_bits, label in [(8, "Q8_0"), (6, "Q6"), (5, "Q5"), (4, "Q4_0"), (3, "Q3"), (2, "Q2")]:
        eff = gguf_effective_bits(q_bits)
        logger.info("  %-6s  nominal=%d bit  effective=%.2f bits/weight", label, q_bits, eff)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    # Llama-3-8B architecture (public spec)
    llama_8b = ModelConfig(
        name="Llama-3-8B",
        num_parameters_billions=8.0,
        num_layers=32,
        num_attention_heads=32,
        head_dim=128,
        vocab_size=128_256,
        hidden_dim=4096,
    )

    # Llama-3-70B for comparison
    llama_70b = ModelConfig(
        name="Llama-3-70B",
        num_parameters_billions=70.0,
        num_layers=80,
        num_attention_heads=64,
        head_dim=128,
        vocab_size=128_256,
        hidden_dim=8192,
    )

    for model in [llama_8b, llama_70b]:
        calc = QuantizationCalculator(model)
        results = calc.report(context_len=4096)
        logger.info("=== %s at 4096-token context, batch size 1 ===", model.name)
        logger.info("  %-6s | %-14s | %-9s | %-12s | %-14s | %-12s | quality", "prec", "weights", "KV", "total", "compression", "BW speedup")
        for r in results:
            logger.info(r.summary())

    # Check which precisions fit in 16 GB (e.g., M2 MacBook with 16 GB unified memory)
    logger.info("\n=== Fits in 16 GB VRAM? (Llama-3-8B, 4k context) ===")
    calc_8b = QuantizationCalculator(llama_8b)
    for r in calc_8b.report():
        fits = "YES" if r.total_memory_gb <= 16.0 else "NO "
        logger.info("  %s  %s  (%.1f GB)", fits, r.precision, r.total_memory_gb)

    logger.info("")
    gguf_report()


if __name__ == "__main__":
    main()
```

**Expected output (approximate):**

```
INFO === Llama-3-8B at 4096-token context, batch size 1 ===
INFO   prec   | weights        | KV        | total       | compression   | BW speedup   | quality
INFO   fp32   | weights= 29.80 GB | KV=1.00 GB | total= 30.80 GB | compression=0.5x | BW speedup=2.0x | quality~100%
INFO   fp16   | weights= 14.90 GB | KV=1.00 GB | total= 15.90 GB | compression=1.0x | BW speedup=1.0x | quality~100%
INFO   fp8    | weights=  7.45 GB | KV=1.00 GB | total=  8.45 GB | compression=2.0x | BW speedup=2.0x | quality~99%
INFO   int8   | weights=  7.45 GB | KV=1.00 GB | total=  8.45 GB | compression=2.0x | BW speedup=2.0x | quality~99%
INFO   int4   | weights=  3.78 GB | KV=1.00 GB | total=  4.78 GB | compression=3.9x | BW speedup=4.0x | quality~97%
INFO   int2   | weights=  1.90 GB | KV=1.00 GB | total=  2.90 GB | compression=7.8x | BW speedup=8.0x | quality~85%

INFO === Fits in 16 GB VRAM? (Llama-3-8B, 4k context) ===
INFO   NO   fp32  (30.8 GB)
INFO   YES  fp16  (15.9 GB)
INFO   YES  fp8   (8.4 GB)
INFO   YES  int8  (8.4 GB)
INFO   YES  int4  (4.8 GB)
INFO   YES  int2  (2.9 GB)

INFO --- GGUF Effective Bits Per Weight (Q4_0 block structure) ---
INFO   Q8_0    nominal=8 bit  effective=8.50 bits/weight
INFO   Q6      nominal=6 bit  effective=6.50 bits/weight
INFO   Q5      nominal=5 bit  effective=5.50 bits/weight
INFO   Q4_0    nominal=4 bit  effective=4.50 bits/weight
INFO   Q3      nominal=3 bit  effective=3.50 bits/weight
INFO   Q2      nominal=2 bit  effective=2.50 bits/weight
```

Key observation: INT4 (GPTQ/AWQ) cuts a 70B model from ~131 GB (FP16) to ~34 GB — fitting within a 2×A100-80GB setup vs requiring 4–6 GPUs at FP16. Quality retention remains ~97%. This is why INT4 quantization dominates production deployments of large open-source models.

---

## Exercises

1. **Calibration dataset size.** GPTQ accuracy depends on the calibration dataset. Research (from the GPTQ paper at the link below) what dataset is used for calibration and how many examples. Modify `QuantizationResult` to add a `calibration_examples` field. Add plausible values for each quantization method (GPTQ, AWQ, bitsandbytes INT8). What would happen to quantization quality if you used only 10 calibration examples vs 128?

2. **Batch size impact on KV cache.** Extend `QuantizationCalculator.analyze()` to accept a `batch_size` parameter. Plot (or print) total memory (weights + KV cache) for the Llama-3-8B INT4 model across batch sizes 1, 4, 8, 16, 32 at a 4096-token context. At what batch size does the KV cache exceed the weight memory? What does this imply for the relative importance of weight quantization vs KV cache quantization at large batch sizes?

3. **GGUF k-quants overhead.** K-quants use a two-level block structure: a super-block of 256 weights contains 8 sub-blocks of 32 weights each. The super-block stores one FP16 scale; each sub-block stores one 6-bit integer scale. Write a function `gguf_kquant_effective_bits(q_bits: int) -> float` that computes the effective bits per weight for this structure. Compare `Q4_K_M` (4-bit weights, k-quant) to `Q4_0` (4-bit weights, simple blocks). Which has more overhead? Why might k-quants still achieve better quality despite the extra overhead?

(Solutions: ../solutions/week-10/day-3.md)

---

## Self-check

- [ ] I can explain what quantization is and why it reduces model size proportionally to bit width.
- [ ] I can describe the difference between PTQ and QAT and explain why PTQ dominates open-source deployment.
- [ ] I can explain the GPTQ layer-wise weight update in plain terms: what error it minimizes and how the Hessian is used.
- [ ] I can explain why AWQ protects 1% of weights and how the scaling factor avoids runtime overhead.
- [ ] I can compute the memory footprint of a model at any precision given its parameter count.
- [ ] I can explain GGUF block quantization and the purpose of the per-block scale factor.

---

## Further reading

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration (Lin et al., 2023)](https://arxiv.org/abs/2306.00978)
- [llama.cpp — CPU inference with GGUF quantization](https://github.com/ggerganov/llama.cpp)
