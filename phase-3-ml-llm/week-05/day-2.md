# Phase 3 · Week 05 · Day 2 — Scaled Dot-Product Attention (Implement)

> Objective: Derive and implement scaled dot-product attention from first principles in PyTorch.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Queries, Keys, and Values

Attention generalises the lookup table: given a **query** `Q`, retrieve values `V` weighted by how much each **key** `K` matches the query.

Think of it as a soft dictionary lookup:
- Hard lookup: return `V[k]` exactly when `k == Q`.
- Soft lookup: return a weighted mix of all values, weights proportional to similarity between Q and each K.

In transformers every position plays all three roles simultaneously: it acts as a query (what am I looking for?), a key (what do I offer to others?), and a value (what information do I carry?).

### The formula

Given matrices:
- `Q ∈ ℝ^{n×d_k}` — n queries, each of dimension d_k
- `K ∈ ℝ^{m×d_k}` — m keys
- `V ∈ ℝ^{m×d_v}` — m values

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V
```

Step by step:
1. **Dot products** `Q Kᵀ`: shape `(n, m)`. Each entry `[i, j]` = similarity of query `i` to key `j`.
2. **Scale by √d_k**: prevents dot products from growing large (which would push softmax into saturation).
3. **Softmax row-wise**: turns similarities into a probability distribution over keys.
4. **Weighted sum of V**: each query gets a context vector — a mixture of values.

### Why √d_k?

With random init, dot products of d-dim vectors have variance `d`. Dividing by `√d` restores unit variance. Without scaling, softmax receives large inputs and produces near-one-hot outputs — gradients become tiny (softmax saturation).

Quick check: if `q, k ~ N(0, 1)` with dimension d, then `Var(q·k) = d`. Dividing by `√d`: variance → 1. ✓

## How it works internally

### Memory and compute

- `Q Kᵀ` is an `(n, m)` matrix → O(nm·d_k) FLOPS, O(nm) memory.
- For self-attention n = m = sequence length T: O(T²·d) FLOPS, O(T²) memory.
- This is the quadratic cost of attention — it dominates for long sequences.

### Softmax saturation

```
softmax([100, 1, 1]) ≈ [1.0, 0.0, 0.0]  # near one-hot, tiny gradient
softmax([1.0, 0.01, 0.01]) ≈ [0.34, 0.33, 0.33]  # spread out, larger gradient
```

Scaling keeps the logits in a range where softmax produces meaningful gradients early in training.

### Self-attention vs cross-attention

- **Self-attention**: Q, K, V all come from the same sequence. Each position attends to all other positions in the same sequence.
- **Cross-attention**: Q comes from one sequence (e.g. decoder), K and V come from another (e.g. encoder output).

## Worked example

```python
"""
day2_scaled_dot_product.py
Implements scaled dot-product attention from scratch.
Run: python day2_scaled_dot_product.py
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Args:
        q: Queries,  shape (..., n, d_k)
        k: Keys,     shape (..., m, d_k)
        v: Values,   shape (..., m, d_v)
        mask: Optional boolean mask, shape (..., n, m).
              True = position is MASKED (ignored).

    Returns:
        output:  shape (..., n, d_v)
        weights: shape (..., n, m)  — the attention distribution
    """
    d_k = q.size(-1)
    # Step 1: dot products → (..., n, m)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: apply mask (set masked positions to -inf before softmax)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Step 3: softmax over key dimension
    weights = F.softmax(scores, dim=-1)

    # Step 4: weighted sum of values
    output = torch.matmul(weights, v)

    return output, weights


def verify_shapes() -> None:
    """Shape tests — run these as a quick sanity check."""
    torch.manual_seed(0)
    batch, n, m, d_k, d_v = 2, 5, 7, 16, 32

    q = torch.randn(batch, n, d_k)
    k = torch.randn(batch, m, d_k)
    v = torch.randn(batch, m, d_v)

    out, w = scaled_dot_product_attention(q, k, v)

    assert out.shape == (batch, n, d_v), f"output shape wrong: {out.shape}"
    assert w.shape == (batch, n, m), f"weights shape wrong: {w.shape}"
    # Weights should sum to 1 over key dimension
    assert torch.allclose(w.sum(dim=-1), torch.ones(batch, n), atol=1e-5), \
        "weights don't sum to 1"

    log.info("Shape test PASSED: output %s, weights %s", out.shape, w.shape)


def demonstrate_scaling() -> None:
    """Show what happens with and without √d_k scaling."""
    torch.manual_seed(1)
    d = 64
    q = torch.randn(1, 1, d)
    k = torch.randn(1, 10, d)
    v = torch.randn(1, 10, d)

    # Without scaling
    scores_raw = torch.matmul(q, k.transpose(-2, -1))
    w_raw = F.softmax(scores_raw, dim=-1)

    # With scaling
    scores_scaled = scores_raw / math.sqrt(d)
    w_scaled = F.softmax(scores_scaled, dim=-1)

    log.info("Unscaled scores std:  %.3f", scores_raw.std().item())
    log.info("Scaled   scores std:  %.3f", scores_scaled.std().item())
    log.info("Unscaled weights max: %.4f", w_raw.max().item())
    log.info("Scaled   weights max: %.4f", w_scaled.max().item())
    log.info(
        "Unscaled is more peaked (closer to one-hot) — worse gradients"
        if w_raw.max() > w_scaled.max() else "Unexpected: check dim"
    )


def causal_mask_demo() -> None:
    """Demonstrate masking to prevent attending to future positions."""
    seq_len = 4
    # Upper-triangular mask (excluding diagonal): future positions
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    log.info("\nCausal mask (True = blocked):\n%s", mask)

    torch.manual_seed(2)
    d = 8
    q = torch.randn(1, seq_len, d)
    k = torch.randn(1, seq_len, d)
    v = torch.randn(1, seq_len, d)

    _, w = scaled_dot_product_attention(q, k, v, mask=mask.unsqueeze(0))
    log.info("\nAttention weights (should be 0 above diagonal):\n%s",
             w.squeeze(0).detach().round(decimals=3))


if __name__ == "__main__":
    verify_shapes()
    log.info("\n--- Scaling Effect ---")
    demonstrate_scaling()
    log.info("\n--- Causal Masking ---")
    causal_mask_demo()
```

Expected output shows weights sum to 1, unscaled weights are more concentrated, and upper triangle of causal weights is 0.

## Exercises

1. Write a pytest test file `tests/test_attention.py` with at least three tests:
   - Output shape is correct for arbitrary (batch, n, m, d_k, d_v).
   - Weights sum to 1 along the key dimension.
   - With a causal mask of size T×T, `weights[b, i, j] == 0` for all j > i.

2. Implement a `temperature_attention` variant that takes a `temperature: float` parameter (default 1.0) replacing the √d_k divisor. What temperature makes the distribution uniform? What makes it near one-hot?

3. Prove (with a small numerical example, not just words): dividing by √d keeps variance ≈ 1 when q and k are standard normal.

4. What is the output of `scaled_dot_product_attention` when all keys are identical? What does that tell you about the expressivity of attention?

(Solutions: ../solutions/week-05/day-2.md)

## Self-check

- [ ] I can write the attention formula from memory including the scaling factor.
- [ ] I can explain why √d_k is the right divisor (variance argument).
- [ ] I can implement `scaled_dot_product_attention` without reference.
- [ ] I understand why masking works by setting logits to −∞ before softmax.

## Supplementary

- Vaswani et al. (2017) "Attention Is All You Need" — https://arxiv.org/abs/1706.03762 (Section 3.2)
- Illustrated Transformer — https://jalammar.github.io/illustrated-transformer/
- PyTorch `F.scaled_dot_product_attention` docs (compare with your implementation) — https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
