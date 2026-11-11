# Phase 3 · Week 06 · Day 3 — Transformer Blocks: Causal Self-Attention + FFN

> Objective: Assemble a full decoder-only transformer block: multi-head causal self-attention, feed-forward sub-layer, residual connections, and layer norm.  ·  Time: ~1 hr  ·  Prereqs: [[week-05]], [[week-06/day-2]]

---

## Concept

A single transformer block has two sub-layers, each wrapped in a **residual connection** and **layer normalisation**:

```
Block(x):
    x = x + CausalSelfAttention(LayerNorm(x))   # pre-LN style
    x = x + FeedForward(LayerNorm(x))
    return x
```

Stack `N` of these blocks and you have the transformer body.

| Sub-layer | Role |
|---|---|
| **Multi-head causal self-attention** | Tokens gather context from previous tokens |
| **Position-wise feed-forward (FFN)** | Non-linear transformation per token independently |
| **Layer norm** | Stabilises training; applied *before* each sub-layer (pre-LN) |
| **Residual connection** | Gradient highway; enables deep stacks |

### Pre-LN vs Post-LN

Original paper uses Post-LN (`LN(x + sub(x))`).  
GPT-2 and nanoGPT use **Pre-LN** (`x + sub(LN(x))`).  
Pre-LN trains more stably without warmup; use it by default.

---

## How It Works Internally

### Causal self-attention (recap from Week 5)

Query, Key, Value projections followed by scaled dot-product attention with a **causal mask** — upper-triangular positions set to `-inf` before softmax, so position `t` can only attend to positions `0..t`.

For `H` heads each of dimension `d_head = d_model // H`:

```
Q = x W_Q   K = x W_K   V = x W_V        # (B, T, d_model) each
# split into H heads → (B, H, T, d_head)
scores = Q @ K.T / sqrt(d_head)           # (B, H, T, T)
scores = masked_fill(scores, mask, -inf)
attn   = softmax(scores)                  # (B, H, T, T)
out    = attn @ V                         # (B, H, T, d_head)
# merge heads → (B, T, d_model)
out    = out W_O
```

### Feed-forward sub-layer

Two linear layers with GELU in between, expanding to `4 * d_model` hidden units:

```
FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
```

Shapes: `(B, T, d_model) → (B, T, 4*d_model) → (B, T, d_model)`.

---

## Worked Example

```python
"""
day3_transformer_block.py
Full decoder-only transformer block for miniGPT.
Run: python day3_transformer_block.py
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny config
# ---------------------------------------------------------------------------
D_MODEL     = 64
N_HEADS     = 4
CONTEXT_LEN = 32
DROPOUT     = 0.1
BATCH_SIZE  = 2


# ---------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal (decoder) self-attention."""

    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.d_model  = d_model

        # Combined QKV projection
        self.c_attn   = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj   = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Causal mask — upper-triangular ones, registered as buffer
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("mask", mask)  # (T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Project to Q, K, V
        qkv = self.c_attn(x)                      # (B, T, 3D)
        q, k, v = qkv.split(self.d_model, dim=2)  # each (B, T, D)

        # Reshape to (B, H, T, d_head)
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        scale  = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale   # (B, H, T, T)

        # Apply causal mask
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_drop(attn)

        # Weighted sum of values
        out = attn @ v                                        # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, D)
        return self.resid_drop(self.c_proj(out))


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear, 4× expansion."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block (Pre-LN style)
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """One decoder-only transformer block with pre-LN and residuals."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # residual + attention
        x = x + self.ffn(self.ln2(x))    # residual + FFN
        return x


# ---------------------------------------------------------------------------
# Shape and gradient smoke tests
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(0)

    x = torch.randn(BATCH_SIZE, CONTEXT_LEN, D_MODEL, requires_grad=True)

    # Single block
    block = TransformerBlock(D_MODEL, N_HEADS, CONTEXT_LEN, DROPOUT)
    out   = block(x)
    assert out.shape == (BATCH_SIZE, CONTEXT_LEN, D_MODEL), out.shape
    log.info("TransformerBlock output: %s", out.shape)

    # Backward pass
    loss = out.mean()
    loss.backward()
    assert x.grad is not None, "No gradient on input"
    log.info("Backward pass OK — x.grad shape: %s", x.grad.shape)

    # Stack N blocks
    N_LAYERS = 2
    blocks = nn.Sequential(*[
        TransformerBlock(D_MODEL, N_HEADS, CONTEXT_LEN, DROPOUT)
        for _ in range(N_LAYERS)
    ])
    stacked_out = blocks(x.detach())
    assert stacked_out.shape == (BATCH_SIZE, CONTEXT_LEN, D_MODEL)
    log.info("Stacked %d blocks output: %s", N_LAYERS, stacked_out.shape)

    # Parameter count per block
    n_params = sum(p.numel() for p in block.parameters())
    log.info("Params per block: %d", n_params)
    # attn: 3*D*D (QKV) + D*D (proj) = 4*D² = 4*64²=16384
    # ffn: D*4D + 4D*D = 8*D² = 32768
    # LN: 2*D each × 2 = 4*D = 256
    # total ≈ 49408

    # Verify causal mask: later positions must not attend to future tokens
    # Run attention separately, freeze weights, check that attn weights are 0 above diag
    block.eval()
    with torch.no_grad():
        # We can't easily inspect the internal attn weights without refactoring,
        # but we verify causality indirectly: changing token at pos T-1
        # should NOT change output at pos 0.
        x2 = x.detach().clone()
        x2[0, -1] = x2[0, -1] * 0.0   # zero out last token
        out1 = block(x.detach())
        out2 = block(x2)
        # Position 0 output should be identical
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), \
            "Causality violation: pos 0 sees future!"
        log.info("Causal mask verified: changing last token does NOT affect pos 0")


if __name__ == "__main__":
    main()
```

Run with: `python day3_transformer_block.py`

### Block parameter count at `d_model=64, n_heads=4`

| Component | Formula | Count |
|---|---|---|
| QKV proj | `3 × D × D` | 12 288 |
| Output proj | `D × D` | 4 096 |
| FFN W1 | `D × 4D` | 16 384 |
| FFN W2 | `4D × D` | 16 384 |
| LayerNorm ×2 | `2 × 2D` | 256 |
| **Total** | | **49 408** |

---

## Exercises

1. Add an `attn_weights` return option to `CausalSelfAttention` (a `return_attn: bool = False` parameter). When `True`, return `(output, attn)` where `attn` is the averaged attention weights shape `(B, T, T)`. Write a test verifying that the upper triangle of the averaged attention matrix is zero (causality check).
2. Replace the combined `c_attn` (`3D` projection) with three separate `nn.Linear` layers for Q, K, V. Verify the output shape is unchanged.
3. Implement a `count_params(module: nn.Module) -> dict[str, int]` function that returns a dictionary mapping each named sub-module to its parameter count. Call it on a `TransformerBlock`.
4. Change the `FeedForward` activation from `GELU` to `SwiGLU` — or explain in 3 sentences why SwiGLU has become preferred in modern LLMs (Llama, Mistral) and what shape change it introduces.
5. Write a pytest test that instantiates a `TransformerBlock`, runs a forward pass, calls `loss.backward()`, and asserts that every parameter has a non-None `.grad`.

(Solutions: ../solutions/week-06/day-3.md)

---

## Self-check

- [ ] I can draw the data flow through a transformer block from memory
- [ ] I can explain pre-LN vs post-LN and why pre-LN trains more stably
- [ ] I can implement `CausalSelfAttention` from scratch given a blank file
- [ ] I can state the FFN expansion ratio and why `4×` is conventional

---

## Supplementary

- [Attention Is All You Need — Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)
- [nanoGPT model.py — `Block` class](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [GELU paper — Hendrycks & Gimpel 2016](https://arxiv.org/abs/1606.08415)
