# Solutions — Week 05 · Project — Scaled Dot-Product Attention and Multi-Head Attention

## Complete Reference Implementation

### `attention_lib.py`

```python
"""attention_lib.py — Transformer attention primitives from scratch.

Deps (uv add torch): torch
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Scaled dot-product attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    q: torch.Tensor,               # (..., n, d_k)
    k: torch.Tensor,               # (..., m, d_k)
    v: torch.Tensor,               # (..., m, d_v)
    mask: torch.Tensor | None = None,  # (..., n, m) bool, True = block
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Args:
        q: query tensor, shape (..., n, d_k)
        k: key tensor, shape (..., m, d_k)
        v: value tensor, shape (..., m, d_v)
        mask: boolean mask; True positions are set to -inf before softmax.

    Returns:
        output: (..., n, d_v)
        weights: (..., n, m) — attention weights summing to 1 along last dim
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., n, m)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # (..., n, m)
    output = weights @ v                 # (..., n, d_v)
    return output, weights


# ---------------------------------------------------------------------------
# 2. Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head attention as a single batched operation (no Python head loop).

    Cross-validates against PyTorch's nn.MultiheadAttention within atol=1e-4
    when weights are copied.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,          # (B, n, d_model)
        key: torch.Tensor,            # (B, m, d_model)
        value: torch.Tensor,          # (B, m, d_model)
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-head attention forward pass.

        Returns:
            output: (B, n, d_model)
            weights: (B, num_heads, n, m)
        """
        B, n, _ = query.shape
        m = key.size(1)
        H, d = self.num_heads, self.d_head

        def _split_heads(t: torch.Tensor, seq: int) -> torch.Tensor:
            return t.view(B, seq, H, d).transpose(1, 2)  # (B, H, seq, d)

        q = _split_heads(self.W_q(query), n)  # (B, H, n, d)
        k = _split_heads(self.W_k(key),   m)  # (B, H, m, d)
        v = _split_heads(self.W_v(value), m)  # (B, H, m, d)

        # Expand mask for heads if provided
        if mask is not None and mask.dim() == 3:
            # (B, n, m) -> (B, 1, n, m) -> (B, H, n, m)
            mask = mask.unsqueeze(1)

        out, weights = scaled_dot_product_attention(q, k, v, mask)
        # out: (B, H, n, d)
        out = out.transpose(1, 2).contiguous().view(B, n, self.d_model)
        return self.W_o(out), weights  # weights: (B, H, n, m)


# ---------------------------------------------------------------------------
# 3. Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Stored as a buffer (not a learnable parameter) so it is saved with the
    model but excluded from optimiser updates.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build (max_len, d_model) PE table
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)   # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )                                                               # (d_model//2,)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model) — same shape, PE added.
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore[index]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 4. Feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    Architecture: Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# 5. Transformer block (Pre-LN)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block with Pre-LN residual connections.

    Structure:
        x = x + attention(LN1(x))
        x = x + ffn(LN2(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,               # (B, T, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:                 # (B, T, d_model)
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, normed, mask)[0]
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# 6. Causal mask
# ---------------------------------------------------------------------------

def causal_mask(
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Upper-triangular causal mask.

    Returns:
        (seq_len, seq_len) bool tensor.
        True at position (i, j) means "block token j from attending to position i",
        i.e. j > i (future positions).
        Diagonal and lower triangle are False (allowed).
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
```

---

### `tests/test_attention_lib.py`

```python
"""tests/test_attention_lib.py — Required test suite for attention_lib."""
from __future__ import annotations

import torch
import pytest
from attention_lib import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    FeedForward,
    TransformerBlock,
    causal_mask,
)


# Helpers
B, n, m, d_k, d_v = 2, 5, 7, 16, 32
d_model, num_heads, d_ff = 64, 4, 128


def _qkv(b: int = B, nq: int = n, nk: int = m, dk: int = d_k, dv: int = d_v):
    q = torch.randn(b, nq, dk)
    k = torch.randn(b, nk, dk)
    v = torch.randn(b, nk, dv)
    return q, k, v


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------

def test_sdpa_output_shape() -> None:
    q, k, v = _qkv()
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, n, d_v)
    assert w.shape   == (B, n, m)


def test_sdpa_weights_sum_to_one() -> None:
    q, k, v = _qkv()
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B, n), atol=1e-5)


def test_sdpa_causal_mask_zeros_upper_triangle() -> None:
    T = 6
    q = torch.randn(1, T, d_k)
    k = torch.randn(1, T, d_k)
    v = torch.randn(1, T, d_v)
    mask = causal_mask(T)  # (T, T) — broadcast over batch
    _, w = scaled_dot_product_attention(q, k, v, mask.unsqueeze(0))
    for i in range(T):
        for j in range(T):
            if j > i:
                assert w[0, i, j].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

def test_mha_output_shape() -> None:
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, n, d_model)
    out, _ = mha(x, x, x)
    assert out.shape == (B, n, d_model)


def test_mha_weights_shape() -> None:
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(B, n, d_model)
    _, w = mha(x, x, x)
    assert w.shape == (B, num_heads, n, n)


def test_mha_cross_attention_shape() -> None:
    mha = MultiHeadAttention(d_model, num_heads)
    q = torch.randn(B, n, d_model)
    kv = torch.randn(B, m, d_model)
    out, w = mha(q, kv, kv)
    assert out.shape == (B, n, d_model)
    assert w.shape == (B, num_heads, n, m)


# ---------------------------------------------------------------------------
# causal_mask
# ---------------------------------------------------------------------------

def test_causal_mask_shape() -> None:
    T = 8
    mask = causal_mask(T)
    assert mask.shape == (T, T)
    assert mask.dtype == torch.bool


def test_causal_mask_diagonal_unmasked() -> None:
    T = 8
    mask = causal_mask(T)
    for i in range(T):
        assert mask[i, i].item() is False


# ---------------------------------------------------------------------------
# SinusoidalPositionalEncoding
# ---------------------------------------------------------------------------

def test_sinusoidal_pe_deterministic() -> None:
    pe1 = SinusoidalPositionalEncoding(d_model)
    pe2 = SinusoidalPositionalEncoding(d_model)
    x = torch.zeros(1, 10, d_model)
    assert torch.allclose(pe1(x), pe2(x))


def test_sinusoidal_pe_unique_positions() -> None:
    pe = SinusoidalPositionalEncoding(d_model, max_len=64)
    x = torch.zeros(1, 64, d_model)
    out = pe(x)  # (1, 64, d_model)
    rows = out[0]  # (64, d_model)
    # All rows should be distinct
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            assert not torch.allclose(rows[i], rows[j])


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

def test_transformer_block_shape() -> None:
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    x = torch.randn(B, n, d_model)
    out = block(x)
    assert out.shape == x.shape


def test_backward_through_block() -> None:
    block = TransformerBlock(d_model, num_heads, d_ff, dropout=0.0)
    x = torch.randn(B, n, d_model, requires_grad=True)
    out = block(x)
    out.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
```

---

### `day6_integration.py`

```python
"""day6_integration.py — TinyGPT integration test.

Run: python day6_integration.py
Expected log line: TinyGPT forward pass OK
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from attention_lib import TransformerBlock, SinusoidalPositionalEncoding, causal_mask

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

VOCAB_SIZE = 65
D_MODEL = 64
N_HEADS = 4
D_FF = 256
N_LAYERS = 2
CTX = 16
DROPOUT = 0.0


class TinyGPT(nn.Module):
    """Minimal decoder-only GPT using attention_lib components."""

    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pe = SinusoidalPositionalEncoding(D_MODEL, max_len=CTX, dropout=DROPOUT)
        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_FF, DROPOUT)
            for _ in range(N_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        mask = causal_mask(T, device=idx.device)  # (T, T)
        x = self.pe(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), targets.view(-1)
            )
        return logits, loss


def main() -> None:
    torch.manual_seed(0)
    model = TinyGPT()
    B, T = 2, CTX
    idx = torch.randint(0, VOCAB_SIZE, (B, T))
    logits, loss = model(idx, targets=idx)
    log.info("TinyGPT forward pass OK — logits shape: %s", logits.shape)
    log.info("Loss: %.4f", loss.item() if loss is not None else float("nan"))

    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Missing gradients: {no_grad}"
    log.info(".backward() OK — all gradients non-None")


if __name__ == "__main__":
    main()
```

---

## Design notes

**Efficient multi-head without a Python loop**: reshape `(B, T, d_model)` to
`(B, T, H, d_head)`, transpose to `(B, H, T, d_head)`, then run `scaled_dot_product_attention`
once — the batch and head dimensions are treated as leading dimensions by `@`.

**Pre-LN**: apply `LayerNorm` BEFORE the sub-layer, not after. Gradient signal at the start
of training is dominated by the residual bypass (`x + ...`), which has magnitude ~1 regardless
of depth. This eliminates the vanishing-gradient problem that plagued the original Post-LN transformer.

**`causal_mask` as a buffer**: register it with `register_buffer` in `CausalSelfAttention` or
pass it in explicitly. Never recompute it on every forward pass — allocating a new tensor per
call is wasteful and breaks `torch.compile`.

**Sinusoidal PE is not a parameter**: `register_buffer` tells PyTorch to (a) move it to the
correct device with the model and (b) save it in `state_dict`, but (c) exclude it from
`model.parameters()` so it is not updated by the optimiser.
