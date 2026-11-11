# Solutions — Day 6: Saturday Deep Practice

## Reference implementation: `attention_lib.py`

```python
"""
attention_lib.py — complete reference implementation.
This is the answer to the Day 6 closed-book build session.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.
    q: (..., n, d_k), k: (..., m, d_k), v: (..., m, d_v)
    Returns (output (..., n, d_v), weights (..., n, m))
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = query.size(0)
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))
        attn_out, weights = scaled_dot_product_attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out), weights


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T, :]  # type: ignore[index]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, mask)[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )
```

---

## Reference test suite: `tests/test_attention_lib.py`

```python
"""tests/test_attention_lib.py — reference test suite for Day 6."""
import math
import torch
import torch.nn as nn
import pytest

from attention_lib import (
    scaled_dot_product_attention,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    FeedForward,
    TransformerBlock,
    causal_mask,
)


class TestSdpa:
    def test_sdpa_output_shape(self):
        B, n, m, d_k, d_v = 3, 5, 7, 16, 32
        q = torch.randn(B, n, d_k)
        k = torch.randn(B, m, d_k)
        v = torch.randn(B, m, d_v)
        out, w = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, n, d_v)
        assert w.shape == (B, n, m)

    def test_sdpa_weights_sum_to_one(self):
        B, T, d = 2, 8, 16
        q = k = v = torch.randn(B, T, d)
        _, w = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(w.sum(-1), torch.ones(B, T), atol=1e-5)

    def test_sdpa_causal_mask_zeros_upper_triangle(self):
        T, d = 6, 8
        q = k = v = torch.randn(1, T, d)
        mask = causal_mask(T).unsqueeze(0)
        _, w = scaled_dot_product_attention(q, k, v, mask)
        for i in range(T):
            for j in range(i + 1, T):
                assert w[0, i, j].item() < 1e-6


class TestMha:
    def test_mha_output_shape(self):
        B, T, d, h = 2, 10, 64, 8
        mha = MultiHeadAttention(d, h)
        x = torch.randn(B, T, d)
        out, w = mha(x, x, x)
        assert out.shape == (B, T, d)
        assert w.shape == (B, h, T, T)

    def test_mha_weights_shape(self):
        B, n, m, d, h = 2, 4, 7, 32, 4
        mha = MultiHeadAttention(d, h)
        q = torch.randn(B, n, d)
        kv = torch.randn(B, m, d)
        _, w = mha(q, kv, kv)
        assert w.shape == (B, h, n, m)

    def test_mha_cross_attention_shape(self):
        B, n, m, d, h = 1, 3, 9, 64, 4
        mha = MultiHeadAttention(d, h)
        out, _ = mha(torch.randn(B, n, d), torch.randn(B, m, d), torch.randn(B, m, d))
        assert out.shape == (B, n, d)


class TestCausalMask:
    def test_causal_mask_shape(self):
        T = 8
        mask = causal_mask(T)
        assert mask.shape == (T, T)
        assert mask.dtype == torch.bool

    def test_causal_mask_diagonal_not_masked(self):
        mask = causal_mask(5)
        for i in range(5):
            assert not mask[i, i].item(), f"Diagonal position {i} should not be masked"

    def test_causal_mask_upper_triangle_masked(self):
        mask = causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j].item(), f"Position ({i},{j}) should be masked"


class TestSinusoidalPE:
    def test_sinusoidal_pe_shape(self):
        B, T, d = 2, 20, 64
        spe = SinusoidalPositionalEncoding(d)
        out = spe(torch.randn(B, T, d))
        assert out.shape == (B, T, d)

    def test_sinusoidal_pe_deterministic(self):
        d = 32
        spe1 = SinusoidalPositionalEncoding(d)
        spe2 = SinusoidalPositionalEncoding(d)
        assert torch.allclose(spe1.pe, spe2.pe)  # type: ignore[attr-defined]

    def test_sinusoidal_pe_all_positions_unique(self):
        spe = SinusoidalPositionalEncoding(d_model=64, max_len=50)
        pe = spe.pe.squeeze(0)  # (50, 64) type: ignore[attr-defined]
        dists = torch.cdist(pe, pe)
        dists.fill_diagonal_(float("inf"))
        assert dists.min().item() > 1e-4, "Some PE positions are identical"


class TestTransformerBlock:
    def test_transformer_block_shape(self):
        B, T, d = 2, 8, 64
        block = TransformerBlock(d_model=d, num_heads=4, d_ff=256)
        x = torch.randn(B, T, d)
        out = block(x)
        assert out.shape == (B, T, d)

    def test_backward_through_block(self):
        B, T, d = 1, 4, 32
        block = TransformerBlock(d_model=d, num_heads=4, d_ff=128)
        x = torch.randn(B, T, d)
        loss = block(x).sum()
        loss.backward()
        for name, p in block.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
```

---

## Memory scaling verification (Exercise 4)

```python
import tracemalloc
import torch
from attention_lib import scaled_dot_product_attention


def measure_attention_memory(T: int, d: int = 64) -> int:
    """Return peak memory in bytes for one attention call."""
    tracemalloc.start()
    q = k = v = torch.randn(1, T, d)
    _ = scaled_dot_product_attention(q, k, v)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


for T in [16, 32, 64, 128, 256]:
    mem = measure_attention_memory(T)
    print(f"T={T:4d}: ~{mem/1024:.1f} KB")
```

Expected output shows memory roughly quadruples when T doubles (O(T²) from the score matrix).
