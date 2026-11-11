# Phase 3 · Week 06 · Day 2 — Token Embeddings & Positional Encoding

> Objective: Implement learnable token embeddings and sinusoidal / learned positional embeddings, understand their shapes and why both are needed.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]], `nn.Embedding`

---

## Concept

A transformer has no notion of order — its attention operation is a set function. We inject two signals into every token's representation before the first block:

| Signal | What it encodes | Implementation |
|---|---|---|
| **Token embedding** | *What* the token is (identity) | `nn.Embedding(vocab_size, d_model)` |
| **Positional embedding** | *Where* the token sits (position 0…T-1) | sinusoidal formula **or** `nn.Embedding(context_length, d_model)` |

The two embeddings are **added** element-wise → shape `(B, T, d_model)`.

### Why add instead of concatenate?

Concatenation doubles dimensionality. Addition keeps the same dimension, so both sources of information travel through every attention head and feed-forward layer simultaneously. Empirically, addition works at least as well.

---

## How It Works Internally

### Token embedding lookup

`nn.Embedding(V, D)` is a weight matrix `E` of shape `(V, D)`.  
A forward pass with index tensor `(B, T)` performs a gather: for each integer `i` in the input, copy row `E[i]`. Gradient flows back via scatter-add — every token's embedding row accumulates gradients from all positions where that token appeared.

### Sinusoidal positional encoding (original Transformer, Vaswani 2017)

For position `pos` and dimension `i`:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
```

- **Not learned** — fixed at init time.
- Different frequencies at each pair of dimensions: low dimensions → high frequency (nearby positions differ a lot), high dimensions → low frequency (long-range).
- Advantage: can extrapolate beyond the training context length.
- GPT-2 and nanoGPT use **learned** positional embeddings instead (simpler; works well for fixed context lengths).

---

## Worked Example

```python
"""
day2_embeddings.py
Token embeddings + positional embeddings for miniGPT.
Run: python day2_embeddings.py
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny config — CPU-friendly
# ---------------------------------------------------------------------------
VOCAB_SIZE    = 65    # typical char-level shakespeare
D_MODEL       = 64    # embedding dimension
CONTEXT_LEN   = 32   # max sequence length
BATCH_SIZE    = 4


# ---------------------------------------------------------------------------
# Token embedding
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) int64  →  out: (B, T, d_model)
        return self.embed(idx)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding (fixed, not learned)
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, context_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(context_len, d_model)           # (T, D)
        pos = torch.arange(context_len).unsqueeze(1)      # (T, 1)
        # div_term: shape (D/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)          # even dims
        pe[:, 1::2] = torch.cos(pos * div_term)          # odd dims
        # Register as buffer — moves with .to(device), not a learned param
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — add positional signal for first T positions
        T = x.size(1)
        return x + self.pe[:T]   # broadcasts over batch


# ---------------------------------------------------------------------------
# Learned positional embedding (GPT-style)
# ---------------------------------------------------------------------------
class LearnedPE(nn.Module):
    def __init__(self, context_len: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(context_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        pos = torch.arange(T, device=x.device)  # (T,)
        return x + self.embed(pos)               # (T, D) broadcast over B


# ---------------------------------------------------------------------------
# Combined input embedding (token + position)
# ---------------------------------------------------------------------------
class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        learned_pos: bool = True,
    ) -> None:
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb   = (
            LearnedPE(context_len, d_model)
            if learned_pos
            else SinusoidalPE(context_len, d_model)
        )
        self.drop = nn.Dropout(p=0.1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) int64  →  (B, T, d_model)
        tok = self.token_emb(idx)   # (B, T, D)
        return self.drop(self.pos_emb(tok))


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(42)

    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LEN))

    # --- Token embedding ---
    te = TokenEmbedding(VOCAB_SIZE, D_MODEL)
    tok_out = te(idx)
    assert tok_out.shape == (BATCH_SIZE, CONTEXT_LEN, D_MODEL), tok_out.shape
    log.info("Token embed output: %s", tok_out.shape)

    # --- Sinusoidal PE ---
    spe = SinusoidalPE(CONTEXT_LEN, D_MODEL)
    spe_out = spe(tok_out)
    assert spe_out.shape == (BATCH_SIZE, CONTEXT_LEN, D_MODEL)
    log.info("Sinusoidal PE output: %s", spe_out.shape)

    # --- Learned PE ---
    lpe = LearnedPE(CONTEXT_LEN, D_MODEL)
    lpe_out = lpe(tok_out)
    assert lpe_out.shape == (BATCH_SIZE, CONTEXT_LEN, D_MODEL)
    log.info("Learned PE output: %s", lpe_out.shape)

    # --- Full input embedding ---
    emb = InputEmbedding(VOCAB_SIZE, CONTEXT_LEN, D_MODEL, learned_pos=True)
    out = emb(idx)
    assert out.shape == (BATCH_SIZE, CONTEXT_LEN, D_MODEL)
    log.info("InputEmbedding output: %s", out.shape)

    # --- Parameter count ---
    n_params = sum(p.numel() for p in emb.parameters())
    log.info("InputEmbedding param count: %d", n_params)
    # tok_emb: vocab_size * d_model = 65*64 = 4160
    # pos_emb: context_len * d_model = 32*64 = 2048
    # total: 6208
    expected = VOCAB_SIZE * D_MODEL + CONTEXT_LEN * D_MODEL
    assert n_params == expected, f"{n_params} != {expected}"
    log.info("Param count verified: %d", n_params)

    # --- Verify positional signal differs across positions ---
    single = emb(idx[:1, :4])   # (1, 4, D)
    # Different positions should yield different vectors
    assert not torch.allclose(single[0, 0], single[0, 1]), \
        "Position 0 and 1 should differ"
    log.info("Positional uniqueness verified")


if __name__ == "__main__":
    main()
```

Run with: `python day2_embeddings.py`

### Shape summary

```
idx          (B, T)       int64   — token indices
tok_emb(idx) (B, T, D)   float32 — identity signal
pos_emb(x)   (B, T, D)   float32 — position signal (added in-place)
output       (B, T, D)   float32 — combined representation
```

---

## Exercises

1. Count the total parameters in an `InputEmbedding` with `vocab_size=65`, `context_len=128`, `d_model=128` using `learned_pos=True`. Show the breakdown by component.
2. Implement a `RotaryPE` stub that accepts `(B, T, D)` and returns the same shape — leave the rotation logic as `raise NotImplementedError` with a docstring explaining RoPE in 2 sentences.
3. Write a pytest test `test_token_embed_lookup` that creates a `TokenEmbedding(10, 8)`, manually sets weight row 3 to all-ones, and verifies that forward pass on `torch.tensor([[3]])` returns a `(1, 1, 8)` tensor of all-ones.
4. Modify `InputEmbedding` to scale token embeddings by `math.sqrt(d_model)` (the scaling from Vaswani et al.) before adding positional embeddings. Add a `scale: bool = False` parameter.
5. Explain in two sentences why sinusoidal PE can (in principle) handle sequences longer than the training context length, while learned PE cannot.

(Solutions: ../solutions/week-06/day-2.md)

---

## Self-check

- [ ] I can explain why both token and positional embeddings are needed
- [ ] I can state the shape flow: `(B, T)` → `(B, T, D)` through `InputEmbedding`
- [ ] I understand the difference between `nn.Embedding` and sinusoidal PE
- [ ] I can count parameters in an `InputEmbedding` module

---

## Supplementary

- [Attention Is All You Need — Vaswani et al. 2017 (positional encoding, §3.5)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [nanoGPT model.py — learned positional embedding lines](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [RoPE paper — Su et al. 2021](https://arxiv.org/abs/2104.09864)
