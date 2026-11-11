# Solutions — Assessment Week 05

## Section 1 — Conceptual answers

**Q1.** Three RNN limitations:
1. **Vanishing gradients** — BPTT multiplies Jacobians across steps; if they are < 1, gradients shrink exponentially, making early-token learning impossible.
2. **Information bottleneck** — entire history must fit in a fixed-size hidden vector `h_t`. For long sequences, early tokens are increasingly overwritten.
3. **No parallelism** — `h_t = f(h_{t-1}, x_t)` is inherently sequential; each step depends on the previous, so GPUs cannot parallelise across the time dimension.

**Q2.** Without √d_k, dot products of d-dimensional standard-normal vectors have variance d. Large variance → large softmax inputs → softmax saturation → near-one-hot distribution → tiny gradients. Dividing by √d_k restores variance to 1, keeping softmax in a gradient-friendly regime during early training.

**Q3.** Separate projection matrices allow each head to specialise: head i learns to attend for one relationship type (e.g. syntactic), head j for another (e.g. co-reference). A shared projection forces all heads to extract the same subspace — they would collapse to computing the same attention pattern, providing no diversity. The output W^O then has nothing to combine.

**Q4.** Permutation-invariance: if you permute the rows of Q, K, V by any permutation P, the output rows are permuted by the same P. Concretely: swapping token 3 and token 7 in the input produces an output with tokens 3 and 7 swapped, but otherwise identical content. This matters because "dog bites man" and "man bites dog" are different sentences but contain the same tokens — without positional encoding, a transformer cannot distinguish them.

**Q5.** `PE(t, 2i) = sin(t / 10000^(2i/d))`, `PE(t, 2i+1) = cos(t / 10000^(2i/d))`. The 10000 base spaces frequencies logarithmically, giving usable encodings up to ~10000 positions. Trade-off: sinusoidal generalises to unseen lengths and needs no parameters; learned encodings can fit training distribution better but have a hard `max_seq_len` ceiling.

**Q6.** Without residuals, gradients in a 50-layer network must pass through 50 non-linear transformations. Each multiplication by a Jacobian < 1 shrinks the gradient; by layer 1, it is effectively zero. Residual connections `x + f(x)` provide a direct gradient highway: the gradient of the loss with respect to any layer's input is the sum of the residual path gradient (always 1) plus the sublayer gradient. Training remains stable regardless of depth.

**Q7.** Causal mask for T=4 (True = blocked):
```
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
```
`mask[2, 2] = False` — not masked. Position 2 must attend to itself because its own representation is part of its context. The causal constraint is "no attending to positions strictly in the future", i.e., j > i. The diagonal (j == i) is the present, not the future.

**Q8.**
- **Encoder-only** (BERT): no causal mask, full bidirectional context. Best for classification/NER/embeddings — the model sees the complete input at once.
- **Decoder-only** (GPT): causal mask, autoregressive. Best for generation — predicts tokens left-to-right.
- **Encoder-decoder** (T5): encoder builds a full context representation; decoder attends to it via cross-attention while generating autoregressively. Best for seq2seq tasks like translation or summarisation where input and output are separate sequences.

---

## Section 2 — Coding challenge answers

### Task A — Scaled dot-product attention

```python
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights
```

```python
# Tests
def test_output_shape():
    B, n, m, d_k, d_v = 2, 4, 6, 16, 32
    q = torch.randn(B, n, d_k)
    k = torch.randn(B, m, d_k)
    v = torch.randn(B, m, d_v)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, n, d_v)
    assert w.shape == (B, n, m)


def test_causal_mask_upper_triangle_zero():
    T, d = 5, 8
    q = k = v = torch.randn(1, T, d)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    _, w = scaled_dot_product_attention(q, k, v, mask.unsqueeze(0))
    for i in range(T):
        for j in range(i + 1, T):
            assert w[0, i, j].item() < 1e-6
```

### Task B — TransformerBlock

```python
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# Test
def test_block_shape():
    B, T, d = 2, 8, 64
    block = TransformerBlock(d_model=d, num_heads=4, d_ff=256)
    x = torch.randn(B, T, d)
    out = block(x)
    assert out.shape == (B, T, d)
```

---

## Section 3 — Debugging answers

**Bug 1:** Missing `.transpose(-2, -1)` on k.
```python
# Wrong:
scores = torch.matmul(q, k) / math.sqrt(d_k)
# Fixed:
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
```

**Bug 2:** `self.num_heads` is never stored. The `__init__` method computes `self.d_k` but never does `self.num_heads = num_heads`.
```python
def __init__(self, d_model, num_heads):
    super().__init__()
    self.num_heads = num_heads  # MISSING — add this line
    self.d_k = d_model // num_heads
    ...
```

**Bug 3:** `self.pe` is assigned as a plain attribute, not a buffer. It won't be moved to the correct device when calling `.to(device)`, and won't appear in `state_dict`. Fix: `self.register_buffer("pe", pe.unsqueeze(0))`.

**Bug 4:** `diagonal=0` masks the diagonal in addition to the upper triangle. Causal mask should use `diagonal=1` to keep `mask[i,i] = False` (present position should not be blocked).
```python
# Wrong: diagonal=0 masks i==j too
return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
# Fixed:
return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
```

---

## Section 4 — Code quality issues

```python
def attn(q, k, v, m=None):  # Issues below
```

| Issue | Location | Fix |
|---|---|---|
| No type hints | function signature | Add `q: torch.Tensor`, etc. |
| Single-letter variable names `q, k, v, m, x, s, w, o` | throughout | Use `query, key, value, mask, d_k, scores, weights, output` |
| `m != None` uses `!=` with None | line 5 | Use `if m is not None:` |
| `m == True` inside `masked_fill` is redundant | line 6 | `masked_fill(m, ...)` — bool tensor already |
| `softmax(s, -1)` positional dim arg | line 7 | Use keyword: `F.softmax(scores, dim=-1)` |
| No docstring | function | Add Args/Returns docstring |
| Name `attn` too generic | function name | `scaled_dot_product_attention` |
| No logging, uses no visible output | — | Add `logging` import and debug log for shape |
