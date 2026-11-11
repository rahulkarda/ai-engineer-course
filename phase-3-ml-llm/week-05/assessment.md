# Week 05 Assessment — Transformer Architecture & Attention

**Time limit:** 60 minutes  ·  **Pass threshold:** 75%

Complete without opening lesson files or external references.

---

## Section 1 — Conceptual (25 points)

Answer each question concisely (2–5 sentences unless specified).

**Q1.** (3 pts) Name three specific limitations of vanilla RNNs for long-sequence modelling. For each, state what structural property of RNNs causes it.

**Q2.** (3 pts) Explain the purpose of dividing by √d_k in scaled dot-product attention. What happens to training if you omit the scaling? Be precise about the mechanism.

**Q3.** (4 pts) Multi-head attention uses separate projection matrices W_Q^i, W_K^i, W_V^i per head. Why not use a single shared projection? What would be lost?

**Q4.** (3 pts) Attention is permutation-invariant. Define what that means precisely (in terms of input permutations and output), and explain why it matters.

**Q5.** (3 pts) Describe the sinusoidal positional encoding formula. Why is the 10000 base chosen? What is the trade-off between sinusoidal and learned encodings?

**Q6.** (3 pts) In the transformer block, there are two residual connections. What failure mode do they prevent? Why would a 50-layer transformer without residuals be untrainable in practice?

**Q7.** (3 pts) A causal mask prevents attention to future positions. Describe the exact boolean matrix for T=4. What is `mask[2, 2]` (the diagonal) — True or False — and why?

**Q8.** (3 pts) Contrast encoder-only, decoder-only, and encoder-decoder transformers. For each, state one concrete task it is best suited for and why.

---

## Section 2 — Coding Challenge (40 points)

Implement the following **from memory**, with no external references. Type hints required. Use `logging`, not `print`.

### Task A (20 pts) — Scaled dot-product attention with masking

Implement `scaled_dot_product_attention(q, k, v, mask=None)` satisfying:

1. Input shapes: `q (..., n, d_k)`, `k (..., m, d_k)`, `v (..., m, d_v)`.
2. Output: `(output, weights)` where output is `(..., n, d_v)` and weights sum to 1 over the key dimension.
3. If `mask` is provided (bool tensor, True = blocked), those positions receive weight 0.
4. Include the correct scaling factor.

Also write two pytest tests: one verifying output shape, one verifying causal masking zeroes the upper triangle for T=5.

### Task B (20 pts) — Full transformer block

Implement `TransformerBlock(nn.Module)` using:
- Your `scaled_dot_product_attention` from Task A (or `MultiHeadAttention` if you prefer).
- A feed-forward network: `Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)`.
- Pre-LN: LayerNorm applied before each sub-layer, residual connection outside.
- `forward(x, mask=None) -> x` preserving shape `(B, T, d_model)`.

Write one test: input and output shape match for `(B=2, T=8, d_model=64)`.

---

## Section 3 — Debugging (20 points)

Each snippet contains exactly one bug. Identify and fix it.

**Bug 1** (5 pts):
```python
def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

**Bug 2** (5 pts):
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
```

**Bug 3** (5 pts):
```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```

**Bug 4** (5 pts):
```python
def causal_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
```

---

## Section 4 — Code Quality (15 points)

Review this implementation and list all code quality issues (not bugs). Suggest fixes for each.

```python
import torch
import torch.nn.functional as F
import math

def attn(q, k, v, m=None):
    x = q.size(-1)
    s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x)
    if m != None:
        s = s.masked_fill(m == True, float("-inf"))
    w = F.softmax(s, -1)
    o = torch.matmul(w, v)
    return o, w
```

---

## Rubric

| Section | Points | Weight |
|---|---|---|
| Conceptual (Q1–Q8) | 25 | 25% |
| Coding challenge (A + B) | 40 | 40% |
| Debugging (4 bugs) | 20 | 20% |
| Code quality | 15 | 15% |
| **Total** | **100** | |

Pass: ≥ 75 / 100.

Answer key: ../solutions/week-05/assessment.md
