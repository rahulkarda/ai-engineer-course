# Phase 3 · Week 05 · Day 3 — Multi-Head Attention (Implement)

> Objective: Understand why multiple attention heads are needed and implement multi-head attention from scratch.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Why single-head attention is limiting

Single-head attention computes one weighted blend of all values. But a position often needs to attend for multiple distinct reasons simultaneously:

- Token "bank" in "river bank" needs to attend to "river" (semantics) AND to its syntactic head verb (syntax) at the same time.
- Positional dependencies, co-reference, subject-verb agreement — these are structurally independent relationship types.

One attention distribution cannot represent multiple independent relationship types without interference.

### Multi-head attention: parallel projections

Run `h` independent attention heads, each on a different learned linear projection of the input:

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Concatenate and project back:

```
MultiHead(Q, K, V) = Concat(head_1, …, head_h) W^O
```

Dimensions:
- Input: `(batch, n, d_model)`
- Each projection: `W_i^Q, W_i^K ∈ ℝ^{d_model × d_k}`, `W_i^V ∈ ℝ^{d_model × d_v}`
- Standard choice: `d_k = d_v = d_model / h`
- After concat: `(batch, n, h · d_v) = (batch, n, d_model)`
- After W^O ∈ ℝ^{d_model × d_model}: still `(batch, n, d_model)`

Total parameter count: `3 · d_model · d_model + d_model · d_model = 4 · d_model²` — same as one large head but more expressively factored.

### Efficient implementation: reshape, don't loop

Looping over heads is slow. Instead, reshape the input so all heads run in one batched matrix multiply:

1. Project input `(B, T, d_model)` → `(B, T, h, d_k)` via a single `Linear(d_model, h*d_k)`.
2. Transpose to `(B, h, T, d_k)` — heads become a batch dimension.
3. Apply `scaled_dot_product_attention` to the 4D tensors — broadcasts over both B and h.
4. Transpose back to `(B, T, h*d_v)`, reshape to `(B, T, d_model)`.
5. Apply output projection W^O.

## How it works internally

### Each head specialises

Because heads share no parameters (separate projection matrices), they are free to specialise. Empirical analysis (Voita et al. 2019) shows some heads:
- Attend to syntactic dependencies.
- Attend to the previous token (positional).
- Attend to rare/sentinel tokens.
- Are largely redundant and can be pruned.

This emergent specialisation requires no explicit supervision — it follows from gradient descent minimising the loss.

### The role of W^O

After concatenation, the output projection W^O mixes information across heads. Without it, heads are completely independent. W^O lets the model learn which combination of head outputs is useful.

### Complexity

Per-head: O(T² · d_k). Total with h heads: O(h · T² · d_k) = O(T² · d_model). Same asymptotic as single-head with d_model. Multi-head is not more expensive in FLOPS — it's a different factoring of the same compute budget.

## Worked example

```python
"""
day3_multi_head_attention.py
Implements multi-head attention from scratch without using nn.MultiheadAttention.
Run: python day3_multi_head_attention.py
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
    """Scaled dot-product attention — reused from Day 2."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as in 'Attention Is All You Need'.

    Args:
        d_model: Model dimension (must be divisible by num_heads).
        num_heads: Number of parallel attention heads.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # per-head dimension

        # Single matrix for Q, K, V projections (efficient)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (B, T, d_model) → (B, num_heads, T, d_k).
        This allows batched attention over all heads simultaneously.
        """
        B, T, _ = x.shape
        # (B, T, num_heads, d_k) → (B, num_heads, T, d_k)
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, n, d_model)
            key:   (B, m, d_model)
            value: (B, m, d_model)
            mask:  Optional (B, 1, n, m) or (B, num_heads, n, m) bool mask.

        Returns:
            output:       (B, n, d_model)
            attn_weights: (B, num_heads, n, m)
        """
        B = query.size(0)

        # Project and split into heads
        q = self._split_heads(self.W_q(query))  # (B, h, n, d_k)
        k = self._split_heads(self.W_k(key))    # (B, h, m, d_k)
        v = self._split_heads(self.W_v(value))  # (B, h, m, d_k)

        # Batched attention over all heads
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        # attn_out: (B, h, n, d_k)

        # Merge heads: (B, h, n, d_k) → (B, n, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # Output projection
        output = self.W_o(attn_out)  # (B, n, d_model)
        return output, attn_weights


def verify_shapes() -> None:
    """Comprehensive shape tests."""
    torch.manual_seed(42)
    B, T, d_model, h = 2, 10, 64, 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=h)
    x = torch.randn(B, T, d_model)

    out, weights = mha(x, x, x)  # self-attention

    assert out.shape == (B, T, d_model), f"output shape: {out.shape}"
    assert weights.shape == (B, h, T, T), f"weights shape: {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, h, T), atol=1e-5)

    log.info(
        "Shape test PASSED: output %s, weights %s", out.shape, weights.shape
    )


def compare_with_pytorch() -> None:
    """Cross-check output against PyTorch's built-in MultiheadAttention."""
    torch.manual_seed(0)
    B, T, d_model, h = 1, 6, 32, 4

    our_mha = MultiHeadAttention(d_model=d_model, num_heads=h)

    # PyTorch MHA (batch_first=True, no bias for fair comparison)
    pt_mha = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=h, bias=False, batch_first=True
    )

    # Copy our weights into PyTorch's format
    # PyTorch packs W_q, W_k, W_v into in_proj_weight: (3*d, d)
    with torch.no_grad():
        pt_mha.in_proj_weight.copy_(
            torch.cat([our_mha.W_q.weight, our_mha.W_k.weight, our_mha.W_v.weight])
        )
        pt_mha.out_proj.weight.copy_(our_mha.W_o.weight)

    x = torch.randn(B, T, d_model)
    our_out, _ = our_mha(x, x, x)
    pt_out, _ = pt_mha(x, x, x, need_weights=False)

    max_diff = (our_out - pt_out).abs().max().item()
    log.info("Max diff vs PyTorch MHA: %.6f (should be < 1e-5)", max_diff)


if __name__ == "__main__":
    verify_shapes()
    log.info("\n--- Cross-check vs PyTorch ---")
    compare_with_pytorch()
```

## Exercises

1. Add a `dropout: float = 0.0` parameter to `MultiHeadAttention` that applies dropout to the attention weights after softmax (standard practice). Write a test confirming dropout is applied during training but not eval.

2. Implement cross-attention: modify `forward` so that `query` can have a different sequence length `n` from `key`/`value` (length `m`). Verify with `n=3, m=7`.

3. Count the total trainable parameters in a `MultiHeadAttention(d_model=512, num_heads=8)`. Show the calculation by component (W_q, W_k, W_v, W_o). Do the math before calling `sum(p.numel() for p in mha.parameters())`.

4. Why is `.contiguous()` needed before `.view()` after `.transpose()`? What error occurs without it?

(Solutions: ../solutions/week-05/day-3.md)

## Self-check

- [ ] I can state the full multi-head attention formula including all projection matrices.
- [ ] I can implement `MultiHeadAttention` from scratch without looking at notes.
- [ ] I understand why reshaping + batched matmul is preferred over looping over heads.
- [ ] I can explain what W^O does and why it's necessary.

## Supplementary

- Vaswani et al. (2017) Section 3.2.2 — https://arxiv.org/abs/1706.03762
- Voita et al. (2019) "Analyzing Multi-Head Self-Attention" — https://arxiv.org/abs/1905.09418
- PyTorch MHA source (study the in_proj_weight packing) — https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
