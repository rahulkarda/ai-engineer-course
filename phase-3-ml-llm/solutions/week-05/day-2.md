# Solutions — Day 2: Scaled Dot-Product Attention

## Exercise 1 — pytest test file

```python
"""tests/test_day2_attention.py"""
import math
import pytest
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


def causal_mask(T: int) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        torch.manual_seed(0)
        B, n, m, d_k, d_v = 3, 5, 7, 16, 32
        q = torch.randn(B, n, d_k)
        k = torch.randn(B, m, d_k)
        v = torch.randn(B, m, d_v)
        out, w = scaled_dot_product_attention(q, k, v)
        assert out.shape == (B, n, d_v)
        assert w.shape == (B, n, m)

    def test_weights_sum_to_one(self):
        torch.manual_seed(1)
        B, n, m, d = 2, 4, 6, 8
        q = torch.randn(B, n, d)
        k = torch.randn(B, m, d)
        v = torch.randn(B, m, d)
        _, w = scaled_dot_product_attention(q, k, v)
        assert torch.allclose(w.sum(dim=-1), torch.ones(B, n), atol=1e-5)

    def test_causal_mask_zeros_upper_triangle(self):
        torch.manual_seed(2)
        T, d = 5, 8
        q = torch.randn(1, T, d)
        k = torch.randn(1, T, d)
        v = torch.randn(1, T, d)
        mask = causal_mask(T).unsqueeze(0)  # (1, T, T)
        _, w = scaled_dot_product_attention(q, k, v, mask=mask)
        # Upper triangle (j > i) should be 0
        for i in range(T):
            for j in range(i + 1, T):
                assert w[0, i, j].item() < 1e-6, f"Expected 0 at ({i},{j}), got {w[0, i, j].item()}"
```

---

## Exercise 2 — Temperature attention

```python
def temperature_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace √d_k scaling with arbitrary temperature.
    temperature → 0: near one-hot (concentrated)
    temperature → ∞: uniform distribution
    """
    import torch.nn.functional as F
    scores = torch.matmul(q, k.transpose(-2, -1)) / temperature
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights
```

**Uniform distribution:** `temperature → ∞` → all scores approach 0 → softmax outputs 1/m for each key.

**Near one-hot:** `temperature → 0` → scores become infinitely large relative differences → winner takes all.

The standard `1/√d_k` corresponds to a temperature of `√d_k`, chosen so variance ≈ 1 with standard-normal inputs.

---

## Exercise 3 — Numerical variance proof

```python
import torch

torch.manual_seed(42)
d = 64
n_samples = 100_000

q = torch.randn(n_samples, d)
k = torch.randn(n_samples, d)
dots = (q * k).sum(dim=-1)  # element-wise, then sum = dot product

print(f"E[q·k]    = {dots.mean():.4f}  (should be ~0)")
print(f"Var[q·k]  = {dots.var():.4f}  (should be ~{d})")
print(f"Var[q·k/√d] = {(dots / d**0.5).var():.4f}  (should be ~1)")
```

Each term `qᵢkᵢ` has mean 0 and variance 1 (product of two standard normals). Sum of d independent such terms has variance d. Division by √d gives variance 1.

---

## Exercise 4 — Identical keys

If all keys are identical (`k_1 = k_2 = … = k_m`), then all scores `q · kᵢ / √d_k` are identical. Softmax of equal values produces a uniform distribution: `weights[i, j] = 1/m` for all j.

The output is just the uniform average of all values: `output = (1/m) Σ v_j`.

This shows that attention with identical keys degenerates to a mean pooling — the attention mechanism provides no selectivity. The expressivity of attention depends on having diverse keys that vary meaningfully with the input.
