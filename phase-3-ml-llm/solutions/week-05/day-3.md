# Solutions — Day 3: Multi-Head Attention

## Exercise 1 — Dropout in MHA

```python
class MultiHeadAttentionWithDropout(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)  # applied to attention weights

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)  # dropout on weights
        attn_out = torch.matmul(weights, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out), weights
```

```python
# Test: dropout active in training, not in eval
def test_dropout_applied():
    torch.manual_seed(0)
    mha = MultiHeadAttentionWithDropout(d_model=64, num_heads=4, dropout=0.9)
    x = torch.randn(1, 8, 64)

    mha.train()
    _, w_train = mha(x, x, x)

    mha.eval()
    _, w_eval = mha(x, x, x)

    # In eval mode, no dropout — weights should be non-zero
    assert w_eval.min().item() >= 0.0
    # In train mode with 0.9 dropout, many weights should be 0
    zero_frac = (w_train == 0).float().mean().item()
    assert zero_frac > 0.5, f"Expected >50% zeros in train, got {zero_frac:.2f}"
```

---

## Exercise 2 — Cross-attention (n ≠ m)

The implementation in Day 3 already supports cross-attention because Q comes from `query` (length n) and K/V from `key`/`value` (length m). Shape verification:

```python
def test_cross_attention():
    mha = MultiHeadAttention(d_model=64, num_heads=4)
    B, n, m, d = 2, 3, 7, 64
    q = torch.randn(B, n, d)
    kv = torch.randn(B, m, d)
    out, w = mha(q, kv, kv)
    assert out.shape == (B, n, d)
    assert w.shape == (B, 4, n, m)
```

---

## Exercise 3 — Parameter count

For `MultiHeadAttention(d_model=512, num_heads=8)`:

| Matrix | Shape | Parameters |
|---|---|---|
| W_q | (512, 512) | 262,144 |
| W_k | (512, 512) | 262,144 |
| W_v | (512, 512) | 262,144 |
| W_o | (512, 512) | 262,144 |
| **Total** | | **1,048,576** |

```python
mha = MultiHeadAttention(d_model=512, num_heads=8)
total = sum(p.numel() for p in mha.parameters())
assert total == 4 * 512 * 512  # 1,048,576
```

---

## Exercise 4 — Why `.contiguous()` before `.view()`

`.transpose()` returns a view with non-contiguous memory — the stride order doesn't match the logical shape. `.view()` requires the tensor to occupy a contiguous block of memory.

Without `.contiguous()`:
```
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(…) instead.
```

Fix: either call `.contiguous().view(…)` or use `.reshape(…)` (which calls contiguous internally when needed).
