# Solutions — Week 06 · Day 2

## Exercise 1 — Parameter count

```
vocab_size=65, context_len=128, d_model=128, learned_pos=True

Token embedding   : 65  × 128 = 8,320
Positional embed  : 128 × 128 = 16,384
Total             :             24,704
```

Verification:
```python
emb = InputEmbedding(65, 128, 128, learned_pos=True)
assert sum(p.numel() for p in emb.parameters()) == 24704
```

---

## Exercise 2 — `RotaryPE` stub

```python
class RotaryPE(nn.Module):
    """Stub for Rotary Positional Embedding (RoPE).

    RoPE encodes position by rotating Q and K vectors in 2-D subspaces
    using sine/cosine of position-dependent frequencies. Because rotation
    is applied to Q and K (not the residual stream), it does not shift V,
    and the dot-product Q·K naturally incorporates relative position
    information without adding a separate position embedding to the input.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) — would apply rotation to Q and K in practice
        raise NotImplementedError("RoPE rotation not implemented in this stub")
```

---

## Exercise 3 — `test_token_embed_lookup`

```python
import torch
from minigpt.data import TokenEmbedding

def test_token_embed_lookup():
    emb = TokenEmbedding(vocab_size=10, d_model=8)
    # Manually set row 3 to all-ones
    with torch.no_grad():
        emb.embed.weight[3] = 1.0

    out = emb(torch.tensor([[3]]))        # (1, 1, 8)
    assert out.shape == (1, 1, 8)
    assert torch.allclose(out, torch.ones(1, 1, 8))
```

---

## Exercise 4 — `sqrt(d_model)` scaling

```python
class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        learned_pos: bool = True,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb   = (LearnedPE(context_len, d_model)
                          if learned_pos else SinusoidalPE(context_len, d_model))
        self.drop  = nn.Dropout(p=0.1)
        self._scale = math.sqrt(d_model) if scale else 1.0

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        tok = self.token_emb(idx) * self._scale
        return self.drop(self.pos_emb(tok))
```

The Vaswani scaling prevents token embeddings from having much smaller magnitude than positional embeddings early in training, keeping the two signals comparable.

---

## Exercise 5 — Sinusoidal vs learned PE

**Sinusoidal:** the PE values at position `T+k` are computed by the same formula used for any `T`, so the model can in principle attend to positions beyond the training window — though empirically quality degrades for very long extrapolation.

**Learned:** the embedding table only has rows for positions `0..context_len-1`. Querying index `context_len` or beyond would either raise an index error or access garbage memory; there is no mathematically meaningful value for unseen positions.
