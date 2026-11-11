# Solutions — Day 5: Full Transformer Block

## Exercise 1 — Greedy generate

```python
def generate(
    self,
    prompt: list[int],
    max_new_tokens: int,
) -> list[int]:
    """Greedy decoding: repeatedly pick argmax of next-token logits."""
    self.eval()
    tokens = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = self(tokens)       # (1, T, vocab_size)
            next_tok = logits[0, -1].argmax(dim=-1, keepdim=True)  # (1,)
            tokens = torch.cat([tokens, next_tok.unsqueeze(0)], dim=1)
    return tokens.squeeze(0).tolist()


# Test
def test_generate_length():
    model = TinyDecoderOnlyTransformer(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    # Monkey-patch the method
    import types
    model.generate = types.MethodType(generate, model)

    prompt = [1, 2, 3]
    output = model.generate(prompt, max_new_tokens=5)
    assert len(output) == len(prompt) + 5, f"Expected {len(prompt)+5}, got {len(output)}"
```

---

## Exercise 2 — EncoderBlock and EncoderDecoderBlock

```python
class EncoderBlock(nn.Module):
    """No causal mask — all positions attend to all positions."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class EncoderDecoderBlock(nn.Module):
    """
    Three sub-layers:
    1. Causal self-attention (decoder stream).
    2. Cross-attention (Q from decoder, K/V from encoder).
    3. FFN.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # decoder stream  (B, T_dec, d_model)
        enc_out: torch.Tensor,     # encoder output  (B, T_enc, d_model)
        causal_mask_: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Causal self-attention
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, causal_mask_))
        # 2. Cross-attention
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, enc_out, enc_out))
        # 3. FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
```

---

## Exercise 3 — Pre-LN vs Post-LN

```python
class TransformerBlockConfigurable(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int,
        dropout: float = 0.0, pre_ln: bool = True
    ) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.pre_ln:
            x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

**Pre-LN recommended for deep models** because: at initialisation, the residual stream is unmodified (LayerNorm receives x, but x is also added unchanged). Post-LN normalises the residual stream output, which can destabilise very deep networks — the gradient must flow through the LayerNorm's normalisation operation at every layer. Pre-LN allows a direct gradient highway through the residual connections.

---

## Exercise 4 — Causal mask for T=4

```
causal_mask(4) =
[[False, True,  True,  True ],
 [False, False, True,  True ],
 [False, False, False, True ],
 [False, False, False, False]]
```

`mask[i, i] = False` (diagonal is NOT masked).

Reason: position i must be able to attend to itself — it is not a future position. Masking the diagonal would prevent a token from incorporating its own representation, which is almost always wrong. The causal constraint is "no attending to strictly future positions", i.e., j > i, which is the strict upper triangle.
