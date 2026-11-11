# Solutions — Week 06 · Day 3

## Exercise 1 — `return_attn` option

```python
class CausalSelfAttention(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        r = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn   = self.attn_drop(F.softmax(scores, dim=-1))
        out    = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        out    = self.resid_drop(self.c_proj(out))
        if return_attn:
            # Average over heads → (B, T, T)
            return out, attn.mean(dim=1)
        return out
```

Test:
```python
def test_causal_attn_upper_zero():
    cfg = GPTConfig(vocab_size=10, context_len=8, d_model=16, n_heads=2,
                    n_layers=1, dropout=0.0)
    attn = CausalSelfAttention(cfg)
    attn.eval()
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        _, weights = attn(x, return_attn=True)
    # weights: (1, 8, 8) — upper triangle should be 0 (after masked softmax)
    upper = torch.triu(weights[0], diagonal=1)
    assert upper.abs().max().item() < 1e-6, "Upper triangle non-zero — causality broken"
```

---

## Exercise 2 — Separate Q, K, V projections

```python
class CausalSelfAttentionSeparate(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        mask = torch.triu(torch.ones(cfg.context_len, cfg.context_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        r = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        s = s.masked_fill(self.mask[:T, :T], float("-inf"))
        o = (F.softmax(s, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(o)
```

Output shape is identical to the combined version.

---

## Exercise 3 — `count_params`

```python
def count_params(module: nn.Module) -> dict[str, int]:
    return {
        name: sum(p.numel() for p in sub.parameters())
        for name, sub in module.named_children()
    }
```

Example output for `TransformerBlock(d_model=64, n_heads=4)`:
```
{'ln1': 128, 'attn': 32768, 'ln2': 128, 'ffn': 16512}
```

---

## Exercise 4 — SwiGLU explanation

SwiGLU splits the expansion into two parallel linear layers:
```
SwiGLU(x) = (x W_1) * SiLU(x W_2)
```
This doubles the hidden dimension to `8/3 × d_model` (rounded to a multiple of 64) to keep parameter count similar. SwiGLU is preferred in Llama/Mistral because it introduces a gating mechanism that empirically improves loss-per-parameter by ~5–10%. The shape change is that `W_1` and `W_2` are both `(d_model, 4d_model)` instead of one `(d_model, 4d_model)` layer.

---

## Exercise 5 — Gradient flow test

```python
def test_gradients_flow():
    cfg = GPTConfig(vocab_size=10, context_len=8, d_model=16,
                    n_heads=2, n_layers=1, dropout=0.0)
    block = TransformerBlock(cfg)
    x = torch.randn(1, 8, 16, requires_grad=True)
    out = block(x)
    out.mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
```
