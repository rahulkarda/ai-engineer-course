# Phase 3 · Week 06 · Day 4 — Assembling MiniGPT & Training Loop

> Objective: Wire embeddings, transformer blocks, and a linear head into a full GPT model; implement the cross-entropy training loop with loss logging.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]], [[week-06/day-2]], [[week-06/day-3]]

---

## Concept

### Full decoder-only GPT architecture

```
Tokens (B, T) int
  │
  ▼
InputEmbedding  → (B, T, D)   [token emb + pos emb + dropout]
  │
  ▼
TransformerBlock × N          [self-attn + FFN, residuals]
  │
  ▼
LayerNorm (final)             [pre-LN requires one at the top]
  │
  ▼
Linear head (D → vocab_size)  [unnormalised logits, no softmax]
  → (B, T, V)
```

The **language modelling loss** is the mean cross-entropy over all `B×T` positions:

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

### Training loop anatomy

```
for step in range(max_steps):
    x, y     = next(iter(loader))        # (B, T)
    logits   = model(x)                  # (B, T, V)
    loss     = cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

Gradient clipping (norm 1.0) prevents exploding gradients — critical for transformers.

---

## How It Works Internally

### Cross-entropy loss for language models

`F.cross_entropy(input, target)` expects:
- `input`: `(N, C)` — logits (unnormalised)
- `target`: `(N,)` — integer class indices

We reshape `(B, T, V) → (B*T, V)` and `(B, T) → (B*T,)`.  
Loss measures bits of information needed to encode the next token; **lower is better**.  
Random model baseline: `log(vocab_size)` ≈ 4.17 nats for vocab=65.

### Learning rate and AdamW

AdamW (Adam + decoupled weight decay) is the standard for transformers.  
Typical config for a tiny model: `lr=3e-4`, `weight_decay=0.1`, `betas=(0.9, 0.95)`.  
Embedding and LayerNorm weights are excluded from weight decay (they are not "weight" tensors in the same sense).

---

## Worked Example

```python
"""
day4_minigpt.py
Full miniGPT model + training loop on a synthetic corpus.
Run: python day4_minigpt.py
Expected: loss drops from ~3.5 to ~0.5 in 200 steps (overfit tiny data).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse components from previous days (paste inline for self-containment)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size:   int   = 65
    context_len:  int   = 32
    d_model:      int   = 64
    n_heads:      int   = 4
    n_layers:     int   = 2
    dropout:      float = 0.0   # 0 for overfitting demo
    # training
    batch_size:   int   = 8
    max_steps:    int   = 200
    lr:           float = 3e-4
    weight_decay: float = 0.1


# ---------------------------------------------------------------------------
# Building blocks (inline for self-containment)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.c_attn  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.drop    = nn.Dropout(cfg.dropout)
        mask = torch.triu(torch.ones(cfg.context_len, cfg.context_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        reshape = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        attn   = self.drop(F.softmax(scores, dim=-1))
        out    = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full MiniGPT
# ---------------------------------------------------------------------------
class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg      = cfg
        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb  = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)
        self.blocks   = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f     = nn.LayerNorm(cfg.d_model)
        self.head     = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding and head weights (GPT-2 style)
        self.head.weight = self.tok_emb.weight

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        log.info("MiniGPT: %d parameters", n)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)          # (T,)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # (B, T, D)
        x    = self.blocks(x)                              # (B, T, D)
        x    = self.ln_f(x)                                # (B, T, D)
        logits = self.head(x)                              # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
        return logits, loss

    def configure_optimizer(self) -> torch.optim.AdamW:
        """AdamW with weight decay on weight matrices only, not biases/LN/emb."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                decay.append(p)
            else:
                no_decay.append(p)
        groups = [
            {"params": decay,    "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.cfg.lr, betas=(0.9, 0.95))


# ---------------------------------------------------------------------------
# Inline dataset (from Day 1, self-contained)
# ---------------------------------------------------------------------------
from torch.utils.data import Dataset as TorchDataset

SYNTHETIC_TEXT = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
) * 120


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._stoi = {c: i for i, c in enumerate(chars)}
        self._itos = {i: c for c, i in self._stoi.items()}

    def encode(self, t: str) -> list[int]: return [self._stoi[c] for c in t]
    def decode(self, ids: list[int]) -> str: return "".join(self._itos[i] for i in ids)


class CharDataset(TorchDataset):
    def __init__(self, tokens: list[int], ctx: int) -> None:
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.ctx  = ctx

    def __len__(self) -> int: return len(self.data) - self.ctx

    def __getitem__(self, i: int):
        c = self.data[i : i + self.ctx + 1]
        return c[:-1], c[1:]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(cfg: GPTConfig) -> MiniGPT:
    tok  = CharTokenizer(SYNTHETIC_TEXT)
    cfg.vocab_size = tok.vocab_size

    tokens = tok.encode(SYNTHETIC_TEXT)
    ds     = CharDataset(tokens, cfg.context_len)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = MiniGPT(cfg)
    opt   = model.configure_optimizer()

    loader_iter = iter(loader)
    losses: list[float] = []

    for step in range(cfg.max_steps):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)

        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        losses.append(loss.item())
        if step % 50 == 0 or step == cfg.max_steps - 1:
            log.info("step %4d | loss %.4f", step, loss.item())

    log.info("Final loss: %.4f  (baseline random=%.4f)",
             losses[-1], math.log(tok.vocab_size))
    return model


def main() -> None:
    torch.manual_seed(42)
    cfg   = GPTConfig()
    model = train(cfg)
    # Verify final loss is well below baseline
    # Baseline for vocab ~28: ln(28) ≈ 3.33
    # After 200 steps we expect < 0.5 (overfitting tiny corpus)


if __name__ == "__main__":
    main()
```

Run with: `python day4_minigpt.py`

Expected loss curve: starts ~3.3, drops to < 0.5 within 200 steps (overfitting intentional — the tiny corpus is easy to memorise).

### Architecture summary at default config

| Component | Shape in | Shape out |
|---|---|---|
| Token + pos emb | `(B, T)` int | `(B, T, 64)` |
| 2× Block | `(B, T, 64)` | `(B, T, 64)` |
| Final LN | `(B, T, 64)` | `(B, T, 64)` |
| Head | `(B, T, 64)` | `(B, T, vocab)` |

---

## Exercises

1. Add a `val_loss` computation every 50 steps (use a held-out 10% split from `CharDataset`). Log both `train_loss` and `val_loss`.
2. Implement a cosine learning rate schedule that decays from `lr` to `lr/10` over `max_steps`. Apply it in the training loop.
3. Add gradient norm logging every 50 steps (before the clip). Observe how clipping affects it.
4. Implement weight tying verification: assert that `model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()` after construction.
5. Increase `n_layers` to 4 and `d_model` to 128. Log parameter count and observe how training time changes. At what config does training on CPU become > 60 s for 200 steps?

(Solutions: ../solutions/week-06/day-4.md)

---

## Self-check

- [ ] I can draw the full MiniGPT forward pass from `idx` to `loss`
- [ ] I understand why we reshape logits before cross-entropy
- [ ] I can explain weight tying and its benefit (parameter reduction)
- [ ] I can implement `configure_optimizer` with parameter group separation

---

## Supplementary

- [nanoGPT training loop — train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py)
- [AdamW paper — Loshchilov & Hutter 2019](https://arxiv.org/abs/1711.05101)
- [Gradient clipping in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
