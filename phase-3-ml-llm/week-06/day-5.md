# Phase 3 · Week 06 · Day 5 — Sampling: Greedy, Temperature & Top-k

> Objective: Implement three text generation strategies (greedy, temperature, top-k) and understand how each shapes the output distribution.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-4]]

---

## Concept

A trained model produces **logits** `(1, T, V)` at each forward pass.  
The **last-position logit** `logits[:, -1, :]` is the distribution over the next token.  
Sampling strategies decide *which token to pick* from that distribution.

| Strategy | Idea | Trade-off |
|---|---|---|
| **Greedy** | Always pick highest-probability token | Deterministic, repetitive |
| **Temperature** | Divide logits by `T` before softmax | `T<1` → sharper; `T>1` → flatter |
| **Top-k** | Zero out all but top-k logits, then sample | Prevents tail-token chaos |
| **Top-p (nucleus)** | Keep minimum tokens whose cumulative prob ≥ p | Adaptive; not built today |

Sampling strategies are applied **at inference**, not during training.

---

## How It Works Internally

### Temperature scaling

```
probs = softmax(logits / temperature)
```

- `temperature → 0`: greedy limit (argmax)
- `temperature = 1`: original trained distribution
- `temperature > 1`: flatter distribution → more creative / random

### Top-k

1. Keep only the top-k logits; set all others to `-inf`.
2. Apply softmax → probabilities sum to 1 over k tokens.
3. `torch.multinomial` samples one index.

### Autoregressive generation loop

```
for _ in range(max_new_tokens):
    # crop context to model's context_len
    idx_cond = idx[:, -context_len:]
    logits, _ = model(idx_cond)
    next_logit = logits[:, -1, :]          # (B, V)
    next_token = sample(next_logit)        # (B, 1)
    idx = torch.cat([idx, next_token], dim=1)
```

Each generated token is appended and fed back in — this is **autoregressive** decoding.

---

## Worked Example

```python
"""
day5_sampling.py
Text generation: greedy, temperature, top-k.
Run: python day5_sampling.py  (trains a quick model then samples)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal MiniGPT (self-contained, copy of Day 4 essentials)
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size:  int   = 65
    context_len: int   = 32
    d_model:     int   = 64
    n_heads:     int   = 4
    n_layers:    int   = 2
    dropout:     float = 0.0
    batch_size:  int   = 8
    max_steps:   int   = 300
    lr:          float = 3e-4


import math

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.c_attn  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj  = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        mask = torch.triu(torch.ones(cfg.context_len, cfg.context_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        r = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        s = s.masked_fill(self.mask[:T, :T], float("-inf"))
        out = (F.softmax(s, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(out)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x   = self.tok_emb(idx) + self.pos_emb(pos)
        x   = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------
def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """Greedy: always pick argmax. Shape: (B, V) → (B, 1)."""
    return logits.argmax(dim=-1, keepdim=True)


def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Temperature sampling. Shape: (B, V) → (B, 1)."""
    if temperature <= 0:
        return sample_greedy(logits)
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=-1)          # (B, V)
    return torch.multinomial(probs, num_samples=1)  # (B, 1)


def sample_topk(logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> torch.Tensor:
    """Top-k + temperature sampling. Shape: (B, V) → (B, 1)."""
    # Zero out all logits below the k-th largest
    topk_vals, _ = torch.topk(logits, k, dim=-1)
    threshold = topk_vals[:, -1:].expand_as(logits)  # (B, V)
    filtered  = logits.masked_fill(logits < threshold, float("-inf"))
    return sample_temperature(filtered, temperature)


SamplingStrategy = Literal["greedy", "temperature", "topk"]


@torch.no_grad()
def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    strategy: SamplingStrategy = "temperature",
    temperature: float = 0.8,
    top_k: int = 10,
) -> torch.Tensor:
    """Autoregressive generation loop.

    Args:
        model: trained MiniGPT (in eval mode)
        idx:   seed token ids, shape (1, T_seed)
        max_new_tokens: how many tokens to generate
        strategy: "greedy" | "temperature" | "topk"
        temperature: softmax temperature (ignored for greedy)
        top_k: number of candidates for top-k (ignored otherwise)

    Returns:
        token ids, shape (1, T_seed + max_new_tokens)
    """
    model.eval()
    ctx = model.cfg.context_len

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -ctx:]                          # crop to context
        logits, _ = model(idx_cond)
        last_logit = logits[:, -1, :]                     # (1, V)

        if strategy == "greedy":
            next_tok = sample_greedy(last_logit)
        elif strategy == "temperature":
            next_tok = sample_temperature(last_logit, temperature)
        else:  # topk
            next_tok = sample_topk(last_logit, k=top_k, temperature=temperature)

        idx = torch.cat([idx, next_tok], dim=1)

    return idx


# ---------------------------------------------------------------------------
# Inline corpus + training (abbreviated — 300 steps, tiny model)
# ---------------------------------------------------------------------------
CORPUS = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
    "or to take arms against a sea of troubles\n"
) * 150


class CharTok:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._s = {c: i for i, c in enumerate(chars)}
        self._i = {i: c for c, i in self._s.items()}

    def encode(self, t: str) -> list[int]: return [self._s[c] for c in t]
    def decode(self, ids: list[int]) -> str: return "".join(self._i[i] for i in ids)


class CharDS(Dataset):
    def __init__(self, toks: list[int], ctx: int) -> None:
        self.d, self.ctx = torch.tensor(toks, dtype=torch.long), ctx
    def __len__(self): return len(self.d) - self.ctx
    def __getitem__(self, i): c = self.d[i:i+self.ctx+1]; return c[:-1], c[1:]


def quick_train(cfg: GPTConfig, tok: CharTok) -> MiniGPT:
    tokens = tok.encode(CORPUS)
    loader = DataLoader(CharDS(tokens, cfg.context_len),
                        batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    model = MiniGPT(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    it    = iter(loader)
    for step in range(cfg.max_steps):
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            log.info("train step %d | loss %.4f", step, loss.item())
    return model


def main() -> None:
    torch.manual_seed(42)
    tok = CharTok(CORPUS)
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = quick_train(cfg, tok)

    # Seed: first 8 chars of the corpus
    seed_str = "to be or"
    seed_ids = tok.encode(seed_str)
    prompt   = torch.tensor([seed_ids], dtype=torch.long)  # (1, 8)

    log.info("=== Greedy ===")
    greedy_ids = generate(model, prompt, max_new_tokens=60, strategy="greedy")
    log.info("%r", tok.decode(greedy_ids[0].tolist()))

    log.info("=== Temperature=0.8 ===")
    temp_ids = generate(model, prompt, max_new_tokens=60, strategy="temperature",
                        temperature=0.8)
    log.info("%r", tok.decode(temp_ids[0].tolist()))

    log.info("=== Top-k=5 T=0.8 ===")
    topk_ids = generate(model, prompt, max_new_tokens=60, strategy="topk",
                        temperature=0.8, top_k=5)
    log.info("%r", tok.decode(topk_ids[0].tolist()))

    log.info("=== Temperature=1.5 (high entropy) ===")
    hot_ids = generate(model, prompt, max_new_tokens=60, strategy="temperature",
                       temperature=1.5)
    log.info("%r", tok.decode(hot_ids[0].tolist()))


if __name__ == "__main__":
    main()
```

Run with: `python day5_sampling.py`

### Sampling cheat sheet

| Strategy | Call | Deterministic? |
|---|---|---|
| Greedy | `sample_greedy(logits)` | Yes |
| Temperature | `sample_temperature(logits, T=0.8)` | No |
| Top-k + temp | `sample_topk(logits, k=10, T=0.8)` | No |

---

## Exercises

1. Implement **top-p (nucleus) sampling**: keep the smallest set of tokens whose cumulative probability exceeds `p`, zero out the rest, then sample. Test with `p=0.9`.
2. Add a `repetition_penalty` parameter to `sample_temperature`: divide logits for already-generated tokens by a penalty factor > 1 before sampling. Observe its effect on greedy output.
3. Write a `compare_strategies` function that generates 3 different continuations (greedy, temperature=0.7, top-k=5) for the same prompt and prints them side by side in a table.
4. Write a pytest test `test_greedy_deterministic` that calls `generate(..., strategy="greedy")` twice with the same seed and asserts the outputs are identical.
5. Write a pytest test `test_topk_respects_k` that patches the model to output uniform logits and verifies that `sample_topk` only ever returns tokens within the top-k indices.

(Solutions: ../solutions/week-06/day-5.md)

---

## Self-check

- [ ] I can explain what temperature does to the output distribution mathematically
- [ ] I can implement greedy, temperature, and top-k from scratch
- [ ] I understand the autoregressive loop and why we crop the context
- [ ] I can state when greedy is preferred vs temperature vs top-k

---

## Supplementary

- [The Curious Case of Neural Text Degeneration (Holtzman et al. 2020) — nucleus sampling](https://arxiv.org/abs/1904.09751)
- [nanoGPT generate method](https://github.com/karpathy/nanoGPT/blob/master/model.py)
- [Hugging Face generation strategies overview](https://huggingface.co/docs/transformers/generation_strategies)
