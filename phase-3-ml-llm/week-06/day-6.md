# Phase 3 · Week 06 · Day 6 — Full MiniGPT: Train, Evaluate & Generate

> Objective: Run a 4-hour end-to-end training session — full pipeline from raw text to a model you can interact with, with loss curves and generation samples.  ·  Time: ~4 hr  ·  Prereqs: [[week-06/day-1]] through [[week-06/day-5]]

---

## Overview

Today you build and run the **complete miniGPT pipeline** as a single coherent script:

1. Load/generate corpus → tokenize → split train/val
2. Train for multiple epochs with loss logging and checkpointing
3. Plot train vs val loss curve (ASCII in-terminal)
4. Generate text at multiple temperatures

Target: by the end, your model memorises the training corpus well enough to generate plausible continuations, and you observe the train/val gap that signals overfitting.

---

## Concept Review: What "Training" Means Here

The model has no world knowledge. It learns **statistical patterns in your corpus** — bigrams, trigrams, phrase fragments. With enough capacity and steps it will perfectly memorise a tiny corpus. That's fine for a build week: the goal is to verify that the pipeline works and that loss moves in the right direction.

### Monitoring overfitting

| Signal | Meaning |
|---|---|
| `train_loss` ≫ `log(V)` | Model hasn't started learning |
| `train_loss` decreasing, `val_loss` flat/rising | Overfitting |
| Both decreasing | Generalising |
| Both converged ≈ equal | Well-regularised |

For a tiny corpus with no regularisation, expect severe overfitting — that's the point.

---

## Full Pipeline Script

```python
"""
day6_full_pipeline.py
Full miniGPT: train, eval, checkpoint, generate.
Time: ~4 hours if you tune hyperparameters. Fast CPU run (~2 min) at default config.

Usage:
    python day6_full_pipeline.py                    # use synthetic corpus
    python day6_full_pipeline.py --data input.txt   # use your own text file
"""
from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # model
    context_len:  int   = 64
    d_model:      int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 3
    dropout:      float = 0.1
    # training
    batch_size:   int   = 16
    max_steps:    int   = 500
    eval_interval: int  = 50
    eval_steps:   int   = 20
    lr:           float = 3e-4
    weight_decay: float = 0.1
    grad_clip:    float = 1.0
    # generation
    gen_prompt:   str   = "to be or not"
    gen_tokens:   int   = 200
    temperature:  float = 0.8
    top_k:        int   = 10
    # I/O
    checkpoint_path: Optional[Path] = None
    vocab_size:   int   = 0   # set after tokenizer is built


SYNTHETIC_CORPUS = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them to die to sleep\n"
    "no more and by a sleep to say we end\n"
    "the heartache and the thousand natural shocks\n"
    "that flesh is heir to tis a consummation\n"
    "devoutly to be wished to die to sleep\n"
    "to sleep perchance to dream ay there the rub\n"
) * 200   # ~10 k chars


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._s = {c: i for i, c in enumerate(chars)}
        self._i = {i: c for c, i in self._s.items()}
        log.info("Vocab size: %d", self.vocab_size)

    def encode(self, text: str) -> list[int]:
        return [self._s[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset):
    def __init__(self, tokens: list[int], ctx: int) -> None:
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.ctx  = ctx

    def __len__(self) -> int:
        return len(self.data) - self.ctx

    def __getitem__(self, i: int):
        c = self.data[i : i + self.ctx + 1]
        return c[:-1], c[1:]


def make_loaders(
    tokens: list[int], cfg: Config
) -> tuple[DataLoader, DataLoader]:
    split = int(len(tokens) * 0.9)
    train_ds = CharDataset(tokens[:split], cfg.context_len)
    val_ds   = CharDataset(tokens[split:], cfg.context_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, drop_last=True)
    log.info("Train windows: %d  Val windows: %d", len(train_ds), len(val_ds))
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model (self-contained, same as Day 4/5)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.nh = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.dm = cfg.d_model
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
        mask = torch.triu(torch.ones(cfg.context_len, cfg.context_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(self.dm, dim=2)
        r = lambda t: t.view(B, T, self.nh, self.dh).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s = s.masked_fill(self.mask[:T, :T], float("-inf"))
        a = self.drop(F.softmax(s, dim=-1))
        o = (a @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(o)


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self._init()
        n = sum(p.numel() for p in self.parameters())
        log.info("MiniGPT parameters: %d", n)

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        p = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(p))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))
        return logits, loss

    def configure_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.lr, betas=(0.9, 0.95),
        )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model: MiniGPT, idx: torch.Tensor,
             max_new: int, temp: float, k: int) -> torch.Tensor:
    model.eval()
    ctx = model.cfg.context_len
    for _ in range(max_new):
        logits, _ = model(idx[:, -ctx:])
        lgt = logits[:, -1, :] / max(temp, 1e-6)
        # top-k filter
        topv, _ = torch.topk(lgt, k)
        lgt = lgt.masked_fill(lgt < topv[:, -1:], float("-inf"))
        probs = F.softmax(lgt, dim=-1)
        nxt   = torch.multinomial(probs, 1)
        idx   = torch.cat([idx, nxt], dim=1)
    return idx


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model: MiniGPT, loader: DataLoader, steps: int) -> float:
    model.eval()
    total = 0.0
    count = 0
    for i, (x, y) in enumerate(loader):
        if i >= steps:
            break
        _, loss = model(x, y)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# ASCII loss plot
# ---------------------------------------------------------------------------
def ascii_loss_plot(train_losses: list[float], val_losses: list[float]) -> None:
    if not train_losses:
        return
    all_vals = train_losses + val_losses
    lo, hi   = min(all_vals), max(all_vals)
    W, H     = 60, 15
    canvas   = [[" "] * W for _ in range(H)]

    def row(v: float) -> int:
        if hi == lo: return H // 2
        frac = (v - lo) / (hi - lo)
        return H - 1 - int(frac * (H - 1))

    def col(i: int, total: int) -> int:
        return int(i / max(total - 1, 1) * (W - 1))

    for i, v in enumerate(train_losses):
        r, c = row(v), col(i, len(train_losses))
        if 0 <= r < H and 0 <= c < W:
            canvas[r][c] = "t"

    for i, v in enumerate(val_losses):
        r, c = row(v), col(i, len(val_losses))
        if 0 <= r < H and 0 <= c < W:
            canvas[r][c] = "v"

    log.info("Loss curve  t=train  v=val  (top=high loss)")
    log.info("%.3f ┤%s", hi, "─" * W)
    for row_data in canvas:
        log.info("      │%s", "".join(row_data))
    log.info("%.3f ┤%s", lo, "─" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None,
                        help="Path to a .txt file (optional; uses synthetic corpus otherwise)")
    args = parser.parse_args()

    torch.manual_seed(42)
    cfg = Config()

    # Load corpus
    if args.data and args.data.exists():
        text = args.data.read_text(encoding="utf-8")
        log.info("Loaded %d chars from %s", len(text), args.data)
    else:
        text = SYNTHETIC_CORPUS
        log.info("Using synthetic corpus (%d chars)", len(text))

    tok = CharTokenizer(text)
    cfg.vocab_size = tok.vocab_size
    tokens = tok.encode(text)

    train_loader, val_loader = make_loaders(tokens, cfg)

    model = MiniGPT(cfg)
    opt   = model.configure_optimizer()

    train_losses: list[float] = []
    val_losses:   list[float] = []
    t0 = time.time()

    it = iter(train_loader)
    for step in range(cfg.max_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        model.train()
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            tl = estimate_loss(model, train_loader, cfg.eval_steps)
            vl = estimate_loss(model, val_loader,   cfg.eval_steps)
            train_losses.append(tl)
            val_losses.append(vl)
            elapsed = time.time() - t0
            log.info("step %4d | train %.4f | val %.4f | %.1f s",
                     step, tl, vl, elapsed)

    # Checkpoint
    if cfg.checkpoint_path:
        cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg}, cfg.checkpoint_path)
        log.info("Saved checkpoint → %s", cfg.checkpoint_path)

    ascii_loss_plot(train_losses, val_losses)

    # Generation
    log.info("=" * 60)
    log.info("Prompt: %r", cfg.gen_prompt)
    seed_ids = tok.encode(cfg.gen_prompt)
    prompt   = torch.tensor([seed_ids], dtype=torch.long)

    for temp, label in [(0.0, "greedy"), (0.8, "T=0.8"), (1.2, "T=1.2")]:
        ids = generate(model, prompt.clone(), cfg.gen_tokens,
                       temp=max(temp, 1e-6), k=cfg.top_k)
        log.info("[%s] %r", label, tok.decode(ids[0].tolist()))


if __name__ == "__main__":
    main()
```

Run with: `python day6_full_pipeline.py`

Or with tiny shakespeare (if you download it):
```
python day6_full_pipeline.py --data input.txt
```

To download tiny shakespeare manually (≈ 1 MB):
```
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

## 4-Hour Lab Plan

| Time | Activity |
|---|---|
| 0:00–0:30 | Run default config. Verify loss drops. Read all log output. |
| 0:30–1:00 | Add val loss tracking; observe train/val gap. |
| 1:00–1:30 | Download tiny shakespeare; re-run. Compare loss curves. |
| 1:30–2:30 | Tune: try `n_layers=4`, `d_model=256`. Note time increase. |
| 2:30–3:00 | Experiment with temperatures (0.3, 0.8, 1.2). Note output quality. |
| 3:00–3:30 | Add top-p sampling (see Day 5 exercise 1). Compare to top-k. |
| 3:30–4:00 | Write tests (Day 6 exercises). Clean up code. |

---

## Exercises

1. Add a `--steps N` CLI flag to `day6_full_pipeline.py` so you can control training length without editing the source.
2. Implement and plot a **cosine decay** learning rate schedule. Log the LR at each eval step. Observe whether it improves final val loss.
3. Save **generation samples** to a file `samples.txt` — one per temperature — with the config logged at the top.
4. Modify the checkpoint to also save the tokenizer vocabulary so you can reload and generate without the original corpus.
5. Run with `dropout=0.2` vs `dropout=0.0` at 500 steps. Report (in a comment) whether dropout reduces the val/train gap on the tiny corpus.

(Solutions: ../solutions/week-06/day-6.md)

---

## Self-check

- [ ] I can run the full pipeline end-to-end and explain each stage
- [ ] I can read a loss curve and identify overfitting
- [ ] I can tune temperature and explain the output quality change
- [ ] I understand what checkpointing saves and why vocab must be saved too

---

## Supplementary

- [nanoGPT — full training script](https://github.com/karpathy/nanoGPT/blob/master/train.py)
- [Andrej Karpathy — Let's build GPT (YouTube, 2023)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Tiny Shakespeare dataset — char-rnn repo](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
