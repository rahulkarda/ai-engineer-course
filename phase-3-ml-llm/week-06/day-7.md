# Phase 3 · Week 06 · Day 7 — Week Project: Working MiniGPT

> Objective: Consolidate the week into a clean, tested, importable `minigpt` package with CLI train and generate commands.  ·  Time: ~3 hr project + 1 hr assessment  ·  Prereqs: [[week-06/day-1]] through [[week-06/day-6]]

---

## Project Spec

Build a self-contained `minigpt/` Python package that:

1. Exposes a clean public API: `MiniGPT`, `CharTokenizer`, `CharDataset`, `train`, `generate`
2. Provides a `python -m minigpt.train` CLI (corpus path or synthetic, config via args)
3. Provides a `python -m minigpt.generate` CLI (loads checkpoint, samples text)
4. Has pytest tests for: shape correctness, encode/decode round-trip, causal mask, generation smoke test
5. Trains in < 5 minutes on CPU with the default config

---

## Concept: Packaging & Separation of Concerns

Until now, everything was in single scripts. A production ML project separates:

| Module | Responsibility |
|---|---|
| `minigpt/config.py` | `GPTConfig` dataclass (single source of truth) |
| `minigpt/data.py` | `CharTokenizer`, `CharDataset`, `make_loaders` |
| `minigpt/model.py` | `MiniGPT`, `Block`, `CausalSelfAttention`, `FeedForward` |
| `minigpt/generate.py` | `generate()`, `sample_*` functions |
| `minigpt/train.py` | Training loop + `__main__` CLI |
| `tests/` | pytest suite |

This layout lets you import any piece independently and swap components without touching others.

---

## How It Works Internally: Weight Tying Revisited

`head.weight = tok_emb.weight` means a single `(V, D)` matrix serves both as:
- Input lookup: row `i` → embedding of token `i`
- Output projection: `(B, T, D) @ (D, V)` → logits

This saves `V × D` parameters and forces the model to learn a single geometric space where similar tokens cluster together in both input and output roles. GPT-2 uses weight tying; GPT-3 does not (large model, marginal benefit).

---

## Package Scaffold

```
minigpt/
├── __init__.py
├── config.py
├── data.py
├── model.py
├── generate.py
└── train.py
tests/
├── test_data.py
├── test_model.py
└── test_generate.py
```

### `minigpt/config.py`

```python
"""MiniGPT configuration."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GPTConfig:
    # Architecture
    vocab_size:   int   = 65       # set after tokenizer built
    context_len:  int   = 64
    d_model:      int   = 128
    n_heads:      int   = 4
    n_layers:     int   = 3
    dropout:      float = 0.1
    # Training
    batch_size:   int   = 16
    max_steps:    int   = 500
    eval_interval: int  = 50
    eval_steps:   int   = 20
    lr:           float = 3e-4
    weight_decay: float = 0.1
    grad_clip:    float = 1.0
    # Generation
    temperature:  float = 0.8
    top_k:        int   = 10
    # I/O
    checkpoint_path: Optional[Path] = None
```

### `minigpt/data.py`

```python
"""Char-level tokenizer and PyTorch dataset."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset, DataLoader
from .config import GPTConfig

log = logging.getLogger(__name__)

SYNTHETIC_CORPUS = (
    "to be or not to be that is the question\n"
    "whether tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune\n"
    "or to take arms against a sea of troubles\n"
    "and by opposing end them to die to sleep\n"
    "no more and by a sleep to say we end\n"
) * 200


class CharTokenizer:
    """Character-level tokenizer."""
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self._s: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._i: dict[int, str] = {i: c for c, i in self._s.items()}

    def encode(self, text: str) -> list[int]:
        return [self._s[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._i.get(i, "?") for i in ids)

    def state_dict(self) -> dict:
        return {"stoi": self._s, "itos": self._i, "vocab_size": self.vocab_size}

    @classmethod
    def from_state_dict(cls, d: dict) -> "CharTokenizer":
        obj = object.__new__(cls)
        obj._s = d["stoi"]
        obj._i = {int(k): v for k, v in d["itos"].items()}
        obj.vocab_size = d["vocab_size"]
        return obj


class CharDataset(Dataset):
    """Sliding-window char dataset returning (x, y) pairs."""
    def __init__(self, tokens: list[int], ctx: int) -> None:
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.ctx  = ctx

    def __len__(self) -> int:
        return len(self.data) - self.ctx

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.data[i : i + self.ctx + 1]
        return c[:-1], c[1:]


def load_corpus(path: Optional[Path] = None) -> str:
    if path is not None and path.exists():
        text = path.read_text(encoding="utf-8")
        log.info("Loaded %d chars from %s", len(text), path)
        return text
    log.info("Using synthetic corpus")
    return SYNTHETIC_CORPUS


def make_loaders(
    tokens: list[int], cfg: GPTConfig
) -> tuple[DataLoader, DataLoader]:
    split = int(len(tokens) * 0.9)
    tr = CharDataset(tokens[:split], cfg.context_len)
    vl = CharDataset(tokens[split:], cfg.context_len)
    return (
        DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,  drop_last=True),
        DataLoader(vl, batch_size=cfg.batch_size, shuffle=False, drop_last=True),
    )
```

### `minigpt/model.py`

```python
"""MiniGPT decoder-only transformer."""
from __future__ import annotations
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import GPTConfig

log = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.nh = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.dm = cfg.d_model
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model,     bias=False)
        self.drop   = nn.Dropout(cfg.dropout)
        mask = torch.triu(torch.ones(cfg.context_len, cfg.context_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.c_attn(x).split(self.dm, dim=2)
        r = lambda t: t.view(B, T, self.nh, self.dh).transpose(1, 2)
        q, k, v = r(q), r(k), r(v)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        s = s.masked_fill(self.mask[:T, :T], float("-inf"))
        o = (self.drop(F.softmax(s, dim=-1)) @ v)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.c_proj(o)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout),
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


class MiniGPT(nn.Module):
    """Decoder-only GPT, char-level, CPU-friendly."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying
        self._init_weights()
        log.info("MiniGPT: %d parameters", sum(p.numel() for p in self.parameters()))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        p = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(p))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size), targets.view(-1)
            )
        return logits, loss

    def configure_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for _, p in self.named_parameters():
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay,    "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.lr, betas=(0.9, 0.95),
        )
```

### `minigpt/generate.py`

```python
"""Text generation strategies."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import MiniGPT


def generate(
    model: MiniGPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int = 10,
) -> torch.Tensor:
    """Autoregressive generation with top-k + temperature.

    Args:
        model: trained MiniGPT (eval mode set internally)
        idx: seed token ids, shape (1, T_seed) or (B, T_seed)
        max_new_tokens: number of tokens to generate
        temperature: softmax temperature (0 → greedy)
        top_k: top-k filter width

    Returns:
        token ids, shape (B, T_seed + max_new_tokens)
    """
    model.eval()
    ctx = model.cfg.context_len
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -ctx:])
            last      = logits[:, -1, :]           # (B, V)
            if temperature <= 0:
                nxt = last.argmax(dim=-1, keepdim=True)
            else:
                scaled = last / temperature
                topv, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled   = scaled.masked_fill(scaled < topv[:, -1:], float("-inf"))
                probs    = F.softmax(scaled, dim=-1)
                nxt      = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
    return idx
```

### `minigpt/train.py` (CLI)

```python
"""Training CLI: python -m minigpt.train [--data PATH] [--steps N] [--out PATH]"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import GPTConfig
from .data import CharTokenizer, load_corpus, make_loaders
from .model import MiniGPT
from .generate import generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@torch.no_grad()
def _eval(model: MiniGPT, loader: DataLoader, steps: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= steps: break
        _, loss = model(x, y)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)


def run(cfg: GPTConfig, corpus_path: Path | None = None) -> MiniGPT:
    text = load_corpus(corpus_path)
    tok  = CharTokenizer(text)
    cfg.vocab_size = tok.vocab_size

    train_loader, val_loader = make_loaders(tok.encode(text), cfg)
    model = MiniGPT(cfg)
    opt   = model.configure_optimizer()

    it = iter(train_loader)
    t0 = time.time()
    for step in range(cfg.max_steps):
        try: x, y = next(it)
        except StopIteration:
            it = iter(train_loader); x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            tl = _eval(model, train_loader, cfg.eval_steps)
            vl = _eval(model, val_loader,   cfg.eval_steps)
            log.info("step %4d | train %.4f | val %.4f | %.0f s",
                     step, tl, vl, time.time() - t0)

    if cfg.checkpoint_path:
        cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "config": cfg,
                    "tokenizer": tok.state_dict()}, cfg.checkpoint_path)
        log.info("Checkpoint saved → %s", cfg.checkpoint_path)
    return model, tok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data",  type=Path,  default=None)
    p.add_argument("--steps", type=int,   default=500)
    p.add_argument("--out",   type=Path,  default=Path("checkpoint.pt"))
    p.add_argument("--prompt", type=str,  default="to be or not")
    args = p.parse_args()

    cfg = GPTConfig(max_steps=args.steps, checkpoint_path=args.out)
    model, tok = run(cfg, corpus_path=args.data)

    # Quick generation demo
    seed = tok.encode(args.prompt)
    idx  = torch.tensor([seed], dtype=torch.long)
    ids  = generate(model, idx, max_new_tokens=200, temperature=0.8, top_k=10)
    print("\n--- Generated ---")
    print(tok.decode(ids[0].tolist()))


if __name__ == "__main__":
    main()
```

---

## Tests Scaffold

```python
# tests/test_data.py
import torch, pytest
from minigpt.data import CharTokenizer, CharDataset

def test_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello world")) == "hello world"

def test_dataset_shapes():
    tok = CharTokenizer("abcdefghijklmnop" * 20)
    ds  = CharDataset(tok.encode("abcdefghijklmnop" * 20), ctx=8)
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)

def test_shift():
    tok = CharTokenizer("abcde" * 30)
    ds  = CharDataset(tok.encode("abcde" * 30), ctx=4)
    x, y = ds[5]
    assert torch.all(y[:-1] == x[1:])
```

```python
# tests/test_model.py
import torch, pytest
from minigpt.config import GPTConfig
from minigpt.model import MiniGPT

CFG = GPTConfig(vocab_size=28, context_len=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)

def test_output_shape():
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.context_len))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, CFG.context_len, CFG.vocab_size)
    assert loss is not None

def test_loss_is_scalar():
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.context_len))
    _, loss = m(idx, targets=idx)
    assert loss.shape == ()

def test_causal_mask():
    """Changing last token must not affect output at position 0."""
    m = MiniGPT(CFG); m.eval()
    x  = torch.randint(0, CFG.vocab_size, (1, CFG.context_len))
    x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size
    with torch.no_grad():
        o1, _ = m(x)
        o2, _ = m(x2)
    assert torch.allclose(o1[0, 0], o2[0, 0], atol=1e-5), "Causality violated"

def test_gradient_flows():
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.context_len))
    _, loss = m(idx, targets=idx)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"No grad: {name}"
```

```python
# tests/test_generate.py
import torch, pytest
from minigpt.config import GPTConfig
from minigpt.model import MiniGPT
from minigpt.generate import generate

CFG = GPTConfig(vocab_size=28, context_len=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)

def test_generate_length():
    m   = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (1, 4))
    out = generate(m, idx, max_new_tokens=10)
    assert out.shape[1] == 4 + 10

def test_greedy_deterministic():
    m   = MiniGPT(CFG); m.eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 4))
    o1  = generate(m, idx.clone(), max_new_tokens=8, temperature=0.0)
    o2  = generate(m, idx.clone(), max_new_tokens=8, temperature=0.0)
    assert torch.equal(o1, o2), "Greedy generation should be deterministic"

def test_smoke():
    """Train 5 steps, generate 20 tokens — no crash."""
    import torch.nn.functional as F
    from minigpt.data import CharDataset, CharTokenizer
    from torch.utils.data import DataLoader
    text = "hello world " * 50
    tok  = CharTokenizer(text)
    cfg  = GPTConfig(vocab_size=tok.vocab_size, context_len=8,
                     d_model=32, n_heads=2, n_layers=1, dropout=0.0)
    m   = MiniGPT(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ds  = CharDataset(tok.encode(text), ctx=8)
    dl  = DataLoader(ds, batch_size=4, drop_last=True)
    for x, y in dl:
        _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
        break
    seed = torch.tensor([tok.encode("hello")[:4]], dtype=torch.long)
    out  = generate(m, seed, max_new_tokens=20)
    assert out.shape[1] == 24
```

---

## Running the Full Project

```bash
# Create package (from project root)
# Assumes minigpt/ and tests/ directories are created as above

# Run tests
uv run pytest tests/ -v

# Train on synthetic corpus (fast, ~2 min CPU)
uv run python -m minigpt.train --steps 500 --out checkpoint.pt

# Train on tiny shakespeare (download first)
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
uv run python -m minigpt.train --data input.txt --steps 1000 --out shakespeare.pt

# Generate from checkpoint
uv run python -m minigpt.train --data input.txt --prompt "ROMEO:"
```

---

## Sunday Assessment

See [[assessment.md]] for the week assessment. After completing the project, attempt the assessment to benchmark your understanding.

---

## Self-check

- [ ] I can build a full decoder-only GPT from scratch without reference
- [ ] I can explain every component: tokenizer, embeddings, blocks, head, loss
- [ ] My tests cover shape, causality, gradient flow, and generation
- [ ] I can load a checkpoint and generate text from a new prompt
- [ ] I understand the train/val loss dynamics on a tiny corpus

---

## Supplementary

- [nanoGPT — Karpathy](https://github.com/karpathy/nanoGPT)
- [Andrej Karpathy — Let's build GPT from scratch (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [The Illustrated GPT-2 — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- [Language Models are Few-Shot Learners (GPT-3 paper)](https://arxiv.org/abs/2005.14165)
