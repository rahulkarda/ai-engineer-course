# Solutions — Week 06 · Project — MiniGPT: Working Char-Level Language Model

## Complete Reference Implementation

The complete module code is embedded in `week-06/day-7.md` (the project spec day). Copy each
module from there into the corresponding file under `minigpt/`. This solutions file adds the
`minigpt/__init__.py`, a complete `pyproject.toml`, and the `.env.example` / `.gitignore`.

---

### `minigpt/__init__.py`

```python
"""minigpt — minimal char-level GPT package.

Public API:
    MiniGPT, CharTokenizer, GPTConfig, generate
"""
from .config import GPTConfig
from .data import CharTokenizer, CharDataset, load_corpus, make_loaders
from .model import MiniGPT
from .generate import generate

__all__ = [
    "GPTConfig",
    "CharTokenizer",
    "CharDataset",
    "load_corpus",
    "make_loaders",
    "MiniGPT",
    "generate",
]
```

---

### `minigpt/generate.py` — `__main__` entry point

```python
"""Generation CLI: python -m minigpt.generate --checkpoint PATH [--prompt TEXT]"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from .config import GPTConfig
from .data import CharTokenizer
from .model import MiniGPT
from .generate import generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--prompt", type=str, default="to be or not")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=10)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg: GPTConfig = ckpt["config"]
    tok = CharTokenizer.from_state_dict(ckpt["tokenizer"])

    model = MiniGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Loaded checkpoint from %s", args.checkpoint)

    seed_ids = tok.encode(args.prompt)
    idx = torch.tensor([seed_ids], dtype=torch.long)
    out_ids = generate(
        model, idx,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\n--- Generated ---")
    print(tok.decode(out_ids[0].tolist()))


if __name__ == "__main__":
    main()
```

---

### `pyproject.toml`

```toml
[project]
name = "minigpt"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"
```

---

### `.env.example`

```
# Copy to .env and fill in values — never commit .env
# No API key needed for minigpt (fully local training)
MINIGPT_DATA_PATH=          # optional: path to .txt corpus
MINIGPT_CHECKPOINT=checkpoint.pt
```

---

### `.gitignore`

```
.env
__pycache__/
*.pyc
*.pt
*.pth
checkpoint*.pt
shakespeare.pt
input.txt
.venv/
dist/
*.egg-info/
```

---

## Full test suite

All tests from `week-06/day-7.md` are the reference tests. Here is the complete consolidated
suite with the smoke test from `test_generate.py` expanded to exercise `__main__` CLI:

```python
# tests/test_data.py
"""Tests for CharTokenizer and CharDataset."""
import torch
import pytest
from minigpt.data import CharTokenizer, CharDataset


def test_roundtrip() -> None:
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello world")) == "hello world"


def test_dataset_shapes() -> None:
    text = "abcdefghijklmnop" * 20
    tok = CharTokenizer(text)
    ds = CharDataset(tok.encode(text), ctx=8)
    x, y = ds[0]
    assert x.shape == (8,) and y.shape == (8,)


def test_shift() -> None:
    """y should be x shifted left by one position."""
    tok = CharTokenizer("abcde" * 30)
    ds = CharDataset(tok.encode("abcde" * 30), ctx=4)
    x, y = ds[5]
    assert torch.all(y[:-1] == x[1:])


def test_tokenizer_state_dict_roundtrip() -> None:
    tok = CharTokenizer("hello world foo bar")
    state = tok.state_dict()
    tok2 = CharTokenizer.from_state_dict(state)
    assert tok.encode("hello") == tok2.encode("hello")
    assert tok.decode(tok.encode("world")) == "world"
```

```python
# tests/test_model.py
"""Tests for MiniGPT model."""
import torch
import pytest
from minigpt.config import GPTConfig
from minigpt.model import MiniGPT

CFG = GPTConfig(
    vocab_size=28, context_len=16, d_model=32,
    n_heads=4, n_layers=2, dropout=0.0
)


def test_output_shape() -> None:
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.context_len))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, CFG.context_len, CFG.vocab_size)
    assert loss is not None


def test_loss_is_scalar() -> None:
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.context_len))
    _, loss = m(idx, targets=idx)
    assert loss is not None and loss.shape == ()


def test_causal_mask() -> None:
    """Changing the last token must NOT affect output at position 0."""
    m = MiniGPT(CFG)
    m.eval()
    x = torch.randint(0, CFG.vocab_size, (1, CFG.context_len))
    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % CFG.vocab_size
    with torch.no_grad():
        o1, _ = m(x)
        o2, _ = m(x2)
    assert torch.allclose(o1[0, 0], o2[0, 0], atol=1e-5), "Causality violated"


def test_gradient_flows() -> None:
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.context_len))
    _, loss = m(idx, targets=idx)
    assert loss is not None
    loss.backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad: {name}"


def test_weight_tying() -> None:
    m = MiniGPT(CFG)
    assert m.head.weight is m.tok_emb.weight, "Weight tying not active"
```

```python
# tests/test_generate.py
"""Tests for text generation."""
import torch
import pytest
from minigpt.config import GPTConfig
from minigpt.model import MiniGPT
from minigpt.generate import generate

CFG = GPTConfig(
    vocab_size=28, context_len=16, d_model=32,
    n_heads=4, n_layers=2, dropout=0.0
)


def test_generate_length() -> None:
    m = MiniGPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (1, 4))
    out = generate(m, idx, max_new_tokens=10)
    assert out.shape[1] == 4 + 10


def test_greedy_deterministic() -> None:
    m = MiniGPT(CFG)
    m.eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 4))
    o1 = generate(m, idx.clone(), max_new_tokens=8, temperature=0.0)
    o2 = generate(m, idx.clone(), max_new_tokens=8, temperature=0.0)
    assert torch.equal(o1, o2), "Greedy generation should be deterministic"


def test_smoke() -> None:
    """Train 1 batch, generate 20 tokens — no crash."""
    from minigpt.data import CharDataset, CharTokenizer
    from torch.utils.data import DataLoader
    text = "hello world " * 50
    tok = CharTokenizer(text)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, context_len=8,
        d_model=32, n_heads=2, n_layers=1, dropout=0.0
    )
    m = MiniGPT(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    ds = CharDataset(tok.encode(text), ctx=8)
    dl = DataLoader(ds, batch_size=4, drop_last=True)
    x, y = next(iter(dl))
    _, loss = m(x, y)
    assert loss is not None
    opt.zero_grad()
    loss.backward()
    opt.step()
    seed = torch.tensor([tok.encode("hello")[:4]], dtype=torch.long)
    out = generate(m, seed, max_new_tokens=20)
    assert out.shape[1] == 24
```

---

## Acceptance criteria verification

```bash
# Install
uv pip install -e ".[dev]"

# Tests
uv run pytest tests/ -v

# Train (fast, ~90 s CPU)
uv run python -m minigpt.train --steps 500 --out checkpoint.pt

# Generate
uv run python -m minigpt.generate --checkpoint checkpoint.pt --prompt "to be"

# Lint
ruff check minigpt/ tests/

# No print in library code
grep -r "^print" minigpt/

# Weight tying check
python -c "from minigpt import MiniGPT, GPTConfig; m = MiniGPT(GPTConfig()); print(m.head.weight is m.tok_emb.weight)"
```
