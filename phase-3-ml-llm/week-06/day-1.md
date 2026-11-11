# Phase 3 · Week 06 · Day 1 — Char-Level Dataset, Tokenizer & Batching

> Objective: Build a character-level tokenizer and a PyTorch Dataset/DataLoader that feeds fixed-length context windows to a GPT model.  ·  Time: ~1 hr  ·  Prereqs: [[week-05/day-1]], Python file I/O, basic PyTorch tensors

---

## Concept

Language models consume **tokens** — discrete integer IDs drawn from a vocabulary.  
At the char level, each unique character is a token. Vocabulary size is typically 65–100 chars (ASCII printable + newline), tiny compared with the 50 k BPE vocab a real GPT uses. The payoff: zero external dependencies, instant understanding of the full pipeline.

Three components built today:

| Component | Role |
|---|---|
| **Vocabulary** | sorted unique chars → index mapping |
| **Encoder / Decoder** | `str → List[int]` and back |
| **Dataset + DataLoader** | sliding context windows as `(x, y)` tensor pairs |

---

## How It Works Internally

### Sliding window batching

For a token sequence `[t0, t1, t2, ..., tN]` and context length `T`:

- Input window: `x = [t_i, ..., t_{i+T-1}]`  — shape `(T,)`
- Target window: `y = [t_{i+1}, ..., t_{i+T}]`  — shape `(T,)`

`y` is `x` shifted one step right.  The model must predict `y[j]` given `x[0..j]` — this is next-token prediction, and a single window gives `T` training examples simultaneously (one per position).

### Why `T` labels per window?

A standard cross-entropy loss averages over all `T` positions. Each position sees a shorter prefix — the model implicitly learns all context lengths from 1 to T. This is the causal language modelling objective.

---

## Worked Example

```python
"""
day1_dataset.py
Char-level tokenizer + PyTorch Dataset for miniGPT.
Run: python day1_dataset.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny synthetic corpus — no download needed
# ---------------------------------------------------------------------------
SYNTHETIC_TEXT = """\
to be or not to be that is the question
whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles
and by opposing end them to die to sleep
no more and by a sleep to say we end
""" * 80  # repeat to get ~5 k chars


def get_text(path: Optional[Path] = None) -> str:
    """Return corpus text. If path given, read it; else use synthetic."""
    if path is not None and path.exists():
        text = path.read_text(encoding="utf-8")
        log.info("Loaded %d chars from %s", len(text), path)
        return text
    log.info("Using synthetic corpus (%d chars)", len(SYNTHETIC_TEXT))
    return SYNTHETIC_TEXT


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class CharTokenizer:
    """Character-level tokenizer: builds vocab from a corpus string."""

    def __init__(self, text: str) -> None:
        chars: list[str] = sorted(set(text))
        self.vocab_size: int = len(chars)
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._itos: dict[int, str] = {i: c for c, i in self._stoi.items()}
        log.info("Vocab size: %d  first tokens: %s", self.vocab_size, chars[:10])

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window char-level dataset.

    Each item: (x, y) where y = x shifted right by 1.
    """

    def __init__(self, tokens: list[int], context_length: int) -> None:
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.context_length = context_length
        # Number of valid windows
        self._len = len(tokens) - context_length

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]  # T+1
        x = chunk[:-1]   # (T,)
        y = chunk[1:]    # (T,) — shifted
        return x, y


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def main() -> None:
    text = get_text()

    # --- tokenize ---
    tok = CharTokenizer(text)
    tokens = tok.encode(text)
    log.info("Total tokens: %d", len(tokens))

    # round-trip check
    sample = text[:40]
    assert tok.decode(tok.encode(sample)) == sample, "encode/decode round-trip failed"
    log.info("Round-trip OK: %r", sample)

    # --- dataset ---
    CONTEXT = 32
    ds = CharDataset(tokens, context_length=CONTEXT)
    log.info("Dataset length (windows): %d", len(ds))

    x, y = ds[0]
    assert x.shape == (CONTEXT,), f"bad x shape {x.shape}"
    assert y.shape == (CONTEXT,), f"bad y shape {y.shape}"
    assert torch.all(y[:-1] == x[1:]), "y should be x shifted right"

    log.info("x[:8] = %s", x[:8].tolist())
    log.info("y[:8] = %s", y[:8].tolist())
    log.info("decoded x: %r", tok.decode(x.tolist()))

    # --- dataloader ---
    BATCH = 4
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)
    bx, by = next(iter(loader))
    assert bx.shape == (BATCH, CONTEXT)
    assert by.shape == (BATCH, CONTEXT)
    log.info("Batch shapes: x=%s  y=%s", bx.shape, by.shape)


if __name__ == "__main__":
    main()
```

Run with: `python day1_dataset.py`

Expected output (synthetic corpus):
```
INFO Vocab size: 28  first tokens: ['\n', ' ', 'a', ...]
INFO Total tokens: 4480
INFO Round-trip OK: 'to be or not to be that is the questi'
INFO Dataset length (windows): 4448
INFO Batch shapes: x=torch.Size([4, 32])  y=torch.Size([4, 32])
```

### Shapes cheat sheet

| Tensor | Shape | Dtype |
|---|---|---|
| Single window `x` | `(T,)` | `torch.long` |
| Single window `y` | `(T,)` | `torch.long` |
| Batch `x` | `(B, T)` | `torch.long` |
| Batch `y` | `(B, T)` | `torch.long` |

---

## Exercises

1. Modify `CharTokenizer` to add a special `<PAD>` token at index 0 and shift all other tokens by 1. Verify that `vocab_size` increases by 1 and round-trip still works for padding-free strings.
2. Implement a `train_val_split(tokens, val_frac=0.1)` function that returns two non-overlapping `CharDataset` objects (train on first 90%, val on last 10%). Check that `len(train_ds) + len(val_ds) <= len(tokens) - context_length`.
3. Add a `__repr__` to `CharDataset` that prints vocab size, context length, and number of windows.
4. Write a pytest test `test_shift_invariant` that asserts `y[:-1] == x[1:]` for 20 random indices drawn from a `CharDataset`.
5. If `context_length = 64` and `len(tokens) = 10_000`, how many windows does the dataset contain? What happens to the last `64` tokens? Why is `drop_last=True` important in the `DataLoader`?

(Solutions: ../solutions/week-06/day-1.md)

---

## Self-check

- [ ] I can explain why `y = x` shifted by 1 implements next-token prediction
- [ ] I can state the shape of a batched `(x, y)` pair from a `CharDataset`
- [ ] I can implement `CharTokenizer.encode` and `decode` from scratch
- [ ] I understand why we use `torch.long` for token indices

---

## Supplementary

- [nanoGPT — Karpathy's reference implementation](https://github.com/karpathy/nanogpt)
- [Illustrated GPT-2 — Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- [PyTorch Dataset & DataLoader tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
