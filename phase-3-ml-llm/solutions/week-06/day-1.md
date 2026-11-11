# Solutions — Week 06 · Day 1

## Exercise 1 — `<PAD>` token

```python
class CharTokenizerWithPad:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        # Insert PAD at index 0, shift all others by 1
        self._pad_id = 0
        self._stoi: dict[str, int] = {c: i + 1 for i, c in enumerate(chars)}
        self._stoi["<PAD>"] = 0
        self._itos: dict[int, str] = {v: k for k, v in self._stoi.items()}
        self.vocab_size: int = len(self._stoi)   # +1 for PAD

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids if i != self._pad_id)
```

Verification:
```python
tok = CharTokenizerWithPad("hello")
assert tok.vocab_size == len(set("hello")) + 1
assert tok.decode(tok.encode("hello")) == "hello"
```

---

## Exercise 2 — `train_val_split`

```python
def train_val_split(
    tokens: list[int], context_length: int, val_frac: float = 0.1
) -> tuple[CharDataset, CharDataset]:
    split = int(len(tokens) * (1 - val_frac))
    train_ds = CharDataset(tokens[:split],  context_length)
    val_ds   = CharDataset(tokens[split:],  context_length)
    # Sanity: no overlap, counts consistent
    assert len(train_ds) + len(val_ds) <= len(tokens) - context_length
    return train_ds, val_ds
```

---

## Exercise 3 — `__repr__`

```python
class CharDataset(Dataset):
    # ... (existing code) ...

    def __repr__(self) -> str:
        return (
            f"CharDataset(context_len={self.ctx}, "
            f"windows={len(self)}, "
            f"tokens={len(self.data)})"
        )
```

---

## Exercise 4 — `test_shift_invariant`

```python
import pytest, torch
from minigpt.data import CharTokenizer, CharDataset

def test_shift_invariant():
    text = "abcdefghij" * 50
    tok  = CharTokenizer(text)
    ds   = CharDataset(tok.encode(text), ctx=8)
    rng  = torch.randint(0, len(ds), (20,))
    for idx in rng.tolist():
        x, y = ds[idx]
        assert torch.all(y[:-1] == x[1:]), f"Shift violated at index {idx}"
```

---

## Exercise 5 — Window count analysis

With `context_length=64` and `len(tokens)=10_000`:

- Number of windows = `10_000 - 64 = 9_936`
- The last 64 tokens appear **only as targets** in the final window starting at index 9,935. They cannot form a complete input window themselves (no 65 tokens remain).
- `drop_last=True` in `DataLoader` discards the final incomplete batch so every batch has exactly `batch_size` windows. Without it, the last batch may be smaller, which can cause shape mismatches in batch-norm layers or logging inconsistencies.
