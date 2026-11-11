# Solutions — Week 06 Assessment

## Conceptual answers

**Q1.** Two sub-layers: (1) Multi-head causal self-attention — lets tokens gather contextual information from earlier tokens. (2) Position-wise feed-forward network — applies a non-linear transformation independently to each token's representation. Both are wrapped in residual connections so the original signal passes through unchanged; the sub-layer only needs to learn a *correction*, making gradients flow more easily through deep stacks and preventing vanishing gradients.

**Q2.** The causal mask is an upper-triangular matrix of `-inf` values added to attention scores *before* softmax. It zeros out (via exp(-inf)=0) all attention weights from token `t` to tokens `>t`. Forgetting it allows every token to attend to future tokens — a data leak at training time that makes the task trivially easy and produces a model useless for autoregressive generation.

**Q3.** The head linear layer `(d_model, vocab_size)` shares weights with the token embedding matrix `(vocab_size, d_model)` (stored as its transpose). Parameter saving: `50_257 × 768 = 38,597,376` — nearly 39 M parameters freed, which is ~15% of GPT-2 small's total.

**Q4.** Crop the context to the last `context_len` tokens at each generation step: `idx[:, -context_len:]`. The model never sees tokens that were generated more than `context_len` steps ago. This means early tokens of the prompt are forgotten once the generation grows beyond `context_len`.

**Q5.**
- Greedy: deterministic; always picks argmax. Preferred when reproducibility is required or for diagnostic runs.
- Temperature: non-deterministic; scales logits. Preferred for creative generation where some diversity is desired.
- Top-k: non-deterministic; restricts to top-k candidates before sampling. Preferred when you want diversity but want to avoid extremely low-probability tokens.

**Q6.** A random model assigns equal probability to all `V` tokens: `p = 1/V`. Cross-entropy = `-log(1/V) = log(V)`. For V=28: `log(28) ≈ 3.332` nats. Any trained model should achieve loss below this.

**Q7.** Post-LN (`LN(x + sub(x))`) causes gradients to vanish in deep networks because the LN normalises after adding the residual, shrinking gradient magnitude. Pre-LN (`x + sub(LN(x))`) leaves the residual stream unnormalised, giving a direct gradient path from the loss to early layers. This allows training without learning rate warmup and supports larger learning rates.

**Q8.** The 4× expansion creates a wide bottleneck where token representations are projected into a high-dimensional space before being compressed back. This gives the model extra representational capacity per token (a "memory" of patterns). Using 1× (no expansion) removes this capacity; empirically, models with no expansion underfit and converge to worse loss at equivalent parameter budgets.

---

## Coding Challenge — `BPETokenizer` stub

```python
"""bpe_tokenizer.py — BPETokenizer stub with char-level fallback."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class BPETokenizer:
    """BPE tokenizer stub. Falls back to char-level when vocab_size <= 128."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    def _build_vocab_if_needed(self, text: str) -> None:
        """Lazily build char-level vocab from text."""
        if self._stoi:
            return
        chars = sorted(set(text))
        self._stoi = {c: i for i, c in enumerate(chars)}
        self._itos = {i: c for c, i in self._stoi.items()}
        log.info("Built char vocab of size %d", len(self._stoi))

    def encode(self, text: str) -> list[int]:
        self._build_vocab_if_needed(text)
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos.get(i, "?") for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"stoi": self._stoi, "vocab_size": self.vocab_size}),
            encoding="utf-8",
        )
        log.info("Tokenizer saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        obj = cls(vocab_size=data["vocab_size"])
        obj._stoi = data["stoi"]
        obj._itos = {int(v): k for k, v in data["stoi"].items()}
        return obj
```

Tests:
```python
def test_bpe_char_fallback():
    tok = BPETokenizer(vocab_size=64)
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text

def test_bpe_save_load(tmp_path):
    tok = BPETokenizer(vocab_size=64)
    tok.encode("hello world")
    tok.save(tmp_path / "vocab.json")
    tok2 = BPETokenizer.load(tmp_path / "vocab.json")
    assert tok2.decode(tok2.encode("hello world")) == "hello world"
```

---

## Debug section — answers

**Bug A** — Wrong reshape for cross_entropy:
```python
# Bug: logits is (B, T, V), targets is (B, T)
# cross_entropy expects (N, C) and (N,)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

**Bug B** — Mask polarity inverted:
```python
# Bug: torch.tril gives the LOWER triangle (positions we WANT to attend to)
# masked_fill(mask, -inf) hides those positions → only future is visible
# Fix: use upper triangle for masking
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float("-inf"))
```

**Bug C** — Fixed-length position vector:
```python
# Bug: pos always has length context_len even if idx has fewer tokens
# This causes pos_emb to return wrong-length embeddings when T < context_len
# Fix: use the actual sequence length T
B, T = idx.shape
pos = torch.arange(T, device=idx.device)   # (T,) not (context_len,)
x   = tok_emb(idx) + pos_emb(pos)
```
