# Phase 3 · Week 05 · Day 4 — Positional Encodings

> Objective: Understand why position information must be injected and implement sinusoidal and learned positional encodings.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

## Concept

### Attention is permutation-invariant

Attention with no modifications treats its input as a **set**, not a sequence. Swap the order of tokens and the output changes only in the corresponding output positions — the model cannot distinguish "dog bites man" from "man bites dog" at the attention level.

Proof: if you permute the rows of Q, K, V by permutation matrix P:
- Scores: `(PQ)(PK)ᵀ = P(QKᵀ)Pᵀ` — reordered rows and columns.
- After softmax and multiply by PV: output rows are just reordered.
- Result: output is a permuted version of the original output.

The model sees the same content, just reordered — positions are invisible.

### Solution: add position information to token embeddings

Inject a position-dependent signal into the input before the first attention layer:

```
x_t = embedding(token_t) + positional_encoding(t)
```

Two main approaches:

**1. Sinusoidal (Vaswani et al. 2017)**
Deterministic formula, no learnable parameters:

```
PE(t, 2i)   = sin(t / 10000^(2i/d_model))
PE(t, 2i+1) = cos(t / 10000^(2i/d_model))
```

Properties:
- Each position has a unique encoding.
- Neighbouring positions have similar encodings (smooth).
- Relative positions can be represented: PE(t+k) is a linear function of PE(t) for fixed k.
- Generalises to positions not seen during training.

**2. Learned absolute (BERT, GPT)**
`nn.Embedding(max_seq_len, d_model)` — learnable table indexed by position. Simpler but cannot extrapolate beyond `max_seq_len`.

**3. Relative / RoPE (modern LLMs — covered in advanced weeks)**
Encode relative distances rather than absolute positions. Better length generalisation. Used in LLaMA, GPT-NeoX.

## How it works internally

### Why sinusoidal works

The frequencies span a range from 1 cycle per position (fast oscillation, captures local position) to 1 cycle per 10000 positions (slow oscillation, captures global position). This is analogous to a Fourier decomposition of position.

The 10000 base gives a usable encoding up to ~10000 positions. For d_model=512 there are 256 sin/cos pairs, each with a different frequency.

After addition to the token embedding:
- The model can attend to "position 3" by learning a Q/K that fires on the PE(3) component.
- Relative attention patterns emerge naturally from the linear transformation properties.

### Why learned works well in practice

For a fixed maximum context length (e.g. 512 for BERT), the learned table simply memorises good encodings via gradient descent. It often outperforms sinusoidal on tasks where the training distribution covers the full context length. Disadvantage: hard ceiling at `max_seq_len`.

## Worked example

```python
"""
day4_positional_encoding.py
Implements sinusoidal and learned positional encodings.
Run: python day4_positional_encoding.py
"""
import logging
import math

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention Is All You Need'.
    Adds PE to the input embedding and optionally applies dropout.

    Args:
        d_model:  Embedding dimension.
        max_len:  Maximum sequence length to pre-compute.
        dropout:  Dropout probability.
    """

    def __init__(
        self, d_model: int, max_len: int = 512, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute PE table: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Frequencies: 1 / 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        # Register as buffer (saved in state_dict, not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with PE added
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]  # type: ignore[index]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional encoding (as in BERT/GPT-2).

    Args:
        d_model:      Embedding dimension.
        max_seq_len:  Hard ceiling on sequence length.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with PE added
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        return x + self.embedding(positions)


def verify_shapes() -> None:
    """Shape and buffer tests."""
    torch.manual_seed(0)
    B, T, d = 2, 20, 64

    spe = SinusoidalPositionalEncoding(d_model=d, max_len=100)
    lpe = LearnedPositionalEncoding(d_model=d, max_seq_len=100)

    x = torch.randn(B, T, d)

    out_s = spe(x)
    out_l = lpe(x)

    assert out_s.shape == (B, T, d), f"sinusoidal shape: {out_s.shape}"
    assert out_l.shape == (B, T, d), f"learned shape: {out_l.shape}"

    # Sinusoidal PE is deterministic — same call = same result
    spe2 = SinusoidalPositionalEncoding(d_model=d, max_len=100)
    assert torch.allclose(spe.pe, spe2.pe), "Sinusoidal PE should be deterministic"  # type: ignore[attr-defined]

    log.info("Shape tests PASSED")


def visualise_pe_properties() -> None:
    """Show smoothness and uniqueness of sinusoidal PE."""
    d = 16
    max_len = 50
    spe = SinusoidalPositionalEncoding(d_model=d, max_len=max_len)
    pe = spe.pe.squeeze(0)  # (max_len, d)

    # Cosine similarity between adjacent positions
    similarities: list[float] = []
    for t in range(max_len - 1):
        cos_sim = torch.nn.functional.cosine_similarity(
            pe[t].unsqueeze(0), pe[t + 1].unsqueeze(0)
        ).item()
        similarities.append(cos_sim)

    avg_sim = sum(similarities) / len(similarities)
    log.info(
        "Average cosine similarity between adjacent PE vectors: %.4f"
        " (high → smooth)", avg_sim
    )

    # Verify all positions are unique
    # Pairwise L2 distance — min non-zero distance should be > 0
    dists = torch.cdist(pe, pe)
    dists.fill_diagonal_(float("inf"))
    min_dist = dists.min().item()
    log.info("Min L2 distance between distinct PE vectors: %.6f (> 0 → unique)", min_dist)


if __name__ == "__main__":
    verify_shapes()
    log.info("\n--- PE Properties ---")
    visualise_pe_properties()
```

## Exercises

1. Plot (using ASCII or by printing values) the first 8 dimensions of `SinusoidalPositionalEncoding` for positions 0–31. Describe the pattern you observe.

2. Verify the "linear relation" claim: for d=4, show numerically that `PE(t+k, :)` can be expressed as `M · PE(t, :)` for some matrix `M` (independent of `t`). What is `M` for k=1?

3. Implement `RelativePositionBias`, a simplified version: for each pair `(i, j)`, output a scalar bias `clip(j - i, -max_dist, max_dist)` (learnable embedding indexed by clipped distance). This scalar is added to attention scores before softmax. Write shape tests.

4. What happens to `LearnedPositionalEncoding` when you pass a sequence of length `T > max_seq_len` at inference time? How would you handle this gracefully?

(Solutions: ../solutions/week-05/day-4.md)

## Self-check

- [ ] I can explain why attention is permutation-invariant and why that's a problem.
- [ ] I can derive the sinusoidal PE formula and explain the role of the 10000 base.
- [ ] I understand the trade-offs between sinusoidal and learned positional encodings.
- [ ] I can implement `SinusoidalPositionalEncoding` and `LearnedPositionalEncoding` from scratch.

## Supplementary

- Vaswani et al. (2017) Section 3.5 — https://arxiv.org/abs/1706.03762
- Press et al. (2022) "Train Short, Test Long: ALiBi" — https://arxiv.org/abs/2108.12409
- Su et al. (2023) "RoFormer: Enhanced Transformer with Rotary Position Embedding" — https://arxiv.org/abs/2104.09864
