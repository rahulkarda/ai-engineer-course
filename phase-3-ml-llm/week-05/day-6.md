# Phase 3 · Week 05 · Day 6 — Saturday Deep Practice: Build Attention & MHA from Scratch

> Objective: Consolidate the week's implementation skills by building a complete, tested attention library from scratch without reference.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Overview

Today is a closed-book build session. You will implement a small but complete `attention` library covering everything from this week: scaled dot-product attention, multi-head attention, positional encodings, and a full transformer block — all with comprehensive tests.

Target: a module you can import in next week's project without changes.

### Session structure

| Block | Time | Goal |
|---|---|---|
| Warm-up | 30 min | Recall formulas, sketch architecture on paper |
| Core build | 90 min | Implement without looking at lesson files |
| Testing | 45 min | pytest suite with ≥10 tests |
| Integration | 45 min | Assemble full model, run forward pass |
| Review | 30 min | Compare with lessons, note gaps |

## Part 1 — Warm-up: write from memory

Before touching code, write on paper (or a scratch file):

1. The scaled dot-product attention formula.
2. The multi-head attention formula including all projection matrices.
3. The sinusoidal PE formula.
4. The transformer block equations (Pre-LN variant).
5. The causal mask: what matrix shape, what values.

Only after writing all five should you open the editor.

## Part 2 — Implementation spec

Implement the following in `attention_lib.py`. The spec is the interface contract — implement it cleanly.

```python
"""attention_lib.py — public interface (implement this)"""
import torch
import torch.nn as nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (output, weights).
    output:  (..., n, d_v)
    weights: (..., n, m)  — sums to 1 over m
    """
    ...


class MultiHeadAttention(nn.Module):
    """
    d_model divisible by num_heads.
    forward(query, key, value, mask=None) -> (output, weights)
    output: (B, n, d_model), weights: (B, num_heads, n, m)
    """
    ...


class SinusoidalPositionalEncoding(nn.Module):
    """
    forward(x: Tensor (B, T, d_model)) -> Tensor (B, T, d_model)
    PE is deterministic, not a learnable parameter.
    """
    ...


class FeedForward(nn.Module):
    """
    Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)
    """
    ...


class TransformerBlock(nn.Module):
    """
    Pre-LN variant.
    forward(x, mask=None) -> x  same shape
    """
    ...


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Upper-triangular bool mask (T, T). True = blocked."""
    ...
```

## Part 3 — Test suite spec

Write `tests/test_attention_lib.py`. Must include:

```python
# Shape tests
def test_sdpa_output_shape(): ...
def test_sdpa_weights_sum_to_one(): ...
def test_mha_output_shape(): ...
def test_mha_weights_shape(): ...

# Masking tests
def test_sdpa_causal_mask_zeros_upper_triangle(): ...
def test_causal_mask_shape(): ...
def test_causal_mask_diagonal_not_masked(): ...

# PE tests
def test_sinusoidal_pe_deterministic(): ...
def test_sinusoidal_pe_shape(): ...
def test_sinusoidal_pe_all_positions_unique(): ...

# Block test
def test_transformer_block_shape(): ...
def test_transformer_block_residual_identity_at_zero_init(): ...
```

Run all tests: `python -m pytest tests/test_attention_lib.py -v`

## Part 4 — Integration: tiny language model forward pass

Using only `attention_lib.py`, build and run:

```python
"""day6_integration.py"""
import logging
import torch
import torch.nn as nn
from attention_lib import (
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    causal_mask,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class TinyGPT(nn.Module):
    """3-layer decoder-only transformer, sinusoidal PE, vocab=256."""

    def __init__(self) -> None:
        super().__init__()
        self.d_model = 128
        self.vocab_size = 256
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pe = SinusoidalPositionalEncoding(self.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=self.d_model, num_heads=4, d_ff=512)
             for _ in range(3)]
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        x = self.pe(self.embedding(tokens))
        mask = causal_mask(T).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyGPT()
    tokens = torch.randint(0, 256, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, 256)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("TinyGPT forward pass OK. Logits: %s. Params: %d", logits.shape, n_params)

    # Greedy decode one token
    next_tok = logits[0, -1].argmax().item()
    log.info("Greedy next token for sequence 0: %d", next_tok)
```

## Part 5 — Review & reflection

After completing the build:

1. Compare each implementation with the lesson files. Note any formula you got wrong or any shape error you hit.
2. Write a 5-bullet summary of the most common mistakes in this domain (your own hits + what to watch for).
3. Which component was hardest to remember precisely? Why?

## Exercises (build checkpoints)

1. After implementing `scaled_dot_product_attention`, run the shape tests before moving on. Do not proceed with MHA until all SDPA tests pass.

2. After implementing `MultiHeadAttention`, add a `test_mha_equals_pytorch` test (as in Day 3) to cross-validate your implementation.

3. Add a `test_forward_pass_is_differentiable` test: call `.backward()` on the sum of logits from `TinyGPT` and verify that all parameter `.grad` attributes are non-None.

4. Measure memory: use `torch.cuda.memory_allocated()` or `tracemalloc` (CPU) to compare attention memory usage for T=16 vs T=64 vs T=128. Does it scale as O(T²)? Report your measurements.

(Solutions: ../solutions/week-05/day-6.md)

## Self-check

- [ ] I implemented all six components from memory without referencing lesson files during the build phase.
- [ ] All 12+ tests pass.
- [ ] `TinyGPT` forward pass runs without error.
- [ ] I can articulate at least three things I got wrong on first attempt.
