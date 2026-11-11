# Week 05 Project — Implement Scaled Dot-Product Attention and Multi-Head Attention from Scratch

## Overview

Build a clean, tested attention library in PyTorch that implements the core components of a transformer from first principles. The deliverable is a module importable by future weeks' projects.

**Time budget:** Saturday (~4 hr) + Sunday morning (~90 min polish)

## Specification

### Files to deliver

```
week-05/
  attention_lib.py       # main implementation
  day6_integration.py    # TinyGPT integration test (from Day 6)
  tests/
    test_attention_lib.py
```

### `attention_lib.py` — required components

#### 1. `scaled_dot_product_attention`

```python
def scaled_dot_product_attention(
    q: torch.Tensor,      # (..., n, d_k)
    k: torch.Tensor,      # (..., m, d_k)
    v: torch.Tensor,      # (..., m, d_v)
    mask: torch.Tensor | None = None,  # (..., n, m) bool, True = block
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    # Returns: output (..., n, d_v), weights (..., n, m)
```

Acceptance criteria:
- Output shape correct for arbitrary leading batch dimensions.
- Weights sum to 1 along key dimension (after masking).
- Masked positions have weight 0.

#### 2. `MultiHeadAttention(nn.Module)`

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None: ...
    def forward(
        self,
        query: torch.Tensor,  # (B, n, d_model)
        key: torch.Tensor,    # (B, m, d_model)
        value: torch.Tensor,  # (B, m, d_model)
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...
        # Returns: output (B, n, d_model), weights (B, num_heads, n, m)
```

Acceptance criteria:
- Uses `scaled_dot_product_attention` internally.
- Efficient: single matmul over all heads (no Python loop).
- Cross-validates output against `nn.MultiheadAttention` (after weight copy) within atol=1e-4.

#### 3. `SinusoidalPositionalEncoding(nn.Module)`

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    # x: (B, T, d_model) → (B, T, d_model)
```

Acceptance criteria:
- PE is stored as a buffer (not a parameter).
- Two instances with the same args produce identical PE.
- All T position vectors are unique.

#### 4. `FeedForward(nn.Module)`

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    # x: (B, T, d_model) → (B, T, d_model)
```

Acceptance criteria:
- Uses GELU activation.
- Inner dimension is `d_ff`, output dimension is `d_model`.
- Applied position-wise (same network at each position).

#### 5. `TransformerBlock(nn.Module)`

```python
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None: ...
    def forward(
        self,
        x: torch.Tensor,       # (B, T, d_model)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...     # (B, T, d_model)
```

Acceptance criteria:
- Pre-LN variant.
- Two residual connections (one per sub-layer).
- Output shape unchanged from input.

#### 6. `causal_mask`

```python
def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    ...
    # Returns: (seq_len, seq_len) bool, True = future position (j > i)
```

Acceptance criteria:
- Upper triangle excluding diagonal is True.
- Diagonal and lower triangle are False.

### `tests/test_attention_lib.py` — minimum required tests

| Test name | What it tests |
|---|---|
| `test_sdpa_output_shape` | Arbitrary (B, n, m, d_k, d_v) |
| `test_sdpa_weights_sum_to_one` | Row-sum = 1 |
| `test_sdpa_causal_mask_zeros_upper_triangle` | Upper triangle = 0 |
| `test_mha_output_shape` | (B, n, d_model) |
| `test_mha_weights_shape` | (B, num_heads, n, m) |
| `test_mha_cross_attention_shape` | n ≠ m |
| `test_causal_mask_shape` | (T, T) bool |
| `test_causal_mask_diagonal_unmasked` | mask[i,i] = False |
| `test_sinusoidal_pe_deterministic` | Two instances = same output |
| `test_sinusoidal_pe_unique_positions` | All rows distinct |
| `test_transformer_block_shape` | Input = output shape |
| `test_backward_through_block` | Gradients non-None |

## Rubric

| Dimension | Weight | Criteria |
|---|---|---|
| Independent implementation | 40% | All 6 components work without referencing lesson code |
| Correctness & testing | 25% | All 12+ tests pass; cross-validation with PyTorch passes |
| Code quality | 20% | Type hints, logging, no god functions, clean structure |
| Conceptual understanding | 15% | Reflection notes show genuine understanding of failure modes |

Pass threshold: 75%

## Submission

Package the three files. Verify `python -m pytest tests/test_attention_lib.py -v` exits with 0. Verify `python day6_integration.py` logs "TinyGPT forward pass OK".
