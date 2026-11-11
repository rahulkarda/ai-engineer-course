# Phase 3 · Week 05 · Day 5 — Full Transformer Block: FFN, Residual, LayerNorm, Masking, Encoder/Decoder

> Objective: Assemble a complete transformer block and understand encoder, decoder, and decoder-only architectures.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### The transformer block (one layer)

A single transformer layer wraps attention in two sub-layers, each guarded by a residual connection and layer normalisation:

```
Sub-layer 1 (attention):
  x = LayerNorm(x + Dropout(MultiHeadAttention(x, x, x)))

Sub-layer 2 (feed-forward):
  x = LayerNorm(x + Dropout(FFN(x)))
```

This is "Post-LN" (original paper). "Pre-LN" (most modern models) moves LayerNorm inside:

```
x = x + Dropout(MultiHeadAttention(LayerNorm(x), ...))
x = x + Dropout(FFN(LayerNorm(x)))
```

Pre-LN trains more stably at large depth.

### Feed-Forward Network (FFN)

Two linear layers with a non-linearity:

```
FFN(x) = Linear(ReLU(Linear(x)))
       = W₂ · max(0, W₁ · x + b₁) + b₂
```

Dimensions: input `d_model` → inner `d_ff` → output `d_model`. Standard: `d_ff = 4 · d_model`.

Modern models use GELU instead of ReLU, and some use SwiGLU (gated variant).

The FFN applies position-wise: the same network runs independently at each token position. This is where most of the network's "knowledge" is stored — attention routes information, FFN transforms it.

### Residual connections

`x + sublayer(x)` ensures:
- Gradients flow directly from loss to early layers (no vanishing).
- Each sublayer only needs to learn a **residual** correction, not a full transformation.
- If the sublayer is uninitialised/useless, x passes unchanged (identity initialisation).

Deep networks (GPT-3: 96 layers) require residual connections to be trainable at all.

### Layer Normalisation

Normalises over the feature (d_model) dimension, not the batch dimension:

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β
```

Where μ and σ are computed over the d_model features of each token independently.

Why not BatchNorm? LN is computed per-token regardless of batch size — stable for variable-length sequences and batch size 1 at inference.

### Masking: causal (autoregressive) mask

For decoder-only (GPT-style) models: token at position t must not see positions > t.

```
mask[i, j] = True  if  j > i  (upper triangle, excluding diagonal)
```

Set masked logits to −∞ before softmax → probability 0. This is "causal" or "autoregressive" masking.

For encoder models (BERT-style): no causal mask — all positions attend to all others.

For encoder-decoder: encoder has no mask; decoder has causal mask for self-attention AND uses cross-attention (Q from decoder, K/V from encoder output) with no mask.

## How it works internally

### Three architecture families

| Architecture | Self-attn mask | Cross-attn | Example |
|---|---|---|---|
| Encoder-only | None (bidirectional) | No | BERT, RoBERTa |
| Decoder-only | Causal | No | GPT series, LLaMA |
| Encoder-decoder | Encoder: none; Decoder: causal | Yes (dec Q → enc K/V) | T5, BART, original transformer |

**Encoder-only**: best for classification, NER, understanding tasks — sees full context.

**Decoder-only**: best for generation — predicts next token autoregressively.

**Encoder-decoder**: best for seq2seq (translation, summarisation) — encoder builds full context representation, decoder generates token by token conditioned on it.

### Why most modern LLMs are decoder-only

Decoder-only is simpler (one module type), scales better empirically at large data, and handles generation naturally. The "encoding" happens implicitly in the prefill phase.

## Worked example

```python
"""
day5_transformer_block.py
Full transformer block: MHA + FFN + residual + LayerNorm + causal masking.
Run: python day5_transformer_block.py
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = query.size(0)
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(attn_out)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Upper-triangular boolean mask: True = blocked (future positions)."""
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


class TransformerBlock(nn.Module):
    """
    One transformer decoder-style block (Pre-LN variant).

    Args:
        d_model:   Model dimension.
        num_heads: Number of attention heads.
        d_ff:      FFN inner dimension (typically 4 * d_model).
        dropout:   Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pre-LN transformer block.
        Args:
            x:    (B, T, d_model)
            mask: Optional causal mask (T, T) or (B, 1, T, T)
        Returns:
            (B, T, d_model)
        """
        # Sub-layer 1: self-attention with residual
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, mask))
        # Sub-layer 2: FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TinyDecoderOnlyTransformer(nn.Module):
    """
    Minimal decoder-only (GPT-style) transformer for demonstration.

    Args:
        vocab_size: Vocabulary size.
        d_model:    Model dimension.
        num_heads:  Attention heads.
        num_layers: Number of transformer blocks.
        max_len:    Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_len: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff=4 * d_model)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, T) integer token ids
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(positions)

        mask = causal_mask(T, device=tokens.device).unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, T, T) — broadcasts over batch and heads

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.head(x)


def verify_shapes() -> None:
    torch.manual_seed(42)
    vocab, B, T, d, h, layers = 100, 2, 12, 64, 4, 3
    model = TinyDecoderOnlyTransformer(
        vocab_size=vocab, d_model=d, num_heads=h, num_layers=layers
    )
    tokens = torch.randint(0, vocab, (B, T))
    logits = model(tokens)
    assert logits.shape == (B, T, vocab), f"logits shape: {logits.shape}"
    log.info("Shape test PASSED: logits %s", logits.shape)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)


if __name__ == "__main__":
    verify_shapes()
```

## Exercises

1. Extend `TinyDecoderOnlyTransformer` with a `generate(prompt: list[int], max_new_tokens: int) -> list[int]` method that greedily samples the next token (argmax of logits at the last position) `max_new_tokens` times. Test that the output is `len(prompt) + max_new_tokens` tokens.

2. Implement an `EncoderBlock` (no causal mask) and an `EncoderDecoderBlock` (adds cross-attention sub-layer between self-attention and FFN). The cross-attention sub-layer takes Q from the decoder stream and K, V from the encoder output.

3. Post-LN vs Pre-LN: modify `TransformerBlock` to support a `pre_ln: bool` flag. Write a test that verifies both produce the correct output shape. Which is recommended for very deep models and why?

4. What is the output of `causal_mask(4)`? Draw the 4×4 boolean matrix. Explain why the diagonal is NOT masked.

(Solutions: ../solutions/week-05/day-5.md)

## Self-check

- [ ] I can assemble a full transformer block (MHA + FFN + residual + LayerNorm) from the components built this week.
- [ ] I can explain Pre-LN vs Post-LN and the training stability argument.
- [ ] I can describe the three transformer families (encoder-only / decoder-only / encoder-decoder) and when to use each.
- [ ] I understand why causal masking preserves the autoregressive property.

## Supplementary

- Vaswani et al. (2017) full paper — https://arxiv.org/abs/1706.03762
- Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture" (Pre-LN analysis) — https://arxiv.org/abs/2002.04745
- The Annotated Transformer — https://nlp.seas.harvard.edu/annotated-transformer/
