# Phase 3 · Week 07 · Day 6 — Saturday Deep Practice: Build Sampling Toolkit + KV Cache Demo

> Objective: Implement a production-quality sampling/decoding toolkit and KV cache from scratch using NumPy.  ·  Time: ~4 hr  ·  Prereqs: [[day-3]], [[day-4]], [[day-5]]

---

## Overview

Today you build a complete, testable implementation of the components that power autoregressive LLM decoding:

| Part | Time | Deliverable |
|------|------|-------------|
| 1 — Sampling Toolkit | 60 min | `src/sampling.py` |
| 2 — KV Cache | 45 min | `src/kv_cache.py` |
| 3 — Mock Model | 45 min | `src/mock_model.py` |
| 4 — Demo Script | 30 min | `src/demo.py` |
| 5 — Test Suite | 60 min | `tests/test_sampling.py`, `tests/test_kv_cache.py` |

Project structure:
```
week-07/
  src/
    sampling.py
    kv_cache.py
    mock_model.py
    demo.py
  tests/
    test_sampling.py
    test_kv_cache.py
```

Install dependencies first:
```bash
uv add numpy
uv add pytest
```

---

## Concept review: what each component does

**Sampling** transforms a raw logit vector (one float per vocabulary token) into a discrete token ID. The pipeline is:
1. Optionally apply repetition penalty to suppress already-generated tokens.
2. Apply temperature scaling to sharpen or flatten the distribution.
3. Apply top-k filtering to zero out all but the k highest-logit tokens.
4. Apply top-p (nucleus) filtering to zero out tokens outside the top-p probability mass.
5. Convert to probabilities via softmax, then sample.

**KV Cache** stores the key and value tensors from past attention steps so they don't need to be recomputed. At decode step t, each transformer layer needs keys K and values V for positions 0..t. Without a cache, this requires a full forward pass over the entire sequence. With the cache, only position t needs to be computed and appended.

**Autoregressive generation** is a loop:
```
input_ids = prompt_tokens
for step in range(max_new_tokens):
    logits, kv_cache = model.get_logits(input_ids[-1:], kv_cache)
    next_token = sample_token(logits, ...)
    input_ids.append(next_token)
    if next_token == EOS_TOKEN:
        break
```

---

## Part 1 — Sampling Toolkit

### How it works internally

**Temperature scaling**: logits are divided by T before softmax. T < 1 sharpens the distribution (more deterministic), T > 1 flattens it (more random). T -> 0 approaches greedy (argmax). The math: `softmax(logits / T)`. Dividing by a small T amplifies differences between logits exponentially after softmax.

**Top-k filtering**: sort logits descending, keep top k, set the rest to -inf. After softmax, -inf becomes probability 0. This prevents the model from ever sampling low-probability "garbage" tokens.

**Top-p (nucleus) filtering**: sort by descending probability, accumulate until the cumulative probability >= p, then cut off the tail. Adapts to the distribution: if the model is confident, nucleus might include only 1-2 tokens; if uncertain, it includes more.

**Repetition penalty**: for each token already generated, divide its logit by the penalty factor (if > 1.0, this reduces the logit, making the token less likely). The standard HuggingFace formulation: if logit > 0, divide by penalty; if logit < 0, multiply by penalty. This ensures the penalty always decreases probability regardless of logit sign.

### src/sampling.py

```python
# src/sampling.py
"""
Sampling and decoding utilities for autoregressive text generation.

All functions operate on raw logit arrays (pre-softmax scores).
Vocab dimension is always the last axis.

Install: uv add numpy
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis."""
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def greedy_decode(logits: np.ndarray) -> int:
    """
    Return the token with the highest logit (argmax).

    Args:
        logits: shape (vocab_size,) — raw pre-softmax scores.

    Returns:
        Token ID with the maximum logit.
    """
    return int(np.argmax(logits))


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Scale logits by temperature.

    Args:
        logits: shape (vocab_size,).
        temperature: float > 0.
            - temperature < 1.0: sharper distribution (more deterministic).
            - temperature > 1.0: flatter distribution (more random).
            - temperature -> 0: approaches greedy argmax.

    Returns:
        Scaled logits, same shape.

    Raises:
        ValueError: if temperature <= 0.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if temperature == 1.0:
        return logits.copy()
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Zero out (set to -inf) all logits except the top k.

    Args:
        logits: shape (vocab_size,).
        k: number of top tokens to keep. If k <= 0 or k >= vocab_size, returns logits unchanged.

    Returns:
        Filtered logits, same shape. Non-top-k positions are -inf.
    """
    vocab_size = logits.shape[-1]
    if k <= 0 or k >= vocab_size:
        return logits.copy()

    filtered = logits.copy()
    # Partition: find the k-th largest value
    # np.partition puts the k-th smallest at index k from the end
    threshold_idx = vocab_size - k
    kth_value = np.partition(filtered, threshold_idx)[threshold_idx]
    # Set everything below the k-th value to -inf
    filtered[filtered < kth_value] = -np.inf
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability >= p.

    Args:
        logits: shape (vocab_size,).
        p: float in (0, 1]. p=1.0 means no filtering.

    Returns:
        Filtered logits, same shape. Tokens outside the nucleus are -inf.
    """
    if p >= 1.0:
        return logits.copy()
    if p <= 0.0:
        raise ValueError(f"p must be > 0, got {p}")

    # Compute probabilities
    probs = _softmax(logits)

    # Sort tokens by descending probability
    sorted_indices = np.argsort(probs)[::-1]  # descending
    sorted_probs = probs[sorted_indices]

    # Cumulative probability
    cumulative_probs = np.cumsum(sorted_probs)

    # Find cutoff: include tokens until cumulative prob reaches p.
    # Shift by 1 so we include the token that pushes over the threshold.
    cutoff_mask = cumulative_probs - sorted_probs > p
    # Tokens marked True in cutoff_mask should be removed
    tokens_to_remove = sorted_indices[cutoff_mask]

    filtered = logits.copy()
    filtered[tokens_to_remove] = -np.inf
    return filtered


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float,
) -> np.ndarray:
    """
    Apply repetition penalty to logits for tokens that have already been generated.

    Formula (HuggingFace convention):
        if logit[token] > 0: logit[token] /= penalty
        if logit[token] < 0: logit[token] *= penalty

    This always reduces the probability of repeated tokens regardless of logit sign.

    Args:
        logits: shape (vocab_size,).
        generated_ids: list of token IDs already generated.
        penalty: float >= 1.0. 1.0 means no penalty.

    Returns:
        Modified logits, same shape.
    """
    if penalty == 1.0 or not generated_ids:
        return logits.copy()
    if penalty < 1.0:
        raise ValueError(f"penalty must be >= 1.0, got {penalty}")

    penalized = logits.copy()
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if 0 <= token_id < len(penalized):
            if penalized[token_id] > 0:
                penalized[token_id] /= penalty
            else:
                penalized[token_id] *= penalty
    return penalized


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[list[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Full sampling pipeline: penalty -> temperature -> top-k -> top-p -> sample.

    Args:
        logits: shape (vocab_size,) — raw pre-softmax scores.
        temperature: temperature for scaling. Must be > 0.
        top_k: if > 0, keep only the top k tokens before sampling.
        top_p: nucleus probability threshold. 1.0 = no filtering.
        repetition_penalty: penalty for repeated tokens. 1.0 = no penalty.
        generated_ids: token IDs already generated (for repetition penalty).
        rng: numpy random generator. If None, uses np.random default.

    Returns:
        Sampled token ID.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: repetition penalty
    if repetition_penalty != 1.0 and generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Step 2: temperature == 0 -> greedy
    if temperature == 0.0:
        return greedy_decode(logits)

    # Step 3: temperature scaling
    logits = temperature_scale(logits, temperature)

    # Step 4: top-k filtering
    if top_k > 0:
        logits = top_k_filter(logits, top_k)

    # Step 5: top-p filtering
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    # Step 6: softmax -> probabilities
    probs = _softmax(logits)

    # Guard: if all probs are 0 (shouldn't happen but guard against NaN)
    if not np.isfinite(probs).all() or probs.sum() == 0:
        log.warning("Invalid probability distribution, falling back to greedy.")
        return greedy_decode(logits)

    # Step 7: categorical sample
    vocab_size = len(probs)
    return int(rng.choice(vocab_size, p=probs))
```

---

## Part 2 — KV Cache

### How it works internally

Each transformer layer computes, for each new token at position t:
```
K_t = W_K @ x_t   (shape: n_heads x d_head)
V_t = W_V @ x_t
```

Attention for token t requires all keys K_0..K_t and values V_0..V_t. Storing them in a cache avoids recomputing positions 0..t-1 at every decode step.

The cache stores one (K, V) pair per layer. When a new token is processed, its K and V are **appended** along the sequence dimension. Memory usage grows as: `2 * n_layers * n_heads * d_head * seq_len * bytes_per_element`.

For a 7B model (32 layers, 32 heads, 128 d_head, FP16, 2048 seq len):
`2 * 32 * 32 * 128 * 2048 * 2 bytes = 1,073,741,824 bytes = 1 GB`

This is why context length dramatically affects memory requirements.

### src/kv_cache.py

```python
# src/kv_cache.py
"""
KV cache for transformer attention.

Stores per-layer key/value tensors and supports incremental appending
during autoregressive decoding.

Install: uv add numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class KVCache:
    """
    Per-layer key-value cache for autoregressive decoding.

    Each layer stores:
        k: shape (seq_len, n_heads, d_head)
        v: shape (seq_len, n_heads, d_head)

    seq_len grows by 1 on each append call (one new token per decode step).

    Attributes:
        n_layers: number of transformer layers.
        n_heads: number of attention heads.
        d_head: dimension per head.
        dtype: numpy dtype for storage (e.g., np.float16, np.float32).
        cache: internal per-layer storage, initialized as empty arrays.
    """

    n_layers: int
    n_heads: int
    d_head: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))

    # Internal storage: list of dicts, one per layer.
    # Each dict: {"k": np.ndarray shape (seq_len, n_heads, d_head),
    #             "v": np.ndarray shape (seq_len, n_heads, d_head)}
    cache: list[dict[str, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize empty cache for each layer."""
        self.cache = [
            {
                "k": np.empty((0, self.n_heads, self.d_head), dtype=self.dtype),
                "v": np.empty((0, self.n_heads, self.d_head), dtype=self.dtype),
            }
            for _ in range(self.n_layers)
        ]
        log.debug(
            "KVCache initialized: %d layers, %d heads, %d d_head, dtype=%s",
            self.n_layers,
            self.n_heads,
            self.d_head,
            self.dtype,
        )

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """
        Append key/value tensors for a new token at the given layer.

        Args:
            layer_idx: which transformer layer (0-indexed).
            k: shape (1, n_heads, d_head) or (n_heads, d_head) — new token keys.
            v: shape (1, n_heads, d_head) or (n_heads, d_head) — new token values.

        Raises:
            IndexError: if layer_idx is out of range.
            ValueError: if k or v have wrong shape.
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")

        # Normalize to (1, n_heads, d_head)
        if k.ndim == 2:
            k = k[np.newaxis, ...]
        if v.ndim == 2:
            v = v[np.newaxis, ...]

        expected_shape_suffix = (self.n_heads, self.d_head)
        if k.shape[1:] != expected_shape_suffix:
            raise ValueError(
                f"k shape {k.shape} incompatible with (*, {self.n_heads}, {self.d_head})"
            )
        if v.shape[1:] != expected_shape_suffix:
            raise ValueError(
                f"v shape {v.shape} incompatible with (*, {self.n_heads}, {self.d_head})"
            )

        layer = self.cache[layer_idx]
        layer["k"] = np.concatenate([layer["k"], k.astype(self.dtype)], axis=0)
        layer["v"] = np.concatenate([layer["v"], v.astype(self.dtype)], axis=0)

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve accumulated key/value tensors for a layer.

        Args:
            layer_idx: which transformer layer (0-indexed).

        Returns:
            Tuple (k, v) where each has shape (seq_len, n_heads, d_head).
        """
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")
        layer = self.cache[layer_idx]
        return layer["k"], layer["v"]

    def memory_bytes(self) -> int:
        """
        Total memory used by cached tensors in bytes.

        Returns:
            Sum of nbytes across all k and v arrays in all layers.
        """
        total = 0
        for layer in self.cache:
            total += layer["k"].nbytes
            total += layer["v"].nbytes
        return total

    def seq_len(self) -> int:
        """
        Current cached sequence length (number of tokens stored).

        Returns 0 if the cache is empty.
        """
        return int(self.cache[0]["k"].shape[0])

    def clear(self) -> None:
        """Reset the cache to empty (seq_len = 0)."""
        for layer in self.cache:
            layer["k"] = np.empty((0, self.n_heads, self.d_head), dtype=self.dtype)
            layer["v"] = np.empty((0, self.n_heads, self.d_head), dtype=self.dtype)
        log.debug("KVCache cleared.")

    def __repr__(self) -> str:
        return (
            f"KVCache(n_layers={self.n_layers}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, seq_len={self.seq_len()}, "
            f"memory={self.memory_bytes() / 1024:.1f} KB)"
        )
```

---

## Part 3 — Mock Model

### How it works internally

A real transformer layer does:
1. Layer norm the input.
2. Compute Q, K, V projections.
3. Scale dot-product attention with KV cache.
4. Add residual and apply FFN.

The mock model uses random (fixed-seed) weight matrices but follows the same data-flow. This lets us verify that:
- Logit shape is correct (vocab_size,).
- KV cache is properly used and updated.
- The generation loop produces repeatable results.

The attention computation: `attn = softmax(Q @ K^T / sqrt(d_head)) @ V`

For decoding, Q is shape (1, n_heads, d_head) for the new token. K and V are from the cache, shape (seq_len, n_heads, d_head). The output is shape (1, n_heads, d_head), flattened to (d_model,).

### src/mock_model.py

```python
# src/mock_model.py
"""
Tiny mock autoregressive transformer model using random weights (NumPy only).

Not trained — used only to demonstrate the shape and flow of:
    - Multi-head attention with KV cache
    - Logit generation for sampling

Architecture:
    n_vocab = 100
    n_layers = 2
    n_heads = 4
    d_head = 16
    d_model = 64  (= n_heads * d_head)

Install: uv add numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from kv_cache import KVCache

log = logging.getLogger(__name__)


@dataclass
class MockModelConfig:
    n_vocab: int = 100
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 16
    d_model: int = 64  # must equal n_heads * d_head
    seed: int = 42


class MockModel:
    """
    Minimal mock transformer for decoding demos.

    Uses fixed random weights initialized from a seed. Does not compute
    any gradients or meaningful semantics — output logits are random
    but structurally correct.
    """

    def __init__(self, config: MockModelConfig | None = None) -> None:
        self.cfg = config or MockModelConfig()
        assert self.cfg.d_model == self.cfg.n_heads * self.cfg.d_head, (
            f"d_model ({self.cfg.d_model}) must equal n_heads ({self.cfg.n_heads}) "
            f"* d_head ({self.cfg.d_head})"
        )
        self._init_weights()
        log.info("MockModel initialized: %s", self.cfg)

    def _init_weights(self) -> None:
        """Initialize random weight matrices for each layer."""
        rng = np.random.default_rng(self.cfg.seed)
        cfg = self.cfg

        # Weight scale for random init: 1/sqrt(d_model)
        scale = 1.0 / np.sqrt(cfg.d_model)

        # Token embedding: (n_vocab, d_model)
        self.embed = rng.normal(0, scale, (cfg.n_vocab, cfg.d_model)).astype(np.float32)

        # Per-layer attention weights
        self.w_q: list[np.ndarray] = []
        self.w_k: list[np.ndarray] = []
        self.w_v: list[np.ndarray] = []
        self.w_o: list[np.ndarray] = []

        for _ in range(cfg.n_layers):
            # Q, K, V projections: (d_model, d_model)
            self.w_q.append(rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32))
            self.w_k.append(rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32))
            self.w_v.append(rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32))
            # Output projection: (d_model, d_model)
            self.w_o.append(rng.normal(0, scale, (cfg.d_model, cfg.d_model)).astype(np.float32))

        # LM head (unembedding): (d_model, n_vocab)
        self.lm_head = rng.normal(0, scale, (cfg.d_model, cfg.n_vocab)).astype(np.float32)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Simple layer normalization (no learned scale/bias for brevity)."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _attention(
        self,
        x: np.ndarray,
        layer_idx: int,
        kv_cache: KVCache,
    ) -> tuple[np.ndarray, KVCache]:
        """
        Multi-head attention for a single new token, using KV cache.

        Args:
            x: shape (d_model,) — embedding of the current token.
            layer_idx: which layer (for KV cache lookup).
            kv_cache: current cache state.

        Returns:
            (output, updated_kv_cache)
            output: shape (d_model,)
        """
        cfg = self.cfg

        # Project Q, K, V for the current token
        # Shape: (d_model,) @ (d_model, d_model) = (d_model,)
        q = x @ self.w_q[layer_idx]  # (d_model,)
        k = x @ self.w_k[layer_idx]  # (d_model,)
        v = x @ self.w_v[layer_idx]  # (d_model,)

        # Reshape to (n_heads, d_head)
        q_h = q.reshape(cfg.n_heads, cfg.d_head)
        k_h = k.reshape(cfg.n_heads, cfg.d_head)
        v_h = v.reshape(cfg.n_heads, cfg.d_head)

        # Append current K, V to cache
        kv_cache.append(layer_idx, k_h, v_h)

        # Retrieve full K, V history: (seq_len, n_heads, d_head)
        k_cache, v_cache = kv_cache.get(layer_idx)
        seq_len = k_cache.shape[0]

        # Compute attention scores per head
        # q_h: (n_heads, d_head), k_cache: (seq_len, n_heads, d_head)
        # scores[h, t] = q_h[h] @ k_cache[t, h] / sqrt(d_head)
        scale = 1.0 / np.sqrt(cfg.d_head)

        # Vectorized: scores shape (n_heads, seq_len)
        # Transpose k_cache to (n_heads, seq_len, d_head) for batched matmul
        k_t = k_cache.transpose(1, 0, 2)  # (n_heads, seq_len, d_head)
        v_t = v_cache.transpose(1, 0, 2)  # (n_heads, seq_len, d_head)

        # q_h: (n_heads, d_head) -> (n_heads, 1, d_head)
        q_h_exp = q_h[:, np.newaxis, :]  # (n_heads, 1, d_head)

        # scores: (n_heads, 1, seq_len)
        scores = q_h_exp @ k_t.transpose(0, 2, 1) * scale  # (n_heads, 1, seq_len)

        # Softmax over seq_len dimension
        scores_shifted = scores - scores.max(axis=-1, keepdims=True)
        attn_weights = np.exp(scores_shifted)
        attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

        # Weighted sum of values: (n_heads, 1, d_head)
        # v_t: (n_heads, seq_len, d_head) -> (n_heads, d_head, seq_len) for matmul
        out_h = attn_weights @ v_t  # (n_heads, 1, d_head)
        out_h = out_h.squeeze(1)  # (n_heads, d_head)

        # Flatten and apply output projection
        out_flat = out_h.reshape(cfg.d_model)  # (d_model,)
        out = out_flat @ self.w_o[layer_idx]   # (d_model,)

        return out, kv_cache

    def get_logits(
        self,
        input_ids: list[int],
        kv_cache: KVCache | None = None,
    ) -> tuple[np.ndarray, KVCache]:
        """
        Forward pass for a single new token (decode step).

        Processes only the LAST token in input_ids; all previous tokens
        are assumed to already be in the KV cache.

        Args:
            input_ids: list of token IDs. Uses only input_ids[-1] for the
                       new token embedding. All prior context is in kv_cache.
            kv_cache: existing KV cache. Creates a new empty cache if None.

        Returns:
            (logits, updated_kv_cache)
            logits: shape (n_vocab,) — raw pre-softmax scores.
        """
        cfg = self.cfg

        if kv_cache is None:
            kv_cache = KVCache(
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                d_head=cfg.d_head,
                dtype=np.dtype(np.float32),
            )

        # Embed the last token (current decode position)
        token_id = input_ids[-1]
        if token_id < 0 or token_id >= cfg.n_vocab:
            raise ValueError(f"token_id {token_id} out of range [0, {cfg.n_vocab})")

        x = self.embed[token_id].copy()  # (d_model,)

        # Pass through transformer layers
        for layer_idx in range(cfg.n_layers):
            # Pre-norm attention
            x_norm = self._layer_norm(x)
            attn_out, kv_cache = self._attention(x_norm, layer_idx, kv_cache)
            x = x + attn_out  # residual connection

            # Simple FFN: two linear layers with ReLU (no weights learned, use identity)
            # For a mock model, we skip the FFN or use a trivial identity
            x = self._layer_norm(x)

        # Project to vocabulary
        logits = x @ self.lm_head  # (n_vocab,)
        return logits, kv_cache

    def new_cache(self) -> KVCache:
        """Create a fresh empty KV cache matching this model's dimensions."""
        return KVCache(
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            d_head=self.cfg.d_head,
            dtype=np.dtype(np.float32),
        )
```

---

## Part 4 — Demo Script

### src/demo.py

```python
# src/demo.py
"""
Demonstration of sampling strategies and KV cache memory comparison.

Run from the week-07/ directory:
    uv run python src/demo.py

Install: uv add numpy
"""

from __future__ import annotations

import logging
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np

from mock_model import MockModel, MockModelConfig
from sampling import sample_token, greedy_decode
from kv_cache import KVCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def generate(
    model: MockModel,
    prompt_ids: list[int],
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Autoregressive generation loop using the mock model and KV cache.

    Processes the prompt token-by-token to fill the cache, then samples
    max_new_tokens new tokens.

    Args:
        model: the mock model.
        prompt_ids: list of prompt token IDs.
        max_new_tokens: number of new tokens to generate.
        temperature, top_k, top_p, repetition_penalty: sampling params.
        rng: numpy random generator for reproducibility.

    Returns:
        List of generated token IDs (not including prompt).
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    kv_cache = model.new_cache()
    generated: list[int] = []

    # Process prompt: fill KV cache with prompt tokens
    log.info("Processing prompt (%d tokens)...", len(prompt_ids))
    current_ids = list(prompt_ids)
    for i, token_id in enumerate(prompt_ids):
        # For each prompt token, run forward pass to populate the cache
        _, kv_cache = model.get_logits(current_ids[: i + 1], kv_cache)
        # Reset cache between prompt tokens to avoid double-counting
        # Actually: for this demo, build cache incrementally
        # We re-create the cache and process the full prompt once
        # to avoid complexity. In production, this is done in a single batched prefill.
        pass

    # Simpler approach: process full prompt in one shot by rebuilding cache
    kv_cache = model.new_cache()
    for i in range(len(prompt_ids)):
        logits, kv_cache = model.get_logits(prompt_ids[: i + 1], kv_cache)

    log.info("Prompt processed. Cache seq_len=%d, memory=%.1f KB",
             kv_cache.seq_len(), kv_cache.memory_bytes() / 1024)

    # Generate new tokens
    for step in range(max_new_tokens):
        # Input is just the last token (rest is in KV cache)
        last_token = prompt_ids[-1] if step == 0 else generated[-1]
        # We need to pass the full sequence so get_logits knows which token to embed
        # (it uses input_ids[-1])
        full_ids = prompt_ids + generated
        logits, kv_cache = model.get_logits(full_ids, kv_cache)

        next_token = sample_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_ids=generated,
            rng=rng,
        )
        generated.append(next_token)

    return generated


def compare_kv_cache_memory(n_layers: int, n_heads: int, d_head: int) -> None:
    """
    Compare KV cache memory usage vs naive recomputation at different sequence lengths.
    """
    log.info("\n--- KV Cache Memory Comparison ---")
    log.info("Config: %d layers, %d heads, %d d_head, float32", n_layers, n_heads, d_head)

    for seq_len in [128, 512, 2048, 8192]:
        cache = KVCache(n_layers=n_layers, n_heads=n_heads, d_head=d_head)
        # Fill cache with dummy data
        dummy_k = np.zeros((1, n_heads, d_head), dtype=np.float32)
        dummy_v = np.zeros((1, n_heads, d_head), dtype=np.float32)
        for _ in range(seq_len):
            for layer in range(n_layers):
                cache.append(layer, dummy_k, dummy_v)

        cache_mb = cache.memory_bytes() / (1024 * 1024)
        # Naive: recompute all tokens every step = store full activations
        # Approximate: d_model * seq_len * n_layers * 4 bytes (float32) * 2 (k+v)
        d_model = n_heads * d_head
        naive_bytes = 2 * n_layers * seq_len * d_model * 4
        naive_mb = naive_bytes / (1024 * 1024)

        log.info(
            "seq_len=%5d | KV cache: %7.2f MB | naive activations: %7.2f MB",
            seq_len, cache_mb, naive_mb,
        )


def main() -> None:
    log.info("=== Sampling Toolkit + KV Cache Demo ===")

    config = MockModelConfig(n_vocab=100, n_layers=2, n_heads=4, d_head=16, d_model=64)
    model = MockModel(config)

    prompt = [10, 23, 45, 7]  # arbitrary token IDs within vocab
    max_new = 15

    # Strategy 1: Greedy
    log.info("\n--- Greedy Decoding ---")
    rng = np.random.default_rng(seed=1)
    greedy_out = generate(model, prompt, max_new_tokens=max_new,
                          temperature=0.0001, top_k=1, rng=rng)
    log.info("Greedy output tokens: %s", greedy_out)

    # Strategy 2: Temperature sampling
    log.info("\n--- Temperature Sampling (T=0.8) ---")
    rng = np.random.default_rng(seed=2)
    temp_out = generate(model, prompt, max_new_tokens=max_new,
                        temperature=0.8, rng=rng)
    log.info("Temperature output tokens: %s", temp_out)

    # Strategy 3: Top-k sampling
    log.info("\n--- Top-k Sampling (k=10) ---")
    rng = np.random.default_rng(seed=3)
    topk_out = generate(model, prompt, max_new_tokens=max_new,
                        temperature=1.0, top_k=10, rng=rng)
    log.info("Top-k output tokens: %s", topk_out)

    # Strategy 4: Top-p sampling
    log.info("\n--- Top-p Sampling (p=0.9) ---")
    rng = np.random.default_rng(seed=4)
    topp_out = generate(model, prompt, max_new_tokens=max_new,
                        temperature=1.0, top_p=0.9, rng=rng)
    log.info("Top-p output tokens: %s", topp_out)

    # Strategy 5: Combined with repetition penalty
    log.info("\n--- Top-k + Top-p + Repetition Penalty (k=20, p=0.95, rep=1.3) ---")
    rng = np.random.default_rng(seed=5)
    combined_out = generate(model, prompt, max_new_tokens=max_new,
                            temperature=0.9, top_k=20, top_p=0.95,
                            repetition_penalty=1.3, rng=rng)
    log.info("Combined output tokens: %s", combined_out)

    # KV cache memory comparison
    compare_kv_cache_memory(n_layers=2, n_heads=4, d_head=16)

    # Real model memory estimate
    log.info("\n--- Real Model KV Cache Estimate (7B FP16 equivalent) ---")
    # 7B model: 32 layers, 32 heads, 128 d_head, float16 (2 bytes)
    for seq_len in [2048, 4096, 8192]:
        n_layers, n_heads, d_head = 32, 32, 128
        bytes_per_elem = 2  # float16
        kv_bytes = 2 * n_layers * n_heads * d_head * seq_len * bytes_per_elem
        log.info("7B model, seq_len=%d: KV cache = %.2f GB", seq_len, kv_bytes / 1e9)


if __name__ == "__main__":
    main()
```

---

## Part 5 — Test Suite

### tests/test_sampling.py

```python
# tests/test_sampling.py
"""
Tests for src/sampling.py

Run: uv run pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from sampling import (
    greedy_decode,
    temperature_scale,
    top_k_filter,
    top_p_filter,
    apply_repetition_penalty,
    sample_token,
    _softmax,
)


class TestGreedyDecode:
    def test_greedy_selects_argmax(self) -> None:
        """Greedy decode must return the index of the maximum logit."""
        logits = np.array([0.1, 0.5, 0.9, 0.3, 0.2], dtype=np.float32)
        result = greedy_decode(logits)
        assert result == 2, f"Expected argmax=2, got {result}"

    def test_greedy_with_negative_logits(self) -> None:
        """Works correctly when all logits are negative."""
        logits = np.array([-3.0, -1.0, -2.0, -0.5, -4.0], dtype=np.float32)
        result = greedy_decode(logits)
        assert result == 3

    def test_greedy_single_token(self) -> None:
        """Single-token vocab should always return 0."""
        logits = np.array([5.0], dtype=np.float32)
        assert greedy_decode(logits) == 0


class TestTemperatureScale:
    def test_temperature_one_is_identity(self) -> None:
        """Temperature=1.0 should return logits unchanged."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = temperature_scale(logits, 1.0)
        np.testing.assert_array_almost_equal(result, logits)

    def test_temperature_below_one_sharpens(self) -> None:
        """T < 1 should amplify differences between logits."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        scaled = temperature_scale(logits, 0.5)
        # After dividing by 0.5, differences double
        expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_temperature_above_one_flattens(self) -> None:
        """T > 1 should reduce differences between logits."""
        logits = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        scaled = temperature_scale(logits, 2.0)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(scaled, expected)

    def test_temperature_zero_raises(self) -> None:
        """Temperature=0 should raise ValueError."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError):
            temperature_scale(logits, 0.0)

    def test_temperature_negative_raises(self) -> None:
        """Negative temperature must raise ValueError."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError):
            temperature_scale(logits, -1.0)


class TestTopKFilter:
    def test_top_k_filters_correctly(self) -> None:
        """Only the top-k logits should remain; others become -inf."""
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float32)
        filtered = top_k_filter(logits, k=2)
        # Top-2 are indices 1 (5.0) and 4 (4.0)
        assert filtered[1] == 5.0
        assert filtered[4] == 4.0
        assert filtered[0] == -np.inf
        assert filtered[2] == -np.inf
        assert filtered[3] == -np.inf

    def test_top_k_zero_returns_unchanged(self) -> None:
        """k=0 should return logits unchanged."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = top_k_filter(logits, k=0)
        np.testing.assert_array_equal(result, logits)

    def test_top_k_ge_vocab_returns_unchanged(self) -> None:
        """k >= vocab_size should return logits unchanged."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = top_k_filter(logits, k=10)
        np.testing.assert_array_equal(result, logits)

    def test_top_k_one_keeps_only_max(self) -> None:
        """k=1 should keep only the argmax token."""
        logits = np.array([0.5, 3.0, 1.0, 2.5], dtype=np.float32)
        filtered = top_k_filter(logits, k=1)
        assert np.isfinite(filtered[1])  # index 1 is max
        assert np.all(np.isinf(filtered[[0, 2, 3]]))

    def test_top_k_preserves_values(self) -> None:
        """Original logit values for kept tokens must not be modified."""
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float32)
        filtered = top_k_filter(logits, k=3)
        # Indices 1 (5.0), 4 (4.0), 2 (3.0) are top-3
        assert filtered[1] == pytest.approx(5.0)
        assert filtered[4] == pytest.approx(4.0)
        assert filtered[2] == pytest.approx(3.0)


class TestTopPFilter:
    def test_top_p_sums_to_one(self) -> None:
        """After top-p filtering + softmax, the remaining probabilities must sum to ~1."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        filtered = top_p_filter(logits, p=0.9)
        # Compute softmax of filtered (finite) logits
        finite_mask = np.isfinite(filtered)
        finite_logits = filtered[finite_mask]
        exp_vals = np.exp(finite_logits - finite_logits.max())
        probs = exp_vals / exp_vals.sum()
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_top_p_one_returns_unchanged(self) -> None:
        """p=1.0 should not filter any tokens."""
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = top_p_filter(logits, p=1.0)
        np.testing.assert_array_equal(result, logits)

    def test_top_p_filters_low_prob_tokens(self) -> None:
        """Very low-probability tokens should be filtered out with small p."""
        # Create a peaky distribution where one token dominates
        logits = np.array([10.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        filtered = top_p_filter(logits, p=0.9)
        # Token 0 has ~99.9% of probability mass, so p=0.9 should keep only it
        assert np.isfinite(filtered[0])
        # Others should be -inf
        assert np.all(np.isinf(filtered[1:]))

    def test_top_p_zero_raises(self) -> None:
        """p=0 should raise ValueError."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError):
            top_p_filter(logits, p=0.0)


class TestRepetitionPenalty:
    def test_repetition_penalty_reduces_repeated(self) -> None:
        """Positive logit for a repeated token should decrease after penalty."""
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        original_logit_2 = logits[2]
        penalized = apply_repetition_penalty(logits, generated_ids=[2], penalty=1.5)
        # Positive logit is divided by penalty -> decreases
        assert penalized[2] < original_logit_2
        assert penalized[2] == pytest.approx(original_logit_2 / 1.5)
        # Unpenalized tokens should be unchanged
        np.testing.assert_array_almost_equal(penalized[[0, 1, 3]], logits[[0, 1, 3]])

    def test_repetition_penalty_negative_logit(self) -> None:
        """Negative logit for a repeated token should become more negative (multiplied by penalty)."""
        logits = np.array([-3.0, 1.0, 2.0], dtype=np.float32)
        penalized = apply_repetition_penalty(logits, generated_ids=[0], penalty=1.5)
        # Negative logit is multiplied by penalty -> more negative
        assert penalized[0] == pytest.approx(-3.0 * 1.5)

    def test_repetition_penalty_one_is_identity(self) -> None:
        """penalty=1.0 should not change logits."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_repetition_penalty(logits, generated_ids=[0, 1, 2], penalty=1.0)
        np.testing.assert_array_equal(result, logits)

    def test_repetition_penalty_empty_generated(self) -> None:
        """No generated tokens -> logits unchanged."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_repetition_penalty(logits, generated_ids=[], penalty=1.5)
        np.testing.assert_array_equal(result, logits)

    def test_repetition_penalty_below_one_raises(self) -> None:
        """penalty < 1.0 should raise ValueError."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError):
            apply_repetition_penalty(logits, generated_ids=[0], penalty=0.5)


class TestSampleToken:
    def test_sample_token_returns_valid_index(self) -> None:
        """sample_token must return an index in [0, vocab_size)."""
        vocab_size = 50
        rng = np.random.default_rng(seed=42)
        logits = rng.normal(0, 1, vocab_size).astype(np.float32)
        rng2 = np.random.default_rng(seed=0)
        for _ in range(20):
            token = sample_token(logits, rng=rng2)
            assert 0 <= token < vocab_size

    def test_temperature_near_zero_is_greedy(self) -> None:
        """Very low temperature should consistently pick the argmax."""
        logits = np.array([0.1, 5.0, 0.3, 0.2], dtype=np.float32)
        rng = np.random.default_rng(seed=0)
        # With very low temperature, softmax is extremely peaked at argmax
        results = {sample_token(logits, temperature=0.0001, rng=rng) for _ in range(20)}
        assert results == {1}, f"Expected only token 1 (argmax), got {results}"

    def test_top_k_one_always_returns_argmax(self) -> None:
        """top_k=1 constrains to argmax regardless of temperature."""
        logits = np.array([0.1, 5.0, 0.3, 2.0], dtype=np.float32)
        rng = np.random.default_rng(seed=0)
        results = {sample_token(logits, temperature=1.0, top_k=1, rng=rng) for _ in range(10)}
        assert results == {1}

    def test_repetition_penalty_in_pipeline(self) -> None:
        """Repetition penalty in sample_token must reduce probability of repeated tokens."""
        # Set up logits where token 0 is clearly the best
        logits = np.array([10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rng = np.random.default_rng(seed=0)
        # Without penalty, token 0 should dominate
        no_penalty = [sample_token(logits, temperature=1.0, rng=rng) for _ in range(50)]
        rng = np.random.default_rng(seed=0)
        # With heavy penalty on token 0
        with_penalty = [
            sample_token(logits, temperature=1.0, repetition_penalty=10.0,
                         generated_ids=[0], rng=rng)
            for _ in range(50)
        ]
        count_0_no_penalty = no_penalty.count(0)
        count_0_with_penalty = with_penalty.count(0)
        assert count_0_no_penalty > count_0_with_penalty, (
            f"Penalty should reduce token 0 frequency: "
            f"{count_0_no_penalty} vs {count_0_with_penalty}"
        )
```

### tests/test_kv_cache.py

```python
# tests/test_kv_cache.py
"""
Tests for src/kv_cache.py

Run: uv run pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from kv_cache import KVCache


@pytest.fixture
def small_cache() -> KVCache:
    """A KVCache with small dimensions for testing."""
    return KVCache(n_layers=2, n_heads=2, d_head=4, dtype=np.dtype(np.float32))


class TestKVCacheInit:
    def test_initial_seq_len_is_zero(self, small_cache: KVCache) -> None:
        """Fresh cache should have seq_len == 0."""
        assert small_cache.seq_len() == 0

    def test_initial_memory_is_zero(self, small_cache: KVCache) -> None:
        """Fresh cache should use 0 bytes."""
        assert small_cache.memory_bytes() == 0

    def test_correct_number_of_layers(self, small_cache: KVCache) -> None:
        """Cache should have one entry per layer."""
        assert len(small_cache.cache) == 2


class TestKVCacheAppend:
    def test_kv_cache_append_grows(self, small_cache: KVCache) -> None:
        """Appending tokens should increase seq_len by 1 each time."""
        rng = np.random.default_rng(seed=42)
        for step in range(5):
            k = rng.normal(size=(1, 2, 4)).astype(np.float32)
            v = rng.normal(size=(1, 2, 4)).astype(np.float32)
            for layer in range(2):
                small_cache.append(layer, k, v)
            assert small_cache.seq_len() == step + 1

    def test_append_accepts_2d_input(self, small_cache: KVCache) -> None:
        """Append should accept (n_heads, d_head) and reshape correctly."""
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        small_cache.append(0, k, v)
        assert small_cache.seq_len() == 1

    def test_append_wrong_layer_raises(self, small_cache: KVCache) -> None:
        """Layer index out of range must raise IndexError."""
        k = np.ones((1, 2, 4), dtype=np.float32)
        v = np.ones((1, 2, 4), dtype=np.float32)
        with pytest.raises(IndexError):
            small_cache.append(5, k, v)

    def test_append_wrong_shape_raises(self, small_cache: KVCache) -> None:
        """Wrong n_heads or d_head must raise ValueError."""
        k_wrong = np.ones((1, 3, 4), dtype=np.float32)  # n_heads=3, expected 2
        v = np.ones((1, 2, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            small_cache.append(0, k_wrong, v)


class TestKVCacheGet:
    def test_kv_cache_get_returns_concatenated(self, small_cache: KVCache) -> None:
        """Get should return all appended k/v tensors concatenated along seq dim."""
        rng = np.random.default_rng(seed=0)
        k_list = []
        v_list = []
        for _ in range(3):
            k = rng.normal(size=(1, 2, 4)).astype(np.float32)
            v = rng.normal(size=(1, 2, 4)).astype(np.float32)
            k_list.append(k)
            v_list.append(v)
            small_cache.append(0, k, v)

        k_cached, v_cached = small_cache.get(0)
        assert k_cached.shape == (3, 2, 4), f"Expected (3, 2, 4), got {k_cached.shape}"
        assert v_cached.shape == (3, 2, 4)

        # Values should match what was appended
        expected_k = np.concatenate(k_list, axis=0)
        np.testing.assert_array_almost_equal(k_cached, expected_k)

    def test_get_wrong_layer_raises(self, small_cache: KVCache) -> None:
        """Get with out-of-range layer must raise IndexError."""
        with pytest.raises(IndexError):
            small_cache.get(99)


class TestKVCacheMemoryBytes:
    def test_kv_cache_memory_bytes(self, small_cache: KVCache) -> None:
        """Memory calculation should match seq_len * n_heads * d_head * bytes * 2 (k+v) * n_layers."""
        n_steps = 4
        k = np.ones((1, 2, 4), dtype=np.float32)
        v = np.ones((1, 2, 4), dtype=np.float32)
        for _ in range(n_steps):
            for layer in range(2):
                small_cache.append(layer, k, v)

        # Expected: 2 layers * 2 arrays (k,v) * 4 steps * 2 heads * 4 d_head * 4 bytes
        expected = 2 * 2 * n_steps * 2 * 4 * 4
        assert small_cache.memory_bytes() == expected

    def test_memory_scales_linearly_with_seq_len(self) -> None:
        """Memory usage should scale linearly with sequence length."""
        cache = KVCache(n_layers=1, n_heads=2, d_head=4, dtype=np.dtype(np.float32))
        k = np.ones((1, 2, 4), dtype=np.float32)
        v = np.ones((1, 2, 4), dtype=np.float32)

        mem_per_step = []
        for step in range(1, 6):
            cache.append(0, k, v)
            mem_per_step.append(cache.memory_bytes())

        # Each step should add the same amount of memory
        diffs = [mem_per_step[i] - mem_per_step[i - 1] for i in range(1, len(mem_per_step))]
        assert all(d == diffs[0] for d in diffs), f"Non-linear growth: {diffs}"


class TestKVCacheClear:
    def test_clear_resets_seq_len(self, small_cache: KVCache) -> None:
        """After clear(), seq_len should be 0."""
        k = np.ones((1, 2, 4), dtype=np.float32)
        v = np.ones((1, 2, 4), dtype=np.float32)
        for _ in range(3):
            for layer in range(2):
                small_cache.append(layer, k, v)

        assert small_cache.seq_len() == 3
        small_cache.clear()
        assert small_cache.seq_len() == 0
        assert small_cache.memory_bytes() == 0
```

---

## Running the project

From the `week-07/` directory:

```bash
# Run tests
uv run pytest tests/ -v

# Run demo
cd src && uv run python demo.py
```

Expected test output:
```
tests/test_sampling.py::TestGreedyDecode::test_greedy_selects_argmax PASSED
tests/test_sampling.py::TestGreedyDecode::test_greedy_with_negative_logits PASSED
tests/test_sampling.py::TestGreedyDecode::test_greedy_single_token PASSED
tests/test_sampling.py::TestTemperatureScale::test_temperature_one_is_identity PASSED
...
tests/test_kv_cache.py::TestKVCacheInit::test_initial_seq_len_is_zero PASSED
tests/test_kv_cache.py::TestKVCacheInit::test_initial_memory_is_zero PASSED
...
================================ 30 passed in X.XXs ================================
```

---

## Reflection questions

After completing all parts, consider:

1. The `generate()` function rebuilds the KV cache by running a forward pass for each prompt token sequentially. A real LLM inference server processes all prompt tokens in a single batched prefill step (O(T^2) once). How would you refactor `MockModel.get_logits` to support this? What would the API change look like?

2. `top_p_filter` has a subtle edge case: if a single token has probability > p, it is kept but the cumulative sum check might exclude it. Trace through the code for `logits = [100.0, 0.1, 0.1]` with `p = 0.5` and verify the behavior is correct.

3. The KV cache memory grows unboundedly. Real systems use techniques like **sliding window attention** (Mistral) or **KV cache eviction** (H2O). Design a `truncate(max_seq_len: int)` method for `KVCache` that drops the oldest tokens.

(Solutions: ../solutions/week-07/day-6.md)

---

## Self-check

- [ ] I implemented all six sampling functions with correct behavior verified by tests.
- [ ] I can explain why `apply_repetition_penalty` uses division for positive logits and multiplication for negative logits.
- [ ] I understand the KV cache's memory formula and can calculate it for a given model architecture.
- [ ] All 30 tests pass with `uv run pytest tests/ -v`.
- [ ] I can trace a full generation loop from prompt token IDs to output token IDs through the mock model.
