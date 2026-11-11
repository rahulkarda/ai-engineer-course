# Phase 3 · Week 07 · Day 3 — Autoregressive Decoding + KV Cache

> Objective: Understand autoregressive generation and how the KV cache eliminates redundant attention computation.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-2]]

## Concept

### Autoregressive Decoding

Language model generation is *autoregressive*: the model produces one token per forward pass, then feeds that token back as input for the next pass.

```
Input:  [T1, T2, T3]          -> forward pass -> predict T4
Input:  [T1, T2, T3, T4]      -> forward pass -> predict T5
Input:  [T1, T2, T3, T4, T5]  -> forward pass -> predict T6
...
```

Each forward pass runs the full transformer stack: embedding, n_layers of attention + FFN, output projection over vocabulary, softmax. For a sequence of length n, each forward pass computes attention over all n tokens.

### Naive cost: O(n^2) per step, O(n^3) total

At generation step t (sequence now has t tokens):
- Each of n_layers attention layers computes Q = XW_Q, K = XW_K, V = XW_V for all t tokens.
- Attention score matrix: t x t dot products.

Generating T tokens from a prompt of P tokens:
- Steps: T
- Attention matrix at step t: (P+t) x (P+t)
- Total attention FLOPs ~ sum_{t=1}^{T} (P+t)^2 ≈ O(T * (P+T)^2)

For P=2048, T=512: that is 512 * 2560^2 ≈ 3.3 billion element-wise operations per layer. Naively, all keys and values are recomputed from scratch at each step.

### KV Cache

The insight: in attention, Q_new = x_new W_Q only uses the *new* token's representation. But K and V for all previous tokens are *identical* across steps — they only depend on the token at that position, which never changes.

So: store K and V for all previous tokens. At step t:
1. Compute Q, K, V only for the *new* token (one row each).
2. Append K_new and V_new to the cache.
3. Compute attention: Q_new (shape: 1 x d_head) against full K cache (shape: t x d_head).
4. Weighted sum of V cache (shape: t x d_head) with the attention weights (shape: 1 x t).

**Per-step cost drops from O(t^2) to O(t)** (the t multiplications for the Q·K dot product are now over cached keys, but Q has only 1 row instead of t rows, so the matrix product is 1 x t instead of t x t).

### KV Cache Memory Cost

Each layer stores two tensors (K and V), each of shape [seq_len, n_heads, d_head]:

```
bytes = 2 * n_layers * seq_len * n_heads * d_head * bytes_per_element
```

Example — LLaMA-3 8B (approx numbers):
- n_layers = 32
- n_heads = 32, d_head = 128 (d_model = 4096)
- seq_len = 8192
- float16 (2 bytes)

```
2 * 32 * 8192 * 32 * 128 * 2 = 4,294,967,296 bytes ≈ 4 GB
```

For 128k context this scales to 64 GB — which is why KV cache eviction, quantization (INT8/INT4 KV), and sliding window attention are active research areas.

## How it works internally

**Why can K and V be cached but not Q?**
In causal (autoregressive) attention, token i attends only to tokens j <= i. For token i at step t, its Q is computed from its own embedding at that time step. But its K and V are computed from the same input embedding (position i never changes once generated). The "new" Q needs to attend to all past K and V — hence Q cannot be cached (it is the new query), but K and V can be.

**Multi-query attention (MQA) and grouped-query attention (GQA):**
LLaMA-2 and later models use GQA: instead of n_heads independent K/V heads, multiple Q heads share one K/V head. This directly reduces KV cache size by n_heads/n_kv_heads. Example: LLaMA-3 8B has 32 Q heads and 8 KV heads — cache is 4x smaller.

**Prefill vs decode phases:**
Modern inference engines split generation into two phases:
1. *Prefill*: process the entire prompt in one batched forward pass. All K and V for prompt tokens are computed and stored at once. This is compute-bound.
2. *Decode*: generate one token at a time using the cache. This is memory-bandwidth-bound (reading the full cache per step).

## Worked example

```python
"""
KV cache implementation in pure NumPy.
Simulates multi-head attention with a KV cache for autoregressive decoding.
Run: python day-3-example.py
Dependencies: numpy (uv add numpy)
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_head: int = 16      # d_model / n_heads
    vocab_size: int = 256
    max_seq_len: int = 128

    def __post_init__(self) -> None:
        assert self.d_head == self.d_model // self.n_heads


CFG = ModelConfig()


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

@dataclass
class KVCache:
    """
    Stores K and V tensors per layer for autoregressive decoding.
    Shape per layer: [seq_len, n_heads, d_head]
    """
    n_layers: int
    n_heads: int
    d_head: int

    # Each list element is (K, V) for one layer; initially None (not allocated)
    _cache: list[tuple[np.ndarray, np.ndarray] | None] = field(
        default_factory=list, repr=False
    )

    def __post_init__(self) -> None:
        self._cache = [None] * self.n_layers

    def append(
        self,
        layer_idx: int,
        k_new: np.ndarray,  # shape: [1, n_heads, d_head]
        v_new: np.ndarray,  # shape: [1, n_heads, d_head]
    ) -> None:
        """Append K and V for one new token to the specified layer's cache."""
        assert k_new.shape == v_new.shape == (1, self.n_heads, self.d_head)
        entry = self._cache[layer_idx]
        if entry is None:
            self._cache[layer_idx] = (k_new.copy(), v_new.copy())
        else:
            k_cached, v_cached = entry
            self._cache[layer_idx] = (
                np.concatenate([k_cached, k_new], axis=0),
                np.concatenate([v_cached, v_new], axis=0),
            )

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (K, V) tensors of shape [seq_len, n_heads, d_head] for a layer."""
        return self._cache[layer_idx]

    def seq_len(self) -> int:
        """Return current cached sequence length (0 if empty)."""
        entry = self._cache[0]
        if entry is None:
            return 0
        return entry[0].shape[0]

    def memory_bytes(self, bytes_per_element: int = 4) -> int:
        """Total bytes used by the KV cache."""
        total = 0
        for entry in self._cache:
            if entry is not None:
                k, v = entry
                total += k.nbytes + v.nbytes
        return total


# ---------------------------------------------------------------------------
# Attention with and without KV cache
# ---------------------------------------------------------------------------

def attention_naive(
    x: np.ndarray,      # [seq_len, d_model]
    W_Q: np.ndarray,    # [d_model, n_heads * d_head]
    W_K: np.ndarray,
    W_V: np.ndarray,
) -> np.ndarray:
    """
    Full multi-head attention without cache.
    Computes Q, K, V for ALL tokens — O(n^2) in memory for the attention matrix.
    Returns: [seq_len, d_model]
    """
    seq_len = x.shape[0]
    n_heads = CFG.n_heads
    d_head = CFG.d_head
    scale = 1.0 / math.sqrt(d_head)

    # Project [seq_len, d_model] -> [seq_len, n_heads * d_head]
    Q = (x @ W_Q).reshape(seq_len, n_heads, d_head)  # [seq_len, n_heads, d_head]
    K = (x @ W_K).reshape(seq_len, n_heads, d_head)
    V = (x @ W_V).reshape(seq_len, n_heads, d_head)

    # Per-head attention: Q[h] @ K[h].T -> [seq_len, seq_len]
    out_heads = []
    for h in range(n_heads):
        scores = Q[:, h, :] @ K[:, h, :].T * scale      # [seq_len, seq_len]
        # Causal mask: upper triangle = -inf
        mask = np.triu(np.full_like(scores, -np.inf), k=1)
        scores = scores + mask
        weights = softmax(scores, axis=-1)               # [seq_len, seq_len]
        out_heads.append(weights @ V[:, h, :])           # [seq_len, d_head]

    return np.concatenate(out_heads, axis=-1)             # [seq_len, n_heads * d_head]


def attention_cached(
    x_new: np.ndarray,      # [1, d_model] — only the NEW token
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    cache: KVCache,
    layer_idx: int,
) -> np.ndarray:
    """
    Single-step attention using KV cache.
    Compute Q, K, V only for x_new, append to cache, attend over full cache.
    Returns: [1, d_model]
    """
    d_head = CFG.d_head
    n_heads = CFG.n_heads
    scale = 1.0 / math.sqrt(d_head)

    # New token projections: shape [1, n_heads, d_head]
    q_new = (x_new @ W_Q).reshape(1, n_heads, d_head)
    k_new = (x_new @ W_K).reshape(1, n_heads, d_head)
    v_new = (x_new @ W_V).reshape(1, n_heads, d_head)

    # Update cache
    cache.append(layer_idx, k_new, v_new)
    k_full, v_full = cache.get(layer_idx)  # [seq_len, n_heads, d_head]

    # Attention: q_new attends to full k_full
    out_heads = []
    for h in range(n_heads):
        q_h = q_new[:, h, :]     # [1, d_head]
        k_h = k_full[:, h, :]    # [seq_len, d_head]
        v_h = v_full[:, h, :]    # [seq_len, d_head]
        scores = q_h @ k_h.T * scale              # [1, seq_len]
        weights = softmax(scores, axis=-1)        # [1, seq_len]
        out_heads.append(weights @ v_h)            # [1, d_head]

    return np.concatenate(out_heads, axis=-1)      # [1, n_heads * d_head]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Mock token embeddings and weight matrices
# ---------------------------------------------------------------------------

def make_weights(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Initialize random projection matrices for one layer."""
    d = CFG.d_model
    h = CFG.n_heads * CFG.d_head
    return {
        "W_Q": rng.standard_normal((d, h)).astype(np.float32) * 0.02,
        "W_K": rng.standard_normal((d, h)).astype(np.float32) * 0.02,
        "W_V": rng.standard_normal((d, h)).astype(np.float32) * 0.02,
    }


def embed_tokens(token_ids: list[int], rng: np.random.Generator) -> np.ndarray:
    """Mock embedding: random but deterministic per token id."""
    embeddings = []
    for tid in token_ids:
        # Deterministic per token id: seed from id
        local_rng = np.random.default_rng(tid)
        embeddings.append(local_rng.standard_normal(CFG.d_model).astype(np.float32))
    return np.stack(embeddings)  # [seq_len, d_model]


# ---------------------------------------------------------------------------
# Demonstration: naive vs cached, with memory comparison
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    weights_per_layer = [make_weights(rng) for _ in range(CFG.n_layers)]

    # Simulate a prompt + generation sequence
    prompt_ids = [10, 20, 30, 40]   # 4 prompt tokens
    gen_ids = [50, 60, 70]          # 3 generated tokens
    all_ids = prompt_ids + gen_ids

    logger.info("=== Naive vs Cached Attention ===")
    logger.info("Config: %d layers, %d heads, d_model=%d, d_head=%d",
                CFG.n_layers, CFG.n_heads, CFG.d_model, CFG.d_head)

    # --- Naive: recompute everything at each generation step ---
    naive_flop_counts = []
    for t in range(len(prompt_ids), len(all_ids) + 1):
        # sequence length at step t
        seq_len = t
        # attention matrix size per head per layer: seq_len * seq_len
        attn_ops = seq_len * seq_len * CFG.n_heads * CFG.n_layers
        naive_flop_counts.append(attn_ops)
        logger.info("Naive step t=%d: seq_len=%d, attention ops=%d", t, seq_len, attn_ops)

    # --- Cached: only new token's Q against full K cache ---
    logger.info("")
    cache = KVCache(n_layers=CFG.n_layers, n_heads=CFG.n_heads, d_head=CFG.d_head)

    # Prefill: process all prompt tokens (would normally be batched)
    logger.info("Prefill: processing %d prompt tokens", len(prompt_ids))
    prompt_embeddings = embed_tokens(prompt_ids, rng)
    for layer_idx, w in enumerate(weights_per_layer):
        for i, tok_emb in enumerate(prompt_embeddings):
            tok_emb_2d = tok_emb[np.newaxis, :]  # [1, d_model]
            attention_cached(tok_emb_2d, w["W_Q"], w["W_K"], w["W_V"], cache, layer_idx)
    logger.info("Cache after prefill: seq_len=%d, memory=%d bytes",
                cache.seq_len(), cache.memory_bytes())

    # Decode: generate tokens one at a time
    cached_flop_counts = []
    for t, gen_id in enumerate(gen_ids, start=1):
        tok_emb = embed_tokens([gen_id], rng)  # [1, d_model]
        for layer_idx, w in enumerate(weights_per_layer):
            attention_cached(tok_emb, w["W_Q"], w["W_K"], w["W_V"], cache, layer_idx)
        seq_len = cache.seq_len()
        # Cached attention ops: 1 * seq_len per head per layer (Q is size 1)
        attn_ops = 1 * seq_len * CFG.n_heads * CFG.n_layers
        cached_flop_counts.append(attn_ops)
        logger.info("Cached step t=%d: seq_len=%d, attention ops=%d, cache memory=%d bytes",
                    t, seq_len, attn_ops, cache.memory_bytes())

    logger.info("")
    logger.info("=== Memory cost formula ===")
    logger.info(
        "KV cache bytes = 2 * n_layers * seq_len * n_heads * d_head * bytes_per_element"
    )
    for seq_len in [512, 2048, 8192, 32768]:
        cost = 2 * CFG.n_layers * seq_len * CFG.n_heads * CFG.d_head * 2  # float16
        logger.info("  seq_len=%6d -> %8d bytes (%.1f KB)", seq_len, cost, cost / 1024)

    logger.info("")
    logger.info("=== Comparison at last step ===")
    logger.info("Naive attention ops (last step): %d", naive_flop_counts[-1])
    logger.info("Cached attention ops (last step): %d", cached_flop_counts[-1])
    ratio = naive_flop_counts[-1] / max(cached_flop_counts[-1], 1)
    logger.info("Speedup factor (ops): %.1fx", ratio)


if __name__ == "__main__":
    main()
```

Expected output (abridged):
```
INFO: Naive step t=4: seq_len=4, attention ops=256
INFO: Naive step t=5: seq_len=5, attention ops=400
...
INFO: Cached step t=1: seq_len=5, attention ops=40, cache memory=2048 bytes
INFO: Cached step t=2: seq_len=6, attention ops=48, cache memory=2304 bytes
```

Note: install numpy with `uv add numpy` before running.

## Exercises

1. **Memory calculation**: A production model has 40 layers, 40 attention heads, d_head = 128, and uses float16 (2 bytes per element). Calculate the KV cache size in GB for a sequence length of 32,768 tokens. Show your work using the formula. Then compute how many such sequences fit in 80 GB of GPU VRAM alongside 14 GB of model weights.

2. **Implement cache append**: Extend the `KVCache` class with a `trim(max_len: int)` method that keeps only the most recent `max_len` tokens in the cache (sliding window). Write a test that verifies after trimming to max_len=4, `cache.seq_len()` returns 4 and earlier tokens are removed.

3. **Cache-aware attention**: Modify `attention_cached` to support grouped-query attention (GQA): accept an `n_kv_heads` parameter where `n_kv_heads < n_heads`. Each K/V head is shared by `n_heads // n_kv_heads` Q heads. Update `KVCache` accordingly and verify the output shape is still `[1, n_heads * d_head]`.

(Solutions: ../solutions/week-07/day-3.md)

## Self-check

- [ ] I can explain why naive autoregressive attention is O(n^2) per generation step and O(n^3) total for a sequence of n tokens.
- [ ] I can state why K and V can be cached but Q cannot.
- [ ] I can write the KV cache memory formula and apply it to a realistic model configuration.
- [ ] I can implement a KVCache class with append and get methods, and demonstrate it reduces per-step attention ops.

## Supplementary reading

- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- Pope et al., "Efficiently Scaling Transformer Inference" (introduces prefill/decode split analysis): https://arxiv.org/abs/2211.05100
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints": https://arxiv.org/abs/2305.13245
- Dao et al., "FlashAttention-2": https://arxiv.org/abs/2307.08691
