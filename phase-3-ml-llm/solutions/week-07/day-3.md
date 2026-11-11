# Solutions — Phase 3 · Week 07 · Day 3

## Exercise 1 — KV Cache Memory Calculation

**Given:**
- Layers: 32
- Attention heads: 32
- Head dimension (d_head): 128
- Precision: FP16 = 2 bytes per element
- Sequence length: 2048

**Calculation:**

Each layer stores a K tensor and a V tensor.

Shape of K (or V) per layer:
```
[sequence_length, num_heads, d_head] = [2048, 32, 128]
```

Elements per K or V tensor:
```
2048 x 32 x 128 = 8,388,608 elements
```

Bytes per K or V tensor (FP16):
```
8,388,608 x 2 bytes = 16,777,216 bytes
```

Bytes per layer (K + V):
```
16,777,216 x 2 = 33,554,432 bytes
```

Total across all 32 layers:
```
33,554,432 x 32 = 1,073,741,824 bytes
```

Convert to megabytes (1 MB = 1,048,576 bytes):
```
1,073,741,824 / 1,048,576 = 1024 MB = 1 GB
```

**Answer: 1024 MB (1 GB)**

**Formula in general:**
```
KV cache bytes = 2 (K and V)
              x num_layers
              x num_heads
              x d_head
              x seq_len
              x bytes_per_element
```

```
= 2 x 32 x 32 x 128 x 2048 x 2
= 1,073,741,824 bytes
= 1024 MB
```

**Practical implication:** For a batch of 8 sequences, KV cache requires 8 GB — comparable to the weight memory for a 7B FP16 model (~14 GB). At longer sequences (e.g., 128K tokens), KV cache memory becomes the dominant constraint: 1024 MB x (128000/2048) = 64 GB per batch item.

---

## Exercise 2 — KV Cache Append Method

```python
"""
KVCache implementation with append method using numpy.

The cache stores key and value tensors for each layer. During autoregressive
decoding, each new token adds one row to the K and V tensors.

Shape convention:
    K, V: (seq_len, num_heads, d_head)
    New K, V (one new token): (1, num_heads, d_head)

Dependencies: numpy
Run: python day-3-solution-ex2.py
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class KVCache:
    """
    Key-Value cache for transformer attention layers.

    Stores accumulated K and V tensors for all layers.
    Grows by 1 along the sequence dimension with each new token.

    Attributes:
        num_layers: number of transformer layers
        num_heads: number of attention heads per layer
        d_head: dimension of each attention head

    Internal state:
        keys:   list of K tensors, one per layer, shape (seq_len, num_heads, d_head)
        values: list of V tensors, one per layer, shape (seq_len, num_heads, d_head)
    """

    num_layers: int
    num_heads: int
    d_head: int
    keys: list[np.ndarray] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize empty cache (zero-length sequence for each layer)."""
        for _ in range(self.num_layers):
            self.keys.append(
                np.empty((0, self.num_heads, self.d_head), dtype=np.float16)
            )
            self.values.append(
                np.empty((0, self.num_heads, self.d_head), dtype=np.float16)
            )
        logger.debug(
            "KVCache initialized: %d layers, %d heads, d_head=%d",
            self.num_layers, self.num_heads, self.d_head,
        )

    def append(
        self,
        layer_idx: int,
        new_k: np.ndarray,
        new_v: np.ndarray,
    ) -> None:
        """
        Append new key and value tensors for one new token to the specified layer.

        Args:
            layer_idx: which transformer layer (0-indexed)
            new_k: new key tensor, shape (1, num_heads, d_head) or (num_heads, d_head)
            new_v: new value tensor, shape (1, num_heads, d_head) or (num_heads, d_head)

        Raises:
            ValueError: if layer_idx is out of range or tensor shapes are wrong
        """
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"layer_idx={layer_idx} out of range [0, {self.num_layers})"
            )

        # Accept both (num_heads, d_head) and (1, num_heads, d_head) shapes
        if new_k.ndim == 2:
            new_k = new_k[np.newaxis, :, :]  # (1, num_heads, d_head)
        if new_v.ndim == 2:
            new_v = new_v[np.newaxis, :, :]  # (1, num_heads, d_head)

        expected_shape = (1, self.num_heads, self.d_head)
        if new_k.shape != expected_shape:
            raise ValueError(
                f"new_k shape {new_k.shape} != expected {expected_shape}"
            )
        if new_v.shape != expected_shape:
            raise ValueError(
                f"new_v shape {new_v.shape} != expected {expected_shape}"
            )

        # Concatenate along sequence dimension (axis=0)
        self.keys[layer_idx] = np.concatenate(
            [self.keys[layer_idx], new_k.astype(np.float16)], axis=0
        )
        self.values[layer_idx] = np.concatenate(
            [self.values[layer_idx], new_v.astype(np.float16)], axis=0
        )

        logger.debug(
            "Layer %d cache after append: K shape=%s, V shape=%s",
            layer_idx,
            self.keys[layer_idx].shape,
            self.values[layer_idx].shape,
        )

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the full cached K and V tensors for a layer.

        Returns:
            (K, V): both of shape (current_seq_len, num_heads, d_head)
        """
        return self.keys[layer_idx], self.values[layer_idx]

    def seq_len(self, layer_idx: int = 0) -> int:
        """Return current cached sequence length."""
        return self.keys[layer_idx].shape[0]

    def memory_bytes(self) -> int:
        """Return total memory used by the cache in bytes."""
        total = sum(k.nbytes + v.nbytes for k, v in zip(self.keys, self.values, strict=True))
        return total

    def evict_oldest(self, n_tokens: int) -> None:
        """
        Remove the n_tokens oldest tokens from all layers.

        Used when the sequence exceeds the context window limit.
        """
        for layer_idx in range(self.num_layers):
            self.keys[layer_idx] = self.keys[layer_idx][n_tokens:]
            self.values[layer_idx] = self.values[layer_idx][n_tokens:]
        logger.debug("Evicted %d tokens from all layers", n_tokens)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    num_layers = 4
    num_heads = 8
    d_head = 64
    rng = np.random.default_rng(seed=42)

    cache = KVCache(num_layers=num_layers, num_heads=num_heads, d_head=d_head)
    logger.info("Initial seq_len: %d", cache.seq_len())
    logger.info("Initial memory: %d bytes", cache.memory_bytes())

    # Simulate 5 decode steps: each step appends one new K/V per layer
    for step in range(5):
        for layer_idx in range(num_layers):
            # In production: these come from the attention projection of the new token
            new_k = rng.standard_normal((1, num_heads, d_head)).astype(np.float32)
            new_v = rng.standard_normal((1, num_heads, d_head)).astype(np.float32)
            cache.append(layer_idx, new_k, new_v)

        logger.info(
            "After step %d: seq_len=%d, memory=%d bytes (%.2f KB)",
            step + 1,
            cache.seq_len(),
            cache.memory_bytes(),
            cache.memory_bytes() / 1024,
        )

    # Verify shapes
    k, v = cache.get(layer_idx=0)
    assert k.shape == (5, num_heads, d_head), f"Unexpected K shape: {k.shape}"
    assert v.shape == (5, num_heads, d_head), f"Unexpected V shape: {v.shape}"
    logger.info("Shape verification passed: K=%s, V=%s", k.shape, v.shape)

    # Test eviction
    cache.evict_oldest(2)
    logger.info("After evicting 2 oldest tokens: seq_len=%d", cache.seq_len())
    assert cache.seq_len() == 3


if __name__ == "__main__":
    main()
```

---

## Exercise 3 — Cache-Aware Single-Head Attention

```python
"""
Cache-aware single-head attention for autoregressive decoding.

During autoregressive generation, we only have a query for the NEW token
(shape [1, d_k]). The keys and values from all previous tokens are stored
in the KV cache (shape [seq_len, d_k] and [seq_len, d_v]).

The attention output for the new token is computed against the full cache.

This is the core of the decode phase in a transformer: one forward pass,
O(seq_len) attention, no recomputation of K/V for previous tokens.

Dependencies: numpy
Run: python day-3-solution-ex3.py
"""

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def softmax_1d(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax for a 1D array.

    Subtracts max to prevent exp overflow.

    Args:
        x: 1D array of logits, shape (N,)

    Returns:
        probability distribution, shape (N,), sums to 1.0
    """
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)


def cache_aware_single_head_attention(
    query: np.ndarray,
    cached_keys: np.ndarray,
    cached_values: np.ndarray,
) -> np.ndarray:
    """
    Single-head attention for one new token using the KV cache.

    During autoregressive decoding, only the query for the new token is
    computed. The K and V tensors for all previous tokens are retrieved
    from the cache. This avoids O(T^2) recomputation: each decode step
    is O(T) where T is the current sequence length.

    Args:
        query: query vector for the new token, shape (1, d_k) or (d_k,)
        cached_keys: all previous keys, shape (seq_len, d_k)
        cached_values: all previous values, shape (seq_len, d_v)

    Returns:
        attention output for the new token, shape (1, d_v) or (d_v,)
        (matches input query rank)

    Notes:
    - Scaling: divide attention scores by sqrt(d_k) to prevent softmax
      saturation in high-dimensional spaces (Vaswani et al., 2017).
    - No causal masking needed here because we are decoding one token at
      a time — the cache only contains past tokens by construction.
    - In multi-head attention, this function is called independently for
      each head; outputs are concatenated and projected.
    """
    # Normalize input shape
    squeezed = query.ndim == 1
    if query.ndim == 1:
        query = query[np.newaxis, :]  # (1, d_k)

    if query.shape[0] != 1:
        raise ValueError(
            f"Expected single-token query shape (1, d_k), got {query.shape}. "
            "Use batched attention for multiple queries."
        )

    d_k: int = query.shape[1]
    seq_len: int = cached_keys.shape[0]

    if cached_keys.shape != (seq_len, d_k):
        raise ValueError(
            f"cached_keys shape {cached_keys.shape} incompatible with "
            f"query d_k={d_k}"
        )
    if cached_values.shape[0] != seq_len:
        raise ValueError(
            f"cached_values seq_len {cached_values.shape[0]} != "
            f"cached_keys seq_len {seq_len}"
        )

    # Step 1: compute attention scores
    # query: (1, d_k), cached_keys^T: (d_k, seq_len)
    # scores: (1, seq_len) -> (seq_len,)
    scale: float = 1.0 / np.sqrt(d_k)
    scores: np.ndarray = (query @ cached_keys.T).squeeze(0) * scale  # (seq_len,)

    logger.debug(
        "Attention scores [min=%.3f, max=%.3f, mean=%.3f]",
        float(scores.min()), float(scores.max()), float(scores.mean()),
    )

    # Step 2: softmax over sequence dimension
    attention_weights: np.ndarray = softmax_1d(scores)  # (seq_len,)

    logger.debug(
        "Attention weights [min=%.4f, max=%.4f, sum=%.6f]",
        float(attention_weights.min()),
        float(attention_weights.max()),
        float(attention_weights.sum()),
    )

    # Step 3: weighted sum of values
    # attention_weights: (seq_len,), cached_values: (seq_len, d_v)
    # output: (d_v,)
    output: np.ndarray = attention_weights @ cached_values  # (d_v,)

    # Restore rank to match query input
    if not squeezed:
        output = output[np.newaxis, :]  # (1, d_v)

    return output


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(seed=42)

    d_k: int = 64
    d_v: int = 64
    seq_len: int = 10  # 10 tokens already in cache

    # Simulate: previous tokens have been encoded into the cache
    cached_keys = rng.standard_normal((seq_len, d_k)).astype(np.float32)
    cached_values = rng.standard_normal((seq_len, d_v)).astype(np.float32)

    # New token's query (computed from its embedding + projection)
    new_query = rng.standard_normal((1, d_k)).astype(np.float32)

    logger.info("Query shape:         %s", new_query.shape)
    logger.info("Cached keys shape:   %s", cached_keys.shape)
    logger.info("Cached values shape: %s", cached_values.shape)

    output = cache_aware_single_head_attention(new_query, cached_keys, cached_values)

    logger.info("Output shape:        %s", output.shape)
    logger.info("Output norm:         %.4f", float(np.linalg.norm(output)))
    logger.info(
        "Output (first 8 dims): %s",
        np.round(output.squeeze()[:8], 4),
    )

    # Verify output is a valid weighted combination of values
    # Output should lie within the convex hull of cached_values rows
    # (bounded by min/max of values, since weights sum to 1)
    v_min = float(cached_values.min())
    v_max = float(cached_values.max())
    out_min = float(output.min())
    out_max = float(output.max())
    logger.info(
        "Output in value range [%.3f, %.3f]: %s",
        v_min, v_max,
        v_min <= out_min and out_max <= v_max,
    )

    # Show which past token received the most attention
    scale = 1.0 / np.sqrt(d_k)
    scores = (new_query @ cached_keys.T).squeeze() * scale
    weights = softmax_1d(scores)
    top_token = int(np.argmax(weights))
    logger.info(
        "Most attended past token: position %d (weight=%.4f)",
        top_token, float(weights[top_token]),
    )

    # Verify 1D query input also works
    output_1d = cache_aware_single_head_attention(
        new_query.squeeze(), cached_keys, cached_values
    )
    logger.info("1D query input works: output shape=%s", output_1d.shape)
    assert np.allclose(output.squeeze(), output_1d, atol=1e-6), "1D and 2D outputs differ"
    logger.info("1D/2D output consistency: OK")


if __name__ == "__main__":
    main()
```

Expected output (values vary by rng seed but shapes and checks are deterministic):
```
INFO: Query shape:         (1, 64)
INFO: Cached keys shape:   (10, 64)
INFO: Cached values shape: (10, 64)
INFO: Output shape:        (1, 64)
INFO: Output norm:         ...
INFO: Output in value range [...]: True
INFO: Most attended past token: position X (weight=...)
INFO: 1D/2D output consistency: OK
```

**Key implementation notes:**

1. **Scaling by 1/sqrt(d_k)**: without this, for large d_k the dot products grow large, pushing the softmax into near-zero gradient regions (saturation). The scale ensures attention scores have unit variance when queries and keys are unit-normal.

2. **No causal mask**: during decode, the cache contains only past tokens by construction. The new query can attend to all of them freely. Causal masking is only needed during prefill (processing the full input sequence in parallel), where future tokens must be masked from past positions.

3. **O(T) per step**: the query-key dot product is (1, d_k) @ (d_k, T) = (1, T), and the weighted sum is (1, T) @ (T, d_v) = (1, d_v). Both are O(T * d_k) and O(T * d_v) — linear in sequence length. Without the KV cache, we would recompute all K and V projections (O(T) each) and recompute all pairwise dot products O(T^2).
