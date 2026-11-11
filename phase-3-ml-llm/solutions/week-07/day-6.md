# Solutions — Phase 3 · Week 07 · Day 6 — Saturday Build Day

This file contains complete, runnable implementations for both build-day modules and their full test suites.

---

## Section 1: src/sampling.py

```python
"""Sampling strategies for autoregressive language model decoding."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def greedy_decode(logits: np.ndarray) -> int:
    """Return the token id with the highest logit (argmax).

    Args:
        logits: Raw logit scores, shape [vocab_size].

    Returns:
        Index of the maximum logit.
    """
    token_id = int(np.argmax(logits))
    logger.debug("greedy_decode -> token %d (logit=%.4f)", token_id, logits[token_id])
    return token_id


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by temperature before softmax.

    temperature < 1.0 sharpens the distribution (more confident sampling).
    temperature > 1.0 flattens the distribution (more random sampling).
    As temperature approaches 0 the distribution collapses to argmax; we
    clamp at 1e-8 to avoid division by zero.

    Args:
        logits: Raw logit scores, shape [vocab_size].
        temperature: Scaling factor. Must be > 0.

    Returns:
        Temperature-scaled logits (not yet softmaxed).

    Raises:
        ValueError: If temperature <= 0.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    temp = max(temperature, 1e-8)
    scaled = logits / temp
    logger.debug(
        "temperature_scale T=%.3f: logit range [%.3f, %.3f] -> [%.3f, %.3f]",
        temperature,
        float(logits.min()),
        float(logits.max()),
        float(scaled.min()),
        float(scaled.max()),
    )
    return scaled


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Set all logits outside the top-k to -inf.

    Args:
        logits: Raw logit scores, shape [vocab_size].
        k: Number of top tokens to keep. If k <= 0, returns unchanged logits.

    Returns:
        Filtered logits with non-top-k positions set to -inf.
    """
    if k <= 0:
        return logits.copy()
    vocab_size = logits.shape[0]
    k = min(k, vocab_size)
    # argpartition is O(n) vs argsort O(n log n)
    top_k_indices = np.argpartition(logits, -k)[-k:]
    filtered = np.full_like(logits, -np.inf)
    filtered[top_k_indices] = logits[top_k_indices]
    logger.debug("top_k_filter k=%d: kept %d tokens", k, k)
    return filtered


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest set of tokens with cumulative prob >= p.

    Algorithm:
    1. Convert logits to probabilities via stable softmax.
    2. Sort tokens by probability descending.
    3. Compute cumulative sum.
    4. Remove tokens where the cumulative sum BEFORE including that token
       already exceeds p — this ensures the nucleus always sums to at least p.

    Args:
        logits: Raw logit scores, shape [vocab_size].
        p: Cumulative probability threshold in (0, 1]. 1.0 = no filtering.

    Returns:
        Filtered logits with tokens outside the nucleus set to -inf.
    """
    if p >= 1.0:
        return logits.copy()

    # Numerically stable softmax
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum()

    # Sort descending by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # cumsum[i] = total prob of tokens 0..i inclusive
    cumsum = np.cumsum(sorted_probs)

    # Remove tokens where the cumulative sum BEFORE this token exceeds p.
    # This retains the token that first pushes cumsum over p.
    cutoff_mask = (cumsum - sorted_probs) > p
    remove_indices = sorted_indices[cutoff_mask]

    filtered = logits.copy()
    filtered[remove_indices] = -np.inf
    nucleus_size = len(sorted_indices) - int(cutoff_mask.sum())
    logger.debug(
        "top_p_filter p=%.3f: nucleus size=%d out of %d",
        p,
        nucleus_size,
        len(logits),
    )
    return filtered


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float,
) -> np.ndarray:
    """Discourage repetition by penalizing logits of already-generated tokens.

    For a token that has already been generated:
    - If its logit is positive, divide by penalty (reduces attraction).
    - If its logit is negative, multiply by penalty (increases repulsion).

    This asymmetric formulation preserves the sign, consistent with the
    Keskar et al. implementation used in most production inference engines.

    Args:
        logits: Raw logit scores, shape [vocab_size].
        generated_ids: Token IDs already generated in this sequence.
        penalty: Penalty factor >= 1.0. 1.0 = no penalty.

    Returns:
        Logits with repetition penalty applied.

    Raises:
        ValueError: If penalty < 1.0.
    """
    if penalty == 1.0 or not generated_ids:
        return logits.copy()
    if penalty < 1.0:
        raise ValueError(f"penalty must be >= 1.0, got {penalty}")

    penalized = logits.copy()
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if 0 <= token_id < len(logits):
            if penalized[token_id] > 0:
                penalized[token_id] /= penalty
            else:
                penalized[token_id] *= penalty
    logger.debug(
        "apply_repetition_penalty: penalized %d unique tokens with factor=%.2f",
        len(unique_ids),
        penalty,
    )
    return penalized


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum()


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
) -> int:
    """Unified sampling: apply all strategies in order, then sample.

    Order of operations:
    1. Apply repetition penalty to raw logits.
    2. Apply temperature scaling.
    3. Apply top-k filter (if top_k > 0).
    4. Apply top-p (nucleus) filter (if top_p < 1.0).
    5. Convert filtered logits to probability distribution via softmax.
    6. Sample from distribution. If temperature < 1e-7, fall back to greedy.

    Args:
        logits: Raw logit scores, shape [vocab_size].
        temperature: Temperature for scaling. Default 1.0 (no scaling effect
            on relative probabilities beyond what's already in logits).
        top_k: Keep top-k tokens. 0 = no filtering.
        top_p: Nucleus probability. 1.0 = no filtering.
        repetition_penalty: Penalty for repeated tokens. 1.0 = no penalty.
        generated_ids: Previously generated token IDs for repetition penalty.

    Returns:
        Sampled token ID.
    """
    if generated_ids is None:
        generated_ids = []

    processed = logits.copy()

    # Step 1: repetition penalty applied to raw logits before any scaling
    processed = apply_repetition_penalty(processed, generated_ids, repetition_penalty)

    # Step 2: near-zero temperature -> greedy (short-circuit before top_p)
    if temperature < 1e-7:
        return greedy_decode(processed)
    processed = temperature_scale(processed, temperature)

    # Step 3: top-k filter
    if top_k > 0:
        processed = top_k_filter(processed, top_k)

    # Step 4: top-p filter
    if top_p < 1.0:
        processed = top_p_filter(processed, top_p)

    # Step 5: probability distribution
    probs = _softmax(processed)

    # Step 6: categorical sample
    vocab_size = len(probs)
    token_id = int(np.random.choice(vocab_size, p=probs))
    logger.debug("sample_token -> %d (prob=%.4f)", token_id, probs[token_id])
    return token_id
```

---

## Section 2: src/kv_cache.py

```python
"""Key-Value cache for efficient autoregressive transformer decoding."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KVCache:
    """Per-layer key-value cache for transformer inference.

    During autoregressive generation, the K and V projections for all past
    tokens are identical across decode steps. This cache stores them once
    so each new token only needs to compute Q for the current step, reducing
    per-step compute from O(T) to O(1) matmuls (at the cost of O(T) memory).

    Attributes:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        d_head: Dimension per attention head.
        dtype: NumPy dtype for stored tensors. Default float32.
    """

    n_layers: int
    n_heads: int
    d_head: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    _cache: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize empty K/V arrays for each layer.

        Each layer's K and V tensors start with shape [n_heads, 0, d_head].
        The sequence-length dimension grows by 1 on each call to append().
        """
        self._cache = [
            {
                "k": np.empty((self.n_heads, 0, self.d_head), dtype=self.dtype),
                "v": np.empty((self.n_heads, 0, self.d_head), dtype=self.dtype),
            }
            for _ in range(self.n_layers)
        ]
        logger.debug(
            "KVCache initialized: %d layers, %d heads, d_head=%d, dtype=%s",
            self.n_layers,
            self.n_heads,
            self.d_head,
            self.dtype,
        )

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append new K/V tensors for one new token in one layer.

        Each call concatenates along the sequence dimension (axis=1), growing
        the stored cache by one position.

        Args:
            layer_idx: Layer index, 0-indexed. Must be in [0, n_layers).
            k: New key tensor, shape [n_heads, 1, d_head].
            v: New value tensor, shape [n_heads, 1, d_head].

        Raises:
            IndexError: If layer_idx is out of range.
            ValueError: If k or v shape does not match expected.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )
        expected_shape = (self.n_heads, 1, self.d_head)
        if k.shape != expected_shape:
            raise ValueError(f"k shape {k.shape} != expected {expected_shape}")
        if v.shape != expected_shape:
            raise ValueError(f"v shape {v.shape} != expected {expected_shape}")

        self._cache[layer_idx]["k"] = np.concatenate(
            [self._cache[layer_idx]["k"], k], axis=1
        )
        self._cache[layer_idx]["v"] = np.concatenate(
            [self._cache[layer_idx]["v"], v], axis=1
        )
        logger.debug(
            "KVCache.append layer=%d, new seq_len=%d", layer_idx, self.seq_len()
        )

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the full accumulated K/V cache for one layer.

        Args:
            layer_idx: Layer index, 0-indexed. Must be in [0, n_layers).

        Returns:
            Tuple of (K, V), each with shape [n_heads, seq_len, d_head].

        Raises:
            IndexError: If layer_idx is out of range.
        """
        if not (0 <= layer_idx < self.n_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.n_layers})"
            )
        layer = self._cache[layer_idx]
        return layer["k"], layer["v"]

    def memory_bytes(self) -> int:
        """Total memory used by all cached K and V tensors in bytes.

        Formula: 2 * n_layers * n_heads * seq_len * d_head * bytes_per_element

        The factor of 2 accounts for both K and V being stored.
        """
        bytes_per_element = np.dtype(self.dtype).itemsize
        seq_len = self.seq_len()
        return (
            2
            * self.n_layers
            * self.n_heads
            * seq_len
            * self.d_head
            * bytes_per_element
        )

    def seq_len(self) -> int:
        """Current number of token positions stored in the cache."""
        if not self._cache:
            return 0
        return int(self._cache[0]["k"].shape[1])

    def clear(self) -> None:
        """Reset all cache entries to empty, as if just initialized."""
        self.__post_init__()
        logger.debug("KVCache cleared")
```

---

## Section 3: tests/test_sampling.py

```python
"""Tests for src/sampling.py."""

from __future__ import annotations

import numpy as np
import pytest

from sampling import (
    apply_repetition_penalty,
    greedy_decode,
    sample_token,
    temperature_scale,
    top_k_filter,
    top_p_filter,
    _softmax,
)


# ---------------------------------------------------------------------------
# greedy_decode
# ---------------------------------------------------------------------------

def test_greedy_selects_argmax() -> None:
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    assert greedy_decode(logits) == 1


def test_greedy_ties_returns_first() -> None:
    """np.argmax returns the FIRST maximum index on a tie."""
    logits = np.array([5.0, 5.0, 2.0], dtype=np.float32)
    assert greedy_decode(logits) == 0


# ---------------------------------------------------------------------------
# temperature_scale
# ---------------------------------------------------------------------------

def test_temperature_scales_correctly() -> None:
    logits = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    scaled = temperature_scale(logits, 2.0)
    np.testing.assert_allclose(scaled, logits / 2.0)


def test_temperature_one_is_identity() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(temperature_scale(logits, 1.0), logits)


def test_temperature_zero_raises() -> None:
    with pytest.raises(ValueError, match="temperature must be > 0"):
        temperature_scale(np.array([1.0, 2.0]), 0.0)


def test_temperature_negative_raises() -> None:
    with pytest.raises(ValueError):
        temperature_scale(np.array([1.0, 2.0]), -0.5)


# ---------------------------------------------------------------------------
# top_k_filter
# ---------------------------------------------------------------------------

def test_top_k_keeps_only_k() -> None:
    logits = np.array([1.0, 3.0, 5.0, 2.0, 4.0], dtype=np.float32)
    filtered = top_k_filter(logits, k=2)
    finite_count = int(np.sum(np.isfinite(filtered)))
    assert finite_count == 2


def test_top_k_keeps_highest_values() -> None:
    logits = np.array([1.0, 3.0, 5.0, 2.0, 4.0], dtype=np.float32)
    filtered = top_k_filter(logits, k=2)
    # Indices 2 (5.0) and 4 (4.0) should survive
    assert np.isfinite(filtered[2])
    assert np.isfinite(filtered[4])
    assert filtered[0] == -np.inf


def test_top_k_zero_no_filter() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    filtered = top_k_filter(logits, k=0)
    np.testing.assert_array_equal(filtered, logits)


def test_top_k_larger_than_vocab_keeps_all() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    filtered = top_k_filter(logits, k=10)
    assert np.all(np.isfinite(filtered))


# ---------------------------------------------------------------------------
# top_p_filter
# ---------------------------------------------------------------------------

def test_top_p_nucleus_coverage() -> None:
    """After applying top_p=0.95, the prob mass of kept tokens must be >= 0.95."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=100).astype(np.float32)
    filtered = top_p_filter(logits, p=0.95)
    probs = _softmax(filtered)
    kept_mass = float(np.sum(probs[np.isfinite(filtered)]))
    assert kept_mass >= 0.95


def test_top_p_one_no_filter() -> None:
    logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    filtered = top_p_filter(logits, p=1.0)
    np.testing.assert_array_equal(filtered, logits)


def test_top_p_small_p_keeps_few_tokens() -> None:
    """p=0.01 should keep only the highest-probability token(s)."""
    logits = np.array([10.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    filtered = top_p_filter(logits, p=0.01)
    finite_count = int(np.sum(np.isfinite(filtered)))
    assert finite_count == 1


# ---------------------------------------------------------------------------
# apply_repetition_penalty
# ---------------------------------------------------------------------------

def test_repetition_penalty_reduces_positive() -> None:
    """Positive logit for a repeated token should decrease after penalty."""
    logits = np.array([0.0, 5.0, 2.0], dtype=np.float32)
    penalized = apply_repetition_penalty(logits, generated_ids=[1], penalty=2.0)
    assert penalized[1] < logits[1]
    assert penalized[1] == pytest.approx(5.0 / 2.0)


def test_repetition_penalty_increases_negative() -> None:
    """Negative logit for a repeated token should become more negative."""
    logits = np.array([0.0, -3.0, 2.0], dtype=np.float32)
    penalized = apply_repetition_penalty(logits, generated_ids=[1], penalty=2.0)
    assert penalized[1] < logits[1]
    assert penalized[1] == pytest.approx(-3.0 * 2.0)


def test_repetition_penalty_one_no_change() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    penalized = apply_repetition_penalty(logits, generated_ids=[0, 1], penalty=1.0)
    np.testing.assert_array_equal(penalized, logits)


def test_repetition_penalty_empty_generated_no_change() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    penalized = apply_repetition_penalty(logits, generated_ids=[], penalty=2.0)
    np.testing.assert_array_equal(penalized, logits)


def test_repetition_penalty_below_one_raises() -> None:
    logits = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        apply_repetition_penalty(logits, generated_ids=[0], penalty=0.5)


# ---------------------------------------------------------------------------
# sample_token
# ---------------------------------------------------------------------------

def test_sample_token_temperature_zero_is_greedy() -> None:
    """Near-zero temperature must produce the argmax deterministically."""
    logits = np.array([1.0, 2.0, 10.0, 3.0], dtype=np.float32)
    results = {sample_token(logits, temperature=1e-10) for _ in range(20)}
    assert results == {2}


def test_sample_token_returns_int() -> None:
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = sample_token(logits)
    assert isinstance(result, int)


def test_sample_token_result_in_vocab() -> None:
    vocab_size = 50
    logits = np.ones(vocab_size, dtype=np.float32)
    for _ in range(100):
        token_id = sample_token(logits)
        assert 0 <= token_id < vocab_size


def test_sample_token_top_k_restricts_range() -> None:
    """With top_k=1, only the argmax token can be sampled."""
    logits = np.array([1.0, 2.0, 10.0, 3.0], dtype=np.float32)
    results = {sample_token(logits, temperature=1.0, top_k=1) for _ in range(30)}
    assert results == {2}
```

---

## Section 4: tests/test_kv_cache.py

```python
"""Tests for src/kv_cache.py."""

from __future__ import annotations

import numpy as np
import pytest

from kv_cache import KVCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cache(n_layers: int = 4, n_heads: int = 8, d_head: int = 64) -> KVCache:
    return KVCache(n_layers=n_layers, n_heads=n_heads, d_head=d_head)


def make_kv(
    n_heads: int = 8, d_head: int = 64, dtype: np.dtype = np.dtype(np.float32)
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    k = rng.random((n_heads, 1, d_head)).astype(dtype)
    v = rng.random((n_heads, 1, d_head)).astype(dtype)
    return k, v


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_initial_seq_len_zero() -> None:
    cache = make_cache()
    assert cache.seq_len() == 0


def test_initial_memory_bytes_zero() -> None:
    cache = make_cache()
    assert cache.memory_bytes() == 0


# ---------------------------------------------------------------------------
# append / seq_len
# ---------------------------------------------------------------------------

def test_append_grows_seq_len() -> None:
    cache = make_cache()
    k, v = make_kv()
    cache.append(0, k, v)
    cache.append(0, k, v)
    cache.append(0, k, v)
    assert cache.seq_len() == 3


def test_append_all_layers_same_seq_len() -> None:
    """All layers must have the same seq_len after appending to each."""
    n_layers = 4
    cache = make_cache(n_layers=n_layers)
    k, v = make_kv()
    for layer_idx in range(n_layers):
        cache.append(layer_idx, k, v)
    assert cache.seq_len() == 1


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def test_get_returns_concatenated() -> None:
    """After 2 appends to layer 0, K shape must be [n_heads, 2, d_head]."""
    n_heads, d_head = 8, 64
    cache = make_cache(n_heads=n_heads, d_head=d_head)
    k, v = make_kv(n_heads=n_heads, d_head=d_head)
    cache.append(0, k, v)
    cache.append(0, k, v)
    k_out, v_out = cache.get(0)
    assert k_out.shape == (n_heads, 2, d_head)
    assert v_out.shape == (n_heads, 2, d_head)


def test_get_returns_correct_values() -> None:
    """Values retrieved must match what was appended."""
    n_heads, d_head = 4, 32
    cache = make_cache(n_layers=2, n_heads=n_heads, d_head=d_head)
    rng = np.random.default_rng(1)
    k1 = rng.random((n_heads, 1, d_head)).astype(np.float32)
    v1 = rng.random((n_heads, 1, d_head)).astype(np.float32)
    k2 = rng.random((n_heads, 1, d_head)).astype(np.float32)
    v2 = rng.random((n_heads, 1, d_head)).astype(np.float32)
    cache.append(0, k1, v1)
    cache.append(0, k2, v2)
    k_out, v_out = cache.get(0)
    np.testing.assert_array_equal(k_out[:, 0:1, :], k1)
    np.testing.assert_array_equal(k_out[:, 1:2, :], k2)


# ---------------------------------------------------------------------------
# memory_bytes
# ---------------------------------------------------------------------------

def test_memory_bytes_formula() -> None:
    """memory_bytes() must match the analytical formula."""
    n_layers, n_heads, d_head = 3, 4, 16
    cache = KVCache(n_layers=n_layers, n_heads=n_heads, d_head=d_head)
    k, v = make_kv(n_heads=n_heads, d_head=d_head)
    n_appends = 5
    for _ in range(n_appends):
        for layer_idx in range(n_layers):
            cache.append(layer_idx, k, v)

    expected = (
        2
        * n_layers
        * n_heads
        * n_appends
        * d_head
        * np.dtype(np.float32).itemsize
    )
    assert cache.memory_bytes() == expected


def test_memory_bytes_empty() -> None:
    cache = make_cache()
    assert cache.memory_bytes() == 0


def test_memory_bytes_fp16_half_of_fp32() -> None:
    """FP16 cache should use exactly half the memory of an equivalent FP32 cache."""
    n_layers, n_heads, d_head = 2, 4, 16
    cache_fp32 = KVCache(n_layers=n_layers, n_heads=n_heads, d_head=d_head, dtype=np.dtype(np.float32))
    cache_fp16 = KVCache(n_layers=n_layers, n_heads=n_heads, d_head=d_head, dtype=np.dtype(np.float16))
    k32 = np.ones((n_heads, 1, d_head), dtype=np.float32)
    v32 = np.ones((n_heads, 1, d_head), dtype=np.float32)
    k16 = k32.astype(np.float16)
    v16 = v32.astype(np.float16)
    for layer_idx in range(n_layers):
        cache_fp32.append(layer_idx, k32, v32)
        cache_fp16.append(layer_idx, k16, v16)
    assert cache_fp16.memory_bytes() == cache_fp32.memory_bytes() // 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_append_wrong_k_shape_raises() -> None:
    cache = make_cache(n_heads=8, d_head=64)
    wrong_k = np.ones((8, 2, 64), dtype=np.float32)  # seq=2, not 1
    v = np.ones((8, 1, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="k shape"):
        cache.append(0, wrong_k, v)


def test_append_wrong_v_shape_raises() -> None:
    cache = make_cache(n_heads=8, d_head=64)
    k = np.ones((8, 1, 64), dtype=np.float32)
    wrong_v = np.ones((8, 1, 32), dtype=np.float32)  # d_head=32, not 64
    with pytest.raises(ValueError, match="v shape"):
        cache.append(0, k, wrong_v)


def test_layer_index_out_of_range_append_raises() -> None:
    cache = make_cache(n_layers=4)
    k, v = make_kv()
    with pytest.raises(IndexError):
        cache.append(4, k, v)


def test_layer_index_negative_raises() -> None:
    cache = make_cache(n_layers=4)
    k, v = make_kv()
    with pytest.raises(IndexError):
        cache.append(-1, k, v)


def test_layer_index_out_of_range_get_raises() -> None:
    cache = make_cache(n_layers=4)
    with pytest.raises(IndexError):
        cache.get(10)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

def test_clear_resets_seq_len() -> None:
    cache = make_cache()
    k, v = make_kv()
    for _ in range(5):
        cache.append(0, k, v)
    assert cache.seq_len() == 5
    cache.clear()
    assert cache.seq_len() == 0


def test_clear_resets_memory() -> None:
    cache = make_cache()
    k, v = make_kv()
    cache.append(0, k, v)
    assert cache.memory_bytes() > 0
    cache.clear()
    assert cache.memory_bytes() == 0


def test_clear_allows_reuse() -> None:
    """After clear(), appending should work as if freshly initialized."""
    cache = make_cache()
    k, v = make_kv()
    cache.append(0, k, v)
    cache.clear()
    cache.append(0, k, v)
    assert cache.seq_len() == 1
```
