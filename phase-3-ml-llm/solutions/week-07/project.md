# Solutions — Week 07 · Project — Sampling & Decoding Toolkit

## Complete Reference Implementation

---

### `src/sampling.py`

```python
"""sampling.py — Token sampling and filtering functions for LLM inference.

All functions are pure NumPy. No external API calls, no GPU required.

Deps (uv add numpy): numpy
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# 1. Greedy decode
# ---------------------------------------------------------------------------

def greedy_decode(logits: np.ndarray) -> int:
    """Return the token index with the highest logit (argmax).

    Args:
        logits: 1-D array of shape (vocab_size,).

    Returns:
        Integer index of the maximum logit.
    """
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# 2. Temperature scaling
# ---------------------------------------------------------------------------

def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by temperature. Returns pre-softmax scaled logits.

    Args:
        logits: 1-D array of shape (vocab_size,).
        temperature: positive float; smaller = more peaked distribution.

    Returns:
        Scaled logits (same shape as input).

    Raises:
        ValueError: if temperature <= 0.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


# ---------------------------------------------------------------------------
# 3. Top-k filter
# ---------------------------------------------------------------------------

def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Mask all but the top-k logits to -inf.

    Args:
        logits: 1-D array of shape (vocab_size,).
        k: number of logits to keep. Must satisfy 1 <= k <= len(logits).

    Returns:
        Copy of logits with all but the top-k values set to -np.inf.

    Raises:
        ValueError: if k is out of range.
    """
    if not (1 <= k <= len(logits)):
        raise ValueError(f"k must be in [1, {len(logits)}], got {k}")
    result = np.full_like(logits, -np.inf)
    top_indices = np.argpartition(logits, -k)[-k:]
    result[top_indices] = logits[top_indices]
    return result


# ---------------------------------------------------------------------------
# 4. Top-p (nucleus) filter
# ---------------------------------------------------------------------------

def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Mask tokens outside the top-p nucleus to -inf.

    Sorts tokens by descending probability, accumulates probability mass,
    and keeps all tokens whose cumulative sum is <= p. Always keeps at
    least one token.

    Args:
        logits: 1-D array of shape (vocab_size,).
        p: nucleus probability threshold in (0, 1].

    Returns:
        Copy of logits with out-of-nucleus tokens set to -np.inf.

    Raises:
        ValueError: if p is out of range.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits.copy()

    probs = _softmax(logits)
    sorted_idx = np.argsort(-probs)          # descending probability
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)

    # Keep tokens until cumulative prob exceeds p; always keep at least 1
    cutoff = np.searchsorted(cumulative, p, side="right")
    cutoff = max(cutoff, 1)

    result = np.full_like(logits, -np.inf)
    keep = sorted_idx[:cutoff]
    result[keep] = logits[keep]
    return result


# ---------------------------------------------------------------------------
# 5. Repetition penalty
# ---------------------------------------------------------------------------

def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float,
) -> np.ndarray:
    """Penalize tokens that have already been generated.

    For each token ID in generated_ids:
    - If logit > 0: divide by penalty (reduce confidence).
    - If logit < 0: multiply by penalty (increase magnitude of negative).

    Args:
        logits: 1-D array of shape (vocab_size,).
        generated_ids: list of previously generated token IDs.
        penalty: penalty factor >= 1.0. 1.0 is a no-op.

    Returns:
        Modified copy of logits.

    Raises:
        ValueError: if penalty < 1.0.
    """
    if penalty < 1.0:
        raise ValueError(f"penalty must be >= 1.0, got {penalty}")
    result = logits.copy()
    for token_id in set(generated_ids):
        if 0 <= token_id < len(result):
            if result[token_id] > 0:
                result[token_id] /= penalty
            else:
                result[token_id] *= penalty
    return result


# ---------------------------------------------------------------------------
# 6. Unified sample_token
# ---------------------------------------------------------------------------

def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
) -> int:
    """Sample the next token applying temperature, top-k, top-p, and
    repetition penalty in order.

    Pipeline:
        1. Apply repetition penalty (if penalty > 1.0).
        2. If temperature <= 1e-8, return greedy_decode immediately.
        3. Apply temperature scaling.
        4. Apply top-k filtering (if top_k is not None).
        5. Apply top-p filtering (if top_p < 1.0).
        6. Convert to probabilities via softmax.
        7. Sample one token from the resulting distribution.

    Args:
        logits: 1-D array of shape (vocab_size,).
        temperature: softmax temperature. 0 → greedy (deterministic).
        top_k: if set, keep only top-k logits.
        top_p: nucleus probability threshold.
        repetition_penalty: penalty for previously generated tokens.
        generated_ids: list of token IDs generated so far.

    Returns:
        Sampled token ID as int.
    """
    if generated_ids is None:
        generated_ids = []

    logits = logits.copy()

    # Step 1: repetition penalty
    if repetition_penalty > 1.0:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Step 2: greedy shortcut
    if temperature <= 1e-8:
        return greedy_decode(logits)

    # Step 3: temperature
    logits = temperature_scale(logits, temperature)

    # Step 4: top-k
    if top_k is not None:
        logits = top_k_filter(logits, top_k)

    # Step 5: top-p
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    # Step 6: softmax
    probs = _softmax(logits)

    # Step 7: multinomial sample
    token_id = int(np.random.choice(len(probs), p=probs))
    log.debug("sample_token → %d (p=%.4f)", token_id, probs[token_id])
    return token_id
```

---

### `src/kv_cache.py`

```python
"""kv_cache.py — KV cache for transformer inference.

Deps: numpy
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class KVCache:
    """Stores key and value tensors for each transformer layer.

    Each layer's cache grows by one step per generation call. All layers
    are kept in sync (same seq_len).

    Attributes:
        num_layers: number of transformer layers.
        cache: list of per-layer dicts, each with keys "k" and "v".
            Shape of each entry after seq_len steps:
                "k": (seq_len, num_heads, d_head)
                "v": (seq_len, num_heads, d_head)
    """

    def __init__(self, num_layers: int) -> None:
        """Initialize an empty cache for num_layers layers."""
        self.num_layers = num_layers
        self.cache: list[dict[str, np.ndarray]] = [
            {"k": np.empty((0, 0, 0)), "v": np.empty((0, 0, 0))}
            for _ in range(num_layers)
        ]
        self._seq_len: int = 0

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append key and value tensors for one generation step.

        Args:
            layer_idx: index of the transformer layer (0-indexed).
            k: key tensor for this step, shape (num_heads, d_head).
            v: value tensor for this step, shape (num_heads, d_head).

        Raises:
            IndexError: if layer_idx is out of range.
        """
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        entry = self.cache[layer_idx]
        if entry["k"].shape[0] == 0:
            # First append: initialise storage
            entry["k"] = k[np.newaxis]  # (1, H, d)
            entry["v"] = v[np.newaxis]
        else:
            entry["k"] = np.concatenate([entry["k"], k[np.newaxis]], axis=0)
            entry["v"] = np.concatenate([entry["v"], v[np.newaxis]], axis=0)

        # Update seq_len from layer 0
        if layer_idx == 0:
            self._seq_len = entry["k"].shape[0]
        log.debug("KVCache.append layer=%d seq_len=%d", layer_idx, self._seq_len)

    def get(self, layer_idx: int) -> dict[str, np.ndarray]:
        """Return the full cached K and V arrays for a layer.

        Returns:
            dict with keys "k" and "v", each of shape (seq_len, num_heads, d_head).
        """
        return self.cache[layer_idx]

    def seq_len(self) -> int:
        """Return the number of steps currently cached (same for all layers)."""
        return self._seq_len

    def memory_bytes(self) -> int:
        """Return total bytes occupied by all cached tensors."""
        total = 0
        for layer in self.cache:
            for arr in layer.values():
                if arr.size > 0:
                    total += arr.nbytes
        return total
```

---

### `src/mock_model.py`

```python
"""mock_model.py — Deterministic mock autoregressive model.

Produces plausible-looking logits from a seeded RNG. No real weights,
no API calls, no file I/O.

Deps: numpy
"""
from __future__ import annotations

import logging

import numpy as np

from kv_cache import KVCache

log = logging.getLogger(__name__)

VOCAB_SIZE: int = 1000
NUM_LAYERS: int = 4
NUM_HEADS: int = 4
D_HEAD: int = 32


def get_logits(
    input_ids: list[int],
    kv_cache: KVCache | None = None,
) -> np.ndarray:
    """Return mock next-token logits for the given input sequence.

    Uses a seeded RNG derived from the last token ID for reproducibility.
    If kv_cache is provided, appends dummy K/V tensors to simulate growth.

    Args:
        input_ids: token IDs generated so far (full sequence).
        kv_cache: optional KVCache to update.

    Returns:
        1-D NumPy array of shape (VOCAB_SIZE,) representing next-token logits.
    """
    last_token = input_ids[-1] if input_ids else 0
    rng = np.random.default_rng(seed=last_token % (2**31))
    logits = rng.standard_normal(VOCAB_SIZE).astype(np.float32)
    # Give token (last_token + 1) a slight boost for mild repetition structure
    next_hint = (last_token + 1) % VOCAB_SIZE
    logits[next_hint] += 2.0

    if kv_cache is not None:
        k_dummy = rng.standard_normal((NUM_HEADS, D_HEAD)).astype(np.float32)
        v_dummy = rng.standard_normal((NUM_HEADS, D_HEAD)).astype(np.float32)
        for layer_idx in range(NUM_LAYERS):
            kv_cache.append(layer_idx, k_dummy, v_dummy)
        log.debug("mock_model: updated KVCache seq_len=%d", kv_cache.seq_len())

    return logits
```

---

### `src/demo.py`

```python
"""demo.py — Compare sampling strategies on the mock model.

Usage:
    uv run python src/demo.py
"""
from __future__ import annotations

import logging
import sys

import numpy as np

sys.path.insert(0, "src")
from sampling import sample_token
from kv_cache import KVCache
from mock_model import get_logits, VOCAB_SIZE

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_generation(
    prompt: list[int],
    n_tokens: int = 20,
    seed: int = 42,
    **kwargs,
) -> list[int]:
    """Generate n_tokens starting from prompt, using sample_token(**kwargs)."""
    np.random.seed(seed)
    ids = list(prompt)
    cache = KVCache(num_layers=4)
    for _ in range(n_tokens):
        logits = get_logits(ids, kv_cache=cache)
        token = sample_token(logits, generated_ids=ids, **kwargs)
        ids.append(token)
    return ids[len(prompt):]


def main() -> None:
    prompt = [1, 2, 3]

    strategies: list[tuple[str, dict]] = [
        ("Greedy (temperature=0)",                  {"temperature": 0.0}),
        ("Temperature=0.8, no top-k/p",             {"temperature": 0.8}),
        ("Temperature=1.0, top-p=0.9",              {"temperature": 1.0, "top_p": 0.9}),
        ("Temperature=1.0, top-k=50",               {"temperature": 1.0, "top_k": 50}),
        ("T=0.9, top-p=0.92, rep_penalty=1.3",      {"temperature": 0.9, "top_p": 0.92,
                                                      "repetition_penalty": 1.3}),
    ]

    for name, kwargs in strategies:
        tokens = run_generation(prompt, n_tokens=20, **kwargs)
        n_unique = len(set(tokens))
        log.warning("Strategy '%s' — unique tokens: %d", name, n_unique)
        print(f"\n  {name}:")
        print(f"    tokens    = {tokens}")
        print(f"    unique    = {n_unique}")


if __name__ == "__main__":
    main()
```

---

### `tests/test_sampling.py`

```python
"""Unit tests for sampling.py."""
from __future__ import annotations

import numpy as np
import pytest
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from sampling import (
    greedy_decode,
    temperature_scale,
    top_k_filter,
    top_p_filter,
    apply_repetition_penalty,
    sample_token,
)

VOCAB = 20


@pytest.fixture
def logits() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(VOCAB).astype(np.float64)


def test_greedy_returns_argmax(logits: np.ndarray) -> None:
    assert greedy_decode(logits) == int(np.argmax(logits))


def test_temperature_scale_noop(logits: np.ndarray) -> None:
    scaled = temperature_scale(logits, 1.0)
    np.testing.assert_array_almost_equal(scaled, logits)


def test_temperature_scale_raises_on_zero(logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        temperature_scale(logits, 0.0)


def test_temperature_scale_raises_on_negative(logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        temperature_scale(logits, -0.5)


def test_top_k_keeps_exactly_k(logits: np.ndarray) -> None:
    k = 5
    filtered = top_k_filter(logits, k)
    finite_count = np.sum(np.isfinite(filtered))
    assert finite_count == k


def test_top_k_raises_out_of_range(logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        top_k_filter(logits, 0)
    with pytest.raises(ValueError):
        top_k_filter(logits, VOCAB + 1)


def test_top_p_noop_at_one(logits: np.ndarray) -> None:
    filtered = top_p_filter(logits, 1.0)
    np.testing.assert_array_almost_equal(filtered, logits)


def test_top_p_keeps_at_least_one(logits: np.ndarray) -> None:
    # Even with very small p, at least one token survives
    filtered = top_p_filter(logits, 0.0001)
    assert np.sum(np.isfinite(filtered)) >= 1


def test_top_p_raises_out_of_range(logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        top_p_filter(logits, 0.0)
    with pytest.raises(ValueError):
        top_p_filter(logits, 1.5)


def test_repetition_penalty_noop(logits: np.ndarray) -> None:
    result = apply_repetition_penalty(logits, generated_ids=[0, 1, 2], penalty=1.0)
    np.testing.assert_array_almost_equal(result, logits)


def test_repetition_penalty_reduces_positive_logit(logits: np.ndarray) -> None:
    # Force token 0 to have a large positive logit
    logits[0] = 5.0
    result = apply_repetition_penalty(logits, generated_ids=[0], penalty=2.0)
    assert result[0] < logits[0]


def test_repetition_penalty_raises_below_one(logits: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_repetition_penalty(logits, generated_ids=[], penalty=0.5)


def test_sample_token_greedy_deterministic(logits: np.ndarray) -> None:
    t1 = sample_token(logits, temperature=0.0)
    t2 = sample_token(logits, temperature=0.0)
    assert t1 == t2 == greedy_decode(logits)


def test_sample_token_returns_valid_id(logits: np.ndarray) -> None:
    token = sample_token(logits, temperature=1.0)
    assert 0 <= token < VOCAB
```

---

### `tests/test_kv_cache.py`

```python
"""Unit tests for kv_cache.KVCache."""
from __future__ import annotations

import numpy as np
import pytest
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from kv_cache import KVCache

NUM_LAYERS = 4
NUM_HEADS = 4
D_HEAD = 32


def _dummy_kv() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    k = rng.standard_normal((NUM_HEADS, D_HEAD)).astype(np.float32)
    v = rng.standard_normal((NUM_HEADS, D_HEAD)).astype(np.float32)
    return k, v


def test_initial_cache_empty() -> None:
    cache = KVCache(NUM_LAYERS)
    assert cache.seq_len() == 0


def test_append_increases_seq_len() -> None:
    cache = KVCache(NUM_LAYERS)
    k, v = _dummy_kv()
    for layer in range(NUM_LAYERS):
        cache.append(layer, k, v)
    assert cache.seq_len() == 1

    for layer in range(NUM_LAYERS):
        cache.append(layer, k, v)
    assert cache.seq_len() == 2


def test_get_returns_correct_shape() -> None:
    cache = KVCache(NUM_LAYERS)
    k, v = _dummy_kv()
    for step in range(3):
        for layer in range(NUM_LAYERS):
            cache.append(layer, k, v)
    entry = cache.get(0)
    assert entry["k"].shape == (3, NUM_HEADS, D_HEAD)
    assert entry["v"].shape == (3, NUM_HEADS, D_HEAD)


def test_memory_bytes_zero_initially() -> None:
    cache = KVCache(NUM_LAYERS)
    assert cache.memory_bytes() == 0


def test_memory_bytes_increases_after_append() -> None:
    cache = KVCache(NUM_LAYERS)
    k, v = _dummy_kv()
    for layer in range(NUM_LAYERS):
        cache.append(layer, k, v)
    assert cache.memory_bytes() > 0

    mem1 = cache.memory_bytes()
    for layer in range(NUM_LAYERS):
        cache.append(layer, k, v)
    assert cache.memory_bytes() > mem1


def test_out_of_range_layer_raises_index_error() -> None:
    cache = KVCache(NUM_LAYERS)
    k, v = _dummy_kv()
    with pytest.raises(IndexError):
        cache.append(NUM_LAYERS, k, v)
    with pytest.raises(IndexError):
        cache.append(-1, k, v)
```

---

### `pyproject.toml`

```toml
[project]
name = "sampling-toolkit"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"
```

---

## Design notes

**`top_p_filter` always keeps at least one token**: if `p` is very small and the top-1 token
has cumulative probability > p, `searchsorted` returns 0. `max(cutoff, 1)` ensures the
highest-probability token is always preserved — otherwise softmax receives all-`-inf` input
and produces `NaN` probabilities.

**Repetition penalty sign convention**: the HuggingFace convention divides positive logits (to
reduce their probability) and multiplies negative logits by the penalty (to make them more
negative). This consistently reduces the probability of repeated tokens regardless of their
initial sign.

**`KVCache` initialised with empty arrays**: using `np.empty((0, 0, 0))` as a sentinel
avoids a `None` check on every `append` and makes shape introspection consistent.

**`greedy_decode` short-circuit in `sample_token`**: dividing logits by temperature=0 causes
`+inf / 0 = NaN`. Short-circuiting before any temperature computation prevents NaN propagation
through the subsequent filter and softmax stages.
