# Week 07 Project — Sampling & Decoding Toolkit

## Overview

Build a production-quality sampling and decoding library for autoregressive language model inference. The library runs entirely on CPU with NumPy — no GPU or model download required.

By the end of this project you will have implemented every major token sampling strategy used in modern LLM inference, wired them into a minimal generation loop, and backed the whole thing with tests and documentation.

---

## Objectives

- Implement all major sampling strategies as composable, well-tested functions.
- Build a reusable `KVCache` class that tracks per-layer key/value tensors and exposes memory accounting.
- Wire together a mini inference loop using a mock autoregressive model.
- Write comprehensive tests achieving >80% coverage on `sampling.py` and `kv_cache.py`.
- Document every public function and class with docstrings.

---

## Deliverables

| File | Description |
|---|---|
| `src/sampling.py` | All sampling and filtering functions |
| `src/kv_cache.py` | `KVCache` class |
| `src/mock_model.py` | Mock autoregressive model |
| `src/demo.py` | Demo comparing all sampling strategies |
| `tests/test_sampling.py` | Unit tests for sampling functions |
| `tests/test_kv_cache.py` | Unit tests for `KVCache` |
| `pyproject.toml` | Project config and dependencies |

---

## Setup

```bash
uv init sampling-toolkit
cd sampling-toolkit
uv add numpy pytest pytest-cov
```

Recommended directory layout after setup:

```
sampling-toolkit/
  src/
    sampling.py
    kv_cache.py
    mock_model.py
    demo.py
  tests/
    test_sampling.py
    test_kv_cache.py
  pyproject.toml
```

---

## Functional Requirements

Implement each function and class to the specification below. Signatures are required; docstrings are required; implementation is yours to design.

### 1. `greedy_decode(logits: np.ndarray) -> int`

Returns the index of the maximum logit. No randomness.

```python
def greedy_decode(logits: np.ndarray) -> int:
    """Return the token index with the highest logit (argmax)."""
    ...
```

### 2. `temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray`

Divides logits by `temperature` and returns the result before softmax. Caller is responsible for applying softmax afterward.

Precondition: `temperature > 0`. Raise `ValueError` if `temperature <= 0`.

```python
def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by temperature. Returns pre-softmax scaled logits."""
    ...
```

### 3. `top_k_filter(logits: np.ndarray, k: int) -> np.ndarray`

Returns a copy of `logits` where every element outside the top-k values is replaced with `-np.inf`. The top-k values are unchanged.

Precondition: `1 <= k <= len(logits)`. Raise `ValueError` otherwise.

```python
def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Mask all but the top-k logits to -inf."""
    ...
```

### 4. `top_p_filter(logits: np.ndarray, p: float) -> np.ndarray`

Implements nucleus sampling filtering. Sort tokens by descending probability, accumulate probabilities, and mask out all tokens whose cumulative probability exceeds `p`. Always keep at least one token.

Precondition: `0 < p <= 1.0`. Raise `ValueError` otherwise.

```python
def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Mask tokens outside the top-p nucleus to -inf."""
    ...
```

### 5. `apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int], penalty: float) -> np.ndarray`

For each token ID present in `generated_ids`, apply the repetition penalty:
- If the logit is positive, divide it by `penalty`.
- If the logit is negative, multiply it by `penalty`.

`penalty >= 1.0`. A penalty of `1.0` is a no-op. Raise `ValueError` if `penalty < 1.0`.

```python
def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float,
) -> np.ndarray:
    """Penalize tokens that have already been generated."""
    ...
```

### 6. `sample_token(...) -> int`

Unified sampling function that applies the full pipeline in order:

1. Apply repetition penalty (if `penalty > 1.0`).
2. If `temperature <= 1e-8`, return `greedy_decode(logits)` immediately.
3. Apply temperature scaling.
4. Apply top-k filtering (if `k` is not `None`).
5. Apply top-p filtering (if `p < 1.0`).
6. Convert to probabilities with softmax.
7. Sample one token from the resulting distribution.

```python
def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: list[int] | None = None,
) -> int:
    """
    Sample the next token applying temperature, top-k, top-p,
    and repetition penalty.
    """
    ...
```

### 7. `KVCache` class

Stores per-layer key and value tensors and provides append, retrieval, and memory accounting.

```python
class KVCache:
    """
    Stores key and value tensors for each transformer layer.

    Attributes
    ----------
    num_layers : int
    cache : list[dict[str, np.ndarray]]
        One dict per layer, each with keys "k" and "v".
    """

    def __init__(self, num_layers: int) -> None:
        """Initialize an empty cache for `num_layers` layers."""
        ...

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """
        Append key and value tensors for a single generation step to the
        specified layer's cache.

        Parameters
        ----------
        layer_idx : int
            Index of the transformer layer (0-indexed).
        k : np.ndarray
            Key tensor for this step, shape (num_heads, d_head).
        v : np.ndarray
            Value tensor for this step, shape (num_heads, d_head).
        """
        ...

    def get(self, layer_idx: int) -> dict[str, np.ndarray]:
        """
        Return the full cached K and V arrays for a layer.

        Returns a dict with keys "k" and "v", each of shape
        (seq_len, num_heads, d_head).
        """
        ...

    def seq_len(self) -> int:
        """Return the number of steps currently cached (same for all layers)."""
        ...

    def memory_bytes(self) -> int:
        """Return the total number of bytes occupied by all cached tensors."""
        ...
```

### 8. Mock model: `get_logits(input_ids: list[int], kv_cache: KVCache | None) -> np.ndarray`

A deterministic mock that returns plausible-looking logits without a real transformer. The implementation must:

- Accept `input_ids` (full sequence so far) and an optional `KVCache`.
- Return a 1-D NumPy array of shape `(vocab_size,)` representing next-token logits.
- Use a fixed random seed derived from the last token ID so the output is reproducible.
- Update `kv_cache` if one is provided (append dummy K/V tensors for a single layer to simulate cache growth).
- Not call any external API or load any file.

```python
VOCAB_SIZE: int = 1000
NUM_LAYERS: int = 4
NUM_HEADS: int = 4
D_HEAD: int = 32

def get_logits(
    input_ids: list[int],
    kv_cache: KVCache | None = None,
) -> np.ndarray:
    """
    Return mock next-token logits for the given input sequence.

    Uses a seeded RNG for reproducibility. If kv_cache is provided,
    appends dummy K/V tensors to simulate cache growth.
    """
    ...
```

---

## Demo (`src/demo.py`)

Write a script that:

1. Initialises a token sequence (e.g., `[1, 2, 3]`).
2. Runs generation with each of the following configurations for 20 tokens each:
   - Greedy (temperature=0)
   - Temperature=0.8, no top-k, no top-p
   - Temperature=1.0, top-p=0.9
   - Temperature=1.0, top-k=50
   - Temperature=0.9, top-p=0.92, repetition_penalty=1.3
3. Prints the generated token ID list for each configuration.
4. Logs (not prints) a line for each configuration reporting the number of unique tokens generated.

The demo must run without errors:

```bash
uv run python src/demo.py
```

---

## Testing Requirements

### `tests/test_sampling.py`

Cover at minimum:

- `greedy_decode` returns the index of the maximum value.
- `temperature_scale` with T=1.0 is a no-op.
- `temperature_scale` raises `ValueError` for T <= 0.
- `top_k_filter` keeps exactly k non-`-inf` values.
- `top_k_filter` raises `ValueError` for k < 1.
- `top_p_filter` with p=1.0 is a no-op (no tokens masked).
- `top_p_filter` always keeps at least one token.
- `apply_repetition_penalty` with penalty=1.0 is a no-op.
- `apply_repetition_penalty` reduces logit magnitude for repeated tokens.
- `sample_token` with temperature=0.0 is deterministic.
- `sample_token` returns an integer in `[0, vocab_size)`.

### `tests/test_kv_cache.py`

Cover at minimum:

- `KVCache` initializes with empty cache for each layer.
- `append` increases `seq_len()` by 1.
- `get` returns arrays of the correct shape after multiple appends.
- `memory_bytes` returns 0 for an empty cache.
- `memory_bytes` increases after each `append`.
- Appending to an out-of-range layer index raises `IndexError`.

### Running tests

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

Target: >80% coverage on `src/sampling.py` and `src/kv_cache.py`.

---

## Non-functional Requirements

- All public functions and classes have type hints on all parameters and return values.
- All public functions and classes have docstrings describing behavior, parameters, and return values.
- Diagnostic output uses the `logging` module. No `print()` calls in `sampling.py`, `kv_cache.py`, or `mock_model.py`. `demo.py` may use `print()` for human-readable output only.
- No hardcoded secrets, API keys, or credentials anywhere in the project.
- `pyproject.toml` lists all dependencies. The project must be reproducible from a clean `uv sync`.

---

## Stretch Goals

Implement any of the following for additional depth. None are required for a passing score.

- [ ] **Beam search**: implement `beam_search(get_logits, prompt_ids, beam_width=2, max_new_tokens=20) -> list[int]`. Track the top-2 partial sequences by cumulative log-probability and return the highest-scoring complete sequence.
- [ ] **Min-p sampling**: implement `min_p_filter(logits, p_min) -> np.ndarray`. Filter out any token whose probability is below `p_min * max_prob` before sampling.
- [ ] **Typical-p sampling**: implement `typical_p_filter(logits, p) -> np.ndarray`. Rank tokens by their distance from the entropy of the distribution (|log p(x) - H|) and keep the nucleus whose typical mass sums to p.
- [ ] **Benchmark**: add `src/benchmark.py` that measures tokens/sec for the mock model at sequence lengths [64, 128, 256, 512, 1024] and prints a table. Use `time.perf_counter` for timing.

---

## Evaluation Checklist

Before submitting, verify:

- [ ] `uv run pytest tests/ -v` — all tests pass.
- [ ] `uv run pytest tests/ --cov=src --cov-report=term-missing` — >80% coverage on `sampling.py` and `kv_cache.py`.
- [ ] `uv run python src/demo.py` — runs without exceptions.
- [ ] Every public function in `sampling.py` and `kv_cache.py` has a type hint and a docstring.
- [ ] No `print()` in `sampling.py`, `kv_cache.py`, or `mock_model.py`.
- [ ] `pyproject.toml` exists and `uv sync` succeeds from a clean environment.

---

## Conceptual Connections

This project operationalises the theory from Week 07 lectures. As you build each component, trace it back to the concepts:

- `temperature_scale` and `top_p_filter` implement what you derived mathematically in the assessment.
- `KVCache.memory_bytes()` should match your hand-calculated estimate from assessment question 5 when you configure the mock model with the same hyperparameters.
- The demo output gives you intuition for how sampling strategy affects output diversity — greedy output will repeat tokens; top-p with a moderate temperature will not.

---

## Reference Signatures (summary)

```python
# sampling.py
def greedy_decode(logits: np.ndarray) -> int: ...
def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray: ...
def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray: ...
def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray: ...
def apply_repetition_penalty(logits: np.ndarray, generated_ids: list[int], penalty: float) -> np.ndarray: ...
def sample_token(logits: np.ndarray, temperature: float = 1.0, top_k: int | None = None, top_p: float = 1.0, repetition_penalty: float = 1.0, generated_ids: list[int] | None = None) -> int: ...

# kv_cache.py
class KVCache:
    def __init__(self, num_layers: int) -> None: ...
    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None: ...
    def get(self, layer_idx: int) -> dict[str, np.ndarray]: ...
    def seq_len(self) -> int: ...
    def memory_bytes(self) -> int: ...

# mock_model.py
def get_logits(input_ids: list[int], kv_cache: KVCache | None = None) -> np.ndarray: ...
```
