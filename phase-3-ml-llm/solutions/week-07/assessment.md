# Answer Key — Week 07 Assessment — LLM Training & Inference Fundamentals

---

## Part 1: Conceptual Questions

### 1. Pretraining vs Instruction Tuning

| Dimension | Pretraining | Instruction Tuning (SFT) |
|-----------|-------------|--------------------------|
| Objective | Next-token prediction (cross-entropy on all tokens) | Cross-entropy on response tokens only — prompt tokens are masked |
| Dataset | Hundreds of billions of tokens from web crawls, books, code | Thousands to millions of (instruction, response) pairs |
| Compute | Millions of GPU-hours (e.g., GPT-3: ~3,640 petaflop-days) | Thousands of GPU-hours on top of a pretrained checkpoint |
| What the model learns | World knowledge, language structure, emergent reasoning | Behavioral format: follow instructions, respond helpfully |

**Key insight:** Pretraining builds the model's knowledge and language capability. Instruction tuning shapes *how* the model uses that capability when prompted by a user. Neither step teaches the model what humans actually value — that requires RLHF.

---

### 2. Why RLHF Produces Better Aligned Models Than SFT Alone

SFT minimizes cross-entropy loss on training responses. This means the model learns to assign high probability to whatever tokens appear in the training data — it has no way to distinguish a high-quality response from a low-quality one if both appear in the dataset.

RLHF proceeds in three stages:
1. **SFT warm-start** — fine-tune on a small set of high-quality demonstrations.
2. **Reward model training** — human raters compare pairs of responses (A vs B) and indicate which is preferable. A reward model is trained to predict these preferences. The reward model learns a scalar "goodness" signal over (prompt, response) pairs.
3. **RL fine-tuning (PPO)** — the policy (language model) is fine-tuned to maximize the reward model score while staying close to the SFT checkpoint (via a KL penalty). This explicitly steers the model toward responses that score high on human-judged quality dimensions: safety, helpfulness, coherence, factual accuracy.

The critical difference: RLHF's optimization objective directly encodes human preferences. SFT's objective (token prediction loss) does not.

---

### 3. Context Length Overflow

The model's learned position embeddings only cover positions 0 to `max_position_embeddings - 1` (here: 0 to 1023). When a prompt exceeds this length, one of the following happens depending on the framework:

- **Truncation (most common):** The framework silently keeps only the most recent `max_seq_len` tokens. For a 3800-token prompt with max_seq_len=1024, the first **3800 - 1024 = 2776 tokens** are discarded. Any information in the truncated prefix is permanently lost.
- **Error:** Some frameworks raise an explicit exception rather than silently truncating.
- **Garbage output:** If the model runs with out-of-range position indices (no guard rails), the position embeddings for positions >= 1024 are undefined, producing incoherent outputs.

**Mitigation strategies:** Use a model with a larger context window; apply summarization or retrieval to compress the prompt; use RoPE (Rotary Position Encoding) which has better extrapolation properties beyond training length.

---

### 4. How KV Cache Reduces Compute

**Without KV cache:**
At decode step `t`, the model processes all `t` tokens in the context window. The attention mechanism computes Q, K, V projections for all `t` tokens, then computes scaled dot-product attention (Q @ K^T): `O(t)` matrix multiplications per step, `O(t^2)` total across all steps.

**With KV cache:**
K and V projections for tokens 0 to `t-1` are computed once and stored. At step `t`, only Q for the new token needs to be computed, then multiplied against the cached K/V. This reduces per-step cost to `O(1)` matmuls.

**Tradeoff:**
Memory grows as `O(seq_len)`:
```
bytes = 2 * n_layers * n_heads * seq_len * d_head * bytes_per_element
```
The factor of 2 accounts for both K and V. For long sequences and large models, this can exceed GPU VRAM (see Q2 in day-7 reflections for mitigation strategies).

---

### 5. KV Cache Memory Calculation

**Setup:** GPT-2 style: 12 transformer layers, 12 attention heads, d_head = 64, FP16 precision (2 bytes/element), sequence length = 1024.

```
memory = 2 * n_layers * n_heads * seq_len * d_head * bytes_per_element
       = 2 * 12 * 12 * 1024 * 64 * 2
       = 2 * 12 * 12 * 1024 * 128
       = 37,748,736 bytes
       = 36.0 MB
```

**Verification:**
- Per token, per layer: 2 (K+V) * 12 (heads) * 64 (d_head) * 2 (FP16) = 3,072 bytes
- All tokens: 3,072 * 1024 = 3,145,728 bytes = 3 MB per layer
- All layers: 3 MB * 12 = 36 MB

At 36 MB for 1024 tokens, GPT-2's KV cache is negligible compared to the ~548 MB model weights. For much larger models and longer sequences (e.g., 70B model, 32K context), the KV cache reaches tens of GB.

---

### 6. Temperature Approaches 0

As temperature T → 0, logit_i / T → ±∞ for all tokens except the maximum:

```
scaled_logit_i = logit_i / T

For T → 0+:
  - If logit_i = max(logits): scaled_logit_i → +∞
  - If logit_i < max(logits): scaled_logit_i → -∞
```

After softmax:
```
softmax(logits / T)_i = exp(logit_i / T) / sum_j(exp(logit_j / T))
```

Divide numerator and denominator by `exp(max_logit / T)`:
```
= exp((logit_i - max_logit) / T) / sum_j(exp((logit_j - max_logit) / T))
```

As T → 0, every term in the denominator except the maximum goes to 0. The result is:
```
lim_{T→0} softmax(logits / T) = one_hot(argmax(logits))
```

This is why temperature=0 is equivalent to greedy (argmax) sampling.

---

### 7. Top-k vs Top-p

**Top-k:** Always keeps exactly `k` tokens regardless of the distribution's shape. If k=50, the model samples from the top-50 tokens even if the remaining 49 tokens all have near-zero probability (the dominant token has 99.9% probability). This can introduce many unnecessary low-probability candidates.

**Top-p (nucleus sampling):** Keeps the smallest set of tokens whose cumulative probability mass reaches `p`. The number of kept tokens is adaptive:
- For a peaked distribution (one dominant token): nucleus may be size 1-3.
- For a flat distribution (many roughly equal tokens): nucleus may be size 50-200.

**When top-p outperforms top-k:**
Across an autoregressive generation sequence, the distribution's "peakedness" varies dramatically. At high-certainty steps (e.g., completing "The capital of France is ___"), the distribution is highly peaked and top-k unnecessarily adds noise. At high-uncertainty steps (e.g., generating the next word of a creative story), the distribution is flat and top-k may be too restrictive. Top-p adapts to both situations automatically.

**Practical recommendation:** Top-p with p=0.9 or p=0.95 is generally preferred. Many systems combine top-k and top-p (apply top-k first as a hard maximum, then top-p).

---

### 8. Chinchilla Scaling Laws

The Hoffmann et al. (2022) paper found that the compute-optimal allocation of a fixed training budget follows:

```
N_optimal = sqrt(C / 6)      (parameters)
D_optimal = 20 * N_optimal   (training tokens)
```

Where `C` is the total compute budget in FLOPs and the constant 6 comes from the `6ND` FLOPs approximation for training.

**Key finding:** Prior large models (GPT-3 175B, Gopher 280B) were significantly undertrained — they used too many parameters and too few tokens for their compute budget. The Chinchilla model (70B parameters, 1.4T tokens) outperformed Gopher (280B parameters, 300B tokens) despite using 4x fewer parameters, because it was trained on 4.7x more data.

**Rule of thumb:** Each parameter should see approximately 20 training tokens for compute-optimal training.

---

### 9. Quantization Trade-offs

**What quantization does:**
Maps continuous float weights (FP32: 4 bytes, ~7 significant decimal digits of precision) to discrete integer representations (INT8: 1 byte, 256 distinct values; INT4: 0.5 bytes, 16 distinct values).

| Format | Bytes | Values | Memory vs FP32 | Typical accuracy loss |
|--------|-------|--------|----------------|----------------------|
| FP32   | 4     | ~4B    | 1x             | Baseline             |
| FP16   | 2     | 65K    | 0.5x           | Negligible           |
| INT8   | 1     | 256    | 0.25x          | < 1% on most tasks  |
| INT4   | 0.5   | 16     | 0.125x         | 1-5%, needs calibration |

**INT4-specific challenges:**
With only 16 representable values per weight, outlier weights (common in larger models — Dettmers et al. 2022 showed LLM.int8() specifically addresses outliers) cause disproportionate quantization error. Naive INT4 can degrade accuracy significantly. Calibration-based methods like GPTQ (post-training quantization using Hessian information) and AWQ (activation-aware weight quantization) achieve INT4 with 1-2% accuracy loss by compensating for outliers.

---

### 10. TTFT vs TBT

**TTFT (Time to First Token):**
Measures time from request submission to when the first output token is produced. This is dominated by the **prefill** phase, where the full prompt is processed in a single forward pass. Prefill processes all prompt tokens in parallel (like standard transformer inference) — it is **compute-bound** (GPU utilization is high, matrix multiplications are the bottleneck). TTFT scales with prompt length.

**TBT (Time Between Tokens):**
Measures time between successive generated tokens in the decode phase. Each token is generated sequentially — one forward pass produces one token. The model must read all weight matrices once per token, making decode **memory-bandwidth-bound** (the GPU's arithmetic units are underutilized; the bottleneck is loading weights from HBM). TBT is largely independent of prompt length (assuming KV cache is populated) and scales with model size and hardware memory bandwidth.

**Optimization implications:**
- To reduce TTFT: use speculative decoding, optimize prefill batching, use tensor parallelism to increase compute throughput.
- To reduce TBT: use quantization (fewer bytes to read per step), increase HBM bandwidth (better GPU), use continuous batching (amortize weight reads across multiple sequences), or use speculative decoding with a small draft model.

---

## Part 2: Coding Challenge — Reference Implementation

```python
"""TextGenerator: combines sampling strategies for autoregressive generation."""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    return exp_l / exp_l.sum()


def _temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature; clamp to avoid division by zero."""
    return logits / max(temperature, 1e-8)


def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep smallest set of tokens covering probability mass >= p."""
    if p >= 1.0:
        return logits.copy()
    shifted = logits - np.max(logits)
    exp_l = np.exp(shifted)
    probs = exp_l / exp_l.sum()
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    remove = sorted_idx[cumsum - sorted_probs > p]
    filtered = logits.copy()
    filtered[remove] = -np.inf
    return filtered


def _apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float,
) -> np.ndarray:
    """Penalize logits for tokens already generated in the sequence."""
    if penalty == 1.0 or not generated_ids:
        return logits.copy()
    penalized = logits.copy()
    for tid in set(generated_ids):
        if 0 <= tid < len(logits):
            if penalized[tid] > 0:
                penalized[tid] /= penalty
            else:
                penalized[tid] *= penalty
    return penalized


class TextGenerator:
    """Autoregressive text generator with configurable sampling strategies.

    Args:
        vocab_size: Size of the token vocabulary.
        get_logits: Callable mapping current token_ids to next-token logits
            of shape [vocab_size]. Simulates a language model forward pass.
    """

    def __init__(
        self,
        vocab_size: int,
        get_logits: Callable[[list[int]], np.ndarray],
    ) -> None:
        self._vocab_size = vocab_size
        self._get_logits = get_logits
        logger.info("TextGenerator initialized, vocab_size=%d", vocab_size)

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.3,
    ) -> list[int]:
        """Generate max_new_tokens tokens autoregressively.

        For each step:
        1. Call get_logits with the current context.
        2. Apply repetition penalty.
        3. If temperature < 1e-7: greedy (argmax).
        4. Otherwise: temperature scale, top_p filter, softmax, sample.

        Args:
            prompt_ids: Input token IDs (prompt context).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature. 0 = greedy.
            top_p: Nucleus sampling threshold. 1.0 = no filtering.
            repetition_penalty: Penalty >= 1.0 for repeated tokens.

        Returns:
            List of exactly max_new_tokens generated token IDs.

        Raises:
            ValueError: If get_logits returns wrong-shaped output.
        """
        context = list(prompt_ids)
        generated: list[int] = []

        for step in range(max_new_tokens):
            logits = self._get_logits(context)
            if logits.shape[0] != self._vocab_size:
                raise ValueError(
                    f"get_logits returned shape {logits.shape},"
                    f" expected [{self._vocab_size}]"
                )

            # Apply repetition penalty to raw logits
            logits = _apply_repetition_penalty(logits, generated, repetition_penalty)

            if temperature < 1e-7:
                # Greedy: no randomness
                token_id = int(np.argmax(logits))
            else:
                logits = _temperature_scale(logits, temperature)
                logits = _top_p_filter(logits, top_p)
                probs = _softmax(logits)
                token_id = int(np.random.choice(self._vocab_size, p=probs))

            generated.append(token_id)
            context.append(token_id)
            logger.debug("step=%d, token_id=%d", step, token_id)

        return generated
```

---

## Part 2: Coding Challenge — Tests

```python
"""Tests for TextGenerator reference implementation."""
from __future__ import annotations

import numpy as np
import pytest

from text_generator import TextGenerator


VOCAB_SIZE = 50


def make_peaked_logits(peak_token: int) -> np.ndarray:
    """Create logits strongly peaked at peak_token."""
    logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
    logits[peak_token] = 10.0
    return logits


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_generate_returns_correct_length() -> None:
    """generate() must return exactly max_new_tokens tokens."""
    def get_logits(ids: list[int]) -> np.ndarray:
        return make_peaked_logits(5)

    gen = TextGenerator(VOCAB_SIZE, get_logits)
    result = gen.generate([1, 2, 3], max_new_tokens=7, temperature=0.0)
    assert len(result) == 7


def test_generate_returns_list_of_ints() -> None:
    def get_logits(ids: list[int]) -> np.ndarray:
        return make_peaked_logits(0)

    gen = TextGenerator(VOCAB_SIZE, get_logits)
    result = gen.generate([0], max_new_tokens=3, temperature=0.0)
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


def test_generate_tokens_in_vocab() -> None:
    rng = np.random.default_rng(0)

    def get_logits(ids: list[int]) -> np.ndarray:
        return rng.random(VOCAB_SIZE).astype(np.float32)

    gen = TextGenerator(VOCAB_SIZE, get_logits)
    result = gen.generate([0], max_new_tokens=20, temperature=1.0)
    assert all(0 <= t < VOCAB_SIZE for t in result)


# ---------------------------------------------------------------------------
# Temperature = 0 is deterministic
# ---------------------------------------------------------------------------

def test_temperature_zero_is_deterministic() -> None:
    """Temperature=0 must produce identical results on repeated calls."""
    peak_token = 42

    def get_logits(ids: list[int]) -> np.ndarray:
        return make_peaked_logits(peak_token)

    gen = TextGenerator(VOCAB_SIZE, get_logits)
    r1 = gen.generate([0], max_new_tokens=5, temperature=0.0)
    r2 = gen.generate([0], max_new_tokens=5, temperature=0.0)
    assert r1 == r2 == [peak_token] * 5


def test_temperature_zero_selects_argmax() -> None:
    """Temperature=0 must always select the highest-logit token."""
    def get_logits(ids: list[int]) -> np.ndarray:
        logits = np.zeros(VOCAB_SIZE, dtype=np.float32)
        # Highest logit at token 17
        logits[17] = 8.0
        logits[5] = 3.0
        return logits

    gen = TextGenerator(VOCAB_SIZE, get_logits)
    result = gen.generate([0], max_new_tokens=10, temperature=0.0)
    assert all(t == 17 for t in result)


# ---------------------------------------------------------------------------
# Repetition penalty
# ---------------------------------------------------------------------------

def test_repetition_penalty_reduces_repetition() -> None:
    """With very high repetition penalty, a dominant token should not repeat excessively."""
    call_count = [0]

    def get_logits(ids: list[int]) -> np.ndarray:
        call_count[0] += 1
        logits = np.ones(VOCAB_SIZE, dtype=np.float32)
        logits[7] = 5.0  # token 7 is dominant at every step
        return logits

    gen = TextGenerator(VOCAB_SIZE, get_logits)
    np.random.seed(42)
    result = gen.generate(
        [0], max_new_tokens=10, temperature=0.1, repetition_penalty=10.0
    )
    token_7_count = result.count(7)
    # With penalty=10, after first selection token 7's logit drops by 10x
    # It should appear at most 3 times out of 10 with temperature=0.1
    assert token_7_count <= 3


def test_repetition_penalty_one_no_effect() -> None:
    """penalty=1.0 should not change results vs no penalty at all."""
    def get_logits(ids: list[int]) -> np.ndarray:
        return make_peaked_logits(5)

    np.random.seed(0)
    gen = TextGenerator(VOCAB_SIZE, get_logits)
    r_penalty = gen.generate([0], max_new_tokens=5, temperature=0.0, repetition_penalty=1.0)

    np.random.seed(0)
    r_no_penalty = gen.generate([0], max_new_tokens=5, temperature=0.0, repetition_penalty=1.0)
    assert r_penalty == r_no_penalty


# ---------------------------------------------------------------------------
# Wrong-shaped logits raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_vocab_size_raises() -> None:
    def bad_logits(ids: list[int]) -> np.ndarray:
        return np.ones(VOCAB_SIZE + 1, dtype=np.float32)

    gen = TextGenerator(VOCAB_SIZE, bad_logits)
    with pytest.raises(ValueError, match="get_logits returned shape"):
        gen.generate([0], max_new_tokens=1)
```

---

## Grading Notes

### Full Marks — Independent Implementation
Award full marks when the student wrote all function bodies from scratch without copying the reference. Signs of independent work: variable names differ from reference, alternative but equivalent algorithms (e.g., using `np.sort` instead of `np.argsort` in top_p), tests with different edge case values.

### Conceptual Questions — Partial Credit
- Q6 (Temperature math): Award 3/5 for "approaches argmax as T→0" without the softmax derivation.
- Q8 (Scaling laws): Award 3/5 for stating "20 tokens per parameter" without the compute-optimal formula.
- Q9 (Quantization): Award 3/5 for naming accuracy loss without explaining outlier weights.
- Q10 (TTFT vs TBT): Award 3/5 for correctly distinguishing prefill vs decode without naming compute-bound vs memory-bandwidth-bound.

### Common Bugs — Debugging Exercise

Present students with this broken `top_p_filter`:

```python
# BUG: wrong cumsum condition
def top_p_filter_broken(logits: np.ndarray, p: float) -> np.ndarray:
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.exp(shifted).sum()
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    # BUG: uses cumsum > p instead of (cumsum - sorted_probs) > p
    # This removes the token that FIRST pushes cumsum over p,
    # shrinking the nucleus below the requested coverage.
    cutoff_mask = cumsum > p
    remove_indices = sorted_indices[cutoff_mask]
    filtered = logits.copy()
    filtered[remove_indices] = -np.inf
    return filtered
```

Expected student answer: The condition `cumsum > p` removes the token that first pushes the cumulative sum past `p`. This means the actual nucleus probability mass is `< p` rather than `>= p`. The fix is `(cumsum - sorted_probs) > p`, which only removes tokens where the cumulative sum BEFORE adding that token already exceeded `p`, ensuring the nucleus always covers at least probability mass `p`.

### Code Quality Rubric (5 points)
- 2 pts: All public functions have type annotations
- 1 pt: All public functions have docstrings
- 1 pt: No bare `print()` — logging used throughout
- 1 pt: No function body exceeds ~30 lines (single responsibility)
