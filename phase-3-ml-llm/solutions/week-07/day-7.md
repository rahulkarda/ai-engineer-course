# Solutions — Phase 3 · Week 07 · Day 7 — Sunday Project Completion

---

## Self-Review Checklist

Before submitting the week-07 project, verify all of the following:

| # | Check | How to verify |
|---|-------|---------------|
| 1 | All functions have complete implementations (no `pass` or `...`) | Read each function body |
| 2 | Every public function has a type-annotated signature | `mypy src/` or read signatures |
| 3 | Every public function has a docstring | Read each function |
| 4 | No bare `print()` calls — all output goes through `logging` | `grep -r "print(" src/` returns nothing |
| 5 | `tests/test_sampling.py` covers all 6 public functions | Count test functions: greedy, temperature, top_k, top_p, repetition_penalty, sample_token |
| 6 | `tests/test_kv_cache.py` covers append, get, memory_bytes, seq_len, clear | Count test functions |
| 7 | All tests pass: `pytest tests/ -v` exits 0 | Run the command |
| 8 | No hardcoded magic numbers — constants are named or documented | Read all numeric literals |
| 9 | `import numpy as np` is the only numerical dependency in src/ | `grep -r "^import\|^from" src/` |
| 10 | `sample_token()` integrates all sub-functions in the documented order | Trace the function body step by step |

---

## Reflection Questions

### Q1: What would break if temperature=0 but top_p=0.5?

**Answer:**

There are two possible behaviors depending on implementation quality:

**In the reference implementation (correct behavior):**
When `temperature < 1e-7`, `sample_token()` short-circuits with `greedy_decode()` before reaching the top_p filter. Temperature effectively "overrides" top_p because we never reach that code path. The result is deterministic argmax selection — exactly one token is produced, and top_p has no effect.

**In a naive implementation that does not short-circuit:**
Dividing logits by temperature=0 causes a division-by-zero. In NumPy, positive logits become `+inf`, negative logits become `-inf`, and zero logits become `NaN`. The top_p filter then receives logits with `inf`/`NaN` values. The softmax over these produces `NaN` probabilities (from `inf - inf` in the stable softmax). `np.random.choice` with a `NaN` probability array either raises an error or silently produces garbage.

**Correct mental model:**
Temperature=0 means "infinite confidence in the highest-probability token." The one-hot distribution this produces makes top_p irrelevant — the nucleus is always just the argmax token. Any implementation that computes top_p after setting temperature=0 is wastefully computing a filter on an already-deterministic distribution. Short-circuiting to greedy is both mathematically correct and computationally cheaper.

---

### Q2: If KV cache memory exceeds GPU VRAM, what strategies exist?

**Answer:**

Five practical strategies, ordered from lowest to highest engineering complexity:

1. **Reduce context length or batch size.** The simplest mitigation: use a shorter `max_seq_len` or decrease the number of concurrent requests. KV cache memory scales as `O(batch_size * seq_len)`, so halving either frees half the cache.

2. **KV cache quantization (INT8 / INT4).** Store cached K and V tensors in reduced precision. INT8 KV cache halves memory versus FP16 with negligible accuracy impact on most models. INT4 KV cache (used in some quantized inference engines) achieves 4x reduction. The tradeoff: slightly slower dequantization overhead per attention step.

3. **Sliding window attention.** Discard cached K/V entries for the oldest tokens, keeping only the most recent `window_size` positions. Effective for tasks where distant context is less critical (code generation, chat). The tradeoff: early prompt tokens are permanently forgotten, which breaks tasks that require long-range retrieval.

4. **Paged attention (vLLM's PagedAttention).** Instead of a contiguous K/V cache block per sequence, store cache in fixed-size non-contiguous "pages" analogous to OS virtual memory. Pages can be shared across sequences with the same prefix (prompt caching), and memory is allocated on demand rather than pre-allocated for `max_seq_len`. This dramatically increases effective GPU utilization and enables batching more sequences simultaneously. Tradeoff: requires a modified attention kernel; not drop-in compatible with standard transformer code.

5. **CPU offloading.** Move K/V pages for less-recently-accessed tokens to CPU RAM, fetching them back to GPU just-in-time for attention computation. CPU RAM is typically 8-16x larger than GPU VRAM. Tradeoff: PCIe bandwidth (typically 32-64 GB/s) becomes a bottleneck for long sequences, adding latency per decode step proportional to the number of offloaded pages.

---

### Q3: Why does instruction tuning alone (without RLHF) sometimes produce harmful outputs?

**Answer:**

Supervised fine-tuning (SFT) for instruction following trains the model to minimize cross-entropy loss on response tokens given instruction-response pairs. This objective has three structural limitations that allow harmful outputs to persist:

**1. The loss function is indifferent to response quality.**
Cross-entropy measures whether the model assigns high probability to the training response tokens. It has no mechanism to distinguish "this is a helpful, safe response" from "this is a harmful response" — both score equally well if the tokens match. If the training dataset contains a mix of helpful and harmful demonstrations (or if the model encounters prompts not covered by the training distribution), the model can generate harmful completions that match the token distribution it learned.

**2. SFT amplifies base-model tendencies.**
The pretrained base model has absorbed potentially harmful patterns from web-scale data (toxic text, misinformation, manipulation tactics). SFT teaches the model to follow instruction format but does not selectively suppress these capabilities. For prompts that weren't in the SFT dataset, the model falls back on base-model behavior, which can include harmful outputs.

**3. Instruction format does not imply intent alignment.**
A model trained on SFT learns that "here is a question / here is an answer" is the expected structure. It does not learn that safety and helpfulness are properties the human user actually cares about. RLHF fixes this by: (a) collecting human preference data where raters explicitly rank responses by quality, safety, and helpfulness; (b) training a reward model to predict these preferences; and (c) using reinforcement learning (PPO) to fine-tune the policy to maximize reward-model scores. This explicitly encodes human values into the optimization objective rather than hoping the SFT dataset implicitly represents them.

**Practical consequence:**
Models like InstructGPT (the first RLHF model) showed significant reductions in harmful outputs versus SFT-only models of the same size, even though the SFT-only model was technically a stronger next-token predictor on held-out instruction data.

---

## Demo: Five Sampling Strategies Compared

The following `demo.py` shows how the same logits produce different token sequences under different sampling configurations.

```python
"""demo.py — compare sampling strategies on a fixed logit distribution."""
from __future__ import annotations

import logging
import sys

import numpy as np

# Adjust this import path to match your project layout
sys.path.insert(0, "src")
from sampling import sample_token

logging.basicConfig(level=logging.WARNING)  # suppress debug noise in demo


def run_demo() -> None:
    rng = np.random.default_rng(seed=7)
    vocab_size = 20
    # Logits: token 3 is dominant, token 12 is second, rest are noisy
    logits = rng.normal(loc=0.0, scale=1.5, size=vocab_size).astype(np.float32)
    logits[3] = 6.0
    logits[12] = 4.0

    strategies: list[tuple[str, dict]] = [
        (
            "Greedy (temperature=0)",
            {"temperature": 0.0},
        ),
        (
            "High temperature (T=2.0)",
            {"temperature": 2.0},
        ),
        (
            "Top-k=3, T=1.0",
            {"temperature": 1.0, "top_k": 3},
        ),
        (
            "Nucleus p=0.9, T=1.0",
            {"temperature": 1.0, "top_p": 0.9},
        ),
        (
            "Nucleus p=0.9, T=1.0, rep_penalty=1.5",
            {"temperature": 1.0, "top_p": 0.9, "repetition_penalty": 1.5},
        ),
    ]

    n_tokens = 10
    generated_ids: list[int] = []

    for name, kwargs in strategies:
        np.random.seed(42)
        generated_ids = []
        tokens: list[int] = []
        for _ in range(n_tokens):
            token_id = sample_token(logits, generated_ids=generated_ids, **kwargs)
            tokens.append(token_id)
            generated_ids.append(token_id)
        # Demo output — using sys.stdout directly for display, not logging
        sys.stdout.write(f"  {name}:\n    tokens = {tokens}\n\n")


if __name__ == "__main__":
    run_demo()
```

**Expected output (approximate — stochastic strategies vary by seed):**

```
  Greedy (temperature=0):
    tokens = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

  High temperature (T=2.0):
    tokens = [3, 12, 7, 3, 1, 15, 3, 0, 12, 4]

  Top-k=3, T=1.0:
    tokens = [3, 3, 12, 3, 3, 12, 3, 3, 3, 12]

  Nucleus p=0.9, T=1.0:
    tokens = [3, 3, 12, 3, 12, 3, 3, 3, 12, 3]

  Nucleus p=0.9, T=1.0, rep_penalty=1.5:
    tokens = [3, 12, 3, 12, 3, 12, 7, 3, 12, 3]
```

**Observations from the output:**

- Greedy always selects token 3 (highest logit = 6.0) — zero diversity.
- High temperature increases diversity dramatically, allowing low-probability tokens to appear.
- Top-k=3 restricts sampling to tokens 3, 12, and the third-highest — more diverse than greedy, more focused than high temperature.
- Nucleus p=0.9 produces similar diversity to top-k=3 in this case because the top-2 tokens already account for most of the probability mass.
- Repetition penalty with p=0.9 noticeably increases variety by penalizing token 3 after it first appears, giving token 12 and others a better chance on subsequent steps.
