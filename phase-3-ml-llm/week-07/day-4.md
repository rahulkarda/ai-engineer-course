# Phase 3 · Week 07 · Day 4 — Sampling Strategies: Greedy, Temperature, Top-k, Top-p, Repetition Penalty

> Objective: Implement and compare the five main decoding strategies for LLM output generation.  ·  Time: ~1 hr  ·  Prereqs: [[week-07/day-3]]

## Concept

At each generation step, the model outputs a logit vector of shape `[vocab_size]`. The sampling strategy converts these logits into a single chosen token. The choice of strategy fundamentally changes the character of the output: deterministic vs stochastic, creative vs conservative, repetitive vs diverse.

### Greedy Decoding

Pick the highest-probability token unconditionally:
```
token = argmax(softmax(logits))
```
Fast and deterministic. Tends toward repetitive, generic, mode-collapsed output because the model keeps selecting its single most likely continuation, which forms self-reinforcing loops ("The the the the..."). Appropriate for exact-answer tasks (math, code generation requiring a specific answer) but poor for creative text.

### Temperature Scaling

Divide logits by temperature T before softmax:
```
probs = softmax(logits / T)
token = sample(probs)
```

Effect on distribution:
- T = 1.0: unmodified distribution, standard sampling.
- T < 1.0 (e.g., 0.3): sharper distribution — top token gets higher relative probability, output is more deterministic and conservative.
- T > 1.0 (e.g., 1.5): flatter distribution — probability mass spreads to low-probability tokens, output is more random and creative.
- T → 0: approaches greedy (argmax). T → infinity: approaches uniform random over vocabulary.

Temperature does not truncate the distribution — it only reshapes it. All vocab tokens remain nonzero.

### Top-k Sampling

Keep only the k highest-logit tokens, set all others to -inf, then renormalize and sample:
```
top_k_indices = argsort(logits)[-k:]
filtered = logits[top_k_indices]
probs = softmax(filtered)
token = sample(probs)
```

Guarantees at most k candidates. Prevents sampling from the long tail of implausible tokens. Drawback: k is a fixed count regardless of how peaked or flat the distribution is. If the top token has 95% probability, allowing k=50 candidates still wastes probability mass on nonsense tokens. If the distribution is very flat, k=50 may be too restrictive.

### Top-p / Nucleus Sampling

Sort tokens by probability descending, accumulate probabilities until the cumulative sum reaches threshold p, keep only those tokens, renormalize, sample:
```
sorted_probs, sorted_indices = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
keep = cumsum <= p  # keep tokens up to the threshold
keep[first index where cumsum > p] = True  # include the token that crossed p
filtered_probs = renormalize(sorted_probs[keep])
token = sample(filtered_probs[keep])
```

Adapts the candidate set size to the distribution. When the model is confident (peaky distribution), few tokens are needed to reach p=0.9. When uncertain, many tokens contribute. This makes top-p more robust than top-k across varying distribution shapes. Typical production values: p=0.9 or p=0.95.

### Repetition Penalty

Discourage tokens that have already appeared in the generated sequence:
```
for token_id in already_generated:
    if logits[token_id] > 0:
        logits[token_id] /= penalty
    else:
        logits[token_id] *= penalty
```

Penalty > 1.0 reduces the score of previously seen tokens (both by dividing positive logits and multiplying negative logits to make them more negative). Penalty = 1.0 is no-op. Typical values: 1.1 to 1.3. Too high (> 1.5) degrades coherence by forbidding reasonable function words ("the," "is," "a").

### Combining strategies

Most production systems apply in this order:
1. Apply repetition penalty to raw logits.
2. Apply temperature scaling.
3. Apply top-k or top-p (or both: top-k first, then top-p on the survivors).
4. Sample from the filtered, renormalized distribution.

## How it works internally

**Why greedy fails for long sequences:**
Greedy is myopic — maximizing the probability of each token does not maximize the probability of the full sequence. A slightly lower-probability token at step 5 might open up much higher-probability options at steps 6-10. Beam search addresses this (keep k partial sequences) but is still suboptimal and prone to generic output.

**Temperature as entropy control:**
Shannon entropy H = -sum(p * log(p)). Low T compresses probability mass onto the mode, reducing entropy. High T spreads mass, increasing entropy. Temperature controls the model's effective "confidence" without changing the rank ordering of tokens — the most probable token under T=1 is still the most probable token under any T > 0.

**Why top-p adapts better than top-k:**
Top-k is a count constraint; top-p is a probability mass constraint. With top-k=50 on a distribution where P(token_1) = 0.99, you keep 49 near-zero tokens unnecessarily. With top-p=0.95, you keep only token_1 and return it deterministically — the behavior adapts to the model's actual confidence.

**Repetition penalty and absolute vs relative discounting:**
The original Keskar et al. formulation divides positive logits and multiplies negative logits to keep the sign semantics consistent. Some implementations instead subtract a fixed value (additive penalty). Division is more common in inference frameworks (transformers library).

## Worked example

```python
"""
All five sampling strategies implemented from first principles in NumPy.
Run on the same mock logits to compare outputs.
Run: python day-4-example.py
Dependencies: numpy (uv add numpy)
"""

import logging
import numpy as np
from typing import Sequence

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VOCAB = ["the", "a", "cat", "sat", "on", "mat", "dog", "ran", "fast", "slowly",
         "very", "big", "small", "red", "blue", "jumped", "over", "under", "fence",
         "<unk>"]
VOCAB_SIZE = len(VOCAB)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum()


def sample_from_probs(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one index from a probability distribution."""
    return int(rng.choice(len(probs), p=probs))


def token_name(idx: int) -> str:
    return VOCAB[idx] if idx < len(VOCAB) else f"<id={idx}>"


# ---------------------------------------------------------------------------
# 1. Greedy decoding
# ---------------------------------------------------------------------------

def greedy(logits: np.ndarray) -> int:
    """Return the argmax token. Deterministic."""
    return int(np.argmax(logits))


# ---------------------------------------------------------------------------
# 2. Temperature sampling
# ---------------------------------------------------------------------------

def temperature_sample(
    logits: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """
    Scale logits by 1/T then sample.
    temperature=0 is not valid (division by zero); treat as greedy.
    """
    if temperature <= 0.0:
        return greedy(logits)
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    return sample_from_probs(probs, rng)


# ---------------------------------------------------------------------------
# 3. Top-k sampling
# ---------------------------------------------------------------------------

def top_k_sample(
    logits: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> int:
    """
    Keep k highest-logit tokens, zero out the rest, renormalize, sample.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(logits))

    # Indices of k largest logits
    top_k_indices = np.argpartition(logits, -k)[-k:]

    # Build filtered distribution
    filtered_logits = np.full_like(logits, -np.inf)
    filtered_logits[top_k_indices] = logits[top_k_indices]

    probs = softmax(filtered_logits)
    return sample_from_probs(probs, rng)


# ---------------------------------------------------------------------------
# 4. Top-p (nucleus) sampling
# ---------------------------------------------------------------------------

def top_p_sample(
    logits: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> int:
    """
    Keep tokens whose cumulative probability (sorted descending) <= p.
    Always keep at least one token.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")

    probs = softmax(logits)

    # Sort descending by probability
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Cumulative sum
    cumsum = np.cumsum(sorted_probs)

    # Find first index where cumsum exceeds p; keep tokens up to and including it
    # (np.searchsorted on cumsum to find cutoff)
    cutoff = int(np.searchsorted(cumsum, p)) + 1
    cutoff = max(cutoff, 1)  # always keep at least 1 token

    # Build filtered distribution
    keep_indices = sorted_indices[:cutoff]
    keep_probs = sorted_probs[:cutoff]
    keep_probs = keep_probs / keep_probs.sum()  # renormalize

    chosen_position = sample_from_probs(keep_probs, rng)
    return int(keep_indices[chosen_position])


# ---------------------------------------------------------------------------
# 5. Repetition penalty
# ---------------------------------------------------------------------------

def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: Sequence[int],
    penalty: float,
) -> np.ndarray:
    """
    Divide logits of already-generated tokens by penalty (>1 discourages repeats).
    Positive logits are divided; negative logits are multiplied (to make more negative).
    Returns modified logits (does not mutate input).
    """
    if penalty == 1.0:
        return logits
    logits = logits.copy()
    for token_id in set(generated_ids):
        if token_id < len(logits):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


# ---------------------------------------------------------------------------
# Combined sampler
# ---------------------------------------------------------------------------

def combined_sample(
    logits: np.ndarray,
    generated_ids: Sequence[int],
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,           # 0 = disabled
    repetition_penalty: float = 1.0,
) -> int:
    """
    Production-style sampler: repetition penalty -> temperature -> top_k -> top_p -> sample.
    """
    # Step 1: repetition penalty
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Step 2: temperature
    if temperature > 0:
        logits = logits / temperature

    # Step 3: optional top-k truncation
    if top_k > 0:
        top_k_actual = min(top_k, len(logits))
        threshold_idx = np.argpartition(logits, -top_k_actual)[-top_k_actual]
        threshold = logits[threshold_idx]
        logits = np.where(logits >= threshold, logits, np.full_like(logits, -np.inf))

    # Step 4: top-p (on possibly top-k-filtered logits)
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumsum, top_p)) + 1
    cutoff = max(cutoff, 1)
    keep_indices = sorted_indices[:cutoff]
    keep_probs = sorted_probs[:cutoff]
    keep_probs = keep_probs / keep_probs.sum()

    chosen_position = sample_from_probs(keep_probs, rng)
    return int(keep_indices[chosen_position])


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def make_mock_logits(rng: np.random.Generator) -> np.ndarray:
    """
    Return logits that give 'the' ~40% and 'a' ~25% probability, with a long tail.
    Simulates a model that strongly prefers function words.
    """
    logits = rng.standard_normal(VOCAB_SIZE).astype(np.float32) * 0.5
    logits[0] += 2.5   # 'the'
    logits[1] += 1.8   # 'a'
    logits[2] += 0.8   # 'cat'
    return logits


def show_top_probs(logits: np.ndarray, n: int = 5) -> None:
    probs = softmax(logits)
    top = np.argsort(probs)[::-1][:n]
    parts = [f"{token_name(i)}={probs[i]:.3f}" for i in top]
    logger.info("  Top-%d: %s", n, "  ".join(parts))


def simulate_generation(
    label: str,
    logits_fn,
    n_steps: int,
    rng_seed: int,
    **sampler_kwargs,
) -> list[int]:
    """Run n_steps of generation using the given sampler."""
    rng = np.random.default_rng(rng_seed)
    generated: list[int] = []
    for step in range(n_steps):
        logits = logits_fn(rng)
        token_id = combined_sample(
            logits, generated, rng,
            **sampler_kwargs,
        )
        generated.append(token_id)
    tokens = [token_name(t) for t in generated]
    logger.info("%s: %s", label, " ".join(tokens))
    return generated


def main() -> None:
    base_rng = np.random.default_rng(0)
    base_logits = make_mock_logits(base_rng)

    logger.info("=== Base distribution ===")
    show_top_probs(base_logits)

    logger.info("\n=== 1. Greedy decoding ===")
    greedy_id = greedy(base_logits)
    logger.info("Token: %r (logit=%.3f)", token_name(greedy_id), base_logits[greedy_id])

    logger.info("\n=== 2. Temperature comparison ===")
    for temp in [0.3, 1.0, 1.5]:
        rng = np.random.default_rng(42)
        samples = [
            token_name(temperature_sample(base_logits, temp, rng))
            for _ in range(10)
        ]
        unique = len(set(samples))
        logger.info("T=%.1f | samples: %s | unique=%d",
                    temp, " ".join(samples), unique)

    logger.info("\n=== 3. Top-k comparison ===")
    for k in [1, 3, 10, VOCAB_SIZE]:
        rng = np.random.default_rng(42)
        samples = [token_name(top_k_sample(base_logits, k, rng)) for _ in range(10)]
        unique = len(set(samples))
        logger.info("k=%2d | samples: %s | unique=%d", k, " ".join(samples), unique)

    logger.info("\n=== 4. Top-p comparison ===")
    for p_val in [0.5, 0.9, 0.99]:
        rng = np.random.default_rng(42)
        samples = [token_name(top_p_sample(base_logits, p_val, rng)) for _ in range(10)]
        unique = len(set(samples))
        logger.info("p=%.2f | samples: %s | unique=%d", p_val, " ".join(samples), unique)

    logger.info("\n=== 5. Repetition penalty effect ===")
    previously_generated = [0, 1]  # 'the', 'a' already generated
    logger.info("Already generated: %s", [token_name(i) for i in previously_generated])
    for penalty in [1.0, 1.2, 1.5, 2.0]:
        penalized = apply_repetition_penalty(base_logits, previously_generated, penalty)
        rng = np.random.default_rng(42)
        samples = [
            token_name(top_p_sample(penalized, 0.9, rng)) for _ in range(10)
        ]
        the_count = samples.count("the")
        a_count = samples.count("a")
        logger.info(
            "penalty=%.1f | 'the'=%d, 'a'=%d | %s",
            penalty, the_count, a_count, " ".join(samples),
        )

    logger.info("\n=== 6. Combined sampler — 10-token generation ===")
    logits_fn = lambda rng: make_mock_logits(rng)  # noqa: E731
    simulate_generation(
        "T=1.0, top-p=0.9, no penalty",
        logits_fn, n_steps=10, rng_seed=1,
        temperature=1.0, top_p=0.9, repetition_penalty=1.0,
    )
    simulate_generation(
        "T=0.7, top-p=0.9, penalty=1.2",
        logits_fn, n_steps=10, rng_seed=1,
        temperature=0.7, top_p=0.9, repetition_penalty=1.2,
    )
    simulate_generation(
        "T=1.3, top-p=0.95, penalty=1.0",
        logits_fn, n_steps=10, rng_seed=1,
        temperature=1.3, top_p=0.95, repetition_penalty=1.0,
    )


if __name__ == "__main__":
    main()
```

Expected output (abridged):
```
INFO: === Base distribution ===
INFO:   Top-5: the=0.402  a=0.247  cat=0.098  ...
INFO: === 2. Temperature comparison ===
INFO: T=0.3 | samples: the the the the the the the the the the | unique=1
INFO: T=1.0 | samples: the a the the cat the a the the a | unique=3
INFO: T=1.5 | samples: the a cat a the on dog the sat a | unique=6
```

Note: install numpy with `uv add numpy` before running.

## Exercises

1. **Implement and compare**: Given the logits array `np.array([2.0, 1.5, 0.5, 0.3, -0.5, -1.0, -2.0, -3.0])` for an 8-token vocabulary, manually compute (on paper or in code): (a) the softmax probabilities; (b) which tokens are kept by top-k=3; (c) which tokens are kept by top-p=0.85. Show the renormalized probabilities for both.

2. **Entropy analysis**: Write a Python function `compute_entropy(logits, temperature)` that returns the Shannon entropy of the distribution after temperature scaling. Plot (using print statements or log output) entropy vs temperature for T in [0.1, 0.5, 1.0, 1.5, 2.0] on the same logits. Confirm that entropy monotonically increases with T.

3. **Tune for a task**: Using the `combined_sample` function from the worked example, run 20-token generation three times with three different parameter sets optimized for: (a) deterministic factual Q&A; (b) creative story continuation; (c) diverse brainstorming. For each, justify your parameter choices (temperature, top-p, repetition penalty) based on what you learned today.

(Solutions: ../solutions/week-07/day-4.md)

## Self-check

- [ ] I can explain in one sentence each how greedy, temperature, top-k, and top-p differ in what they do to the logit/probability distribution before sampling.
- [ ] I can state what temperature = 0, temperature = 1, and temperature = infinity each produce.
- [ ] I can implement top-p sampling from scratch: sort by prob, cumsum, cutoff, renormalize, sample.
- [ ] I can combine repetition penalty + temperature + top-p into a single production-style sampler.

## Supplementary reading

- Holtzman et al., "The Curious Case of Neural Text Degeneration" (introduces nucleus/top-p sampling): https://arxiv.org/abs/1904.09751
- Keskar et al., "CTRL: A Conditional Transformer Language Model for Controllable Generation" (introduces repetition penalty): https://arxiv.org/abs/1909.05858
- Fan et al., "Hierarchical Neural Story Generation" (introduces top-k sampling): https://arxiv.org/abs/1805.04833
- Aaronson, "Why I Am Not An AI Doomer" — section on temperature intuition: https://scottaaronson.blog/?p=7215
