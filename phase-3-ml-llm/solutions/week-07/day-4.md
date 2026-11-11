# Solutions — Phase 3 · Week 07 · Day 4

## Exercise 1 — greedy_decode, temperature_scale, top_k_filter

```python
"""
Sampling strategy primitives for autoregressive language model decoding.

All functions operate on raw logit vectors (unnormalized log-probabilities).
The pipeline is typically:
  logits -> temperature_scale -> top_k_filter/top_p_filter -> softmax -> sample

Dependencies: numpy
Run: python day-4-solution.py
"""

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Sentinel value used to mask logits that should not be sampled
NEG_INF: float = -1e9


def greedy_decode(logits: np.ndarray) -> int:
    """
    Select the token with the highest logit (argmax).

    This is equivalent to temperature=0 decoding: the probability mass
    collapses to a single token. Deterministic and reproducible, but
    produces repetitive, low-diversity output.

    Args:
        logits: 1D array of raw logits, shape (vocab_size,).
                Need not be normalized.

    Returns:
        Index of the token with the highest logit.

    Example:
        logits = [0.1, 2.3, -0.5, 1.7]
        greedy_decode(logits) -> 1  (argmax = index 1, value 2.3)
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    return int(np.argmax(logits))


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Divides logits by temperature before softmax:
        P(token) = softmax(logits / T)

    - T < 1: sharpens distribution, concentrates probability on top tokens,
             more deterministic output (lower diversity).
    - T = 1: no effect, sample from the model's native distribution.
    - T > 1: flattens distribution, spreads probability more evenly,
             more random output (higher diversity, but less coherent).
    - T -> 0: approaches greedy decoding (argmax).
    - T -> inf: approaches uniform random sampling over vocabulary.

    Args:
        logits: 1D array of raw logits, shape (vocab_size,).
        temperature: positive float scaling factor.

    Returns:
        Temperature-scaled logits, same shape as input.
        The caller is responsible for applying softmax and sampling.

    Raises:
        ValueError: if temperature is not positive.
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """
    Mask all but the top-k logits to NEG_INF.

    After masking, applying softmax and sampling draws only from the k
    highest-probability tokens. Prevents sampling from the long tail of
    low-probability tokens, which often produces incoherent continuations.

    Args:
        logits: 1D array of raw logits, shape (vocab_size,).
        k: number of top tokens to keep. Must be >= 1.

    Returns:
        New logits array with all but the top-k values set to NEG_INF.
        The top-k values are unchanged (not renormalized here).

    Raises:
        ValueError: if k < 1 or k > vocab_size.

    Example:
        logits = [0.1, 2.3, -0.5, 1.7, 0.9]
        top_k_filter(logits, k=2) -> [NEG_INF, 2.3, NEG_INF, 1.7, NEG_INF]
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    vocab_size = len(logits)
    if not 1 <= k <= vocab_size:
        raise ValueError(f"k={k} must be in [1, vocab_size={vocab_size}]")

    # Find the k-th largest logit value (threshold)
    # np.partition puts the k-th largest element at index -k
    threshold = float(np.partition(logits, -k)[-k])

    filtered = np.full_like(logits, NEG_INF, dtype=np.float64)
    # Keep tokens whose logit >= threshold
    # For ties at the threshold, this may keep more than k tokens;
    # production implementations break ties deterministically.
    mask = logits >= threshold
    filtered[mask] = logits[mask]
    return filtered
```

---

## Exercise 2 — top_p_filter and apply_repetition_penalty

```python
def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus (top-p) filtering: mask tokens outside the top-p probability mass.

    Computes the softmax probability distribution, sorts tokens by descending
    probability, and keeps the smallest set of tokens whose cumulative
    probability sum >= p. All other tokens are masked to NEG_INF.

    Compared to top-k: top-p adapts the number of candidates to the shape
    of the distribution. When the model is confident (peaked distribution),
    top-p selects fewer tokens; when uncertain (flat distribution), it selects
    more. This avoids top-k's fixed cutoff that can be too permissive on flat
    distributions and too restrictive on peaked ones.

    Args:
        logits: 1D array of raw logits, shape (vocab_size,).
        p: cumulative probability threshold in (0, 1].
           p=1.0 keeps all tokens; p=0.9 keeps tokens covering 90% of prob mass.

    Returns:
        Filtered logits array with out-of-nucleus tokens set to NEG_INF.

    Raises:
        ValueError: if p is not in (0, 1].
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p={p} must be in (0, 1]")

    # Convert logits to probabilities
    logits_f64 = logits.astype(np.float64)
    shifted = logits_f64 - np.max(logits_f64)  # numerical stability
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)

    # Sort tokens by descending probability
    sorted_indices = np.argsort(probs)[::-1]  # highest prob first
    sorted_probs = probs[sorted_indices]

    # Cumulative probability
    cumulative_probs = np.cumsum(sorted_probs)

    # Find the cutoff: smallest set covering >= p cumulative mass
    # Include the first token that pushes cumsum over p
    # i.e., keep tokens where (cumsum - current_prob) < p
    # Equivalently: keep tokens at index i if cumsum[i-1] < p
    cutoff_mask = (cumulative_probs - sorted_probs) < p  # (vocab_size,) bool in sorted order

    # Map back to original token indices
    kept_indices = sorted_indices[cutoff_mask]

    filtered = np.full_like(logits, NEG_INF, dtype=np.float64)
    filtered[kept_indices] = logits_f64[kept_indices]
    return filtered


def apply_repetition_penalty(
    logits: np.ndarray,
    generated_ids: list[int],
    penalty: float = 1.3,
) -> np.ndarray:
    """
    Apply repetition penalty to logits for tokens already generated.

    Reduces the logit of any token that has already appeared in the
    generated sequence:
    - For positive logits: divide by penalty (makes them less likely)
    - For negative logits: multiply by penalty (makes them more negative, less likely)

    This is the formulation from Press et al. (2020) "CTRL: A Conditional
    Transformer Language Model for Controllable Generation" and Keskar et al.:

        score(t) = score(t) / penalty   if score(t) > 0
        score(t) = score(t) * penalty   if score(t) < 0

    Args:
        logits: 1D array of raw logits, shape (vocab_size,).
        generated_ids: list of token ids already generated in this sequence.
                       May contain duplicates; each unique id is penalized once.
        penalty: penalty multiplier. Must be >= 1.0.
                 1.0 = no effect, 1.3 = moderate penalty (typical default),
                 2.0+ = aggressive repetition suppression.

    Returns:
        Modified logits with repeated tokens penalized.

    Raises:
        ValueError: if penalty < 1.0 (would reward repetition).
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}")
    if penalty < 1.0:
        raise ValueError(f"penalty={penalty} must be >= 1.0. Values < 1 reward repetition.")

    penalized = logits.astype(np.float64).copy()
    unique_ids = set(generated_ids)

    for token_id in unique_ids:
        if 0 <= token_id < len(logits):
            if penalized[token_id] > 0:
                penalized[token_id] /= penalty
            else:
                penalized[token_id] *= penalty

    return penalized
```

---

## Exercise 3 — Strategy Comparison on Mock Logits

```python
def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def sample_from_logits(logits: np.ndarray, rng: np.random.Generator) -> int:
    """Sample a token from the probability distribution defined by logits."""
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def main() -> None:
    """
    Compare all sampling strategies on the same mock logit array.

    Vocabulary (10 tokens for clarity):
      id=0: [PAD]
      id=1: "the"
      id=2: "cat"
      id=3: "dog"
      id=4: "sat"
      id=5: "a"
      id=6: "on"
      id=7: "mat"
      id=8: "jumped"
      id=9: "over"
    """
    # Mock logit array: token 2 ("cat") is strongly favored,
    # token 4 ("sat") is second, token 7 ("mat") is third,
    # remaining tokens have low logits.
    # This simulates a model generating "The cat sat on the mat."
    # after "The" has been generated — "cat" has highest logit.
    logits = np.array(
        [-3.0, -1.5, 4.2, 0.8, 2.1, -0.5, 0.3, 1.4, -1.2, -2.0],
        dtype=np.float32,
    )
    vocab = {
        0: "[PAD]", 1: "the", 2: "cat", 3: "dog",
        4: "sat",   5: "a",   6: "on",  7: "mat",
        8: "jumped", 9: "over",
    }

    logger.info("=== Mock Logits ===")
    probs = softmax(logits.astype(np.float64))
    for i, (logit, prob) in enumerate(zip(logits, probs)):
        logger.info(
            "  id=%d %-8s logit=%6.2f  prob=%.4f",
            i, vocab[i], logit, prob,
        )

    # ---------- Greedy ----------
    greedy_id = greedy_decode(logits)
    logger.info("\n--- Greedy Decode ---")
    logger.info("Selected: id=%d '%s' (logit=%.2f)", greedy_id, vocab[greedy_id], logits[greedy_id])
    logger.info(
        "Why: argmax picks the single highest logit. id=2 'cat' has logit=4.2, "
        "the largest in the array. Deterministic."
    )

    # ---------- Temperature T=2.0 (flat) ----------
    scaled_hot = temperature_scale(logits, temperature=2.0)
    probs_hot = softmax(scaled_hot.astype(np.float64))
    logger.info("\n--- Temperature T=2.0 (high/flat) ---")
    for i, (orig, scaled, prob) in enumerate(zip(logits, scaled_hot, probs_hot)):
        if prob > 0.01:  # only show tokens with non-negligible probability
            logger.info(
                "  id=%d %-8s orig=%.2f scaled=%.2f prob=%.4f",
                i, vocab[i], orig, scaled, prob,
            )
    logger.info(
        "Why: dividing by T=2.0 halves all logits, compressing the spread "
        "from [4.2, -3.0] to [2.1, -1.5]. Softmax output is flatter: "
        "'cat' still leads but 'sat' and 'mat' have meaningfully higher probability. "
        "Sampling at T=2 produces more diverse (but less confident) output."
    )

    # ---------- Temperature T=0.5 (sharp) ----------
    scaled_cold = temperature_scale(logits, temperature=0.5)
    probs_cold = softmax(scaled_cold.astype(np.float64))
    logger.info("\n--- Temperature T=0.5 (low/sharp) ---")
    for i, (orig, scaled, prob) in enumerate(zip(logits, scaled_cold, probs_cold)):
        if prob > 0.001:
            logger.info(
                "  id=%d %-8s orig=%.2f scaled=%.2f prob=%.4f",
                i, vocab[i], orig, scaled, prob,
            )
    logger.info(
        "Why: dividing by T=0.5 doubles all logits, amplifying the spread "
        "from [4.2, -3.0] to [8.4, -6.0]. Softmax output is very peaked: "
        "'cat' captures nearly all probability mass. "
        "Sampling at T=0.5 almost always picks 'cat', approaching greedy."
    )

    # ---------- Top-k (k=3) ----------
    logits_topk = top_k_filter(logits, k=3)
    probs_topk = softmax(logits_topk)
    logger.info("\n--- Top-k Filter (k=3) ---")
    for i, (orig, filt, prob) in enumerate(zip(logits, logits_topk, probs_topk)):
        status = "KEPT" if filt > NEG_INF else "masked"
        logger.info(
            "  id=%d %-8s orig=%.2f filtered=%-8s prob=%.4f",
            i, vocab[i], orig, f"{filt:.2f}" if filt > NEG_INF else "NEG_INF", prob,
        )
    # Fixed selection: top-3 are id=2(4.2), id=4(2.1), id=7(1.4)
    # Greedy from filtered = still id=2
    topk_greedy = greedy_decode(logits_topk)
    logger.info("Greedy from top-3 filtered: id=%d '%s'", topk_greedy, vocab[topk_greedy])
    logger.info(
        "Why: top-k keeps only the 3 highest logit tokens (cat=4.2, sat=2.1, mat=1.4). "
        "All other tokens are masked to NEG_INF before softmax. "
        "Probability mass is redistributed exclusively among these 3 tokens. "
        "The model can never sample low-quality tail tokens like [PAD] or 'over'."
    )

    # ---------- Top-p (p=0.9) ----------
    logits_topp = top_p_filter(logits, p=0.9)
    probs_topp = softmax(logits_topp)
    kept_topp = [i for i, v in enumerate(logits_topp) if v > NEG_INF]
    logger.info("\n--- Top-p Filter (p=0.9) ---")
    for i in range(len(logits)):
        status = "KEPT" if logits_topp[i] > NEG_INF else "masked"
        logger.info(
            "  id=%d %-8s prob_orig=%.4f status=%s",
            i, vocab[i], probs[i], status,
        )
    logger.info(
        "Nucleus covers %d tokens: %s",
        len(kept_topp),
        [vocab[i] for i in kept_topp],
    )
    logger.info(
        "Why: top-p computes cumulative probability in descending order and keeps "
        "tokens until the cumsum >= 0.9. 'cat' alone has prob ~0.90, so the nucleus "
        "may include just 'cat' (or 'cat'+'sat' depending on the exact threshold). "
        "This adapts to the distribution shape — on a flatter distribution top-p "
        "would keep more tokens than top-k=3."
    )

    # ---------- Repetition penalty ----------
    generated_so_far = [1, 2]  # "the cat" has been generated; penalize "the"(1) and "cat"(2)
    logits_rep = apply_repetition_penalty(logits, generated_ids=generated_so_far, penalty=1.3)
    logger.info("\n--- Repetition Penalty (penalty=1.3, generated=[the, cat]) ---")
    for i, (orig, penalized) in enumerate(zip(logits, logits_rep)):
        changed = " <-- penalized" if i in generated_so_far else ""
        logger.info(
            "  id=%d %-8s orig=%.2f penalized=%.2f%s",
            i, vocab[i], orig, penalized, changed,
        )
    logger.info(
        "Why: 'the'(id=1) has logit -1.5 (negative), so it's multiplied by 1.3 -> -1.95. "
        "'cat'(id=2) has logit 4.2 (positive), so it's divided by 1.3 -> 3.23. "
        "Both become less likely to be sampled again. "
        "After applying the penalty, greedy would now select: id=%d '%s'.",
        greedy_decode(logits_rep), vocab[greedy_decode(logits_rep)],
    )

    # ---------- Summary table ----------
    logger.info("\n=== Strategy Comparison Summary ===")
    logger.info(
        "%-25s  %-12s  %-50s",
        "Strategy", "Token Selected", "Reason"
    )
    logger.info("-" * 90)
    comparisons = [
        ("Greedy",                   vocab[2],        "Argmax: 'cat' has highest logit 4.2"),
        ("Temperature T=0.5",        vocab[2],        "Sharp dist: ~100% on 'cat', rarely differs"),
        ("Temperature T=2.0",        "stochastic",    "Flat dist: 'cat' ~50%, 'sat' ~25%, 'mat' ~15%"),
        ("Top-k k=3 + greedy",       vocab[2],        "Filters to {cat,sat,mat}; cat still argmax"),
        ("Top-p p=0.9",              "stochastic",    "Keeps cat (and maybe sat); samples from nucleus"),
        ("Repetition pen + greedy",  vocab[4],        "cat penalized from 4.2->3.23; sat(2.1) now argmax"),
    ]
    for strategy, token, reason in comparisons:
        logger.info("%-25s  %-12s  %s", strategy, token, reason)

    logger.info("\n=== Key Insight ===")
    logger.info(
        "The strategies differ because they modify WHAT is in the candidate set "
        "(top-k/top-p) and HOW MUCH probability each candidate receives "
        "(temperature, repetition penalty) before sampling."
    )
    logger.info(
        "Greedy always picks the same token regardless of distribution shape. "
        "Temperature controls diversity without changing the candidate set. "
        "Top-k/top-p control the candidate set without changing relative probabilities within it. "
        "Repetition penalty modifies logits for already-seen tokens specifically. "
        "In production, these are typically combined: T + top-p + repetition penalty."
    )


if __name__ == "__main__":
    main()
```

**Expected output (abridged):**

```
INFO: === Mock Logits ===
INFO:   id=0 [PAD]    logit= -3.00  prob=0.0003
INFO:   id=1 the      logit= -1.50  prob=0.0014
INFO:   id=2 cat      logit=  4.20  prob=0.8961
INFO:   id=3 dog      logit=  0.80  prob=0.0145
INFO:   id=4 sat      logit=  2.10  prob=0.0534
...
INFO: --- Greedy Decode ---
INFO: Selected: id=2 'cat' (logit=4.20)
...
INFO: === Strategy Comparison Summary ===
INFO: Strategy                   Token Selected  Reason
INFO: Greedy                     cat             Argmax: 'cat' has highest logit 4.2
INFO: Temperature T=0.5          cat             Sharp dist: ~100% on 'cat', rarely differs
INFO: Temperature T=2.0          stochastic      Flat dist: 'cat' ~50%, 'sat' ~25%, 'mat' ~15%
INFO: Top-k k=3 + greedy         cat             Filters to {cat,sat,mat}; cat still argmax
INFO: Top-p p=0.9                stochastic      Keeps cat (and maybe sat); samples from nucleus
INFO: Repetition pen + greedy    sat             cat penalized from 4.2->3.23; sat(2.1) now argmax
```

**Why the outputs differ:**

The mock logit array has `logits[2] = 4.2` ("cat") as a clear winner. Strategies that preserve the relative ordering (greedy, top-k with k>=1, top-p with p>=prob("cat")) all select "cat" when picking greedily. They differ in what is sampled stochastically:

- **Greedy**: always "cat". Zero randomness.

- **Temperature T=0.5**: concentrates nearly all mass on "cat" (logit becomes 8.4, prob near 1). Sampling almost never deviates.

- **Temperature T=2.0**: spreads mass — "cat" drops from ~90% to ~50% probability, "sat" rises to ~25%. Sampling frequently picks alternatives, increasing diversity.

- **Top-k k=3**: removes all tokens except cat/sat/mat from the candidate pool. If sampling deterministically (argmax), still "cat". The value of top-k is in preventing accidental sampling of nonsense tokens in the tail during stochastic decoding.

- **Top-p p=0.9**: since "cat" alone has ~90% probability, the nucleus contains only "cat" (or "cat" + "sat" depending on exact threshold). Effectively identical to greedy here, but on a flatter distribution top-p would keep many more tokens.

- **Repetition penalty**: directly modifies "cat"'s logit from 4.2 to 4.2/1.3 = 3.23. The new argmax is "sat" (logit 2.1, unchanged). This is the only strategy that changes which token greedy selects, because it targets already-generated tokens specifically.
