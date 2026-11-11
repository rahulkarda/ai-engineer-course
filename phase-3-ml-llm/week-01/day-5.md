# Phase 3 · Week 01 · Day 5 — Probability, Distributions & Cross-Entropy

> Objective: Understand the probability machinery behind ML models — distributions, expectation, Bayes, maximum likelihood, and cross-entropy loss.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]], [[day-4]]

---

## Concept

### Why Probability in ML?

ML models output **distributions**, not point predictions. A classifier outputs `P(class | input)`. A language model outputs `P(next_token | context)`. To train these models, we need a loss function that measures how bad the predicted distribution is — that's **cross-entropy**.

### Key Distributions

| Distribution | NumPy | Use in ML |
|---|---|---|
| Bernoulli(p) | `rng.binomial(1, p)` | Binary classification labels |
| Categorical | `rng.multinomial(1, probs)` | Multi-class labels, token sampling |
| Gaussian N(μ,σ²) | `rng.normal(μ, σ)` | Weight init, noise, continuous predictions |
| Uniform U(a,b) | `rng.uniform(a, b)` | Weight init (Xavier), random sampling |

### Expectation & Variance

For a random variable X:  
- **Expected value:** `E[X] = Σ xᵢ P(xᵢ)` (discrete) or `∫ x p(x) dx` (continuous)  
- **Variance:** `Var(X) = E[(X - μ)²] = E[X²] - E[X]²`

In ML: loss = expectation of per-sample losses over the data distribution.

### Bayes' Theorem

```
P(A | B) = P(B | A) * P(A) / P(B)
```

- `P(A)` — **prior**: what we believe before seeing data
- `P(B | A)` — **likelihood**: probability of data given hypothesis
- `P(A | B)` — **posterior**: updated belief after seeing data

In ML: Bayesian inference, Naive Bayes classifier, priors as regularization.

### Maximum Likelihood Estimation (MLE)

Given data `D = {x₁, ..., xₙ}` and model parameters `θ`, choose `θ` to maximize:
```
L(θ) = Π P(xᵢ | θ)   →   log L(θ) = Σ log P(xᵢ | θ)
```

Maximizing log-likelihood is equivalent to minimizing **negative log-likelihood** (NLL). That's exactly what cross-entropy loss computes.

### Cross-Entropy

For true distribution `p` and predicted distribution `q`:  
`H(p, q) = -Σ p(x) log q(x)`

For a single example with one-hot true label `y` and softmax predictions `ŷ`:  
`CE = -log ŷ[y_true_class]`  — just the log of the predicted probability for the correct class.

Minimizing cross-entropy = maximizing log-likelihood = MLE.

**Softmax:** converts raw logits `z` to a probability distribution:  
`softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`

---

## How It Works Internally

`log_softmax` is numerically preferred over `log(softmax(z))`: it subtracts the max logit before exponentiating to avoid overflow (`z - max(z)`). PyTorch's `F.cross_entropy` internally calls `log_softmax` then `nll_loss` — this is why you pass raw logits, not softmax outputs, to PyTorch's CE loss.

---

## Worked Example

```python
"""
day5_probability_cross_entropy.py
Probability fundamentals and cross-entropy loss.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def demo_distributions() -> None:
    rng = np.random.default_rng(42)

    # Gaussian: mean=0, std=1 — standard normal
    samples: np.ndarray = rng.normal(0.0, 1.0, size=10_000)
    log.info("Gaussian  mean=%.4f  std=%.4f", samples.mean(), samples.std())

    # Bernoulli p=0.3
    coins: np.ndarray = rng.binomial(1, 0.3, size=10_000)
    log.info("Bernoulli p_est=%.4f (true=0.3)", coins.mean())


def demo_expectation_variance() -> None:
    rng = np.random.default_rng(0)
    x: np.ndarray = rng.normal(3.0, 2.0, size=100_000)

    mean: float    = float(np.mean(x))
    var: float     = float(np.var(x))
    std: float     = float(np.std(x))

    log.info("E[X] = %.4f  (true 3.0)", mean)
    log.info("Var  = %.4f  (true 4.0)", var)
    log.info("Std  = %.4f  (true 2.0)", std)


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_shifted = z - np.max(z)
    exp_z     = np.exp(z_shifted)
    return exp_z / exp_z.sum()


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax."""
    z_shifted = z - np.max(z)
    return z_shifted - np.log(np.sum(np.exp(z_shifted)))


def cross_entropy_loss(logits: np.ndarray, y_true: int) -> float:
    """
    Cross-entropy loss for a single example.

    logits : raw model outputs (before softmax)
    y_true : integer class index
    """
    log_probs = log_softmax(logits)
    return float(-log_probs[y_true])


def batched_cross_entropy(
    logits: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """
    Mean cross-entropy over a batch.

    logits : (B, C) — one row per example, one col per class
    y_true : (B,) integer class indices
    """
    batch_size = logits.shape[0]
    total_loss = 0.0
    for i in range(batch_size):
        total_loss += cross_entropy_loss(logits[i], int(y_true[i]))
    return total_loss / batch_size


def demo_cross_entropy() -> None:
    # 3 examples, 4 classes
    rng = np.random.default_rng(9)
    logits: np.ndarray  = rng.standard_normal((3, 4))
    y_true: np.ndarray  = np.array([2, 0, 3])

    probs = np.apply_along_axis(softmax, 1, logits)
    loss  = batched_cross_entropy(logits, y_true)

    log.info("Logits:\n%s", np.round(logits, 3))
    log.info("Probs :\n%s", np.round(probs,  3))
    log.info("True labels: %s", y_true)
    log.info("CE Loss: %.4f", loss)

    # Perfect predictions should give loss → 0
    perfect_logits: np.ndarray = np.array([
        [-10.0, -10.0, 100.0, -10.0],  # class 2
        [100.0, -10.0, -10.0, -10.0],  # class 0
        [-10.0, -10.0, -10.0, 100.0],  # class 3
    ])
    log.info("Perfect CE Loss: %.4f", batched_cross_entropy(perfect_logits, y_true))


def demo_bayes() -> None:
    """Naive Bayes email spam filter — pure NumPy."""
    # P(word | spam), P(word | ham) for 3 words
    # words: "discount", "meeting", "urgent"
    p_word_given_spam = np.array([0.8, 0.1, 0.6])
    p_word_given_ham  = np.array([0.05, 0.7, 0.1])
    p_spam = 0.2  # 20% of emails are spam (prior)
    p_ham  = 0.8

    # Email contains words 0 and 2 (not word 1)
    word_present = np.array([1, 0, 1])
    word_absent  = 1 - word_present

    # Log-space to avoid underflow
    log_p_email_spam = (
        np.log(p_spam)
        + np.sum(word_present * np.log(p_word_given_spam + 1e-9))
        + np.sum(word_absent  * np.log(1 - p_word_given_spam + 1e-9))
    )
    log_p_email_ham = (
        np.log(p_ham)
        + np.sum(word_present * np.log(p_word_given_ham + 1e-9))
        + np.sum(word_absent  * np.log(1 - p_word_given_ham + 1e-9))
    )

    # Posterior (unnormalized)
    log.info("log P(email|spam)*P(spam) = %.4f", log_p_email_spam)
    log.info("log P(email|ham)*P(ham)   = %.4f", log_p_email_ham)
    log.info("Predicted: %s", "SPAM" if log_p_email_spam > log_p_email_ham else "HAM")


if __name__ == "__main__":
    demo_distributions()
    demo_expectation_variance()
    demo_cross_entropy()
    demo_bayes()
```

---

## Exercises

1. Implement `kl_divergence(p, q)` — KL divergence between two probability vectors: `KL(p||q) = Σ p(x) log(p(x)/q(x))`. Verify that `KL(p||p) = 0`.
2. A classifier outputs logits `[2.0, 0.5, -1.0]` for classes [cat, dog, fish]. Compute the softmax probabilities by hand (show the steps), then verify with your `softmax` function.
3. Show that **cross-entropy is asymmetric**: compute `H(p, q)` and `H(q, p)` for `p = [0.8, 0.2]` and `q = [0.5, 0.5]`. Why does this asymmetry matter for training?
4. Implement `accuracy(logits, y_true)` — fraction of correctly predicted classes. Compare it to cross-entropy loss as training progresses on random logits. Do they always move together?
5. The cross-entropy gradient w.r.t. logits is `softmax(z) - one_hot(y)`. Verify this numerically using `numerical_gradient` from Day 3 and the `cross_entropy_loss` function.

(Solutions: ../solutions/week-01/day-5.md)

---

## Self-check

- [ ] I can write the cross-entropy formula from memory and explain what it measures
- [ ] I can implement numerically stable softmax and explain why stability matters
- [ ] I can explain why minimizing CE is equivalent to MLE
- [ ] I know the CE gradient formula and where it comes from

---

## Supplementary

- [Visual information theory — Chris Olah](https://colah.github.io/posts/2015-09-Visual-Information/)
- [Probability for ML (fast.ai)](https://www.fast.ai/posts/part2-2023-chapter9-foundations.html)
- [CS229 Notes on probability (Stanford)](https://cs229.stanford.edu/notes2022fall/cs229-prob_review.pdf)
