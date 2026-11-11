# Phase 3 · Week 03 · Day 2 — Forward Pass Mechanics and Loss Functions

> Objective: Derive MSE, cross-entropy, and binary cross-entropy from first principles, then implement forward pass + loss in NumPy.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-1]]

---

## Concept

The **forward pass** is the sequence of matrix multiplications and activations that transforms raw input `x` into a prediction `y_hat`. Once we have a prediction, a **loss function** measures how wrong it is — producing a single scalar that training will minimize.

Loss functions are not arbitrary. Each one is derived from a probabilistic assumption about the target distribution. Choosing the wrong loss gives a network the wrong objective.

### Mean Squared Error (MSE)

Assumption: targets are drawn from a Gaussian distribution around the prediction.

Derivation via maximum likelihood:

```
P(y | x, theta) = N(y; f_theta(x), sigma^2)
log P = -1/(2*sigma^2) * (y - y_hat)^2 + const
```

Maximizing log-likelihood is equivalent to minimizing:

```
MSE = (1/N) * sum_i (y_i - y_hat_i)^2
```

Use MSE for **regression** tasks where the output is a real-valued scalar.

**Gradient of MSE** w.r.t. `y_hat`:

```
dL/dy_hat = (2/N) * (y_hat - y)
```

### Cross-Entropy Loss (Categorical)

Assumption: targets are one-hot vectors drawn from a categorical distribution; network outputs a softmax probability vector.

Derivation via negative log-likelihood:

```
P(y | x) = prod_k p_k^{y_k}      (y is one-hot, so only the true class term survives)
-log P = -sum_k y_k * log(p_k) = -log(p_{true_class})
```

Over a batch of `N` examples:

```
CE = -(1/N) * sum_i sum_k y_{ik} * log(p_{ik})
```

Use CE for **multiclass classification**. Works with softmax output.

**Gradient of CE w.r.t. softmax input (logits)** has a beautiful closed form:

```
dL/dz_k = p_k - y_k
```

This is why cross-entropy + softmax is the standard combination — gradients are clean.

### Binary Cross-Entropy (BCE)

Special case of CE for two classes. Assumption: output is a Bernoulli probability from a sigmoid.

```
BCE = -(1/N) * sum_i [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

Use BCE for **binary classification** (single sigmoid output) or **multi-label classification** (independent sigmoid per label).

**Numerical stability**: `log(p)` with `p` near 0 diverges. Clip predictions: `p = clip(p, 1e-7, 1-1e-7)`.

---

## How it works internally

### Matrix multiplication as batched dot products

For batch size `B`, input dim `n_in`, output dim `n_out`:

```
X: (B, n_in)
W: (n_out, n_in)
Z = X @ W.T : (B, n_out)
```

Each element `Z[b, j] = dot(X[b, :], W[j, :])` — the activation of neuron `j` for sample `b`.

NumPy computes this via BLAS `dgemm`. The key insight: processing a full batch as a matrix operation is orders of magnitude faster than looping over samples. This is the reason mini-batch gradient descent is both a computational and statistical win.

### Why loss is a scalar

The optimizer needs a single number to differentiate. When computing over a batch, we average (or sum) per-sample losses into one scalar. The gradient of this scalar w.r.t. every parameter is what backprop computes tomorrow.

### Log-sum-exp trick for numerical CE

```python
# Unstable:
loss = -np.log(softmax(z)[true_class])

# Stable — fuse softmax + log + CE into one operation:
# log(softmax(z)_k) = z_k - log(sum(exp(z)))
#                   = z_k - logsumexp(z)
```

Using `scipy.special.logsumexp` or the manual equivalent avoids intermediate overflow when computing CE directly from logits.

---

## Worked example

```python
"""
day_2_forward_loss.py — Forward pass + all three loss functions in NumPy.
Derives losses from logits, no ML framework.
"""

import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---- Activations (reused from day-1) ----

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


# ---- Linear layer ----

def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z = x @ W.T + b."""
    return x @ W.T + b


# ---- Loss functions ----

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error for regression.

    Args:
        y_true: Ground-truth values, shape (batch,) or (batch, out).
        y_pred: Predictions, same shape.

    Returns:
        Scalar loss.
    """
    diff = y_pred - y_true
    return float(np.mean(diff ** 2))


def cross_entropy_loss(
    y_true_onehot: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Categorical cross-entropy from raw logits.

    Numerically stable: uses log-softmax trick.

    Args:
        y_true_onehot: One-hot targets, shape (batch, n_classes).
        logits: Pre-softmax activations, shape (batch, n_classes).
        eps: Clipping floor to avoid log(0).

    Returns:
        Scalar loss.
    """
    # log-softmax: log(p_k) = z_k - log(sum(exp(z)))
    log_probs = logits - np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(
        axis=1, keepdims=True
    ) + eps) - logits.max(axis=1, keepdims=True)

    # Only the true-class log-prob survives the sum
    per_sample = -np.sum(y_true_onehot * log_probs, axis=1)  # (batch,)
    return float(np.mean(per_sample))


def binary_cross_entropy_loss(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Binary cross-entropy from sigmoid probabilities.

    Args:
        y_true: Binary labels {0, 1}, shape (batch,).
        y_pred_prob: Sigmoid output in (0, 1), shape (batch,).
        eps: Clipping for numerical stability.

    Returns:
        Scalar loss.
    """
    p = np.clip(y_pred_prob, eps, 1.0 - eps)
    per_sample = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return float(np.mean(per_sample))


# ---- Gradient of MSE w.r.t. y_pred ----

def mse_grad(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """dMSE/dy_pred = 2/N * (y_pred - y_true)."""
    n = y_true.shape[0]
    return (2.0 / n) * (y_pred - y_true)


# ---- Gradient of CE w.r.t. logits (fused softmax + CE) ----

def cross_entropy_grad_logits(
    y_true_onehot: np.ndarray,
    logits: np.ndarray,
) -> np.ndarray:
    """dCE/dz = (softmax(z) - y_true) / batch.

    This is the elegant closed-form gradient for softmax + CE.
    """
    probs = softmax(logits)
    return (probs - y_true_onehot) / logits.shape[0]


# ---- Full 2-layer MLP forward + loss ----

def forward_with_loss(
    x: np.ndarray,
    y_true_onehot: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Forward pass + cross-entropy loss.

    Returns a cache dict with all intermediate values needed for backprop.
    """
    z1 = linear(x, W1, b1)
    a1 = relu(z1)

    z2 = linear(a1, W2, b2)
    # Do NOT apply softmax here — cross_entropy_loss takes logits directly
    loss = cross_entropy_loss(y_true_onehot, z2)
    probs = softmax(z2)

    return {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "probs": probs,
        "loss": loss,
    }


def main() -> None:
    rng = np.random.default_rng(seed=0)

    # ---- Regression demo: MSE ----
    log.info("=== MSE Loss Demo ===")
    y_true_reg = np.array([1.0, 2.0, 3.0])
    y_pred_perfect = np.array([1.0, 2.0, 3.0])
    y_pred_off = np.array([1.5, 2.5, 2.0])

    log.info("MSE (perfect prediction): %.6f", mse_loss(y_true_reg, y_pred_perfect))
    log.info("MSE (imperfect prediction): %.6f", mse_loss(y_true_reg, y_pred_off))
    log.info("MSE gradient: %s", mse_grad(y_true_reg, y_pred_off))

    # ---- Multiclass demo: CE ----
    log.info("=== Cross-Entropy Loss Demo ===")
    batch, n_classes = 4, 3

    # Toy network weights
    n_in, n_hidden = 5, 6
    W1 = rng.standard_normal((n_hidden, n_in)) * 0.1
    b1 = np.zeros(n_hidden)
    W2 = rng.standard_normal((n_classes, n_hidden)) * 0.1
    b2 = np.zeros(n_classes)

    x = rng.standard_normal((batch, n_in))
    labels = rng.integers(0, n_classes, size=batch)

    # One-hot encode labels
    y_onehot = np.zeros((batch, n_classes))
    y_onehot[np.arange(batch), labels] = 1.0
    log.info("Labels: %s", labels)
    log.info("One-hot targets:\n%s", y_onehot)

    cache = forward_with_loss(x, y_onehot, W1, b1, W2, b2)
    log.info("Predicted probabilities:\n%s", np.round(cache["probs"], 4))
    log.info("Cross-entropy loss: %.6f", cache["loss"])

    grad_z2 = cross_entropy_grad_logits(y_onehot, cache["z2"])
    log.info("Gradient of loss w.r.t. logits z2:\n%s", np.round(grad_z2, 6))

    # ---- Binary classification demo: BCE ----
    log.info("=== Binary Cross-Entropy Loss Demo ===")
    y_binary = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    sigmoid_logits = np.array([2.0, -1.5, 0.5, -0.1, -3.0])
    probs_binary = sigmoid(sigmoid_logits)
    bce = binary_cross_entropy_loss(y_binary, probs_binary)
    log.info("Sigmoid probs: %s", np.round(probs_binary, 4))
    log.info("BCE loss: %.6f", bce)

    # Sanity check: loss should decrease as prediction improves
    bce_better = binary_cross_entropy_loss(
        y_binary,
        np.array([0.99, 0.01, 0.99, 0.99, 0.01]),  # near-perfect predictions
    )
    log.info("BCE (near-perfect): %.6f  (should be << %.6f)", bce_better, bce)


if __name__ == "__main__":
    main()
```

---

## Exercises

1. **Conceptual**: MSE penalizes large errors quadratically. What does this mean for outliers during training? Name one loss function that is more robust to outliers and explain why.

2. **Coding**: Implement `log_softmax` from scratch (without calling `softmax` then `log`). Use the log-sum-exp trick: `log_softmax(z)_k = z_k - log(sum(exp(z - max(z)))) - max(z)`. Verify it matches `np.log(softmax(z))` for a sample input.

3. **Derivation**: Derive the gradient `dBCE/dp` (w.r.t. sigmoid probability `p`, not logit `z`). Then use the chain rule to get `dBCE/dz` (w.r.t. the sigmoid input). Notice the clean form.

4. **Coding**: Implement `accuracy(logits, y_true_labels)` — returns the fraction of correctly classified samples (take argmax of logits as predicted class). Run it on the multiclass demo outputs.

5. **Experiment**: Increase batch size from 4 to 64 with random data. Log the CE loss over 10 random weight initializations. Observe the variance in loss across initializations. What does this suggest about weight initialization?

(Solutions: ../solutions/week-03/day-2.md)

---

## Self-check

- [ ] I can state the probabilistic assumption behind each loss function (MSE, CE, BCE).
- [ ] I can implement all three loss functions in NumPy from memory.
- [ ] I can explain why `dCE/dz = softmax(z) - y_true` and reproduce this derivation.
- [ ] I can explain why cross-entropy should be computed from logits, not post-softmax probabilities, for numerical stability.
- [ ] I can one-hot encode integer labels and verify shapes match before computing loss.

---

## Supplementary Links

- Bishop "Pattern Recognition and Machine Learning" Ch. 4-5 (loss derivations): https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf
- Log-sum-exp trick explained: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
- CE + softmax gradient derivation: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
- NumPy einsum for batched operations: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
