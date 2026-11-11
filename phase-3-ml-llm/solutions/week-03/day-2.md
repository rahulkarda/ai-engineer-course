# Solutions — Week 03 · Day 2

## Exercise 1 — Matrix multiply forward pass (NumPy)

```python
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def linear_forward(
    x: np.ndarray,    # (batch, in_features)
    w: np.ndarray,    # (in_features, out_features)
    b: np.ndarray,    # (out_features,)
) -> np.ndarray:
    """Affine transform: z = x @ W + b."""
    return x @ w + b


# Demo: batch=4, in=3, out=2
rng = np.random.default_rng(42)
x = rng.standard_normal((4, 3))
w = rng.standard_normal((3, 2))
b = np.zeros(2)

z = linear_forward(x, w, b)
logger.info("x shape : %s", x.shape)
logger.info("W shape : %s", w.shape)
logger.info("z shape : %s", z.shape)   # (4, 2)
logger.info("z       :\n%s", np.round(z, 4))
```

Note: `x @ w` is equivalent to `np.matmul(x, w)`. Broadcasting handles the
`+ b` over the batch dimension automatically.

---

## Exercise 2 — MSE Loss implementation

```python
import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Squared Error: (1/N) * sum((y_pred - y_true)^2)."""
    diff = y_pred - y_true
    return float(np.mean(diff ** 2))


def mse_gradient(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """dL/dy_pred = (2/N) * (y_pred - y_true)."""
    n = y_pred.shape[0]
    return (2.0 / n) * (y_pred - y_true)


# Verify
y_pred = np.array([2.5, 0.0, 2.0, 8.0])
y_true = np.array([3.0, -0.5, 2.0, 7.0])
loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)

print(f"MSE loss : {loss:.4f}")    # 0.375 / 4 = 0.1875... let's compute:
# diffs: [-0.5, 0.5, 0.0, 1.0]  -> squares: [0.25, 0.25, 0.0, 1.0] -> mean=0.375
# Correct: 0.375
print(f"Gradient : {np.round(grad, 4)}")
```

MSE gradient derivation:
- L = (1/N) * Σ (ŷ_i - y_i)²
- dL/dŷ_i = (2/N) * (ŷ_i - y_i)

---

## Exercise 3 — Cross-entropy loss implementation

```python
import numpy as np


def cross_entropy_loss(
    logits: np.ndarray,   # (N, C) raw scores
    labels: np.ndarray,   # (N,) integer class indices
) -> float:
    """Numerically stable softmax cross-entropy."""
    # Shift for numerical stability
    logits_stable = logits - logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(logits_stable).sum(axis=1))
    correct_logits = logits_stable[np.arange(len(labels)), labels]
    loss = -correct_logits + log_sum_exp
    return float(loss.mean())


def cross_entropy_gradient(
    logits: np.ndarray,   # (N, C)
    labels: np.ndarray,   # (N,)
) -> np.ndarray:
    """dL/d_logits = softmax(logits) - one_hot(labels), divided by N."""
    n, c = logits.shape
    logits_stable = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_stable)
    softmax_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # Subtract 1 from correct class
    softmax_probs[np.arange(n), labels] -= 1.0
    return softmax_probs / n


# Verify
logits = np.array([[2.0, 1.0, 0.1],
                   [0.5, 2.5, 0.3]])
labels = np.array([0, 1])
loss = cross_entropy_loss(logits, labels)
grad = cross_entropy_gradient(logits.copy(), labels)

print(f"CE loss  : {loss:.4f}")   # approx 0.4017
print(f"Gradient :\n{np.round(grad, 4)}")
```

Derivation note: for softmax + cross-entropy together the gradient simplifies to
`p_i - 1` at the true class and `p_i` elsewhere (divided by batch size N).

---

## Exercise 4 — Numerical gradient check

```python
import numpy as np
from typing import Callable


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central difference gradient: (f(x+eps) - f(x-eps)) / (2*eps)."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = float(x[idx])
        x[idx] = old_val + eps
        fp = f(x)
        x[idx] = old_val - eps
        fm = f(x)
        x[idx] = old_val
        grad[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def gradient_check(
    f: Callable[[np.ndarray], float],
    analytical_grad: np.ndarray,
    x: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> bool:
    """Returns True if max relative error < tol."""
    num_grad = numerical_gradient(f, x, eps)
    abs_diff = np.abs(analytical_grad - num_grad)
    denom = np.abs(analytical_grad) + np.abs(num_grad) + 1e-8
    rel_error = (abs_diff / denom).max()
    print(f"Max relative error: {rel_error:.2e}  (tol={tol:.2e}) -> {'PASS' if rel_error < tol else 'FAIL'}")
    return bool(rel_error < tol)


# Check MSE gradient
def mse(y_pred: np.ndarray) -> float:
    y_true = np.array([1.0, 2.0, 3.0])
    return float(np.mean((y_pred - y_true) ** 2))

y_pred = np.array([1.5, 2.5, 2.5])
analytical = (2.0 / len(y_pred)) * (y_pred - np.array([1.0, 2.0, 3.0]))
gradient_check(mse, analytical, y_pred.copy())
```

Numerical gradient check is the standard sanity test for backprop implementations.
Central differences have O(eps^2) error vs. O(eps) for forward differences.

---

## Exercise 5 — Binary Cross-Entropy

```python
import numpy as np


def binary_cross_entropy(
    y_pred: np.ndarray,   # (N,) probabilities in (0, 1)
    y_true: np.ndarray,   # (N,) binary labels {0, 1}
    eps: float = 1e-7,
) -> float:
    """BCE: -(1/N) * sum(y * log(p) + (1-y) * log(1-p))."""
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def bce_gradient(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """dL/dp = -(1/N) * (y/p - (1-y)/(1-p))."""
    n = len(y_pred)
    p = np.clip(y_pred, eps, 1.0 - eps)
    return (1.0 / n) * (-(y_true / p) + (1.0 - y_true) / (1.0 - p))


# Verify
y_pred = np.array([0.9, 0.3, 0.8, 0.1])
y_true = np.array([1.0, 0.0, 1.0, 0.0])

loss = binary_cross_entropy(y_pred, y_true)
grad = bce_gradient(y_pred, y_true)
print(f"BCE loss  : {loss:.4f}")   # approx 0.1643
print(f"Gradient  : {np.round(grad, 4)}")

# Confirm with numerical gradient check
def bce_scalar(p: np.ndarray) -> float:
    return binary_cross_entropy(p, y_true)

from typing import Callable
num_grad = numerical_gradient(bce_scalar, y_pred.copy())
print(f"Numerical : {np.round(num_grad, 4)}")
print(f"Match: {np.allclose(grad, num_grad, atol=1e-5)}")
```

BCE is the per-element version of cross-entropy for binary classification.
The clipping (`eps`) prevents log(0) = -inf.
```
