# Solutions — Week 03 · Day 3

## Exercise 1 — Chain Rule derivation (written out)

Given a composed function L = f(g(h(x))):

    dL/dx = (dL/df) * (df/dg) * (dg/dh) * (dh/dx)

In a neural network with layers z1 = W1·x + b1, a1 = ReLU(z1),
z2 = W2·a1 + b2, L = MSE(z2, y):

    dL/dW2 = dL/dz2 · dz2/dW2
           = (2/N)(z2 - y) · a1^T

    dL/da1 = W2^T · dL/dz2

    dL/dz1 = dL/da1 * ReLU'(z1)          (element-wise product)

    dL/dW1 = dL/dz1 · x^T

    dL/db2 = sum(dL/dz2, axis=batch)
    dL/db1 = sum(dL/dz1, axis=batch)

The chain rule flows gradients *backward* through each operation, which is
why the algorithm is called backpropagation.

---

## Exercise 2 — Full gradient derivation for 2-layer MLP

Network:
- Input:   x ∈ R^{n_in}
- Layer 1: z1 = W1 x + b1, a1 = ReLU(z1),  W1 ∈ R^{n_h × n_in}
- Layer 2: z2 = W2 a1 + b2, output a2 = z2,  W2 ∈ R^{n_out × n_h}
- Loss:    L = (1/N) ||a2 - y||^2  (MSE, regression)

Forward pass (single sample, column vectors):

    z1 = W1 x + b1
    a1 = max(0, z1)           element-wise
    z2 = W2 a1 + b2
    L  = (1/2)||z2 - y||^2   (using 1/2 for clean derivative)

Backward pass:

    delta2 = dL/dz2 = (z2 - y)               shape (n_out,)

    dL/dW2 = delta2 · a1^T                   shape (n_out, n_h)
    dL/db2 = delta2                           shape (n_out,)

    dL/da1 = W2^T · delta2                   shape (n_h,)

    delta1 = dL/dz1 = dL/da1 ⊙ 1[z1 > 0]   shape (n_h,)   (⊙ = element-wise)

    dL/dW1 = delta1 · x^T                    shape (n_h, n_in)
    dL/db1 = delta1                           shape (n_h,)

For batch size N, replace column vectors with matrices and divide weight/bias
gradients by N (mean over batch).

---

## Exercise 3 — NumPy backprop for XOR

```python
import logging
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


@dataclass
class MLPParams:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def init_params(
    n_in: int,
    n_hidden: int,
    n_out: int,
    seed: int = 0,
) -> MLPParams:
    rng = np.random.default_rng(seed)
    scale = 0.5
    return MLPParams(
        W1=rng.standard_normal((n_hidden, n_in)) * scale,
        b1=np.zeros(n_hidden),
        W2=rng.standard_normal((n_out, n_hidden)) * scale,
        b2=np.zeros(n_out),
    )


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def forward(
    x: np.ndarray,      # (N, n_in)
    p: MLPParams,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Returns output probabilities and cache for backprop."""
    z1 = x @ p.W1.T + p.b1          # (N, n_hidden)
    a1 = relu(z1)                    # (N, n_hidden)
    z2 = a1 @ p.W2.T + p.b2         # (N, n_out)
    a2 = sigmoid(z2)                 # (N, n_out)
    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
    return a2, cache


def bce_loss(a2: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(a2, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def backward(
    y: np.ndarray,
    cache: dict[str, np.ndarray],
    p: MLPParams,
) -> MLPParams:
    """Returns gradients as an MLPParams dataclass."""
    n = y.shape[0]
    a2, a1, z1, x = cache["a2"], cache["a1"], cache["z1"], cache["x"]

    # Output layer: BCE + sigmoid gradient simplifies to (a2 - y) / n
    delta2 = (a2 - y) / n                   # (N, n_out)

    dW2 = delta2.T @ a1                     # (n_out, n_hidden)
    db2 = delta2.sum(axis=0)                # (n_out,)

    # Hidden layer
    da1  = delta2 @ p.W2                    # (N, n_hidden)
    delta1 = da1 * (z1 > 0).astype(float)  # ReLU derivative

    dW1 = delta1.T @ x                      # (n_hidden, n_in)
    db1 = delta1.sum(axis=0)               # (n_hidden,)

    return MLPParams(W1=dW1, b1=db1, W2=dW2, b2=db2)


def update(p: MLPParams, grads: MLPParams, lr: float) -> MLPParams:
    return MLPParams(
        W1=p.W1 - lr * grads.W1,
        b1=p.b1 - lr * grads.b1,
        W2=p.W2 - lr * grads.W2,
        b2=p.b2 - lr * grads.b2,
    )


def train_xor(n_epochs: int = 5000, lr: float = 0.1, seed: int = 0) -> MLPParams:
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    p = init_params(n_in=2, n_hidden=4, n_out=1, seed=seed)

    for epoch in range(n_epochs):
        a2, cache = forward(X, p)
        loss = bce_loss(a2, y)
        grads = backward(y, cache, p)
        p = update(p, grads, lr)
        if epoch % 1000 == 0:
            preds = (a2 > 0.5).astype(int)
            acc = float((preds == y).mean())
            logger.info("epoch=%5d  loss=%.4f  acc=%.2f", epoch, loss, acc)

    return p


if __name__ == "__main__":
    p = train_xor()
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    a2, _ = forward(X, p)
    logger.info("Final predictions: %s", np.round(a2, 3))
    logger.info("Targets          : %s", y.T)
```

Expected: loss < 0.02, accuracy = 100% by epoch 5000.

---

## Exercise 4 — Gradient check test

```python
import numpy as np
import pytest
from typing import Callable


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps
        fp = f(x)
        x[idx] = orig - eps
        fm = f(x)
        x[idx] = orig
        grad[idx] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8)))


def test_mlp_gradient_check() -> None:
    """Verify analytical gradients match numerical gradients for a mini MLP."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((3, 2))
    y = np.array([[1.0], [0.0], [1.0]])

    from day3_solution import init_params, forward, backward, bce_loss, MLPParams

    p = init_params(2, 4, 1, seed=7)
    a2, cache = forward(X, p)
    grads = backward(y, cache, p)

    # Check dW2
    def loss_W2(w2_flat: np.ndarray) -> float:
        p_test = MLPParams(p.W1, p.b1, w2_flat.reshape(p.W2.shape), p.b2)
        a, _ = forward(X, p_test)
        return bce_loss(a, y)

    num_dW2 = numerical_gradient(loss_W2, p.W2.ravel().copy())
    err = rel_error(grads.W2.ravel(), num_dW2)
    assert err < 1e-5, f"dW2 relative error too large: {err:.2e}"

    # Check dW1
    def loss_W1(w1_flat: np.ndarray) -> float:
        p_test = MLPParams(w1_flat.reshape(p.W1.shape), p.b1, p.W2, p.b2)
        a, _ = forward(X, p_test)
        return bce_loss(a, y)

    num_dW1 = numerical_gradient(loss_W1, p.W1.ravel().copy())
    err = rel_error(grads.W1.ravel(), num_dW1)
    assert err < 1e-5, f"dW1 relative error too large: {err:.2e}"


if __name__ == "__main__":
    test_mlp_gradient_check()
    print("Gradient check PASSED")
```

---

## Exercise 5 — Learning rate experiment

```python
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Reuse train_xor from Exercise 3 but return loss history
def train_xor_history(
    lr: float,
    n_epochs: int = 3000,
    seed: int = 0,
) -> list[float]:
    from day3_ex3 import init_params, forward, backward, update, bce_loss
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    p = init_params(2, 4, 1, seed=seed)
    history: list[float] = []
    for _ in range(n_epochs):
        a2, cache = forward(X, p)
        history.append(bce_loss(a2, y))
        grads = backward(y, cache, p)
        p = update(p, grads, lr)
    return history


learning_rates = [0.001, 0.01, 0.1, 1.0, 5.0]
results: dict[float, list[float]] = {}

for lr in learning_rates:
    history = train_xor_history(lr)
    final_loss = history[-1]
    results[lr] = history
    logger.info("lr=%.3f  final_loss=%.4f", lr, final_loss)

# Observations:
# lr=0.001 -> slow convergence, loss still high at 3000 epochs
# lr=0.01  -> good convergence
# lr=0.1   -> fast convergence, best
# lr=1.0   -> may oscillate but converges
# lr=5.0   -> likely diverges (loss NaN or very high)

# Key insight: too small = slow; too large = divergence.
# Optimal lr for this problem ≈ 0.1-0.5 with vanilla SGD.
```

Takeaways:
- Learning rate is the most critical hyperparameter for training stability.
- Too small: gradients are correct but tiny updates — slow convergence.
- Too large: overshoots minima, loss oscillates or diverges.
- Adaptive optimizers (Adam, RMSProp) mitigate sensitivity to lr choice.
```
