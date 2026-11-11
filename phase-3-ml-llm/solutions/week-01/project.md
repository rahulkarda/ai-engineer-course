# Solutions — Phase 3 · Week 01 · Project

## Complete `ml_math` Implementation

### `ml_math/ops.py`

```python
"""Vector and matrix operations."""
import logging
import numpy as np

log = logging.getLogger(__name__)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1-D vectors."""
    assert a.ndim == 1 and b.ndim == 1, "dot requires 1-D arrays"
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    return float(np.dot(a, b))


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication with shape assertion."""
    assert A.ndim == 2 and B.ndim == 2, "matmul requires 2-D arrays"
    assert A.shape[1] == B.shape[0], (
        f"Inner dimensions must match: {A.shape} @ {B.shape}"
    )
    return A @ B


def l2_norm(x: np.ndarray) -> float:
    """L2 (Euclidean) norm."""
    return float(np.linalg.norm(x))


def l1_norm(x: np.ndarray) -> float:
    """L1 (Manhattan) norm."""
    return float(np.linalg.norm(x, 1))


def normalize(x: np.ndarray) -> np.ndarray:
    """Return unit vector (L2-normalized)."""
    n = l2_norm(x)
    if n < 1e-10:
        raise ValueError("Cannot normalize near-zero vector")
    return x / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]."""
    denom = l2_norm(a) * l2_norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector projection of b onto a."""
    return (np.dot(a, b) / np.dot(a, a)) * a
```

### `ml_math/activations.py`

```python
"""Activation functions and their analytic gradients."""
import numpy as np

_CLIP: float = 500.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, numerically stable."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -_CLIP, _CLIP)))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: s(x)(1-s(x))."""
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Sub-gradient of ReLU (0 at x=0)."""
    return (x > 0.0).astype(float)


def tanh_act(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent."""
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh: 1 - tanh²(x)."""
    return 1.0 - np.tanh(x) ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    z_s = z - np.max(z, axis=-1, keepdims=True)
    e   = np.exp(z_s)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over last axis."""
    z_s = z - np.max(z, axis=-1, keepdims=True)
    return z_s - np.log(np.sum(np.exp(z_s), axis=-1, keepdims=True))
```

### `ml_math/losses.py`

```python
"""Loss functions and their gradients."""
import numpy as np


def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_pred - y_true) ** 2))


def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. y_pred."""
    return 2.0 * (y_pred - y_true) / y_pred.size


def binary_cross_entropy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Binary CE. probs in (0,1), y_true in {0,1}."""
    eps = 1e-9
    return float(-np.mean(
        y_true * np.log(probs + eps) + (1 - y_true) * np.log(1 - probs + eps)
    ))


def cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Multi-class CE. logits (N, C), y_true (N,) int indices."""
    n = logits.shape[0]
    z_s   = logits - logits.max(axis=1, keepdims=True)
    log_p = z_s - np.log(np.sum(np.exp(z_s), axis=1, keepdims=True))
    return float(-log_p[np.arange(n), y_true].mean())
```

### `ml_math/optimizer.py`

```python
"""Gradient descent optimizers."""
import logging
from collections.abc import Callable
import numpy as np

log = logging.getLogger(__name__)


def gradient_descent(
    loss_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    init_params: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[float]]:
    """Generic gradient descent. Returns (final_params, loss_history)."""
    params = init_params.copy()
    history: list[float] = []

    for step in range(n_steps):
        loss = loss_fn(params)
        history.append(loss)
        if step > 0 and abs(history[-2] - loss) < tol:
            log.info("Converged at step %d", step)
            break
        params -= lr * grad_fn(params)

    return params, history


def mini_batch_gd(
    X: np.ndarray,
    y: np.ndarray,
    loss_and_grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[float, np.ndarray]],
    init_params: np.ndarray,
    lr: float = 0.01,
    batch_size: int = 32,
    n_epochs: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """
    Mini-batch gradient descent.
    loss_and_grad_fn(X_batch, y_batch, params) -> (loss, grad)
    Returns (final_params, epoch_losses).
    """
    params = init_params.copy()
    rng    = np.random.default_rng(0)
    epoch_losses: list[float] = []

    for epoch in range(n_epochs):
        idx     = rng.permutation(len(X))
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, len(X), batch_size):
            batch = idx[start : start + batch_size]
            loss, grad = loss_and_grad_fn(X[batch], y[batch], params)
            params -= lr * grad
            ep_loss += loss
            n_batches += 1

        epoch_losses.append(ep_loss / max(n_batches, 1))

    return params, epoch_losses
```

### `ml_math/stats.py`

```python
"""Probability and statistics utilities."""
from collections.abc import Callable
import numpy as np


def empirical_mean(x: np.ndarray) -> float:
    """Sample mean."""
    return float(np.mean(x))


def empirical_variance(x: np.ndarray) -> float:
    """Sample variance (ddof=0)."""
    return float(np.var(x))


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference numerical gradient of scalar-valued f at x."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp, xm = x.copy().astype(float), x.copy().astype(float)
        xp.flat[i] += eps
        xm.flat[i] -= eps
        grad.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def gradient_check(
    analytic: np.ndarray,
    numeric: np.ndarray,
    tol: float = 1e-4,
) -> tuple[bool, float]:
    """Returns (passed, max_absolute_diff)."""
    max_diff = float(np.max(np.abs(analytic - numeric)))
    return max_diff < tol, max_diff
```

---

## `demo_train.py`

```python
"""demo_train.py — logistic regression on Iris (binary) using ml_math."""
import logging
import numpy as np
from sklearn.datasets import load_iris  # data loading only

from ml_math.activations import sigmoid
from ml_math.losses import binary_cross_entropy
from ml_math.optimizer import gradient_descent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

LEARNING_RATE: float = 0.1
N_STEPS: int = 1000


def make_iris_binary() -> tuple[np.ndarray, np.ndarray]:
    """Iris classes 0 vs 1, features 0 and 1 only."""
    iris = load_iris()
    mask = iris.target < 2
    X    = iris.data[mask, :2].astype(float)
    y    = iris.target[mask].astype(float)
    # Standardize
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Add bias column
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    return X, y


def main() -> None:
    X, y = make_iris_binary()
    w0   = np.zeros(X.shape[1])

    def loss_fn(w: np.ndarray) -> float:
        return binary_cross_entropy(sigmoid(X @ w), y)

    def grad_fn(w: np.ndarray) -> np.ndarray:
        probs = sigmoid(X @ w)
        return (1.0 / len(y)) * X.T @ (probs - y)

    w_final, history = gradient_descent(loss_fn, grad_fn, w0, lr=LEARNING_RATE, n_steps=N_STEPS)

    final_loss = history[-1]
    preds      = (sigmoid(X @ w_final) >= 0.5).astype(int)
    accuracy   = float(np.mean(preds == y.astype(int)))

    log.info("Initial loss : %.4f", history[0])
    log.info("Final loss   : %.4f", final_loss)
    log.info("Accuracy     : %.4f", accuracy)

    assert final_loss < 0.15, f"Loss too high: {final_loss:.4f}"
    assert accuracy   >= 0.95, f"Accuracy too low: {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

---

## Test Suite Sample

```python
# tests/test_ops.py
import numpy as np
import pytest
from ml_math.ops import dot, matmul, normalize, cosine_similarity


def test_dot_commutative():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert dot(a, b) == dot(b, a)


def test_matmul_shape():
    A = np.ones((3, 4))
    B = np.ones((4, 2))
    C = matmul(A, B)
    assert C.shape == (3, 2)


def test_matmul_inner_mismatch():
    with pytest.raises(AssertionError):
        matmul(np.ones((3, 4)), np.ones((5, 2)))


def test_normalize_unit():
    v = np.array([3.0, 4.0])
    u = normalize(v)
    assert abs(np.linalg.norm(u) - 1.0) < 1e-7


def test_cosine_self_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-7
```

```python
# tests/test_activations.py
import numpy as np
from ml_math.activations import sigmoid, relu, softmax, sigmoid_grad
from ml_math.stats import numerical_gradient


def test_sigmoid_range():
    z = np.linspace(-10, 10, 200)
    s = sigmoid(z)
    assert np.all(s > 0) and np.all(s < 1)


def test_relu_nonneg():
    z = np.array([-3.0, 0.0, 5.0])
    assert np.all(relu(z) >= 0)


def test_softmax_sums_to_one():
    z = np.array([1.0, 2.0, 3.0])
    assert abs(softmax(z).sum() - 1.0) < 1e-6


def test_sigmoid_grad_check():
    x = np.array([0.0, 1.5, -1.0])
    analytic = sigmoid_grad(x)
    numeric  = numerical_gradient(lambda v: sigmoid(v).sum(), x)
    assert np.max(np.abs(analytic - numeric)) < 1e-5
```
