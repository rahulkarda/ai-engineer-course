# Phase 3 · Week 01 · Project — NumPy Math Engine

> Build a small, well-tested NumPy math engine covering every concept from this week, then use it to train a model from scratch.  ·  Time: ~3 hr  ·  Due: Day 7

---

## Goal

Write a self-contained Python package (`ml_math/`) that implements the core linear algebra, calculus, and probability primitives needed for ML — without using any high-level ML library. Then demonstrate it by training a logistic regression model on real data.

This is not a toy exercise. The code you write here will be reference material you return to throughout the course.

---

## Package Structure

```
ml_math/
├── __init__.py
├── ops.py          # vector/matrix operations
├── activations.py  # activation functions + gradients
├── losses.py       # MSE, binary CE, multi-class CE
├── optimizer.py    # gradient descent + mini-batch GD
└── stats.py        # probability utilities

tests/
├── test_ops.py
├── test_activations.py
├── test_losses.py
├── test_optimizer.py
└── test_stats.py

demo_train.py       # end-to-end training demo
```

---

## Module Specs

### `ops.py`

```python
import numpy as np

def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two 1-D vectors."""
    ...

def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication with shape assertion."""
    ...

def l2_norm(x: np.ndarray) -> float:
    """L2 (Euclidean) norm."""
    ...

def l1_norm(x: np.ndarray) -> float:
    """L1 (Manhattan) norm."""
    ...

def normalize(x: np.ndarray) -> np.ndarray:
    """Return unit vector (L2-normalized)."""
    ...

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, range [-1, 1]."""
    ...

def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector projection of b onto a."""
    ...
```

### `activations.py`

```python
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray: ...
def sigmoid_grad(x: np.ndarray) -> np.ndarray: ...

def relu(x: np.ndarray) -> np.ndarray: ...
def relu_grad(x: np.ndarray) -> np.ndarray: ...

def tanh_act(x: np.ndarray) -> np.ndarray: ...
def tanh_grad(x: np.ndarray) -> np.ndarray: ...

def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    ...

def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over last axis."""
    ...
```

### `losses.py`

```python
import numpy as np

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error."""
    ...

def mse_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. y_pred."""
    ...

def binary_cross_entropy(
    probs: np.ndarray, y_true: np.ndarray
) -> float:
    """Binary CE loss. probs in (0,1), y_true in {0,1}."""
    ...

def cross_entropy(
    logits: np.ndarray, y_true: np.ndarray
) -> float:
    """Multi-class CE. logits shape (N, C), y_true shape (N,) int."""
    ...
```

### `optimizer.py`

```python
import numpy as np
from collections.abc import Callable

def gradient_descent(
    loss_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    init_params: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 1000,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[float]]:
    """Generic gradient descent. Returns (final_params, loss_history)."""
    ...

def mini_batch_gd(
    X: np.ndarray,
    y: np.ndarray,
    loss_and_grad_fn: Callable,
    init_params: np.ndarray,
    lr: float = 0.01,
    batch_size: int = 32,
    n_epochs: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """Mini-batch gradient descent. Returns (final_params, epoch_losses)."""
    ...
```

### `stats.py`

```python
import numpy as np

def empirical_mean(x: np.ndarray) -> float: ...
def empirical_variance(x: np.ndarray) -> float: ...

def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference numerical gradient."""
    ...

def gradient_check(
    analytic: np.ndarray,
    numeric: np.ndarray,
    tol: float = 1e-4,
) -> tuple[bool, float]:
    """Returns (passed, max_absolute_diff)."""
    ...
```

---

## Demo Script: `demo_train.py`

Train logistic regression using your engine on the **Iris dataset** (binary: class 0 vs. class 1, features 0–1 only):

```python
"""demo_train.py — end-to-end logistic regression with ml_math."""
import logging
import numpy as np
# Dataset loading only — no ML models from sklearn
from sklearn.datasets import load_iris  # data only

from ml_math.activations import sigmoid
from ml_math.losses import binary_cross_entropy
from ml_math.optimizer import gradient_descent

# ... implement full training loop here
```

Expected output:
```
Initial loss : ~0.69 (random chance for binary)
Final loss   : < 0.15
Accuracy     : ≥ 95% on Iris binary (it's linearly separable)
```

---

## Test Requirements

Minimum 10 tests across all test files. Each module must have at least 2 tests.

Suggested tests:
- `test_ops.py`: `test_dot_commutative`, `test_matmul_shape`, `test_normalize_unit`, `test_cosine_self_is_one`
- `test_activations.py`: `test_sigmoid_range`, `test_relu_nonneg`, `test_softmax_sums_to_one`, `test_sigmoid_grad_check`
- `test_losses.py`: `test_mse_zero_on_perfect`, `test_ce_decreases_toward_zero`
- `test_optimizer.py`: `test_gd_converges_on_quadratic`
- `test_stats.py`: `test_gradient_check_passes_for_l2_norm`

Run with: `pytest tests/ -v`

---

## Grading

| Criterion | Points |
|---|---|
| All modules implemented with correct signatures | 30 |
| `demo_train.py` reaches ≥ 95% accuracy | 20 |
| ≥ 10 passing tests | 20 |
| Code quality (types, logging, docstrings) | 15 |
| Gradient check passes on at least sigmoid and MSE | 15 |
| **Total** | **100** |

Pass threshold: 75 / 100

---

## Setup

```bash
# Install NumPy (required — do not install automatically)
uv add numpy

# Optional: sklearn for data loading only
uv add scikit-learn

# Run tests
pytest tests/ -v
```

---

## Notes

- Do not use `np.linalg.solve` for logistic regression — implement GD yourself.
- `sklearn` is only allowed for dataset loading (`load_iris`, `make_classification`). No `sklearn.linear_model`.
- All matrix operations must work correctly for both single samples and batches.
- Use `logging` throughout; no bare `print` calls.
