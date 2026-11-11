# Solutions — Phase 3 · Week 01 · Day 6

## Problem 1 — Logistic Regression

```python
"""
day6_logistic_regression_solution.py
"""
import logging
from dataclasses import dataclass, field
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


@dataclass
class LogisticRegression:
    lr: float        = 0.1
    n_steps: int     = 1000
    l2_lambda: float = 0.0

    weights_: np.ndarray = field(default_factory=lambda: np.array([]))
    bias_:    float      = 0.0
    losses_:  list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n, d = X.shape
        self.weights_ = np.zeros(d)
        self.bias_    = 0.0
        self.losses_  = []

        for _ in range(self.n_steps):
            z      = X @ self.weights_ + self.bias_
            y_pred = sigmoid(z)
            # Binary CE loss
            loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
            loss += self.l2_lambda * float(np.sum(self.weights_ ** 2))
            self.losses_.append(loss)

            diff = y_pred - y
            dw   = (1.0 / n) * X.T @ diff + 2 * self.l2_lambda * self.weights_
            db   = float(np.mean(diff))

            self.weights_ -= self.lr * dw
            self.bias_     -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
```

### Tests

```python
# tests/test_logistic.py
import numpy as np
import pytest
from day6_logistic_regression_solution import LogisticRegression, sigmoid


def make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X0  = rng.normal([-2, -2], 0.5, (100, 2))
    X1  = rng.normal([+2, +2], 0.5, (100, 2))
    X   = np.vstack([X0, X1])
    y   = np.array([0] * 100 + [1] * 100, dtype=float)
    return X, y


def test_sigmoid_range():
    z = np.linspace(-10, 10, 100)
    s = sigmoid(z)
    assert np.all(s > 0) and np.all(s < 1)


def test_predict_shape():
    X, y = make_data()
    model = LogisticRegression().fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (200,)


def test_loss_decreases():
    X, y = make_data()
    model = LogisticRegression(lr=0.1, n_steps=100).fit(X, y)
    assert model.losses_[-1] < model.losses_[0]
```

## Problem 2 — Vector Search

```python
import numpy as np
import time

rng  = np.random.default_rng(0)
N, D = 10_000, 64
docs = rng.standard_normal((N, D))
docs /= np.linalg.norm(docs, axis=1, keepdims=True)  # L2-normalize

def top_k_search_vectorized(query: np.ndarray, docs: np.ndarray, k: int = 5):
    q = query / np.linalg.norm(query)
    scores = docs @ q  # (N,) — cosine similarity (since both normalized)
    top_k  = np.argsort(scores)[::-1][:k]
    return top_k, scores[top_k]

def top_k_search_loop(query: np.ndarray, docs: np.ndarray, k: int = 5):
    q = query / np.linalg.norm(query)
    scores = [float(np.dot(docs[i], q)) for i in range(len(docs))]
    top_k  = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return top_k, [scores[i] for i in top_k]

# Self-search test
idx = 42
q   = docs[idx].copy()
idxs, scores = top_k_search_vectorized(q, docs)
assert idxs[0] == idx, f"Self not rank 1: got {idxs[0]}"

# Speedup
t0 = time.perf_counter(); top_k_search_vectorized(q, docs); t1 = time.perf_counter()
t2 = time.perf_counter(); top_k_search_loop(q, docs);       t3 = time.perf_counter()
print(f"vectorized={t1-t0:.4f}s  loop={t3-t2:.4f}s  speedup={((t3-t2)/(t1-t0)):.1f}x")
# Typical: 50-200x speedup
```

## Problem 3 — Gradient Checker

```python
from typing import NamedTuple
import numpy as np

class GradCheckResult(NamedTuple):
    passed: bool
    max_diff: float
    rel_error: float

def gradient_checker(
    f,
    x: np.ndarray,
    analytic_grad: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-4,
) -> GradCheckResult:
    numeric = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps; xm.flat[i] -= eps
        numeric.flat[i] = (f(xp) - f(xm)) / (2 * eps)

    diff      = np.abs(analytic_grad - numeric)
    max_diff  = float(diff.max())
    denom     = np.abs(analytic_grad) + np.abs(numeric) + 1e-10
    rel_error = float((diff / denom).max())
    return GradCheckResult(max_diff < tol, max_diff, rel_error)

# Test on ‖x‖²
x = np.array([1.0, 2.0, -1.0])
r = gradient_checker(lambda v: float(np.sum(v**2)), x, 2 * x)
assert r.passed, f"Failed with max_diff={r.max_diff}"
```

## Problem 4 — CE vs Accuracy Analysis

**Key insight:** Accuracy is a step function — it only changes when a prediction crosses 0.5. Cross-entropy loss is continuous. A model can become more confident in correct predictions (CE decreases) without any prediction switching class (accuracy unchanged).

```python
def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Returns 2x2 array [[TN, FP], [FN, TP]]."""
    cm = np.zeros((2, 2), dtype=int)
    for p, t in zip(y_pred, y_true):
        cm[int(t), int(p)] += 1
    return cm

# Row = actual, Col = predicted
# cm[0,0]=TN  cm[0,1]=FP  cm[1,0]=FN  cm[1,1]=TP
```

**Why track both:** Accuracy tells you final decision correctness; CE tells you calibration quality (is the model confident when right?). A model can have 90% accuracy but poor CE if it's barely crossing the threshold. Track CE for training health, accuracy for business interpretation.
