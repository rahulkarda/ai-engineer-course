# Phase 3 · Week 01 · Day 6 — Applied Problem Set (4 hr)

> Objective: Consolidate the full week — vectors, norms, gradients, GD, cross-entropy — by working applied ML problems that combine multiple concepts.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

---

## Overview

Today is a deep-practice session. Four extended problems, each mixing multiple topics from the week. Work independently; refer to your notes, not solutions. Time guidance is approximate — depth > speed.

---

## Problem 1 — From-Scratch Logistic Regression (90 min)

Build a binary logistic regression classifier in NumPy. No scikit-learn.

### Spec

A logistic regression model predicts:  
`P(y=1 | x) = sigmoid(w · x + b)`

Training loss (binary cross-entropy):  
`L = -(1/N) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]`

Gradient:  
`∂L/∂w = (1/N) Xᵀ(ŷ - y)`  
`∂L/∂b = (1/N) Σ(ŷ - y)`

### Tasks

1. Implement `LogisticRegression` as a Python class with:
   - `fit(X, y, lr, n_steps)` — gradient descent training
   - `predict_proba(X)` — returns sigmoid probabilities
   - `predict(X, threshold=0.5)` — returns 0/1 labels
   - `accuracy(X, y)` — fraction correct
2. Generate a linearly separable 2D dataset using `sklearn.datasets.make_classification` or pure NumPy (no sklearn required — write a generator if you prefer).
3. Train your model. Plot (or log) the loss curve and final accuracy.
4. Add L2 regularization (`λ = 0.01`). Compare trained weights with and without regularization.

### Acceptance criteria

- Loss decreases monotonically for `lr = 0.1`
- Accuracy ≥ 90% on a linearly separable synthetic dataset
- Gradient shapes match input dimensions
- At least 3 pytest unit tests:
  - `test_sigmoid_range`: sigmoid output ∈ (0, 1)
  - `test_predict_shape`: output shape matches input batch size
  - `test_loss_decreases`: final loss < initial loss after 100 steps

```python
# starter scaffold — fill in the TODOs
"""
day6_logistic_regression.py
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
    lr: float      = 0.1
    n_steps: int   = 1000
    l2_lambda: float = 0.0

    weights_: np.ndarray = field(default_factory=lambda: np.array([]))
    bias_:    float      = 0.0
    losses_:  list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        # TODO
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
```

---

## Problem 2 — Cosine Similarity Search (45 min)

Build a minimal vector search engine.

### Tasks

1. Generate 1000 random 64-dimensional "document" vectors. L2-normalize each.
2. Implement `top_k_search(query, docs, k=5)` — returns indices and scores of the top-k most similar documents using cosine similarity.
3. Verify that searching with a document as its own query returns itself as the top result.
4. Measure wall-clock time for `k=10` search over 10,000 documents. Does vectorizing (matrix multiply vs. loop) make a difference? Log both times.

### Acceptance criteria

- Self-search returns rank 1
- Vectorized version ≥ 5× faster than loop version for 10k docs

---

## Problem 3 — Numerical Gradient Checker (30 min)

Build a reusable gradient checker you can use for any scalar loss function.

### Tasks

1. Implement `gradient_checker(f, x, analytic_grad, eps=1e-5, tol=1e-4)` that:
   - Computes numerical gradient via central differences
   - Compares element-wise to `analytic_grad`
   - Returns `(passed: bool, max_diff: float, rel_error: float)`
2. Test it on:
   - `f(x) = ‖x‖²`, gradient = `2x`
   - `f(x) = sum(sigmoid(x))`, gradient = `sigmoid_grad(x)`
   - Your logistic regression loss from Problem 1

### Acceptance criteria

- Max absolute difference < `tol` for all three test functions
- Function returns a typed result (use a dataclass or NamedTuple)

---

## Problem 4 — Cross-Entropy vs. Accuracy (45 min)

Investigate the relationship between CE loss and accuracy during training.

### Tasks

1. Train your logistic regression on a noisy dataset (`noise_std=2.0`).
2. Log both CE loss and accuracy every 10 steps.
3. Observe and answer:
   - Does accuracy improve every time loss decreases?
   - Can you construct a case where loss decreases but accuracy doesn't change? (Hint: think about predictions near the 0.5 threshold.)
   - Why do practitioners track both metrics instead of just accuracy?
4. Implement a `confusion_matrix(y_pred, y_true)` function — returns a 2×2 NumPy array. Log true positives, false positives, true negatives, false negatives.

---

## Session Checklist

- [ ] Problem 1: logistic regression implemented, tests passing
- [ ] Problem 2: vector search implemented, speedup logged
- [ ] Problem 3: gradient checker implemented and verified
- [ ] Problem 4: analysis written up (even just log statements)

---

(Solutions: ../solutions/week-01/day-6.md)

---

## Supplementary

- [StatQuest Logistic Regression (YouTube)](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [The Illustrated Machine Learning website — all basics](https://illustrated-machine-learning.github.io/)
