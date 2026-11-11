# Solutions — Phase 3 · Week 01 · Day 4

## Exercise 1 — Momentum

```python
import numpy as np
from dataclasses import dataclass, field

@dataclass
class MomentumGD:
    lr: float = 0.05
    beta: float = 0.9
    n_steps: int = 2000

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[float]]:
        n, d = X.shape
        w = np.zeros(d)
        v = np.zeros(d)  # velocity
        losses: list[float] = []

        for _ in range(self.n_steps):
            y_pred = X @ w
            loss   = float(np.mean((y_pred - y) ** 2))
            losses.append(loss)
            grad = (2.0 / n) * X.T @ (y_pred - y)

            v = self.beta * v + (1 - self.beta) * grad
            w -= self.lr * v

        return w, losses
```

Momentum typically converges in roughly half the steps of vanilla GD because it accumulates velocity in consistent gradient directions and dampens oscillations.

## Exercise 2 — Mini-batch gradient descent

```python
def mini_batch_gd(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    batch_size: int = 32,
    n_epochs: int = 100,
) -> tuple[np.ndarray, list[float]]:
    n, d = X.shape
    w = np.zeros(d)
    epoch_losses: list[float] = []
    rng = np.random.default_rng(0)

    for epoch in range(n_epochs):
        indices = rng.permutation(n)
        epoch_loss = 0.0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb, yb    = X[batch_idx], y[batch_idx]
            nb        = Xb.shape[0]

            y_pred = Xb @ w
            loss   = float(np.mean((y_pred - yb) ** 2))
            epoch_loss += loss
            grad   = (2.0 / nb) * Xb.T @ (y_pred - yb)
            w -= lr * grad

        epoch_losses.append(epoch_loss / (n // batch_size))

    return w, epoch_losses
```

The loss curve is noisier per step but each epoch covers the full dataset, so wall-clock convergence is faster.

## Exercise 3 — Recovery vs. dataset size

```
n=10  : recovered w often very far from truth (high variance, underdetermined)
n=50  : recovers rough direction but significant error
n=200 : recovers within ~0.05 of true weights
```

With `n < d` (fewer samples than features), the system is underdetermined. Linear regression requires at minimum `n >> d` for reliable recovery.

## Exercise 4 — L2 regularization

New loss: `L = MSE + λ * ‖w‖²`  
Gradient: `∇L = (2/N) * Xᵀ(ŷ - y) + 2λw`

```python
def mse_gradient_l2(X, y_pred, y_true, w, lam=0.1):
    n = X.shape[0]
    return (2.0 / n) * X.T @ (y_pred - y_true) + 2.0 * lam * w
```

With `λ = 0.1`, fitted weights will be slightly shrunk toward zero. The bias column weight should be less affected if not regularized separately.

## Exercise 5 — Learning rate shapes

| LR | Behavior |
|---|---|
| Very small (0.001) | Slow steady descent, many steps needed |
| Good (0.05) | Smooth exponential decay to minimum |
| Too large (0.5) | Oscillates or converges slowly; may appear stable |
| Way too large (2.0) | Diverges — loss grows to `inf` or `nan` |

A good LR shows a smooth decreasing curve that flattens near the minimum. Too large shows oscillation or explosion. Too small shows barely perceptible progress.
