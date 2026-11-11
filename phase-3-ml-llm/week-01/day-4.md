# Phase 3 · Week 01 · Day 4 — Partial Derivatives & Gradient Descent

> Objective: Implement gradient descent from scratch in NumPy and develop intuition for why it works.  ·  Time: ~1 hr  ·  Prereqs: [[day-3]]

---

## Concept

### Partial Derivatives

When a function depends on multiple variables, the **partial derivative** `∂f/∂xᵢ` measures how `f` changes as `xᵢ` changes, *holding all other variables fixed*.

Example: `f(x, y) = x² + 3xy + y²`  
`∂f/∂x = 2x + 3y`  
`∂f/∂y = 3x + 2y`

The gradient `∇f = [∂f/∂x, ∂f/∂y]` collects all partials — one per input dimension.

### Gradient Descent

Start at some point `θ`. Repeatedly move in the direction that decreases the loss:

```
θ ← θ - α * ∇L(θ)
```

- `α` is the **learning rate** (step size).
- Too large: overshoot, diverge.
- Too small: converge slowly.

**Convergence intuition:** for a convex loss (bowl-shaped), gradient descent always reaches the global minimum. For non-convex (neural nets), it finds *a* local minimum — in practice, often good enough.

### Variants

| Variant | Gradient over | Trade-off |
|---|---|---|
| Batch GD | Full dataset | Stable but slow per step |
| Stochastic GD (SGD) | 1 sample | Noisy but fast; escapes local minima |
| Mini-batch GD | Batch of B samples | Best of both; standard in DL |

### The Gradient Descent Update in Matrix Form

For a linear model `ŷ = X @ w`:  
Loss `L = (1/N) * ‖ŷ - y‖²` (MSE)  
Gradient: `∇L(w) = (2/N) * Xᵀ @ (ŷ - y)`  
Update: `w ← w - α * ∇L(w)`

Deriving this is the canonical first backprop exercise.

---

## How It Works Internally

Gradient descent is simple — but choosing `α` matters enormously. Modern optimizers (Adam, AdaGrad) adaptively scale the learning rate per parameter using running statistics of past gradients. We implement vanilla GD here; the mechanics are identical.

For a quadratic loss (MSE on a linear model), the loss surface is a convex bowl. The optimal `α` is `1/λ_max` where `λ_max` is the largest eigenvalue of `(2/N) * Xᵀ @ X`. Too large → gradient explodes. This is why learning rate warm-up and scheduling exist.

---

## Worked Example

```python
"""
day4_gradient_descent.py
Gradient descent on linear regression — from scratch.
"""
import logging
from dataclasses import dataclass
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class GDResult:
    weights: np.ndarray
    loss_history: list[float]
    n_steps: int


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mse_gradient(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """Gradient of MSE w.r.t. weights: (2/N) * Xᵀ @ (ŷ - y)."""
    n = X.shape[0]
    return (2.0 / n) * X.T @ (y_pred - y_true)


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    n_steps: int = 1000,
    tol: float = 1e-6,
) -> GDResult:
    """
    Fit linear regression via batch gradient descent.

    Parameters
    ----------
    X      : (N, D) feature matrix — include a bias column if desired
    y      : (N,) target values
    lr     : learning rate (step size)
    n_steps: max iterations
    tol    : stop if loss change < tol
    """
    n, d = X.shape
    w: np.ndarray = np.zeros(d)           # start at origin
    loss_history: list[float] = []

    for step in range(n_steps):
        y_pred = X @ w
        loss   = mse_loss(y_pred, y)
        loss_history.append(loss)

        # Check convergence
        if step > 0 and abs(loss_history[-2] - loss) < tol:
            log.info("Converged at step %d, loss=%.6f", step, loss)
            break

        grad = mse_gradient(X, y_pred, y)
        w -= lr * grad

        if step % 100 == 0:
            log.info("step %4d | loss = %.6f", step, loss)

    return GDResult(weights=w, loss_history=loss_history, n_steps=len(loss_history))


def make_dataset(
    n: int = 200,
    true_w: np.ndarray | None = None,
    noise_std: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = X @ w_true + noise with a bias column in X."""
    rng = np.random.default_rng(seed)
    if true_w is None:
        true_w = np.array([2.0, -1.5, 0.5])

    # 2 features + 1 bias column
    X_raw: np.ndarray = rng.standard_normal((n, len(true_w) - 1))
    X: np.ndarray     = np.hstack([X_raw, np.ones((n, 1))])
    y: np.ndarray     = X @ true_w + rng.normal(0, noise_std, n)
    return X, y


def demo() -> None:
    true_w = np.array([2.0, -1.5, 0.5])
    X, y = make_dataset(true_w=true_w)

    result = gradient_descent(X, y, lr=0.05, n_steps=2000)

    log.info("True weights  : %s", true_w)
    log.info("Fitted weights: %s", np.round(result.weights, 4))
    log.info("Final loss    : %.6f", result.loss_history[-1])
    log.info("Steps taken   : %d", result.n_steps)


def demo_learning_rate_comparison() -> None:
    """Show effect of learning rate on convergence."""
    X, y = make_dataset()

    for lr in [0.001, 0.05, 0.5, 2.0]:
        try:
            result = gradient_descent(X, y, lr=lr, n_steps=500)
            final_loss = result.loss_history[-1]
            log.info("lr=%.3f  final_loss=%.4f  steps=%d", lr, final_loss, result.n_steps)
        except Exception as exc:
            log.warning("lr=%.3f  FAILED: %s", lr, exc)


if __name__ == "__main__":
    demo()
    demo_learning_rate_comparison()
```

---

## Exercises

1. Add **momentum** to the gradient descent function: `v = β*v + (1-β)*grad`, then `w -= lr * v`. Use `β = 0.9`. Compare convergence steps to vanilla GD.
2. Implement **mini-batch gradient descent** — split `X` into batches of size `B=32` per step. Observe how the loss curve changes (noisier but faster per epoch).
3. The true `w` is `[2.0, -1.5, 0.5]`. After training, how close is the recovered `w`? At what dataset size does recovery become unreliable? Experiment with `n=10, 50, 200`.
4. Add an `L2` regularization term: `loss = MSE + λ * ‖w‖²`. Derive the new gradient analytically and update the code. Test with `λ = 0.1`.
5. Plot (or log) the loss curve for 3 learning rates. Describe the shape for a good LR vs. an LR that's too large.

(Solutions: ../solutions/week-01/day-4.md)

---

## Self-check

- [ ] I can write the gradient of MSE w.r.t. weights from memory
- [ ] I can implement gradient descent in 15 lines of NumPy
- [ ] I understand why learning rate is the most important hyperparameter
- [ ] I can explain the trade-off between batch GD, SGD, and mini-batch GD

---

## Supplementary

- [Gradient Descent, How Neural Networks Learn — 3Blue1Brown (YouTube)](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Why Momentum Really Works — distill.pub](https://distill.pub/2017/momentum/)
- [An overview of gradient descent optimization algorithms — Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/)
