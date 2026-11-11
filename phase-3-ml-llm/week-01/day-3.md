# Phase 3 · Week 01 · Day 3 — Derivatives, Gradients & Chain Rule

> Objective: Understand derivatives and gradients well enough to read and implement backpropagation by hand.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]]

---

## Concept

### Derivative

The derivative `f'(x)` (or `df/dx`) measures how much `f` changes per unit change in `x` at a given point. Mechanically: slope of the tangent line.

```
f'(x) = lim_{h→0} [f(x+h) - f(x)] / h
```

**ML meaning:** gradient descent moves `x` in the direction that *decreases* `f`. The derivative tells us which direction that is.

Common derivatives you must know:

| `f(x)` | `f'(x)` |
|---|---|
| `x²` | `2x` |
| `eˣ` | `eˣ` |
| `ln(x)` | `1/x` |
| `sigmoid(x) = 1/(1+e⁻ˣ)` | `sigmoid(x) * (1 - sigmoid(x))` |
| `ReLU(x) = max(0,x)` | `1 if x>0 else 0` |

### Gradient

When `f` takes a *vector* input `x = [x₁, x₂, ..., xₙ]`, the **gradient** is the vector of partial derivatives:

```
∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

The gradient points in the direction of steepest *ascent*. Gradient descent steps in the opposite direction: `x ← x - α * ∇f(x)`.

### Chain Rule

If `z = f(g(x))` then `dz/dx = f'(g(x)) * g'(x)`.

In neural nets, each layer is a composed function. The chain rule propagates gradients backward through the composition.

Example: `loss = MSE(sigmoid(W @ x + b))`

```
d_loss/dW = d_loss/d_sigmoid * d_sigmoid/d_linear * d_linear/dW
```

Each term is computed during the backward pass.

**Numerical gradient check:** finite difference approximation to verify your analytic gradient:  
`∂f/∂xᵢ ≈ [f(x + εeᵢ) - f(x - εeᵢ)] / (2ε)`  
Should match your analytic gradient to ~1e-5 or better.

---

## How It Works Internally

Autograd libraries (PyTorch, JAX) build a **computation graph** as the forward pass runs. Each node records its operation and stores the inputs needed to compute the local gradient. Backprop is then just a traversal of this graph applying the chain rule at each node.

NumPy has no autograd. We implement it manually — which is the best way to understand it.

---

## Worked Example

```python
"""
day3_derivatives_gradients.py
Derivatives, gradients, chain rule — manual implementation.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# --- Activation functions and their derivatives ---

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


# --- Numerical gradient check ---

def numerical_gradient(
    f,
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference numerical gradient of scalar-valued f at x."""
    grad = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy(); x_plus.flat[i] += eps
        x_minus = x.copy(); x_minus.flat[i] -= eps
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def demo_sigmoid_grad() -> None:
    x = np.array([0.0, 1.0, -2.0])
    analytic = sigmoid_grad(x)
    numeric  = numerical_gradient(lambda v: sigmoid(v).sum(), x)

    log.info("sigmoid_grad analytic : %s", analytic)
    log.info("sigmoid_grad numeric  : %s", numeric)
    log.info("max diff              : %.2e", np.max(np.abs(analytic - numeric)))


# --- Chain rule: MSE ∘ linear ∘ sigmoid ---

def forward(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, float]:
    """One-neuron forward pass. Returns (sigmoid output, MSE loss)."""
    z: np.ndarray = w @ x + b           # linear
    a: np.ndarray = sigmoid(z)          # activation
    loss: float = float(np.mean((a - y_true) ** 2))  # MSE
    return a, loss


def backward(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Chain rule: d_loss/dw and d_loss/db."""
    z: np.ndarray = w @ x + b
    a: np.ndarray = sigmoid(z)

    # d_loss/d_a  (MSE gradient)
    d_loss_da: np.ndarray = 2.0 * (a - y_true) / y_true.size  # scalar per element

    # d_a/d_z  (sigmoid gradient)
    da_dz: np.ndarray = sigmoid_grad(z)

    # d_z/d_w = x,  d_z/d_b = 1  (linear gradient)
    d_loss_dz: np.ndarray = d_loss_da * da_dz   # chain rule, element-wise

    dw: np.ndarray = np.outer(d_loss_dz, x)     # (out, in)
    db: np.ndarray = d_loss_dz                   # (out,)

    return dw, db


def demo_chain_rule() -> None:
    rng = np.random.default_rng(7)
    x      = rng.standard_normal(4)
    w      = rng.standard_normal((2, 4))
    b      = np.zeros(2)
    y_true = np.array([0.0, 1.0])

    a, loss = forward(x, w, b, y_true)
    dw, db  = backward(x, w, b, y_true)

    log.info("loss  = %.6f", loss)
    log.info("dw shape = %s", dw.shape)
    log.info("db shape = %s", db.shape)
    log.info("db (analytic): %s", db)

    # Numerical check on b
    numeric_db = numerical_gradient(
        lambda bv: forward(x, w, bv, y_true)[1], b
    )
    log.info("db (numeric) : %s", numeric_db)
    log.info("max diff     : %.2e", np.max(np.abs(db - numeric_db)))


if __name__ == "__main__":
    demo_sigmoid_grad()
    demo_chain_rule()
```

---

## Exercises

1. Implement `tanh_grad(x)` — the derivative of `tanh`. Verify against a numerical gradient check.
2. For `f(x) = x³ + 2x`, compute `f'(x)` analytically. Write Python code to verify numerically at `x = 2.0`.
3. A loss function is `L = (y_pred - y_true)²`. Compute `dL/dy_pred`. If `y_pred = sigmoid(w * x)`, use the chain rule to find `dL/dw` where `x = 1.5`, `w = 0.5`, `y_true = 1.0`.
4. In the worked example, add a `numerical_gradient` check for `dw` and verify the max difference is < 1e-4.
5. What happens to `sigmoid_grad(x)` at `x = 10` and `x = -10`? Why is this a problem for deep networks? (This is the vanishing gradient problem — explain in 2–3 sentences.)

(Solutions: ../solutions/week-01/day-3.md)

---

## Self-check

- [ ] I can write down the derivative of sigmoid, ReLU, and MSE from memory
- [ ] I can implement a numerical gradient check and know when a difference is "too large"
- [ ] I can trace the chain rule through a 2-layer composition step by step
- [ ] I understand why backprop is just the chain rule applied to a computation graph

---

## Supplementary

- [Karpathy "micrograd" walkthrough (YouTube)](https://www.youtube.com/watch?v=VMj-3S1tku0) — build autograd from scratch
- [CS231n backprop notes (Stanford)](https://cs231n.github.io/optimization-2/)
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)
