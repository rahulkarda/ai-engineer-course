# Solutions — Phase 3 · Week 01 · Day 3

## Exercise 1 — `tanh_grad`

```python
import numpy as np

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2

# Numerical check
def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps; xm.flat[i] -= eps
        grad.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad

x = np.array([0.0, 1.0, -2.0])
analytic = tanh_grad(x)
numeric  = numerical_gradient(lambda v: tanh_act(v).sum(), x)
assert np.max(np.abs(analytic - numeric)) < 1e-5
```

## Exercise 2 — `f(x) = x³ + 2x`

Analytic: `f'(x) = 3x² + 2`  
At `x = 2.0`: `f'(2) = 3*4 + 2 = 14`

```python
def f(x): return x**3 + 2*x
def f_prime(x): return 3*x**2 + 2

x0 = 2.0
analytic = f_prime(x0)
eps = 1e-5
numeric  = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
print(f"analytic={analytic}, numeric={numeric:.6f}, diff={abs(analytic-numeric):.2e}")
```

## Exercise 3 — Chain rule for `dL/dw`

```
L = (y_pred - y_true)²    where y_pred = sigmoid(w * x)
dL/dy_pred = 2(y_pred - y_true)
dy_pred/dw = sigmoid_grad(w*x) * x   [chain rule: d/dw of sigmoid(w*x)]
dL/dw      = 2(y_pred - y_true) * sigmoid_grad(w*x) * x
```

```python
def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def sigmoid_grad(z): return sigmoid(z) * (1.0 - sigmoid(z))

x, w, y_true = 1.5, 0.5, 1.0
z      = w * x
y_pred = sigmoid(z)
dL_dw  = 2.0 * (y_pred - y_true) * sigmoid_grad(z) * x
print(f"y_pred={y_pred:.4f}  dL/dw={dL_dw:.6f}")
```

## Exercise 4 — Numerical check for `dw`

Using the backward function from the lesson, compute `numerical_gradient` for a scalar loss w.r.t. `w` (vectorized):
```python
# In the demo, add:
numeric_dw = numerical_gradient(
    lambda wv: forward(x, wv.reshape(w.shape), b, y_true)[1],
    w.flatten()
).reshape(w.shape)
print("max diff dw:", np.max(np.abs(dw - numeric_dw)))  # should be < 1e-4
```

## Exercise 5 — Vanishing gradient

At `x = 10`: `sigmoid(10) ≈ 0.99995`, `sigmoid_grad(10) ≈ 0.000045` — nearly zero.  
At `x = -10`: `sigmoid(-10) ≈ 0.000045`, `sigmoid_grad(-10) ≈ 0.000045` — nearly zero.

**Explanation:** When the pre-activation `z` is large in magnitude, sigmoid saturates: it outputs values very close to 0 or 1, and its gradient vanishes toward zero. During backprop, this near-zero gradient is multiplied through every preceding layer via the chain rule. For deep networks, the product of many near-zero terms → the gradient reaching early layers becomes effectively zero, making those layers impossible to train. This is called the **vanishing gradient problem** and is the primary reason ReLU replaced sigmoid in deep networks.
