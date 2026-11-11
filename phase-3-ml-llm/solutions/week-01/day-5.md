# Solutions — Phase 3 · Week 01 · Day 5

## Exercise 1 — KL Divergence

```python
import numpy as np

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(p || q) — not symmetric."""
    p = p + eps  # avoid log(0)
    q = q + eps
    return float(np.sum(p * np.log(p / q)))

# Verify KL(p||p) = 0
p = np.array([0.3, 0.5, 0.2])
assert abs(kl_divergence(p, p)) < 1e-8

p2 = np.array([0.8, 0.2])
q2 = np.array([0.5, 0.5])
print(f"KL(p||q) = {kl_divergence(p2, q2):.4f}")  # ~0.195
print(f"KL(q||p) = {kl_divergence(q2, p2):.4f}")  # ~0.223  — asymmetric
```

## Exercise 2 — Softmax by hand

Logits: `z = [2.0, 0.5, -1.0]`

1. `exp(z) = [e², e^0.5, e^-1] ≈ [7.389, 1.649, 0.368]`
2. `sum = 9.406`
3. `softmax = [7.389/9.406, 1.649/9.406, 0.368/9.406] ≈ [0.785, 0.175, 0.039]`

```python
def softmax(z):
    z_s = z - np.max(z)
    e   = np.exp(z_s)
    return e / e.sum()

z = np.array([2.0, 0.5, -1.0])
print(softmax(z))  # [0.785, 0.175, 0.039]
```

## Exercise 3 — CE asymmetry

```python
def cross_entropy(p, q, eps=1e-10):
    return float(-np.sum(p * np.log(q + eps)))

p = np.array([0.8, 0.2])
q = np.array([0.5, 0.5])

hpq = cross_entropy(p, q)  # H(p, q) ≈ 0.722
hqp = cross_entropy(q, p)  # H(q, p) ≈ 0.832
print(f"H(p,q)={hpq:.4f}  H(q,p)={hqp:.4f}")
```

**Why asymmetry matters:** During training, `p` is the *true* distribution (fixed labels) and `q` is the *model* output. We optimize `H(p, q)` — we want the model `q` to match the true `p`. Reversing them (`H(q, p)`) would give a different loss with different gradients and different convergence properties.

## Exercise 4 — Accuracy vs. CE

```python
def accuracy(logits: np.ndarray, y_true: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y_true))
```

Accuracy and CE don't always move together because accuracy is a **step function** — it only changes when a prediction crosses the classification boundary. CE can decrease (the model becomes more confident) without a prediction flipping from wrong to correct. This is why CE is differentiable and useful for training while accuracy is the interpretable evaluation metric.

## Exercise 5 — CE gradient via numerical check

CE gradient w.r.t. logits `z` for a single example: `softmax(z) - one_hot(y)`

```python
def one_hot(y: int, n_classes: int) -> np.ndarray:
    oh = np.zeros(n_classes)
    oh[y] = 1.0
    return oh

def ce_grad_analytic(z: np.ndarray, y_true: int) -> np.ndarray:
    return softmax(z) - one_hot(y_true, len(z))

z = np.array([2.0, 0.5, -1.0])
y = 0

analytic = ce_grad_analytic(z, y)
numeric  = numerical_gradient(lambda v: cross_entropy_loss(v, y), z)
print("max diff:", np.max(np.abs(analytic - numeric)))  # < 1e-5
```
