# Phase 3 · Week 03 · Day 3 — Backpropagation: Chain Rule, Gradient Derivation, NumPy Implementation

> Objective: Derive backprop for a 2-layer MLP from the chain rule, then implement the full training loop in NumPy.  ·  Time: ~1 hr (4-hour conceptual equivalent — see pointers below)  ·  Prereqs: [[week-03/day-2]]

---

## Concept

**Backpropagation** is the algorithm that computes the gradient of the loss `L` with respect to every parameter in the network. It applies the chain rule of calculus repeatedly, flowing gradients from the output layer back to the input layer — hence "back" propagation.

Without backprop there is no training. This is the most important algorithm in deep learning.

### The Chain Rule

For composed functions `L = f(g(x))`:

```
dL/dx = (dL/dg) * (dg/dx)
```

For a layer's computation `z = W @ a + b`, then `a_next = activation(z)`, then loss `L`:

```
dL/dW = dL/da_next  *  da_next/dz  *  dz/dW
      = delta       *  activation'(z)  *  a_prev
```

where `delta = dL/dz` is called the **error signal** for this layer.

### Notation for 2-Layer MLP

Network:
```
a0 = x                          (input)
z1 = W1 @ a0 + b1               (layer 1 pre-activation)
a1 = relu(z1)                   (layer 1 post-activation)
z2 = W2 @ a1 + b2               (layer 2 pre-activation / logits)
L  = cross_entropy(y, softmax(z2))
```

### Gradient Derivations

**Layer 2 (output layer):**

The softmax + cross-entropy gradient is derived in Day 2:

```
delta2 = dL/dz2 = softmax(z2) - y_true     shape: (batch, n_out)

dL/dW2 = (1/B) * delta2.T @ a1             shape: (n_out, n_hidden)
dL/db2 = (1/B) * sum(delta2, axis=0)       shape: (n_out,)
```

**Layer 1 (hidden layer):**

Error signal propagated back through W2, then gated by ReLU derivative:

```
da1    = delta2 @ W2                        shape: (batch, n_hidden)
relu'(z) = 1 if z > 0 else 0               (Heaviside step function)
delta1 = da1 * relu_grad(z1)               shape: (batch, n_hidden)  [element-wise *]

dL/dW1 = (1/B) * delta1.T @ a0             shape: (n_hidden, n_in)
dL/db1 = (1/B) * sum(delta1, axis=0)       shape: (n_hidden,)
```

### Parameter Update (Gradient Descent)

```
W = W - lr * dL/dW
b = b - lr * dL/db
```

`lr` is the learning rate — a scalar hyperparameter controlling step size. Too large: oscillates/diverges. Too small: slow convergence.

### What the Gradient Means

`dL/dW1[i, j]` tells us: "if we increase `W1[i, j]` by a tiny `epsilon`, the loss changes by `dL/dW1[i, j] * epsilon`." Gradient descent moves every parameter slightly against this direction to reduce loss.

---

## How it works internally

### Forward and backward passes are symmetric

Forward pass: data flows left to right, each layer computes and caches its output.
Backward pass: gradients flow right to left, each layer consumes the incoming gradient and produces the outgoing gradient.

The cache from the forward pass is essential — `z1` and `a1` must be stored during forward to compute `delta1` during backward.

### The ReLU gradient

```
relu'(z) = 1  if z > 0
relu'(z) = 0  if z <= 0
```

In NumPy: `(z > 0).astype(float)`. This is the **Heaviside** function. It zeroes out gradients for neurons that were inactive (did not fire during forward pass). This is the source of the "dying ReLU" problem — a neuron that never fires never receives a gradient and never updates.

### Gradient checking

Numerical approximation of gradient via finite differences:

```
df/dx ≈ (f(x + h) - f(x - h)) / (2h)   for small h (e.g. 1e-5)
```

If your analytical gradient matches the numerical approximation (relative error < 1e-5), backprop is correct. This is a standard debugging technique — always check before trusting a new backprop implementation.

### Shape discipline

Every gradient must have the same shape as its parameter:

| Parameter | Shape | Gradient | Shape |
|---|---|---|---|
| W1 | `(n_hidden, n_in)` | dW1 | `(n_hidden, n_in)` |
| b1 | `(n_hidden,)` | db1 | `(n_hidden,)` |
| W2 | `(n_out, n_hidden)` | dW2 | `(n_out, n_hidden)` |
| b2 | `(n_out,)` | db2 | `(n_out,)` |

Mismatched shapes are the most common backprop bug. Add `assert` statements during development.

---

## Worked example

```python
"""
day_3_backprop.py — Manual backpropagation for a 2-layer MLP.
Includes gradient checking to verify correctness.
"""

import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---- Activations and loss ----

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 where z > 0, else 0."""
    return (z > 0).astype(np.float64)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def cross_entropy_from_logits(
    y_onehot: np.ndarray,
    logits: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Stable CE: log(softmax(z)) via log-sum-exp."""
    max_z = logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(logits - max_z).sum(axis=1, keepdims=True) + eps) + max_z
    log_probs = logits - log_sum_exp
    return float(-np.mean(np.sum(y_onehot * log_probs, axis=1)))


# ---- Forward pass with full cache ----

def forward(
    x: np.ndarray,
    params: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Forward pass.

    Returns:
        logits: Pre-softmax output, shape (batch, n_out).
        cache: Intermediate values needed for backward.
    """
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    z1 = x @ W1.T + b1     # (batch, n_hidden)
    a1 = relu(z1)           # (batch, n_hidden)
    z2 = a1 @ W2.T + b2    # (batch, n_out)

    cache = {"x": x, "z1": z1, "a1": a1, "z2": z2}
    return z2, cache


# ---- Backward pass ----

def backward(
    y_onehot: np.ndarray,
    logits: np.ndarray,
    cache: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Backpropagation through a 2-layer ReLU MLP with CE loss.

    Returns:
        grads: Dict with keys dW1, db1, dW2, db2.
    """
    batch = y_onehot.shape[0]
    W2 = params["W2"]

    # ---- Layer 2 gradients (output) ----
    # dL/dz2 = softmax(z2) - y  (fused softmax + CE gradient)
    delta2 = (softmax(logits) - y_onehot) / batch   # (batch, n_out)

    dW2 = delta2.T @ cache["a1"]                    # (n_out, n_hidden)
    db2 = delta2.sum(axis=0)                         # (n_out,)

    # ---- Layer 1 gradients (hidden) ----
    # Propagate error back through W2
    da1 = delta2 @ W2                               # (batch, n_hidden)
    # Gate by ReLU derivative
    delta1 = da1 * relu_grad(cache["z1"])           # (batch, n_hidden)

    dW1 = delta1.T @ cache["x"]                    # (n_hidden, n_in)
    db1 = delta1.sum(axis=0)                        # (n_hidden,)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # Shape assertions — catch transposition bugs early
    for key, param_key in [("dW1", "W1"), ("db1", "b1"), ("dW2", "W2"), ("db2", "b2")]:
        assert grads[key].shape == params[param_key].shape, (
            f"Shape mismatch: {key} {grads[key].shape} vs {param_key} {params[param_key].shape}"
        )

    return grads


# ---- Gradient descent update ----

def sgd_update(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    lr: float,
) -> dict[str, np.ndarray]:
    """Vanilla SGD: W = W - lr * dW."""
    return {
        "W1": params["W1"] - lr * grads["dW1"],
        "b1": params["b1"] - lr * grads["db1"],
        "W2": params["W2"] - lr * grads["dW2"],
        "b2": params["b2"] - lr * grads["db2"],
    }


# ---- Gradient check ----

def gradient_check(
    x: np.ndarray,
    y_onehot: np.ndarray,
    params: dict[str, np.ndarray],
    h: float = 1e-5,
) -> None:
    """Verify analytical gradients against numerical finite differences.

    Uses two-sided estimate: (f(x+h) - f(x-h)) / (2h).
    Reports relative error for each parameter.
    """
    # Analytical gradients
    logits, cache = forward(x, params)
    grads = backward(y_onehot, logits, cache, params)

    for param_name, grad_name in [
        ("W1", "dW1"), ("b1", "db1"),
        ("W2", "dW2"), ("b2", "db2"),
    ]:
        param = params[param_name]
        analytical_grad = grads[grad_name]
        numerical_grad = np.zeros_like(param)

        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = param[idx]

            param[idx] = orig + h
            loss_plus, _ = forward(x, params)
            l_plus = cross_entropy_from_logits(y_onehot, loss_plus)

            param[idx] = orig - h
            loss_minus, _ = forward(x, params)
            l_minus = cross_entropy_from_logits(y_onehot, loss_minus)

            numerical_grad[idx] = (l_plus - l_minus) / (2.0 * h)
            param[idx] = orig  # restore
            it.iternext()

        rel_error = (
            np.abs(analytical_grad - numerical_grad).max()
            / (np.abs(analytical_grad).max() + np.abs(numerical_grad).max() + 1e-12)
        )
        status = "PASS" if rel_error < 1e-4 else "FAIL"
        log.info("Gradient check %s (%s): relative error = %.2e", param_name, status, rel_error)


# ---- Training loop ----

def train(
    x: np.ndarray,
    y_onehot: np.ndarray,
    params: dict[str, np.ndarray],
    lr: float = 0.1,
    n_epochs: int = 200,
    log_every: int = 50,
) -> list[float]:
    """Train MLP with SGD. Returns loss history."""
    loss_history: list[float] = []

    for epoch in range(n_epochs):
        logits, cache = forward(x, params)
        loss = cross_entropy_from_logits(y_onehot, logits)
        grads = backward(y_onehot, logits, cache, params)
        params = sgd_update(params, grads, lr)

        loss_history.append(loss)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            probs = softmax(logits)
            preds = probs.argmax(axis=1)
            true_labels = y_onehot.argmax(axis=1)
            accuracy = float((preds == true_labels).mean())
            log.info(
                "Epoch %4d/%d  loss=%.4f  accuracy=%.2f",
                epoch + 1, n_epochs, loss, accuracy,
            )

    return loss_history


# ---- Main ----

def main() -> None:
    rng = np.random.default_rng(seed=7)

    n_in, n_hidden, n_out = 4, 8, 3
    batch = 30

    # Random classification data
    x = rng.standard_normal((batch, n_in))
    labels = rng.integers(0, n_out, size=batch)
    y_onehot = np.zeros((batch, n_out))
    y_onehot[np.arange(batch), labels] = 1.0

    # He init for W1 (ReLU), Xavier for W2
    params: dict[str, np.ndarray] = {
        "W1": rng.standard_normal((n_hidden, n_in)) * np.sqrt(2.0 / n_in),
        "b1": np.zeros(n_hidden),
        "W2": rng.standard_normal((n_out, n_hidden)) * np.sqrt(1.0 / n_hidden),
        "b2": np.zeros(n_out),
    }

    log.info("=== Gradient Check ===")
    gradient_check(x[:4], y_onehot[:4], params)  # small subset for speed

    log.info("=== Training ===")
    loss_history = train(x, y_onehot, params, lr=0.05, n_epochs=300, log_every=50)

    log.info("Initial loss: %.4f", loss_history[0])
    log.info("Final loss:   %.4f", loss_history[-1])
    log.info("Loss decreased: %s", loss_history[-1] < loss_history[0])


if __name__ == "__main__":
    main()
```

Expected: gradient check passes for all parameters (relative error < 1e-4), and loss decreases monotonically over 300 epochs.

---

## Exercises

1. **Derivation (pen and paper)**: Derive `dL/dW1` step by step from the chain rule. Write out every intermediate step explicitly. Compare to the code. Where does the `1/B` factor come from?

2. **Coding**: Implement gradient descent with **momentum**: keep a velocity `v = beta * v + (1-beta) * grad` and update `W = W - lr * v`. Use `beta=0.9`. Verify it converges faster than vanilla SGD.

3. **Gradient check**: The current gradient check tests W1 element `[0,0]` through `[n_hidden-1, n_in-1]` via full iteration. This is slow. Implement a **sampled gradient check** that randomly selects 20 elements per parameter and checks only those. Compare running time.

4. **Dying ReLU**: Initialize `b1` to `-10.0` (very negative). Run the training loop. Observe whether the network learns. Explain what happens to `relu_grad(z1)` when `z1` is always negative.

5. **Conceptual**: Explain in one paragraph why storing the forward pass cache is necessary for backprop. What would happen if we only stored the final loss and discarded intermediate activations?

(Solutions: ../solutions/week-03/day-3.md)

---

## Self-check

- [ ] I can derive `delta2 = softmax(z2) - y_true` from first principles without reference.
- [ ] I can state why `delta1 = (delta2 @ W2) * relu_grad(z1)` and identify what each term represents.
- [ ] I can implement the full backward pass from scratch given the forward cache.
- [ ] I can implement gradient checking and explain what relative error threshold indicates correctness.
- [ ] I can implement a complete training loop (forward, loss, backward, update) that achieves decreasing loss.

---

## Deep-dive pointers (for the full 4-hour version)

This lesson compresses a substantial topic. For deeper mastery:

- Andrej Karpathy's "micrograd" (55-line autograd engine): https://github.com/karpathy/micrograd — build your own autograd from scratch
- 3Blue1Brown "Backpropagation calculus" (visual): https://www.youtube.com/watch?v=tIeHLnjs5U8
- CS231n Lecture 4 (canonical backprop treatment): https://cs231n.github.io/optimization-2/
- Nielsen "Neural Networks and Deep Learning" Ch. 2 (proof of the four backprop equations): http://neuralnetworksanddeeplearning.com/chap2.html

---

## Supplementary Links

- Stanford CS231n backprop notes: https://cs231n.github.io/optimization-2/
- Karpathy micrograd: https://github.com/karpathy/micrograd
- Gradient checking technique: https://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes-backprop.pdf
- Nielsen book Ch. 2: http://neuralnetworksanddeeplearning.com/chap2.html
