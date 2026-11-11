# Phase 3 · Week 03 · Day 6 — Build a Neural Network from Scratch

> Objective: Implement a 2-layer neural net in pure NumPy (manual backprop), then re-implement in PyTorch and verify both produce equivalent results.  ·  Time: ~4 hr  ·  Prereqs: [[week-03/day-5]]

---

## Concept

Building a network from scratch forces you to internalize what PyTorch does automatically:

- The **forward pass** is matrix multiplication + activation functions.
- The **loss** is a scalar reduction over predictions and targets.
- **Backpropagation** is repeated application of the chain rule, moving backward through every operation.
- The **parameter update** is gradient descent: subtract a scaled gradient from each weight.

Until you have traced a gradient back through a `tanh` or a `ReLU` by hand, the PyTorch autograd graph is a black box. After this day it is not.

---

## How it works internally

### Notation for a 2-layer network

```
Input X (N, D)
  -> Linear: Z1 = X @ W1 + b1          (N, H)
  -> Activation: A1 = relu(Z1)          (N, H)
  -> Linear: Z2 = A1 @ W2 + b2          (N, C)
  -> Output: probabilities / logits      (N, C)
```

For binary classification with a single output unit, use sigmoid + binary cross-entropy:

```
loss = -mean(y * log(sigma(Z2)) + (1-y) * log(1 - sigma(Z2)))
```

### Backward pass — chain rule step by step

```
dL/dZ2 = sigma(Z2) - y                  # BCE + sigmoid combined gradient
dL/dW2 = A1.T @ dL/dZ2 / N
dL/db2 = mean(dL/dZ2, axis=0)

dL/dA1 = dL/dZ2 @ W2.T
dL/dZ1 = dL/dA1 * relu_prime(Z1)       # relu_prime = 1 where Z1 > 0 else 0
dL/dW1 = X.T @ dL/dZ1 / N
dL/db1 = mean(dL/dZ1, axis=0)
```

The division by `N` is because the loss is a mean over the batch.

### Why the gradients have the shapes they do

`W1` is `(D, H)`, so `dL/dW1` must also be `(D, H)`. `X.T` is `(D, N)`, `dL/dZ1` is `(N, H)`, product is `(D, H)`. The shape arithmetic always checks out.

---

## Worked example

### Part 1 — NumPy implementation

```python
# day6_numpy_nn.py
"""
2-layer neural network in NumPy. Manual forward pass, loss, backprop, parameter update.
Trained on the XOR dataset (4 samples) then on a 2D spiral dataset.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def relu(z: FloatArray) -> FloatArray:
    return np.maximum(0.0, z)


def relu_grad(z: FloatArray) -> FloatArray:
    """Elementwise derivative of ReLU."""
    return (z > 0).astype(np.float64)


def sigmoid(z: FloatArray) -> FloatArray:
    # Numerically stable: split on sign of z
    pos = z >= 0
    result = np.empty_like(z)
    result[pos]  = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    result[~pos] = exp_z / (1.0 + exp_z)
    return result


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class TwoLayerNet:
    """
    2-layer MLP: input -> ReLU -> sigmoid output.
    All parameters stored as NumPy arrays.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)   # He initialization for ReLU
        self.W1: FloatArray = rng.normal(0, scale, (in_dim, hidden))
        self.b1: FloatArray = np.zeros(hidden)
        self.W2: FloatArray = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, out_dim))
        self.b2: FloatArray = np.zeros(out_dim)

        # Cache from forward pass needed for backward
        self._X: FloatArray | None = None
        self._Z1: FloatArray | None = None
        self._A1: FloatArray | None = None
        self._Z2: FloatArray | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: FloatArray) -> FloatArray:
        """Returns raw logits (pre-sigmoid) of shape (N, out_dim)."""
        self._X  = X
        self._Z1 = X @ self.W1 + self.b1       # (N, H)
        self._A1 = relu(self._Z1)               # (N, H)
        self._Z2 = self._A1 @ self.W2 + self.b2 # (N, out_dim)
        return self._Z2

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def bce_loss(self, logits: FloatArray, y: FloatArray) -> float:
        """Binary cross-entropy with numerically stable log."""
        probs = sigmoid(logits)
        eps = 1e-8
        loss = -np.mean(
            y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)
        )
        return float(loss)

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, y: FloatArray) -> dict[str, FloatArray]:
        """
        Compute gradients for all parameters.
        Returns dict with keys W1, b1, W2, b2.
        Assumes forward() was called immediately before.
        """
        assert self._X is not None, "Call forward() first."
        N = self._X.shape[0]

        # Gradient of BCE loss w.r.t. Z2 (sigmoid + BCE combined)
        dZ2 = (sigmoid(self._Z2) - y) / N       # (N, out_dim)

        dW2 = self._A1.T @ dZ2                   # (H, out_dim)
        db2 = dZ2.sum(axis=0)                    # (out_dim,)

        dA1 = dZ2 @ self.W2.T                    # (N, H)
        dZ1 = dA1 * relu_grad(self._Z1)          # (N, H)

        dW1 = self._X.T @ dZ1                    # (in_dim, H)
        db1 = dZ1.sum(axis=0)                    # (H,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ------------------------------------------------------------------
    # SGD update
    # ------------------------------------------------------------------

    def step(self, grads: dict[str, FloatArray], lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def predict(self, X: FloatArray) -> FloatArray:
        logits = self.forward(X)
        return (sigmoid(logits) > 0.5).astype(np.float64)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def xor_dataset() -> tuple[FloatArray, FloatArray]:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)
    return X, y


def spiral_dataset(n_per_class: int = 200, seed: int = 42) -> tuple[FloatArray, FloatArray]:
    """Binary spiral: two interleaved arms."""
    rng = np.random.default_rng(seed)
    def arm(label: int) -> tuple[FloatArray, FloatArray]:
        t = np.linspace(0, 1, n_per_class)
        theta = t * 3 * np.pi + label * np.pi
        r = t + rng.normal(0, 0.05, n_per_class)
        x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        y_arr = np.full((n_per_class, 1), float(label))
        return x, y_arr
    X0, y0 = arm(0)
    X1, y1 = arm(1)
    idx = rng.permutation(2 * n_per_class)
    X = np.vstack([X0, X1])[idx]
    y = np.vstack([y0, y1])[idx]
    return X.astype(np.float64), y.astype(np.float64)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_numpy(
    model: TwoLayerNet,
    X: FloatArray,
    y: FloatArray,
    lr: float = 0.01,
    epochs: int = 5000,
    log_every: int = 500,
) -> list[float]:
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        logits = model.forward(X)
        loss   = model.bce_loss(logits, y)
        grads  = model.backward(y)
        model.step(grads, lr)
        losses.append(loss)
        if epoch % log_every == 0:
            acc = (model.predict(X) == y).mean()
            log.info("epoch %5d  loss=%.6f  acc=%.4f", epoch, loss, acc)
    return losses


def main_numpy() -> None:
    log.info("--- XOR dataset ---")
    X_xor, y_xor = xor_dataset()
    net_xor = TwoLayerNet(in_dim=2, hidden=8, out_dim=1, seed=0)
    train_numpy(net_xor, X_xor, y_xor, lr=0.1, epochs=5000, log_every=1000)
    acc = (net_xor.predict(X_xor) == y_xor).mean()
    log.info("XOR final accuracy: %.4f (should be 1.0)", acc)

    log.info("--- Spiral dataset ---")
    X_sp, y_sp = spiral_dataset()
    net_sp = TwoLayerNet(in_dim=2, hidden=64, out_dim=1, seed=42)
    train_numpy(net_sp, X_sp, y_sp, lr=0.01, epochs=5000, log_every=1000)
    acc = (net_sp.predict(X_sp) == y_sp).mean()
    log.info("Spiral training accuracy: %.4f", acc)


if __name__ == "__main__":
    main_numpy()
```

### Part 2 — PyTorch re-implementation

```python
# day6_pytorch_nn.py
"""
Same 2-layer MLP re-implemented in PyTorch.
Compare final training accuracy and loss against the NumPy version.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")


class TwoLayerMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def spiral_tensors(n_per_class: int = 200, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    def arm(label: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 1, n_per_class)
        theta = t * 3 * np.pi + label * np.pi
        r = t + rng.normal(0, 0.05, n_per_class)
        x = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        return x, np.full((n_per_class, 1), float(label))
    X0, y0 = arm(0)
    X1, y1 = arm(1)
    idx = rng.permutation(2 * n_per_class)
    X = np.vstack([X0, X1])[idx].astype(np.float32)
    y = np.vstack([y0, y1])[idx].astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def train_pytorch(epochs: int = 5000, lr: float = 0.01) -> None:
    torch.manual_seed(42)
    X, y = spiral_tensors()
    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=len(dataset))  # full-batch to match NumPy

    model     = TwoLayerMLP(2, 64, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
        if epoch % 1000 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(X)
                final_loss = criterion(logits, y).item()
                acc = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean().item()
            log.info("epoch %5d  loss=%.6f  acc=%.4f", epoch, final_loss, acc)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        acc = ((torch.sigmoid(logits) > 0.5).float() == y).float().mean().item()
    log.info("PyTorch spiral training accuracy: %.4f", acc)


if __name__ == "__main__":
    train_pytorch()
```

Run both scripts:

```bash
python day6_numpy_nn.py
python day6_pytorch_nn.py
```

Both should reach ~95%+ training accuracy on the spiral after 5000 epochs. The NumPy and PyTorch numbers should be close but not identical — random seed handling and floating-point ordering differ.

---

## Exercises

**Challenge 1 — Gradient check**: Implement a numerical gradient checker for `TwoLayerNet`. For each parameter, perturb it by `eps=1e-5`, compute `(f(theta+eps) - f(theta-eps)) / (2*eps)`, and compare against the analytical gradient from `backward()`. The relative difference should be below `1e-5` for all parameters. This is the gold standard for verifying backprop implementations.

**Challenge 2 — Momentum from scratch**: Extend `TwoLayerNet` with SGD + momentum. Add a `velocities` dict in `__init__` (matching parameter shapes, initialized to zero). In `step()`, compute `v = momentum * v_prev + grad` then `param -= lr * v`. Verify that training loss decreases faster than vanilla SGD on the spiral dataset.

**Challenge 3 — PyTorch gradient inspection**: After one backward pass in `train_pytorch`, print the `.grad` attribute of `model.fc1.weight` and `model.fc2.weight`. Then manually compute the same gradients using the NumPy formulas. Verify the shapes and that the values are close (within floating-point precision). You will need to match the He initialization weights manually or set identical starting weights in both.

(Solutions: ../solutions/week-03/day-6.md)

---

## Self-check

- [ ] I can derive the gradient of BCE + sigmoid w.r.t. the pre-sigmoid logit from scratch.
- [ ] I can implement a full forward pass + backward pass in NumPy for a 2-layer network without reference.
- [ ] I can explain why dividing gradients by N is correct when using mean loss.
- [ ] I can write the PyTorch equivalent of the NumPy network and explain what each PyTorch call corresponds to in the manual version.
- [ ] I can implement a numerical gradient check and interpret its result.

---

## Supplementary links

- Backpropagation notes (CS231n): https://cs231n.github.io/optimization-2/
- He initialization paper: https://arxiv.org/abs/1502.01852
- "Yes you should understand backprop" (Karpathy): https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
- NumPy broadcasting guide: https://numpy.org/doc/stable/user/basics.broadcasting.html
- PyTorch autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
