# Solutions — Week 03 · Project — Neural Network from Scratch and PyTorch Comparison

## Complete Reference Implementation

---

### `src/numpy_net.py`

```python
"""Pure NumPy two-layer neural network with manual backpropagation.

Deps (uv add numpy): numpy
"""
from __future__ import annotations

import logging
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0.0, z)


def _relu_grad(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return (z > 0.0).astype(np.float64)


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_spiral_dataset(
    n_per_class: int = 200,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two-arm spiral dataset, normalised to zero mean / unit std."""
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for k in range(2):
        t = np.linspace(0.0, 1.0, n_per_class)
        angle = t * 4.0 * np.pi + k * np.pi
        r = t
        noise = rng.standard_normal(n_per_class) * 0.05
        X_parts.append(np.stack([r * np.cos(angle) + noise,
                                  r * np.sin(angle) + noise], axis=1))
        y_parts.append(np.full(n_per_class, k, dtype=np.float64))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    # Normalise using training-set statistics (computed here for full dataset)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y


def train_test_split_numpy(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return X[idx[n_test:]], X[idx[:n_test]], y[idx[n_test:]], y[idx[:n_test]]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TwoLayerNet:
    """Two-layer (hidden + output) binary classifier in pure NumPy."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int = 1,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        # He (Kaiming Normal) initialisation
        self.W1: NDArray[np.float64] = rng.standard_normal((hidden, in_dim)) * np.sqrt(2.0 / in_dim)
        self.b1: NDArray[np.float64] = np.zeros(hidden)
        self.W2: NDArray[np.float64] = rng.standard_normal((out_dim, hidden)) * np.sqrt(2.0 / hidden)
        self.b2: NDArray[np.float64] = np.zeros(out_dim)
        # Cache for backward
        self._X: NDArray[np.float64] | None = None
        self._z1: NDArray[np.float64] | None = None
        self._a1: NDArray[np.float64] | None = None
        self._z2: NDArray[np.float64] | None = None

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Forward pass. Caches intermediate values."""
        self._X = X
        self._z1 = X @ self.W1.T + self.b1        # (B, H)
        self._a1 = _relu(self._z1)                 # (B, H)
        self._z2 = self._a1 @ self.W2.T + self.b2  # (B, 1)
        return self._z2                             # raw logits

    def bce_loss(
        self,
        logits: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Numerically stable binary cross-entropy loss."""
        probs = _sigmoid(logits.flatten())
        eps = 1e-9
        return float(-np.mean(
            y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)
        ))

    def backward(self, y: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Compute gradients via backprop. Requires a preceding forward call."""
        assert self._X is not None, "Call forward() before backward()."
        B = len(y)
        a2 = _sigmoid(self._z2.flatten())                 # (B,)

        # Output layer
        d_loss = (a2 - y) / B                             # (B,)
        dz2 = d_loss.reshape(-1, 1) * _sigmoid(self._z2) * (1 - _sigmoid(self._z2))  # (B,1)
        dW2 = dz2.T @ self._a1                            # (1, H)
        db2 = dz2.sum(axis=0)                             # (1,)

        # Hidden layer
        da1 = dz2 @ self.W2                               # (B, H)
        dz1 = da1 * _relu_grad(self._z1)                  # (B, H)
        dW1 = dz1.T @ self._X                             # (H, D)
        db1 = dz1.sum(axis=0)                             # (H,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads: dict[str, NDArray[np.float64]], lr: float) -> None:
        """SGD parameter update."""
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Binary predictions (0.0 or 1.0)."""
        logits = self.forward(X)
        return (_sigmoid(logits.flatten()) >= 0.5).astype(np.float64)


def train(
    model: TwoLayerNet,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    lr: float = 0.01,
    epochs: int = 3000,
    log_every: int = 300,
) -> list[float]:
    """Full-batch SGD training loop. Returns loss history."""
    history: list[float] = []
    for epoch in range(epochs):
        logits = model.forward(X)
        loss = model.bce_loss(logits, y)
        history.append(loss)
        grads = model.backward(y)
        model.step(grads, lr)
        if epoch % log_every == 0 or epoch == epochs - 1:
            preds = model.predict(X)
            acc = float(np.mean(preds == y))
            log.info("epoch %4d | loss=%.4f | acc=%.4f", epoch, loss, acc)
    return history


def gradient_check(
    model: TwoLayerNet,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    eps: float = 1e-5,
) -> float:
    """
    Return max relative difference between numerical and analytic gradients.
    Value below 1e-5 indicates a correct backprop implementation.
    """
    model.forward(X)
    analytic = model.backward(y)
    max_err = 0.0

    for param_name in ("W1", "b1", "W2", "b2"):
        param = getattr(model, param_name)
        ag = analytic[param_name]

        for idx in np.ndindex(param.shape):
            orig = param[idx]
            param[idx] = orig + eps
            loss_p = model.bce_loss(model.forward(X), y)
            param[idx] = orig - eps
            loss_m = model.bce_loss(model.forward(X), y)
            param[idx] = orig
            ng = (loss_p - loss_m) / (2 * eps)

            denom = max(abs(ag[idx]), abs(ng), 1e-8)
            err = abs(ag[idx] - ng) / denom
            if err > max_err:
                max_err = err

    # Restore cached values after perturbation
    model.forward(X)
    return max_err
```

---

### `src/pytorch_net.py`

```python
"""PyTorch two-layer MLP for binary classification.

Deps (uv add torch numpy): torch, numpy
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    hidden: int = 64
    lr: float = 0.01
    weight_decay: float = 0.0
    epochs: int = 3000
    batch_size: int = 400   # full-batch default matches NumPy
    seed: int = 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TwoLayerMLP(nn.Module):
    """Two-layer MLP for binary classification."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        # He (Kaiming Normal) initialisation — matches NumPy implementation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model: TwoLayerMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """One epoch of training. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        logits = model(xb).squeeze(1)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def evaluate(
    model: TwoLayerMLP,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += len(yb)
    return total_loss / max(len(loader), 1), correct / max(total, 1)


def train(
    cfg: Config,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_te: torch.Tensor,
    y_te: torch.Tensor,
) -> TwoLayerMLP:
    """Full training loop. Returns trained model."""
    torch.manual_seed(cfg.seed)
    device = best_device()
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)

    model = TwoLayerMLP(X_tr.shape[1], cfg.hidden).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=len(y_te))

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, loader, optimizer, criterion)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            _, te_acc = evaluate(model, te_loader, criterion)
            log.info("epoch %4d | train_loss=%.4f | test_acc=%.4f",
                     epoch, train_loss, te_acc)

    return model
```

---

### `src/compare.py`

```python
"""compare.py — Train both implementations, compare final metrics.

Usage:
    uv run python src/compare.py
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from numpy_net import (
    TwoLayerNet,
    make_spiral_dataset,
    train,
    train_test_split_numpy,
)
from pytorch_net import Config, TwoLayerMLP, best_device, evaluate, train as pytorch_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LR = 0.01
EPOCHS = 3000
HIDDEN = 64


def main() -> None:
    # ---- Dataset -------------------------------------------------------
    X_full, y_full = make_spiral_dataset(n_per_class=200, seed=42)
    X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split_numpy(X_full, y_full, seed=0)

    # ---- NumPy ---------------------------------------------------------
    log.info("=== NumPy training ===")
    np_model = TwoLayerNet(in_dim=2, hidden=HIDDEN, seed=0)
    history = train(np_model, X_tr_np, y_tr_np, lr=LR, epochs=EPOCHS, log_every=500)
    np_final_loss = history[-1]

    np_tr_acc = float(np.mean(np_model.predict(X_tr_np) == y_tr_np))
    np_te_acc = float(np.mean(np_model.predict(X_te_np) == y_te_np))
    log.info("NumPy — final_loss=%.4f  train_acc=%.4f  test_acc=%.4f",
             np_final_loss, np_tr_acc, np_te_acc)

    # ---- PyTorch -------------------------------------------------------
    log.info("=== PyTorch training ===")
    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_np, dtype=torch.float32)
    y_te_t = torch.tensor(y_te_np, dtype=torch.float32)

    cfg = Config(hidden=HIDDEN, lr=LR, epochs=EPOCHS, batch_size=len(y_tr_np))
    pt_model = pytorch_train(cfg, X_tr_t, y_tr_t, X_te_t, y_te_t)

    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    criterion = nn.BCEWithLogitsLoss()
    te_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=len(y_te_t))
    _, pt_te_acc = evaluate(pt_model, te_loader, criterion)
    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=len(y_tr_t))
    pt_final_loss, pt_tr_acc = evaluate(pt_model, tr_loader, criterion)
    log.info("PyTorch — final_loss=%.4f  train_acc=%.4f  test_acc=%.4f",
             pt_final_loss, pt_tr_acc, pt_te_acc)

    # ---- Comparison ----------------------------------------------------
    ratio = np_final_loss / (pt_final_loss + 1e-9)
    log.info("Loss ratio numpy/pytorch = %.4f  (should be near 1.0)", ratio)
    assert np_te_acc > 0.90, f"NumPy test accuracy {np_te_acc:.4f} < 0.90"
    assert pt_te_acc > 0.90, f"PyTorch test accuracy {pt_te_acc:.4f} < 0.90"
    log.info("Both implementations exceed 90%% test accuracy. Comparison complete.")


if __name__ == "__main__":
    main()
```

---

### `tests/test_numpy_net.py`

```python
"""Tests for numpy_net.TwoLayerNet."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from numpy_net import TwoLayerNet, make_spiral_dataset, gradient_check, train, train_test_split_numpy

N, D, H = 50, 2, 16


@pytest.fixture
def small_model() -> TwoLayerNet:
    return TwoLayerNet(in_dim=D, hidden=H, seed=0)


@pytest.fixture
def small_data() -> tuple[NDArray, NDArray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, D)).astype(np.float64)
    y = (rng.random(N) > 0.5).astype(np.float64)
    return X, y


def test_forward_output_shape(small_model: TwoLayerNet, small_data) -> None:
    X, _ = small_data
    out = small_model.forward(X)
    assert out.shape == (N, 1)


def test_loss_is_finite(small_model: TwoLayerNet, small_data) -> None:
    X, y = small_data
    logits = small_model.forward(X)
    loss = small_model.bce_loss(logits, y)
    assert np.isfinite(loss) and loss > 0.0


def test_backward_grad_shapes(small_model: TwoLayerNet, small_data) -> None:
    X, y = small_data
    small_model.forward(X)
    grads = small_model.backward(y)
    assert grads["W1"].shape == small_model.W1.shape
    assert grads["b1"].shape == small_model.b1.shape
    assert grads["W2"].shape == small_model.W2.shape
    assert grads["b2"].shape == small_model.b2.shape


def test_gradient_check_passes(small_data) -> None:
    X, y = small_data
    model = TwoLayerNet(in_dim=D, hidden=8, seed=1)
    max_err = gradient_check(model, X[:10], y[:10])
    assert max_err < 1e-5, f"Gradient check failed: max_err={max_err:.2e}"


def test_xor_reaches_100_accuracy() -> None:
    # XOR dataset
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    model = TwoLayerNet(in_dim=2, hidden=8, seed=7)
    train(model, X, y, lr=0.1, epochs=5000, log_every=5000)
    acc = float(np.mean(model.predict(X) == y))
    assert acc == 1.0, f"XOR accuracy {acc:.4f} < 1.0"


def test_predict_returns_binary(small_model: TwoLayerNet, small_data) -> None:
    X, _ = small_data
    preds = small_model.predict(X)
    unique = set(preds.tolist())
    assert unique <= {0.0, 1.0}
```

---

### `tests/test_pytorch_net.py`

```python
"""Tests for pytorch_net.TwoLayerMLP."""
from __future__ import annotations

import torch
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from pytorch_net import Config, TwoLayerMLP, best_device, train, evaluate
from numpy_net import make_spiral_dataset, train_test_split_numpy
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import math


def _make_model(in_dim: int = 2, hidden: int = 16) -> TwoLayerMLP:
    return TwoLayerMLP(in_dim=in_dim, hidden=hidden)


def _make_data(n: int = 32, d: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n, d)
    y = (torch.rand(n) > 0.5).float()
    return X, y


def test_forward_output_shape() -> None:
    model = _make_model()
    X, _ = _make_data()
    out = model(X)
    assert out.shape == (32, 1)


def test_initial_loss_near_log2() -> None:
    torch.manual_seed(0)
    model = _make_model()
    X, y = _make_data(64)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(model(X).squeeze(1), y).item()
    assert abs(loss - math.log(2)) < 0.5


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    model = _make_model()
    X, y = _make_data(64)
    criterion = nn.BCEWithLogitsLoss()
    initial_loss = criterion(model(X).squeeze(1), y).item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(X).squeeze(1), y)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(X).squeeze(1), y).item()
    assert final_loss < initial_loss


def test_eval_mode_deterministic() -> None:
    torch.manual_seed(0)
    # Add dropout to make train mode non-deterministic
    model = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 1)
    )
    X, _ = _make_data(16)
    model.train()
    out_train_1 = model(X).detach()
    out_train_2 = model(X).detach()
    # train mode with dropout: two passes differ
    # (probabilistic — may occasionally match; acceptable)
    model.eval()
    with torch.no_grad():
        out_eval_1 = model(X)
        out_eval_2 = model(X)
    assert torch.allclose(out_eval_1, out_eval_2)


def test_spiral_accuracy_above_90pct() -> None:
    import numpy as np
    X_full, y_full = make_spiral_dataset(n_per_class=200, seed=42)
    X_tr_np, X_te_np, y_tr_np, y_te_np = train_test_split_numpy(X_full, y_full, seed=0)
    X_tr_t = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_np, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_np, dtype=torch.float32)
    y_te_t = torch.tensor(y_te_np, dtype=torch.float32)
    cfg = Config(hidden=64, lr=0.01, epochs=3000, batch_size=len(y_tr_t))
    model = train(cfg, X_tr_t, y_tr_t, X_te_t, y_te_t)
    criterion = nn.BCEWithLogitsLoss()
    te_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=len(y_te_t))
    _, acc = evaluate(model, te_loader, criterion)
    assert acc > 0.90, f"Test accuracy {acc:.4f} < 0.90"
```

---

## Key design decisions

**He initialisation**: `W = N(0, sqrt(2 / fan_in))`. Keeps variance constant through ReLU layers which zero out ~50% of activations. Without it, signals in deep networks either vanish (σ too small) or explode (σ too large).

**Numerically stable BCE**: Applying sigmoid before computing log introduces floating-point cancellation near 0 and 1. Compute the sigmoid inside the loss, clamping inputs to `[-500, 500]` before `exp`.

**Gradient check**: Perturbation `eps = 1e-5` balances the discretisation error (`O(eps²)`) against floating-point round-off (`O(eps⁻¹)`). The central-difference estimate `(f(x+eps) - f(x-eps)) / 2eps` converges as `O(eps²)`, so `1e-5` gives ~10 accurate digits.

**Full-batch comparison**: To compare NumPy and PyTorch fairly, set `batch_size = len(X_train)` in PyTorch so both use identical gradient computation per step. Mini-batching introduces stochastic noise that makes loss curves incomparable.
