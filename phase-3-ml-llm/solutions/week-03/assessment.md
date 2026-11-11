# Assessment Answer Key — Week 03: Neural Networks & PyTorch

---

## Part 1 — Conceptual Questions (5 questions, 10 points each)

---

### Q1. Explain the vanishing gradient problem. Why does sigmoid cause it and how does ReLU mitigate it?

**Answer:**

The vanishing gradient problem occurs during backpropagation when gradients of the
loss w.r.t. early-layer weights become extremely small, effectively preventing those
layers from learning.

**Why sigmoid causes it:**
The sigmoid derivative is s'(x) = s(x)(1 - s(x)), which has a maximum value of
0.25 at x = 0 and approaches 0 for large |x| (saturation). During backprop, the
gradient passes through L multiplicative terms — one sigmoid derivative per layer:

    dL/dW1 ∝ s'(z_L) * s'(z_{L-1}) * ... * s'(z_1)

Each factor ≤ 0.25. For L = 10: 0.25^10 ≈ 9.5e-7. Weights in early layers
receive essentially zero gradient signal and do not update.

**Why ReLU mitigates it:**
The ReLU derivative is 1 for x > 0 and 0 for x ≤ 0. For active (positive) units,
multiplying L ReLU derivatives gives 1 — gradient magnitude is preserved across
layers. The dead neuron problem (ReLU derivative = 0 when x ≤ 0) is a different
failure mode, partially addressed by Leaky ReLU or ELU.

**Full marks require:** gradient product argument with numbers, saturation
explanation, and ReLU derivative = 1 argument.

---

### Q2. Derive the gradient of cross-entropy loss with respect to the softmax input (logits). Show that it simplifies to (p - y) where p is the predicted probability and y is the one-hot label.

**Answer:**

Let z ∈ R^C be logits, p_i = softmax(z)_i = exp(z_i) / Σ_j exp(z_j).
Cross-entropy loss for true class k: L = -log(p_k).

We want dL/dz_i for all i.

**Case i = k (correct class):**

    dL/dz_k = d/dz_k [-log(p_k)]
            = -(1/p_k) * dp_k/dz_k

    dp_k/dz_k = p_k(1 - p_k)    (standard softmax derivative, same index)

    dL/dz_k = -(1 - p_k) = p_k - 1 = p_k - y_k   (since y_k = 1)

**Case i ≠ k (incorrect class):**

    dL/dz_i = -(1/p_k) * dp_k/dz_i

    dp_k/dz_i = -p_k * p_i    (softmax cross-term derivative)

    dL/dz_i = p_i = p_i - 0 = p_i - y_i   (since y_i = 0)

**Combined:** dL/dz_i = p_i - y_i for all i, i.e., dL/dz = p - y.

This remarkably simple gradient is why softmax + cross-entropy is the standard
combination for multi-class output layers.

---

### Q3. What is the difference between model.train() and model.eval() in PyTorch? Name two layers that behave differently between these modes.

**Answer:**

`model.train()` and `model.eval()` switch the module's training flag, which
changes the behavior of certain layers:

**Dropout:**
- `train()`: randomly zeros activations with probability p, scales surviving
  activations by 1/(1-p) (inverted dropout) so expected value is preserved.
- `eval()`: identity function — all activations pass through unchanged.
  Forgetting to switch to eval() causes noisy, non-deterministic inference.

**BatchNorm1d / BatchNorm2d:**
- `train()`: computes mean and variance from the current mini-batch; uses them
  for normalization; updates running mean/variance via exponential moving average.
- `eval()`: uses the stored running statistics (accumulated during training)
  for normalization; batch statistics are ignored. This ensures deterministic,
  batch-size-independent inference.

**Rule of thumb:** always call `model.eval()` before any inference or validation
loop, and `model.train()` when resuming training.

---

### Q4. Explain the role of the learning rate. What happens when it is too large vs too small? Describe one method to set it automatically.

**Answer:**

The learning rate (lr) controls how far parameters move in the direction of the
negative gradient at each step: θ = θ - lr * g.

**Too large:** updates overshoot the minimum; loss oscillates or diverges to NaN.
Gradients explode because large steps generate large next gradients.

**Too small:** updates are tiny; convergence is extremely slow; model may get
stuck in flat regions or local minima before reaching a good solution.

**Automatic methods:**

1. **Learning rate schedulers:** Start with a moderate lr and decay it over
   training (StepLR, CosineAnnealingLR, ReduceLROnPlateau). PyTorch provides
   `torch.optim.lr_scheduler` for this.

2. **Adaptive optimizers (Adam, RMSProp):** Maintain per-parameter running
   statistics of gradient magnitudes and scale each parameter's effective lr
   inversely proportional to its gradient history. Parameters with large
   historical gradients get smaller effective lr; those with small gradients
   get larger effective lr.

3. **Learning rate range test (LR Finder):** Train for a few steps with lr
   increasing exponentially from 1e-7 to 10. Plot loss vs. lr. Choose the lr
   just before loss begins to rise — the "sweet spot" of fast descent without
   instability (Leslie Smith, 2015).

---

### Q5. What is weight decay (L2 regularization)? Write the regularized loss function and explain how it prevents overfitting.

**Answer:**

Weight decay adds a penalty proportional to the squared norm of all weights:

    L_reg = L_data + (lambda/2) * ||W||^2_F
          = (1/N) * Σ loss(f(x_i), y_i)  +  (lambda/2) * Σ_l ||W_l||^2_F

where lambda is the regularization coefficient.

The gradient of the regularized loss w.r.t. W is:

    dL_reg/dW = dL_data/dW + lambda * W

Each gradient step therefore shrinks the weights slightly:

    W_new = W - lr * (dL_data/dW + lambda * W)
          = W * (1 - lr * lambda) - lr * dL_data/dW

**How it prevents overfitting:**
1. **Shrinks weights toward zero**, penalizing large weights that encode
   memorized noise patterns in the training data.
2. **Smooths the decision boundary** — small weights mean the output changes
   slowly with the input, discouraging sharp, data-specific features.
3. **Bayesian interpretation:** equivalent to placing a zero-mean Gaussian prior
   on weights; MAP estimation under this prior gives the L2-regularized solution.

In PyTorch, pass `weight_decay=lambda` to any optimizer constructor; it applies
the decay automatically without modifying the loss function explicitly.

---

## Part 2 — Coding Challenge

Build a 2-layer MLP in PyTorch that achieves ≥ 90% accuracy on a toy
classification dataset (moons). Include pytest tests.

### Solution code

```python
"""
week03_assessment.py

Two-layer MLP that achieves >= 90% test accuracy on the moons dataset.
Runnable as a script; also importable for pytest.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer MLP for binary classification."""

    def __init__(
        self,
        in_features: int = 2,
        hidden_dim: int = 32,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2),   # 2 logits for CrossEntropyLoss
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_moons(
    n_samples: int = 1000,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two-moons dataset without sklearn dependency."""
    rng = np.random.default_rng(seed)
    n_each = n_samples // 2
    t = np.linspace(0, np.pi, n_each)
    X0 = np.stack([np.cos(t), np.sin(t)], axis=1)
    X1 = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], axis=1)
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(n_each), np.ones(n_each)])
    X += noise * rng.standard_normal(X.shape)
    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx].astype(np.int64)


def split_normalize(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(seed)
    n_test = int(len(X) * test_frac)
    idx = rng.permutation(len(X))
    train_idx, test_idx = idx[n_test:], idx[:n_test]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    mu = X_tr.mean(axis=0)
    sigma = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mu) / sigma
    X_te = (X_te - mu) / sigma
    return (
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(X_te),
        torch.from_numpy(y_te),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    model: nn.Module
    train_losses: list[float]
    test_accuracies: list[float]


def train(
    X_train: Tensor,
    y_train: Tensor,
    X_test: Tensor,
    y_test: Tensor,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> TrainResult:
    torch.manual_seed(seed)
    device = torch.device("cpu")

    model = MLP(in_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_test_d = X_test.to(device)
    y_test_d = y_test.to(device)

    train_losses: list[float] = []
    test_accuracies: list[float] = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            avg_loss = epoch_loss / len(X_train)
            preds = model(X_test_d).argmax(dim=1)
            acc = (preds == y_test_d).float().mean().item()

        train_losses.append(avg_loss)
        test_accuracies.append(acc)

        if (epoch + 1) % 20 == 0:
            logger.info("epoch=%3d  loss=%.4f  test_acc=%.3f",
                        epoch + 1, avg_loss, acc)

    return TrainResult(model=model, train_losses=train_losses,
                       test_accuracies=test_accuracies)


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

def test_model_output_shape() -> None:
    model = MLP(in_features=2, hidden_dim=16)
    x = torch.randn(8, 2)
    out = model(x)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"


def test_model_forward_no_nan() -> None:
    torch.manual_seed(0)
    model = MLP()
    x = torch.randn(32, 2)
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in forward pass"
    assert not torch.isinf(out).any(), "Inf in forward pass"


def test_loss_decreases() -> None:
    """Loss after 20 epochs should be lower than initial loss."""
    X, y = make_moons(n_samples=400, seed=0)
    X_tr, y_tr, X_te, y_te = split_normalize(X, y, seed=0)
    result = train(X_tr, y_tr, X_te, y_te, n_epochs=20, seed=0)
    assert result.train_losses[-1] < result.train_losses[0], (
        f"Loss did not decrease: {result.train_losses[0]:.4f} -> {result.train_losses[-1]:.4f}"
    )


def test_achieves_90_percent_accuracy() -> None:
    """The model should reach >= 90% test accuracy within 100 epochs."""
    X, y = make_moons(n_samples=1000, noise=0.2, seed=42)
    X_tr, y_tr, X_te, y_te = split_normalize(X, y, seed=42)
    result = train(X_tr, y_tr, X_te, y_te, n_epochs=100, seed=42)
    final_acc = result.test_accuracies[-1]
    assert final_acc >= 0.90, (
        f"Expected >= 90% accuracy, got {final_acc:.1%}"
    )


def test_eval_mode_deterministic() -> None:
    """Predictions should be identical across two eval-mode forward passes."""
    model = MLP(dropout_p=0.5)
    model.eval()
    x = torch.randn(16, 2)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2), "Eval mode outputs are not deterministic"


def test_no_grad_outside_training() -> None:
    """Inference tensors should not require grad when wrapped in no_grad."""
    model = MLP()
    model.eval()
    x = torch.randn(8, 2)
    with torch.no_grad():
        out = model(x)
    assert not out.requires_grad, "Output should not require grad during inference"


def test_gradient_flows_to_all_layers() -> None:
    """All parameter gradients should be non-None and non-zero after a backward pass."""
    model = MLP()
    x = torch.randn(16, 2)
    y = torch.randint(0, 2, (16,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = make_moons(n_samples=1000, noise=0.2, seed=42)
    X_tr, y_tr, X_te, y_te = split_normalize(X, y, seed=42)
    result = train(X_tr, y_tr, X_te, y_te, n_epochs=100, seed=42)
    final_acc = result.test_accuracies[-1]
    logger.info("Final test accuracy: %.1f%%", final_acc * 100)
    assert final_acc >= 0.90, f"Did not reach 90%: {final_acc:.1%}"
    logger.info("PASS — >= 90%% accuracy achieved")
```

---

## Part 3 — Grade Rubric

| Dimension | Points | Criteria for full marks |
|---|---|---|
| Conceptual Q1 (vanishing gradient) | 10 | Gradient product argument with numbers; saturation explanation; ReLU derivative = 1 |
| Conceptual Q2 (softmax CE gradient) | 10 | Correct derivation of both i=k and i≠k cases; final result p-y stated |
| Conceptual Q3 (train/eval modes) | 10 | Two correct layers (Dropout + BatchNorm); behavior in each mode described correctly |
| Conceptual Q4 (learning rate) | 10 | Too large = diverge; too small = slow; one automatic method described with mechanism |
| Conceptual Q5 (weight decay) | 10 | Regularized loss formula written; shrinkage update derived; overfitting mechanism explained |
| Code: model architecture | 10 | Two layers with ReLU, Dropout; correct output size; He init |
| Code: training loop | 15 | zero_grad + backward + step; model.train()/eval() used correctly; loss logged |
| Code: achieves >= 90% accuracy | 15 | Test passes (test_achieves_90_percent_accuracy) |
| Code: pytest tests (6 tests) | 10 | All 6 test functions present; each tests a distinct property |
| Code: type hints + logging | 10 | All functions typed; `logging` used instead of `print` |
| **Total** | **100** | |

**Passing threshold:** 70/100.
**Distinction:** 85/100.

---

## Grading notes

- Partial credit: Q2 and Q5 — award 5/10 if the correct formula is stated but
  derivation is missing or incomplete.
- Code: deduct 5 points for each missing `model.eval()` / `model.train()` call
  in the appropriate location, and 5 points for using `print` instead of `logging`.
- The `make_moons` helper in the solution avoids sklearn to keep the solution
  self-contained. Accept solutions that import `sklearn.datasets.make_moons` —
  this is valid and common practice.
- test_achieves_90_percent_accuracy may take 5-10 seconds on CPU — this is
  expected and should not be penalized.
```
