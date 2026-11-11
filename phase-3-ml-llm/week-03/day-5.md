# Phase 3 · Week 03 · Day 5 — nn.Module, Optimizers, and the Full Training Loop

> Objective: Build, train, and evaluate a PyTorch neural network using nn.Module, Adam/SGD, and proper regularization.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-1]], [[week-03/day-4]]

---

## Concept

A neural network in PyTorch is a Python class that inherits from `nn.Module`. It encapsulates:

- **Parameters** — tensors that require gradients, registered via `nn.Linear` and similar layers.
- **Forward logic** — what happens during inference, defined in `forward()`.
- **State** — buffers, training flags, device placement.

The training loop has four repeating steps:

1. **Forward pass** — compute predictions.
2. **Loss** — scalar measuring error.
3. **Backward pass** — `loss.backward()` populates `.grad` on every leaf tensor.
4. **Optimizer step** — update parameters using accumulated gradients, then zero them.

Regularization prevents overfitting by penalizing model complexity. Two practical techniques:

- **Dropout** — randomly zeroes activations during training (disabled during eval).
- **Weight decay (L2)** — adds a penalty proportional to the squared parameter norm, implemented directly inside most optimizers via the `weight_decay` argument.

---

## How it works internally

### nn.Module parameter registration

When you assign `self.fc1 = nn.Linear(in, out)` inside `__init__`, PyTorch wraps the weight and bias tensors in `nn.Parameter` objects and registers them. `model.parameters()` traverses a linked list of these registrations depth-first, which is why nested modules work automatically.

### Autograd and .grad accumulation

`loss.backward()` walks the computation graph in reverse, applying the chain rule. Each leaf tensor's `.grad` attribute is **accumulated**, not replaced. This means you must call `optimizer.zero_grad()` (or `optimizer.zero_grad(set_to_none=True)` for slightly better performance) before each backward pass, otherwise gradients from previous batches add to the current batch.

### SGD with momentum — update rule

```
v_t = momentum * v_{t-1} + grad_t
theta_t = theta_{t-1} - lr * v_t
```

Momentum builds a velocity vector that smooths noisy gradient directions.

### Adam — update rule

Adam maintains per-parameter first and second moment estimates:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t        # first moment (mean)
v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t**2      # second moment (variance)

m_hat = m_t / (1 - beta1**t)   # bias correction
v_hat = v_t / (1 - beta2**t)   # bias correction

theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

Defaults: `lr=1e-3`, `beta1=0.9`, `beta2=0.999`, `eps=1e-8`. The adaptive per-parameter learning rate makes Adam less sensitive to the global `lr` choice than SGD.

### .train() and .eval() modes

`model.train()` sets a flag read by `nn.Dropout` and `nn.BatchNorm`. In training mode, dropout randomly masks activations. In eval mode (`model.eval()`), dropout becomes a no-op and batch norm uses running statistics. Forgetting to call `.eval()` before validation is one of the most common PyTorch bugs.

---

## Worked example

Install once in the project:

```bash
uv add torch
```

```python
# day5_training_loop.py
"""
Full PyTorch training loop: 2-layer MLP on synthetic binary classification.
Demonstrates nn.Module, Adam optimizer, dropout, weight decay, and train/eval modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    hidden_size: int = 64
    dropout: float = 0.3
    weight_decay: float = 1e-4
    lr: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    seed: int = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Two-layer multi-layer perceptron for binary classification."""

    def __init__(self, in_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),   # raw logit; use BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_features) -> (B, 1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_dataset(n: int = 500, seed: int = 42) -> Tuple[TensorDataset, TensorDataset]:
    """Toy binary classification: two concentric rings."""
    rng = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, generator=rng) * 2 * torch.pi
    r = torch.where(
        torch.rand(n, generator=rng) > 0.5,
        0.5 + 0.1 * torch.randn(n, generator=rng),   # inner ring
        1.5 + 0.1 * torch.randn(n, generator=rng),   # outer ring
    )
    X = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    y = (r > 1.0).float().unsqueeze(1)

    split = int(0.8 * n)
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds   = TensorDataset(X[split:], y[split:])
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Train / eval utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg: TrainConfig | None = None) -> None:
    if cfg is None:
        cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    log.info("Config: %s", cfg)

    train_ds, val_ds = make_dataset(seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size)

    model = MLP(in_features=2, hidden=cfg.hidden_size, dropout=cfg.dropout)
    log.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # weight_decay implements L2 regularization inside Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f",
                epoch, cfg.epochs, train_loss, val_loss, val_acc,
            )

    log.info("Best val accuracy: %.3f", best_val_acc)


if __name__ == "__main__":
    main()
```

Run with:

```bash
python day5_training_loop.py
```

Expected output (approximate): val accuracy above 0.95 within 30 epochs on this toy dataset.

---

## Exercises

1. **Optimizer swap**: Replace `Adam` with `SGD(momentum=0.9)`. Keep everything else the same. Does it converge? Does it need a different `lr`? Log the best val accuracy for both and compare.

2. **Dropout sweep**: Train three variants with `dropout` set to `0.0`, `0.3`, and `0.6`. Plot (or log) train vs. val loss for each. Identify which value shows signs of underfitting vs. overfitting.

3. **Weight decay ablation**: Set `weight_decay=0.0` and `weight_decay=0.1`. How does the magnitude of `model.net[0].weight` change at the end of training? Print the Frobenius norm before and after for both runs.

4. **Train/eval mode bug**: Remove the `model.eval()` call in `evaluate`. What changes in the metrics across multiple runs? Why is this non-deterministic?

5. **Custom scheduler**: Add `torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)` and call `scheduler.step()` after each epoch. Log the current LR and observe how it affects convergence.

(Solutions: ../solutions/week-03/day-5.md)

---

## Self-check

- [ ] I can explain what `nn.Module.__init__` does when you assign a sub-layer.
- [ ] I can write the Adam update rule from memory (moments, bias correction, parameter step).
- [ ] I can implement a full train/eval loop without reference, including `.train()`/`.eval()` and `zero_grad()`.
- [ ] I can explain why dropout must be disabled during evaluation.
- [ ] I can describe what `weight_decay` does inside an optimizer and how it relates to L2 regularization.

---

## Supplementary links

- PyTorch `nn.Module` docs: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Adam paper (Kingma & Ba, 2014): https://arxiv.org/abs/1412.6980
- PyTorch optimizer docs: https://pytorch.org/docs/stable/optim.html
- Dropout paper (Srivastava et al., 2014): https://jmlr.org/papers/v15/srivastava14a.html
- "A Recipe for Training Neural Networks" (Karpathy): https://karpathy.github.io/2019/04/25/recipe/
