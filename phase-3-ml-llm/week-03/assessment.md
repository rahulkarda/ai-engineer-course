# Week 03 Assessment — Neural Networks and PyTorch

> Prereqs: [[day-5]], [[day-6]], [[day-7]]  ·  Time: ~1.5 hr  ·  Pass: >= 75%

---

## Conceptual (interview-style)

Answer in 3-5 sentences each. No code required.

**Q1.** Explain the role of `optimizer.zero_grad()` in the PyTorch training loop. What goes wrong if you omit it, and why does the problem compound over batches?

**Q2.** Derive the Adam update rule from scratch: define the two moment estimates, explain bias correction, and write the final parameter update. Why does Adam often converge faster than SGD with a fixed learning rate?

**Q3.** A model achieves 99% training accuracy and 60% validation accuracy. List three regularization techniques you could apply. For each, describe mechanistically how it reduces overfitting.

**Q4.** You call `model(X)` before setting `model.eval()`, and your validation loss is higher than expected. Which components of the network behave differently in training vs. eval mode? Why does this produce incorrect validation metrics?

**Q5.** Explain the difference between `BCELoss` and `BCEWithLogitsLoss`. Under what numerical conditions does using `BCELoss(sigmoid(logits))` produce incorrect results, and how does the fused version avoid the problem?

---

## Coding challenge

### Task: Train a 2-layer MLP to >90% accuracy on toy data

**Dataset**: Use the spiral dataset from Day 6 (`n_per_class=200`, `seed=42`). Split 80/20 train/val.

**Requirements**:

1. Define a `TwoLayerMLP` class inheriting from `nn.Module` with configurable `in_dim`, `hidden`, and `out_dim`.
2. Use `nn.BCEWithLogitsLoss`.
3. Use `Adam` optimizer with `weight_decay >= 1e-5`.
4. Implement a `train_epoch` function and an `evaluate` function as separate functions (not methods).
5. Implement a full training loop that logs train loss and val accuracy every 10 epochs using `logging`, not `print`.
6. Achieve `>= 0.90` validation accuracy within 100 epochs.
7. Write a `pytest` test suite in `tests/test_mlp.py` with the following test functions:
   - `test_forward_shape`: asserts output shape is `(N, 1)` for input `(N, 2)`.
   - `test_loss_finite`: asserts initial loss is finite and near `log(2)` (within 0.5).
   - `test_training_reduces_loss`: asserts loss after 20 epochs is lower than initial loss.
   - `test_val_accuracy_above_threshold`: asserts val accuracy exceeds 0.90 after full training.

**Acceptance criteria**:

- [ ] Code runs without error: `python solution.py`
- [ ] Tests pass: `pytest tests/test_mlp.py -v`
- [ ] Val accuracy logged >= 0.90 in the final epoch
- [ ] No `print` statements (use `logging`)
- [ ] All functions and class methods have type hints
- [ ] No hardcoded magic numbers — use a config dataclass or named constants

**Starter structure** (fill in the blanks):

```python
# assessment_solution.py
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


@dataclass
class Config:
    hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    seed: int = 42


class TwoLayerMLP(nn.Module):
    # TODO: implement
    ...


def make_spiral_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    # TODO: return train_loader, val_loader
    ...


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    # TODO: return average train loss
    ...


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    # TODO: return (avg_loss, accuracy)
    ...


def main(cfg: Config | None = None) -> None:
    # TODO: full training loop, log every 10 epochs
    ...


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    main()
```

---

## Rubric (pass >= 75%)

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Independent implementation | 40% | Model, train loop, and evaluate written without copy-paste from lessons; correct shapes and dtypes throughout |
| Conceptual understanding | 25% | Conceptual questions answered accurately; code choices (optimizer, loss fn) can be justified verbally |
| Debugging and problem-solving | 20% | Code handles edge cases (eval mode, zero_grad, device placement); can describe how to diagnose a NaN loss |
| Code quality and testing | 15% | Type hints present, logging used, config dataclass used, pytest tests pass |

**Scoring guide**:

- 90-100%: All conceptual questions fully correct, tests pass, accuracy >90%, clean code.
- 75-89%: Minor gaps in one conceptual question, tests pass, accuracy >=90%.
- 60-74%: One conceptual question mostly wrong OR accuracy <90% but >80%, tests partially pass.
- Below 60%: Multiple conceptual gaps OR training loop has fundamental bugs (missing zero_grad, missing eval mode, etc.).

Answer key: ../solutions/week-03/assessment.md
