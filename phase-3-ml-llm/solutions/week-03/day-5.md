# Solutions — Week 03 · Day 5

## Exercise 1 — Custom nn.Module

```python
import logging
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class TwoLayerMLP(nn.Module):
    """Configurable 2-layer MLP with optional dropout and batch norm."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        dropout_p: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# Inspect the module
model = TwoLayerMLP(in_features=4, hidden_dim=8, out_features=2, dropout_p=0.2)
logger.info("Model:\n%s", model)

# Count parameters
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info("Trainable params: %d", n_params)

# Forward pass
x = torch.randn(5, 4)    # batch=5
out = model(x)
logger.info("Output shape: %s", out.shape)   # (5, 2)

# Named parameters
for name, param in model.named_parameters():
    logger.info("  %-20s shape=%s", name, list(param.shape))
```

---

## Exercise 2 — Adam optimizer step by step

```python
import logging
import math
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def adam_step_manual(
    param: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,       # 1st moment estimate (in-place modified)
    v: torch.Tensor,       # 2nd moment estimate (in-place modified)
    t: int,                # step count (1-indexed)
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Returns updated parameter, modifies m and v in place."""
    # Update biased first and second moment estimates
    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

    # Bias-correction factors
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t

    # Compute effective learning rate
    step_size = lr * math.sqrt(bc2) / bc1

    # Update parameter
    return param - step_size * m / (v.sqrt() + eps)


# Compare with PyTorch Adam on a simple quadratic
torch.manual_seed(0)
param_manual = torch.tensor([2.0, -1.5])
param_torch  = param_manual.clone().requires_grad_(True)

optimizer = torch.optim.Adam([param_torch], lr=1e-3)

m = torch.zeros_like(param_manual)
v = torch.zeros_like(param_manual)

for step in range(1, 6):
    # Target: minimize ||param - [1.0, 0.5]||^2
    target = torch.tensor([1.0, 0.5])
    grad = 2.0 * (param_manual - target)

    param_manual = adam_step_manual(param_manual, grad, m, v, t=step)

    # PyTorch side
    optimizer.zero_grad()
    loss = ((param_torch - target) ** 2).sum()
    loss.backward()
    optimizer.step()

    logger.info("step=%d  manual=%s  torch=%s",
                step,
                param_manual.numpy().round(5),
                param_torch.detach().numpy().round(5))
```

Adam algorithm summary:
1. m_t = beta1 * m_{t-1} + (1 - beta1) * g_t         (1st moment: gradient EMA)
2. v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2       (2nd moment: squared gradient EMA)
3. m_hat = m_t / (1 - beta1^t)                        (bias-corrected)
4. v_hat = v_t / (1 - beta2^t)                        (bias-corrected)
5. theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

---

## Exercise 3 — Training loop with loss logging

```python
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")


def make_spiral_data(
    n_per_class: int = 200,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    for k in range(n_classes):
        t = torch.linspace(0, 4, n_per_class)
        angle = t * 4 + k * (2 * 3.14159 / n_classes)
        r = t / 4 + 0.05 * torch.randn(n_per_class)
        X_list.append(torch.stack([r * torch.cos(angle), r * torch.sin(angle)], dim=1))
        y_list.append(torch.full((n_per_class,), k, dtype=torch.long))
    return torch.cat(X_list), torch.cat(y_list)


def train(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int,
    device: torch.device,
) -> list[float]:
    model.train()
    history: list[float] = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(xb)
        avg_loss = epoch_loss / total
        acc = correct / total
        history.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            logger.info("epoch=%3d  loss=%.4f  acc=%.3f", epoch + 1, avg_loss, acc)
    return history


def main() -> None:
    device = torch.device("cpu")
    X, y = make_spiral_data(n_per_class=200, n_classes=3)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 3),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    history = train(model, loader, optimizer, criterion, n_epochs=50, device=device)
    logger.info("Final loss: %.4f", history[-1])


if __name__ == "__main__":
    main()
```

---

## Exercise 4 — Dropout effect demonstration

```python
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class MLPWithDropout(nn.Module):
    def __init__(self, dropout_p: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p=dropout_p),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = MLPWithDropout(dropout_p=0.5)
x = torch.randn(4, 2)

# Training mode: dropout is active
model.train()
out_train_1 = model(x)
out_train_2 = model(x)
logger.info("Train mode (different each call):")
logger.info("  pass 1: %s", out_train_1.detach().round(decimals=4))
logger.info("  pass 2: %s", out_train_2.detach().round(decimals=4))
logger.info("  identical: %s", torch.allclose(out_train_1, out_train_2))  # False

# Eval mode: dropout is disabled, outputs are deterministic
model.eval()
with torch.no_grad():
    out_eval_1 = model(x)
    out_eval_2 = model(x)
logger.info("Eval mode (deterministic):")
logger.info("  pass 1: %s", out_eval_1.round(decimals=4))
logger.info("  pass 2: %s", out_eval_2.round(decimals=4))
logger.info("  identical: %s", torch.allclose(out_eval_1, out_eval_2))  # True
```

Key points:
- `model.train()` activates dropout (each call zeros different units randomly).
- `model.eval()` disables dropout — always call before inference.
- Dropout scales surviving activations by `1/(1-p)` during training so the
  expected value matches eval (inverted dropout). PyTorch does this automatically.
- Regularization effect: forces redundant representations, reduces co-adaptation.

---

## Exercise 5 — Weight decay comparison

```python
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def generate_overfit_data(
    n_train: int = 30,
    n_test: int = 200,
    noise: float = 0.3,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    X_train = torch.linspace(-3, 3, n_train).unsqueeze(1)
    y_train = torch.sin(X_train) + noise * torch.randn_like(X_train)
    X_test = torch.linspace(-3, 3, n_test).unsqueeze(1)
    y_test = torch.sin(X_test)
    return X_train, y_train, X_test, y_test


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def run_experiment(weight_decay: float, n_epochs: int = 500) -> tuple[float, float]:
    X_train, y_train, X_test, y_test = generate_overfit_data()
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=30, shuffle=True)

    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = criterion(model(X_train), y_train).item()
        test_loss = criterion(model(X_test), y_test).item()
    return train_loss, test_loss


logger.info("%-12s  %-12s  %-12s  %-12s", "weight_decay", "train_MSE", "test_MSE", "gap")
for wd in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
    tr, te = run_experiment(wd)
    logger.info("%-12.4f  %-12.4f  %-12.4f  %-12.4f", wd, tr, te, te - tr)
```

Expected result pattern:
| weight_decay | train_MSE | test_MSE | gap  |
|---|---|---|---|
| 0.0    | very low  | high   | large  (overfit)   |
| 1e-4   | low       | medium | smaller            |
| 1e-3   | medium    | lower  | smaller still      |
| 1e-2   | higher    | lowest | tight              |
| 1e-1   | high      | high   | under-regularized  |

Weight decay (L2 regularization) adds `lambda * ||w||^2` to the loss, which
penalizes large weights and pushes the model toward simpler functions. It is
equivalent to Gaussian prior on weights in a Bayesian view. The `weight_decay`
parameter in Adam applies it as an additive gradient term each step.
```
