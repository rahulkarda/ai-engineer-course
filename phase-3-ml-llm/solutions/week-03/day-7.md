# Solutions — Week 03 · Day 7

## Exercise 1 — GPU / device basics

### 1a. Check device availability and move tensors

```python
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
logger.info("Using device: %s", device)

# Create tensor on CPU, move to device
x_cpu = torch.randn(3, 4)
x_dev = x_cpu.to(device)
logger.info("x_cpu device: %s", x_cpu.device)
logger.info("x_dev device: %s", x_dev.device)

# Alternative: create directly on device
y = torch.zeros(3, 4, device=device)
logger.info("y device    : %s", y.device)

# Move model to device
model = nn.Linear(4, 2)
model = model.to(device)
logger.info("model weight device: %s", model.weight.device)

# All tensors in an operation must be on the same device
try:
    z = x_cpu + x_dev   # RuntimeError if devices differ
except RuntimeError as e:
    logger.info("Mixed-device error: %s", str(e)[:80])

# Correct: both on device
z = x_dev + y
logger.info("z device: %s", z.device)
```

### 1b. Write device-agnostic training code

```python
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def device_agnostic_train(
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 50,
) -> nn.Module:
    """Training loop that works on CPU, CUDA, or MPS without modification."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Training on: %s", device)

    # Move data to device once
    X = X.to(device)
    y = y.to(device)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, int(y.max().item()) + 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            # Batches from DataLoader are already on the right device
            # because the underlying tensors were moved above
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == y).float().mean().item()
            logger.info("epoch=%3d  acc=%.3f", epoch + 1, acc)

    return model
```

### 1c. Transfer overhead demonstration (CPU only, conceptual)

```python
import logging
import time
import torch

logger = logging.getLogger(__name__)


def benchmark_device_ops(size: int = 1000) -> None:
    """Show that unnecessary CPU<->GPU transfers are expensive."""
    # On CPU, repeated .numpy() calls that force synchronization
    x = torch.randn(size, size)

    t0 = time.perf_counter()
    for _ in range(100):
        _ = x.sum()           # stays in tensor world
    t_tensor = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(100):
        _ = x.sum().item()    # forces sync + Python float allocation
    t_item = time.perf_counter() - t0

    logger.info("sum() 100x         : %.4f s", t_tensor)
    logger.info("sum().item() 100x  : %.4f s (sync cost)", t_item)
    logger.info("Lesson: accumulate loss as tensor, call .item() only for logging.")


benchmark_device_ops()
```

---

## Exercise 2 — Debugging checklist answers

### Bug 1: Loss not decreasing

Checklist question → answer:

1. **Is the learning rate reasonable?**
   Too high (>1.0) causes divergence. Too low (<1e-6) causes imperceptible progress.
   Try: lr = 1e-3 for Adam, lr = 0.01-0.1 for SGD.

2. **Did you call `optimizer.zero_grad()` before `loss.backward()`?**
   Missing zero_grad accumulates gradients across steps — effectively multiplies
   the gradient by the number of steps, causing instability.

3. **Is the loss function correct for the task?**
   Classification → CrossEntropyLoss (raw logits input, NOT softmax output).
   Binary → BCEWithLogitsLoss (raw logits, not sigmoid output).
   Regression → MSELoss.

4. **Are labels the right dtype?**
   CrossEntropyLoss expects `torch.long` (int64) labels, not float.
   MSELoss expects float targets.

5. **Is the model in training mode?**
   `model.train()` enables Dropout and BatchNorm in training mode.
   `model.eval()` disables them for inference. Forgetting eval() causes
   dropout randomness during validation — metrics are noisy but not wrong.
   Forgetting train() causes dropout to be disabled during training — no
   regularization.

```python
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def debug_training_example() -> None:
    """Demonstrate common bugs and their fixes."""
    torch.manual_seed(0)
    X = torch.randn(100, 4)
    y = torch.randint(0, 3, (100,))

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # BUG 1: wrong loss input (softmax THEN CE)
    criterion_buggy = nn.CrossEntropyLoss()
    logits = model(X[:8])
    probs = torch.softmax(logits, dim=1)    # <-- DO NOT do this
    loss_buggy = criterion_buggy(probs, y[:8])
    logger.info("Buggy loss (double-softmax): %.4f", loss_buggy.item())

    # FIX 1: pass raw logits to CrossEntropyLoss
    loss_correct = criterion_buggy(logits, y[:8])
    logger.info("Correct loss (raw logits)  : %.4f", loss_correct.item())

    # BUG 2: missing zero_grad
    total_grad_buggy = 0.0
    for _ in range(5):
        loss = criterion_buggy(model(X), y)
        loss.backward()   # accumulates
        total_grad_buggy = model[0].weight.grad.abs().mean().item()
    logger.info("Grad norm (accumulated, 5 steps): %.4f", total_grad_buggy)

    # FIX 2: zero_grad each step
    optimizer.zero_grad()
    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion_buggy(model(X), y)
        loss.backward()
    logger.info("Grad norm (zeroed each step)    : %.4f",
                model[0].weight.grad.abs().mean().item())

    # BUG 3: float labels for CrossEntropyLoss
    y_float = y.float()
    try:
        criterion_buggy(model(X), y_float)
    except RuntimeError as e:
        logger.info("Float label error: %s", str(e)[:80])
    # Fix: y.long() or ensure dtype=torch.long at creation
```

### Bug 2: NaN loss

Causes and fixes:

```python
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def debug_nan_loss() -> None:
    torch.manual_seed(0)

    # Cause 1: very large learning rate
    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1000.0)   # way too high
    for step in range(5):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X), y)
        loss.backward()
        opt.step()
        logger.info("lr=1000, step=%d, loss=%.4f", step, loss.item())
    # Gradient explodes -> weights become inf -> loss = nan

    # Fix 1: reduce lr or use gradient clipping
    model2 = nn.Linear(4, 2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=1000.0)
    for step in range(5):
        opt2.zero_grad()
        loss = nn.CrossEntropyLoss()(model2(X), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
        opt2.step()
        logger.info("clipped, step=%d, loss=%.4f", step, loss.item())

    # Cause 2: log(0) in manual BCE
    y_pred = torch.tensor([0.0, 0.5, 1.0])   # 0.0 and 1.0 cause log(0)
    y_true = torch.tensor([0.0, 1.0, 1.0])
    loss_bad = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).mean()
    logger.info("Manual BCE with 0/1 predictions: %.4f", loss_bad.item())   # nan

    # Fix 2: use BCEWithLogitsLoss or clamp
    y_pred_safe = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
    loss_safe = -(y_true * torch.log(y_pred_safe) + (1 - y_true) * torch.log(1 - y_pred_safe)).mean()
    logger.info("Manual BCE clamped              : %.4f", loss_safe.item())


debug_nan_loss()
```

### Debugging checklist (printable reference)

```
DEBUGGING CHECKLIST — Neural Network Training
==============================================
Loss not decreasing:
  [ ] optimizer.zero_grad() called before each backward?
  [ ] loss.backward() called?
  [ ] optimizer.step() called after backward?
  [ ] Learning rate reasonable? (Adam: 1e-3, SGD: 0.01-0.1)
  [ ] model.train() set before training loop?

Wrong loss / NaN:
  [ ] Classification: passing raw logits to CrossEntropyLoss (not softmax output)?
  [ ] Binary: using BCEWithLogitsLoss with raw logits (not BCELoss with sigmoid output)?
  [ ] Labels dtype correct? (CrossEntropyLoss needs torch.long)
  [ ] Any log(0) in manual loss? Add clamp or use built-in losses.
  [ ] Gradient exploding? Add clip_grad_norm_(params, max_norm=1.0).

Wrong predictions:
  [ ] model.eval() called before inference?
  [ ] torch.no_grad() wrapping inference (memory + correctness for BN/dropout)?
  [ ] Input normalization same at train and inference time?
  [ ] Correct argmax dimension? (dim=1 for (N, C) logits)

Device errors:
  [ ] All tensors and model on same device?
  [ ] DataLoader tensors moved to device inside loop or pre-moved?
  [ ] Calling .item() / .numpy() only for logging (not in training hot path)?

Shape errors:
  [ ] Print tensor shapes at each layer when stuck.
  [ ] Check expected vs actual shape in error message.
  [ ] Batch dimension first? (N, C, H, W) convention.
```
