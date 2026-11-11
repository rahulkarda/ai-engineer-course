# Solutions — Week 06 · Day 4

## Exercise 1 — Val loss every 50 steps

```python
def train_with_val(cfg: GPTConfig) -> MiniGPT:
    tok  = CharTokenizer(SYNTHETIC_TEXT)
    cfg.vocab_size = tok.vocab_size
    tokens = tok.encode(SYNTHETIC_TEXT)

    split = int(len(tokens) * 0.9)
    train_ds = CharDataset(tokens[:split], cfg.context_len)
    val_ds   = CharDataset(tokens[split:], cfg.context_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                              shuffle=False, drop_last=True)

    model = MiniGPT(cfg)
    opt   = model.configure_optimizer()

    @torch.no_grad()
    def eval_loss(loader, steps=20):
        model.eval()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(loader):
            if i >= steps: break
            _, loss = model(x, y)
            total += loss.item(); n += 1
        model.train()
        return total / max(n, 1)

    it = iter(train_loader)
    for step in range(cfg.max_steps):
        try: x, y = next(it)
        except StopIteration: it = iter(train_loader); x, y = next(it)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == cfg.max_steps - 1:
            tl, vl = eval_loss(train_loader), eval_loss(val_loader)
            log.info("step %4d | train %.4f | val %.4f", step, tl, vl)
    return model
```

---

## Exercise 2 — Cosine LR schedule

```python
import math

def cosine_lr(step: int, max_steps: int, lr: float, min_lr_frac: float = 0.1) -> float:
    """Cosine decay from lr to lr * min_lr_frac over max_steps."""
    progress = step / max(max_steps - 1, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)

# In training loop:
for step in range(cfg.max_steps):
    # Update LR for this step
    lr_now = cosine_lr(step, cfg.max_steps, cfg.lr)
    for group in opt.param_groups:
        group["lr"] = lr_now
    # ... rest of loop
```

---

## Exercise 3 — Gradient norm logging

```python
for step in range(cfg.max_steps):
    # ...
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if step % 50 == 0:
        log.info("step %d | loss %.4f | grad_norm %.4f", step, loss.item(), grad_norm)
    opt.step()
```

Observation: early in training `grad_norm` is often > 1.0, so clipping activates. As loss converges, `grad_norm` tends to drop below 1.0 and clipping stops activating.

---

## Exercise 4 — Weight tying verification

```python
def test_weight_tying():
    cfg   = GPTConfig(vocab_size=28, context_len=16, d_model=32,
                      n_heads=2, n_layers=1, dropout=0.0)
    model = MiniGPT(cfg)
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr(), \
        "head and tok_emb do not share storage — weight tying broken"
```

---

## Exercise 5 — Config scaling observations

Representative timings on an M2 MacBook Pro CPU for 200 steps:

| n_layers | d_model | Params | ~Time (200 steps) |
|---|---|---|---|
| 2 | 64  | ~50 k  | < 5 s  |
| 4 | 128 | ~800 k | ~30 s  |
| 6 | 256 | ~6 M   | ~180 s |

Exceeding ~2–3 M parameters on CPU typically pushes 200 steps beyond 60 s. Exact threshold depends on machine; measure with `time.time()`.
