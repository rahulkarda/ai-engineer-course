# Solutions — Week 06 · Day 6

## Exercise 1 — `--steps` CLI flag

```python
# At the top of main():
parser = argparse.ArgumentParser()
parser.add_argument("--data",  type=Path, default=None)
parser.add_argument("--steps", type=int,  default=500)
args = parser.parse_args()

cfg = Config(max_steps=args.steps)
```

---

## Exercise 2 — Cosine LR decay

```python
import math

def cosine_lr(step: int, max_steps: int, lr: float, min_lr_frac: float = 0.1) -> float:
    progress = step / max(max_steps - 1, 1)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_frac + (1.0 - min_lr_frac) * coeff)

# In training loop, after computing loss:
lr_now = cosine_lr(step, cfg.max_steps, cfg.lr)
for g in opt.param_groups:
    g["lr"] = lr_now
```

Log LR at eval steps:
```python
if step % cfg.eval_interval == 0:
    log.info("step %d | lr %.2e | train %.4f | val %.4f", step, lr_now, tl, vl)
```

---

## Exercise 3 — Save generation samples

```python
def save_samples(path: Path, cfg: Config, tok: CharTokenizer,
                 model: MiniGPT) -> None:
    seed_ids = tok.encode(cfg.gen_prompt)
    prompt   = torch.tensor([seed_ids], dtype=torch.long)
    lines    = [
        f"# MiniGPT generation samples",
        f"# config: {cfg}",
        "",
    ]
    for temp, label in [(0.0, "greedy"), (0.8, "T=0.8"), (1.2, "T=1.2")]:
        ids  = generate(model, prompt.clone(), cfg.gen_tokens,
                        temp=max(temp, 1e-6), k=cfg.top_k)
        text = tok.decode(ids[0].tolist())
        lines += [f"## {label}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Samples saved → %s", path)
```

---

## Exercise 4 — Save tokenizer in checkpoint

```python
# In MiniGPT.save():
torch.save({
    "model":     model.state_dict(),
    "config":    cfg,
    "tokenizer": tok.state_dict(),   # dict with stoi, itos, vocab_size
}, cfg.checkpoint_path)

# Reload:
ckpt = torch.load("checkpoint.pt", weights_only=False)
tok2  = CharTokenizer.from_state_dict(ckpt["tokenizer"])
cfg2  = ckpt["config"]
model2 = MiniGPT(cfg2)
model2.load_state_dict(ckpt["model"])
```

---

## Exercise 5 — Dropout effect on val/train gap (observation)

With `dropout=0.0`: val loss diverges from train loss quickly; severe overfitting visible after ~200 steps on tiny corpus.

With `dropout=0.2`: train loss converges more slowly; val/train gap narrows slightly, but on a corpus this small the regularisation benefit is marginal — the model simply memorises less aggressively.

Conclusion: dropout helps most when the corpus is large enough that perfect memorisation isn't possible within the training budget. On tiny synthetic corpora, the benefit is visible but small.
