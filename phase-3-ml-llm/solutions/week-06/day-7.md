# Solutions — Phase 3 · Week 06 · Day 7 — Working MiniGPT Package

## This is a project-only day

Day 7 of Week 06 is the culminating project session for the week. There are no standalone
exercises — the entire session is dedicated to:

1. Packaging the `minigpt/` module from the week's scaffolds
2. Ensuring all tests pass
3. Training a working model on the synthetic corpus (or tiny shakespeare)
4. Completing the week assessment

See `solutions/week-06/project.md` for the complete reference implementation of every module in
the `minigpt/` package.

---

## Pre-submission self-check

| # | Check | Verification |
|---|-------|--------------|
| 1 | All tests green | `uv run pytest tests/ -v` exits 0 |
| 2 | Package importable | `python -c "from minigpt import MiniGPT"` succeeds |
| 3 | Training CLI works | `uv run python -m minigpt.train --steps 100` logs train/val loss |
| 4 | Val loss < `log(vocab_size)/2` after 500 steps | Visible in training log |
| 5 | Causal mask upheld | `test_causal_mask` passes — position 0 output unchanged when last token changes |
| 6 | Gradient flows through all layers | `test_gradient_flows` — all `p.grad is not None` |
| 7 | Greedy generation deterministic | `test_greedy_deterministic` passes |
| 8 | Checkpoint save/load round-trip | Load checkpoint → generate → no error |
| 9 | No `print` in `minigpt/` | `grep -r "^print" minigpt/` returns empty |
| 10 | Weight tying active | `model.head.weight is model.tok_emb.weight` |

---

## Weight tying — why it works

`head.weight = tok_emb.weight` ties the output projection to the input embedding matrix.
Intuition: a token that is contextually similar to the next token should be nearby in embedding
space, and the same geometry that encodes "token X is similar to token Y" at input should
score token Y highly at output. Empirically, weight tying also acts as implicit regularisation —
halving the parameter count forces the model to learn a single coherent vector space.

---

## Common mistakes

**`StopIteration` in training loop**
The training loop iterates over a `DataLoader` that can run out of batches before `max_steps`.
Fix: wrap with a try/except and reset the iterator:
```python
try:
    x, y = next(it)
except StopIteration:
    it = iter(train_loader)
    x, y = next(it)
```

**Loss plateau at `log(vocab_size)`**
This is the entropy of a uniform distribution over the vocabulary — what the model predicts
before training. If loss never goes below this after 200+ steps, check:
- Learning rate not too high (>1e-2 can cause instability with AdamW)
- `targets` passed to `model.forward()` correctly (not `None`)
- Labels shifted by one: `y = x[1:]` not `y = x[:-1]`

**`generate` function context window clipping**
The model can only process `context_len` tokens at once. Always clip:
```python
logits, _ = model(idx[:, -ctx:])
```
Forgetting this causes an out-of-bounds position embedding lookup once the sequence exceeds
`context_len` tokens.

**Module not found on `python -m minigpt.train`**
Ensure `minigpt/` is a proper package with `__init__.py` and that the project root is in
`PYTHONPATH` (or install with `uv pip install -e .`).

---

## Assessment

See `assessment.md` for the graded assessment. Complete it without opening any lesson file.
