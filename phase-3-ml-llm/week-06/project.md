# Week 06 Project — MiniGPT: Working Char-Level Language Model

> Duration: 1 day (Saturday lab, ~4 hr)  ·  Deliverable: `minigpt/` package + tests  ·  Assessment: [[assessment.md]]

---

## Goal

Deliver a clean, tested, importable `minigpt` Python package that can:

1. Train a char-level GPT model on a synthetic corpus **or** a user-supplied `.txt` file
2. Sample text using greedy, temperature, and top-k strategies
3. Save and load checkpoints (model + tokenizer vocab)
4. Pass all provided pytest tests

---

## Deliverables Checklist

- [ ] `minigpt/__init__.py` — exports `MiniGPT`, `CharTokenizer`, `GPTConfig`, `generate`
- [ ] `minigpt/config.py` — `GPTConfig` dataclass
- [ ] `minigpt/data.py` — `CharTokenizer`, `CharDataset`, `load_corpus`, `make_loaders`
- [ ] `minigpt/model.py` — `MiniGPT`, `Block`, `CausalSelfAttention`, `FeedForward`
- [ ] `minigpt/generate.py` — `generate()` with temperature + top-k
- [ ] `minigpt/train.py` — training loop + `__main__` CLI
- [ ] `tests/test_data.py` — tokenizer round-trip, dataset shapes, shift invariance
- [ ] `tests/test_model.py` — output shapes, loss is scalar, causal mask, gradient flow
- [ ] `tests/test_generate.py` — generation length, greedy determinism, smoke test

---

## Minimum Acceptance Criteria

| Criterion | Verification |
|---|---|
| All tests green | `uv run pytest tests/ -v` |
| Loss < `log(vocab_size) / 2` after 500 steps | Logged during training |
| Model generates coherent character sequences | Visual inspection of output |
| Checkpoint saves and loads correctly | Re-generate from checkpoint without original corpus |
| No `print` statements | `grep -r "^print" minigpt/` returns empty |
| All public functions have type hints | Confirmed by `ruff check` |

---

## Extension Challenges (optional, no rubric weight)

1. **Top-p (nucleus) sampling** — add a `top_p: float = 0.9` parameter to `generate()`. When set, apply nucleus sampling instead of top-k.

2. **Cosine LR decay** — implement a cosine learning rate schedule in `train.py`. Plot train and val loss with the scheduler on vs off.

3. **Beam search** — implement `generate_beam(model, idx, beam_width=3, max_new=50)`. Compare beam output quality to greedy at the same effective temperature.

4. **Byte-Pair Encoding tokenizer** — implement `BPETokenizer` (see assessment coding challenge) and train the model with it on tiny shakespeare. Compare vocab efficiency and loss.

5. **Multi-GPU stub** — wrap the model in `torch.nn.DataParallel` (CPU no-op; activates on multi-GPU). Verify training still works on CPU.

---

## Submission Format

```
week-06-project/
├── minigpt/           # the package
├── tests/             # pytest suite
├── pyproject.toml     # or requirements.txt
└── README.md          # one-paragraph description + run instructions
```

No secrets, no `.env` files. If you use tiny shakespeare, note the download command — do not commit the file.

---

## Grading

See [[assessment.md]] — the assessment covers both theoretical understanding and the coding challenge. The project itself is graded by the acceptance criteria above.

Pass threshold: all mandatory tests green + loss below baseline.

---

## Supplementary

- [nanoGPT — Karpathy](https://github.com/karpathy/nanoGPT)
- [Let's build GPT from scratch — Karpathy YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [The Illustrated Transformer — Alammar](https://jalammar.github.io/illustrated-transformer/)
- [GPT-2 paper — Radford et al. 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
