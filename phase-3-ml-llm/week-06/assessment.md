# Week 06 Assessment — MiniGPT from Scratch

> Time: ~1 hr  ·  Pass threshold: 75%  ·  Answer key: ../solutions/week-06/assessment.md

---

## Conceptual (interview-style)

1. A transformer block has two sub-layers. Name them, state the role of each, and explain why each is wrapped in a residual connection.

2. What is the causal mask? Why must it be applied *before* the softmax in self-attention? What happens to the output if you forget it?

3. Explain weight tying in a GPT model. Which two matrices share weights, and what is the parameter saving for `vocab_size=50_257`, `d_model=768`?

4. You have a trained model with `context_len=64`. At inference time you want to generate 500 tokens starting from a 10-token prompt. How do you handle the context window limit? What information is lost?

5. Compare greedy decoding, temperature sampling, and top-k sampling. For each: (a) is it deterministic? (b) when is it preferred?

6. What is the cross-entropy loss baseline for a randomly initialised char-level model with vocabulary size 28? Show the calculation.

7. Explain the pre-LN (GPT-2) vs post-LN (original Transformer) placement of Layer Norm. What training stability problem does pre-LN solve?

8. Why does a `FeedForward` sub-layer expand to `4 × d_model` before projecting back? What would happen if you used `1 × d_model` instead?

---

## Coding Challenge

Implement a `BPETokenizer` stub (no BPE algorithm required — just the interface) that:

1. Has `encode(text: str) -> list[int]` and `decode(ids: list[int]) -> str`
2. Accepts a `vocab_size` parameter in `__init__`
3. If `vocab_size <= 128`, falls back to character-level tokenisation (i.e., behaves exactly like `CharTokenizer`)
4. Has a `save(path: Path)` method that serialises the vocabulary to JSON
5. Has a `@classmethod load(path: Path) -> "BPETokenizer"` that deserialises it
6. Passes the following tests:

```python
def test_bpe_char_fallback():
    tok = BPETokenizer(vocab_size=64)
    text = "hello world"
    assert tok.decode(tok.encode(text)) == text

def test_bpe_save_load(tmp_path):
    tok = BPETokenizer(vocab_size=64)
    tok.encode("hello world")  # build vocab
    tok.save(tmp_path / "vocab.json")
    tok2 = BPETokenizer.load(tmp_path / "vocab.json")
    assert tok2.decode(tok2.encode("hello world")) == "hello world"
```

### Acceptance criteria

- [ ] `encode` / `decode` round-trip works for arbitrary ASCII text
- [ ] `save` / `load` produces identical tokenisation
- [ ] Type hints on all methods
- [ ] No `print` statements — use `logging`
- [ ] Tests pass with `uv run pytest`

---

## Rubric (pass ≥ 75%)

| Dimension | Weight | Full marks when… |
|---|---|---|
| Independent implementation | 40% | Coding challenge passes all acceptance criteria without reference |
| Conceptual understanding | 25% | 6/8 conceptual questions answered correctly |
| Debugging & problem-solving | 20% | Correct diagnosis of the shape / causality bugs in the debug section below |
| Code quality & testing | 15% | Type hints, logging, clean module structure, tests present |

### Debug section (part of the 20% debugging marks)

Each snippet below has one bug. Identify and fix it.

**Bug A** — Wrong loss target shape:
```python
logits = model(idx)           # (B, T, V)
loss   = F.cross_entropy(logits, targets)  # targets is (B, T)
```

**Bug B** — Causality broken:
```python
mask = torch.tril(torch.ones(T, T)).bool()
scores = scores.masked_fill(mask, float("-inf"))
```

**Bug C** — Positional embedding out-of-bounds:
```python
pos = torch.arange(0, cfg.context_len)   # always length context_len
x   = tok_emb(idx) + pos_emb(pos)        # idx might be shorter than context_len
```

Answer key: ../solutions/week-06/assessment.md
