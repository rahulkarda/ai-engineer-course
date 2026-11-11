# Solutions — Week 06 · Day 5

## Exercise 1 — Top-p (nucleus) sampling

```python
def sample_top_p(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    Keep the smallest set of tokens whose cumulative probability exceeds p,
    set all others to -inf, then sample.
    Shape: (B, V) → (B, 1)
    """
    # Sort logits descending, compute cumulative probs
    sorted_logits, sorted_indices = torch.sort(logits / max(temperature, 1e-6),
                                               dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative   = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens once cumulative prob exceeds p
    # Shift right: include the token that crosses the threshold
    remove_mask = (cumulative - sorted_probs) > p          # (B, V)
    sorted_logits[remove_mask] = float("-inf")

    # Scatter back to original order
    filtered = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
    return torch.multinomial(F.softmax(filtered, dim=-1), num_samples=1)
```

---

## Exercise 2 — Repetition penalty

```python
def sample_temperature_with_penalty(
    logits: torch.Tensor,
    temperature: float,
    generated_ids: torch.Tensor,  # (1, T_so_far) — previously generated tokens
    penalty: float = 1.2,
) -> torch.Tensor:
    """Divide logits for already-seen tokens by `penalty` before sampling."""
    if penalty != 1.0:
        logits = logits.clone()
        for token_id in generated_ids[0].unique():
            logits[:, token_id] /= penalty
    return sample_temperature(logits, temperature)
```

Observation: with `penalty=1.2` on a greedy model, repetitive phrases break up. Too high a penalty (> 2.0) causes the model to avoid common words and degrade quality.

---

## Exercise 3 — `compare_strategies`

```python
def compare_strategies(
    model: MiniGPT,
    tok: CharTok,
    prompt: str,
    max_new: int = 80,
) -> None:
    seed = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    strategies = [
        ("greedy",        dict(strategy="greedy")),
        ("temp=0.7",      dict(strategy="temperature", temperature=0.7)),
        ("top-k=5 T=0.8", dict(strategy="topk", top_k=5, temperature=0.8)),
    ]
    header = f"{'Strategy':<16} | Generated text"
    log.info(header)
    log.info("-" * 70)
    for label, kwargs in strategies:
        ids = generate(model, seed.clone(), max_new_tokens=max_new, **kwargs)
        text = tok.decode(ids[0].tolist())
        log.info("%-16s | %r", label, text[len(prompt):])
```

---

## Exercise 4 — `test_greedy_deterministic`

```python
def test_greedy_deterministic():
    cfg = GPTConfig(vocab_size=28, context_len=16, d_model=32,
                    n_heads=2, n_layers=1, dropout=0.0)
    model = MiniGPT(cfg)
    model.eval()
    seed = torch.randint(0, cfg.vocab_size, (1, 4))

    o1 = generate(model, seed.clone(), max_new_tokens=20, strategy="greedy")
    o2 = generate(model, seed.clone(), max_new_tokens=20, strategy="greedy")
    assert torch.equal(o1, o2), "Greedy must be deterministic"
```

---

## Exercise 5 — `test_topk_respects_k`

```python
def test_topk_respects_k():
    """With uniform logits, top-k should only return tokens in [0, k)."""
    V = 20
    k = 5
    # Uniform logits: top-k = first k tokens (indices 0..k-1) by tie-breaking
    # To make it deterministic, make first k logits slightly higher
    logits = torch.zeros(1, V)
    logits[0, :k] = 1.0   # top-k tokens are 0..k-1

    results = set()
    for _ in range(200):
        tok_id = sample_topk(logits, k=k, temperature=1.0).item()
        results.add(tok_id)

    assert all(r < k for r in results), \
        f"top-k violated: got tokens outside [0, {k}): {results}"
```
