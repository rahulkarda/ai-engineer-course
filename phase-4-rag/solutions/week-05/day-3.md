# Solutions — Week 05 · Day 3

## Exercise 1: relevance_scores parameter

```python
def mmr_select(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    texts: list[str],
    k: int,
    lambda_mult: float = 0.5,
    relevance_scores: list[float] | None = None,
) -> list[MMRResult]:
    n = len(texts)
    k = min(k, n)

    if relevance_scores is not None:
        if len(relevance_scores) != n:
            raise ValueError("relevance_scores length must match texts length")
        relevance = np.array(relevance_scores, dtype=float)
    else:
        relevance = doc_embs @ query_emb

    remaining: set[int] = set(range(n))
    selected_indices: list[int] = []
    results: list[MMRResult] = []

    for step in range(k):
        if step == 0:
            best_idx = int(np.argmax(relevance))
            best_score = float(relevance[best_idx])
        else:
            selected_embs = doc_embs[selected_indices]
            best_score = -1e9
            best_idx = -1
            for idx in remaining:
                rel = float(relevance[idx])
                sims = selected_embs @ doc_embs[idx]
                redundancy = float(np.max(sims))
                score = lambda_mult * rel - (1.0 - lambda_mult) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx = idx

        selected_indices.append(best_idx)
        remaining.discard(best_idx)
        results.append(MMRResult(
            index=best_idx,
            relevance=float(relevance[best_idx]),
            mmr_score=best_score,
            text=texts[best_idx],
        ))

    return results
```

```python
# test
def test_relevance_scores_override():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([1.0, 0.0])
    # Override: make doc 1 more "relevant" via custom scores
    results = mmr_select(q, embs, ["a", "b"], k=1, relevance_scores=[0.1, 0.9])
    assert results[0].index == 1
```

## Exercise 2: caplog test for logging

```python
import logging
import pytest
import numpy as np
from day3_mmr import mmr_select

def test_logging_messages(caplog):
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([1.0, 0.0])
    with caplog.at_level(logging.DEBUG, logger="day3_mmr"):
        mmr_select(q, embs, ["doc-a", "doc-b"], k=2, lambda_mult=0.5)
    # Check that step log messages were emitted
    step_logs = [r for r in caplog.records if "Step" in r.message]
    assert len(step_logs) == 2
    # First step has score info
    assert "selected" in step_logs[0].message.lower() or "step" in step_logs[0].message.lower()
```

## Exercise 3: lambda_sweep

```python
def lambda_sweep(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    texts: list[str],
    k: int,
    lambdas: list[float] | None = None,
) -> dict[float, list[int]]:
    """Run MMR with multiple lambda values. Returns {lambda: [selected_indices]}."""
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    return {
        lam: [r.index for r in mmr_select(query_emb, doc_embs, texts, k=k, lambda_mult=lam)]
        for lam in lambdas
    }
```

At λ=0.0, all results should be maximally diverse (orthogonal embeddings preferred). At λ=1.0, results are sorted by cosine relevance to query only — identical to standard ranking.
