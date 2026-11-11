# Solutions — Week 05 · Day 1

## Exercise 1: Swap to bge-small-en-v1.5

```python
import logging
from day1_biencoder_vs_crossencoder import biencoder_rank, CORPUS, QUERY

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

for model_name in models:
    ranked = biencoder_rank(QUERY, CORPUS, model_name=model_name)
    top3 = [(r.rank, r.score, r.text[:60]) for r in ranked[:3]]
    log.info("Model: %s → top3: %s", model_name, top3)
```

The ranking often differs because bge-small uses a different training regime (BAAI's C-PACK contrastive training vs SBERT's NLI/STS fine-tuning), producing a different semantic space. The cross-encoder stage reduces this sensitivity because it attends to the full text jointly.

## Exercise 2: Misleading documents

```python
MISLEADING = [
    "Cross-encoder circuits in signal processing achieve high precision at 50 Hz.",
    "Encoder precision in optical sensors depends on cross-sectional area.",
]

CORPUS_EXT = CORPUS + MISLEADING

bi_ranked = biencoder_rank(QUERY, CORPUS_EXT)
top5_texts = [d.text for d in bi_ranked[:5]]
ce_ranked = crossencoder_rerank(QUERY, top5_texts)

log.info("CE top 3:")
for doc in ce_ranked[:3]:
    log.info("  #%d %.4f %s", doc.rank, doc.score, doc.text[:70])
```

The bi-encoder may surface the misleading documents because "cross-encoder" and "precision" appear in both. The cross-encoder demotes them because the surrounding context ("signal processing", "optical sensors") is semantically distant from information retrieval.

## Exercise 3: top_n + pytest mock

```python
# day1_extended.py
from day1_biencoder_vs_crossencoder import RankedDoc
import numpy as np
from typing import Optional

def crossencoder_rerank(
    query: str,
    candidates: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: Optional[int] = None,
) -> list[RankedDoc]:
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    pairs = [[query, doc] for doc in candidates]
    scores: np.ndarray = model.predict(pairs)
    order = np.argsort(scores)[::-1]
    results = [
        RankedDoc(rank=i + 1, score=float(scores[idx]), text=candidates[idx])
        for i, idx in enumerate(order)
    ]
    if top_n is not None:
        results = results[:top_n]
    return results
```

```python
# tests/test_day1_ex3.py
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from day1_extended import crossencoder_rerank

def test_top_n_limits_results():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2.0, 5.0, 1.0])
    with patch("day1_extended.CrossEncoder", return_value=mock_model):
        results = crossencoder_rerank("q", ["a", "b", "c"], top_n=2)
    assert len(results) == 2

def test_ranking_order_correct():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2.0, 5.0, 1.0])
    with patch("day1_extended.CrossEncoder", return_value=mock_model):
        results = crossencoder_rerank("q", ["a", "b", "c"])
    # b has highest score (5.0) → rank 1
    assert results[0].text == "b"
    assert results[0].rank == 1
```
