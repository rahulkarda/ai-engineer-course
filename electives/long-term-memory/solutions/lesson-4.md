# Solutions: Lesson 4 — Retrieval & Relevance Scoring

---

## Exercise 1 — `weights` parameter with validation

```python
def composite_score(
    record: MemoryRecord,
    query_vector: list[float],
    decay_rate: float = 0.01,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
) -> tuple[float, dict[str, float]]:
    total = alpha + beta + gamma
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    relevance = max(0.0, cosine_sim(query_vector, record.vector))
    recency = compute_recency(record, decay_rate)
    importance = record.importance

    score = alpha * relevance + beta * recency + gamma * importance
    breakdown = {
        "relevance": round(relevance, 4),
        "recency": round(recency, 4),
        "importance": round(importance, 4),
        "composite": round(score, 4),
    }
    return score, breakdown
```

Tests:

```python
import pytest
from lesson_4_retrieval import composite_score, _make_record


def test_default_weights_valid():
    r = _make_record("test", importance=0.5, hours_old=0)
    score, bd = composite_score(r, r.vector)
    assert 0.0 <= score <= 1.0
    assert "composite" in bd


@pytest.mark.parametrize("alpha,beta,gamma", [
    (0.5, 0.3, 0.2),   # valid: sum = 1.0
    (1.0, 0.0, 0.0),   # valid
])
def test_valid_weights(alpha, beta, gamma):
    r = _make_record("test", importance=0.5, hours_old=0)
    score, _ = composite_score(r, r.vector, alpha=alpha, beta=beta, gamma=gamma)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("alpha,beta,gamma", [
    (0.5, 0.5, 0.5),   # sum = 1.5
    (0.1, 0.1, 0.1),   # sum = 0.3
])
def test_invalid_weights(alpha, beta, gamma):
    r = _make_record("test", importance=0.5, hours_old=0)
    with pytest.raises(ValueError, match="Weights must sum"):
        composite_score(r, r.vector, alpha=alpha, beta=beta, gamma=gamma)
```

---

## Exercise 2 — `TimeAwareRetriever`

```python
TIME_SENSITIVE_KEYWORDS = ["latest", "recent", "today", "now", "current", "just"]

class TimeAwareRetriever(MemoryRetriever):
    """Increases recency weight when query is time-sensitive."""

    TIME_SENSITIVE_BETA = 0.5

    def retrieve(
        self,
        query_vector: list[float],
        top_k: int = 5,
        candidate_n: int = 20,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        # Check if query is time-sensitive by checking against keyword vectors
        # Simple heuristic: embed each keyword and check cosine similarity
        time_sensitive = self._is_time_sensitive(query_vector)
        if time_sensitive:
            # Temporarily boost recency weight
            original_beta = self.beta
            original_alpha = self.alpha
            self.beta = self.TIME_SENSITIVE_BETA
            self.alpha = 1.0 - self.TIME_SENSITIVE_BETA - self.gamma
            results = super().retrieve(query_vector, top_k, candidate_n, min_score)
            self.beta = original_beta
            self.alpha = original_alpha
            return results
        return super().retrieve(query_vector, top_k, candidate_n, min_score)

    def _is_time_sensitive(self, query_vector: list[float]) -> bool:
        # Embed keywords and check if any are similar to the query
        from lesson_4_retrieval import mock_embed
        for kw in TIME_SENSITIVE_KEYWORDS:
            kw_vec = mock_embed(kw)
            if cosine_sim(query_vector, kw_vec) > 0.85:
                return True
        return False
```

---

## Exercise 3 — Strictly decreasing recency

```python
import pytest
from lesson_4_retrieval import compute_recency, _make_record


def test_recency_strictly_decreasing():
    hours_values = [0, 10, 24, 48, 72, 120, 168, 240]
    records = [_make_record("x", importance=0.5, hours_old=h) for h in hours_values]
    scores = [compute_recency(r, decay_rate=0.01) for r in records]

    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], (
            f"Expected recency[{i}]={scores[i]:.4f} > recency[{i+1}]={scores[i+1]:.4f} "
            f"(hours_old: {hours_values[i]} vs {hours_values[i+1]})"
        )
```

---

## Exercise 4 — Low `candidate_n` failure

If `candidate_n` is set to 3, the candidate pool contains only the 3 records with
the highest raw cosine similarity. The composite re-ranking step only operates on
these 3 candidates.

**Concrete failure**: Suppose the most relevant memory (highest composite score) has
moderate raw cosine similarity of 0.70, but was accessed recently (high recency) and
is very important (high importance). If there are 4 records with raw cosine > 0.70,
this memory never enters the candidate pool. The agent retrieves a less useful,
lower-importance memory that happened to use more similar vocabulary, missing the
truly relevant one.

The trade-off: `candidate_n` is a **precision-recall knob** for the candidate stage.
Too low → recall drops. Too high → composite re-ranking has more work but is still
fast (it is O(n) not O(n²)).
