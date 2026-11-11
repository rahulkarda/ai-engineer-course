# Solutions — Week 05 · Day 2

## Exercise 1: batch_size parameter

```python
# In CrossEncoderReranker.__init__:
def __init__(
    self,
    model_name: str = DEFAULT_MODEL,
    max_length: int = 512,
    batch_size: int = 32,
) -> None:
    self._model_name = model_name
    self._max_length = max_length
    self._batch_size = batch_size
    self._model = None

# In rerank():
raw: np.ndarray = np.asarray(
    self._model.predict(
        pairs,
        batch_size=self._batch_size,
        show_progress_bar=False,
    )
)
```

```python
# tests/test_batch_size.py
def test_batch_size_forwarded():
    reranker = CrossEncoderReranker(batch_size=16)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.0, 2.0])
    with patch("day2_reranker.CrossEncoder", return_value=mock_model):
        reranker.rerank("q", ["a", "b"])
    _, kwargs = mock_model.predict.call_args
    assert kwargs.get("batch_size") == 16
```

## Exercise 2: rerank_with_metadata

```python
def rerank_with_metadata(
    self,
    query: str,
    candidates: list[dict],
    top_n: int | None = None,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Like rerank() but accepts dicts with a 'text' key.
    Returns dicts with an added 'rerank_score' field.
    """
    texts = [c["text"] for c in candidates]
    results = self.rerank(query, texts, top_n=top_n, score_threshold=score_threshold)
    # Map back to original dicts
    output = []
    for r in results:
        original = candidates[r.original_index].copy()
        original["rerank_score"] = r.score
        output.append(original)
    return output
```

## Exercise 3: Thread-safe lazy loading (sketch)

```python
import threading

class CrossEncoderReranker:
    def __init__(self, ...):
        ...
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        # Double-checked locking pattern:
        # 1. Fast path: check without acquiring lock (most calls after first load)
        if self._model is not None:
            return
        # 2. Slow path: acquire lock, check again (only one thread loads the model)
        with self._lock:
            if self._model is None:  # second check inside lock
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name, max_length=self._max_length)
```

The double-checked locking pattern avoids acquiring the lock on every call after the model is loaded. The second check *inside* the lock is necessary to prevent two threads that both passed the first check from both loading the model.

Note: in Python, due to the GIL, simple attribute assignment is atomic. But for correctness in any language (and for clarity), the full double-checked pattern is recommended.
