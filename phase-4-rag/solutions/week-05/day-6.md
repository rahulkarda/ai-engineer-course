# Solutions — Week 05 · Day 6 (Saturday Build)

This is a build day. The day-6 lesson contains the full reference implementation. Solutions here cover specific gotchas and common mistakes.

## Common mistakes and fixes

### 1. Index mismatch after MMR

The MMR step receives the *reranked* list (size ≤ `rerank_top_n`), not the original full candidate list. `MMRResult.index` refers to position in the reranked list, not the original `all_chunks` list.

Correct mapping in `pipeline.py`:
```python
for mmr_result in mmr_results:
    # mmr_result.index → position in reranked_texts → position in reranked list
    rerank_entry = reranked[mmr_result.index]
    # rerank_entry.original_index → position in all_chunks
    original_chunk = all_chunks[rerank_entry.original_index]
```

Incorrect (off-by-one bug):
```python
# WRONG: mmr_result.index does not index into all_chunks directly
original_chunk = all_chunks[mmr_result.index]
```

### 2. Empty reranker output

If `score_threshold` is set too high, `reranker.rerank()` may return an empty list. The pipeline must handle this:

```python
if not reranked:
    log.warning("No candidates survived reranking threshold for query: %s", query)
    return []
```

### 3. Embedding shape for single query

`retriever.embed([query])` returns shape `(1, D)`. Extract the vector:
```python
query_emb = self._retriever.embed([query])[0]  # shape (D,)
```

Not:
```python
query_emb = self._retriever.embed([query])  # shape (1, D) — breaks MMR
```

### 4. pydantic-settings env variable precedence

`PipelineSettings` with `env_prefix="RAG_"` reads `RAG_MMR_K`, `RAG_RERANKER_MODEL` etc. from the environment. In tests, set environment variables via `monkeypatch.setenv` rather than modifying `os.environ` directly:

```python
def test_config_from_env(monkeypatch):
    monkeypatch.setenv("RAG_MMR_K", "10")
    from pipeline import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.mmr_k == 10
```

## Self-check answers

1. Empty retriever results: handled by `if not all_chunks: return []`.
2. All ML calls are mockable because `CrossEncoderReranker._load` does a local import, and `retriever` is injected (no hardcoded import).
3. `PipelineConfig` uses Pydantic validators — invalid threshold raises `ValidationError` before runtime.
4. Tests mock `CrossEncoder` and `SentenceTransformer` — no network calls.
5. Metrics show improvement only when reranking corrects genuine bi-encoder mistakes in the eval set.
