# Week 04 Project — Tiny BPE Tokenizer + Semantic Search Engine

## Goal

Build two production-quality, fully-tested Python modules:

1. **`bpe_tokenizer.py`** — A BPE tokenizer that trains on a corpus, encodes/decodes text, and persists to disk.
2. **`semantic_search.py`** — A semantic search engine that indexes a document corpus using any pluggable embedding provider and answers natural language queries.

These form the foundation for the Phase 3 RAG project (Phase 4, Week 1).

---

## Deliverables

```
week04_project/
├── src/
│   ├── __init__.py
│   ├── bpe_tokenizer.py
│   └── semantic_search.py
├── tests/
│   ├── __init__.py
│   ├── test_bpe_tokenizer.py
│   └── test_semantic_search.py
├── demo.py          # runnable end-to-end demo using StubEmbedder
└── pyproject.toml   # or requirements.txt
```

---

## Functional requirements

### BPETokenizer

| Requirement | Description |
|---|---|
| `fit(corpus, num_merges)` | Learns merge rules; supports corpora up to ~10,000 sentences |
| `encode(text) -> list[str]` | Returns BPE tokens for any input text |
| `encode_ids(text) -> list[int]` | Returns token integer IDs |
| `decode(tokens) -> str` | Reconstructs original text (whitespace-separated) |
| `save(path)` | Serialises merges + vocabulary to JSON |
| `load(path)` classmethod | Restores tokenizer; `encode` must produce identical output |
| Unknown word handling | Degrades to character tokens; never raises on unseen input |
| Empty / punctuation-only input | Returns empty list without error |

### SemanticSearchEngine

| Requirement | Description |
|---|---|
| `build_index(documents)` | Embeds and L2-normalises all documents in batches |
| `search(query, top_k)` | Returns `top_k` (document, cosine_score) pairs |
| `save_index(path)` | Saves embedding matrix (.npy) + document list (.txt) |
| `load_index(path)` | Restores from disk; search works without re-embedding |
| Pluggable provider | Constructor takes any object with `embed(texts) -> ndarray` |
| Batch size | Configurable; respected during `build_index` |

---

## Demo script requirements

`demo.py` must:
1. Train a `BPETokenizer` on 20+ sentences.
2. Encode and decode 5 test phrases; print token sequences.
3. Build a `SemanticSearchEngine` using `StubEmbedder` (from day-5) on a 10-document corpus.
4. Run 3 queries and print ranked results.
5. Save both artefacts to `./artefacts/` and reload them; verify results are identical.

All output via `logging`. No `print`. No external API calls.

---

## Test requirements

Minimum test counts:
- `test_bpe_tokenizer.py`: ≥ 10 tests
- `test_semantic_search.py`: ≥ 8 tests (all with mocked embedder)

Required test cases (in addition to day-6 scaffolds):
- `test_bpe_save_load_preserves_encode_ids` — verify integer IDs match after round-trip
- `test_search_top_k_respects_corpus_size` — asking for top_k > corpus size returns ≤ corpus size results
- `test_build_index_batching_produces_same_result` — batch_size=1 vs batch_size=100 give same index
- `test_cosine_scores_decrease_monotonically` — scores in returned list are non-increasing

---

## Stretch goals (optional, for distinction)

1. **MMR search**: Implement Maximal Marginal Relevance in `SemanticSearchEngine` — `search_mmr(query, top_k, lambda_=0.5)` that balances relevance and diversity.
2. **Byte-level BPE**: Extend `BPETokenizer` to operate on raw bytes (integers 0–255) rather than characters, replicating GPT-2's approach. Test that it handles Unicode without crashing.
3. **Async embedding**: Wrap `OpenAIEmbedder` with an async variant `AsyncOpenAIEmbedder` that uses `httpx.AsyncClient` directly, without the `openai` SDK. Mock it in tests.
4. **PPMI embeddings**: Compute Positive PMI from your co-occurrence matrix (day-2), apply truncated SVD (numpy's `np.linalg.svd`), and compare nearest neighbours to word2vec-style embeddings from pre-trained vectors.

---

## Rubric

| Criterion | Pass | Distinction |
|---|---|---|
| BPETokenizer all requirements | All functional + tests pass | + save/load + encode_ids pass |
| SemanticSearchEngine all requirements | All functional + tests pass | + save/load + batch test pass |
| Tests | ≥ 18 total, all green | ≥ 22, including ≥ 1 parametrised test |
| Code quality | Type hints, logging, docstrings | + module-level `__all__`, ruff clean |
| Demo script runs end-to-end | ✓ with stub embedder | ✓ + artefacts saved/reloaded correctly |
| Stretch goals | 0 | ≥ 1 implemented and tested |

**Pass gate:** BPETokenizer + SemanticSearchEngine requirements met + ≥ 18 tests passing.

---

## Submission checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] `demo.py` runs without errors: `python demo.py`
- [ ] No hardcoded API keys (any key via `os.getenv`)
- [ ] `.env` (if present) is in `.gitignore`
- [ ] `logging` used throughout; no bare `print`
- [ ] All functions/classes have type hints and docstrings
- [ ] Code passes `ruff check src/ tests/`
