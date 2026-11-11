# Solutions — Phase 3 · Week 04 · Day 7 — Review + Project Submission

## This is a review and submission day

Day 7 of Week 04 has no new exercises. The session is structured as:
- 0–30 min: review days 1–5 notes
- 30–150 min: polish and complete the Week 4 project
- 150–180 min: self-grade against the project rubric
- 180–240 min: assessment (`assessment.md`)

See `solutions/week-04/project.md` for the complete reference implementation and
`solutions/week-04/day-6.md` for the full `BPETokenizer` and `SemanticSearchEngine`
reference code (the project is built on Day 6).

---

## Reflection: model answers

### Q1 — TF-IDF + logistic regression vs. sentence embeddings + nearest-neighbour

| Scenario | Better choice |
|---|---|
| Short-document classification (spam, sentiment) with labelled training data | TF-IDF + logreg — fast, interpretable, no embedding API cost |
| Zero-shot semantic search over a large unlabelled corpus | Sentence embeddings + ANN — captures meaning even for paraphrases |

Rule of thumb: if you have labels and the task is classification, TF-IDF is a strong baseline that is often competitive. If you need semantic retrieval without labels, embeddings win.

### Q2 — BPE produces fewer tokens than expected after loading

Most likely cause: the `id2token` mapping is reconstructed as `dict[str, str]` instead of `dict[int, str]` because JSON always serialises dict keys as strings. On load, `{0: "a"}` becomes `{"0": "a"}`. If `encode_ids` uses integer lookups on the restored dict, every lookup misses and falls back to the `<unk>` ID, collapsing token counts.

Fix: restore with explicit int cast: `{int(k): v for k, v in data.items()}`.

### Q3 — Risks of a Python dict keyed by document string at scale

- **Memory**: full string keys are repeated; the dict holds a reference to every document string twice (in the key and the doc list).
- **Serialisation**: `pickle` / `json` cannot store numpy arrays natively; custom serialisation required.
- **Concurrent access**: Python dicts are not thread-safe for concurrent writes.
- **Scale**: 1 M documents × 512-dim float32 = ~2 GB. A numpy matrix is O(1) on axis lookups; a dict with string keys requires hashing.

Production alternative: store vectors in a matrix `(N, D)` and a parallel `list[str]` for documents, or use a purpose-built ANN index (FAISS, Annoy, hnswlib).

### Q4 — BPE training complexity

The dominant term is the inner merge step: for each merge round the algorithm scans all adjacent symbol pairs in the vocabulary (O(V × avg_word_length)). For large corpora, V and average word length are both large. Production libraries (SentencePiece, Hugging Face tokenizers) use priority queues and incremental updates to reduce this to near O(N log N) where N is total token count.

### Q5 — Evaluating semantic search quality

1. **Mean Reciprocal Rank (MRR)**: for each query in a labelled set, find the rank of the first relevant document. MRR = mean(1/rank). Higher is better.
2. **Precision@K / Recall@K**: given a known relevant set, what fraction of the top-K results are relevant (P@K), and what fraction of all relevant documents appear in the top-K (R@K).

---

## Pre-assessment self-check

| # | Check | Pass criterion |
|---|-------|---------------|
| 1 | BPE round-trip | `decode(encode(text)) == text` for any whitespace-separated ASCII text |
| 2 | Unknown word handling | `encode("zyxwvuts")` returns non-empty list (char tokens) |
| 3 | Save/load idempotent | `loaded.encode("lower") == original.encode("lower")` |
| 4 | Search scores in range | All cosine scores in `[-1, 1]` |
| 5 | No real network calls in tests | All tests pass with `unset OPENAI_API_KEY` |
| 6 | Batch indexing | `embed` called `ceil(N / batch_size)` times |
| 7 | Index persistence | `save_index` + `load_index` + `search` returns same top-1 |
| 8 | All tests pass | `pytest tests/ -v` exits 0 |
| 9 | No `print` in `src/` | `grep -r "print(" src/` empty |
| 10 | Type hints everywhere | Confirmed by reading each function signature |

---

## Assessment

See `assessment.md` for the graded assessment (complete independently without lesson references).
