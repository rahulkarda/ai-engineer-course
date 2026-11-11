# Solutions — Week 04 · Day 2

## Exercise 1 — Parameter sensitivity

```python
import itertools

for k1, b in itertools.product([0.5, 1.2, 2.0], [0, 0.5, 1.0]):
    scores = bm25_score(query, CORPUS, k1=k1, b=b)
    top1 = rank(scores, CORPUS)[0]
    log.info("k1=%.1f b=%.1f  score=%.4f  %s", k1, b, top1[0], top1[1][:50])
```

When `b=0`: length normalisation disabled — long documents may rank higher (more term occurrences). When `k1` is large (e.g. 10): TF saturation is slow — frequent-term docs score disproportionately high, approaching raw TF-IDF behaviour.

## Exercise 2 — Length normalisation demo

```python
long_doc = "Python " * 50 + "completely unrelated content about cooking recipes"
CORPUS_WITH_LONG = CORPUS + [long_doc]

# b=0.75 (BM25): long_doc scores low because dl >> avgdl
bm_scores = bm25_score("Python data science", CORPUS_WITH_LONG, b=0.75)
log.info("b=0.75 rank of long_doc: %d", sorted(bm_scores, reverse=True).index(bm_scores[-1]) + 1)

# b=0.0 (no normalisation): long_doc may rank first due to raw TF
bm_scores_no_norm = bm25_score("Python data science", CORPUS_WITH_LONG, b=0.0)
log.info("b=0.00 rank of long_doc: %d", sorted(bm_scores_no_norm, reverse=True).index(bm_scores_no_norm[-1]) + 1)
```

With b=0.75, the long doc with 50 "Python" tokens is penalised. With b=0, it unfairly dominates.

## Exercise 3 — Stop words

```python
STOP_WORDS = {"is", "a", "for", "in", "and", "the", "of", "to", "are", "used"}

def tokenize_with_stops(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP_WORDS]
```

Removing stop words reduces noise for most queries. It hurts queries where stop words carry meaning (e.g. "to be or not to be") or when the stop word is itself the search term (rare but happens for code search: "is" as a Python keyword check).

## Exercise 4 — IDF variants

```python
def idf_classic(t: str, N: int, df: dict) -> float:
    return math.log(N / max(df.get(t, 1), 1))

def idf_smooth(t: str, N: int, df: dict) -> float:
    return math.log(1 + N / max(df.get(t, 1), 1))
```

When a term appears in every document (df = N): `idf_classic` = log(1) = 0 — correct, but negative scores possible if df > N (shouldn't happen in theory). `idf_smooth` = log(2) ≈ 0.693 — always positive. BM25 uses a third variant: `log((N - df + 0.5)/(df + 0.5) + 1)` which is always ≥ 0.

## Exercise 5 — Unit tests

```python
# tests/test_day2.py
import pytest
import math
from day2_tfidf_bm25_theory import tokenize, bm25_score, tfidf_score, build_tfidf, CORPUS

QUERY = "Python data science"


def test_tokenize_lowercases():
    assert tokenize("Hello WORLD") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    assert "." not in tokenize("end.")


def test_bm25_python_doc_ranks_first():
    scores = bm25_score(QUERY, CORPUS)
    best_idx = scores.index(max(scores))
    assert "Python" in CORPUS[best_idx]


def test_tfidf_python_doc_ranks_first():
    tf_docs, idf = build_tfidf(CORPUS)
    scores = tfidf_score(QUERY, tf_docs, idf)
    best_idx = scores.index(max(scores))
    assert "Python" in CORPUS[best_idx]


def test_bm25_zero_scores_no_overlap():
    scores = bm25_score("xyzzy quantum_flux_capacitor", CORPUS)
    assert all(s == 0.0 for s in scores)


def test_tfidf_zero_scores_no_overlap():
    tf_docs, idf = build_tfidf(CORPUS)
    scores = tfidf_score("xyzzy quantum_flux_capacitor", tf_docs, idf)
    assert all(s == 0.0 for s in scores)
```
