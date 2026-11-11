# Solutions — Week 04 · Day 1

## Exercise 1 — Extend `preprocess()` with number and short-token stripping

```python
def preprocess(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
```

Numbers are excluded because `re.findall(r"\b[a-z]+\b", ...)` only matches lowercase letters — digits never enter the list. Single-character tokens (e.g. "a", "i") are dropped by `len(t) > 1`.

Vocabulary size typically drops 5–15% when single-char tokens are removed from a small English corpus.

---

## Exercise 2 — Bigrams with `ngram_range=(1, 2)`

```python
tfidf = TfidfVectorizer(stop_words="english", max_features=200, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(CORPUS)
print("Vocab size:", len(tfidf.vocabulary_))
```

With `ngram_range=(1, 2)`, the vocabulary includes both unigrams and all adjacent word pairs. For a 5-sentence corpus the vocab grows roughly 3–5×.

**Why bigrams improve classifiers:** Phrases like "not good" (negative sentiment), "machine learning" (technical domain), or "New York" (named entity) carry different meaning from their parts. Bigrams capture these compositional units that unigrams miss.

---

## Exercise 3 — Cosine similarity with numpy, find most similar document pair

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0

CORPUS = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "Machine learning is a branch of artificial intelligence",
    "Deep learning uses neural networks for artificial intelligence tasks",
    "The fox ran quickly through the forest",
]

tfidf = TfidfVectorizer(stop_words="english")
mat = tfidf.fit_transform(CORPUS).toarray()

best_i, best_j, best_score = 0, 1, -1.0
n = len(CORPUS)
for i in range(n):
    for j in range(i + 1, n):
        s = cosine_similarity(mat[i], mat[j])
        if s > best_score:
            best_score, best_i, best_j = s, i, j

print(f"Most similar: docs {best_i} & {best_j}  (score {best_score:.3f})")
# Expected: docs 2 & 3 (both about ML/AI) have the highest score
```

---

## Exercise 4 — Sklearn IDF differences and matching

Sklearn's default `TfidfVectorizer` uses:

```
IDF(t) = log( (1 + n) / (1 + df(t)) ) + 1
```

Two differences from the manual formula:
1. The `1+` is added to **both** numerator and denominator (not just denominator).
2. `+1` is added to the entire log (additive smoothing so IDF is always ≥ 1).

Adjusted manual implementation:

```python
def compute_tfidf_sklearn_compat(corpus: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    all_terms = sorted({tok for doc in corpus for tok in doc})
    vocab = {t: i for i, t in enumerate(all_terms)}
    n_docs = len(corpus)
    mat = np.zeros((n_docs, len(vocab)), dtype=np.float64)
    for i, doc in enumerate(corpus):
        counts = Counter(doc)
        doc_len = len(doc)
        for term, cnt in counts.items():
            if term in vocab:
                tf = cnt / doc_len
                df = sum(1 for d in corpus if term in d)
                idf = np.log((1 + n_docs) / (1 + df)) + 1  # sklearn formula
                mat[i, vocab[term]] = tf * idf
    # L2 normalise rows
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms == 0, 1, norms)
    return mat, all_terms
```

---

## Exercise 5 — pytest tests

```python
import pytest
import numpy as np
from day1_tfidf_pipeline import preprocess, compute_tfidf

def test_preprocess_lower_stop_word_removal() -> None:
    result = preprocess("The Running foxes")
    assert result == ["running", "foxes"]

def test_preprocess_returns_list_of_strings() -> None:
    result = preprocess("Hello world")
    assert all(isinstance(t, str) for t in result)

def test_manual_tfidf_shape() -> None:
    corpus = [
        ["cat", "sat", "mat"],
        ["dog", "ran", "mat"],
        ["bird", "flew", "sky"],
    ]
    mat, vocab = compute_tfidf(corpus)
    assert mat.shape == (3, len(vocab))

def test_tfidf_values_nonnegative() -> None:
    corpus = [["hello", "world"], ["world", "peace"]]
    mat, _ = compute_tfidf(corpus)
    assert (mat >= 0).all()
```
