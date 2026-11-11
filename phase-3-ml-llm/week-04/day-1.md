# Phase 3 · Week 04 · Day 1 — Text Preprocessing & Bag-of-Words / TF-IDF

> Objective: Build a classical text feature pipeline from raw strings to TF-IDF vectors.  ·  Time: ~1 hr  ·  Prereqs: [[week-03/day-7]]

## Concept

Every NLP pipeline begins by converting raw text into numbers a model can consume. Classical methods do this with two steps:

1. **Preprocessing** — normalise the surface form of text (lower-case, remove noise, split tokens, reduce vocabulary by stemming/lemmatising).
2. **Vectorisation** — turn a list of tokens into a fixed-length numeric vector.

**Bag-of-Words (BoW)** is the simplest vectoriser: count how many times each vocabulary term appears in a document. Word order is discarded — only frequency counts survive.

**TF-IDF** (Term Frequency × Inverse Document Frequency) improves on BoW by down-weighting terms that appear in almost every document (stop-words, common verbs) and up-weighting rare, discriminative terms.

```
TF(t, d)  = count of t in d / total tokens in d
IDF(t, D) = log( |D| / (1 + |{d ∈ D : t ∈ d}|) )
TF-IDF    = TF × IDF
```

The `1+` in the denominator prevents division by zero for unseen terms (Scikit-learn uses a slightly different smoothing; see docs).

## How it works internally

### Preprocessing pipeline

| Step | Purpose | Library |
|---|---|---|
| Lower-case | `Apple == apple` | built-in `str.lower()` |
| Tokenise | Split on whitespace/punctuation | `str.split()`, `re`, or NLTK |
| Remove stop-words | Drop "the", "is", … | custom list or NLTK |
| Stem / lemmatise | `running → run` | `nltk.stem`, `spacy` |
| Rejoin or keep as list | Depends on vectoriser | — |

### BoW construction

`CountVectorizer` builds a vocabulary dict `{term: index}` during `fit()`, then fills a sparse matrix during `transform()`. Each row = one document; each column = one vocabulary term; the value = raw count.

### TF-IDF weighting

`TfidfVectorizer` wraps `CountVectorizer` and applies the TF-IDF formula. The output matrix is L2-normalised by default (each document vector has unit length), making cosine similarity trivial later.

### Sparsity

A corpus of 10,000 documents and 50,000-term vocabulary produces a 10,000 × 50,000 matrix — but most cells are zero. Scikit-learn stores this as a **CSR sparse matrix** to save memory.

## Worked example

```python
"""day1_tfidf_pipeline.py — Text preprocessing and TF-IDF vectorisation."""

import logging
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Manual preprocessing
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can not no nor so yet "
    "both either or and but if then because as when while how".split()
)


def preprocess(text: str) -> list[str]:
    """Lower-case, tokenise on word boundaries, remove stop-words."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z]+\b", text)
    return [t for t in tokens if t not in STOP_WORDS]


# ---------------------------------------------------------------------------
# 2. Manual Bag-of-Words
# ---------------------------------------------------------------------------

class ManualBoW:
    """Minimal bag-of-words vectoriser for illustration."""

    vocab: dict[str, int]

    def fit(self, corpus: list[list[str]]) -> "ManualBoW":
        all_tokens = [tok for doc in corpus for tok in doc]
        self.vocab = {tok: i for i, tok in enumerate(sorted(set(all_tokens)))}
        logger.info("Vocabulary size: %d", len(self.vocab))
        return self

    def transform(self, corpus: list[list[str]]) -> np.ndarray:
        mat = np.zeros((len(corpus), len(self.vocab)), dtype=np.float32)
        for i, doc in enumerate(corpus):
            for tok, cnt in Counter(doc).items():
                if tok in self.vocab:
                    mat[i, self.vocab[tok]] = cnt
        return mat


# ---------------------------------------------------------------------------
# 3. Scikit-learn pipeline
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "Machine learning is a branch of artificial intelligence",
    "Deep learning uses neural networks for artificial intelligence tasks",
    "The fox ran quickly through the forest",
]


def run_sklearn_pipeline(corpus: list[str]) -> None:
    """Demonstrate CountVectorizer and TfidfVectorizer."""
    # BoW
    cv = CountVectorizer(stop_words="english", max_features=20)
    bow_matrix = cv.fit_transform(corpus)
    logger.info("BoW shape: %s", bow_matrix.shape)
    logger.info("BoW vocabulary (first 10): %s", list(cv.vocabulary_)[:10])

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", max_features=20)
    tfidf_matrix = tfidf.fit_transform(corpus)
    logger.info("TF-IDF shape: %s", tfidf_matrix.shape)

    # Inspect top terms for document 2 (ML sentence)
    doc_idx = 2
    feature_names = tfidf.get_feature_names_out()
    scores = tfidf_matrix[doc_idx].toarray().flatten()
    top_idx = scores.argsort()[::-1][:5]
    logger.info(
        "Top TF-IDF terms for doc %d: %s",
        doc_idx,
        [(feature_names[i], round(float(scores[i]), 3)) for i in top_idx],
    )


# ---------------------------------------------------------------------------
# 4. Manual TF-IDF (no libraries)
# ---------------------------------------------------------------------------

def compute_tfidf(corpus: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Return TF-IDF matrix and vocabulary list."""
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
                idf = np.log(n_docs / (1 + df)) + 1  # sklearn-style smoothing
                mat[i, vocab[term]] = tf * idf

    return mat, all_terms


if __name__ == "__main__":
    # Manual pipeline
    preprocessed = [preprocess(doc) for doc in CORPUS]
    logger.info("Preprocessed[0]: %s", preprocessed[0])

    bow = ManualBoW().fit(preprocessed)
    bow_mat = bow.transform(preprocessed)
    logger.info("Manual BoW matrix shape: %s", bow_mat.shape)

    tfidf_mat, vocab = compute_tfidf(preprocessed)
    logger.info("Manual TF-IDF shape: %s", tfidf_mat.shape)

    # Sklearn
    run_sklearn_pipeline(CORPUS)
```

Run: `python day1_tfidf_pipeline.py`
Deps: `numpy`, `scikit-learn` (install via `uv add numpy scikit-learn`)

## Exercises

1. Extend `preprocess()` to also strip numbers (`\d+`) and single-character tokens. Run it on the corpus and compare vocabulary size before/after.
2. `TfidfVectorizer` has a `ngram_range` parameter. Set `ngram_range=(1, 2)` (unigrams + bigrams) and report how the vocabulary size changes. Why might bigrams improve a text classifier?
3. Implement a `cosine_similarity(a, b)` function using only `numpy` (no sklearn) and use it to find the most similar document pair in the TF-IDF matrix.
4. The manual `compute_tfidf` uses a simplified smoothing. Read sklearn's `TfidfVectorizer` docs and note two differences in its default IDF formula. Adjust your implementation to match and verify the outputs are close.
5. Write a `pytest` test that verifies: (a) `preprocess("The Running foxes")` returns `["running", "foxes"]`; (b) the manual TF-IDF matrix has the correct shape for a 3-document corpus.

(Solutions: ../solutions/week-04/day-1.md)

## Self-check
- [ ] I can explain BoW and its limitations (word order loss, sparsity).
- [ ] I can derive the TF-IDF formula from scratch.
- [ ] I can build a preprocessing pipeline with regex tokenisation and stop-word removal.
- [ ] I can use `TfidfVectorizer` and inspect per-document top terms.
- [ ] I can implement TF-IDF without libraries.

## Supplementary links
- [Sklearn TfidfVectorizer docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [NLTK book — ch. 1 intro to NLP](https://www.nltk.org/book/ch01.html)
- [Wikipedia: tf–idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
