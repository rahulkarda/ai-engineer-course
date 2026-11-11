# Phase 3 · Week 04 · Day 2 — Word Embeddings & Cosine Similarity

> Objective: Understand word2vec / GloVe intuitions and implement cosine similarity from scratch.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

**Dense word embeddings** replace sparse BoW counts with compact, fixed-size vectors (e.g. 300 dimensions) where **geometry encodes meaning**.

Key properties:
- Similar words cluster in vector space: `vec("cat")` is close to `vec("dog")`.
- Analogies become arithmetic: `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`.
- Every word maps to the same number of dimensions regardless of vocabulary size — no curse of dimensionality.

Two landmark models:

| Model | Learning signal | Core idea |
|---|---|---|
| **word2vec** (Mikolov et al. 2013) | Predict context from centre word (Skip-gram) or vice-versa (CBOW) | Train a shallow neural net; the weight matrix *is* the embedding |
| **GloVe** (Pennington et al. 2014) | Factorise a global word–word co-occurrence matrix | Log-bilinear model; captures global statistics |

Both produce embeddings that capture analogical reasoning — a property that emerges from the training objective, not by design.

## How it works internally

### Skip-gram in one paragraph

Given a centre word `w_c`, predict each surrounding context word `w_o` within a window of size `k`. The network has:
- Input: one-hot vector of `w_c` (size `|V|`)
- Hidden: weight matrix `W ∈ ℝ^{|V|×d}` — the embedding lookup
- Output: softmax over `|V|` classes

After training, row `i` of `W` is the embedding for word `i`. The model never sees word meaning — just raw co-occurrence statistics — yet geometry encodes meaning because synonyms appear in similar contexts.

**Negative sampling** makes training tractable by replacing the full softmax with a binary classification task: "is this a real context word, or a randomly drawn noise word?"

### GloVe in one paragraph

Build a matrix `X` where `X[i,j]` = count of times word `j` appears in the context of word `i` across the whole corpus. Train embeddings `u_i, v_j` such that:

```
u_i · v_j + b_i + b_j ≈ log(X[i,j])
```

The loss weights rare co-occurrences less than frequent ones. The resulting vectors have similar analogical properties to word2vec but train faster on large corpora.

### Cosine similarity

Measures the angle between two vectors, ignoring magnitude:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Range: `[-1, 1]`. For normalised unit vectors, `cos(a, b) = a · b`. Used everywhere in NLP because frequency differences affect vector magnitude but not direction.

## Worked example

```python
"""day2_embeddings_cosine.py — Word embedding intuitions and cosine similarity."""

import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Cosine similarity from scratch
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity for all rows of a matrix."""
    # L2-normalise each row, then dot product = cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)   # avoid division by zero
    normalised = matrix / norms
    return normalised @ normalised.T


# ---------------------------------------------------------------------------
# 2. Toy embedding space to demonstrate analogical reasoning
# ---------------------------------------------------------------------------

# Hand-crafted 4-D embeddings: dims ≈ [royalty, female, plural, abstract]
EMBEDDINGS: dict[str, np.ndarray] = {
    "king":   np.array([0.9,  0.1,  0.0,  0.8]),
    "queen":  np.array([0.9,  0.9,  0.0,  0.8]),
    "man":    np.array([0.1,  0.1,  0.0,  0.8]),
    "woman":  np.array([0.1,  0.9,  0.0,  0.8]),
    "kings":  np.array([0.9,  0.1,  0.9,  0.8]),
    "dog":    np.array([0.0,  0.0,  0.0,  0.3]),
    "cat":    np.array([0.0,  0.05, 0.0,  0.3]),
    "puppy":  np.array([0.0,  0.0,  0.0,  0.2]),
}


def analogy(a: str, b: str, c: str, vocab: dict[str, np.ndarray]) -> str:
    """Find d such that a - b + c ≈ d  (i.e. a:b :: c:?)."""
    query = vocab[a] - vocab[b] + vocab[c]
    exclude = {a, b, c}
    best_word, best_score = "", -2.0
    for word, vec in vocab.items():
        if word in exclude:
            continue
        score = cosine_similarity(query, vec)
        if score > best_score:
            best_score, best_word = score, word
    return best_word


def nearest_neighbours(word: str, vocab: dict[str, np.ndarray], k: int = 3) -> list[tuple[str, float]]:
    """Return top-k most similar words."""
    query = vocab[word]
    scores = [
        (w, cosine_similarity(query, v))
        for w, v in vocab.items()
        if w != word
    ]
    return sorted(scores, key=lambda x: -x[1])[:k]


# ---------------------------------------------------------------------------
# 3. Skip-gram training loop (tiny demonstration)
# ---------------------------------------------------------------------------

def build_cooccurrence(sentences: list[list[str]], window: int = 2) -> tuple[np.ndarray, dict[str, int]]:
    """Build a symmetric co-occurrence matrix from tokenised sentences."""
    vocab_set: set[str] = {tok for sent in sentences for tok in sent}
    vocab: dict[str, int] = {w: i for i, w in enumerate(sorted(vocab_set))}
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=np.float32)
    for sent in sentences:
        for idx, centre in enumerate(sent):
            start = max(0, idx - window)
            end = min(len(sent), idx + window + 1)
            for ctx_idx in range(start, end):
                if ctx_idx != idx:
                    ci = vocab[centre]
                    cj = vocab[sent[ctx_idx]]
                    matrix[ci, cj] += 1
    return matrix, vocab


SENTENCES: list[list[str]] = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "floor"],
    ["the", "cat", "chased", "the", "dog"],
    ["a", "dog", "chased", "a", "cat"],
    ["the", "puppy", "played", "with", "the", "cat"],
]


if __name__ == "__main__":
    # Cosine similarity
    v1 = np.array([1.0, 0.5, 0.2])
    v2 = np.array([0.9, 0.4, 0.3])
    v3 = np.array([-1.0, 0.0, 0.0])
    logger.info("cos(v1, v2) = %.4f  (similar)", cosine_similarity(v1, v2))
    logger.info("cos(v1, v3) = %.4f  (opposite)", cosine_similarity(v1, v3))

    # Nearest neighbours
    nn = nearest_neighbours("dog", EMBEDDINGS)
    logger.info("Nearest to 'dog': %s", nn)

    # Analogy
    result = analogy("king", "man", "woman", EMBEDDINGS)
    logger.info("king - man + woman = '%s'  (expected: queen)", result)

    # Co-occurrence matrix
    cooc, vocab = build_cooccurrence(SENTENCES)
    logger.info("Co-occurrence matrix shape: %s", cooc.shape)
    logger.info("Vocabulary: %s", list(vocab))

    # Pairwise similarity of hand-crafted embeddings
    words = list(EMBEDDINGS)
    mat = np.array([EMBEDDINGS[w] for w in words])
    sim_mat = cosine_similarity_matrix(mat)
    logger.info("Similarity matrix shape: %s", sim_mat.shape)

    # Most similar pair
    np.fill_diagonal(sim_mat, -1)
    i, j = np.unravel_index(sim_mat.argmax(), sim_mat.shape)
    logger.info(
        "Most similar pair: '%s' & '%s'  (score %.3f)",
        words[i], words[j], sim_mat[i, j]
    )
```

## Exercises

1. Add five new words to `EMBEDDINGS` (e.g. "prince", "princess", "boy", "girl", "horse") with sensible 4-D coordinates. Verify that `nearest_neighbours("prince", ...)` returns at least one royalty-related word.
2. Implement `euclidean_distance(a, b)` and compare it with `cosine_similarity` on `EMBEDDINGS`. When do the two measures disagree? Provide a concrete example using the toy embeddings.
3. Using `build_cooccurrence`, compute the PMI (Pointwise Mutual Information) matrix from the co-occurrence matrix. PMI(i,j) = log( P(i,j) / (P(i) * P(j)) ) where probabilities are estimated from counts. Clamp negative PMI to 0 (PPMI).
4. The analogy function uses cosine similarity on the offset vector. An alternative is to find d = argmax cos(c, d) + cos(a, d) - cos(b, d). Implement this "3CosMul" variant and test whether it changes any results on the toy vocabulary.
5. Write a `pytest` test suite with at least three test cases: (a) `cosine_similarity` of identical vectors equals 1.0; (b) orthogonal vectors give 0.0; (c) `cosine_similarity_matrix` is symmetric.

(Solutions: ../solutions/week-04/day-2.md)

## Self-check
- [ ] I can explain how Skip-gram learns embeddings from co-occurrence.
- [ ] I can explain the key difference between word2vec and GloVe.
- [ ] I can implement cosine similarity from scratch with numpy.
- [ ] I can find nearest neighbours and solve word analogies using vector arithmetic.
- [ ] I understand why cosine similarity is preferred over Euclidean distance for text.

## Supplementary links
- [word2vec paper (Mikolov et al. 2013)](https://arxiv.org/abs/1301.3781)
- [GloVe project page](https://nlp.stanford.edu/projects/glove/)
- [Visualising word vectors — Google Embedding Projector](https://projector.tensorflow.org/)
- [The Illustrated Word2Vec — Jay Alammar](https://jalammar.github.io/illustrated-word2vec/)
