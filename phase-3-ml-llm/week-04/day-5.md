# Phase 3 · Week 04 · Day 5 — Contextual & Sentence Embeddings + Embedding APIs

> Objective: Understand contextual embeddings, sentence embeddings, and integrate embedding APIs with proper mocking.  ·  Time: ~1 hr  ·  Prereqs: [[day-4]]

## Concept

### Static vs contextual embeddings

Word2vec gives **static embeddings**: `"bank"` always maps to the same vector regardless of context — riverbank or financial bank.

**Contextual embeddings** (BERT, GPT, etc.) produce a different vector for each occurrence of a word, conditioned on its surrounding context:
- `"I went to the bank to deposit money."` → `bank` gets a "finance" vector.
- `"The hikers sat on the river bank."` → `bank` gets a "geography" vector.

This is because transformer models process the full sequence jointly before producing token representations. Each token's vector is the output of all attention layers — a function of all other tokens.

### Sentence embeddings

Token-level contextual embeddings are not directly useful for comparing whole documents. **Sentence embeddings** compress an entire sentence into a single fixed-size vector that captures its semantic meaning.

Common approaches:
1. **[CLS] token pooling** — use the BERT `[CLS]` token vector as the sentence representation.
2. **Mean pooling** — average all non-padding token vectors.
3. **Sentence-BERT (SBERT)** — fine-tune BERT with a Siamese network on NLI/STS datasets to produce semantically aligned sentence vectors. This is the foundation of `sentence-transformers`.

### Embedding APIs

Production systems rarely run embedding models locally. Instead they call embedding APIs:
- **OpenAI**: `text-embedding-3-small` / `text-embedding-3-large` (1536 / 3072 dims)
- **Cohere**: `embed-english-v3.0`
- **Google**: `text-embedding-004`
- **sentence-transformers**: local models (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`)

All return a list of floats. The pattern is identical regardless of provider.

## How it works internally

### Transformer as an embedding extractor

A transformer processes tokens as follows:
1. Token → embedding (lookup table) + positional encoding
2. Stack of N attention + FFN layers
3. Final hidden states = contextual token vectors

For sentence embeddings, we take the final-layer output and pool across the sequence dimension.

**Mean pooling with attention mask:**

```python
# hidden: (batch, seq_len, hidden_dim)
# mask:   (batch, seq_len)  — 1 for real tokens, 0 for padding
mask_expanded = mask.unsqueeze(-1).float()           # (batch, seq_len, 1)
sum_embeddings = (hidden * mask_expanded).sum(dim=1)  # (batch, hidden_dim)
count = mask_expanded.sum(dim=1).clamp(min=1e-9)     # avoid /0
mean_pooled = sum_embeddings / count
```

L2-normalisation after pooling makes cosine similarity equal to dot product — a common optimisation in vector databases.

### API rate limits and batching

Embedding APIs charge per token and impose rate limits. Best practices:
- Batch documents (most APIs accept up to 2048 inputs per request).
- Cache embeddings for repeated texts (hash the text, store `{hash: vector}`).
- Use async clients for high-throughput pipelines.

## Worked example

```python
"""day5_sentence_embeddings.py — Sentence embeddings and embedding API patterns."""

import hashlib
import logging
import os
from typing import Protocol
from unittest.mock import MagicMock, patch

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Embedding provider protocol (interface abstraction)
# ---------------------------------------------------------------------------

class EmbeddingProvider(Protocol):
    """Any callable that maps a list of texts to a 2-D float array."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), embedding_dim)."""
        ...


# ---------------------------------------------------------------------------
# 2. OpenAI embedding provider (real — requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

class OpenAIEmbedder:
    """Wraps the OpenAI embeddings endpoint."""

    MODEL = "text-embedding-3-small"
    DIM = 1536

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")

    def embed(self, texts: list[str]) -> np.ndarray:
        import openai  # lazy import; install: uv add openai  # noqa: PLC0415

        client = openai.OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self.MODEL, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. sentence-transformers provider (local, no API key needed)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Lazy import so the module loads even if not installed
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        logger.info("Loaded sentence-transformer: %s", model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# 4. Stub provider for testing (no external dependencies)
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Returns deterministic pseudo-random embeddings — suitable for tests."""

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    def embed(self, texts: list[str]) -> np.ndarray:
        # Seed per-text for determinism: same text → same vector
        vectors = []
        for text in texts:
            text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(text_hash)
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)  # normalise to unit sphere
            vectors.append(vec)
        return np.array(vectors)


# ---------------------------------------------------------------------------
# 5. Semantic search over a corpus
# ---------------------------------------------------------------------------

class SemanticSearchEngine:
    """Index a corpus with embeddings and search by cosine similarity."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self._embedder = embedder
        self._docs: list[str] = []
        self._index: np.ndarray | None = None  # shape (n_docs, dim)

    def build_index(self, documents: list[str]) -> None:
        """Embed all documents and store the index."""
        self._docs = documents
        self._index = self._embedder.embed(documents)
        logger.info("Index built: %d docs × %d dims", *self._index.shape)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """Return top-k documents most similar to query."""
        if self._index is None:
            raise RuntimeError("Call build_index first")
        query_vec = self._embedder.embed([query])[0]
        # Cosine similarity = dot product if vectors are L2-normalised
        scores = self._index @ query_vec
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self._docs[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# 6. Caching wrapper
# ---------------------------------------------------------------------------

class CachedEmbedder:
    """Wraps any EmbeddingProvider with a simple in-memory cache."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        results: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_vectors = self._provider.embed(uncached_texts)
            for idx, text, vec in zip(uncached_indices, uncached_texts, new_vectors):
                self._cache[text] = vec
                results[idx] = vec

        return np.array(results)


# ---------------------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------------------

CORPUS: list[str] = [
    "Machine learning models learn patterns from data.",
    "Python is a popular programming language for data science.",
    "Neural networks are inspired by the human brain.",
    "Gradient descent optimises model parameters iteratively.",
    "Transformers use attention mechanisms to process sequences.",
    "Random forests combine many decision trees for robust predictions.",
    "Natural language processing enables computers to understand text.",
    "Embeddings map words and sentences to dense numeric vectors.",
]


def demo_semantic_search() -> None:
    """Run semantic search using the stub embedder (no external calls)."""
    embedder = CachedEmbedder(StubEmbedder(dim=64))
    engine = SemanticSearchEngine(embedder)
    engine.build_index(CORPUS)

    queries = [
        "How do language models represent words?",
        "What optimisation algorithms are used in deep learning?",
    ]
    for query in queries:
        results = engine.search(query, top_k=3)
        logger.info("\nQuery: %r", query)
        for doc, score in results:
            logger.info("  [%.3f] %s", score, doc)


def demo_mock_openai() -> None:
    """Show how to mock the OpenAI client in unit tests."""
    fake_vectors = np.random.default_rng(0).standard_normal((2, 1536)).astype(np.float32)

    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=fake_vectors[0].tolist()),
        MagicMock(embedding=fake_vectors[1].tolist()),
    ]

    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-fake-key"}):
        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_response

            embedder = OpenAIEmbedder()
            result = embedder.embed(["hello", "world"])

    logger.info("Mocked OpenAI result shape: %s", result.shape)
    assert result.shape == (2, 1536), "Expected 2 vectors of dim 1536"
    logger.info("Mock test passed.")


if __name__ == "__main__":
    demo_semantic_search()
    demo_mock_openai()
```

## Exercises

1. Add a `batch_size` parameter to `SemanticSearchEngine.build_index` that embeds documents in chunks (useful when the corpus is large or the API has input limits). Write a test that verifies it produces the same index regardless of `batch_size`.
2. Implement `mmr_search` (Maximal Marginal Relevance) on `SemanticSearchEngine`: return top-k results that are both relevant to the query AND diverse from each other. Use λ=0.5 weighting between relevance and dissimilarity.
3. Extend `CachedEmbedder` to persist the cache to disk as a `.npy` file. Add `save_cache(path)` and `load_cache(path)` methods using `numpy.save/load` with `pathlib.Path`.
4. Write a `pytest` test for `SemanticSearchEngine.search` that mocks `StubEmbedder.embed` to return known vectors, then asserts that the top result is the expected document.
5. The `StubEmbedder` uses MD5 hashing for determinism. Replace it with SHA-256. Write a test that confirms the same text always returns the same vector and two different texts return different vectors.

(Solutions: ../solutions/week-04/day-5.md)

## Self-check
- [ ] I can explain the difference between static word embeddings and contextual embeddings.
- [ ] I can describe mean pooling with attention mask.
- [ ] I can implement an `EmbeddingProvider` interface and swap providers without changing downstream code.
- [ ] I can mock an OpenAI API call with `unittest.mock.patch` so tests never hit the network.
- [ ] I understand when to use [CLS] pooling vs mean pooling.

## Supplementary links
- [Sentence-BERT paper (Reimers & Gurevych 2019)](https://arxiv.org/abs/1908.10084)
- [sentence-transformers library](https://www.sbert.net/)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [The Illustrated BERT — Jay Alammar](https://jalammar.github.io/illustrated-bert/)
