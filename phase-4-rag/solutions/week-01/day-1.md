# Solutions — Phase 4 · Week 01 · Day 1

## Exercise 1 — RAG vs Fine-tuning vs Long Context Comparison Table

| Dimension | RAG | Fine-tuning | Long Context |
|---|---|---|---|
| **Cost** | Low–medium (retrieval infra + inference on retrieved chunks only) | High upfront (GPU training); low per-query after | High per-query (scales with context window tokens billed) |
| **Latency** | Medium (retrieval round-trip adds ~50–300 ms depending on store) | Low (no retrieval step; pure inference) | High (long-context models are slower; O(n²) attention) |
| **Knowledge freshness** | Excellent — update the index, queries reflect new data immediately | Poor — requires retraining or LoRA re-run for every update | Good — inject fresh docs at query time, but limited by context window |
| **Accuracy on private data** | High — grounded in retrieved source; hallucination reduced by citation | High — domain vocabulary baked in, but no guarantee of factual grounding | Medium-high — depends on whether answer actually appears in the injected context |
| **Implementation complexity** | Medium (chunking, embedding, vector store, retrieval pipeline) | High (dataset prep, training infra, eval, deployment, versioning) | Low (stuff docs into prompt) — until you exceed the window |

**When to choose each:**
- RAG: private, frequently-updated corpora; need citation/attribution; cost-sensitive at scale.
- Fine-tuning: stable domain vocabulary/style that the base model lacks; latency-critical inference with no retrieval budget.
- Long context: rapid prototyping, small corpora that fit the window, or when retrieval precision is hard to tune.


## Exercise 2 — Confidence Filter

Full modified version of the `simple_retrieve` worked example adding a `min_score` threshold:

```python
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    source: str


def simple_retrieve(
    query: str,
    documents: list[Document],
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[Document]:
    """Keyword-based retrieval with optional confidence filter.

    Args:
        query: User query string.
        documents: Corpus of documents to search.
        top_k: Maximum number of results to return.
        min_score: Minimum Jaccard-like overlap score in [0, 1].
                   Documents scoring below this threshold are discarded
                   before the top-k cut, so the result may have fewer
                   than top_k items.

    Returns:
        Up to top_k Documents whose score >= min_score, ranked descending.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be in [0, 1], got {min_score}")

    query_terms = set(query.lower().split())
    scored: list[tuple[float, Document]] = []

    for doc in documents:
        doc_terms = set(doc.text.lower().split())
        score = len(query_terms & doc_terms) / max(len(query_terms), 1)
        if score >= min_score:
            scored.append((score, doc))
            logger.debug("doc=%r score=%.3f (above threshold %.3f)", doc.source, score, min_score)
        else:
            logger.debug("doc=%r score=%.3f filtered out (below min_score=%.3f)", doc.source, score, min_score)

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [doc for _, doc in scored[:top_k]]
    logger.info(
        "retrieve query=%r top_k=%d min_score=%.2f candidates=%d returned=%d",
        query, top_k, min_score, len(scored), len(results),
    )
    return results


def build_augmented_prompt(query: str, retrieved: list[Document]) -> str:
    """Build a RAG-style prompt from retrieved documents."""
    if not retrieved:
        logger.warning("No documents retrieved — LLM will answer without context.")
        context = "(no context retrieved)"
    else:
        context = "\n\n".join(f"[{doc.source}]: {doc.text}" for doc in retrieved)
    return f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"


# --- quick smoke test (run with: python day1_solution.py) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    corpus = [
        Document("Python is a high-level programming language.", "wiki_python"),
        Document("Java is a statically typed language used in enterprise.", "wiki_java"),
        Document("Python supports dynamic typing and is easy to learn.", "blog_python"),
        Document("Machine learning with Python is very popular.", "article_ml"),
    ]

    query = "Python programming language"

    print("=== No filter ===")
    for doc in simple_retrieve(query, corpus, top_k=3, min_score=0.0):
        print(" ", doc.source)

    print("\n=== min_score=0.5 ===")
    for doc in simple_retrieve(query, corpus, top_k=3, min_score=0.5):
        print(" ", doc.source)

    print("\n=== min_score=0.99 (should return empty) ===")
    results = simple_retrieve(query, corpus, top_k=3, min_score=0.99)
    print(f"  returned {len(results)} docs")

    print("\n=== Prompt with filtered results ===")
    retrieved = simple_retrieve(query, corpus, top_k=2, min_score=0.4)
    print(build_augmented_prompt(query, retrieved))
```

**Key design decisions:**
- Validation raises `ValueError` for out-of-range `min_score` rather than silently clamping — fail fast.
- Filtering happens before the top-k slice so `top_k` always means "at most k from the passing set."
- `logger.debug` per-document, `logger.info` for the aggregate — keeps prod logs clean while DEBUG traces every score.
- Return type is still `list[Document]`; callers that passed `min_score=0.0` (default) see identical behaviour to the original.


## Exercise 3 — Scenario Analysis

**Scenario A: Internal HR policy chatbot — 2,000-page policy manual, updated quarterly.**

Choose **RAG**.
- Policy manual is private; fine-tuning a public LLM on it is expensive and leaks data to the training run.
- Quarterly updates are easily handled by re-indexing chunks — no retraining cycle.
- 2,000 pages far exceeds any practical context window for every query.
- Attribution ("per Section 4.2 of the Travel Policy") is critical for HR — RAG with source citations delivers this naturally.

**Scenario B: Code completion tool for a niche internal DSL with unusual syntax.**

Choose **Fine-tuning** (LoRA on a code-completion base model).
- The DSL's token patterns, keywords, and idioms are absent from the base model's pretraining distribution; retrieval of similar snippets helps, but the model will still hallucinate DSL syntax without domain adaptation.
- Code completion is latency-sensitive (< 200 ms target) — retrieval adds unacceptable round-trip overhead for keystroke-level suggestions.
- The DSL syntax is stable (not changing daily), so retraining cost is a one-time investment.
- Hybrid option: fine-tune for syntax fluency + RAG for project-specific API docs.

**Scenario C: Customer support agent needs to answer questions about today's product pricing.**

Choose **RAG** (with a real-time or near-real-time index).
- Pricing changes daily or hourly — fine-tuning cannot keep pace; long context risks stale snapshots.
- A lightweight retrieval step against a pricing database or structured JSON feed keeps answers current.
- Structured data (prices) is well-suited to keyword/metadata filtering before semantic ranking.
- Fine-tuning would bake in yesterday's prices and require constant re-runs — operationally impractical.
