# Phase 4 · Week 01 · Day 1 — RAG Overview: What RAG Is & When to Use It

> Objective: Understand Retrieval-Augmented Generation from first principles and decide when to use RAG vs alternatives.  ·  Time: ~1 hr  ·  Prereqs: [[phase-3-ml-llm]]

## Concept

**The core problem:** LLMs store knowledge in weights at training time. That knowledge is static, has a cutoff date, and — crucially — the model cannot cite *where* it came from. When asked about facts not in its weights, the model fabricates plausible-sounding text. This is hallucination, and it is a structural property of autoregressive generation, not a bug to be patched.

**RAG (Retrieval-Augmented Generation)** grounds model responses in retrieved documents fetched at inference time. The pipeline:

1. At query time, convert the user question to a vector and search a document store.
2. Retrieve the top-k most relevant document chunks.
3. Inject those chunks into the LLM prompt as context.
4. The LLM generates an answer *conditioned on that retrieved context*, not solely on its weights.

The LLM now has access to fresh, specific, citable information — without retraining.

---

### Parametric vs Non-Parametric Knowledge

| Knowledge type | Where it lives | How it's updated |
|---|---|---|
| **Parametric** | Model weights | Retraining or fine-tuning |
| **Non-parametric** | External store (vector DB, search index) | Update the store |

RAG makes an LLM's non-parametric knowledge dynamic and verifiable.

---

### RAG vs Fine-Tuning vs Long Context Window

| Dimension | RAG | Fine-Tuning | Long Context |
|---|---|---|---|
| **Knowledge freshness** | Live — update the store | Stale until retrain | Stale until retraining |
| **Factual grounding** | High — retrieved chunks | Medium — baked in weights | High — full doc in prompt |
| **Cost per query** | Low–medium (retrieval + smaller prompt) | Low (inference only) | High (token cost scales with context length) |
| **Latency** | Adds retrieval step (~50–200 ms) | None | Increases with doc size |
| **When to use** | Dynamic, large, or proprietary knowledge bases | Style, format, domain-specific behavior | Small number of known docs, one-off analysis |
| **When NOT to use** | Real-time streaming data needing sub-10 ms | Factual recall that changes frequently | Corpora > model context limit |

**Decision heuristic:**
- Data changes frequently → RAG
- Need a specific output style or task behavior, data is stable → fine-tune
- Small fixed set of docs, single session, cost secondary → long context
- Need all three → RAG + fine-tuning is common in production

---

### Why Hallucination Happens Without RAG

The model generates the next token by sampling from a probability distribution over its vocabulary, conditioned on prior tokens. When asked about a fact it never saw (or saw rarely), the probability mass spreads across plausible-sounding tokens — the model completes the pattern confidently because fluency and factual accuracy are independent signals in the loss function.

RAG short-circuits this by making the answer a *conditioned retrieval* problem: if the retrieved chunk contains the answer, the model mostly copies/paraphrases it. Error modes shift from hallucination to retrieval failure — which is far easier to detect and audit.

---

## How It Works Internally

### Three-Stage Pipeline

```
OFFLINE (indexing)                ONLINE (query time)
─────────────────                 ──────────────────────────────────
Raw documents                     User query
    │                                 │
    ▼                                 ▼
Chunking                         Embed query → query vector
    │                                 │
    ▼                                 ▼
Embed chunks → vectors           Vector DB similarity search
    │                                 │
    ▼                                 ▼
Store in vector DB               Top-k chunks retrieved
                                      │
                                      ▼
                                 Augmented prompt: [chunks + query]
                                      │
                                      ▼
                                 LLM → grounded answer
```

**Indexing (offline):** Done once (or on updates). Slow operations — PDF parsing, chunking, embedding — are paid upfront. This is the ETL layer of RAG.

**Retrieval (online):** Embed the query using the same model used at indexing time. Compute similarity against stored vectors. Return top-k chunks by score. This is typically 50–200 ms.

**Generation (online):** Construct a prompt that includes retrieved chunks and the original query. The LLM generates conditioned on this prompt.

---

### Why Semantic Search Works

Keyword search (BM25, TF-IDF) matches exact terms. "What is the capital of France?" would miss a document saying "Paris is the seat of French government" if the query doesn't share tokens.

Embedding models (e.g., `text-embedding-3-small`) map text to a high-dimensional vector where *semantic proximity ≈ geometric proximity*. Similar meaning → similar vector direction. Cosine similarity measures the angle between vectors — it captures meaning, not token overlap.

This is why RAG can find "the answer is Paris" when you query "capital of France" — both live in nearby regions of embedding space.

---

### Vector Databases as Enabling Technology

A vector database (Chroma, Pinecone, Weaviate, pgvector) stores embeddings and supports Approximate Nearest Neighbor (ANN) search at scale — returning top-k vectors by cosine/dot-product similarity in milliseconds, even over millions of chunks. We introduce these in Week 02; for now, treat them as "semantic search engines".

---

## Worked Example

```python
"""
day_1_rag_sketch.py

Minimal conceptual RAG sketch — no external dependencies.
Uses keyword matching as a stand-in for semantic retrieval.
Demonstrates the retrieve → augment → generate pattern.
"""

import logging
import os
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Document:
    """A retrieved document chunk with its source metadata."""
    text: str
    source: str
    score: float  # relevance score (0.0–1.0)


# ---------------------------------------------------------------------------
# Retrieval (keyword stand-in for semantic search)
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    corpus: list[dict[str, str]],
    top_k: int = 2,
) -> list[Document]:
    """
    Retrieve top_k documents most relevant to query.

    Uses simple keyword overlap as a proxy for semantic similarity.
    In production this would call a vector database.

    Args:
        query:   User question.
        corpus:  List of {"text": ..., "source": ...} dicts.
        top_k:   Number of chunks to return.

    Returns:
        Ranked list of Document objects.
    """
    query_tokens = set(query.lower().split())
    scored: list[Document] = []

    for item in corpus:
        doc_tokens = set(item["text"].lower().split())
        overlap = len(query_tokens & doc_tokens)
        total = len(query_tokens | doc_tokens)
        score = overlap / total if total > 0 else 0.0
        scored.append(Document(text=item["text"], source=item["source"], score=score))

    ranked = sorted(scored, key=lambda d: d.score, reverse=True)
    top = ranked[:top_k]
    logger.info("Retrieved %d documents (top score: %.3f)", len(top), top[0].score if top else 0.0)
    return top


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_augmented_prompt(query: str, retrieved: list[Document]) -> str:
    """
    Construct the LLM prompt by injecting retrieved chunks as context.

    Args:
        query:     Original user question.
        retrieved: Ranked document chunks.

    Returns:
        Augmented prompt string ready to send to an LLM.
    """
    context_block = "\n\n".join(
        f"[Source: {doc.source}]\n{doc.text}"
        for doc in retrieved
    )
    prompt = (
        "You are a helpful assistant. Answer using ONLY the context below.\n"
        "If the context does not contain the answer, say 'I don't know.'\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    logger.debug("Augmented prompt length: %d chars", len(prompt))
    return prompt


# ---------------------------------------------------------------------------
# Full RAG sketch (generation mocked)
# ---------------------------------------------------------------------------

def rag_query(
    query: str,
    corpus: list[dict[str, str]],
    top_k: int = 2,
) -> str:
    """
    End-to-end RAG pipeline (retrieval + augmentation; generation mocked).

    Replace the final step with a real LLM call using os.getenv("OPENAI_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY") when ready.

    Args:
        query:   User question.
        corpus:  Document store.
        top_k:   Chunks to retrieve.

    Returns:
        Augmented prompt (mock: returns prompt instead of LLM response).
    """
    logger.info("RAG query: %r", query)

    # Stage 1 — Retrieve
    retrieved = retrieve(query, corpus, top_k=top_k)

    # Stage 2 — Augment
    prompt = build_augmented_prompt(query, retrieved)

    # Stage 3 — Generate (mocked — swap in real LLM call here)
    # api_key = os.getenv("ANTHROPIC_API_KEY")
    # response = call_llm(prompt, api_key)
    mock_response = f"[MOCK LLM — would receive {len(prompt)}-char prompt]"
    logger.info("Generation complete (mocked).")

    return mock_response


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_corpus: list[dict[str, str]] = [
        {
            "text": "Python 3.12 introduced a new type parameter syntax for generic classes.",
            "source": "python-3.12-release-notes.md",
        },
        {
            "text": "Vector databases store embeddings and support approximate nearest neighbor search.",
            "source": "rag-fundamentals.md",
        },
        {
            "text": "Fine-tuning updates model weights on a domain-specific dataset.",
            "source": "llm-training-guide.md",
        },
        {
            "text": "RAG grounds LLM responses in retrieved documents to reduce hallucination.",
            "source": "rag-fundamentals.md",
        },
    ]

    result = rag_query(
        query="What is RAG and why does it reduce hallucination?",
        corpus=sample_corpus,
        top_k=2,
    )
    print(result)
```

**Run it:**
```bash
python day_1_rag_sketch.py
```

No dependencies. Expected output: logged retrieval info + mock generation message.

---

## Exercises

1. **Comparison table.** Create a Markdown table comparing RAG, fine-tuning, and long-context for these three dimensions: (a) knowledge update frequency, (b) maximum corpus size, (c) ability to cite sources. Fill each cell with a 1-sentence justification. No code required.

2. **Confidence filter.** Modify `retrieve()` in the worked example to accept a `min_score: float = 0.0` parameter. Documents scoring below `min_score` should be excluded from results. Log a warning if *zero* documents pass the filter. Write one pytest test that asserts an empty list is returned when all scores fall below the threshold.

3. **Architecture decision exercise.** For each scenario below, choose RAG, fine-tuning, or long-context window and write a 2-sentence justification:
   - (a) A law firm's internal QA system over 500,000 case documents, updated daily.
   - (b) A customer support bot that must match the brand's exact tone and abbreviations.
   - (c) A one-off contract review tool that summarizes a single 40-page PDF uploaded by the user.

(Solutions: ../solutions/week-01/day-1.md)

---

## Self-check

- [ ] I can explain RAG as a three-stage pipeline (indexing, retrieval, generation) in under 60 seconds.
- [ ] I can articulate why LLMs hallucinate and how RAG structurally reduces (not eliminates) it.
- [ ] I can compare RAG vs fine-tuning vs long context on at least three dimensions and choose the right one for a given scenario.
- [ ] I can identify at least two situations where RAG is the *wrong* choice.
