# Phase 4 · Week 06 · Day 2 — HyDE + Contextual Retrieval

> Objective: Improve retrieval precision using hypothetical document embeddings (HyDE) and Anthropic-style contextual chunk prefixing.  ·  Time: ~1 hr  ·  Prereqs: [[week-06/day-1]] (query rewriting)

## Concept

Two techniques that attack retrieval quality from opposite ends of the pipeline:

**HyDE (Hypothetical Document Embeddings)** transforms the *query* before retrieval. Instead of embedding the question "What are the risks of using shell=True in subprocess?" you ask an LLM to generate a plausible answer — a hypothetical document — and embed that instead. The intuition: the embedding space of a generated answer is geometrically closer to real answer chunks than the question is, because questions and answers occupy different regions of the embedding manifold.

**Contextual Retrieval** transforms the *chunks* before indexing. Instead of storing bare chunks, each chunk is prefixed with a short LLM-generated summary that situates it within the source document. "Bill was fired" becomes "In Chapter 3 of 'The Enron Files', discussing the accounting scandal's aftermath: Bill was fired." This enriches the embedding so the chunk can be retrieved by queries that reference context the raw chunk lacks.

Both techniques add LLM calls. HyDE adds one call per query at retrieval time. Contextual retrieval adds one call per chunk at index time — a one-time cost that dramatically improves every subsequent retrieval.

**When to use which**

| Technique | Best for | Cost |
|---|---|---|
| HyDE | Question-answer shaped corpora, FAQ docs, technical Q&A | One LLM call per query |
| Contextual retrieval | Long documents with context-dependent chunks, narrative text, legal/medical docs | One LLM call per chunk (index time only) |
| Both combined | Complex corpora where both problems exist | Both costs |

## How it works internally

### HyDE — the geometry argument

Standard retrieval:
```
Q (question) ──embed──► q_vec ──ANN──► nearest doc chunks
```

The query embedding `q_vec` sits in "question space". Doc chunks sit in "answer space". The cosine similarity between a question and its answer is often lower than between two questions or two answers.

HyDE:
```
Q ──LLM──► Hypothetical_Answer ──embed──► ha_vec ──ANN──► nearest doc chunks
```

`ha_vec` now sits in "answer space" alongside the real chunks. The ANN search finds real chunks that are semantically similar to what the LLM imagined the answer looks like. If the LLM's hypothesis is reasonable, this is dramatically closer to the ground truth than the raw question embedding.

Edge case: if the LLM hallucinates wildly, `ha_vec` points at nothing useful. HyDE works best when the LLM has general knowledge about the domain even without access to the corpus.

### Contextual Retrieval — the context window trick

During indexing, for each chunk `c_i` in document `D`:

```
prompt = f"""
Document excerpt (full):
{D[:8000]}  # truncated to fit context window

Chunk to situate:
{c_i}

Write a 1-2 sentence context that situates this chunk within the document.
Be specific: mention the section, the subject being discussed, and key entities.
"""
context_prefix = LLM(prompt)
contextual_chunk = context_prefix + "\n\n" + c_i
```

`contextual_chunk` is what gets embedded and stored. The retrieval query hits both the prefix (context) and the original text — dramatically improving recall for queries that reference surrounding context.

Anthropic's benchmarks (2024) show contextual retrieval reduces retrieval failure rate by 49% on their internal eval set.

**Practical note on cost:** One LLM call per chunk. For a 10,000-chunk corpus with 100-token prompts, this is ~1M tokens at index time — a one-time cost, cheap with prompt caching enabled. Use `anthropic.Anthropic` with beta `prompt_caching` to cache the document preamble across all chunk calls.

## Worked example

```python
"""
day2_hyde_contextual.py

HyDE retriever and contextual chunk builder.
Both use the Anthropic SDK with os.getenv for API key.
Mock all LLM calls in tests — see exercises.
"""

import logging
import os
from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL = "claude-sonnet-5-5"  # or claude-opus-5-5 for longer/better hypothetical docs


def _client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
class HypotheticalDocument(BaseModel):
    original_query: str
    hypothetical_text: str


class ContextualChunk(BaseModel):
    chunk_id: str
    original_text: str
    context_prefix: str

    @property
    def full_text(self) -> str:
        """The text that gets embedded and stored."""
        return f"{self.context_prefix}\n\n{self.original_text}"


@dataclass
class Chunk:
    id: str
    text: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# VectorRetriever from week 5 — embed text, ANN search
# ---------------------------------------------------------------------------
from week_05.retriever import VectorRetriever  # type: ignore[import]


# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------
class HyDERetriever:
    """
    Retrieves chunks by embedding a hypothetical answer rather than the question.

    Pipeline:
      query → LLM generates hypothetical_answer → embed(hypothetical_answer) → ANN search
    """

    SYSTEM_PROMPT = (
        "You are a technical writer. "
        "Given a question, write a short (3-5 sentence) passage that directly answers it "
        "in the style of technical documentation. "
        "Do not hedge or say 'I don't know'. Write as if you are certain. "
        "Return only the passage, no preamble."
    )

    def __init__(self, retriever: VectorRetriever, model: str = MODEL) -> None:
        self._retriever = retriever
        self._model = model
        self._client = _client()

    def _generate_hypothesis(self, query: str) -> HypotheticalDocument:
        log.info("Generating hypothetical document for: %s", query)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        hyp_text = response.content[0].text.strip()
        log.info("Hypothetical doc (%d chars): %s …", len(hyp_text), hyp_text[:80])
        return HypotheticalDocument(original_query=query, hypothetical_text=hyp_text)

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Generate hypothesis, embed it, retrieve real chunks by proximity."""
        hyp_doc = self._generate_hypothesis(query)
        # Retrieve using the hypothetical answer text, not the original query
        chunks = self._retriever.retrieve(hyp_doc.hypothetical_text, top_k=top_k)
        log.info("HyDE retrieved %d chunks", len(chunks))
        return chunks

    def retrieve_with_fallback(self, query: str, top_k: int = 5) -> list[Chunk]:
        """
        Retrieve via HyDE; if hypothesis retrieves fewer than top_k / 2 chunks,
        fall back to raw query retrieval and merge.
        """
        hyde_chunks = self.retrieve(query, top_k=top_k)
        if len(hyde_chunks) >= top_k // 2:
            return hyde_chunks
        log.warning("HyDE underperformed — merging with raw query retrieval")
        raw_chunks = self._retriever.retrieve(query, top_k=top_k)
        seen: dict[str, Chunk] = {c.id: c for c in hyde_chunks}
        for c in raw_chunks:
            if c.id not in seen:
                seen[c.id] = c
        return list(seen.values())[:top_k]


# ---------------------------------------------------------------------------
# ContextualChunkBuilder — adds LLM-generated context prefix to each chunk
# ---------------------------------------------------------------------------
class ContextualChunkBuilder:
    """
    Enriches chunks with a context summary at index time.

    For each chunk, calls the LLM with the surrounding document excerpt
    and the chunk text, producing a 1-2 sentence situating prefix.
    The full_text property of the returned ContextualChunk is what gets embedded.
    """

    SYSTEM_PROMPT = (
        "You are a document analyst. "
        "You will be given a document excerpt and a specific chunk from that document. "
        "Write 1-2 sentences that situate the chunk within the document: "
        "mention the section, the topic being discussed, and key entities referenced. "
        "Be specific and factual. Return only the situating sentences."
    )

    # Max chars of document context to pass (fits comfortably in 8k context window)
    _DOC_CONTEXT_LIMIT = 6000

    def __init__(self, model: str = MODEL) -> None:
        self._model = model
        self._client = _client()

    def build_contextual_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        document_text: str,
    ) -> ContextualChunk:
        """
        Generate a context prefix for a single chunk.
        Call this for each chunk at index-build time, not at query time.
        """
        doc_excerpt = document_text[: self._DOC_CONTEXT_LIMIT]
        user_message = (
            f"DOCUMENT EXCERPT:\n{doc_excerpt}\n\n"
            f"CHUNK TO SITUATE:\n{chunk_text}"
        )
        log.info("Building context for chunk %s …", chunk_id)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        prefix = response.content[0].text.strip()
        log.info("Context prefix: %s", prefix)
        return ContextualChunk(
            chunk_id=chunk_id,
            original_text=chunk_text,
            context_prefix=prefix,
        )

    def build_all(
        self,
        chunks: list[tuple[str, str]],  # (chunk_id, chunk_text)
        document_text: str,
    ) -> list[ContextualChunk]:
        """
        Process a list of (id, text) tuples. Returns ContextualChunk objects
        whose .full_text should be embedded and stored in the vector index.
        """
        results: list[ContextualChunk] = []
        for chunk_id, chunk_text in chunks:
            try:
                contextual = self.build_contextual_chunk(chunk_id, chunk_text, document_text)
                results.append(contextual)
            except anthropic.APIError as exc:
                log.error("API error on chunk %s: %s — storing without prefix", chunk_id, exc)
                results.append(
                    ContextualChunk(
                        chunk_id=chunk_id,
                        original_text=chunk_text,
                        context_prefix="",
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Demo — illustrates both patterns (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------
def main() -> None:
    # --- HyDE demo (no real retriever wired, just the hypothesis generation) ---
    log.info("=== HyDE: Hypothetical Document Generation ===")
    raw_client = _client()
    query = "What are the risks of using shell=True in Python subprocess?"
    response = raw_client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=(
            "Write a 3-5 sentence passage from technical documentation that "
            "directly answers the question. Return only the passage."
        ),
        messages=[{"role": "user", "content": query}],
    )
    hyp_text = response.content[0].text.strip()
    log.info("Query       : %s", query)
    log.info("Hypothesis  : %s", hyp_text)
    log.info(
        "Observation : Embed the hypothesis above and retrieve — it lands in "
        "'answer space', nearer to real documentation chunks."
    )

    # --- Contextual chunk demo ---
    log.info("\n=== Contextual Chunk Building ===")
    document = (
        "Chapter 5: Security Considerations in Python\n\n"
        "5.1 Subprocess Security\n"
        "The subprocess module allows running external commands. When shell=True is passed, "
        "the command is executed through the shell interpreter. This introduces shell injection "
        "vulnerabilities when the command string contains user-supplied data. "
        "An attacker can inject shell metacharacters to run arbitrary commands.\n\n"
        "5.2 Pickle Security\n"
        "Pickle deserialisation executes arbitrary Python code. Never unpickle data "
        "from untrusted sources."
    )
    chunk_text = "An attacker can inject shell metacharacters to run arbitrary commands."

    builder = ContextualChunkBuilder()
    contextual = builder.build_contextual_chunk(
        chunk_id="ch5-subprocess-01",
        chunk_text=chunk_text,
        document_text=document,
    )
    log.info("Original chunk : %s", contextual.original_text)
    log.info("Context prefix : %s", contextual.context_prefix)
    log.info("Full text (embedded): %s", contextual.full_text)


if __name__ == "__main__":
    main()
```

**Key observations:**

- `HyDERetriever._generate_hypothesis` produces a passage in "answer style" — note the assertive tone specified in the system prompt. Hedged or question-style outputs hurt HyDE performance.
- `ContextualChunkBuilder.build_contextual_chunk` receives the *full document* (truncated), not just surrounding chunks. The LLM sees global context to produce accurate situating prefixes.
- `ContextualChunk.full_text` is what you pass to your embedding model and store in the vector index. Original text alone is still stored separately for display to the user.
- The `build_all` method wraps each call in a `try/except anthropic.APIError` — at index time, one failed chunk should not abort the entire indexing job.

## Exercises

1. **HyDE vs. raw retrieval comparison.** Wire `HyDERetriever` to your Week-5 `VectorRetriever`. For 5 test queries against your corpus, log the top-3 chunk IDs retrieved by raw query vs. HyDE. Compute the Jaccard similarity of the result sets. Write a pytest test that mocks `HyDERetriever._client` and asserts that when the LLM returns a known hypothesis string, `retrieve()` calls `self._retriever.retrieve` with that string (not the original query).

2. **Contextual chunk unit test.** Write a pytest test for `ContextualChunkBuilder.build_contextual_chunk`. Mock `_client()` so it returns a mock `Anthropic` object. Assert that: (a) `ContextualChunk.full_text` starts with the mocked context prefix, (b) the original chunk text is preserved exactly in `original_text`, (c) `full_text` contains both the prefix and original text. Use `unittest.mock.MagicMock` for the Anthropic response object.

3. **Prompt caching for bulk indexing.** Modify `ContextualChunkBuilder.build_all` to use Anthropic's prompt caching by passing `"cache_control": {"type": "ephemeral"}` on the document excerpt system message. This caches the document prefix across all chunk calls and reduces cost by ~90% for large documents. Test with a document that has 10+ chunks and log token usage before and after.

4. **HyDE ensemble.** Modify `HyDERetriever.retrieve` to generate `n_hypotheses` (default 3) hypothetical documents, embed all of them, retrieve top-k for each, and merge/deduplicate (like multi-query expansion). Write a unit test for the dedup logic.

5. **End-to-end contextual index.** Build a mini pipeline: load a local `.txt` file, chunk it into 200-token segments, call `build_all` to generate contextual chunks, embed each `full_text` with a sentence-transformer, and store in an in-memory FAISS index. Retrieve with a test query. Compare retrieval results against an index built from raw chunks (without context prefixes).

(Solutions: ../solutions/week-06/day-2.md)

## Self-check

- [ ] I can explain why HyDE embeds in "answer space" and why that helps.
- [ ] I can describe when HyDE might fail (wild hallucination) and the fallback strategy.
- [ ] I can explain what `ContextualChunk.full_text` contains and why both parts matter.
- [ ] I can implement `ContextualChunkBuilder.build_contextual_chunk` from memory.
- [ ] I can write a pytest test that mocks the Anthropic client for both classes.

## Supplementary links

- [HyDE original paper (Gao et al., 2022)](https://arxiv.org/abs/2212.10496)
- [Anthropic Contextual Retrieval blog post](https://www.anthropic.com/news/contextual-retrieval)
- [Anthropic prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Embedding space geometry — why questions and answers differ](https://arxiv.org/abs/2104.08821)
