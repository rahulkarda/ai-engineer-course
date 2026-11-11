# Phase 4 · Week 08 · Day 3 — Citations, Grounded Responses, and Streaming

> Objective: Build a `CitationBuilder` that maps answer sentences to source chunks and expose a `/query/stream` endpoint using Server-Sent Events so clients receive tokens as they arrive.  ·  Time: ~1 hr  ·  Prereqs: [[week-08/day-2]]

## Concept

A grounded RAG response is not just an answer string — it is a set of claims, each traceable to a source chunk. **Citations** are the audit trail that separates a verifiable system from an authoritative-sounding hallucinator.

There are two levels of citation granularity:

| Level | What is cited | How it is built |
|---|---|---|
| **Document-level** | "Source: report-2024.pdf" | Attach the parent doc ID to every chunk |
| **Sentence-level** | "Claim X [source: chunk-42, p.3]" | Align each output sentence to its most similar input chunk |

Sentence-level citation is harder but far more useful — it lets the user verify each claim independently.

**Streaming** means the client starts reading tokens before generation is complete. For RAG, streaming is important because:
- Answers can be long (300–800 tokens) — a user staring at a spinner for 3 seconds abandons the product.
- Citations can be emitted as a structured footer after streaming the answer body.

FastAPI delivers streaming via `StreamingResponse` with an `async generator` that yields chunks of bytes.

---

### Server-Sent Events (SSE) Format

SSE is a simple HTTP streaming protocol: the server sends `text/event-stream` responses where each event is:

```
data: <json payload>\n\n
```

The double newline signals event boundaries. Clients (browser `EventSource`, `httpx`) receive events as they arrive. SSE is unidirectional (server → client), which is all RAG needs.

---

## How It Works Internally

`CitationBuilder` works in two passes:

**Pass 1 — Sentence segmentation.** Split the LLM answer into individual sentences (naive: split on `. `, `? `, `! `; production: use `nltk.sent_tokenize` or `spacy`).

**Pass 2 — Chunk alignment.** For each sentence, compute a similarity score against each retrieved chunk. The simplest metric is token overlap (Jaccard similarity); production systems use the same embedding model used at retrieval time to get semantic alignment. The chunk with the highest score above a threshold is attached as the citation.

The `async generator` pattern for streaming: the generator `yield`s SSE-formatted bytes. FastAPI's `StreamingResponse` wraps it and sets `Content-Type: text/event-stream`. The generator calls the LLM with `stream=True` (Anthropic SDK: `client.messages.stream()`), yielding each delta as it arrives.

---

## Worked Example

```python
"""
day_3_citations_streaming.py

CitationBuilder: maps answer sentences to source chunks.
Streaming endpoint: /query/stream via Server-Sent Events.

Run:
    pip install fastapi uvicorn anthropic pydantic
    ANTHROPIC_API_KEY=<key> uvicorn day_3_citations_streaming:app --reload

    # Or run without the LLM (stub mode — no API key needed):
    uvicorn day_3_citations_streaming:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import AsyncGenerator, Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    page: int = 0


class Citation(BaseModel):
    sentence: str
    chunk_id: str
    doc_id: str
    excerpt: str
    similarity: float = Field(ge=0.0, le=1.0)


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    model_id: str
    tokens_used: int = 0


class StreamQueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=20)


# ---------------------------------------------------------------------------
# CitationBuilder
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens, punctuation stripped."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences on sentence-ending punctuation.

    Naive but dependency-free.  Replace with nltk.sent_tokenize for production.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


class CitationBuilder:
    """
    Map each sentence of an LLM answer to its best-matching source chunk.

    The similarity metric is Jaccard over word tokens — fast, no dependencies.
    Swap _score() with a cosine similarity over embeddings for higher accuracy.
    """

    def __init__(self, min_similarity: float = 0.05) -> None:
        self.min_similarity = min_similarity

    def _score(self, sentence: str, chunk: Chunk) -> float:
        return _jaccard(_tokenize(sentence), _tokenize(chunk.text))

    def build(self, answer: str, chunks: list[Chunk]) -> list[Citation]:
        """
        Align each sentence in *answer* to its highest-scoring chunk.

        Sentences whose best chunk scores below *min_similarity* are
        assigned to the overall top chunk rather than dropped — the caller
        can filter low-confidence citations if needed.

        Args:
            answer: Full LLM response text.
            chunks: Retrieved and reranked chunks (ordered by relevance).

        Returns:
            One Citation per sentence.
        """
        if not chunks:
            logger.warning("CitationBuilder received empty chunk list — no citations.")
            return []

        sentences = _split_sentences(answer)
        citations: list[Citation] = []

        for sentence in sentences:
            best_chunk = chunks[0]
            best_score = self._score(sentence, chunks[0])

            for chunk in chunks[1:]:
                score = self._score(sentence, chunk)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk

            # Use best match even if below threshold; log a warning.
            if best_score < self.min_similarity:
                logger.warning(
                    "Sentence has low citation confidence (%.3f): %r",
                    best_score,
                    sentence[:60],
                )

            citations.append(
                Citation(
                    sentence=sentence,
                    chunk_id=best_chunk.chunk_id,
                    doc_id=best_chunk.doc_id,
                    excerpt=best_chunk.text[:120],
                    similarity=round(best_score, 4),
                )
            )

        logger.info(
            "Built %d citations for %d sentences (avg sim: %.3f).",
            len(citations),
            len(sentences),
            sum(c.similarity for c in citations) / max(len(citations), 1),
        )
        return citations


# ---------------------------------------------------------------------------
# Stub retrieval + generation (replace with ComponentRegistry in production)
# ---------------------------------------------------------------------------

_STUB_CHUNKS = [
    Chunk(
        chunk_id="c-001",
        doc_id="doc-001",
        text=(
            "The Document Intelligence Platform ingests PDF, DOCX, and HTML files. "
            "Each file is split into overlapping chunks for retrieval."
        ),
        page=1,
    ),
    Chunk(
        chunk_id="c-002",
        doc_id="doc-001",
        text=(
            "Hybrid retrieval combines dense vector search with BM25 keyword matching. "
            "A reranker then re-scores the candidate set for higher precision."
        ),
        page=2,
    ),
    Chunk(
        chunk_id="c-003",
        doc_id="doc-002",
        text=(
            "Citations map each answer sentence to its source chunk. "
            "This makes RAG responses auditable and reduces hallucination risk."
        ),
        page=1,
    ),
]


def _stub_retrieve(question: str, top_k: int) -> list[Chunk]:
    """Return stub chunks without hitting a vector store."""
    return _STUB_CHUNKS[:top_k]


def _stub_generate_tokens(question: str, chunks: list[Chunk]) -> Iterator[str]:
    """
    Yield answer tokens one word at a time — simulates LLM streaming.

    In production replace with:
        with anthropic.Anthropic().messages.stream(...) as stream:
            for text in stream.text_stream:
                yield text
    """
    answer = (
        "The platform ingests PDF and DOCX files. "
        "Hybrid retrieval combines vector search with keyword matching. "
        "Citations link each sentence to its source chunk."
    )
    for word in answer.split():
        yield word + " "


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_event(data: dict) -> bytes:
    """Format a single SSE event as bytes."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


async def _stream_answer(
    question: str,
    top_k: int,
) -> AsyncGenerator[bytes, None]:
    """
    Async generator that yields SSE events:
      1. [token] events as the LLM generates.
      2. A [citations] event with the full citation list.
      3. A [done] event signalling end of stream.

    The client reassembles the token stream and displays citations as a footer.
    """
    chunks = _stub_retrieve(question, top_k)
    builder = CitationBuilder()

    # Accumulate the full answer so we can build citations at the end.
    full_answer_parts: list[str] = []

    logger.info("Starting SSE stream for: %r", question)

    for token in _stub_generate_tokens(question, chunks):
        full_answer_parts.append(token)
        yield _sse_event({"type": "token", "text": token})

    full_answer = "".join(full_answer_parts)
    citations = builder.build(full_answer, chunks)

    yield _sse_event(
        {
            "type": "citations",
            "citations": [c.model_dump() for c in citations],
        }
    )

    model_id = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    yield _sse_event(
        {
            "type": "done",
            "model_id": model_id,
            "tokens_used": len(full_answer.split()),  # proxy; real: from API response
        }
    )

    logger.info("SSE stream complete (%d tokens, %d citations).", len(full_answer.split()), len(citations))


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Document Intelligence Platform — Streaming", version="0.1.0")


@app.post("/query")
async def query_sync(request: StreamQueryRequest) -> GroundedAnswer:
    """
    Synchronous query endpoint — returns full grounded answer with citations.
    """
    chunks = _stub_retrieve(request.question, request.top_k)
    tokens = list(_stub_generate_tokens(request.question, chunks))
    answer_text = "".join(tokens)

    builder = CitationBuilder()
    citations = builder.build(answer_text, chunks)

    return GroundedAnswer(
        answer=answer_text,
        citations=citations,
        model_id=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
    )


@app.post("/query/stream")
async def query_stream(request: StreamQueryRequest) -> StreamingResponse:
    """
    Streaming query endpoint — returns SSE stream.

    Events:
      {"type": "token",     "text": "..."}
      {"type": "citations", "citations": [...]}
      {"type": "done",      "model_id": "...", "tokens_used": N}

    Client usage (Python):
        import httpx
        with httpx.stream("POST", "http://localhost:8000/query/stream",
                          json={"question": "...", "top_k": 5}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    print(event)
    """
    return StreamingResponse(
        _stream_answer(request.question, request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

**Run and test the streaming endpoint:**
```bash
uvicorn day_3_citations_streaming:app --reload

curl -X POST http://localhost:8000/query/stream \
     -H "Content-Type: application/json" \
     -d '{"question": "How does the platform handle citations?", "top_k": 3}'
```

Expected SSE output:
```
data: {"type": "token", "text": "The "}
data: {"type": "token", "text": "platform "}
...
data: {"type": "citations", "citations": [...]}
data: {"type": "done", "model_id": "claude-haiku-5-5", "tokens_used": 23}
```

---

## Exercises

1. **Extend CitationBuilder for overlapping citations.** A sentence may draw from two chunks (e.g., a comparison sentence). Modify `CitationBuilder.build()` to return up to `max_citations_per_sentence: int = 2` citations per sentence when the second-best chunk's score is within 20% of the best. Update the `Citation` model if needed.

2. **Add a confidence score field.** The current `Citation.similarity` field is computed but not surfaced in a useful way. Add a `confidence: Literal["high", "medium", "low"]` field to `Citation` and compute it: `>= 0.3` → high, `>= 0.1` → medium, `< 0.1` → low. Update `CitationBuilder.build()` accordingly.

3. **Test that every claim has at least one citation.** Write a pytest test that:
   - Calls `CitationBuilder().build()` with a 3-sentence answer and 3 chunks.
   - Asserts that `len(citations) == 3` (one per sentence).
   - Asserts that every `Citation.chunk_id` is non-empty.
   - Asserts that no `Citation.similarity` is `NaN`.

4. **Real LLM streaming (optional, requires API key).** Replace `_stub_generate_tokens` with a real call using the Anthropic Python SDK streaming API. The model ID should come from `os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")` and the API key from `os.getenv("ANTHROPIC_API_KEY")`. Mock the SDK call in your tests.

(Solutions: ../solutions/week-08/day-3.md)

---

## Self-check

- [ ] I can explain the difference between document-level and sentence-level citations.
- [ ] I can implement `CitationBuilder` using Jaccard similarity without external NLP libraries.
- [ ] I can write a `StreamingResponse` with an `async generator` that yields SSE events.
- [ ] I can structure SSE events so a client can distinguish tokens from citations from a done signal.
- [ ] I can write a test that verifies citation coverage without a real LLM.

---

## Supplementary Links

- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [Server-Sent Events spec (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)
- [Anthropic Python SDK — streaming](https://github.com/anthropics/anthropic-sdk-python#streaming)
- [Jaccard similarity — Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
- [NLTK sentence tokenizer](https://www.nltk.org/api/nltk.tokenize.html)
