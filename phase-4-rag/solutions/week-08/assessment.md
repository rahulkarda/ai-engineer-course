# Answer Key — Week 08 Assessment

## Conceptual Questions

### Q1: Describe the end-to-end flow of a query through a RAG pipeline with hybrid retrieval and reranking. What happens at each stage?

**Full credit answer:**

1. **Auth and rate limit** — the incoming request is validated (API key) and checked against the rate limiter. Rejected early if invalid or over limit.
2. **Query preprocessing** — the raw question is optionally cleaned (PII scrubbed for logging, whitespace normalized) and embedded into a dense vector using the same embedding model used at ingest time.
3. **Hybrid retrieval** — two passes run concurrently:
   - Dense: ANN search in ChromaDB using the query embedding, returning top-K1 chunks by cosine similarity.
   - Sparse: BM25 query over the indexed corpus, returning top-K2 chunks by TF-IDF score.
   - Fusion: Reciprocal Rank Fusion merges both lists. Each chunk scores `sum(1 / (k + rank_in_list))` for each list it appears in. k=60 is standard. The merged list is sorted descending by RRF score.
4. **Reranking** — a cross-encoder reranker scores each (query, chunk) pair jointly. Cross-encoders see both texts together, giving higher quality relevance judgments than bi-encoder embeddings. The top-N (e.g., 5) reranked chunks are kept.
5. **Agentic loop** — the LangGraph agent assembles the context (chunk texts + IDs) into the prompt. If the LLM signals low confidence or missing information, the agent can trigger another retrieval round with a refined sub-query.
6. **Generation** — the LLM (claude-haiku-5-5 by default) generates an answer using the context, citing chunk IDs inline.
7. **Citation extraction** — a post-processing step parses `[chunk_id]` references, looks up source metadata, and assembles the `citations` list.
8. **Response** — `QueryResponse` with `answer`, `citations`, `model`, `tokens_used` is returned.

---

### Q2: Compare fixed-size chunking with recursive character splitting. When would you choose each?

**Full credit answer:**

**Fixed-size chunking** splits text every N characters regardless of content boundaries. It is fast and simple. It is appropriate when documents are uniform and structure-free (e.g., log lines, raw sensor data) or when you need strict chunk size guarantees for a model with a tight context window.

**Recursive character splitting** tries a hierarchy of separators in order: paragraph breaks (`\n\n`), then line breaks (`\n`), then sentences (`. `), then words (` `), finally characters. At each level it checks whether the resulting segment fits within `chunk_size`. It only falls through to finer granularity when necessary. This preserves semantic units — a paragraph that fits stays whole; one that is too large is split at sentence boundaries rather than mid-sentence.

**Choose recursive splitting for:**
- Natural language documents (articles, PDFs, documentation) where meaning is paragraph-scoped
- Cases where chunk coherence affects embedding quality — a coherent chunk embeds more meaningfully

**Choose fixed-size for:**
- Structured data, code, logs where paragraph structure is absent
- When you need exact reproducibility and the content has no natural boundaries

In practice, recursive character splitting with a 512-token target and 64-token overlap is a strong default for document intelligence use cases.

---

### Q3: What is Reciprocal Rank Fusion, and why is it preferred over score normalization for hybrid retrieval?

**Full credit answer:**

RRF scores a document by summing `1 / (k + rank)` across all ranked lists it appears in, where rank is 1-indexed and k is a constant (typically 60). A document ranked 1st in both lists scores `2 / (1 + 60) = 0.033`. A document ranked 50th in one list only scores `1 / (110) = 0.009`.

**Why prefer over score normalization:**

Dense retrieval returns cosine similarities in `[-1, 1]`. BM25 returns TF-IDF scores in `[0, ∞)`. These scales are incompatible. Normalizing them (e.g., min-max) is fragile — the min and max shift with every query and corpus. A BM25 score of 5.0 on a short corpus is not comparable to 5.0 on a long one.

RRF sidesteps this entirely by using only rank order, not raw scores. Rank is always well-defined and comparable across retrieval systems. The k=60 constant prevents very high-ranked documents from dominating — it dampens the influence of position 1 vs position 2.

RRF is also parameter-free beyond k, making it easy to deploy without per-corpus tuning.

---

### Q4: What is a cross-encoder reranker, and how does it differ from a bi-encoder embedding model?

**Full credit answer:**

A **bi-encoder** embeds the query and each document independently. Similarity is computed as the dot product (or cosine) between the two vectors. This is fast for retrieval because documents can be pre-embedded and indexed; at query time only the query is embedded and ANN search retrieves candidates.

A **cross-encoder** takes the concatenated (query, document) pair as a single input and outputs a relevance score directly. The model attends across both texts jointly, allowing it to consider token-level interactions (e.g., whether a query term appears in a key position in the document). This produces substantially higher quality relevance scores than dot-product similarity.

The trade-off is cost: cross-encoders cannot pre-compute document representations. Every (query, candidate) pair requires a separate forward pass. This makes cross-encoders impractical for retrieval from large corpora but fast enough to rescore a small set of candidates (top-20 from retrieval down to top-5 for generation).

In the pipeline, bi-encoders handle retrieval (high recall, fast), cross-encoders handle reranking (high precision, applied to a small candidate set).

---

### Q5: Describe two techniques for ensuring citation accuracy in an LLM-generated answer.

**Full credit answer:**

**Technique 1 — Instruction-based citation enforcement with in-context examples:** The system prompt instructs the model to append `[chunk_id]` inline wherever it draws on a specific source. Including a worked example showing the exact format (e.g., `"According to the documentation [chunk_42], the process involves..."`) dramatically improves compliance. The post-processing step then parses these references and cross-checks each cited ID against the set of retrieved chunk IDs, dropping any ID the model hallucinated.

**Technique 2 — Structured output with forced citation schema:** Instead of asking the model to embed citations in free text, use structured output (e.g., Anthropic tool use / `response_format: json_object`) to require the model to return `{"answer": "...", "cited_chunk_ids": ["id1", "id2"]}`. The schema enforces that citations are a machine-readable list separate from the answer text. The extraction step reads `cited_chunk_ids` directly rather than parsing prose. This eliminates regex fragility and makes it easy to validate that every cited ID was in the context window.

---

### Q6: What observability signals should a production RAG service emit, and why?

**Full credit answer:**

**Logs (structured JSON):**
- Request ID, endpoint, latency, status code — for request tracing
- Retrieval hit count, reranker top-score, token count — for pipeline quality monitoring
- Error traces with context (query, stage that failed) — for debugging
- PII must be scrubbed before emission

**Metrics (Prometheus / OpenTelemetry):**
- `http_request_duration_seconds` histogram — P50/P95/P99 latency by endpoint
- `rag_retrieval_chunk_count` — how many chunks are actually used
- `rag_eval_score` gauge — rolling mean eval score (catch quality regressions)
- `llm_tokens_total` counter — spend tracking
- `rate_limit_rejected_total` — detect abuse spikes
- `health_check_status` per service — for alerting

**Traces (OpenTelemetry):**
- Distributed trace spanning: auth, retrieval, reranking, LLM call, citation extraction
- Allows pinpointing which stage adds latency

**Why it matters:** Without these, a degraded eval score or P99 latency spike is invisible until a user complains. Structured logs enable log-based alerting. Metrics enable dashboards and SLO tracking. Traces enable root-cause analysis of latency regressions.

---

### Q7: List three Docker security best practices relevant to this platform and explain each.

**Full credit answer:**

**1. Run as a non-root user.** The default Docker container runs as root, which means any RCE vulnerability in the app has full container filesystem access. Adding `RUN adduser --disabled-password --gecos "" appuser && USER appuser` to the Dockerfile ensures the app process runs with minimal privileges. Pair with `--read-only` filesystem mounts where possible.

**2. Never bake secrets into the image.** The Dockerfile must not contain `ENV ANTHROPIC_API_KEY=...` or `COPY .env .`. Secrets are injected at runtime via `docker-compose` `environment` referencing host env vars (`${ANTHROPIC_API_KEY}`) or via a secrets manager. This ensures the image can be published to a registry without leaking credentials.

**3. Use a multi-stage build to minimize attack surface.** A build stage installs all dev dependencies and compiles wheels. The final stage copies only the installed packages and source code, excluding build tools, compilers, and test dependencies. A smaller image has fewer packages with potential CVEs. Example: builder stage uses `python:3.12-slim` with `build-essential`; runtime stage uses `python:3.12-slim` without it.

---

### Q8: What eval metrics would you track, and what threshold would you set for a quality gate?

**Full credit answer:**

**Metrics:**

- **Answer similarity score** — semantic similarity between generated answer and ground truth answer (e.g., cosine similarity of answer embeddings). Measures whether the model conveys the correct information even with different wording. Use `sentence-transformers` for this.
- **Citation recall** — fraction of ground-truth answer claims that are traceable to a cited chunk. Low citation recall indicates hallucination or poor retrieval.
- **Retrieval hit rate (Recall@K)** — for each golden question, is the ground-truth source document in the top-K retrieved chunks? Low hit rate points to a retrieval problem, not a generation one.
- **Latency P95** — end-to-end response time. Quality improvements that triple latency are not acceptable in production.

**Thresholds:**

- Answer similarity: gate at `>= 0.70` on the golden dataset. Drop below 0.65 triggers a block on merge.
- Citation recall: gate at `>= 0.80`. Missing citations on more than 20% of answers is a reliability problem.
- Retrieval hit rate: gate at `>= 0.85` (Recall@10). If the right document is not in the top 10, no amount of reranking or generation helps.

Thresholds should be calibrated on your specific golden dataset. Start permissive (0.6) and tighten after two weeks of production traffic reveal what "good" looks like for your domain.

---

## Coding Challenge: BatchEvalEndpoint

Implement `POST /eval/batch` that accepts a list of `{question, expected_answer}` pairs, runs each through the RAG pipeline, scores the result, and returns per-question scores and aggregate statistics.

### Implementation

```python
# src/models/eval.py
from __future__ import annotations

import uuid
from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    question: str
    expected_answer: str
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])


class EvalBatchRequest(BaseModel):
    questions: list[EvalQuestion]
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QuestionResult(BaseModel):
    question_id: str
    question: str
    generated_answer: str
    expected_answer: str
    similarity_score: float
    citation_count: int
    passed: bool


class AggregateStats(BaseModel):
    total_questions: int
    passed_count: int
    failed_count: int
    mean_score: float
    min_score: float
    max_score: float
    pass_rate: float


class EvalBatchResponse(BaseModel):
    run_id: str
    results: list[QuestionResult]
    aggregate: AggregateStats
    eval_threshold: float
```

```python
# src/routers/eval.py
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends

from src.models.eval import (
    AggregateStats,
    EvalBatchRequest,
    EvalBatchResponse,
    EvalQuestion,
    QuestionResult,
)
from src.middleware.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])

EVAL_THRESHOLD: float = float(os.getenv("EVAL_SCORE_THRESHOLD", "0.70"))


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a ** 2 for a in vec_a) ** 0.5
    mag_b = sum(b ** 2 for b in vec_b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def _score_question(
    question_item: EvalQuestion,
    rag_pipeline,
    embed_fn,
) -> QuestionResult:
    """Run a single question through the RAG pipeline and score it."""
    query_response = await rag_pipeline.query(question_item.question)

    generated_vec = await embed_fn(query_response.answer)
    expected_vec = await embed_fn(question_item.expected_answer)
    score = _cosine_similarity(generated_vec, expected_vec)

    return QuestionResult(
        question_id=question_item.question_id,
        question=question_item.question,
        generated_answer=query_response.answer,
        expected_answer=question_item.expected_answer,
        similarity_score=round(score, 4),
        citation_count=len(query_response.citations),
        passed=score >= EVAL_THRESHOLD,
    )


def _compute_aggregate(results: list[QuestionResult], threshold: float) -> AggregateStats:
    scores = [r.similarity_score for r in results]
    passed = [r for r in results if r.passed]
    return AggregateStats(
        total_questions=len(results),
        passed_count=len(passed),
        failed_count=len(results) - len(passed),
        mean_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
        min_score=round(min(scores), 4) if scores else 0.0,
        max_score=round(max(scores), 4) if scores else 0.0,
        pass_rate=round(len(passed) / len(results), 4) if results else 0.0,
    )


@router.post("/batch", response_model=EvalBatchResponse, summary="Run batch evaluation")
async def eval_batch(
    request: EvalBatchRequest,
    _: None = Depends(require_api_key),
) -> EvalBatchResponse:
    from src.services.rag_pipeline import get_rag_pipeline
    from src.services.embeddings import get_embed_fn

    pipeline = get_rag_pipeline()
    embed_fn = get_embed_fn()

    logger.info("Starting eval run %s with %d questions", request.run_id, len(request.questions))

    results = []
    for q in request.questions:
        result = await _score_question(q, pipeline, embed_fn)
        results.append(result)
        logger.debug(
            "Eval question %s: score=%.3f passed=%s",
            q.question_id,
            result.similarity_score,
            result.passed,
        )

    aggregate = _compute_aggregate(results, EVAL_THRESHOLD)
    logger.info(
        "Eval run %s complete: pass_rate=%.2f mean_score=%.3f",
        request.run_id,
        aggregate.pass_rate,
        aggregate.mean_score,
    )

    return EvalBatchResponse(
        run_id=request.run_id,
        results=results,
        aggregate=aggregate,
        eval_threshold=EVAL_THRESHOLD,
    )
```

```python
# tests/unit/test_eval_endpoint.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.eval import EvalBatchRequest, EvalQuestion


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    import os
    return {"Authorization": f"Bearer {os.getenv('API_KEY_SECRET', 'test-key')}"}


def _make_mock_pipeline(answer: str = "RAG combines retrieval and generation."):
    mock_response = MagicMock()
    mock_response.answer = answer
    mock_response.citations = [MagicMock(chunk_id="chunk1")]
    pipeline = AsyncMock()
    pipeline.query = AsyncMock(return_value=mock_response)
    return pipeline


def _make_mock_embed():
    async def embed(text: str) -> list[float]:
        # Return deterministic pseudo-embedding based on text length
        length = len(text)
        return [0.5 + (length % 10) * 0.01, 0.3, 0.2]

    return embed


@patch("src.routers.eval.get_embed_fn")
@patch("src.routers.eval.get_rag_pipeline")
def test_eval_batch_returns_per_question_scores(
    mock_pipeline_fn,
    mock_embed_fn,
    client: TestClient,
    auth_headers: dict,
) -> None:
    mock_pipeline_fn.return_value = _make_mock_pipeline()
    mock_embed_fn.return_value = _make_mock_embed()

    payload = {
        "questions": [
            {
                "question": "What is RAG?",
                "expected_answer": "RAG is retrieval-augmented generation.",
            },
            {
                "question": "What is chunking?",
                "expected_answer": "Chunking splits documents into smaller pieces.",
            },
        ]
    }

    response = client.post("/eval/batch", json=payload, headers=auth_headers)
    assert response.status_code == 200

    body = response.json()
    assert "run_id" in body
    assert len(body["results"]) == 2
    assert "aggregate" in body

    for result in body["results"]:
        assert "question_id" in result
        assert "similarity_score" in result
        assert isinstance(result["similarity_score"], float)
        assert 0.0 <= result["similarity_score"] <= 1.0
        assert "passed" in result
        assert "citation_count" in result


@patch("src.routers.eval.get_embed_fn")
@patch("src.routers.eval.get_rag_pipeline")
def test_eval_batch_aggregate_stats(
    mock_pipeline_fn,
    mock_embed_fn,
    client: TestClient,
    auth_headers: dict,
) -> None:
    mock_pipeline_fn.return_value = _make_mock_pipeline()
    mock_embed_fn.return_value = _make_mock_embed()

    payload = {
        "questions": [
            {"question": f"Q{i}", "expected_answer": f"Answer {i}"}
            for i in range(5)
        ]
    }

    response = client.post("/eval/batch", json=payload, headers=auth_headers)
    assert response.status_code == 200
    agg = response.json()["aggregate"]

    assert agg["total_questions"] == 5
    assert agg["passed_count"] + agg["failed_count"] == 5
    assert 0.0 <= agg["pass_rate"] <= 1.0
    assert agg["min_score"] <= agg["mean_score"] <= agg["max_score"]


def test_eval_batch_requires_auth(client: TestClient) -> None:
    payload = {"questions": [{"question": "Q?", "expected_answer": "A."}]}
    response = client.post("/eval/batch", json=payload)
    assert response.status_code == 401


@patch("src.routers.eval.get_embed_fn")
@patch("src.routers.eval.get_rag_pipeline")
def test_eval_batch_empty_questions(
    mock_pipeline_fn,
    mock_embed_fn,
    client: TestClient,
    auth_headers: dict,
) -> None:
    mock_pipeline_fn.return_value = _make_mock_pipeline()
    mock_embed_fn.return_value = _make_mock_embed()

    payload = {"questions": []}
    response = client.post("/eval/batch", json=payload, headers=auth_headers)
    assert response.status_code == 200
    agg = response.json()["aggregate"]
    assert agg["total_questions"] == 0
    assert agg["pass_rate"] == 0.0
```

---

## Grading Guidance

| Rubric Dimension | Full Credit (4 pts) | Partial (2-3 pts) | Minimal (0-1 pts) |
|---|---|---|---|
| Conceptual understanding (Q1-Q8) | All 8 answered correctly with accurate technical detail | 5-7 correct, minor inaccuracies | <5 correct or significant technical errors |
| Coding challenge — correctness | Endpoint works, models correct, scoring logic sound | Endpoint works, minor schema errors | Does not run or produces wrong scores |
| Coding challenge — tests | All LLM/HTTP mocked, parametrized, covers edge cases | Tests present but use real I/O or miss edge cases | No tests or tests always pass trivially |
| Code quality | Type hints throughout, Pydantic models, logging, no hardcoded secrets | Most quality rules followed, 1-2 violations | Multiple violations (untyped, print, hardcoded values) |
| Integration quality | Eval endpoint correctly wired into FastAPI, auth gated, response schema matches spec | Wired but auth missing or schema inconsistent | Not integrated with app |

**Total: 20 points**
- 18-20: Distinction
- 14-17: Pass — strong foundations
- 10-13: Pass — revisit weak areas before capstone
- Below 10: Revisit Week 8 lessons and resubmit
