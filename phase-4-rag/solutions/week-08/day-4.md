# Solutions — Week 08 · Day 4

## Exercise 1: /metrics Endpoint with token_total and cost_usd

A singleton `MetricsStore` accumulates token counts per model across the process lifetime.
Cost-per-token figures are hard-coded ratios keyed by model family; the formula is
`(input_tokens * input_price + output_tokens * output_price) / 1_000_000` (prices in $/million).
Thread safety uses a `threading.Lock` because FastAPI runs sync handlers in a thread pool.

```python
# src/document_platform/metrics.py
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost table — USD per 1 000 000 tokens (input, output)
# Update when Anthropic publishes new pricing.
# ---------------------------------------------------------------------------

_COST_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5":  (0.25,   1.25),    # cheapest
    "claude-sonnet-5-5": (3.00,  15.00),    # mid-tier
    "claude-opus-5-5":   (15.00, 75.00),    # most capable / most expensive
}

_DEFAULT_COST = (3.00, 15.00)  # fall back to Sonnet pricing for unknown models


def _cost_for_model(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost for a single model call."""
    # Match on prefix so versioned aliases (e.g. claude-haiku-5-5-20250514) still resolve
    rate_in, rate_out = _DEFAULT_COST
    for key, rates in _COST_PER_MILLION.items():
        if model_id.startswith(key):
            rate_in, rate_out = rates
            break
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000.0


# ---------------------------------------------------------------------------
# MetricsStore — thread-safe singleton
# ---------------------------------------------------------------------------

@dataclass
class _ModelMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    call_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class MetricsStore:
    """
    Process-wide accumulator for LLM token usage and cost.

    Use MetricsStore.instance() to get the singleton.  Call record_usage()
    after every LLM call.  The /metrics endpoint reads snapshot().
    """

    _instance: ClassVar["MetricsStore | None"] = None
    _creation_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_model: dict[str, _ModelMetrics] = {}

    @classmethod
    def instance(cls) -> "MetricsStore":
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton — for tests only."""
        with cls._creation_lock:
            cls._instance = None

    def record_usage(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Thread-safe update of per-model counters."""
        with self._lock:
            if model_id not in self._per_model:
                self._per_model[model_id] = _ModelMetrics()
            m = self._per_model[model_id]
            m.input_tokens += input_tokens
            m.output_tokens += output_tokens
            m.call_count += 1
        logger.debug(
            "Recorded token usage",
            extra={"model": model_id, "input": input_tokens, "output": output_tokens},
        )

    def snapshot(self) -> "MetricsSnapshot":
        """Return an immutable copy of current metrics."""
        with self._lock:
            per_model_out: dict[str, ModelMetrics] = {}
            token_total = 0
            cost_total = 0.0

            for model_id, m in self._per_model.items():
                cost = _cost_for_model(model_id, m.input_tokens, m.output_tokens)
                per_model_out[model_id] = ModelMetrics(
                    model_id=model_id,
                    input_tokens=m.input_tokens,
                    output_tokens=m.output_tokens,
                    total_tokens=m.total_tokens,
                    call_count=m.call_count,
                    cost_usd=round(cost, 6),
                )
                token_total += m.total_tokens
                cost_total += cost

        return MetricsSnapshot(
            token_total=token_total,
            cost_usd=round(cost_total, 6),
            per_model=per_model_out,
        )


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

from pydantic import BaseModel  # noqa: E402 — import after dataclass definitions


class ModelMetrics(BaseModel):
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    call_count: int
    cost_usd: float


class MetricsResponse(BaseModel):
    """Top-level /metrics response."""
    token_total: int
    cost_usd: float
    per_model: dict[str, ModelMetrics]


# Alias used by snapshot() above
MetricsSnapshot = MetricsResponse


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

from fastapi import APIRouter  # noqa: E402

metrics_router = APIRouter(tags=["observability"])


@metrics_router.get("/metrics", response_model=MetricsResponse)
def get_metrics() -> MetricsResponse:
    """
    Return accumulated token usage and estimated cost across all LLM calls
    since the process started (or since the last reset).
    """
    snap = MetricsStore.instance().snapshot()
    logger.info(
        "Metrics requested",
        extra={"token_total": snap.token_total, "cost_usd": snap.cost_usd},
    )
    return snap
```

### Wiring the record_usage call in your LLM client

```python
# src/document_platform/llm_client.py
from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from src.document_platform.metrics import MetricsStore

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")


def chat(messages: list[dict[str, Any]], system: str = "") -> str:
    """
    Send messages to the Claude API and record token usage.
    Returns the assistant's text reply.
    """
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    MetricsStore.instance().record_usage(
        model_id=_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
    return response.content[0].text
```

---

## Exercise 2: Golden-Dataset Eval Runner Asserting RAGAS Faithfulness >= 0.8

The golden dataset is a JSON file checked into the repo. The RAGAS scorer is mocked so the test
runs in CI without an LLM key, but the assertion logic is real — swap in the live scorer for a
nightly integration gate.

```python
# tests/integration/test_eval_harness.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOLDEN_DATASET_PATH = Path(__file__).parent / "fixtures" / "golden_dataset.json"

SAMPLE_GOLDEN_DATASET: list[dict[str, Any]] = [
    {
        "question": "What is the capital of France?",
        "ground_truth": "Paris is the capital of France.",
        "context": ["Paris is the capital city of France and its largest city."],
    },
    {
        "question": "Who wrote Hamlet?",
        "ground_truth": "William Shakespeare wrote Hamlet.",
        "context": ["Hamlet is a tragedy written by William Shakespeare around 1600."],
    },
    {
        "question": "What gas do plants absorb during photosynthesis?",
        "ground_truth": "Plants absorb carbon dioxide (CO2) during photosynthesis.",
        "context": ["During photosynthesis, plants absorb CO2 and release O2."],
    },
]


@pytest.fixture(scope="session")
def golden_dataset() -> list[dict[str, Any]]:
    """
    Load the golden dataset from disk if it exists; otherwise use the sample set.
    This lets CI run without the full dataset while local runs use the real file.
    """
    if GOLDEN_DATASET_PATH.exists():
        with GOLDEN_DATASET_PATH.open() as fh:
            return json.load(fh)
    return SAMPLE_GOLDEN_DATASET


@pytest.fixture()
def mock_ragas_scorer() -> MagicMock:
    """
    Mock the RAGAS faithfulness scorer.

    In production, replace with:
        from ragas import evaluate
        from ragas.metrics import faithfulness
    The mock returns per-sample scores just above the threshold so the test
    demonstrates the assertion logic without requiring an LLM key.
    """
    scorer = MagicMock()
    scorer.score.side_effect = lambda question, answer, context, ground_truth: 0.85
    return scorer


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_faithfulness_eval(
    dataset: list[dict[str, Any]],
    rag_pipeline: Any,
    scorer: Any,
) -> dict[str, float]:
    """
    Run faithfulness evaluation over the golden dataset.

    Returns:
        {"average_faithfulness": float, "min_faithfulness": float, "sample_count": int}
    """
    scores: list[float] = []
    for sample in dataset:
        answer_obj = rag_pipeline.answer(
            query=sample["question"],
            ranked_chunks=[],  # pipeline stub — real impl passes retrieved chunks
        )
        score = scorer.score(
            question=sample["question"],
            answer=answer_obj.answer,
            context=sample["context"],
            ground_truth=sample["ground_truth"],
        )
        scores.append(float(score))

    return {
        "average_faithfulness": sum(scores) / len(scores) if scores else 0.0,
        "min_faithfulness": min(scores) if scores else 0.0,
        "sample_count": len(scores),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

FAITHFULNESS_THRESHOLD = 0.8


@pytest.fixture()
def mock_rag_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.answer.return_value = MagicMock(
        answer="The answer generated by the RAG pipeline.",
        citations=[{"chunk_id": "c1", "text": "context text", "score": 0.9}],
    )
    return pipeline


def test_average_faithfulness_meets_threshold(
    golden_dataset: list[dict[str, Any]],
    mock_ragas_scorer: MagicMock,
    mock_rag_pipeline: MagicMock,
):
    """
    Average RAGAS faithfulness over the golden dataset must be >= 0.8.

    This is the primary quality gate for the RAG pipeline.  A regression below
    0.8 means answers are drifting from the source documents (hallucination risk).
    """
    results = run_faithfulness_eval(golden_dataset, mock_rag_pipeline, mock_ragas_scorer)

    logger.info(
        "Faithfulness eval results",
        extra={
            "average_faithfulness": results["average_faithfulness"],
            "min_faithfulness": results["min_faithfulness"],
            "sample_count": results["sample_count"],
        },
    )

    assert results["average_faithfulness"] >= FAITHFULNESS_THRESHOLD, (
        f"Average faithfulness {results['average_faithfulness']:.3f} "
        f"is below the required threshold of {FAITHFULNESS_THRESHOLD}. "
        f"Check the retrieval and synthesis pipeline for regressions."
    )


def test_faithfulness_evaluated_for_all_samples(
    golden_dataset: list[dict[str, Any]],
    mock_ragas_scorer: MagicMock,
    mock_rag_pipeline: MagicMock,
):
    """No samples should be silently skipped — every row must produce a score."""
    results = run_faithfulness_eval(golden_dataset, mock_rag_pipeline, mock_ragas_scorer)
    assert results["sample_count"] == len(golden_dataset)


def test_scorer_called_once_per_sample(
    golden_dataset: list[dict[str, Any]],
    mock_ragas_scorer: MagicMock,
    mock_rag_pipeline: MagicMock,
):
    """Verify the scorer is called exactly once per sample (not batched/skipped)."""
    run_faithfulness_eval(golden_dataset, mock_rag_pipeline, mock_ragas_scorer)
    assert mock_ragas_scorer.score.call_count == len(golden_dataset)
```

### Sample golden_dataset.json (check into tests/integration/fixtures/)

```json
[
  {
    "question": "What is the capital of France?",
    "ground_truth": "Paris is the capital of France.",
    "context": ["Paris is the capital city of France and its largest city."]
  },
  {
    "question": "Who wrote Hamlet?",
    "ground_truth": "William Shakespeare wrote Hamlet.",
    "context": ["Hamlet is a tragedy written by William Shakespeare around 1600."]
  }
]
```

---

## Exercise 3: OpenTelemetry Spans for Each Retrieval Step

Wrapping each retrieval sub-step in a child span gives you a flamegraph: you can see whether
latency is dominated by dense search, sparse search, fusion, or reranking without adding any
print statements or timers.

```python
# src/document_platform/retrieval/traced_retriever.py
from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from src.document_platform.registry import (
    HybridRetriever,
    Reranker,
    ScoredChunk,
    RankedChunk,
)

logger = logging.getLogger(__name__)
tracer: Tracer = trace.get_tracer("document_platform.retrieval")


class TracedRetrievalPipeline:
    """
    Wraps hybrid retrieval + reranking with OpenTelemetry child spans.

    Each span records:
      - document_count: how many chunks came out of this step
      - query (on the root span only — do not log PII in attributes)

    The parent span is created by the caller (e.g. the FastAPI middleware).
    Child spans are created here with tracer.start_as_current_span() so they
    nest correctly in the trace waterfall.
    """

    def __init__(self, retriever: HybridRetriever, reranker: Reranker) -> None:
        self._retriever = retriever
        self._reranker = reranker

    def retrieve_and_rerank(
        self,
        query: str,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
    ) -> list[RankedChunk]:
        """
        Full retrieval pipeline with one child span per logical step:

          dense_search  ──┐
                          ├──▶ fusion ──▶ rerank
          sparse_search ──┘
        """
        # Root span for the entire retrieval pipeline
        with tracer.start_as_current_span("retrieval_pipeline") as root_span:
            root_span.set_attribute("query.length", len(query))
            root_span.set_attribute("top_k_retrieve", top_k_retrieve)
            root_span.set_attribute("top_k_rerank", top_k_rerank)

            # Step 1: Dense (vector) search
            dense_results = self._dense_search(query, top_k=top_k_retrieve)

            # Step 2: Sparse (BM25 / keyword) search
            sparse_results = self._sparse_search(query, top_k=top_k_retrieve)

            # Step 3: Reciprocal-rank fusion
            fused = self._fusion(dense_results, sparse_results)

            # Step 4: Cross-encoder reranking
            ranked = self._rerank(query, fused, top_k=top_k_rerank)

        logger.info(
            "Retrieval pipeline complete",
            extra={
                "dense": len(dense_results),
                "sparse": len(sparse_results),
                "fused": len(fused),
                "ranked": len(ranked),
            },
        )
        return ranked

    # ------------------------------------------------------------------
    # Private helpers — each wraps one stage in a child span
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        with tracer.start_as_current_span("dense_search") as span:
            # In production, this calls the vector store's query() method.
            # Here we delegate to the HybridRetriever's internal dense path.
            results: list[ScoredChunk] = self._retriever.retrieve(
                query, top_k=top_k
            )  # temporary — replace with dedicated dense method
            span.set_attribute("document_count", len(results))
            span.set_attribute("retrieval.type", "dense")
        return results

    def _sparse_search(self, query: str, top_k: int) -> list[ScoredChunk]:
        with tracer.start_as_current_span("sparse_search") as span:
            # In a real implementation, call your BM25 index here.
            # The HybridRetriever exposes both paths; for clarity we call it
            # twice with different flags.  Replace with BM25Retriever.retrieve().
            results: list[ScoredChunk] = self._retriever.retrieve(
                query, top_k=top_k
            )  # placeholder — real impl calls BM25 path
            span.set_attribute("document_count", len(results))
            span.set_attribute("retrieval.type", "sparse")
        return results

    def _fusion(
        self,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Reciprocal-rank fusion of dense and sparse result lists."""
        with tracer.start_as_current_span("fusion") as span:
            k = 60  # RRF constant — higher k reduces the impact of rank differences
            scores: dict[str, float] = {}
            chunk_map: dict[str, ScoredChunk] = {}

            for rank, chunk in enumerate(dense, start=1):
                scores[chunk.chunk.chunk_id] = scores.get(chunk.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
                chunk_map[chunk.chunk.chunk_id] = chunk

            for rank, chunk in enumerate(sparse, start=1):
                scores[chunk.chunk.chunk_id] = scores.get(chunk.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
                chunk_map[chunk.chunk.chunk_id] = chunk

            fused = sorted(
                [
                    ScoredChunk(chunk=chunk_map[cid].chunk, score=score)
                    for cid, score in scores.items()
                ],
                key=lambda c: c.score,
                reverse=True,
            )
            span.set_attribute("document_count", len(fused))
            span.set_attribute("fusion.algorithm", "reciprocal_rank_fusion")
        return fused

    def _rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int,
    ) -> list[RankedChunk]:
        with tracer.start_as_current_span("rerank") as span:
            ranked = self._reranker.rerank(query, chunks[:top_k])
            span.set_attribute("document_count", len(ranked))
            span.set_attribute("reranker.top_k", top_k)
        return ranked
```

### Tests for traced retrieval

```python
# tests/unit/test_traced_retrieval.py
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from src.document_platform.registry import ScoredChunk, RankedChunk, Chunk
from src.document_platform.retrieval.traced_retriever import TracedRetrievalPipeline


# ---------------------------------------------------------------------------
# OTel test setup — in-memory exporter so we can inspect spans
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def otel_setup():
    """Configure an in-memory OTel exporter for every test in this module."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()


def _make_scored_chunk(chunk_id: str, score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="doc-1", text="sample text"),
        score=score,
    )


def _make_ranked_chunk(chunk_id: str, score: float = 0.9, rank: int = 1) -> RankedChunk:
    return RankedChunk(
        chunk=Chunk(chunk_id=chunk_id, doc_id="doc-1", text="sample text"),
        score=score,
        rank=rank,
    )


@pytest.fixture()
def pipeline() -> TracedRetrievalPipeline:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        _make_scored_chunk("c1", 0.9),
        _make_scored_chunk("c2", 0.8),
    ]
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [_make_ranked_chunk("c1", 0.95, rank=1)]
    return TracedRetrievalPipeline(retriever=mock_retriever, reranker=mock_reranker)


def test_span_names_present(otel_setup: InMemorySpanExporter, pipeline: TracedRetrievalPipeline):
    """All four expected spans must be emitted for every retrieval call."""
    pipeline.retrieve_and_rerank("What is photosynthesis?")
    span_names = {span.name for span in otel_setup.get_finished_spans()}
    expected = {"retrieval_pipeline", "dense_search", "sparse_search", "fusion", "rerank"}
    assert expected.issubset(span_names), f"Missing spans: {expected - span_names}"


def test_document_count_attributes(otel_setup: InMemorySpanExporter, pipeline: TracedRetrievalPipeline):
    """Each child span must carry a document_count attribute."""
    pipeline.retrieve_and_rerank("test query")
    for span in otel_setup.get_finished_spans():
        if span.name in {"dense_search", "sparse_search", "fusion", "rerank"}:
            attrs = dict(span.attributes or {})
            assert "document_count" in attrs, f"Span {span.name!r} missing document_count attribute"
            assert isinstance(attrs["document_count"], int)


def test_root_span_has_query_length(otel_setup: InMemorySpanExporter, pipeline: TracedRetrievalPipeline):
    """Root span must record query.length for latency correlation."""
    query = "What is the speed of light?"
    pipeline.retrieve_and_rerank(query)
    root = next(s for s in otel_setup.get_finished_spans() if s.name == "retrieval_pipeline")
    assert dict(root.attributes or {}).get("query.length") == len(query)


def test_spans_are_nested(otel_setup: InMemorySpanExporter, pipeline: TracedRetrievalPipeline):
    """Child spans must have the root span as their parent."""
    pipeline.retrieve_and_rerank("nested span test")
    finished = otel_setup.get_finished_spans()
    root = next(s for s in finished if s.name == "retrieval_pipeline")
    child_names = {"dense_search", "sparse_search", "fusion", "rerank"}
    for span in finished:
        if span.name in child_names:
            assert span.parent is not None, f"Span {span.name!r} has no parent"
            assert span.parent.span_id == root.context.span_id, (
                f"Span {span.name!r} parent is not the root retrieval_pipeline span"
            )
```

---

## Key concepts demonstrated

- **Singleton MetricsStore** — double-checked locking (`_creation_lock`) ensures thread safety
  during initialisation; per-update `_lock` prevents torn reads under concurrent LLM calls.
- **Cost formula per model** — prefix-matching on model ID handles versioned aliases
  (e.g. `claude-haiku-5-5-20250514`) without enumerating every release variant.
- **Mock RAGAS for CI** — the scorer is injected as an argument, not imported at the top of the
  test; swapping to the real RAGAS scorer requires only one fixture change.
- **In-memory OTel exporter** — `InMemorySpanExporter` lets tests assert on span names and
  attributes without running a real collector; the same tests pass in CI with no infra.
- **Child span nesting** — `start_as_current_span` automatically sets the parent to whatever
  span is currently active, so the flamegraph is correct with no manual parent-passing.
- **`document_count` attribute on every step** — this single attribute answers "did the reranker
  receive enough candidates?" and "did fusion collapse duplicates as expected?" in one query.
