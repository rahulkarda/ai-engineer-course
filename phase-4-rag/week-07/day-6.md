# Phase 4 · Week 07 · Day 6 — Build the RAG Evaluation Harness (4-Hour Lab)

> Objective: Assemble all week's components into a single, runnable RAG evaluation harness with retrieval metrics, generation metrics, LLM judge, golden dataset, and regression gate.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]]..[[day-5]]

---

## Overview

Today is the capstone lab for Week 7. You will wire together every concept covered this week into one cohesive, production-ready evaluation harness. By the end of the session you will have:

- A fully typed Python package `rag_eval` with six modules
- Retrieval metrics (precision@k, recall@k, MRR, nDCG@k)
- Generation metrics (token-overlap faithfulness)
- An LLM judge that calls `claude-opus-5-5` and returns structured JSON scores
- A golden dataset loaded from JSON using Pydantic
- A regression gate that fails the CI run when scores drop beyond a threshold
- A security layer that detects prompt injection patterns and PII in retrieved docs
- A top-level `RAGEvalHarness` that orchestrates all of the above
- A full pytest suite with mocked LLM calls

Work through the parts in order. Each part ends with a mini-verification step so you catch issues before they compound.

---

## Part 1: Project Setup (~30 min)

### 1.1 Directory Layout

Create the following structure from your repo root:

```
rag_eval_harness/
    src/
        rag_eval/
            __init__.py
            metrics.py
            judge.py
            dataset.py
            security.py
            gate.py
            harness.py
    tests/
        __init__.py
        test_metrics.py
        test_judge.py
        test_dataset.py
        test_security.py
        test_gate.py
        test_harness.py
    baselines/
        week07_baseline.json        # created in Part 4
    golden_data/
        week07_golden.json          # created in Part 4
    pyproject.toml
    .env.example
    .gitignore
```

### 1.2 pyproject.toml Snippet

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "rag-eval"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "anthropic>=0.28",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 1.3 .env.example

```
ANTHROPIC_API_KEY=your-api-key-here
RAG_EVAL_LOG_LEVEL=INFO
```

### 1.4 Verify

```bash
pip install -e .
python -c "import rag_eval; print('ok')"
```

---

## Part 2: Core Metrics Module (~60 min)

File: `src/rag_eval/metrics.py`

### Concepts Recap

| Metric | What it measures |
|---|---|
| Precision@k | Fraction of top-k retrieved docs that are relevant |
| Recall@k | Fraction of relevant docs captured in top-k |
| MRR | Mean reciprocal rank of the first relevant doc |
| nDCG@k | Gain discounted by rank position |
| Token-overlap faithfulness | How many answer tokens appear in retrieved context |

### Implementation Skeleton

```python
"""Retrieval and generation metrics for RAG evaluation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Computes retrieval quality metrics for a single query."""

    relevant_ids: list[str]
    retrieved_ids: list[str]
    k: int = 3

    def precision_at_k(self) -> float:
        """Fraction of top-k retrieved docs that are relevant."""
        top_k = self.retrieved_ids[: self.k]
        if not top_k:
            return 0.0
        relevant_set = set(self.relevant_ids)
        hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
        score = hits / len(top_k)
        logger.debug("precision@%d = %.4f", self.k, score)
        return score

    def recall_at_k(self) -> float:
        """Fraction of all relevant docs captured in top-k."""
        if not self.relevant_ids:
            return 0.0
        top_k = set(self.retrieved_ids[: self.k])
        relevant_set = set(self.relevant_ids)
        hits = len(top_k & relevant_set)
        score = hits / len(relevant_set)
        logger.debug("recall@%d = %.4f", self.k, score)
        return score

    def mrr(self) -> float:
        """Mean reciprocal rank of the first relevant document."""
        relevant_set = set(self.relevant_ids)
        for rank, doc_id in enumerate(self.retrieved_ids, start=1):
            if doc_id in relevant_set:
                score = 1.0 / rank
                logger.debug("MRR = %.4f (first hit at rank %d)", score, rank)
                return score
        return 0.0

    def ndcg_at_k(self) -> float:
        """Normalised discounted cumulative gain at k."""
        relevant_set = set(self.relevant_ids)
        top_k = self.retrieved_ids[: self.k]

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(top_k, start=1)
            if doc_id in relevant_set
        )

        ideal_hits = min(len(relevant_set), self.k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        score = dcg / idcg if idcg > 0 else 0.0
        logger.debug("nDCG@%d = %.4f", self.k, score)
        return score

    def as_dict(self) -> dict[str, float]:
        return {
            f"precision_at_{self.k}": self.precision_at_k(),
            f"recall_at_{self.k}": self.recall_at_k(),
            "mrr": self.mrr(),
            f"ndcg_at_{self.k}": self.ndcg_at_k(),
        }


@dataclass
class GenerationMetrics:
    """Token-overlap based generation quality metrics."""

    answer: str
    context_docs: list[str]
    ground_truth: str = ""

    def _tokenize(self, text: str) -> set[str]:
        return set(text.lower().split())

    def token_overlap_faithfulness(self) -> float:
        """Fraction of answer tokens present in retrieved context."""
        answer_tokens = self._tokenize(self.answer)
        if not answer_tokens:
            return 0.0
        context_text = " ".join(self.context_docs)
        context_tokens = self._tokenize(context_text)
        overlap = answer_tokens & context_tokens
        score = len(overlap) / len(answer_tokens)
        logger.debug("token_overlap_faithfulness = %.4f", score)
        return score

    def as_dict(self) -> dict[str, float]:
        return {
            "token_overlap_faithfulness": self.token_overlap_faithfulness(),
        }
```

### Mini-Verification

Write a quick smoke test in your REPL:

```python
from rag_eval.metrics import RetrievalMetrics, GenerationMetrics

rm = RetrievalMetrics(relevant_ids=["d1", "d2"], retrieved_ids=["d1", "d3", "d2"], k=3)
assert rm.precision_at_k() == pytest.approx(2/3)
assert rm.recall_at_k() == 1.0
assert rm.mrr() == 1.0

gm = GenerationMetrics(answer="the cat sat", context_docs=["the cat sat on the mat"])
assert gm.token_overlap_faithfulness() >= 0.5
print("metrics ok")
```

---

## Part 3: LLM Judge Integration (~60 min)

File: `src/rag_eval/judge.py`

### Design

The LLM judge calls `claude-opus-5-5` with a structured prompt and parses the JSON response. Use a Pydantic model for the output to catch malformed responses early.

```python
"""LLM-as-judge module using Anthropic claude-opus-5-5."""

from __future__ import annotations

import json
import logging
import os

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = """
You are an expert RAG evaluation judge. Given a query, retrieved context, and generated answer,
score two dimensions on a scale of 0 to 1 (float, two decimal places).

faithfulness: Is the answer grounded in the retrieved context? 1 = fully supported, 0 = hallucinated.
answer_relevance: Does the answer address the query? 1 = fully addresses it, 0 = completely off-topic.

Respond with valid JSON only, no markdown fences.
{"faithfulness": <float>, "answer_relevance": <float>, "reasoning": "<one sentence>"}
""".strip()


class JudgeScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    reasoning: str


class LLMJudge:
    """Calls claude-opus-5-5 to score faithfulness and answer relevance."""

    def __init__(self, model: str = "claude-opus-5-5") -> None:
        self.model = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info("LLMJudge initialised with model=%s", self.model)

    def score(
        self,
        query: str,
        context_docs: list[str],
        answer: str,
    ) -> JudgeScore:
        """Return faithfulness and answer_relevance scores for a single (query, context, answer) triple."""
        context_block = "\n\n".join(
            f"[Doc {i + 1}]\n{doc}" for i, doc in enumerate(context_docs)
        )
        user_message = (
            f"Query: {query}\n\n"
            f"Retrieved Context:\n{context_block}\n\n"
            f"Generated Answer: {answer}"
        )

        logger.debug("Sending judge request to %s", self.model)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text.strip()
        logger.debug("Raw judge response: %s", raw_text)

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Judge returned non-JSON: %s", raw_text)
            raise ValueError(f"LLM judge returned non-JSON response: {raw_text}") from exc

        score = JudgeScore(**data)
        logger.info(
            "Judge scores — faithfulness=%.2f, answer_relevance=%.2f",
            score.faithfulness,
            score.answer_relevance,
        )
        return score
```

### Prompt Engineering Notes

- The system prompt pins the output schema to avoid free-form text.
- `max_tokens=256` is enough for one JSON object; avoids wasted tokens.
- Pydantic validation on `JudgeScore` catches out-of-range floats immediately.

### Mini-Verification

Run the test in `tests/test_judge.py` (written in Part 6) after mocking the Anthropic client to confirm parsing works without a real API call.

---

## Part 4: Golden Dataset + Regression Gate (~60 min)

### 4.1 Dataset Module

File: `src/rag_eval/dataset.py`

```python
"""Golden dataset loader and Pydantic data models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GoldenSample(BaseModel):
    """One ground-truth evaluation record."""

    id: str
    query: str
    relevant_doc_ids: list[str] = Field(min_length=1)
    ground_truth_answer: str
    metadata: dict[str, str] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """Collection of golden samples with optional dataset-level metadata."""

    name: str
    version: str = "1.0.0"
    samples: list[GoldenSample] = Field(min_length=1)

    @classmethod
    def from_json(cls, path: Path | str) -> "EvalDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        dataset = cls(**raw)
        logger.info("Loaded dataset '%s' v%s with %d samples", dataset.name, dataset.version, len(dataset.samples))
        return dataset
```

### 4.2 Golden Dataset JSON

Save as `golden_data/week07_golden.json`:

```json
{
  "name": "week07_rag_golden",
  "version": "1.0.0",
  "samples": [
    {
      "id": "q001",
      "query": "What is retrieval-augmented generation?",
      "relevant_doc_ids": ["doc_rag_intro", "doc_rag_overview"],
      "ground_truth_answer": "RAG combines a retrieval step with a generation step so the LLM grounds its answer in retrieved documents."
    },
    {
      "id": "q002",
      "query": "How does nDCG differ from precision@k?",
      "relevant_doc_ids": ["doc_metrics_ndcg"],
      "ground_truth_answer": "nDCG accounts for the rank position of relevant documents, while precision@k only counts how many of the top-k are relevant regardless of order."
    },
    {
      "id": "q003",
      "query": "What is prompt injection in RAG?",
      "relevant_doc_ids": ["doc_security_injection"],
      "ground_truth_answer": "Prompt injection in RAG occurs when malicious content inside retrieved documents instructs the LLM to ignore its system prompt or perform unintended actions."
    }
  ]
}
```

### 4.3 Regression Gate

File: `src/rag_eval/gate.py`

```python
"""Regression gate: compare current run scores against a stored baseline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RegressionError(Exception):
    """Raised when one or more metrics drop beyond the allowed threshold."""


class BaselineRecord(BaseModel):
    """Persisted metric snapshot."""

    dataset_name: str
    metrics: dict[str, float]


class RegressionGate:
    """Compares an incoming metric dict against a stored baseline."""

    def __init__(self, baseline_path: Path | str, threshold: float = 0.05) -> None:
        self.baseline_path = Path(baseline_path)
        self.threshold = threshold
        self._baseline: BaselineRecord | None = None
        if self.baseline_path.exists():
            self._baseline = BaselineRecord(**json.loads(self.baseline_path.read_text(encoding="utf-8")))
            logger.info("Loaded baseline from %s", self.baseline_path)
        else:
            logger.warning("No baseline file found at %s — first run will create it", self.baseline_path)

    def check(self, dataset_name: str, current_metrics: dict[str, float]) -> None:
        """Raise RegressionError if any metric drops more than self.threshold below baseline."""
        if self._baseline is None:
            self._save(dataset_name, current_metrics)
            logger.info("Baseline created at %s", self.baseline_path)
            return

        failures: list[str] = []
        for key, current_value in current_metrics.items():
            baseline_value = self._baseline.metrics.get(key)
            if baseline_value is None:
                logger.debug("No baseline for metric '%s' — skipping", key)
                continue
            delta = current_value - baseline_value
            if delta < -self.threshold:
                msg = f"{key}: {baseline_value:.4f} -> {current_value:.4f} (delta={delta:+.4f}, threshold={-self.threshold:.4f})"
                failures.append(msg)
                logger.error("Regression detected: %s", msg)

        if failures:
            raise RegressionError("Metric regression(s) detected:\n" + "\n".join(failures))

        logger.info("Regression gate passed — all metrics within threshold")

    def update_baseline(self, dataset_name: str, metrics: dict[str, float]) -> None:
        """Overwrite the baseline with new values (call after a deliberate score change)."""
        self._save(dataset_name, metrics)
        logger.info("Baseline updated at %s", self.baseline_path)

    def _save(self, dataset_name: str, metrics: dict[str, float]) -> None:
        record = BaselineRecord(dataset_name=dataset_name, metrics=metrics)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.baseline_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        self._baseline = record
```

### Mini-Verification

```python
from rag_eval.gate import RegressionGate, RegressionError
import tempfile, pathlib

with tempfile.TemporaryDirectory() as tmp:
    gate = RegressionGate(pathlib.Path(tmp) / "baseline.json")
    gate.check("test", {"precision_at_3": 0.8, "mrr": 0.9})   # creates baseline
    gate.check("test", {"precision_at_3": 0.79, "mrr": 0.88}) # within 5%
    try:
        gate.check("test", {"precision_at_3": 0.5, "mrr": 0.5}) # regression
        assert False, "should have raised"
    except RegressionError:
        print("gate ok")
```

---

## Part 5: Security Layer (~30 min)

File: `src/rag_eval/security.py`

```python
"""Security scanning for retrieved documents: injection detection and PII patterns."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (all )?previous instructions?", re.IGNORECASE),
    re.compile(r"disregard (your )?(system |prior )?prompt", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"act as (a |an )?(?!retrieval|search)", re.IGNORECASE),
    re.compile(r"forget (everything|your instructions)", re.IGNORECASE),
    re.compile(r"<\|.*?\|>"),                            # special token injection
    re.compile(r"\[INST\]|\[\/INST\]"),                 # Llama instruction tokens
]

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


@dataclass
class SecurityFinding:
    doc_index: int
    finding_type: str            # "injection" or "pii"
    pattern_name: str
    matched_text: str


@dataclass
class SecurityReport:
    findings: list[SecurityFinding] = field(default_factory=list)

    @property
    def has_injection(self) -> bool:
        return any(f.finding_type == "injection" for f in self.findings)

    @property
    def has_pii(self) -> bool:
        return any(f.finding_type == "pii" for f in self.findings)

    @property
    def is_clean(self) -> bool:
        return not self.findings


class SecurityScanner:
    """Scans a list of document strings for injection attempts and PII."""

    def scan(self, docs: list[str]) -> SecurityReport:
        report = SecurityReport()
        for idx, doc in enumerate(docs):
            self._scan_injection(doc, idx, report)
            self._scan_pii(doc, idx, report)
        if report.is_clean:
            logger.debug("Security scan clean for %d docs", len(docs))
        else:
            logger.warning(
                "Security findings: %d total (%d injection, %d pii)",
                len(report.findings),
                sum(1 for f in report.findings if f.finding_type == "injection"),
                sum(1 for f in report.findings if f.finding_type == "pii"),
            )
        return report

    def _scan_injection(self, doc: str, idx: int, report: SecurityReport) -> None:
        for pattern in _INJECTION_PATTERNS:
            match = pattern.search(doc)
            if match:
                finding = SecurityFinding(
                    doc_index=idx,
                    finding_type="injection",
                    pattern_name=pattern.pattern,
                    matched_text=match.group(0),
                )
                report.findings.append(finding)
                logger.warning("Injection pattern detected in doc %d: '%s'", idx, match.group(0))

    def _scan_pii(self, doc: str, idx: int, report: SecurityReport) -> None:
        for name, pattern in _PII_PATTERNS.items():
            match = pattern.search(doc)
            if match:
                finding = SecurityFinding(
                    doc_index=idx,
                    finding_type="pii",
                    pattern_name=name,
                    matched_text="[REDACTED]",   # never log actual PII
                )
                report.findings.append(finding)
                logger.warning("PII type '%s' detected in doc %d", name, idx)
```

---

## Part 6: Wire It All Together (~30 min)

### 6.1 EvalReport + RAGEvalHarness

File: `src/rag_eval/harness.py`

```python
"""Top-level RAG evaluation harness."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from rag_eval.dataset import EvalDataset
from rag_eval.gate import RegressionGate
from rag_eval.judge import LLMJudge
from rag_eval.metrics import GenerationMetrics, RetrievalMetrics
from rag_eval.security import SecurityReport, SecurityScanner

logger = logging.getLogger(__name__)

# Type alias: a RAG system callable accepts a query string and returns
# (answer: str, docs: list[str], doc_ids: list[str])
RAGCallable = Callable[[str], tuple[str, list[str], list[str]]]


class SampleResult(BaseModel):
    sample_id: str
    retrieval: dict[str, float]
    generation: dict[str, float]
    judge_faithfulness: float | None = None
    judge_answer_relevance: float | None = None
    security_clean: bool = True
    security_findings_count: int = 0


class EvalReport(BaseModel):
    dataset_name: str
    num_samples: int
    aggregate: dict[str, float] = Field(default_factory=dict)
    per_sample: list[SampleResult] = Field(default_factory=list)
    security_summary: dict[str, int] = Field(default_factory=dict)
    regression_passed: bool = True
    regression_error: str | None = None


class RAGEvalHarness:
    """Orchestrates retrieval, generation, judge, security, and regression checks."""

    def __init__(
        self,
        rag_system: RAGCallable,
        dataset_path: Path | str,
        baseline_path: Path | str,
        k: int = 3,
        regression_threshold: float = 0.05,
        use_llm_judge: bool = True,
    ) -> None:
        self._rag = rag_system
        self._dataset = EvalDataset.from_json(dataset_path)
        self._gate = RegressionGate(baseline_path, threshold=regression_threshold)
        self._scanner = SecurityScanner()
        self._judge = LLMJudge() if use_llm_judge else None
        self._k = k
        logger.info(
            "RAGEvalHarness ready — dataset='%s', samples=%d, k=%d, llm_judge=%s",
            self._dataset.name,
            len(self._dataset.samples),
            self._k,
            use_llm_judge,
        )

    def run(self) -> EvalReport:
        """Execute the full evaluation suite and return an EvalReport."""
        results: list[SampleResult] = []
        total_injection = 0
        total_pii = 0

        for sample in self._dataset.samples:
            logger.info("Evaluating sample id=%s", sample.id)
            answer, docs, doc_ids = self._rag(sample.query)

            # Retrieval metrics
            rm = RetrievalMetrics(
                relevant_ids=sample.relevant_doc_ids,
                retrieved_ids=doc_ids,
                k=self._k,
            )
            retrieval_scores = rm.as_dict()

            # Generation metrics
            gm = GenerationMetrics(
                answer=answer,
                context_docs=docs,
                ground_truth=sample.ground_truth_answer,
            )
            generation_scores = gm.as_dict()

            # LLM judge
            judge_f: float | None = None
            judge_ar: float | None = None
            if self._judge is not None:
                try:
                    js = self._judge.score(sample.query, docs, answer)
                    judge_f = js.faithfulness
                    judge_ar = js.answer_relevance
                except Exception:
                    logger.exception("LLM judge failed for sample %s", sample.id)

            # Security
            sec_report: SecurityReport = self._scanner.scan(docs)
            injection_count = sum(1 for f in sec_report.findings if f.finding_type == "injection")
            pii_count = sum(1 for f in sec_report.findings if f.finding_type == "pii")
            total_injection += injection_count
            total_pii += pii_count

            results.append(
                SampleResult(
                    sample_id=sample.id,
                    retrieval=retrieval_scores,
                    generation=generation_scores,
                    judge_faithfulness=judge_f,
                    judge_answer_relevance=judge_ar,
                    security_clean=sec_report.is_clean,
                    security_findings_count=len(sec_report.findings),
                )
            )

        # Aggregate
        aggregate = self._aggregate(results)

        # Regression gate
        passed = True
        reg_error: str | None = None
        try:
            self._gate.check(self._dataset.name, aggregate)
        except Exception as exc:
            passed = False
            reg_error = str(exc)
            logger.error("Regression gate FAILED: %s", reg_error)

        report = EvalReport(
            dataset_name=self._dataset.name,
            num_samples=len(results),
            aggregate=aggregate,
            per_sample=results,
            security_summary={"injection": total_injection, "pii": total_pii},
            regression_passed=passed,
            regression_error=reg_error,
        )
        logger.info("Evaluation complete — regression_passed=%s", passed)
        return report

    def _aggregate(self, results: list[SampleResult]) -> dict[str, float]:
        if not results:
            return {}
        keys = list(results[0].retrieval.keys()) + list(results[0].generation.keys())
        agg: dict[str, float] = {}
        for key in keys:
            vals = []
            for r in results:
                combined = {**r.retrieval, **r.generation}
                if key in combined:
                    vals.append(combined[key])
            agg[key] = sum(vals) / len(vals) if vals else 0.0

        # Judge aggregate (skip None)
        judge_f_vals = [r.judge_faithfulness for r in results if r.judge_faithfulness is not None]
        judge_ar_vals = [r.judge_answer_relevance for r in results if r.judge_answer_relevance is not None]
        if judge_f_vals:
            agg["judge_faithfulness"] = sum(judge_f_vals) / len(judge_f_vals)
        if judge_ar_vals:
            agg["judge_answer_relevance"] = sum(judge_ar_vals) / len(judge_ar_vals)

        return agg
```

### 6.2 Integration Test with Mocked RAG + Mocked LLM

File: `tests/test_harness.py`

```python
"""Integration test for RAGEvalHarness with mocked RAG system and LLM judge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_eval.harness import RAGEvalHarness
from rag_eval.judge import JudgeScore


GOLDEN_DATA = {
    "name": "test_golden",
    "version": "1.0.0",
    "samples": [
        {
            "id": "q001",
            "query": "What is RAG?",
            "relevant_doc_ids": ["d1", "d2"],
            "ground_truth_answer": "RAG retrieves documents then generates an answer.",
        }
    ],
}


def _make_rag_system(answer: str, docs: list[str], doc_ids: list[str]):
    def rag(query: str) -> tuple[str, list[str], list[str]]:
        return answer, docs, doc_ids
    return rag


@pytest.fixture()
def temp_paths(tmp_path: Path):
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(GOLDEN_DATA), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    return golden_path, baseline_path


def test_harness_run_no_judge(temp_paths):
    golden_path, baseline_path = temp_paths
    rag = _make_rag_system(
        answer="RAG retrieves documents then generates an answer.",
        docs=["RAG retrieves documents then generates an answer."],
        doc_ids=["d1", "d2"],
    )
    harness = RAGEvalHarness(
        rag_system=rag,
        dataset_path=golden_path,
        baseline_path=baseline_path,
        use_llm_judge=False,
    )
    report = harness.run()
    assert report.num_samples == 1
    assert report.regression_passed is True
    assert "precision_at_3" in report.aggregate


def test_harness_run_with_mocked_judge(temp_paths):
    golden_path, baseline_path = temp_paths
    rag = _make_rag_system(
        answer="RAG retrieves documents then generates an answer.",
        docs=["RAG retrieves docs and generates answers."],
        doc_ids=["d1"],
    )
    mock_score = JudgeScore(faithfulness=0.9, answer_relevance=0.85, reasoning="Mostly grounded.")

    with patch("rag_eval.harness.LLMJudge") as MockJudge:
        instance = MockJudge.return_value
        instance.score.return_value = mock_score
        harness = RAGEvalHarness(
            rag_system=rag,
            dataset_path=golden_path,
            baseline_path=baseline_path,
            use_llm_judge=True,
        )
        report = harness.run()

    assert report.aggregate.get("judge_faithfulness") == pytest.approx(0.9)
    assert report.aggregate.get("judge_answer_relevance") == pytest.approx(0.85)


def test_harness_regression_gate_triggers(temp_paths):
    golden_path, baseline_path = temp_paths
    rag_good = _make_rag_system(
        answer="RAG retrieves documents then generates an answer.",
        docs=["RAG retrieves documents then generates an answer."],
        doc_ids=["d1", "d2"],
    )
    rag_bad = _make_rag_system(
        answer="I have no idea.",
        docs=["Unrelated content about penguins."],
        doc_ids=["d99"],
    )

    harness_good = RAGEvalHarness(rag_good, golden_path, baseline_path, use_llm_judge=False)
    harness_good.run()  # creates baseline

    harness_bad = RAGEvalHarness(rag_bad, golden_path, baseline_path, use_llm_judge=False)
    report = harness_bad.run()
    assert report.regression_passed is False
    assert report.regression_error is not None
```

---

## Self-check

- [ ] `pip install -e .` completes without errors
- [ ] `pytest tests/` — all tests pass
- [ ] `ruff check src/ tests/` — zero violations
- [ ] `RetrievalMetrics` returns correct values for a hand-calculated example
- [ ] `SecurityScanner` detects "ignore all previous instructions" in a doc
- [ ] Regression gate raises `RegressionError` when scores drop more than 5%
- [ ] `LLMJudge.score` is never called during tests (confirmed by mock assertions)
- [ ] `EvalReport` serialises to JSON without error via `.model_dump_json()`
- [ ] No hardcoded API keys anywhere in the codebase
- [ ] `.env` is listed in `.gitignore`

---

## Exercises

**Exercise 1 — Extend the harness with BLEU-1.**
Add a `bleu_1` score to `GenerationMetrics` that computes the unigram precision between `answer` and `ground_truth`. Wire it into `as_dict()` so it appears in `EvalReport.aggregate`. Write at least two pytest test cases.

**Exercise 2 — Add a custom metric plugin.**
Modify `RAGEvalHarness.__init__` to accept an optional `extra_metrics: list[Callable[[SampleResult], dict[str, float]]] = []` parameter. Each callable receives a `SampleResult` and returns additional key-value scores to merge into `aggregate`. Write a test that passes a lambda returning `{"custom_score": 1.0}`.

**Exercise 3 — Extend the security scanner.**
Add a check for documents that appear to be base64-encoded (matching `re.compile(r'^[A-Za-z0-9+/]{40,}={0,2}$', re.MULTILINE)`). Such documents may be obfuscated injection payloads. Add a new `finding_type = "obfuscation"` and write a pytest test that catches a synthetic case.

---

## Supplementary Links

- [RAGAS paper (arXiv 2309.15217)](https://arxiv.org/abs/2309.15217) — foundational RAG evaluation framework
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) — reference for `client.messages.create`
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/) — model validation
- [OWASP LLM Top 10 — LLM01: Prompt Injection](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — security context
- [nDCG explained (Wikipedia)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
