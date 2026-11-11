# Week 07 Project — RAG Evaluation Harness

---

## Goal

Build a complete, production-quality evaluation harness for any RAG system. The harness must accept a RAG callable, load a golden dataset, compute retrieval and generation metrics, run a security scan on retrieved documents, compare results against a stored baseline, and write a JSON report. All LLM calls must be mockable so the full test suite runs without network access.

---

## Deliverables

1. Python package `rag_eval` under `src/` with six modules: `metrics.py`, `judge.py`, `dataset.py`, `security.py`, `gate.py`, `harness.py`
2. Pytest test suite under `tests/` with one test file per module
3. Golden dataset JSON file in `golden_data/week07_golden.json` (minimum 3 samples)
4. Baseline snapshot created by the first clean run under `baselines/`
5. `pyproject.toml` with all runtime dependencies declared
6. `.env.example` listing all required environment variables
7. A `README.md` with a quickstart (install, set env var, run tests, run evaluation)

---

## Spec

### RAG System Interface

The harness accepts any callable matching this signature:

```python
from collections.abc import Callable

RAGCallable = Callable[[str], tuple[str, list[str], list[str]]]
# (query) -> (answer, context_docs, retrieved_doc_ids)
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `metrics.py` | `RetrievalMetrics` (precision@k, recall@k, MRR, nDCG@k) and `GenerationMetrics` (token-overlap faithfulness) |
| `judge.py` | `LLMJudge` — calls `claude-opus-5-5` via `os.getenv("ANTHROPIC_API_KEY")`, returns `JudgeScore` (faithfulness, answer_relevance) |
| `dataset.py` | `GoldenSample` and `EvalDataset` Pydantic models; `EvalDataset.from_json(path)` loader |
| `security.py` | `SecurityScanner.scan(docs)` — injection pattern detection (regex) and PII detection (email, phone, SSN, credit card); returns `SecurityReport` |
| `gate.py` | `RegressionGate` — loads baseline JSON, compares current metrics, raises `RegressionError` when any metric drops more than `threshold` (default 0.05) |
| `harness.py` | `RAGEvalHarness` — orchestrates all modules, returns `EvalReport` Pydantic model |

### EvalReport Schema

```python
class EvalReport(BaseModel):
    dataset_name: str
    num_samples: int
    aggregate: dict[str, float]        # mean scores across all samples
    per_sample: list[SampleResult]
    security_summary: dict[str, int]   # {"injection": N, "pii": N}
    regression_passed: bool
    regression_error: str | None
```

### LLM Judge Contract

- Model: `claude-opus-5-5`
- API key: `os.getenv("ANTHROPIC_API_KEY")` — raise `EnvironmentError` if absent
- Output: JSON object `{"faithfulness": float, "answer_relevance": float, "reasoning": str}`
- Validation: Pydantic `JudgeScore` with `Field(ge=0.0, le=1.0)` on both floats
- Tests: fully mocked — no real API calls in the test suite

### Regression Gate Behaviour

1. First run: no baseline file exists — gate writes a new baseline and passes.
2. Subsequent runs: gate loads baseline, computes delta per metric.
3. If `delta < -threshold` for any metric: raises `RegressionError` with a message listing every failing metric.
4. Callers can call `gate.update_baseline(...)` to deliberately advance the baseline after an approved score change.

### Security Scanner Patterns

Injection (any match = `finding_type="injection"`):
- "ignore [all] previous instructions"
- "disregard [your] [system/prior] prompt"
- "you are now"
- "act as a/an" (not followed by "retrieval" or "search")
- "forget everything / your instructions"
- Special token patterns: `<|...|>`, `[INST]`, `[/INST]`

PII (`finding_type="pii"`, `matched_text` stored as `"[REDACTED]"` — never log actual PII):
- Email addresses
- US phone numbers
- US Social Security Numbers (NNN-NN-NNNN)
- Credit card numbers (13–16 digits)

---

## Acceptance Criteria

- [ ] `pytest tests/` passes with zero failures and zero warnings
- [ ] `ruff check src/ tests/` reports zero violations
- [ ] `EvalReport.model_dump_json(indent=2)` produces valid JSON for a real run against the golden dataset
- [ ] Regression gate raises `RegressionError` when scores drop more than 5% (demonstrated by test)
- [ ] `LLMJudge` is never called in the test suite (confirmed by `assert mock_judge.score.call_count == N`)
- [ ] `SecurityScanner` flags at least one injection and one PII finding in dedicated unit tests
- [ ] `ANTHROPIC_API_KEY` does not appear as a literal in any source file
- [ ] `.env` is listed in `.gitignore`
- [ ] All public functions and methods have type annotations
- [ ] No module exceeds 300 lines (split if needed)

---

## Stretch Goals

**S1 — Async judge.** Refactor `LLMJudge.score` into an `async def ascore(...)` method and update `RAGEvalHarness` to call all samples concurrently using `asyncio.gather`. Measure wall-clock speedup on the 3-sample golden dataset.

**S2 — BLEU-1 metric.** Add `bleu_1` (unigram precision between `answer` and `ground_truth_answer`) to `GenerationMetrics.as_dict()`. Include it in `EvalReport.aggregate`.

**S3 — HTML report.** Add a `report.to_html(path)` method that writes a minimal static HTML page with a table of per-sample scores and a bar chart (use only the standard library — no matplotlib or charting deps).

**S4 — CI integration.** Add a `scripts/run_eval.sh` that loads `.env`, runs `pytest`, then runs the harness against the golden dataset, and exits with code 1 if either step fails or the regression gate triggers. Wire it into a GitHub Actions workflow file.

**S5 — Semantic similarity faithfulness.** Replace token-overlap faithfulness with a cosine similarity score using sentence embeddings (e.g. `sentence-transformers`). Compare the two metrics on the golden dataset and document cases where they disagree.
