# Phase 4 · Week 07 · Day 7 — Project Day & Week Review

> Objective: Complete and submit the RAG Evaluation Harness project, then consolidate the week's conceptual understanding through self-assessment.  ·  Time: ~3 hr  ·  Prereqs: [[day-6]]

---

## Project: RAG Evaluation Harness

See [[../project]] for full spec.

Recommended time split:

| Activity | Time |
|---|---|
| Finish any incomplete harness components from Day 6 | 60 min |
| Write missing tests; reach full coverage on `metrics.py` and `gate.py` | 40 min |
| Run `ruff check` + `pytest` and fix all failures | 20 min |
| Generate a sample JSON report using the golden dataset | 15 min |
| Self-assessment against acceptance criteria | 15 min |
| Week recap + flashcard review | 30 min |

---

## Week Recap

**Day 1** — Introduced RAG evaluation fundamentals: why automated metrics matter, the difference between retrieval quality and generation quality, and where human evaluation is still necessary.

**Day 2** — Implemented precision@k, recall@k, and MRR from scratch; explored how rank order affects retrieval scores and when MRR is more informative than precision@k.

**Day 3** — Added nDCG@k and token-overlap faithfulness; built `GenerationMetrics` and validated it against hand-computed examples.

**Day 4** — Designed and built the LLM-as-judge with `claude-opus-5-5`; covered structured JSON prompting, Pydantic output validation, and the trade-offs of using a model as a judge.

**Day 5** — Built the golden dataset schema (`GoldenSample`, `EvalDataset`), the regression gate (`RegressionGate`), and the security scanner (`SecurityScanner`) with injection and PII detection.

**Day 6** — Assembled all components into `RAGEvalHarness`; wrote the full integration test suite with mocked LLM calls; verified regression gate, security findings, and JSON report output.

---

## Self-check

- [ ] All tests in `tests/` pass with `pytest -v`
- [ ] `ruff check src/ tests/` reports zero violations
- [ ] `EvalReport` JSON is produced by a real end-to-end run against the golden dataset
- [ ] Regression gate correctly raises `RegressionError` on degraded scores
- [ ] `LLMJudge` is only invoked when `use_llm_judge=True` and is fully mocked in tests
- [ ] `SecurityScanner` flags at least one injection and one PII pattern in dedicated unit tests
- [ ] No `ANTHROPIC_API_KEY` or other secret appears in any committed file
- [ ] `pyproject.toml` declares all runtime dependencies
- [ ] Can explain the difference between `precision@k` and `nDCG@k` without notes
- [ ] Can explain why LLM-as-judge faithfulness and token-overlap faithfulness can disagree
