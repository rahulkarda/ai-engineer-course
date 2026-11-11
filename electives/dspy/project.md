# Project — Optimize a Classification/Extraction Pipeline with DSPy

> Capstone project · Estimated time: ~3 hours
> Prereqs: All 6 lessons complete

---

## Overview

You will build a **support-ticket triage pipeline** that:

1. Extracts structured fields from a raw ticket (category, urgency, sentiment, topic)
2. Generates a one-sentence routing recommendation

You will compare:

- **Baseline**: hand-written prompt via raw Anthropic API call
- **DSPy uncompiled**: same logic via DSPy Signatures
- **DSPy compiled**: BootstrapFewShot-optimized program

You will evaluate all three using the same metric and produce a comparison report.

---

## Learning goals

- [ ] Build a multi-step extraction + classification pipeline in DSPy
- [ ] Write a composite metric covering multiple output fields
- [ ] Compile with BootstrapFewShot and measure improvement
- [ ] Produce a reproducible eval comparison (baseline vs compiled)
- [ ] Write a full pytest suite mocking all LM calls

---

## Dataset

Use the dataset defined in `project_data.py` (you will create this file). It must
contain at least **30 labeled examples** split 20/10 train/test. Each example has:

- `ticket: str` — raw user message
- `category: str` — one of `billing | technical | account | other`
- `urgency: str` — one of `low | medium | high`
- `sentiment: str` — one of `positive | negative | neutral`
- `recommendation: str` — free-text routing recommendation (1 sentence)

You may invent plausible examples. Use variety in phrasing, length, and domain.

---

## Requirements

### 1. Baseline (manual prompt)

File: `pipeline_baseline.py`

Implement `classify_baseline(ticket: str) -> dict` using the Anthropic Python SDK
directly (no DSPy). Use `claude-haiku-5-5`. Your prompt must request all four fields
in a structured format you parse from the response. Return a dict with keys
`category`, `urgency`, `sentiment`, `recommendation`.

### 2. DSPy pipeline

File: `pipeline_dspy.py`

Implement `TriagePipeline(dspy.Module)` with:

- A `dspy.ChainOfThought` step that extracts `category`, `urgency`, `sentiment`
- A separate `dspy.Predict` step that takes the extracted fields + original ticket and
  produces `recommendation`
- Typed output fields using `Literal` for the categorical fields

### 3. Metric

File: `pipeline_metric.py`

Implement `triage_metric(example, pred, trace=None) -> float`:

- `category` exact match: 0.4 weight
- `urgency` exact match: 0.3 weight
- `sentiment` exact match: 0.2 weight
- Recommendation non-empty: 0.1 weight

Score is weighted sum ∈ [0.0, 1.0].

### 4. Compilation

File: `pipeline_compile.py`

Implement `compile_pipeline(trainset) -> dspy.Module` using `BootstrapFewShot` with
`max_bootstrapped_demos=3`. Save the compiled program to `compiled_triage.json`.
Implement `load_pipeline() -> dspy.Module` that loads from the JSON.

### 5. Evaluation

File: `pipeline_eval.py`

Implement `run_comparison(testset) -> dict` that returns:

```python
{
    "baseline": float,      # average metric score over testset
    "dspy_uncompiled": float,
    "dspy_compiled": float,
}
```

Use `dspy.Evaluate` for the DSPy variants. For the baseline, call `classify_baseline`
manually and compute the metric score per example.

Log the comparison table using Python's `logging` module.

### 6. Tests

File: `tests/test_pipeline.py`

Cover:

- `TriagePipeline.forward` with mocked inner predictors
- `triage_metric` with various prediction/gold combinations (no LM calls)
- `compile_pipeline` with mocked `BootstrapFewShot.compile`
- `load_pipeline` with a temporary JSON fixture (use `tmp_path`)
- `run_comparison` with fully mocked LM calls — assert all three keys are present

---

## Acceptance criteria

| Criterion | Pass condition |
|---|---|
| Pipeline runs end-to-end | `pipeline_eval.py` exits 0 with ANTHROPIC_API_KEY set |
| Compiled ≥ uncompiled | DSPy compiled score ≥ DSPy uncompiled on testset |
| Tests pass | `pytest tests/ -v` exits 0 with no real LM calls |
| Metric covers all fields | Metric function handles missing/empty output fields gracefully |
| Type hints everywhere | `mypy --strict pipeline_dspy.py` (or equivalent ruff check) passes |
| No hardcoded keys | `grep -r "sk-ant" .` returns nothing |

---

## Suggested file layout

```
project/
├── project_data.py          # 30 labeled examples
├── pipeline_baseline.py     # raw Anthropic SDK baseline
├── pipeline_dspy.py         # TriagePipeline module
├── pipeline_metric.py       # triage_metric function
├── pipeline_compile.py      # compile/load helpers
├── pipeline_eval.py         # run_comparison + main
├── compiled_triage.json     # output of compilation (gitignore in real project)
└── tests/
    └── test_pipeline.py     # full pytest suite
```

---

## Extension challenges (optional)

1. **MIPRO** — Run `MIPROv2` with `auto="light"` on the same trainset. Does it beat
   `BootstrapFewShot`? Log both scores in `run_comparison`.

2. **LLM-as-judge for recommendation** — The `recommendation` field is free text.
   Replace its binary "non-empty" check with an LLM-as-judge sub-score. Mock it in
   tests.

3. **LangGraph integration** — Wire `TriagePipeline` into a LangGraph graph (as in
   lesson 6) with routing nodes for each category. Add a human-in-the-loop step for
   high-urgency tickets.

4. **Streaming eval** — Write an async version of `run_comparison` that evaluates all
   three variants concurrently using `asyncio.gather`.

---

## Reference implementation

[solutions/project.md](solutions/project.md)

---

## Self-check

- [ ] I built a multi-step DSPy pipeline with extraction and generation steps
- [ ] My metric covers all output fields and handles edge cases
- [ ] I demonstrated measurable improvement from compilation
- [ ] All tests pass without real LM calls
- [ ] I can articulate in 3 sentences when DSPy optimization is worth the cost
