# Week 02 Project — Customer Churn Prediction Pipeline

> Full brief in [[day-7]]. This file is the standalone project card for portfolio tracking.

## Title

Customer Churn Prediction Pipeline

## Context

A SaaS company's ML engineer must predict which customers will churn so the success team can intervene proactively. The model runs weekly batch inference on the customer feature store.

## Dataset

Simulated CRM dataset via `make_classification`:
- 2000 samples, 20 features (10 informative, 5 redundant, 5 noise).
- Binary target: 0 = retained, 1 = churner.
- Class weights: 70% / 30% — mild imbalance.
- No download required; purely generated.

## Technical Requirements

| Requirement | Detail |
|-------------|--------|
| Language | Python 3.12+ |
| Framework | scikit-learn |
| Deps | `scikit-learn`, `numpy`, `joblib`, `pytest` (install with `uv add`) |
| Models | ≥3 classifiers compared |
| Preprocessing | `StandardScaler` inside `Pipeline` |
| Evaluation | accuracy, precision, recall, F1, ROC-AUC |
| CV | 5-fold stratified |
| Persistence | `joblib.dump` / `joblib.load` |
| Tests | pytest, ≥8 test cases |
| Logging | `logging` only, no `print` |

## Acceptance Criteria

- [ ] ROC-AUC ≥ 0.85 on held-out test set (20%).
- [ ] Recall ≥ 0.75 for class 1 (churners).
- [ ] 5-fold CV AUC std < 0.04.
- [ ] All tests pass (`pytest -v`).
- [ ] No preprocessing outside pipelines.
- [ ] No `print()` statements.

## File Structure

```
week02_project/
├── __init__.py
├── pipeline.py         # data loading, split, pipeline definitions
├── evaluate.py         # metrics, CV, comparison
├── train.py            # entry point: train → evaluate → save
└── tests/
    ├── __init__.py
    └── test_all.py     # ≥8 test cases
```

## Suggested Approach

1. (30 min) Scaffold files; implement `load_data` and `make_split`; write data-shape tests.
2. (45 min) Implement `build_pipelines` with 3 models; write pipeline-type tests.
3. (45 min) Implement `evaluate.py`; write metric-bounds tests; tune for recall criterion.
4. (30 min) Implement `train.py`; add acceptance assertions; save model.
5. (30 min) Add save/load round-trip test; run all tests; fix any failures.

## Learning Outcomes

After completing this project you will be able to:
- Build a multi-model comparison loop with no data leakage.
- Select evaluation metrics appropriate to a business problem (recall vs. accuracy).
- Write ML tests that enforce both correctness and performance criteria.
- Persist and load a sklearn pipeline with joblib.

## Solution

See `../solutions/week-02/project.md` for the reference implementation and discussion.
