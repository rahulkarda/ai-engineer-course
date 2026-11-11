# Phase 3 · Week 02 · Day 7 — Project: End-to-End ML Pipeline + Assessment

> Objective: Complete the week project independently, then sit the assessment.  ·  Time: ~4 hr project + 1 hr assessment  ·  Prereqs: [[day-6]]

## Project Brief

**Title**: Customer Churn Prediction Pipeline

**Dataset**: `make_classification` (simulated CRM data — no download needed):

```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],   # 30% churners — mild imbalance
    random_state=99,
)
```

**Business context**: You are an ML engineer at a SaaS company. The churn prediction model will flag at-risk customers for the success team. The business cares more about catching churners (recall) than avoiding false alarms, but F1 must also be reasonable.

**Acceptance criteria**:
- ROC-AUC ≥ 0.85 on held-out test set (20%).
- Recall ≥ 0.75 for class 1 (churners).
- 5-fold CV AUC std < 0.04.
- All tests pass (`pytest -v`).
- No preprocessing outside pipelines.
- `logging` only — no `print`.

## Deliverables

Produce the following files inside `week02_project/`:

```
week02_project/
├── __init__.py
├── pipeline.py         # data loading, split, pipeline definitions
├── evaluate.py         # metrics, CV, comparison
├── train.py            # entry point: train → evaluate → save
└── tests/
    ├── __init__.py
    └── test_all.py     # pytest suite, ≥8 test cases
```

## Requirements

### `pipeline.py`
- `load_data() -> tuple[np.ndarray, np.ndarray]`
- `make_split(X, y, test_size=0.2) -> tuple[X_tr, X_te, y_tr, y_te]`
- `build_pipelines() -> dict[str, Pipeline]` with at least 3 models
- Each pipeline: `StandardScaler` + classifier

### `evaluate.py`
- `evaluate_on_test(pipe, X_te, y_te, name) -> dict[str, float]` — logs accuracy, precision, recall, F1, ROC-AUC
- `cross_validate_pipeline(pipe, X, y, n_splits=5) -> dict[str, float]` — returns mean/std

### `train.py`
- Loads data, splits, trains all models, selects best by CV AUC, evaluates on test, asserts acceptance criteria, saves with joblib

### `tests/test_all.py`
- At least 8 test cases covering: data shape, split stratification, pipeline type, prediction shape, metric bounds, CV output structure, model save/load round-trip, acceptance criteria

## Hints

- Start with the Day 6 scaffold and adapt it — do not copy blindly, understand each change.
- For imbalanced data, consider `class_weight='balanced'` in `LogisticRegression`.
- Recall ≥ 0.75 may require threshold adjustment or class weight tuning.
- `cross_val_score(..., scoring='recall')` computes recall per fold.
- Use `joblib.dump` / `joblib.load` for model persistence.

## Stretch goals (optional)

1. Add `GridSearchCV` inside the pipeline for the best model's hyperparameters.
2. Add a `ColumnTransformer` with simulated categorical features.
3. Produce a confusion matrix and classification report in the logs.
4. Implement a `predict_new(pipe, X_new)` function with input validation.

---

## Assessment

See `assessment.md` for the formal assessment (separate file).

---

## Self-check
- [ ] I built the full pipeline without referring to Day 6 code after the initial scaffold.
- [ ] All 8+ tests pass.
- [ ] Acceptance criteria assertions pass.
- [ ] I can explain every design decision: why that split, why that metric, why that pipeline.
