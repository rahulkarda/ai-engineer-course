# Phase 3 · Week 02 · Day 5 — Evaluation Metrics, Cross-Validation & Pipelines

> Objective: Master classification and regression evaluation metrics, implement cross-validation correctly, and build leakage-free sklearn Pipelines.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]], [[day-3]]

## Concept

### Classification Metrics

**Confusion matrix** (binary):

|              | Predicted Positive | Predicted Negative |
|--------------|--------------------|--------------------|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

Derived metrics:

| Metric | Formula | Use when |
|--------|---------|---------|
| Accuracy | (TP+TN) / N | Balanced classes |
| Precision | TP / (TP+FP) | FP is costly (spam filter) |
| Recall (Sensitivity) | TP / (TP+FN) | FN is costly (cancer screening) |
| F1 | 2·P·R / (P+R) | Imbalanced classes |
| ROC-AUC | Area under ROC curve | Threshold-independent ranking |

**ROC curve**: plots TPR (recall) vs. FPR (FP/(FP+TN)) at all thresholds. AUC=1 is perfect; AUC=0.5 is random.

**Precision-Recall curve**: better than ROC for highly imbalanced problems (AUC-PR is more informative than ROC-AUC when positive class is rare).

**Macro vs. micro averaging** (multi-class):
- Macro: compute per-class, then average — treats all classes equally.
- Micro: pool all predictions — dominated by the majority class.
- Weighted: weighted by support — good for imbalanced multi-class.

### Regression Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | (1/n)Σ\|yᵢ-ŷᵢ\| | Robust to outliers; same units as y |
| MSE | (1/n)Σ(yᵢ-ŷᵢ)² | Penalises large errors heavily |
| RMSE | √MSE | Same units as y; most common |
| R² | 1 - SS_res/SS_tot | Fraction of variance explained; ≤1 |
| MAPE | (1/n)Σ\|yᵢ-ŷᵢ\|/\|yᵢ\| | Percentage error; undefined at y=0 |

### Cross-Validation

**k-Fold CV**: split data into k folds; train on k-1, evaluate on 1, rotate, average k scores.
- k=5 or k=10 are standard.
- Reduces variance of the performance estimate vs. a single split.

**Stratified k-Fold**: preserves class proportions in each fold. Use for classification.

**Time-series CV** (`TimeSeriesSplit`): always train on past, test on future. Never shuffle temporal data.

**Repeated CV**: repeat k-fold with different random seeds; further reduces variance.

**Nested CV**: outer loop estimates generalisation; inner loop tunes hyperparameters. Prevents optimistic bias from using the same data to both tune and evaluate.

### Pipelines

A `Pipeline` chains preprocessors and a final estimator. Crucially, `fit` is only called on training data at each step — `transform` applies to test data using the fitted parameters. This prevents leakage.

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression()),
])
pipe.fit(X_tr, y_tr)          # fit scaler on X_tr only
pipe.score(X_te, y_te)        # transforms X_te with fitted scaler
```

`cross_val_score(pipe, X, y, cv=5)` fits the entire pipeline inside each fold — scaler is re-fit on each fold's training portion. This is the correct way.

**ColumnTransformer**: apply different transformations to different feature subsets.

## How it works internally

**Why AUC-ROC is threshold-independent**: it summarises model quality across all possible classification thresholds by measuring rank ordering. A model that always puts positives above negatives has AUC=1.

**F1 is the harmonic mean**: harmonic mean punishes extreme imbalance between P and R more than arithmetic mean does. If P=1 and R=0, arithmetic mean=0.5 but F1=0.

**Pipeline internals**: `Pipeline.fit` calls `fit_transform` on each intermediate step and `fit` on the final estimator. `Pipeline.predict` calls `transform` (not `fit_transform`) on intermediates, then `predict` on the final estimator.

**Leakage via cross_val_score without Pipeline**: if you scale X before calling `cross_val_score(model, X_scaled, y)`, the scaler was fit on all data including test folds — subtle leakage that makes performance look better than it is.

## Worked example

```python
"""
day5_evaluation_cv_pipelines.py
Deps: scikit-learn, numpy (uv add scikit-learn numpy)
"""
import logging

import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def classification_metrics_demo() -> None:
    log.info("=== Classification metrics (Breast Cancer) ===")
    data = load_breast_cancer()
    X, y = data.data, data.target  # type: ignore[attr-defined]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    log.info("Confusion matrix:\n%s", confusion_matrix(y_te, y_pred))
    log.info("Accuracy:  %.4f", accuracy_score(y_te, y_pred))
    log.info("Precision: %.4f", precision_score(y_te, y_pred))
    log.info("Recall:    %.4f", recall_score(y_te, y_pred))
    log.info("F1:        %.4f", f1_score(y_te, y_pred))
    log.info("ROC-AUC:   %.4f", roc_auc_score(y_te, y_proba))
    log.info("\n%s", classification_report(y_te, y_pred,
                                            target_names=data.target_names))  # type: ignore[attr-defined]


def regression_metrics_demo() -> None:
    log.info("=== Regression metrics ===")
    X, y = make_regression(n_samples=300, n_features=10, noise=15, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = mean_squared_error(y_te, y_pred) ** 0.5
    r2   = r2_score(y_te, y_pred)
    log.info("MAE:  %.3f  RMSE: %.3f  R²: %.4f", mae, rmse, r2)


def cross_validation_demo() -> None:
    log.info("\n=== Cross-validation (5-fold, stratified) ===")
    data = load_breast_cancer()
    X, y = data.data, data.target  # type: ignore[attr-defined]

    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for scoring in ["accuracy", "roc_auc", "f1"]:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        log.info("%-12s  mean=%.4f  std=%.4f  folds=%s",
                 scoring, scores.mean(), scores.std(),
                 [f"{s:.4f}" for s in scores])


def leakage_comparison() -> None:
    """Show the difference between correct CV (with pipeline) and leaked CV."""
    log.info("\n=== Leakage demonstration ===")
    data = load_breast_cancer()
    X, y = data.data, data.target  # type: ignore[attr-defined]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # WRONG: scale entire X before CV — scaler sees test fold data
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)
    leaked_scores = cross_val_score(
        LogisticRegression(max_iter=500), X_scaled_all, y, cv=cv, scoring="roc_auc"
    )

    # CORRECT: scale inside pipeline — scaler only sees training fold
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    correct_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    log.info("Leaked  ROC-AUC: mean=%.4f  std=%.4f", leaked_scores.mean(), leaked_scores.std())
    log.info("Correct ROC-AUC: mean=%.4f  std=%.4f", correct_scores.mean(), correct_scores.std())
    log.info("(Difference is small here but can be large with aggressive preprocessing)")


def main() -> None:
    classification_metrics_demo()
    regression_metrics_demo()
    cross_validation_demo()
    leakage_comparison()


if __name__ == "__main__":
    main()
```

Run:
```bash
python day5_evaluation_cv_pipelines.py
```

## Exercises

1. **Threshold tuning**: On `load_breast_cancer()`, get `predict_proba` from a logistic regression. Vary the classification threshold from 0.1 to 0.9 in steps of 0.1. Log precision, recall, and F1 at each threshold. At what threshold does F1 peak?

2. **Imbalanced dataset**: Create `make_classification(n_samples=1000, weights=[0.95, 0.05])`. Train a logistic regression. Compare accuracy, F1 (macro), F1 (weighted), and ROC-AUC. Which metrics are misleading and which are informative?

3. **CV scorer selection**: On `load_breast_cancer()`, compare `cross_val_score` with `scoring='accuracy'` vs. `scoring='f1'` vs. `scoring='roc_auc'` (all 5-fold). Do model rankings change if you compare two models (e.g., LR with C=0.01 vs. C=10)? What does this tell you about scorer choice?

4. **ColumnTransformer pipeline**: Using `load_iris()`, add a categorical dummy column (`rng.choice(['A','B','C'], 150)`). Build a pipeline with `ColumnTransformer` that scales numeric features with `StandardScaler` and encodes the categorical column with `OneHotEncoder`. Fit and score.

5. **Nested CV**: Implement nested cross-validation: outer `StratifiedKFold(5)`, inner `GridSearchCV(LogisticRegression, {'C': [0.01, 0.1, 1, 10]}, cv=3)`. Report outer fold test scores. Compare to a non-nested CV estimate — is there optimistic bias?

(Solutions: ../solutions/week-02/day-5.md)

## Self-check
- [ ] I can compute precision, recall, F1, and ROC-AUC and explain when to prefer each.
- [ ] I can build a sklearn Pipeline that prevents train/test leakage.
- [ ] I can run stratified k-fold CV correctly using `cross_val_score` with a pipeline.
- [ ] I can explain nested CV and when it is necessary.

## Supplementary links
- Scikit-learn: [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- Scikit-learn: [Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Precision and Recall — Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
- [The ROC Curve — Tom Fawcett, 2006](https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf)
