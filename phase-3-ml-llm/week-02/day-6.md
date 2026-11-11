# Phase 3 · Week 02 · Day 6 — Build an End-to-End scikit-learn Pipeline

> Objective: Build a complete, production-quality ML pipeline — load → preprocess → train → evaluate → cross-validate — with tests.  ·  Time: ~4 hr  ·  Prereqs: [[day-1]] through [[day-5]]

## Overview

Today you build one coherent system rather than isolated demos. The goal is a reusable, tested, logged pipeline on a real tabular problem. You will:

1. Load and explore a dataset.
2. Preprocess (scale, encode, impute).
3. Train and compare multiple models.
4. Evaluate with multiple metrics.
5. Cross-validate correctly inside a pipeline.
6. Save and load the best model.
7. Write unit and integration tests with pytest.

This mirrors what you will do in production ML work.

## Concepts (review)

**Pipeline correctness checklist**:
- All preprocessing steps are inside the pipeline.
- CV uses the full pipeline, not pre-transformed data.
- Test set is held out until final evaluation only.
- Metrics match the business problem.

**Model selection vs. model assessment**:
- *Model selection*: pick the best algorithm/hyperparameters using CV on train+val.
- *Model assessment*: estimate the selected model's real-world performance on the held-out test set.

**Saving models**: use `joblib.dump` / `joblib.load` (sklearn recommendation). Never use pickle directly for security reasons — joblib is safer and handles large arrays more efficiently.

## Project Specification

**Dataset**: `load_breast_cancer()` — 569 samples, 30 numeric features, binary classification (malignant/benign). Goal: predict malignancy.

**Deliverables** (produce all of these):
1. `pipeline_project/pipeline.py` — full pipeline module.
2. `pipeline_project/evaluate.py` — evaluation utilities.
3. `pipeline_project/tests/test_pipeline.py` — pytest test suite.
4. `pipeline_project/run.py` — entry point: train, evaluate, save.

**Acceptance criteria**:
- ROC-AUC ≥ 0.97 on held-out test set.
- 5-fold CV ROC-AUC std < 0.03.
- All tests pass.
- No preprocessing outside pipelines.
- No `print()` statements — use `logging`.

## Step-by-Step Guide

### Step 1 — Project scaffold

```
pipeline_project/
├── pipeline.py
├── evaluate.py
├── run.py
└── tests/
    └── test_pipeline.py
```

```bash
mkdir -p pipeline_project/tests
touch pipeline_project/__init__.py pipeline_project/tests/__init__.py
```

### Step 2 — `pipeline.py`

Build functions that return fitted sklearn Pipelines. Keep logic modular.

```python
"""
pipeline_project/pipeline.py
Deps: scikit-learn (uv add scikit-learn)
"""
import logging
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load breast cancer dataset. Returns X, y, feature_names."""
    data = load_breast_cancer()
    X: np.ndarray = data.data        # type: ignore[attr-defined]
    y: np.ndarray = data.target      # type: ignore[attr-defined]
    feature_names: list[str] = list(data.feature_names)  # type: ignore[attr-defined]
    log.info("Dataset loaded: X=%s  y=%s  classes=%s",
             X.shape, y.shape, data.target_names)  # type: ignore[attr-defined]
    return X, y, feature_names


def make_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    log.info("Split: train=%d  test=%d", len(y_tr), len(y_te))
    return X_tr, X_te, y_tr, y_te


def build_pipelines() -> dict[str, Pipeline]:
    """Return a dict of named sklearn Pipelines to compare."""
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ]),
        "logreg_strong_reg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(C=0.01, max_iter=500, random_state=42)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=7)),
        ]),
        "random_forest": Pipeline([
            # RF is scale-invariant, but include scaler for interface consistency
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
    }


def train_pipeline(pipe: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray) -> Pipeline:
    """Fit a pipeline and return it."""
    pipe.fit(X_tr, y_tr)
    return pipe
```

### Step 3 — `evaluate.py`

```python
"""
pipeline_project/evaluate.py
"""
import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def evaluate_on_test(
    pipe: Pipeline,
    X_te: np.ndarray,
    y_te: np.ndarray,
    name: str = "model",
) -> dict[str, float]:
    """Compute and log test-set metrics. Returns metric dict."""
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    metrics: dict[str, float] = {
        "accuracy":  accuracy_score(y_te, y_pred),
        "f1":        f1_score(y_te, y_pred),
        "roc_auc":   roc_auc_score(y_te, y_proba),
    }
    log.info("[%s] accuracy=%.4f  f1=%.4f  roc_auc=%.4f",
             name, metrics["accuracy"], metrics["f1"], metrics["roc_auc"])
    log.info("Confusion matrix:\n%s", confusion_matrix(y_te, y_pred))
    return metrics


def cross_validate_pipeline(
    pipe: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
) -> dict[str, float]:
    """Run stratified k-fold CV. Returns mean and std."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    result: dict[str, float] = {"mean": scores.mean(), "std": scores.std()}
    log.info("CV %s: mean=%.4f  std=%.4f  folds=%s",
             scoring, result["mean"], result["std"],
             [f"{s:.4f}" for s in scores])
    return result


def compare_models(
    pipelines: dict[str, Pipeline],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> str:
    """Train, CV, and test all pipelines. Return name of best by CV AUC."""
    best_name  = ""
    best_score = -1.0

    for name, pipe in pipelines.items():
        log.info("\n--- %s ---", name)
        pipe.fit(X_tr, y_tr)
        cv_result = cross_validate_pipeline(pipe, X_tr, y_tr)
        evaluate_on_test(pipe, X_te, y_te, name=name)

        if cv_result["mean"] > best_score:
            best_score = cv_result["mean"]
            best_name  = name

    log.info("\nBest model by CV AUC: %s (%.4f)", best_name, best_score)
    return best_name
```

### Step 4 — `run.py`

```python
"""
pipeline_project/run.py
Entry point: train, evaluate, save best model.
"""
import logging
import pathlib
import joblib

from pipeline_project.pipeline import load_data, make_split, build_pipelines, train_pipeline
from pipeline_project.evaluate import compare_models, evaluate_on_test

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = pathlib.Path("pipeline_project/models")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, _feature_names = load_data()
    X_tr, X_te, y_tr, y_te = make_split(X, y)
    pipelines = build_pipelines()

    best_name = compare_models(pipelines, X_tr, y_tr, X_te, y_te)
    best_pipe  = pipelines[best_name]

    # Final assessment on held-out test set
    log.info("\n=== Final evaluation of best model: %s ===", best_name)
    metrics = evaluate_on_test(best_pipe, X_te, y_te, name=best_name)

    # Acceptance gate
    assert metrics["roc_auc"] >= 0.97, (
        f"ROC-AUC {metrics['roc_auc']:.4f} < 0.97 acceptance criterion"
    )
    log.info("Acceptance gate passed: ROC-AUC=%.4f >= 0.97", metrics["roc_auc"])

    # Save
    out_path = OUTPUT_DIR / f"{best_name}.joblib"
    joblib.dump(best_pipe, out_path)
    log.info("Model saved to %s", out_path)

    # Verify round-trip
    loaded_pipe = joblib.load(out_path)
    loaded_metrics = evaluate_on_test(loaded_pipe, X_te, y_te, name=f"{best_name}_loaded")
    assert loaded_metrics["roc_auc"] == metrics["roc_auc"], "Round-trip mismatch"
    log.info("Round-trip verification passed.")


if __name__ == "__main__":
    main()
```

### Step 5 — `tests/test_pipeline.py`

```python
"""
pipeline_project/tests/test_pipeline.py
Run: pytest pipeline_project/tests/ -v
"""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from pipeline_project.pipeline import load_data, make_split, build_pipelines, train_pipeline
from pipeline_project.evaluate import evaluate_on_test, cross_validate_pipeline


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = load_data()
    return X, y


@pytest.fixture(scope="module")
def splits(
    dataset: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = dataset
    return make_split(X, y)


class TestLoadData:
    def test_shape(self, dataset: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = dataset
        assert X.shape == (569, 30)
        assert y.shape == (569,)

    def test_binary_labels(self, dataset: tuple[np.ndarray, np.ndarray]) -> None:
        _, y = dataset
        assert set(np.unique(y)) == {0, 1}


class TestSplit:
    def test_sizes(
        self, splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_tr, X_te, y_tr, y_te = splits
        total = len(y_tr) + len(y_te)
        assert total == 569
        assert abs(len(y_te) / total - 0.2) < 0.02

    def test_stratification(
        self, splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_tr, X_te, y_tr, y_te = splits
        # Positive rate should be similar in both splits
        rate_tr = y_tr.mean()
        rate_te = y_te.mean()
        assert abs(rate_tr - rate_te) < 0.05


class TestPipelines:
    def test_all_pipelines_exist(self) -> None:
        pipelines = build_pipelines()
        assert {"logreg", "knn", "random_forest"} <= set(pipelines.keys())

    def test_pipeline_type(self) -> None:
        for name, pipe in build_pipelines().items():
            assert isinstance(pipe, Pipeline), f"{name} is not a Pipeline"

    def test_pipeline_fit_predict(
        self, splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_tr, X_te, y_tr, y_te = splits
        pipe = build_pipelines()["logreg"]
        trained = train_pipeline(pipe, X_tr, y_tr)
        preds = trained.predict(X_te)
        assert preds.shape == y_te.shape
        assert set(np.unique(preds)) <= {0, 1}


class TestEvaluation:
    def test_metrics_keys(
        self, splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
        metrics = evaluate_on_test(pipe, X_te, y_te)
        assert {"accuracy", "f1", "roc_auc"} <= set(metrics.keys())

    def test_roc_auc_threshold(
        self, splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
        metrics = evaluate_on_test(pipe, X_te, y_te)
        assert metrics["roc_auc"] >= 0.97, f"ROC-AUC too low: {metrics['roc_auc']}"

    def test_cv_returns_mean_std(
        self, dataset: tuple[np.ndarray, np.ndarray],
        splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        X, y = dataset
        X_tr, X_te, y_tr, y_te = splits
        pipe = build_pipelines()["logreg"]
        result = cross_validate_pipeline(pipe, X_tr, y_tr, n_splits=3)
        assert "mean" in result and "std" in result
        assert 0.0 < result["mean"] <= 1.0
        assert result["std"] >= 0.0
```

### Step 6 — Run everything

```bash
# Install deps
uv add scikit-learn numpy joblib pytest

# Run entry point
python -m pipeline_project.run

# Run tests
pytest pipeline_project/tests/ -v
```

## Exercises

1. **Add a new model**: Extend `build_pipelines()` to include a `DecisionTreeClassifier(max_depth=5)`. Does it beat logistic regression on test ROC-AUC? Why or why not?

2. **Hyperparameter sweep**: Replace the fixed `C=1.0` in the logistic regression pipeline with a `GridSearchCV` wrapping the pipeline. Search `C` over `[0.001, 0.01, 0.1, 1, 10]`. Report the best C and CV score.

3. **Feature importance**: After training the Random Forest pipeline, extract `pipe.named_steps['clf'].feature_importances_`. Log the top-5 features by importance (use `feature_names` from `load_data()`).

4. **Test coverage**: Add two more test cases:
   - Test that the saved/loaded model produces identical predictions on X_te.
   - Test that `cross_validate_pipeline` std < 0.05 for the logistic regression model.

5. **Imbalanced data**: Replace the breast cancer dataset with `make_classification(n_samples=1000, weights=[0.9, 0.1])`. What happens to the acceptance gate (ROC-AUC ≥ 0.97)? Does accuracy still pass? What about F1?

(Solutions: ../solutions/week-02/day-6.md)

## Self-check
- [ ] I built a multi-model comparison pipeline with no preprocessing leakage.
- [ ] My tests cover data shape, split quality, pipeline type, and metric thresholds.
- [ ] I can save/load a sklearn model with joblib and verify round-trip correctness.
- [ ] I can run `pytest` and interpret passing/failing tests.

## Supplementary links
- Scikit-learn: [Persistence](https://scikit-learn.org/stable/model_persistence.html)
- Scikit-learn: [Pipeline](https://scikit-learn.org/stable/modules/compose.html)
- [Testing ML code — Jeremy Jordan](https://www.jeremyjordan.me/testing-ml/)
