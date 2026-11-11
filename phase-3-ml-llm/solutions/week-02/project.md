# Solutions — Project: Customer Churn Prediction Pipeline

## Reference Implementation

### `week02_project/pipeline.py`

```python
"""
week02_project/pipeline.py
Deps: scikit-learn, numpy (uv add scikit-learn numpy)
"""
import logging
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate simulated churn dataset."""
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=99,
    )
    log.info("Dataset: X=%s  class distribution=%s", X.shape, np.bincount(y))
    return X, y


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
    """Return named sklearn Pipelines for comparison."""
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=500, class_weight="balanced", random_state=42
            )),
        ]),
        "logreg_high_recall": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=0.1, max_iter=500, class_weight="balanced", random_state=42
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=11)),
        ]),
        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(
                max_depth=7, class_weight="balanced", random_state=42
            )),
        ]),
    }


def train_pipeline(pipe: Pipeline, X_tr: np.ndarray, y_tr: np.ndarray) -> Pipeline:
    """Fit a pipeline on training data."""
    pipe.fit(X_tr, y_tr)
    return pipe
```

---

### `week02_project/evaluate.py`

```python
"""week02_project/evaluate.py"""
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
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
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    metrics: dict[str, float] = {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall":    recall_score(y_te, y_pred),
        "f1":        f1_score(y_te, y_pred),
        "roc_auc":   roc_auc_score(y_te, y_proba),
    }
    log.info(
        "[%s] acc=%.4f prec=%.4f recall=%.4f f1=%.4f auc=%.4f",
        name, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"], metrics["roc_auc"],
    )
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
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    result: dict[str, float] = {"mean": scores.mean(), "std": scores.std()}
    log.info("CV %s: mean=%.4f  std=%.4f", scoring, result["mean"], result["std"])
    return result


def compare_models(
    pipelines: dict[str, Pipeline],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
) -> str:
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
    log.info("\nBest by CV AUC: %s (%.4f)", best_name, best_score)
    return best_name
```

---

### `week02_project/train.py`

```python
"""week02_project/train.py"""
import logging
import pathlib
import joblib

from week02_project.pipeline import load_data, make_split, build_pipelines
from week02_project.evaluate import compare_models, evaluate_on_test

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = pathlib.Path("week02_project/models")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = make_split(X, y)
    pipelines = build_pipelines()
    best_name = compare_models(pipelines, X_tr, y_tr, X_te, y_te)
    best_pipe  = pipelines[best_name]

    metrics = evaluate_on_test(best_pipe, X_te, y_te, name=best_name)

    # Acceptance gates
    assert metrics["roc_auc"] >= 0.85, f"ROC-AUC {metrics['roc_auc']:.4f} < 0.85"
    assert metrics["recall"]  >= 0.75, f"Recall  {metrics['recall']:.4f} < 0.75"
    log.info("All acceptance criteria passed.")

    out = OUTPUT_DIR / f"{best_name}.joblib"
    joblib.dump(best_pipe, out)
    log.info("Saved: %s", out)

    loaded = joblib.load(out)
    loaded_metrics = evaluate_on_test(loaded, X_te, y_te, name=f"{best_name}_loaded")
    assert loaded_metrics["roc_auc"] == metrics["roc_auc"]
    log.info("Round-trip verified.")


if __name__ == "__main__":
    main()
```

---

### `week02_project/tests/test_all.py`

```python
"""week02_project/tests/test_all.py"""
import pathlib
import numpy as np
import pytest
import joblib
from sklearn.pipeline import Pipeline

from week02_project.pipeline import load_data, make_split, build_pipelines, train_pipeline
from week02_project.evaluate import evaluate_on_test, cross_validate_pipeline


@pytest.fixture(scope="module")
def dataset():
    return load_data()


@pytest.fixture(scope="module")
def splits(dataset):
    X, y = dataset
    return make_split(X, y)


class TestData:
    def test_shape(self, dataset):
        X, y = dataset
        assert X.shape == (2000, 20)
        assert y.shape == (2000,)

    def test_binary(self, dataset):
        _, y = dataset
        assert set(np.unique(y)) == {0, 1}

    def test_class_imbalance(self, dataset):
        _, y = dataset
        # 30% churners ± 5%
        rate = y.mean()
        assert 0.25 <= rate <= 0.35


class TestSplit:
    def test_sizes(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        assert len(y_tr) + len(y_te) == 2000
        assert abs(len(y_te) / 2000 - 0.2) < 0.02

    def test_stratification(self, splits):
        _, _, y_tr, y_te = splits
        assert abs(y_tr.mean() - y_te.mean()) < 0.05


class TestPipelines:
    def test_count(self):
        assert len(build_pipelines()) >= 3

    def test_all_are_pipelines(self):
        for name, pipe in build_pipelines().items():
            assert isinstance(pipe, Pipeline), f"{name} not a Pipeline"

    def test_fit_predict(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
        preds = pipe.predict(X_te)
        assert preds.shape == y_te.shape
        assert set(np.unique(preds)) <= {0, 1}


class TestEvaluation:
    def test_metric_keys(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
        m = evaluate_on_test(pipe, X_te, y_te)
        assert {"accuracy", "precision", "recall", "f1", "roc_auc"} <= set(m.keys())

    def test_roc_auc_threshold(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["random_forest"], X_tr, y_tr)
        m = evaluate_on_test(pipe, X_te, y_te)
        assert m["roc_auc"] >= 0.85

    def test_recall_threshold(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
        m = evaluate_on_test(pipe, X_te, y_te)
        assert m["recall"] >= 0.75

    def test_cv_structure(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        pipe = build_pipelines()["logreg"]
        result = cross_validate_pipeline(pipe, X_tr, y_tr, n_splits=3)
        assert "mean" in result and "std" in result
        assert 0.0 < result["mean"] <= 1.0

    def test_cv_std_threshold(self, splits):
        X_tr, X_te, y_tr, y_te = splits
        pipe = build_pipelines()["logreg"]
        result = cross_validate_pipeline(pipe, X_tr, y_tr, n_splits=5)
        assert result["std"] < 0.04

    def test_save_load_roundtrip(self, splits, tmp_path):
        X_tr, X_te, y_tr, y_te = splits
        pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
        out = tmp_path / "churn_model.joblib"
        joblib.dump(pipe, out)
        loaded = joblib.load(out)
        np.testing.assert_array_equal(pipe.predict(X_te), loaded.predict(X_te))
```

---

## Discussion: Design Decisions

**`class_weight='balanced'`**: With 30% churn rate the classes are unbalanced. Setting `class_weight='balanced'` instructs sklearn to weight class losses inversely proportional to frequency, boosting recall for the minority (churn) class. This is necessary to meet the recall ≥ 0.75 criterion.

**Model choice**: Random Forest typically outperforms a single logistic regression on non-linear tabular data, but the margin here is small. Logistic regression with balanced weights often achieves recall ≥ 0.75 reliably and is more interpretable.

**5-fold over 10-fold**: 2000 samples is large enough that 5-fold gives reliable estimates. 10-fold would be marginally more accurate but 2× slower.

**joblib over pickle**: joblib is sklearn's recommended serialisation format; it handles large numpy arrays more efficiently and avoids some pickle security issues. Never use pickle directly in production pipelines.
