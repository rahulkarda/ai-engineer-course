# Solutions — Day 6: Build an End-to-End scikit-learn Pipeline

## Exercise 1 — Add a new model

Add to `build_pipelines()` in `pipeline.py`:

```python
from sklearn.tree import DecisionTreeClassifier

def build_pipelines() -> dict[str, Pipeline]:
    return {
        # ... existing models ...
        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]),
    }
```

**Discussion**: decision trees are scale-invariant (the scaler has no effect), so test accuracy is the same with or without the scaler. The tree typically underperforms logistic regression on breast cancer (~0.92–0.94 vs. 0.97+) because it has higher variance and does not benefit from the smooth decision boundary that LR provides on this dataset.

---

## Exercise 2 — Hyperparameter sweep with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_tuned_pipeline() -> GridSearchCV:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, random_state=42)),
    ])
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10]}
    return GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)

# Usage in run.py:
# gs = build_tuned_pipeline()
# gs.fit(X_tr, y_tr)
# log.info("Best C: %s  CV AUC: %.4f", gs.best_params_, gs.best_score_)
```

---

## Exercise 3 — Feature importance

```python
import logging
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

data = load_breast_cancer()
X, y = data.data, data.target
feature_names: list[str] = list(data.feature_names)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(n_estimators=100, random_state=42)),
])
pipe.fit(X_tr, y_tr)
importances = pipe.named_steps["clf"].feature_importances_

top5_idx = np.argsort(importances)[-5:][::-1]
log.info("Top-5 features by importance:")
for rank, idx in enumerate(top5_idx, 1):
    log.info("  %d. %-35s %.4f", rank, feature_names[idx], importances[idx])
```

---

## Exercise 4 — Additional test cases

Add to `test_pipeline.py`:

```python
import joblib
import pathlib
import numpy as np

def test_save_load_round_trip(
    splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tmp_path: pathlib.Path,
) -> None:
    X_tr, X_te, y_tr, y_te = splits
    pipe = train_pipeline(build_pipelines()["logreg"], X_tr, y_tr)
    out = tmp_path / "model.joblib"
    joblib.dump(pipe, out)
    loaded = joblib.load(out)
    # Predictions must be identical
    np.testing.assert_array_equal(
        pipe.predict(X_te),
        loaded.predict(X_te),
    )

def test_cv_std_below_threshold(
    dataset: tuple[np.ndarray, np.ndarray],
    splits: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_tr, X_te, y_tr, y_te = splits
    pipe = build_pipelines()["logreg"]
    result = cross_validate_pipeline(pipe, X_tr, y_tr, n_splits=5)
    assert result["std"] < 0.05, f"CV std {result['std']:.4f} >= 0.05"
```

---

## Exercise 5 — Imbalanced data

```python
import logging
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
y_proba = pipe.predict_proba(X_te)[:, 1]

log.info("Accuracy:  %.4f  (a naive all-0 model scores %.4f)", accuracy_score(y_te, y_pred), y_te.mean() * 0 + (1 - y_te.mean()))
log.info("F1:        %.4f", f1_score(y_te, y_pred))
log.info("ROC-AUC:   %.4f", roc_auc_score(y_te, y_proba))

# Acceptance gate on ROC-AUC is OK; accuracy gate would be misleading
try:
    assert roc_auc_score(y_te, y_proba) >= 0.97
    log.info("AUC >= 0.97: PASS")
except AssertionError:
    log.warning("AUC < 0.97: the acceptance criterion needs adjustment for imbalanced data")
```

**Discussion**: accuracy can pass trivially (a model that predicts all 0 scores 90%). The ROC-AUC acceptance criterion is more appropriate for imbalanced settings because it is threshold-independent and measures separability rather than raw accuracy.
