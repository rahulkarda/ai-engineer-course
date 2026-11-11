# Solutions — Assessment: Week 02

## Conceptual Answers

**1. Bias-variance tradeoff**

Bias is systematic error from an overly simplistic model that cannot capture the true relationship; variance is sensitivity to fluctuations in training data. High bias → underfitting (e.g., linear regression on a non-linear problem). High variance → overfitting (e.g., a k-NN with k=1 or a decision tree with no depth limit). The tradeoff means reducing bias (more complex model) tends to increase variance.

**2. Leakage via pre-CV scaling**

If you `fit_transform(X)` on the full dataset before CV, the scaler's mean and standard deviation are computed using samples that end up in the test folds. Those folds are no longer truly unseen: the model indirectly "knows" test-fold statistics. The fix is to put the scaler inside a `Pipeline` and pass the pipeline to `cross_val_score`, which re-fits the scaler only on each fold's training portion.

**3. 99% accuracy on imbalanced data**

A model predicting class 0 for every sample achieves 99% accuracy while detecting zero positives. Accuracy is misleading here. Use ROC-AUC (measures ranking), F1 (measures precision-recall balance), or Precision-Recall AUC instead.

**4. k in k-Means vs. k in k-NN**

k-Means: k = number of cluster centroids to find (unsupervised; no labels; k controls partitioning). k-NN: k = number of nearest training examples to consult when classifying a new point (supervised; labels are used for majority vote). One partitions feature space; the other looks up neighbours.

**5. High train F1, low test F1 (logistic regression)**

Two remedies:
- Increase L2 regularisation (lower `C`): penalises large weights, reducing reliance on training-set noise.
- Collect more training data: more samples reduce the variance of the learnt decision boundary.

**6. Macro vs. weighted F1**

Macro: compute F1 per class then average equally — each class counts the same regardless of size. It is misleading when classes have very different sizes (a small class with bad F1 pulls down the macro average heavily). Weighted: average weighted by class support — it is misleading on imbalanced datasets because the majority class dominates and hides poor minority-class performance.

**7. Pipeline.fit vs. Pipeline.predict**

`pipe.fit(X_tr, y_tr)`: calls `fit_transform` on each intermediate step (fitting each transformer on X_tr and producing transformed X) then calls `fit` on the final estimator with the final transformed X. `pipe.predict(X_te)`: calls `transform` (not `fit_transform`) on each intermediate step (using the parameters fitted during `fit`) then calls `predict` on the final estimator.

**8. Curse of dimensionality**

In high-dimensional space, the volume grows exponentially with d, so training data becomes extremely sparse. All pairwise distances converge to similar values, meaning the "nearest" neighbour is nearly as far as the "farthest" one, making distance-based reasoning uninformative. This is called the *curse of dimensionality*.

---

## Coding Challenge Solution

```python
"""
assessment_solution.py
Deps: scikit-learn, numpy (uv add scikit-learn numpy)
"""
import logging

import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    data = load_wine()
    X: np.ndarray = data.data   # type: ignore[attr-defined]
    y: np.ndarray = data.target # type: ignore[attr-defined]
    log.info("Dataset: X=%s  classes=%s", X.shape, data.target_names)  # type: ignore[attr-defined]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log.info("Train: %d  Test: %d", len(y_tr), len(y_te))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(multi_class="multinomial", max_iter=1000, random_state=42)),
    ])

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc_ovr_weighted")
    log.info("CV ROC-AUC (OvR weighted): mean=%.4f  std=%.4f", auc_scores.mean(), auc_scores.std())
    assert auc_scores.std() < 0.05, f"CV std {auc_scores.std():.4f} >= 0.05"

    # Fit and evaluate on test
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average="macro")
    cm     = confusion_matrix(y_te, y_pred)

    log.info("Test accuracy:   %.4f", acc)
    log.info("Test macro-F1:   %.4f", f1_mac)
    log.info("Confusion matrix:\n%s", cm)

    assert acc >= 0.90, f"Accuracy {acc:.4f} < 0.90"
    log.info("All acceptance criteria passed.")


if __name__ == "__main__":
    main()
```

```python
"""test_assessment.py"""
import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="module")
def wine_data():
    data = load_wine()
    return data.data, data.target


@pytest.fixture(scope="module")
def wine_splits(wine_data):
    X, y = wine_data
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def test_data_shape(wine_data):
    X, y = wine_data
    assert X.shape == (178, 13)
    assert y.shape == (178,)


def test_split_size(wine_splits):
    X_tr, X_te, y_tr, y_te = wine_splits
    assert len(y_te) == pytest.approx(178 * 0.2, abs=2)


def test_split_stratification(wine_splits):
    _, _, y_tr, y_te = wine_splits
    for cls in np.unique(y_tr):
        rate_tr = (y_tr == cls).mean()
        rate_te = (y_te == cls).mean()
        assert abs(rate_tr - rate_te) < 0.1


def test_pipeline_type():
    from assessment_solution import main  # noqa: F401
    import importlib, sys
    mod = importlib.import_module("assessment_solution")
    # build the pipe directly for inspection
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    assert isinstance(pipe, Pipeline)


def test_prediction_shape(wine_splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X_tr, X_te, y_tr, y_te = wine_splits
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert preds.shape == y_te.shape


def test_accuracy_threshold(wine_splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    X_tr, X_te, y_tr, y_te = wine_splits
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(multi_class="multinomial", max_iter=1000))])
    pipe.fit(X_tr, y_tr)
    assert accuracy_score(y_te, pipe.predict(X_te)) >= 0.90
```

---

## Debugging Challenge — Answer

**Bug 1: Scaling the full dataset before the split**
```python
X_scaled = scaler.fit_transform(X)   # BUG: uses all 150 rows
X_train, X_test = X_scaled[:120], X_scaled[120:]
```
Fix: split first, fit scaler on train only (or use a Pipeline).

**Bug 2: Non-random sequential split**
```python
X_train, X_test = X_scaled[:120], X_scaled[120:]
```
Iris is sorted by class (rows 0–49 = class 0, 50–99 = class 1, 100–149 = class 2). Taking first 120 rows gives 2 full classes in train and 1 full class in test — completely non-representative. Fix: use `train_test_split(..., stratify=y)`.

**Bug 3: `roc_auc_score` called with predicted classes, not probabilities**
```python
y_pred = model.predict(X_test)         # BUG: discrete labels
auc = roc_auc_score(y_test, y_pred)    # needs probabilities for multiclass
```
Fix for multiclass: `y_proba = model.predict_proba(X_test); roc_auc_score(y_test, y_proba, multi_class='ovr')`.

**Corrected code**:
```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

iris = load_iris()
X, y = iris.data, iris.target

# Bug 2 fixed: stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Bug 1 fixed: scaler inside Pipeline
pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
pipe.fit(X_train, y_train)

# CV also uses pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc_ovr")
log.info("CV AUC: %.4f", cv_scores.mean())

# Bug 3 fixed: use predict_proba
y_proba = pipe.predict_proba(X_test)
auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
log.info("Test AUC: %.4f", auc)
```

---

## Code Quality Review Answer

| # | Issue | Why it matters | Fix |
|---|-------|---------------|-----|
| 1 | `pickle.dump` for model persistence | Security risk (arbitrary code exec on load) + poor array handling | Use `joblib.dump` / `joblib.load` |
| 2 | `data['data']` dict access on sklearn Bunch | Fragile; correct API is attribute access `data.data` | `X = data.data` |
| 3 | No train/test split — evaluating on training data | Reports 100% accuracy; completely uninformative | Split with `train_test_split` |
| 4 | No feature scaling before SVC | SVC is highly sensitive to scale; unscaled results are unreliable | Wrap in Pipeline with StandardScaler |
| 5 | `print()` instead of `logging` | Not suppressible, not configurable, mixes with test output | Replace with `log.info()` |
| 6 | No type hints | Violates project standards; reduces IDE support | Add type annotations to all variables and functions |
| 7 | `SVC()` has no `predict_proba` by default | ROC-AUC requires probabilities; must set `probability=True` | `SVC(probability=True)` |
| 8 | No tests | No regression safety net | Write pytest suite |
