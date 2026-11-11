# Phase 3 · Week 02 · Day 1 — ML Taxonomy, Train/Test Split & Bias-Variance

> Objective: Understand supervised vs. unsupervised learning, split data correctly, and diagnose overfitting via the bias-variance lens.  ·  Time: ~1 hr  ·  Prereqs: [[../week-01/day-7]]

## Concept

### Supervised vs. Unsupervised Learning

**Supervised learning**: every training example has a label. The model learns a mapping `f(X) → y`.
- *Classification*: y is discrete (spam/not-spam, species).
- *Regression*: y is continuous (house price, temperature).

**Unsupervised learning**: no labels. The model finds structure in X alone.
- *Clustering*: group similar examples (k-means, DBSCAN).
- *Dimensionality reduction*: compress X while retaining variance (PCA).
- *Density estimation*: model P(X) (GMMs, autoencoders).

**Semi-supervised**: a small labelled set + large unlabelled set.

**Self-supervised**: labels derived from the data itself (next-token prediction in LLMs is self-supervised).

### Train / Validation / Test Split

| Split | Purpose | Typical size |
|-------|---------|-------------|
| Train | Fit model parameters | 60–80% |
| Validation | Tune hyperparameters / architecture | 10–20% |
| Test | Final, unbiased performance estimate | 10–20% |

The test set must never influence any modelling decision — not feature selection, not threshold tuning. Using test data to guide choices is **data leakage** and produces optimistically biased estimates.

### Bias-Variance Tradeoff

Decompose expected prediction error for a regression model at point x:

```
E[(y - ŷ)²] = Bias² + Variance + Irreducible noise
```

- **Bias**: systematic error from wrong assumptions (e.g., fitting a line to a quadratic). High bias → underfitting.
- **Variance**: sensitivity to fluctuations in the training set. High variance → overfitting.
- **Irreducible noise**: noise inherent in the problem; no model eliminates it.

The tradeoff: reducing bias (more complex model) usually increases variance, and vice versa. Regularization, cross-validation, and ensemble methods navigate this tradeoff.

## How it works internally

**Why random splits can be dangerous**: if data has temporal ordering, a random split leaks future data into training. Use time-aware splits (`TimeSeriesSplit`).

**Stratified splitting**: when classes are imbalanced (99% negative), a random split may put all positives in one partition. `train_test_split(..., stratify=y)` preserves class proportions in every split.

**Overfitting mechanics**: a model with too many parameters memorises training noise — it models the random fluctuations specific to the training sample. On a new sample those fluctuations are different, so performance degrades. Formally, high-variance models have large differences between training and validation loss.

**Underfitting mechanics**: a model too simple to capture the true signal shows high training *and* validation error — the function class it searches cannot express the real relationship.

## Worked example

```python
"""
day1_ml_taxonomy.py
Deps: scikit-learn, numpy (install with uv: uv add scikit-learn numpy)
"""
import logging
from typing import Any

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def demonstrate_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split data and show class distribution before and after."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    log.info("Full set class distribution:  %s", np.bincount(y))
    log.info("Train set class distribution: %s", np.bincount(y_train))
    log.info("Test  set class distribution: %s", np.bincount(y_test))
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def bias_variance_demo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Vary k in k-NN to illustrate the bias-variance tradeoff.
    k=1  → low bias, high variance (overfitting)
    k=n  → high bias, low variance (underfitting)
    """
    log.info("\n%-5s %-14s %-14s", "k", "Train acc", "Test acc")
    for k in [1, 3, 7, 15, 31, 51]:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        log.info("%-5d %-14.4f %-14.4f", k, train_acc, test_acc)


def main() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target  # type: ignore[attr-defined]
    splits = demonstrate_split(X, y)
    bias_variance_demo(
        splits["X_train"], splits["y_train"],
        splits["X_test"],  splits["y_test"],
    )

    # Unsupervised flavour: no labels needed
    X_unlabelled, _ = make_classification(n_samples=300, n_features=4, random_state=0)
    log.info("\nUnsupervised dataset shape: %s (no labels used)", X_unlabelled.shape)


if __name__ == "__main__":
    main()
```

Run:
```bash
uv add scikit-learn numpy
python day1_ml_taxonomy.py
```

Expected output (abbreviated):
```
INFO: Full set class distribution:  [50 50 50]
INFO: Train set class distribution: [40 40 40]
INFO: Test  set class distribution: [10 10 10]
INFO: k     Train acc      Test acc
INFO: 1     1.0000         0.9667
INFO: 3     0.9583         0.9667
INFO: 7     0.9583         0.9667
INFO: 15    0.9500         0.9667
INFO: 31    0.9500         0.9667
INFO: 51    0.9417         0.9333
```

Observe: k=1 hits 100% training accuracy (memorisation) but test accuracy is not best. k=51 starts losing both.

## Exercises

1. **Split sensitivity**: Using `make_classification(n_samples=200, weights=[0.9, 0.1])`, split *without* stratify and *with* stratify. Compare class distributions in each test set. What happens without stratify?

2. **Bias-variance curve**: Generate data with `make_classification(n_samples=500, n_features=20, n_informative=5)`. Train k-NN models for k = 1, 5, 10, 20, 50, 100. Plot (or log) train vs. test accuracy. Identify the approximate sweet spot.

3. **Leakage detection**: A teammate splits `X` into train/test, then applies `StandardScaler().fit_transform(X)` on the *full* `X` before passing to the model. Explain the leakage and write the corrected code.

4. **Supervised vs. unsupervised categorisation**: Classify each scenario as supervised, unsupervised, or self-supervised:
   a. Predicting customer churn from historical CRM data.
   b. Grouping news articles into topics with no predefined categories.
   c. Training a language model to predict the next word.
   d. Detecting anomalies in server metrics with no labelled outages.

(Solutions: ../solutions/week-02/day-1.md)

## Self-check
- [ ] I can explain the difference between supervised and unsupervised learning with one example each.
- [ ] I can implement a stratified train/test split and explain why stratification matters.
- [ ] I can describe the bias-variance tradeoff and identify overfitting/underfitting from train vs. test accuracy.
- [ ] I can spot a data leakage scenario in a code review.

## Supplementary links
- Scikit-learn: [Model selection intro](https://scikit-learn.org/stable/model_selection.html)
- Bishop, *Pattern Recognition and Machine Learning*, ch. 1 (bias-variance)
- [Understanding the Bias-Variance Tradeoff — Scott Fortmann-Roe](http://scott.fortmann-roe.com/docs/BiasVariance.html)
