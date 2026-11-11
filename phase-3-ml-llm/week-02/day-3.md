# Phase 3 · Week 02 · Day 3 — Decision Trees & k-Nearest Neighbours

> Objective: Understand decision tree splitting, control overfitting via pruning, and implement k-NN with correct distance-based reasoning.  ·  Time: ~1 hr  ·  Prereqs: [[day-2]]

## Concept

### Decision Trees

A decision tree partitions the feature space into axis-aligned rectangles by recursively choosing the best split. At each node, it asks a yes/no question about one feature.

**Splitting criterion** (classification):
- **Gini impurity**: `G = 1 - Σ pₖ²` — probability that two randomly drawn samples from the node have different classes.
- **Entropy (information gain)**: `H = -Σ pₖ log₂(pₖ)` — expected bits to encode a label.
- Both produce similar trees; Gini is faster to compute.

**Splitting criterion** (regression):
- Minimise MSE (or MAE) within each child node.

**Stopping conditions** (control complexity):
- `max_depth`: maximum tree depth.
- `min_samples_split`: minimum samples to consider a split.
- `min_samples_leaf`: minimum samples required at a leaf.
- `max_features`: randomly sampled features per split (used in Random Forest).

**Overfitting**: an unbounded tree will fit training data perfectly (leaf = 1 sample). Setting `max_depth` or `min_samples_leaf` prunes it. Post-pruning via cost-complexity pruning (`ccp_alpha`) is also supported.

### k-Nearest Neighbours (k-NN)

k-NN is a **non-parametric, lazy** algorithm. There is no explicit training phase — the model stores the entire training set.

**Prediction** (classification):
1. Compute distance between query point and all training points.
2. Select the k nearest.
3. Return the majority class (or weighted vote).

**Distance metrics**:
- Euclidean (`p=2`): straight-line distance, sensitive to scale.
- Manhattan (`p=1`): sum of absolute differences, robust to outliers.
- Minkowski (`p`): generalisation.

**Critical implication**: k-NN is extremely sensitive to feature scale. Always standardise before using k-NN.

**Curse of dimensionality**: in high-dimensional space, all points are approximately equidistant — the nearest-neighbour concept breaks down. k-NN degrades quickly beyond ~20 meaningful features.

**k selection**:
- k=1: overfit, zero training error.
- Large k: underfit, dominated by the majority class.
- Optimal k found via cross-validation.

## How it works internally

**Decision tree splits are greedy**: the algorithm picks the locally best split at each node without backtracking. This means the globally optimal tree is not guaranteed; pruning corrects excessive depth after the fact.

**Feature importance in trees**: measured as the total reduction in impurity weighted by the number of samples reaching a node. Available as `tree.feature_importances_`. These importances favour high-cardinality features — use with care.

**k-NN memory and time complexity**:
- Training: O(1) (just store data).
- Prediction: O(n·d) for brute-force. sklearn uses a KD-tree (O(log n) per query, degrades in high d) or Ball tree (better for non-Euclidean metrics).

**Ball tree vs. KD-tree**: KD-tree partitions feature space with axis-aligned hyperplanes, efficient for d < ~20. Ball tree uses hyperspheres, faster for non-Euclidean metrics and higher d.

## Worked example

```python
"""
day3_trees_knn.py
Deps: scikit-learn, numpy (uv add scikit-learn numpy)
"""
import logging

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def decision_tree_demo() -> None:
    log.info("=== Decision Tree ===")
    iris = load_iris()
    X, y = iris.data, iris.target  # type: ignore[attr-defined]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results: list[tuple[str, float, float, int]] = []
    for max_depth in [None, 2, 3, 5, 8]:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_tr, y_tr)
        train_acc = accuracy_score(y_tr, clf.predict(X_tr))
        test_acc  = accuracy_score(y_te, clf.predict(X_te))
        n_leaves  = clf.get_n_leaves()
        results.append((str(max_depth), train_acc, test_acc, n_leaves))

    log.info("%-12s %-12s %-12s %-8s", "max_depth", "Train acc", "Test acc", "Leaves")
    for row in results:
        log.info("%-12s %-12.4f %-12.4f %-8d", *row)

    # Show compact tree text for max_depth=3
    clf3 = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf3.fit(X_tr, y_tr)
    tree_text = export_text(clf3, feature_names=iris.feature_names)  # type: ignore[arg-type]
    log.info("\nTree (max_depth=3):\n%s", tree_text)

    # Feature importances
    log.info("Feature importances (max_depth=3):")
    for feat, imp in sorted(
        zip(iris.feature_names, clf3.feature_importances_),  # type: ignore[arg-type]
        key=lambda t: -t[1],
    ):
        log.info("  %-25s %.4f", feat, imp)


def knn_demo() -> None:
    log.info("\n=== k-NN ===")
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=6, random_state=7
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=7
    )

    log.info("%-6s %-14s %-14s", "k", "No-scale acc", "Scaled acc")
    for k in [1, 3, 5, 11, 21, 51]:
        # Without scaling
        clf_raw = KNeighborsClassifier(n_neighbors=k)
        clf_raw.fit(X_tr, y_tr)
        acc_raw = accuracy_score(y_te, clf_raw.predict(X_te))

        # With scaling (correct approach)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=k))])
        pipe.fit(X_tr, y_tr)
        acc_scaled = accuracy_score(y_te, pipe.predict(X_te))

        log.info("%-6d %-14.4f %-14.4f", k, acc_raw, acc_scaled)


def knn_scale_sensitivity() -> None:
    """Demonstrate that k-NN breaks on unscaled features with very different ranges."""
    log.info("\n=== k-NN scale sensitivity ===")
    rng = np.random.default_rng(0)
    # Feature 0: [0,1], Feature 1: [0,1000] — Euclidean distance dominated by feat 1
    X = np.column_stack([rng.uniform(0, 1, 200), rng.uniform(0, 1000, 200)])
    # Label = 1 if feat 0 > 0.5 (irrelevant to feat 1)
    y = (X[:, 0] > 0.5).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    raw = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    scaled = Pipeline([("sc", StandardScaler()),
                       ("knn", KNeighborsClassifier(n_neighbors=5))]).fit(X_tr, y_tr)

    log.info("Unscaled k-NN accuracy: %.4f", accuracy_score(y_te, raw.predict(X_te)))
    log.info("Scaled   k-NN accuracy: %.4f", accuracy_score(y_te, scaled.predict(X_te)))


def main() -> None:
    decision_tree_demo()
    knn_demo()
    knn_scale_sensitivity()


if __name__ == "__main__":
    main()
```

Run:
```bash
python day3_trees_knn.py
```

## Exercises

1. **Pruning experiment**: On `make_classification(n_samples=1000, n_features=20, n_informative=10)`, train an unpruned decision tree. Record its depth, number of leaves, and train vs. test accuracy. Then sweep `max_depth` from 1 to 15. Plot (or log) the train/test accuracy curve. What depth minimises the gap?

2. **Cost-complexity pruning**: Fit a deep tree on Iris, then call `clf.cost_complexity_pruning_path(X_tr, y_tr)` to get `ccp_alphas`. Retrain a tree for each alpha. Log train/test accuracy vs. alpha. How does this compare to `max_depth` pruning?

3. **k-NN metric comparison**: On `load_iris()`, compare `metric='euclidean'`, `metric='manhattan'`, and `metric='chebyshev'` for k=5 (after scaling). Which performs best on the test set?

4. **Curse of dimensionality**: Generate datasets with `make_classification(n_samples=500, n_features=d)` for d = 2, 5, 10, 20, 50, 100. For each, fit k-NN (k=5, scaled) and record test accuracy. Write two sentences explaining the trend.

5. **Feature importance pitfall**: Fit a decision tree on a dataset where you add a high-cardinality random feature (e.g., a random integer column). Does it appear in feature importances? Why might this be misleading?

(Solutions: ../solutions/week-02/day-3.md)

## Self-check
- [ ] I can explain Gini impurity and when a split is "pure".
- [ ] I can implement a decision tree, tune `max_depth`, and read `feature_importances_`.
- [ ] I can build a k-NN pipeline with scaling and select k via a sweep.
- [ ] I can explain why k-NN degrades in high-dimensional spaces.

## Supplementary links
- Scikit-learn: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- Scikit-learn: [Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html)
- [A Visual Introduction to Machine Learning (Part I) — r2d3.us](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
