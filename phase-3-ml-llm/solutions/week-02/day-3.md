# Solutions — Day 3: Decision Trees & k-NN

## Exercise 1 — Pruning experiment

```python
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Unpruned tree
unpruned = DecisionTreeClassifier(random_state=0)
unpruned.fit(X_tr, y_tr)
log.info("Unpruned: depth=%d  leaves=%d  train=%.4f  test=%.4f",
         unpruned.get_depth(), unpruned.get_n_leaves(),
         accuracy_score(y_tr, unpruned.predict(X_tr)),
         accuracy_score(y_te, unpruned.predict(X_te)))

log.info("%-12s %-12s %-12s", "max_depth", "Train acc", "Test acc")
for depth in range(1, 16):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
    clf.fit(X_tr, y_tr)
    log.info("%-12d %-12.4f %-12.4f",
             depth, accuracy_score(y_tr, clf.predict(X_tr)),
             accuracy_score(y_te, clf.predict(X_te)))
```

**Expected**: unpruned tree has 100% train accuracy but lower test accuracy (overfitting). Test accuracy peaks around max_depth=8–10 then plateaus or slightly decreases.

---

## Exercise 2 — Cost-complexity pruning

```python
import numpy as np
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
ccp_alphas = path.ccp_alphas[:-1]  # exclude last (trivial root)

log.info("%-12s %-12s %-12s", "ccp_alpha", "Train acc", "Test acc")
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)
    clf.fit(X_tr, y_tr)
    log.info("%-12.6f %-12.4f %-12.4f",
             alpha, accuracy_score(y_tr, clf.predict(X_tr)),
             accuracy_score(y_te, clf.predict(X_te)))
```

**Discussion**: as ccp_alpha increases, the tree is pruned more aggressively (fewer nodes). Test accuracy first improves (removing noise) then degrades (removing signal). CCP pruning often finds a better tradeoff than a fixed max_depth because it directly optimises the impurity/complexity tradeoff.

---

## Exercise 3 — k-NN metric comparison

```python
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

for metric in ["euclidean", "manhattan", "chebyshev"]:
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric=metric)),
    ])
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    log.info("%-15s accuracy=%.4f", metric, acc)
```

**Typical result**: all three metrics perform similarly on Iris (~0.97). Euclidean and Manhattan are usually near-equal for normalised data. Chebyshev (max of absolute differences) can be slightly lower.

---

## Exercise 4 — Curse of dimensionality

```python
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

log = logging.getLogger(__name__)

log.info("%-6s %-12s", "d", "Test acc")
for d in [2, 5, 10, 20, 50, 100]:
    X, y = make_classification(n_samples=500, n_features=d, n_informative=max(2, d//4), random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=5))])
    pipe.fit(X_tr, y_tr)
    log.info("%-6d %-12.4f", d, accuracy_score(y_te, pipe.predict(X_te)))
```

**Explanation (2 sentences)**: As dimensionality grows, the volume of the space expands exponentially, so training points become sparse and all pairwise distances converge to similar values. The "nearest" neighbour becomes no more informative than a random point, causing k-NN's accuracy to degrade — this is the curse of dimensionality.

---

## Exercise 5 — Feature importance pitfall

```python
import numpy as np
import logging
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger(__name__)

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, random_state=0)
# Add high-cardinality random integer column
random_col = rng.integers(0, 500, size=(500, 1))
X_aug = np.hstack([X, random_col])

X_tr, X_te, y_tr, y_te = train_test_split(X_aug, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_tr, y_tr)

log.info("Feature importances (last = random int column):")
for i, imp in enumerate(clf.feature_importances_):
    log.info("  feature_%d: %.4f", i, imp)
```

**Discussion**: The high-cardinality random integer column may receive non-trivial importance because decision trees can exploit it as a pseudo-unique ID (high split count → high impurity reduction). This is a known bias of Gini/entropy-based importance for high-cardinality features. Permutation importance (see `sklearn.inspection.permutation_importance`) is more reliable.
