# Solutions — Day 1: ML Taxonomy, Train/Test Split & Bias-Variance

## Exercise 1 — Split sensitivity

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

X, y = make_classification(n_samples=200, weights=[0.9, 0.1], random_state=0)
log.info("Full class distribution: %s", np.bincount(y))

_, _, _, y_te_no_strat = train_test_split(X, y, test_size=0.2, random_state=0)
_, _, _, y_te_strat = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

log.info("Without stratify — test class 1 count: %d", y_te_no_strat.sum())
log.info("With    stratify — test class 1 count: %d", y_te_strat.sum())
```

**Discussion**: without stratify, the 10% minority class may end up with 0–5 samples in the test set by chance. With `stratify=y`, the split is forced to maintain the 90/10 ratio. For rare classes the variance across random seeds without stratification can be extreme.

---

## Exercise 2 — Bias-variance curve

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import logging

log = logging.getLogger(__name__)

X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

log.info("%-6s %-14s %-14s", "k", "Train acc", "Test acc")
for k in [1, 5, 10, 20, 50, 100]:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_tr, y_tr)
    log.info("%-6d %-14.4f %-14.4f",
             k, accuracy_score(y_tr, clf.predict(X_tr)),
             accuracy_score(y_te, clf.predict(X_te)))
```

**Expected pattern**: k=1 has 100% train accuracy but lower test accuracy. Test accuracy rises then falls as k increases. The sweet spot is typically around k=10–20 for this dataset.

---

## Exercise 3 — Leakage detection and fix

**Bug**: `StandardScaler().fit_transform(X)` fits the scaler on the **entire** dataset including future test rows. When the model later sees test data, the scaler's mean and std already incorporate test statistics — the test set is no longer truly held-out.

**Fix**:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),   # fit only on X_tr; transform X_te
    ("model",  LogisticRegression()),
])
pipe.fit(X_tr, y_tr)
pipe.score(X_te, y_te)
```

---

## Exercise 4 — Supervised vs. unsupervised categorisation

| Scenario | Category | Reason |
|----------|---------|--------|
| a. Predicting customer churn from CRM data | **Supervised** (binary classification) | Historical labels (churned/not) available |
| b. Grouping news articles by topic | **Unsupervised** (clustering) | No predefined topic labels |
| c. Training LM to predict next word | **Self-supervised** | Labels (next token) derived from the text itself |
| d. Anomaly detection in server metrics | **Unsupervised** (or semi-supervised) | No labelled outage examples |
