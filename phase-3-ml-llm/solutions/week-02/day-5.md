# Solutions — Day 5: Evaluation Metrics, Cross-Validation & Pipelines

## Exercise 1 — Threshold tuning

```python
import logging
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

data = load_breast_cancer()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
pipe.fit(X_tr, y_tr)
proba = pipe.predict_proba(X_te)[:, 1]

log.info("%-10s %-12s %-12s %-12s", "Threshold", "Precision", "Recall", "F1")
for t in np.arange(0.1, 1.0, 0.1):
    y_pred = (proba >= t).astype(int)
    if y_pred.sum() == 0:
        continue
    log.info("%-10.1f %-12.4f %-12.4f %-12.4f",
             t, precision_score(y_te, y_pred, zero_division=0),
             recall_score(y_te, y_pred),
             f1_score(y_te, y_pred, zero_division=0))
```

**Discussion**: lowering the threshold increases recall but decreases precision. F1 peaks somewhere in the middle (often near 0.4–0.6). Default threshold of 0.5 is not always optimal.

---

## Exercise 2 — Imbalanced dataset

```python
import logging
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
y_proba = pipe.predict_proba(X_te)[:, 1]

log.info("Accuracy:        %.4f  ← misleading (95%% would give 0.95)", accuracy_score(y_te, y_pred))
log.info("F1 macro:        %.4f  ← more honest", f1_score(y_te, y_pred, average="macro"))
log.info("F1 weighted:     %.4f  ← dominated by majority", f1_score(y_te, y_pred, average="weighted"))
log.info("ROC-AUC:         %.4f  ← best single metric here", roc_auc_score(y_te, y_proba))
```

**Discussion**: accuracy is misleading (a model predicting all 0 gets 95%). ROC-AUC and macro-F1 reveal the model's true discriminative power. Weighted F1 can still be inflated by majority-class performance.

---

## Exercise 3 — CV scorer selection

```python
import logging
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

data = load_breast_cancer()
X, y = data.data, data.target
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for C in [0.01, 10]:
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=500))])
    log.info("C=%.2f", C)
    for scoring in ["accuracy", "f1", "roc_auc"]:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        log.info("  %-12s mean=%.4f  std=%.4f", scoring, scores.mean(), scores.std())
```

**Discussion**: on a nearly balanced dataset like breast cancer, rankings are similar across metrics. On imbalanced datasets, a high-C model may rank #1 by accuracy but #2 by F1. Scorer choice should match the business objective.

---

## Exercise 4 — ColumnTransformer pipeline

```python
import logging
import numpy as np
from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

log = logging.getLogger(__name__)
rng = np.random.default_rng(0)

iris = load_iris()
X_num = iris.data
cat_col = rng.choice(["A", "B", "C"], size=150).reshape(-1, 1)
import pandas as pd
X = np.hstack([X_num, cat_col])

ct = ColumnTransformer([
    ("num", StandardScaler(), [0, 1, 2, 3]),
    ("cat", OneHotEncoder(sparse_output=False), [4]),
])
pipe = Pipeline([("prep", ct), ("clf", LogisticRegression(max_iter=500))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, iris.target, cv=cv, scoring="accuracy")
log.info("CV accuracy: mean=%.4f  std=%.4f", scores.mean(), scores.std())
```

---

## Exercise 5 — Nested CV

```python
import logging
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

data = load_breast_cancer()
X, y = data.data, data.target

base_pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])

# Non-nested (optimistic)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
non_nested = cross_val_score(base_pipe, X, y, cv=outer_cv, scoring="roc_auc")

# Nested
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
gs = GridSearchCV(base_pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner_cv, scoring="roc_auc")
nested = cross_val_score(gs, X, y, cv=outer_cv, scoring="roc_auc")

log.info("Non-nested AUC: mean=%.4f  std=%.4f", non_nested.mean(), non_nested.std())
log.info("Nested    AUC: mean=%.4f  std=%.4f", nested.mean(), nested.std())
log.info("Optimistic bias estimate: %.4f", non_nested.mean() - nested.mean())
```

**Discussion**: the nested estimate is slightly lower than the non-nested one because it correctly accounts for the variance introduced by hyperparameter search. On small datasets the bias can be 0.01–0.03 AUC points.
