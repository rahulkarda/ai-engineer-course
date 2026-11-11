# Solutions — Day 4: Clustering & k-Means

## Exercise 1 — Elbow interpretation

```python
import logging
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

for n_centers in [3, 6]:
    log.info("\n=== centers=%d ===", n_centers)
    X, _ = make_blobs(n_samples=400, centers=n_centers, cluster_std=1.0, random_state=0)
    X_s = StandardScaler().fit_transform(X)

    log.info("%-4s %-12s %-12s", "k", "Inertia", "Silhouette")
    for k in range(2, 10):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(X_s)
        sil = silhouette_score(X_s, labels)
        log.info("%-4d %-12.1f %-12.4f", k, km.inertia_, sil)
```

**Expected**: inertia decreases monotonically. The elbow shifts to the true k (3 or 6). Silhouette peaks more sharply at the true k than the elbow does.

---

## Exercise 2 — Scale sensitivity

```python
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

rng = np.random.default_rng(0)
X = np.column_stack([rng.uniform(0, 1, 300), rng.uniform(0, 100, 300)])

for label, data in [("Unscaled", X), ("Scaled", StandardScaler().fit_transform(X))]:
    km = KMeans(n_clusters=3, random_state=0, n_init=10)
    labels = km.fit_predict(data)
    sil = silhouette_score(data, labels)
    log.info("%s  silhouette=%.4f  centroids:\n%s", label, sil, km.cluster_centers_)
```

**Discussion**: unscaled centroids are dominated by feature 1 (range 0–100). Scaling equalises both features, allowing the algorithm to find clusters that reflect structure in both features.

---

## Exercise 3 — DBSCAN comparison

```python
import logging
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

X, y = make_moons(n_samples=300, noise=0.1, random_state=0)
X_s = StandardScaler().fit_transform(X)

# k-Means
km_labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(X_s)
log.info("KMeans silhouette: %.4f", silhouette_score(X_s, km_labels))

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=5)
db_labels = db.fit_predict(X_s)
n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
if n_clusters > 1:
    sil = silhouette_score(X_s, db_labels)
    log.info("DBSCAN  silhouette: %.4f  clusters=%d  noise=%d", sil, n_clusters, n_noise)
```

**Discussion**: k-Means assumes convex clusters, so it draws a straight boundary through the two moon shapes — poor silhouette. DBSCAN finds the non-convex moon shapes naturally (much better silhouette ≈ 0.5–0.7 vs. k-Means ≈ 0.2–0.4).

---

## Exercise 4 — Soft clustering with GMM

```python
import logging
import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

iris = load_iris()
X, y = iris.data, iris.target
X_s = StandardScaler().fit_transform(X)

gm = GaussianMixture(n_components=3, random_state=0)
gm.fit(X_s)
proba = gm.predict_proba(X_s)

# Find points uncertain between any two clusters
max_p = proba.max(axis=1)
uncertain_idx = np.argsort(max_p)[:5]  # 5 most uncertain
log.info("Most uncertain points (lowest max probability):")
for idx in uncertain_idx:
    log.info("  idx=%d  features=%s  probs=%s",
             idx, np.round(X[idx], 2), np.round(proba[idx], 3))
```

**Discussion**: a GMM outputs probability vectors like `[0.48, 0.40, 0.12]` for ambiguous points near cluster boundaries. This soft membership is useful when downstream decisions can benefit from confidence levels.

---

## Exercise 5 — Cluster as feature

```python
import logging
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Baseline
pipe_base = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
pipe_base.fit(X_tr, y_tr)
log.info("Baseline accuracy: %.4f", accuracy_score(y_te, pipe_base.predict(X_te)))

# With cluster feature (CORRECT: fit KMeans only on train)
km = KMeans(n_clusters=5, random_state=0, n_init=10)
km.fit(X_tr)
X_tr_aug = np.column_stack([X_tr, km.predict(X_tr)])
X_te_aug = np.column_stack([X_te, km.predict(X_te)])

pipe_aug = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
pipe_aug.fit(X_tr_aug, y_tr)
log.info("Augmented accuracy: %.4f", accuracy_score(y_te, pipe_aug.predict(X_te_aug)))
```

**Note**: KMeans is fitted only on X_tr to avoid leakage. Improvement is dataset-dependent; for linearly separable data it rarely helps; for non-linear structure it can add signal.
