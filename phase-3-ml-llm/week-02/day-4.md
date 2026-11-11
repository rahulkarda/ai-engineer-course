# Phase 3 · Week 02 · Day 4 — Clustering & k-Means

> Objective: Understand unsupervised clustering, implement k-means, select k with the elbow method and silhouette score, and know k-means limitations.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Clustering

Clustering groups data points so that points within a group are more similar to each other than to points in other groups — with no predefined labels.

**Use cases**: customer segmentation, document grouping, anomaly detection, feature engineering (cluster membership as a new feature), exploratory analysis.

### k-Means

k-Means partitions n points into k clusters by minimising **inertia** (within-cluster sum of squared distances to centroids):

```
Inertia = Σₖ Σᵢ∈Cₖ ‖xᵢ - μₖ‖²
```

**Algorithm (Lloyd's)**:
1. Randomly initialise k centroids.
2. **Assignment**: assign each point to the nearest centroid.
3. **Update**: recompute each centroid as the mean of its assigned points.
4. Repeat 2–3 until centroids stop moving (convergence).

**Convergence**: guaranteed to converge, but to a local minimum. Results depend on initialisation.

**k-Means++** (sklearn default, `init='k-means++'`): initialises centroids to be spread out, dramatically reducing bad local minima.

### Choosing k

| Method | How | Limitation |
|--------|-----|------------|
| Elbow method | Plot inertia vs. k; find the "elbow" | Elbow often ambiguous |
| Silhouette score | Measures cohesion vs. separation; range [-1, 1]; higher = better | Expensive on large n |
| Gap statistic | Compare inertia to a null reference; complex to compute | — |

**Silhouette coefficient** for point i:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
where a(i) = mean intra-cluster distance, b(i) = mean distance to nearest other cluster.

### k-Means Limitations

- Assumes **spherical, equal-size** clusters; fails on elongated or irregular shapes.
- Sensitive to **outliers** (outliers pull centroids).
- Sensitive to **scale** — standardise features first.
- Must specify k in advance.
- Non-deterministic — use `random_state` for reproducibility; multiple `n_init` runs reduce risk of bad local minimum.

### Alternatives

| Algorithm | Strength |
|-----------|---------|
| DBSCAN | Arbitrary shapes; finds outliers; no k needed |
| Agglomerative HC | Hierarchical; dendrograms; no k needed up front |
| GMM | Soft cluster membership; probabilistic |
| Spectral | Graph-based; non-convex clusters |

## How it works internally

**Time complexity**: O(n · k · d · iterations). For large datasets use `MiniBatchKMeans` which processes random batches — faster but slightly less optimal.

**Inertia always decreases with k**: a model with k=n has inertia=0. That is why you need silhouette or gap statistic rather than just minimising inertia.

**Feature scaling effect on centroids**: if one feature has range [0,1000] and another [0,1], the distance metric is dominated by the first. Centralise and scale all features before clustering.

**Evaluating unsupervised models is hard**: when ground-truth labels are available, use external metrics (`adjusted_rand_score`, `adjusted_mutual_info_score`). Without labels, internal metrics (silhouette, Davies-Bouldin) are the only option.

## Worked example

```python
"""
day4_clustering.py
Deps: scikit-learn, numpy (uv add scikit-learn numpy)
"""
import logging

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def elbow_and_silhouette(X: np.ndarray, k_range: range) -> None:
    """Log inertia and silhouette for a range of k values."""
    log.info("%-4s %-12s %-12s", "k", "Inertia", "Silhouette")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        log.info("%-4d %-12.1f %-12.4f", k, km.inertia_, sil)


def known_cluster_demo() -> None:
    """Cluster 4 known blobs; compare with ground truth using ARI."""
    log.info("=== Known blobs (k=4) ===")
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=1.2, random_state=0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=0, n_init=10)
    labels = km.fit_predict(X_scaled)

    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X_scaled, labels)
    log.info("Adjusted Rand Index: %.4f (1.0 = perfect)", ari)
    log.info("Silhouette score:    %.4f", sil)


def select_k_demo() -> None:
    """Use elbow + silhouette to pick k on data with unknown structure."""
    log.info("\n=== Selecting k (5 true clusters, hidden) ===")
    X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=1)
    X_scaled = StandardScaler().fit_transform(X)
    elbow_and_silhouette(X_scaled, range(2, 10))


def iris_clustering() -> None:
    """Cluster Iris (pretend labels unknown), compare with ground truth."""
    log.info("\n=== Iris clustering (unsupervised, k=3) ===")
    iris = load_iris()
    X, y_true = iris.data, iris.target  # type: ignore[attr-defined]
    X_scaled = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X_scaled, labels)
    log.info("ARI: %.4f  Silhouette: %.4f", ari, sil)
    log.info("Cluster sizes: %s", np.bincount(labels))


def minibatch_comparison(n_samples: int = 10_000) -> None:
    """Compare KMeans vs MiniBatchKMeans on a larger dataset."""
    import time
    log.info("\n=== KMeans vs MiniBatchKMeans (n=%d) ===", n_samples)
    X, _ = make_blobs(n_samples=n_samples, centers=8, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)

    for cls, name in [(KMeans, "KMeans"), (MiniBatchKMeans, "MiniBatchKMeans")]:
        t0 = time.perf_counter()
        km = cls(n_clusters=8, random_state=42, n_init=3)  # type: ignore[call-arg]
        labels = km.fit_predict(X_scaled)
        elapsed = time.perf_counter() - t0
        sil = silhouette_score(X_scaled, labels, sample_size=2000)
        log.info("%-20s time=%.3fs  silhouette=%.4f", name, elapsed, sil)


def main() -> None:
    known_cluster_demo()
    select_k_demo()
    iris_clustering()
    minibatch_comparison()


if __name__ == "__main__":
    main()
```

Run:
```bash
python day4_clustering.py
```

Expected: select_k_demo silhouette peaks at k=5 (the true number of clusters).

## Exercises

1. **Elbow interpretation**: Generate data with `make_blobs(centers=3)`. Run k-means for k=2..8. Log inertia. Where is the elbow? Does the silhouette score agree? Now try `make_blobs(centers=6)` — does the elbow shift?

2. **Scale sensitivity**: Create a 2-feature dataset where feature 0 ∈ [0,1] and feature 1 ∈ [0,100]. Run k-means with k=3 on unscaled and scaled versions. Log the cluster centroids and silhouette scores. Explain the difference.

3. **DBSCAN comparison**: Import `from sklearn.cluster import DBSCAN`. On `make_moons(n_samples=300, noise=0.1)`, run both k-means (k=2) and DBSCAN. Log the silhouette scores. Which performs better? Why?

4. **Soft clustering with GMM**: Import `from sklearn.mixture import GaussianMixture`. Fit a 3-component GMM on Iris. Use `gm.predict_proba(X)` to show that some points have uncertain cluster membership (probabilities close to 0.5). Pick three such points and log their feature values and probabilities.

5. **Cluster as feature**: Fit k-means (k=5) on `make_classification(n_features=10)`. Append the cluster label as a new feature column to X. Fit a logistic regression on the augmented X vs. the original X. Does adding the cluster feature improve accuracy?

(Solutions: ../solutions/week-02/day-4.md)

## Self-check
- [ ] I can explain k-means Lloyd's algorithm step by step.
- [ ] I can use silhouette score and the elbow method to select k.
- [ ] I know at least two situations where k-means fails and can name an alternative.
- [ ] I always scale features before clustering and can explain why.

## Supplementary links
- Scikit-learn: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- Scikit-learn: [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [k-Means visualiser — naftaliharris.com](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)
