# Solutions — Day 2: Linear & Logistic Regression

## Exercise 1 — Regularisation path

```python
import numpy as np
import logging
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

X, y = make_regression(n_samples=300, n_features=20, n_informative=5, noise=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

log.info("%-10s %-10s", "alpha", "Test RMSE")
for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    pipe = Pipeline([("sc", StandardScaler()), ("reg", Ridge(alpha=alpha))])
    pipe.fit(X_tr, y_tr)
    rmse = mean_squared_error(y_te, pipe.predict(X_te)) ** 0.5
    log.info("%-10.3f %-10.3f", alpha, rmse)
```

**Expected**: RMSE decreases initially as regularisation reduces overfitting, then increases as the model underfits. The optimal alpha is typically in the 1–10 range for this dataset.

---

## Exercise 2 — Lasso feature selection

```python
import numpy as np
import logging
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

X, y = make_regression(n_samples=300, n_features=50, n_informative=5, noise=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
lasso = Lasso(alpha=0.5).fit(X_tr_s, y_tr)

non_zero = np.where(lasso.coef_ != 0)[0]
log.info("Non-zero features: %s", non_zero.tolist())
log.info("Count: %d / 50", len(non_zero))
```

**Discussion**: Lasso with alpha=0.5 should zero out ~40–45 of the 50 features. The surviving features should overlap significantly with the 5 informative ones, but Lasso may miss some informative features if they are correlated with others (collinearity confuses L1 penalisation — ElasticNet is more reliable then).

---

## Exercise 3 — C parameter in logistic regression

```python
import numpy as np
import logging
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

log.info("%-10s %-12s %-12s", "C", "Train acc", "Test acc")
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=500))])
    pipe.fit(X_tr, y_tr)
    log.info("%-10.3f %-12.4f %-12.4f",
             C, accuracy_score(y_tr, pipe.predict(X_tr)),
             accuracy_score(y_te, pipe.predict(X_te)))
```

**Pattern**: very low C (high regularisation) → underfitting (low train and test accuracy). Very high C → overfitting on this small dataset (train > test). Optimal typically around C=1–10 for Iris.

---

## Exercise 4 — Coefficient meaning

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import logging

log = logging.getLogger(__name__)

data = load_diabetes()
X, y = data.data, data.target
sc = StandardScaler()
X_s = sc.fit_transform(X)
model = LinearRegression().fit(X_s, y)

bmi_idx = list(data.feature_names).index("bmi")
bmi_coef = model.coef_[bmi_idx]
log.info("BMI standardised coefficient: %.3f", bmi_coef)
```

**Interpretation**: "A one standard-deviation increase in BMI is associated with a predicted increase of `{bmi_coef:.1f}` units in disease progression score, holding all other patient characteristics constant."

---

## Exercise 5 — Leakage fix

**Bug**: scaling is applied to the full `X` before splitting. The scaler's `mean_` and `scale_` are computed using test-set statistics, so the test set is no longer truly unseen.

**Fixed code**:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Correct: split first, then scale inside a Pipeline
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
pipe.fit(X_tr, y_tr)
pipe.predict(X_te)
```
