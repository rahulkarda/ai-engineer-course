# Phase 3 · Week 02 · Day 2 — Linear & Logistic Regression

> Objective: Understand and implement linear and logistic regression from first principles, interpret coefficients, and apply regularisation.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

### Linear Regression

Linear regression models a continuous target as a weighted sum of features:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ  =  Xw
```

**Loss function** — Mean Squared Error (MSE):

```
L(w) = (1/n) Σ (yᵢ - ŷᵢ)²
```

**Closed-form solution** (Ordinary Least Squares):

```
w* = (XᵀX)⁻¹ Xᵀy
```

Exists only when XᵀX is invertible (no perfectly collinear features). For large n or high d, gradient descent is preferred.

**Assumptions** (OLS): linearity, homoscedasticity (constant variance), independence of residuals, no multicollinearity.

### Regularisation

| Method | Penalty | Effect |
|--------|---------|--------|
| Ridge (L2) | λ‖w‖₂² | Shrinks all weights; never zeros out |
| Lasso (L1) | λ‖w‖₁ | Sparse solution; some weights → 0 |
| ElasticNet | α·L1 + (1-α)·L2 | Combination; good for correlated features |

λ (alpha in sklearn) controls regularisation strength. Higher λ → higher bias, lower variance.

### Logistic Regression

For binary classification, pass the linear output through the **sigmoid** function:

```
P(y=1 | x) = σ(Xw) = 1 / (1 + e^(-Xw))
```

**Loss** — Binary Cross-Entropy (log-loss):

```
L(w) = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

No closed form; optimised with gradient descent (scipy's L-BFGS-B in sklearn).

For **multi-class**: sklearn uses One-vs-Rest (OvR) or Softmax (multinomial) depending on `multi_class` parameter.

## How it works internally

**Gradient descent intuition**: the gradient ∂L/∂w points uphill in parameter space. We step in the negative direction:
```
w ← w - η · ∂L/∂w
```
Learning rate η balances convergence speed vs. stability.

**Ridge keeps all features**: L2 penalty creates a bowl around the origin — the cost of a weight being large grows quadratically, so weights shrink but rarely reach zero.

**Lasso creates sparsity**: L1 penalty has a sharp corner at w=0. The gradient discontinuity means the optimiser can land exactly at zero for irrelevant features — automatic feature selection.

**Sklearn `LogisticRegression`**: default solver is `lbfgs`, default regularisation is L2 with `C=1.0` where `C = 1/λ`. Higher C = less regularisation.

**Coefficient interpretation**: for linear regression, `wⱼ` is the expected change in y when xⱼ increases by 1 unit, holding all other features constant. For logistic regression, `wⱼ` is the log-odds change per unit increase in xⱼ.

## Worked example

```python
"""
day2_regression.py
Deps: scikit-learn, numpy, pandas (uv add scikit-learn numpy pandas)
"""
import logging

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def linear_regression_demo() -> None:
    log.info("=== Linear Regression ===")
    X, y = make_regression(n_samples=300, n_features=10, n_informative=5,
                            noise=20, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results: list[tuple[str, float, float]] = []

    for name, model in [
        ("OLS",        LinearRegression()),
        ("Ridge(α=1)", Ridge(alpha=1.0)),
        ("Ridge(α=10)",Ridge(alpha=10.0)),
        ("Lasso(α=1)", Lasso(alpha=1.0)),
    ]:
        # Always scale before regularised regression
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        rmse = mean_squared_error(y_te, y_pred) ** 0.5
        r2 = r2_score(y_te, y_pred)
        results.append((name, rmse, r2))

    log.info("%-18s %-8s %-6s", "Model", "RMSE", "R²")
    for name, rmse, r2 in results:
        log.info("%-18s %-8.2f %-6.4f", name, rmse, r2)


def coefficient_inspection() -> None:
    log.info("\n=== Coefficient inspection (Ridge vs Lasso) ===")
    X, y = make_regression(n_samples=200, n_features=8, n_informative=3,
                            noise=10, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    ridge = Ridge(alpha=5).fit(X_tr_s, y_tr)
    lasso = Lasso(alpha=5).fit(X_tr_s, y_tr)

    df = pd.DataFrame(
        {"feature": [f"x{i}" for i in range(8)],
         "ridge_coef": ridge.coef_,
         "lasso_coef": lasso.coef_}
    )
    log.info("\n%s", df.to_string(index=False))
    n_zero_lasso = (lasso.coef_ == 0).sum()
    log.info("Lasso zeroed out %d / 8 features (sparsity)", n_zero_lasso)


def logistic_regression_demo() -> None:
    log.info("\n=== Logistic Regression (Iris, binary: setosa vs rest) ===")
    iris = load_iris()
    X, y = iris.data, (iris.target == 0).astype(int)  # type: ignore[attr-defined]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=200)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    log.info("Accuracy: %.4f", accuracy_score(y_te, y_pred))
    log.info("\n%s", classification_report(y_te, y_pred,
                                           target_names=["not setosa", "setosa"]))

    # Inspect coefficients (from scaled space)
    coefs = pipe.named_steps["clf"].coef_[0]
    for feat, coef in zip(iris.feature_names, coefs):  # type: ignore[attr-defined]
        log.info("  %-25s log-odds coef: %+.3f", feat, coef)


def main() -> None:
    linear_regression_demo()
    coefficient_inspection()
    logistic_regression_demo()


if __name__ == "__main__":
    main()
```

Run:
```bash
python day2_regression.py
```

## Exercises

1. **Regularisation path**: Using `make_regression(n_features=20, n_informative=5)`, fit Ridge with `alpha` values `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`. Record test RMSE for each. Where does it start hurting performance? Plot or log.

2. **Lasso feature selection**: On a 50-feature dataset (`make_regression(n_features=50, n_informative=5)`), fit Lasso with `alpha=0.5` after scaling. Which features survive (non-zero coefficient)? How many? Does the set include all 5 informative features?

3. **C parameter in logistic regression**: Using `load_iris()` (3-class), vary `C` from `0.001` to `100` (log scale). Log train and test accuracy for each. Explain the pattern in terms of bias-variance.

4. **Coefficient meaning**: Fit a linear regression on `load_diabetes()`. After standardising features, interpret the coefficient of `bmi`. Write one sentence explaining it to a non-technical stakeholder.

5. **Leakage fix**: The following code is broken — identify the bug and fix it:
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2)
   model.fit(X_tr, y_tr)
   ```

(Solutions: ../solutions/week-02/day-2.md)

## Self-check
- [ ] I can write the MSE and binary cross-entropy loss functions from memory.
- [ ] I can explain what L1 vs. L2 regularisation does to coefficients.
- [ ] I can build a sklearn Pipeline with scaler + model to avoid leakage.
- [ ] I can interpret a logistic regression coefficient as a log-odds change.

## Supplementary links
- Scikit-learn: [Linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Regularisation visualised — distill.pub](https://distill.pub/2019/visual-exploration-gaussian-processes/) (conceptual)
- ESL ch. 3 (Linear Methods for Regression)
