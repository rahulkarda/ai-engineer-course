# Week 02 Assessment — ML Fundamentals & scikit-learn

> Time: ~1 hr  ·  Pass gate: 75%  ·  Answer key: ../solutions/week-02/assessment.md

## Conceptual (interview-style) — 25%

Answer each question concisely (2–5 sentences). No code required.

1. Explain the bias-variance tradeoff. Give one example of a high-bias model and one example of a high-variance model from the algorithms covered this week.

2. Why must you never fit a `StandardScaler` on the full dataset before cross-validation? What specifically goes wrong?

3. A colleague says: "My model has 99% accuracy — it must be great." The dataset is 99% class 0 and 1% class 1. What is likely happening, and what metric would you use instead?

4. Explain the difference between k-Means and k-NN. They both use "k" — what does k mean in each, and what is each algorithm doing?

5. You have a logistic regression model with high training F1 and low test F1. Suggest two remedies and explain why each might help.

6. What is the difference between macro-averaged F1 and weighted-averaged F1? When would each be misleading?

7. Describe the `sklearn.pipeline.Pipeline` contract: what happens during `pipe.fit(X_tr, y_tr)` and what happens during `pipe.predict(X_te)`?

8. Why does k-NN degrade on high-dimensional data? What is this phenomenon called?

---

## Coding Challenge — 40%

Build a complete, tested ML pipeline on `load_wine()` (multiclass, 3 classes, 13 features).

**Requirements**:

1. Stratified 80/20 train/test split.
2. A `Pipeline` with `StandardScaler` + `LogisticRegression(multi_class='multinomial', max_iter=1000)`.
3. 5-fold stratified CV reporting ROC-AUC (macro, one-vs-rest) mean and std.
4. Test-set report: accuracy, macro-F1, confusion matrix (logged, not printed).
5. At least 5 pytest test cases covering: data shape, split proportions, pipeline type, prediction shape, test accuracy ≥ 0.9.
6. Type hints throughout. `logging` not `print`. Deps noted (not installed by you).

**Acceptance criteria**:
- Accuracy ≥ 0.90 on test set.
- CV AUC std < 0.05.
- All tests pass.

**Deliverable**: a single runnable file `assessment_solution.py` plus `test_assessment.py`.

---

## Debugging Challenge — 20%

The following code has **three bugs**. Identify all three, explain each one, and write the corrected code.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

# Bug-ridden code
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:120], X_scaled[120:]
y_train, y_test = y[:120], y[120:]

model = LogisticRegression()
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.4f}")

y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)
print(f"Test AUC: {auc:.4f}")
```

---

## Code Quality — 15%

Review the following implementation and list **all** code quality issues. For each issue, state: what the problem is, why it matters, and the fix.

```python
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

data = load_breast_cancer()
X = data['data']
y = data['target']

model = SVC()
model.fit(X, y)

result = model.predict(X)
print("accuracy:", sum(result == y) / len(y))

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

---

## Rubric

| Dimension | Weight | Pass threshold |
|-----------|--------|---------------|
| Independent implementation (coding challenge) | 40% | Model meets acceptance criteria, tests pass |
| Conceptual understanding (Q&A) | 25% | ≥5/8 questions answered correctly |
| Debugging & problem-solving | 20% | ≥2/3 bugs found and correctly fixed |
| Code quality & testing | 15% | ≥4 issues identified; test coverage ≥ 5 cases |

**Pass gate: 75% overall.**

Answer key: ../solutions/week-02/assessment.md
