# Solutions — Phase 3 · Week 02 · Day 7 — Project Day

## This is a project-only day

Day 7 of Week 02 has no standalone exercises. The entire session is allocated to completing the
Customer Churn Prediction Pipeline project and sitting the week assessment.

See `solutions/week-02/project.md` for the complete reference implementation of:

- `week02_project/pipeline.py` — `load_data`, `make_split`, `build_pipelines`
- `week02_project/evaluate.py` — `evaluate_on_test`, `cross_validate_pipeline`, `compare_models`
- `week02_project/train.py` — training entry point with acceptance-criteria assertions
- `week02_project/tests/test_all.py` — full pytest suite (≥ 8 test cases)

---

## Self-check before marking done

| # | Check | Pass criterion |
|---|-------|---------------|
| 1 | Data loads correctly | `X.shape == (2000, 20)`, `y.shape == (2000,)` |
| 2 | Stratified split works | `abs(y_tr.mean() - y_te.mean()) < 0.05` |
| 3 | At least 3 pipelines defined | `len(build_pipelines()) >= 3` |
| 4 | All pipelines contain a scaler | Each `Pipeline` step 0 is a `StandardScaler` |
| 5 | ROC-AUC ≥ 0.85 on test set | `metrics["roc_auc"] >= 0.85` |
| 6 | Recall ≥ 0.75 for class 1 | `metrics["recall"] >= 0.75` |
| 7 | 5-fold CV AUC std < 0.04 | `cv_result["std"] < 0.04` |
| 8 | All 8+ tests pass | `pytest -v` exits 0 |
| 9 | No `print` statements | `grep -r "print(" week02_project/` returns nothing |
| 10 | Logging used throughout | Every major step emits a `log.info(...)` line |

---

## Common pitfalls

**Recall not reaching 0.75**
`LogisticRegression` defaults to `class_weight=None`, which optimises accuracy — not recall.
Fix: use `class_weight='balanced'` so the minority class (churners) gets proportionally higher loss weight.

**CV score variance too high**
Using `cross_val_score` without `StratifiedKFold` can produce unbalanced folds on imbalanced data.
Fix: pass `cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` explicitly.

**Preprocessing leakage**
Fitting a `StandardScaler` on the full dataset before splitting leaks test statistics into training.
Fix: always wrap the scaler inside `sklearn.pipeline.Pipeline` so it is fit only on train folds.

**joblib round-trip fails**
`joblib.load` returns a new object — confirm it predicts identically:
```python
np.testing.assert_array_equal(pipe.predict(X_te), loaded.predict(X_te))
```

---

## Assessment

See `assessment.md` for the separate graded assessment (to be completed independently — no references).
