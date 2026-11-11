# Solutions — Week 05 · Day 5

## Exercise 1: F1@k and EvalResult update

```python
def f1_at_k(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall. Returns 0 if both are zero."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
```

```python
@dataclass
class EvalResult:
    precision: float
    recall: float
    mrr: float
    ndcg: float
    f1: float = 0.0   # added field
```

In `evaluate_retrieval`:
```python
result = EvalResult(
    precision=float(np.mean(p_scores)),
    recall=float(np.mean(r_scores)),
    mrr=float(np.mean(rr_scores)),
    ndcg=float(np.mean(ndcg_scores)),
    f1=f1_at_k(float(np.mean(p_scores)), float(np.mean(r_scores))),
)
```

## Exercise 2: compare_pipelines

```python
def compare_pipelines(
    cases: list[RetrievalEvalCase],
    baseline_fn: Callable[[str], list[str]],
    improved_fn: Callable[[str], list[str]],
    k: int = 5,
) -> None:
    baseline = evaluate_retrieval(cases, baseline_fn, k=k)
    improved = evaluate_retrieval(cases, improved_fn, k=k)

    header = f"{'Metric':<12} {'Baseline':>10} {'Improved':>10} {'Delta':>10}"
    sep = "-" * len(header)
    log.info("\n%s\n%s", header, sep)
    for metric in ("precision", "recall", "mrr", "ndcg"):
        b = getattr(baseline, metric)
        i = getattr(improved, metric)
        delta = i - b
        sign = "+" if delta >= 0 else ""
        log.info("%-12s %10.3f %10.3f %s%10.3f", metric.upper(), b, i, sign, delta)
```

## Exercise 3: Synthetic eval set and hand-calculated verification

```python
# Synthetic 10-case eval set
from day5_metrics import RetrievalEvalCase, evaluate_retrieval

# Corpus: 5 documents
# Relevant mappings are crafted so we can hand-verify

cases = [
    RetrievalEvalCase("q1", {"d1", "d2"}, {"d1": 2, "d2": 1}),
    RetrievalEvalCase("q2", {"d3"},       {"d3": 2}),
    RetrievalEvalCase("q3", {"d1", "d3"}, {"d1": 2, "d3": 1}),
    # ... 7 more
]

def perfect_retriever(query: str) -> list[str]:
    """Always returns relevant docs first."""
    mapping = {"q1": ["d1", "d2", "d4"], "q2": ["d3", "d1"], "q3": ["d1", "d3", "d2"]}
    return mapping.get(query, [])

result = evaluate_retrieval(cases[:3], perfect_retriever, k=2)

# Hand-calculated:
# q1: retrieved=[d1,d2], relevant={d1,d2} → P@2=1.0, R@2=1.0, RR=1.0
# q2: retrieved=[d3,d1], relevant={d3}    → P@2=0.5, R@2=1.0, RR=1.0
# q3: retrieved=[d1,d3], relevant={d1,d3} → P@2=1.0, R@2=1.0, RR=1.0
# macro P@2 = (1.0+0.5+1.0)/3 ≈ 0.833
# macro R@2 = (1.0+1.0+1.0)/3 = 1.0
# macro MRR = 1.0

assert abs(result.mrr - 1.0) < 1e-6
assert abs(result.recall - 1.0) < 1e-6
assert abs(result.precision - (1.0 + 0.5 + 1.0) / 3) < 1e-6
```

**Note:** For NDCG hand-calculation at k=2:

```
q1: retrieved=[d1,d2], grades={d1:2, d2:1}
DCG@2 = (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3)
       = 3/1.0 + 1/1.585
       = 3.0 + 0.631 = 3.631
Ideal: [d1,d2] → same order → IDCG = 3.631
NDCG@2(q1) = 1.0
```
