# Week 09 Assessment — Production Troubleshooting & Debugging

## Conceptual (interview-style)

Answer each question without referencing notes.

1. Describe the five-step debugging method. Give a one-sentence purpose for each step.

2. What is the difference between USE and RED as diagnostic frameworks? Which do you apply first when p99 latency rises?

3. A circuit breaker is in HALF_OPEN state. Explain what triggered this state and what the next two possible transitions are and their triggers.

4. A Python service running for 6 hours has RSS memory = 8 GB but only 200 MB of active user data. `tracemalloc` shows `dict` objects dominating. Name three code patterns that could cause this and how you would distinguish between them.

5. An LLM agent costs $0.003/request at 50 RPS. After a prompt change, cost rises to $0.031/request. What three metrics would you pull first to localize the cause?

6. A post-mortem's 5-Whys chain ends with "the engineer forgot to add the index." Is this an acceptable root cause? Explain and provide a better endpoint for the chain.

7. What is a "thundering herd" and how does double-checked locking prevent it?

8. Distinguish between *mitigation* and *fix* in an incident. Give an example where applying a fix before a mitigation would be the wrong choice.

## Coding challenge

Build a `ProductionDiagnosticSuite` that integrates all week's concepts.

### Spec

```python
# Your implementation must satisfy this interface:

class ProductionDiagnosticSuite:
    def __init__(self, pool_size: int = 10, cb_threshold: int = 5) -> None: ...

    def submit_request(self, request_id: str, should_fail: bool = False) -> dict: ...
    # Returns: {"request_id": ..., "latency_ms": ..., "status": "ok"|"error"|"fast-fail", "cost": ...}

    def get_metrics(self) -> dict: ...
    # Returns: {
    #   "error_rate": float,
    #   "p99_latency_ms": float,
    #   "pool_available": int,
    #   "circuit_state": str,
    #   "total_cost_usd": float,
    # }

    def run_rca(self) -> str: ...
    # Returns a one-paragraph root-cause summary based on observed metrics
```

### Acceptance criteria

- [ ] `submit_request` uses the connection pool (context manager) — pool available returns to `pool_size` after each call
- [ ] Circuit breaker opens after `cb_threshold` consecutive `should_fail=True` calls
- [ ] `get_metrics()` returns correct `p99_latency_ms` (within 10% of actual)
- [ ] `run_rca()` returns a non-empty string that references the circuit state and error rate
- [ ] All mock LLM costs use `os.getenv("ANTHROPIC_API_KEY")` pattern (never hardcoded)
- [ ] Minimum 5 pytest test cases covering: pool invariant, circuit open, p99 accuracy, rca non-empty, fast-fail count

### Constraints

- Python 3.12+, type hints required
- No external libraries beyond stdlib and pytest
- All LLM calls mocked — do not require a live API key
- Use `logging`, not `print`

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|-----------|--------|
| Independent implementation | 40% |
| Conceptual understanding (Q&A) | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-09/assessment.md
