# Week 10 Assessment

## Conceptual (interview-style)

1. Explain the difference between Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT). In what scenario would you accept a high TTFT in exchange for a low TBT, and why?

2. What problem does continuous batching solve compared to static batching in LLM serving? Describe the mechanism and quantify the expected throughput improvement in practical terms.

3. Derive the memory cost formula for a KV cache. Given a model with 32 layers, 32 attention heads, a head dimension of 128, a sequence length of 4096 tokens, and FP16 precision, calculate the KV cache memory requirement per request.

4. Compare INT8, INT4, and FP16 quantization for LLM inference. For each, state: typical accuracy degradation, memory reduction vs FP16 baseline, and the class of workloads where it is the best choice.

5. Speculative decoding uses a small draft model to propose tokens that a large verifier model accepts or rejects. What factors determine the acceptance rate, and how does a low acceptance rate affect end-to-end latency compared to standard decoding?

6. You are deploying an open-source 70B-parameter model. Compare vLLM and Text Generation Inference (TGI) as serving frameworks. What are the two or three most decisive factors in choosing one over the other for a production system with strict SLA requirements?

7. Distinguish semantic caching from exact-match caching for LLM responses. Describe a realistic failure mode for each approach — a case where the cache returns a result that should not have been served.

8. A batch processing pipeline sends prompts that share a 2,000-token system prompt followed by a short user query. List three distinct strategies to reduce prompt token cost and latency, and explain the mechanism behind each.

## Coding challenge

### Task

Implement an `InferenceOptimizer` class in Python 3.12 that selects an inference configuration given a workload profile.

### Models

```python
from pydantic import BaseModel, Field

class RequestProfile(BaseModel):
    avg_input_tokens: int = Field(..., gt=0)
    avg_output_tokens: int = Field(..., gt=0)
    requests_per_second: float = Field(..., gt=0)
    latency_budget_ms: int = Field(..., gt=0)
    cost_budget_per_1k_requests: float = Field(..., gt=0)

class OptimizationPlan(BaseModel):
    recommended_model: str
    recommended_quantization: str  # "fp16", "int8", or "int4"
    use_batching: bool
    use_caching: bool
    estimated_cost_per_1k: float
    estimated_p95_latency_ms: int
    rationale: str
```

### Requirements

1. `InferenceOptimizer.optimize(profile: RequestProfile) -> OptimizationPlan` must use pure logic — no LLM calls, no external I/O.
2. The model selection must choose among `claude-haiku-5-5`, `claude-sonnet-5-5`, and `claude-opus-5-5`.
3. The recommendation logic must cover at least three distinct decision paths:
   - Latency-sensitive: `latency_budget_ms < 500`
   - Cost-sensitive: `cost_budget_per_1k_requests < 1.0`
   - High-throughput: `requests_per_second >= 50`
4. `estimated_cost_per_1k` and `estimated_p95_latency_ms` must be computed from the profile, not hardcoded constants.
5. `rationale` must be a non-empty string that names at least one deciding factor.

### Test requirements

Write a `pytest` suite in `tests/test_inference_optimizer.py` with at least three parametrized or separate test cases:

- A latency-sensitive profile (`latency_budget_ms=200`, moderate RPS, adequate cost budget) — assert `recommended_model == "claude-haiku-5-5"` and `estimated_p95_latency_ms <= 200`.
- A cost-sensitive profile (`cost_budget_per_1k_requests=0.50`, low RPS, generous latency) — assert `estimated_cost_per_1k <= 0.50`.
- A high-throughput profile (`requests_per_second=100`, generous latency and cost) — assert `use_batching == True`.

All tests must pass with `pytest -q` and no external network calls.

### Acceptance criteria

- All three test cases pass.
- `RequestProfile` and `OptimizationPlan` are Pydantic v2 `BaseModel` subclasses with typed fields.
- No `print` statements; use `logging` for any diagnostic output.
- No hardcoded secrets; any configuration uses `os.getenv`.
- Logic covers at least three distinct recommendation paths.

## Rubric (pass >= 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

Answer key: ../solutions/week-10/assessment.md
