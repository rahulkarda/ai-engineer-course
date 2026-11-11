# Week 10 Project — Inference Optimization Toolkit

## Overview

Build a four-component Python toolkit that measures, models, and optimizes LLM inference for production workloads. The toolkit exposes a FastAPI service with Server-Sent Events (SSE) streaming and a `/metrics` endpoint. All Anthropic API calls must be mockable so the full test suite runs without network access.

## Goals

- Understand the cost structure of Anthropic model tiers (haiku / sonnet / opus) and translate token counts into dollar estimates.
- Instrument streaming responses to capture Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT) independently.
- Simulate concurrent request load with `asyncio` and compute p50/p95/p99 latency distributions.
- Implement a rule-based decision guide that maps workload constraints to a concrete serving configuration.
- Wire all four components into a FastAPI application with live SSE output and a Prometheus-compatible metrics endpoint.

## Architecture

The toolkit has four independent components that are composed in the FastAPI layer.

**Component 1 — TokenCostCalculator**
Accepts a model name, input token count, and output token count. Returns a cost breakdown using the pricing table for `claude-haiku-5-5`, `claude-sonnet-5-5`, and `claude-opus-5-5`. Pricing is loaded from a Pydantic settings object backed by environment variables, not hardcoded literals. Exposes a `batch_estimate` method that accepts a list of `(input_tokens, output_tokens)` tuples and returns aggregate cost and per-request cost.

**Component 2 — StreamingClient**
Wraps the Anthropic Messages API streaming endpoint. On each streaming event it records the wall-clock timestamp. After the stream completes it computes TTFT (time from request send to first token received) and TBT (median inter-token interval). The client is designed around a protocol / abstract interface so tests can substitute a mock that yields synthetic token events at controlled intervals without any real API call.

**Component 3 — LoadTestHarness**
Uses `asyncio.gather` to fire `n` concurrent requests through `StreamingClient`. Collects per-request TTFT and total latency. After all requests complete, computes p50, p95, and p99 for both metrics. Accepts a `concurrency` parameter and a `target_rps` parameter; when `target_rps` is set it uses a token-bucket rate limiter to pace requests. Results are returned as a `LoadTestResult` Pydantic model.

**Component 4 — DecisionGuide**
Pure-logic component that takes a `WorkloadConstraints` Pydantic model (max_latency_ms, max_cost_per_1k, min_throughput_rps, input_output_token_ratio) and returns a `ServingConfig` Pydantic model (model, quantization, use_batching, use_caching, use_streaming, notes). The decision tree must have at least four distinct branches covering: latency-dominant, cost-dominant, throughput-dominant, and balanced profiles.

## Deliverables

- [ ] `TokenCostCalculator` with per-token pricing for `claude-haiku-5-5`, `claude-sonnet-5-5`, and `claude-opus-5-5`; pricing configurable via environment variables
- [ ] `StreamingClient` with TTFT and TBT tracking; injectable mock interface for tests
- [ ] `LoadTestHarness` using `asyncio` concurrency with p50/p95/p99 latency output as a `LoadTestResult` Pydantic model
- [ ] `DecisionGuide` with constraint-based recommendations covering at least four decision branches
- [ ] FastAPI application with an SSE streaming endpoint (`POST /generate`) and a Prometheus-compatible metrics endpoint (`GET /metrics`)
- [ ] Full `pytest` suite with all Anthropic API calls mocked; minimum 80% line coverage measured by `pytest-cov`

## Folder structure

```
week-10-project/
    src/
        inference_toolkit/
            __init__.py
            calculator.py        # TokenCostCalculator
            streaming.py         # StreamingClient + mock interface
            harness.py           # LoadTestHarness
            guide.py             # DecisionGuide
            models.py            # all Pydantic models
            config.py            # pydantic-settings Config
            api.py               # FastAPI app, /generate, /metrics
    tests/
        unit/
            test_calculator.py
            test_guide.py
        integration/
            test_harness.py
            test_api.py
    pyproject.toml
    .env.example
    .gitignore
```

## Grading

| Dimension | Weight | Pass threshold |
|---|---|---|
| All six deliverables present and functional | 35% | all six |
| Pytest suite passes with >= 80% line coverage | 25% | >= 80% |
| Pydantic v2 models used for all data boundaries | 15% | no untyped dicts crossing component interfaces |
| FastAPI endpoints respond correctly (SSE + metrics) | 15% | manual smoke test passes |
| Code quality: type hints, logging, no secrets, ruff clean | 10% | zero ruff errors |

Overall pass threshold: >= 75%.

## Stretch goals

1. Add prompt caching support to `StreamingClient`: when the same system prompt is reused across requests, send the `cache_control` header and measure cache hit rate and cost reduction. Report cache hit rate in `/metrics`.

2. Implement speculative decoding simulation in `LoadTestHarness`: model a draft acceptance rate parameter and simulate the latency impact of speculative decoding vs standard decoding at varying acceptance rates, then plot p95 latency as a function of acceptance rate.

3. Replace the rule-based `DecisionGuide` with a constrained optimization formulation using `scipy.optimize.linprog` or `pulp`, where cost and latency are objective terms and throughput is a constraint. Compare its recommendations against the rule-based version on a set of benchmark profiles.

## Supplementary links

- vLLM docs: https://docs.vllm.ai
- TGI docs: https://huggingface.co/docs/text-generation-inference
- Anthropic streaming: https://docs.anthropic.com/en/api/messages-streaming
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Continuous batching: https://www.anyscale.com/blog/continuous-batching-llm-inference
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
