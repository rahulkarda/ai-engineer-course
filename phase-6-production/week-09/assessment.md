# Week 09 Assessment — Model Gateways & Routing

> Time: ~90 min  ·  Covers: days 1–7

---

## Conceptual (interview-style)

Answer each question in 3–6 sentences. Precise terminology and concrete examples earn full marks.

1. **Provider abstraction**: What problem does a gateway's provider abstraction solve, and what interface methods should it define? Give at least two reasons why you'd express this as an abstract base class rather than a concrete class with a `provider_name` parameter.

2. **Routing policies**: Describe three routing dimensions a `ModelRouter` might consider. When would you choose a `CHEAPEST` policy over a `BEST` policy? Give a concrete real-world scenario where `CHEAPEST` is the wrong choice even if cost is a constraint.

3. **Token-bucket rate limiter**: How does a token-bucket rate limiter work? What two parameters define its behavior? Explain how it differs from a fixed-window counter and why that difference matters for bursty traffic.

4. **Caching strategies**: Distinguish exact caching from semantic caching: how is the cache key constructed in each, when does each apply, and what are the correctness risks of each? Give one example use case where semantic caching could return a subtly wrong answer.

5. **A/B determinism**: In an A/B test between two models, why must the assignment be deterministic per user rather than random per request? How do you achieve this with Python's `hashlib`? What breaks if you use `random.random()` instead?

---

## Coding challenge

### Spec

Implement a minimal gateway endpoint in FastAPI. The implementation must be self-contained in a single file `gateway_endpoint.py`.

**Endpoint**: `POST /v1/chat`

**Request body**:
```json
{
  "model_tier": "fast",
  "messages": [{"role": "user", "content": "Hello"}],
  "tenant_id": "acme"
}
```

`model_tier` must be one of `"fast"`, `"smart"`, `"best"`.

**Routing**:

| `model_tier` | Model |
|---|---|
| `"fast"` | `claude-haiku-5-5` |
| `"smart"` | `claude-sonnet-5-5` |
| `"best"` | `claude-opus-5-5` |

**Rate limiting**: max 10 requests per minute per `tenant_id`. Return HTTP 429 with `{"detail": "rate_limit_exceeded"}` if exceeded.

**Response** (HTTP 200):
```json
{
  "content": "<model response string>",
  "model": "claude-haiku-5-5",
  "cost_usd": 0.000042
}
```

**Fallback**: if a `"smart"` request fails (provider raises any exception), retry automatically with `"fast"` before returning an error.

**Tests**: write pytest tests in `test_gateway_endpoint.py`. All provider calls must be mocked — no real API calls in tests.

### Acceptance criteria

- [ ] `POST /v1/chat` with valid body returns 200 and correct model in response
- [ ] Sending 11 requests with same `tenant_id` within one minute: first 10 return 200, 11th returns 429
- [ ] `model_tier="smart"` with mocked provider failure falls back to `"fast"` and returns 200
- [ ] Invalid `model_tier` (e.g. `"ultra"`) returns HTTP 422
- [ ] All tests pass with `pytest` and zero real HTTP calls to any LLM provider

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation — endpoint works end-to-end, rate limiting, fallback | 40% |
| Conceptual understanding — interview answers precise and concrete | 25% |
| Debugging & problem-solving — code handles edge cases, errors return correct status codes | 20% |
| Code quality & testing — type hints throughout, tests mock providers, no hardcoded secrets | 15% |

---

Answer key: ../solutions/week-09/assessment.md
