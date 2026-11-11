# Week 06 Assessment — AI-Specific System Design

## Conceptual (interview-style)

1. **RAG Platform Scaling**: You are running a RAG-based customer support system at 10,000 QPS with a 500ms p99 latency SLO. Your vector index currently holds 50M embeddings. Describe how you would scale the retrieval layer. What are the specific bottlenecks and how do you address each?

2. **Vector Store Tradeoffs**: Compare Pinecone (managed), Weaviate (self-hosted), and pgvector (PostgreSQL extension) across the following dimensions: query latency at 100M vectors, operational complexity, multi-tenancy support, and cost at scale. When would you choose each?

3. **Agent Orchestration**: You need to build an agentic pipeline where a planning agent decomposes a user task into subtasks dispatched to 3-5 specialist agents, then an aggregation agent synthesizes results. Walk through the state management design. What happens when one specialist agent fails mid-pipeline? How do you avoid context explosion?

4. **LLM Gateway Design**: Design an LLM gateway that sits in front of multiple model providers (Anthropic, Azure OpenAI, Bedrock). It must handle: request routing, prompt caching, rate-limit management, cost tracking per tenant, and PII redaction. What are the key components? Where does each concern live?

5. **Multi-Tenancy Isolation**: Your enterprise AI platform serves 200 tenants. Describe three distinct isolation models (from weak to strong) you could implement for: vector store namespacing, prompt injection defense, LLM call attribution, and data residency. What are the cost and complexity tradeoffs of each?

6. **PII Handling in LLM Pipelines**: A healthcare customer requires that no patient data ever leaves their VPC, but they want to use a frontier LLM for clinical note summarization. Describe at minimum two architectures that satisfy this requirement. What are the detection, redaction, and audit-trail components needed? What are the accuracy vs. privacy tradeoffs?

7. **Cost SLOs**: Your platform has a $0.05 per-query cost SLO at P95. Current average cost is $0.038 but you have a long tail of expensive queries. Describe how you would instrument, detect, and enforce this SLO. What circuit-breaker or fallback mechanisms would you put in place? How do you communicate cost overruns to tenants?

8. **Latency SLOs**: You have a 200ms p50 / 800ms p99 end-to-end latency SLO for a chat application backed by a 1M-context LLM. The model alone takes 150ms p50. Walk through each component of the request path and identify where you have budget to spend. What optimizations are available at each layer? Where do you accept unavoidable tradeoffs?

9. **Production Eval Patterns**: Your team ships a new prompt weekly. Describe a production evaluation system that: (a) catches regressions before deploy, (b) monitors quality drift post-deploy, (c) collects implicit user feedback signals, and (d) generates a weekly quality report. What does the data pipeline look like? How do you handle ground-truth labeling at scale?

10. **Failure Modes**: Enumerate five distinct failure modes specific to LLM-backed production systems (not generic distributed systems failures). For each: describe the failure, how it manifests to end users, how you detect it, and how you recover from it. At least one must involve model hallucination propagation through an agentic pipeline.

---

## Coding Challenge

### Design and Implement a Production LLM Request Router

**Specification:**

Build a FastAPI service that intelligently routes LLM requests across multiple Claude models based on task complexity, tracks costs in real-time, and implements resilient retry logic with model fallback.

**Requirements:**

1. **Request intake**: Accept a `POST /route` endpoint with a JSON body containing:
   - `prompt: str` — the user prompt
   - `task_type: Literal["simple", "complex", "code"]` — the declared task complexity
   - `max_tokens: int` — the desired output budget
   - `tenant_id: str` — the calling tenant

2. **Model routing**: Route based on `task_type`:
   - `"simple"` → `claude-haiku-5-5`
   - `"complex"` → `claude-sonnet-5-5`
   - `"code"` → `claude-opus-5-5`

3. **Cost tracking**: After every successful call, compute and accumulate per-model cost using token counts from the API response. Track `input_tokens`, `output_tokens`, and total cost per model in an in-memory store (a global dict is acceptable for this exercise).

4. **Exponential backoff retry with model fallback**: If the primary model call fails (rate limit, server error, timeout), retry with exponential backoff. After `max_retries` exhausted on the primary model, fall back to the next model in the priority order: `opus → sonnet → haiku`. If all models fail, raise a `RouterExhaustedError`.

5. **Metrics endpoint**: Expose `GET /metrics` returning a JSON response with:
   - Per-model: `total_requests`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `error_count`
   - Global: `total_requests`, `total_cost_usd`, `uptime_seconds`

**Acceptance Criteria:**

- [ ] `POST /route` returns the LLM response text and the model actually used
- [ ] Costs are computed from live token counts (not estimated); use the pricing from the `claude-api` skill reference ($0.10/$0.50 input/output per 1M tokens for Haiku 5.5, $2/$10 for Sonnet 5.5, $4/$20 for Opus 5.5)
- [ ] Exponential backoff uses jitter (full jitter or decorrelated jitter)
- [ ] Model fallback order is deterministic and logged at WARNING level
- [ ] `/metrics` endpoint is always available even when all LLM calls are failing
- [ ] All functions have type hints; Pydantic models used for all request/response schemas
- [ ] `logging` used throughout — no `print` statements
- [ ] At least the following pytest tests exist:
  - Unit test: routing logic maps each `task_type` to the correct model
  - Unit test: cost calculation is correct given known token counts
  - Unit test: fallback order is correct when mocking API failures
  - Integration test (mocked): full `POST /route` request succeeds and returns expected fields
  - Integration test (mocked): `/metrics` accumulates correctly across multiple requests

**What NOT to implement** (out of scope):
- Persistent storage (in-memory is fine)
- Authentication/authorization
- Async LLM calls (sync Anthropic SDK is fine)
- Streaming responses

---

## Rubric (pass ≥ 75%)

| Dimension | Weight |
|---|---|
| Independent implementation | 40% |
| Conceptual understanding | 25% |
| Debugging & problem-solving | 20% |
| Code quality & testing | 15% |

### Scoring detail

**Independent implementation (40%)**
- Full working implementation with no copy-paste from tutorials: 40%
- Working implementation with minor gaps (missing fallback or metrics): 25%
- Partial implementation (routing and cost only, no retry/fallback): 15%
- Scaffold only, no working logic: 0%

**Conceptual understanding (25%)**
- All 10 conceptual questions answered with specific, accurate technical depth: 25%
- 7–9 answers accurate with appropriate depth: 18%
- 5–6 answers accurate: 12%
- Fewer than 5 accurate answers: 0%

**Debugging & problem-solving (20%)**
- Can diagnose and fix a broken implementation under time pressure; explains root cause: 20%
- Identifies the bug but takes excessive time or hints: 12%
- Requires significant guidance to find the issue: 5%

**Code quality & testing (15%)**
- Type hints throughout, Pydantic models, logging, all 5+ required tests passing: 15%
- Minor gaps (1–2 missing type hints, 1 missing test): 10%
- Tests present but not passing, or significant quality issues: 5%
- No tests: 0%

---

Answer key: `../solutions/week-06/assessment.md`
