# Week 06 Assessment — Answer Key

> Instructor/self-review use only. Do not read before attempting the assessment.

---

## Conceptual Questions — Model Answers

**Q1. What are the tradeoffs between HNSW and IVF-PQ for a 500M-document vector store?**

HNSW: graph-based, O(log n) query, high recall (>95%), memory-intensive (~100 bytes/vector = 50 GB for 500M at 1536 dims before PQ). No training required; supports incremental adds. Best for < 100M vectors or when recall is non-negotiable.

IVF-PQ: inverted file + product quantization. 10-50x memory reduction (PQ compresses 6 KB per vector to ~192 bytes). Requires training (k-means on representative sample, ~1–5% of corpus). Recall drops to 85-92% without re-ranking. Requires nprobe tuning (nprobe=64 typical for recall/latency balance). Best for 100M+ vectors where memory cost or recall < 95% is acceptable. At 500M docs: IVF-PQ with nprobe=64 and a cross-encoder re-ranker achieves 90%+ end-to-end recall at 3-5x lower cost.

Scoring: 2 pts — memory tradeoff, 2 pts — recall characteristics, 1 pt — when to use each.

---

**Q2. Explain hybrid search and how you tune the alpha parameter.**

Hybrid search combines dense retrieval (embedding similarity) with sparse retrieval (BM25/TF-IDF keyword matching). Score = alpha * dense_score + (1 - alpha) * sparse_score.

Alpha tuning: offline grid search over a labeled query set (use NDCG@10 as metric). Typical starting point: alpha=0.7 for general knowledge queries, 0.3 for exact-match/entity-heavy queries. Production approach: train a query classifier (keyword-heavy vs semantic) and route to different alpha values. Monitor retrieval quality metrics in production; retune when embedding model changes.

Scoring: 2 pts — formula/concept, 2 pts — tuning methodology, 1 pt — production monitoring.

---

**Q3. Describe the orchestration layer responsibilities in an agent platform.**

Core responsibilities:
1. State management — maintain typed agent state across tool calls; checkpoint to durable store (Postgres + Redis) for fault recovery
2. Graph execution — resolve next node, handle conditional edges, manage fan-out/fan-in parallelism
3. Tool dispatch — route tool calls to appropriate MCP server; enforce timeouts; handle tool errors
4. HITL integration — pause execution at checkpoints, serialize state, resume on approval
5. Observability — emit trace spans per node execution, correlate with run_id

Not in orchestration layer: tool implementation logic (in MCP servers), prompt construction (in prompt module), business rules (in tool services).

Scoring: 3 pts — covering 3+ responsibilities, 2 pts — correct separation of concerns.

---

**Q4. How does an LLM gateway differ from a standard API gateway?**

Standard API gateway: auth, rate limiting, load balancing, SSL termination, request routing. Stateless, byte-passthrough for payload.

LLM gateway adds: token counting (input/output) for accurate cost attribution; model fallback chains (retry with lower-tier model on overload/error); streaming response handling (SSE chunked forwarding); cost-aware routing (select model based on task complexity and budget); prompt injection detection; PII scrubbing; semantic caching (exact + embedding-similarity cache lookup); response logging with token metadata for chargeback.

Key difference: must parse and understand the LLM request/response payload, not just route bytes.

Scoring: 2 pts — standard gateway baseline, 3 pts — LLM-specific additions.

---

**Q5. Compare namespace-level vs cluster-level tenant isolation. When do you use each?**

Namespace isolation: single Qdrant/Weaviate cluster, each tenant gets a collection prefix or metadata filter. Low cost; shared compute. Risks: noisy neighbor on index rebuild, accidental cross-tenant query if filter omitted. Suitable for: SMB SaaS, < 10K tenants, non-regulated data.

Cluster isolation: dedicated vector store cluster per tenant (or per tenant group). Full resource isolation; no cross-tenant risk. Cost: ~10-20x higher. Complexity: fleet management. Suitable for: enterprise/regulated customers, GDPR data residency requirements, high-value tenants with SLA guarantees, tenants > 10M documents.

Hybrid approach: namespace for standard tier, dedicated cluster for enterprise/regulated tier.

Scoring: 2 pts — namespace tradeoffs, 2 pts — cluster tradeoffs, 1 pt — decision criteria.

---

**Q6. How would you design a PII redaction pipeline for a RAG ingestion system?**

Architecture: detection → classification → redaction/pseudonymization → audit.

Detection: run Presidio (or spaCy NER) on raw chunks before embedding. Detect: names, emails, phone numbers, SSNs, credit cards, addresses, IP addresses.

Handling options:
- Redaction: replace with `[REDACTED_NAME]` — safe but degrades retrieval quality if entity is query target
- Pseudonymization: replace with consistent fake entity (`John Smith` → `Person_7f3a`) — preserves retrieval quality, enables re-identification by authorized parties
- Exclusion: reject chunk entirely if PII density > threshold

Storage: store original in encrypted vault (for erasure requests), store redacted version in vector store. Maintain PII audit log (chunk_id, entity_type, timestamp) for GDPR right-to-erasure.

At query time: apply same redaction to user query before embedding; un-redact in response only if user has clearance.

Scoring: 2 pts — detection approach, 2 pts — redaction vs pseudonymization, 1 pt — GDPR erasure.

---

**Q7. Define a cost SLO for a multi-tenant AI platform. What are the key components?**

Components:
1. Budget cap: per-tenant monthly token budget (e.g., 10M tokens/month standard tier)
2. Rate limiting: per-minute token bucket (prevents single large job consuming month budget instantly)
3. Burst allowance: allow 2x rate for up to 5 minutes with auto-throttle after
4. Chargeback model: hard limit (block at 100% budget) vs soft limit (allow overage with surcharge billing)
5. Alert thresholds: alert at 50%, 80%, 95% of budget consumed
6. Attribution granularity: by tenant, user, application, model tier

Enforcement: token bucket in Redis (INCR + EXPIRE). Pre-flight check before each request: `if tenant_tokens_used + estimated_tokens > budget: raise BudgetExceededError`.

Scoring: 2 pts — budget + rate limiting, 2 pts — alerting/enforcement, 1 pt — chargeback model.

---

**Q8. What P50/P95/P99 latency targets would you set for RAG, agent, and direct LLM calls?**

Direct LLM call (chat, non-streaming):
- P50: 800ms, P95: 2s, P99: 5s
- Dominated by model inference time

RAG query (embed + retrieve + rerank + generate):
- P50: 1.5s, P95: 4s, P99: 8s
- Additional steps: embedding (~50ms), vector search (~20-100ms), reranking (~200ms)
- If semantic cache hit: P50: 50ms, P95: 200ms

Agent run (multi-step, multiple tool calls):
- P50: 10s, P95: 45s, P99: 120s
- Highly variable; set timeout SLOs not latency SLOs for agents
- Individual tool call budget: P95 < 5s per tool

Design consideration: surface latency SLOs to users (UI spinner vs async notification pattern based on SLO tier).

Scoring: 2 pts — correct ordering/rationale, 2 pts — realistic numbers, 1 pt — streaming vs non-streaming distinction.

---

**Q9. Describe online eval vs offline eval vs shadow scoring in a production AI system.**

Offline eval: run on a fixed held-out dataset before deployment. Metrics: RAGAS faithfulness/relevance/context recall, exact-match for classification tasks. Fast iteration; no production risk. Limitation: dataset drift — offline set may not reflect production query distribution.

Online eval: score live production responses using LLM-as-judge or embedding-similarity checks. Continuous signal; reflects real distribution. Cost: ~$0.001/response with haiku judge. Limitation: latency added to response path (run async/post-response), judge calibration required.

Shadow scoring: route a copy of production traffic to new model/system in parallel (without serving results to users). Compare shadow vs production quality metrics before cutover. Best for: model upgrades, prompt changes, retrieval strategy changes.

Combined strategy: offline for go/no-go gating, online for continuous drift detection, shadow for safe A/B model comparison.

Scoring: 2 pts — correct definitions, 2 pts — when to use each, 1 pt — combined strategy.

---

**Q10. What are the three most dangerous agent failure modes and how do you mitigate each?**

1. Tool call loop: agent calls the same tool repeatedly with same/similar arguments (often due to ambiguous stopping condition in system prompt). Detection: sliding window hash of (tool_name, args_hash) — if same hash appears 3+ times in N steps, trigger circuit breaker. Mitigation: loop breaker node, max_iterations hard limit, clearer tool descriptions with explicit stop conditions.

2. Context window overflow: agent accumulates tool results until context exceeds model limit. Silent truncation causes loss of task state and non-deterministic behavior. Detection: monitor token count per step; alert at 70% of context limit. Mitigation: summarize tool results before adding to context; use external state store for large artifacts; implement sliding window context management.

3. Hallucination cascade: agent fabricates a tool result (or misinterprets it), builds on that hallucination in subsequent steps, compounding error. Mitigation: validate tool results against schema before adding to agent state; use structured outputs for tool calls; cross-check critical facts with a verification tool; LLM-as-judge scoring on intermediate reasoning steps.

Scoring: 2 pts per failure mode (description + mitigation). Partial credit for identifying the mode without mitigation.

---

## Coding Challenge — Reference Answer Notes

### Architecture

```
LLMRouter
├── ModelConfig (Pydantic) — model_id, task_types, cost_in/cost_out per 1M tokens
├── RoutingTable — task_type → model priority list  
├── CostLedger — in-memory dict[model_id, CostRecord]; thread-safe with asyncio.Lock
├── ExponentialBackoffRetry — max_retries=3, base_delay=1.0, jitter=True, fallback chain
└── FastAPI app — POST /complete, GET /metrics
```

### Routing Logic

```python
ROUTING_TABLE = {
    "simple": ["claude-haiku-5-5", "claude-sonnet-5-5"],
    "complex": ["claude-sonnet-5-5", "claude-haiku-5-5"],
    "code": ["claude-opus-5-5", "claude-sonnet-5-5"],
}
```

### Cost Formula

```python
cost = (input_tokens / 1_000_000 * model.cost_in_per_1m) + \
       (output_tokens / 1_000_000 * model.cost_out_per_1m)
```

Pricing: haiku-5-5: $0.10/$0.50, sonnet-5-5: $2/$10, opus-5-5: $4/$20.

### Retry with Fallback

```python
async def complete_with_fallback(task_type: str, prompt: str) -> CompletionResult:
    models = ROUTING_TABLE[task_type]
    for i, model_id in enumerate(models):
        delay = 1.0
        for attempt in range(3):
            try:
                result = await call_model(model_id, prompt)
                ledger.record(model_id, result.usage)
                return result
            except RateLimitError:
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay *= 2
            except Exception:
                break  # try next model
    raise AllModelsFailedError()
```

### /metrics Endpoint

```python
@app.get("/metrics")
async def metrics() -> MetricsResponse:
    return MetricsResponse(
        per_model_cost=ledger.get_costs(),
        total_cost_usd=ledger.total_cost(),
        request_counts=ledger.get_counts(),
    )
```

### Test Requirements

Minimum passing tests:
- `test_routes_simple_to_haiku` — simple task → haiku-5-5 selected first
- `test_routes_complex_to_sonnet` — complex task → sonnet-5-5 selected first
- `test_routes_code_to_opus` — code task → opus-5-5 selected first
- `test_fallback_on_rate_limit` — mock first model raises RateLimitError → falls back to second
- `test_cost_accumulation` — two requests → cost ledger sums correctly
- `test_metrics_endpoint` — GET /metrics → returns expected structure
- `test_exponential_backoff_timing` — retry delays approximately double
- `test_all_models_fail_raises` — all models fail → AllModelsFailedError raised

### Common Mistakes

- Not implementing jitter in backoff (introduces thundering herd on shared rate limit)
- Not separating routing logic from HTTP layer (testability issue)
- Using global mutable state without asyncio.Lock (race condition on cost ledger)
- Swallowing exceptions instead of logging + re-raising
- Not testing the fallback path

---

## Rubric Application Guide

| Score | Description |
|---|---|
| 90-100 | Correct routing, cost tracking, retry+fallback, metrics endpoint; 6+ tests pass; type hints throughout |
| 80-89 | Routing + cost correct; retry works; missing jitter or metrics endpoint edge cases; 5+ tests |
| 75-79 | Core routing works; cost tracking has minor bugs; some tests; type hints mostly present |
| 65-74 | Routing works for happy path; no fallback or no cost tracking; few tests |
| < 65 | Missing major components or broken routing logic |

### Dimension Scoring

**Independent implementation (40%):** Did the learner implement the router logic themselves (not just copy a template)? Check: routing dict design, cost formula, retry backoff implementation.

**Conceptual understanding (25%):** Do the 10 conceptual answers show understanding of tradeoffs, not just definitions? Look for: "when to use X vs Y" reasoning, capacity math, failure mode mitigations.

**Debugging & problem-solving (20%):** Can the learner explain their fallback chain? Can they trace why a specific model was selected? Run their code against a failing scenario.

**Code quality & testing (15%):** Type hints on all functions/methods? Pydantic models for request/response? Logging not print? At least 5 meaningful tests?
