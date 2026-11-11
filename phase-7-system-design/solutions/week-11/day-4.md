# Week 11 Day 4 Solutions — GitHub & Portfolio Polish

## Exercise 1: GitHub Profile README for an AI Engineer

**Prompt:** Write a GitHub profile README (approximately 150 words) for an AI engineer completing this course.

**Model answer:**

---

```markdown
## AI Engineer

Building production-grade AI systems: LLM agents, RAG pipelines, and distributed inference infrastructure.

**Current focus:**
- Multi-agent orchestration with LangGraph + MCP
- Production RAG systems with retrieval evaluation
- LLMOps: observability, cost control, eval frameworks

**Projects:**
- [Multi-Agent Operations Platform](link) — LangGraph orchestration, MCP gateway, concurrent agent state management
- [Advanced RAG System](link) — Async ingestion, Qdrant vector store, cross-encoder re-ranking, MRR@5 0.71
- [Production AI Platform](link) — Kubernetes deployment, Prometheus/Grafana observability, distributed inference
- [Python Backend Platform](link) — FastAPI, async SQL, 94% test coverage

**Stack:** Python · LangGraph · FastAPI · Qdrant · Docker · Kubernetes · OpenTelemetry

Open to AI Engineer, MLOps, and Forward-Deployed Engineer roles.

[LinkedIn](link) | [Portfolio](link)
```

---

**What makes this README work:**
- Opens with identity, not backstory
- "Current focus" signals active learning, not just past experience
- Projects listed in reverse impact order (most complex first)
- Each project has one concrete metric or differentiator
- Stack line is scannable
- Closing line signals availability without being desperate

---

## Exercise 2: Pinned-Repo Order for 4 Portfolio Projects

**Prompt:** Rank the 4 portfolio projects in pinned-repo order (left to right, top to bottom) and justify the ordering.

**Model answer:**

**Recommended order:**

1. **Multi-Agent Operations Platform** (pin 1 — top left)
2. **Advanced RAG System** (pin 2 — top right)
3. **Production AI Platform** (pin 3 — bottom left)
4. **Python Backend Platform** (pin 4 — bottom right)

**Justification:**

**Position 1 — Multi-Agent Operations Platform:**
Most differentiated project in the market. Multi-agent orchestration + MCP + security is a rare combination that most candidates don't have. Leads with what's hardest to fake. Also the most interview-rich project — dozens of interesting technical questions can come from it.

**Position 2 — Advanced RAG System:**
RAG is the most common AI engineering interview topic in 2024-2026. Having a production RAG system with actual eval metrics (MRR@5, recall) immediately signals you're above the tutorial-follower tier. Goes second because it's familiar to interviewers — easier to evaluate quickly.

**Position 3 — Production AI Platform:**
Important for MLOps and platform roles. Kubernetes, observability, CI/CD — signals production readiness. Goes third because it's less differentiating than 1 and 2 (many candidates have Docker/k8s projects), but it's the one that seals the "can they actually ship" question.

**Position 4 — Python Backend Platform:**
Table-stakes for any engineering role. Strong project (async, tested, production patterns) but least AI-specific. Goes last among pinned repos — it supports the story, doesn't lead it.

**General rule:** Pin order = interview conversation priority order. Lead with what generates the most interesting technical discussion and is hardest for competitors to replicate.

---

## Exercise 3: Rewrite 3 Weak Commit Messages

**Prompt:** Rewrite 3 weak commit messages as good conventional commits.

**Model answer:**

**Weak commit 1:**
> `fixed stuff`

**Strong commit 1:**
> `fix(rag): correct chunk boundary overlap calculation causing duplicate tokens`
>
> Overlap was measured in characters instead of tokens; high-overlap chunks were consuming 15% more embedding budget. Switched to token-aware splitting via tiktoken.

---

**Weak commit 2:**
> `added new feature`

**Strong commit 2:**
> `feat(agents): add optimistic locking to shared agent state to prevent concurrent write races`
>
> Two agents modifying the same workflow state produced silent data corruption under concurrent execution. State reads now return a version token; writes validate the token before committing, failing fast on stale reads and triggering agent re-planning.

---

**Weak commit 3:**
> `update code`

**Strong commit 3:**
> `refactor(api): extract tool gateway from agent runner into standalone MCP server`
>
> Agent runner was directly invoking tools, making credential management and audit logging scattered across the codebase. All tool calls now route through MCP server; adds ~20ms latency per tool call in exchange for single audit surface.

---

**Conventional commit format:**
```
type(scope): imperative-mood summary under 72 chars

Body: why, not what. What is visible in the diff.
Mention the tradeoff or the bug that was fixed.
```

**Types used most in AI engineering projects:**
- `feat` — new capability
- `fix` — bug correction
- `refactor` — restructuring without behavior change
- `perf` — performance improvement
- `test` — adding or fixing tests
- `chore` — dependency updates, config changes

---

## Exercise 4: "Results/Metrics" Section for the Production AI Platform README

**Prompt:** Write the "Results/Metrics" section for the Production AI Platform README.

**Model answer:**

---

```markdown
## Results & Metrics

### Performance

| Metric | Value | Notes |
|---|---|---|
| Inference latency (p50) | 340ms | End-to-end: API → model → response |
| Inference latency (p95) | 720ms | Under 100 QPS sustained load |
| Inference latency (p99) | 1,100ms | Acceptable for async workloads |
| Throughput (sustained) | 100 QPS | 3-replica k8s deployment |
| Throughput (burst) | 280 QPS | With HPA scaling, ~45s cold-start |

### Cost

| Configuration | Cost per 1,000 requests | Notes |
|---|---|---|
| Baseline (no optimization) | $4.20 | claude-sonnet-5-5, no caching |
| With prompt caching | $2.10 | ~50% cache hit rate on repeated system prompts |
| With batching (async) | $1.80 | Batch size 8, +120ms latency |
| With model routing | $1.20 | Low-complexity tasks routed to smaller model |

### Reliability

- **Uptime:** 99.6% over 30-day test period
- **Error rate:** 0.3% (timeouts + model API errors)
- **Recovery time:** < 90s for pod restart; < 4 min for node failure

### Observability Coverage

- Distributed traces: 100% of requests instrumented (OpenTelemetry → Tempo)
- LLM-specific metrics: token usage, cost per request, prompt/completion ratio tracked in Prometheus
- Alerts configured for: p95 > 1s, error rate > 1%, cost/hour > threshold
- Grafana dashboards: latency heatmap, cost over time, model usage breakdown
```

---

**Why this format works:**
- Tables are scannable for technical readers
- Separates performance/cost/reliability — three different audiences care about different sections
- Includes methodology notes (what load, what configuration) so numbers are reproducible
- Observability section is often skipped but strongly differentiates production-serious projects
