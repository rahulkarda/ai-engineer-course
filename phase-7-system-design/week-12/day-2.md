# Phase 7 · Week 12 · Day 2 — HLD + AI-System-Design Mock Loop

> Objective: Run a full timed HLD and AI-system-design mock loop, score against rubrics, identify structural gaps.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]]

## Concept

HLD and AI-system-design rounds test your ability to reason at scale. They share a common structure but diverge in depth:

| Dimension | HLD | AI-System-Design |
|---|---|---|
| Scope | Service architecture, APIs, storage, scale | All of HLD + model serving, data pipelines, eval, feedback loops |
| Time box | 45 min | 50 min |
| Key artifacts | Service diagram, API contract, capacity math | Same + inference topology, embedding pipeline, guardrails |
| Common failure | Jumping to tech before requirements | Ignoring offline vs. online serving split |

## How it works internally

**The 7-step HLD framework (apply to every prompt)**

```
1. Requirements (5 min)   — functional, non-functional, scale numbers
2. API contract (5 min)   — endpoint/event signatures
3. High-level diagram (5 min) — boxes + arrows, named services
4. Deep dive (15 min)     — 2-3 components the interviewer cares about
5. Data model (5 min)     — schema or document shape
6. Scale & bottlenecks (5 min) — back-of-envelope, hotspot, mitigation
7. Trade-offs (5 min)     — what you'd change with more time/money
```

**AI-specific addenda (layer on top of step 4)**

- Model serving topology: batch vs. online vs. streaming inference
- Embedding pipeline: when to re-index, freshness SLA
- Prompt management & versioning
- Eval harness & feedback loop
- Guardrails placement (input + output)
- Cost model: tokens × price × QPS

## Worked example

### HLD mock prompt

> "Design a URL shortener that handles 1 billion URLs and 10 billion redirects per month."

**Step-by-step model answer**

**Requirements**
- Functional: create short URL, redirect, optional custom alias, optional TTL
- Non-functional: redirect p99 < 10 ms, creation p99 < 200 ms, 99.99% uptime
- Scale: ~400 creates/s, ~4,000 redirects/s (10B/30d/86400 ≈ 3,858/s)

**API contract**
```
POST /v1/urls          body: {url, alias?, ttl_s?} → {short_code, short_url}
GET  /{short_code}     → 301 redirect
DELETE /v1/urls/{code} → 204
```

**High-level diagram**
```
Client
  │
CDN (cache hot redirects, TTL=60s)
  │
API Gateway
  ├─ Write Service ──► URL DB (Postgres, short_code PK)
  │                 └► Cache (Redis, short_code→long_url, TTL)
  └─ Read Service  ──► Cache → URL DB (cache miss)
```

**Data model**
```sql
CREATE TABLE urls (
    short_code  CHAR(8)     PRIMARY KEY,
    long_url    TEXT        NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    expires_at  TIMESTAMPTZ,
    owner_id    UUID
);
CREATE INDEX ON urls (expires_at) WHERE expires_at IS NOT NULL;
```

**Scale math**
- 8-char base-62 codes → 62^8 ≈ 218 trillion — no collision concern for 1B
- Read:write ratio 10:1 → cache hit rate 95%+ with LRU, 99% after CDN
- Redis: 1B × (8B code + 2KB avg URL) ≈ 2 TB — shard by first 2 chars

**Trade-offs**: consistent hash ring for Redis sharding vs. cluster-mode; chose cluster-mode for simplicity.

---

### AI-system-design mock prompt

> "Design a retrieval-augmented Q&A system for an enterprise document corpus (10 M docs, 500 QPS, p99 latency < 2 s, source citations required)."

**Model answer (condensed)**

```
[Ingestion pipeline]
Doc store → Chunker → Embedder (batch, GPU) → Vector DB (Qdrant, HNSW)
                                               │
[Online path]                                  │
User query → Query embedder (online, CPU) ─────┘
           → Retriever (top-k=20, MMR reranking)
           → Prompt builder (inject chunks + citations)
           → LLM (claude-sonnet-5-5, streaming)
           → Response with inline source refs
```

**Key AI-specific decisions**

| Decision | Choice | Reason |
|---|---|---|
| Chunk size | 512 tokens, 50-token overlap | Balances context richness vs. retrieval precision |
| Re-ranker | Cross-encoder (BGE-reranker-large) | MMR alone misses semantic nuance |
| Caching | Semantic cache on query embedding (cosine ≥ 0.97) | Cuts LLM calls ~30% for repeat queries |
| Guardrails | Input: PII redaction; output: faithfulness checker | Compliance + hallucination mitigation |
| Eval | RAGAS (faithfulness, answer relevance, context recall) | Automated offline + nightly regression |

**Cost model**: 500 QPS × 2K tokens avg × $3/M (Sonnet) = $3,600/day → add semantic caching.

## Exercises

1. **HLD drill**: Set a 45-min timer. Design a real-time leaderboard for a multiplayer game (10 M players, updates every 10 s, top-100 global + per-region). Produce: requirements, API, diagram, data model, scale math, trade-offs.

2. **AI-system-design drill**: Design an enterprise AI coding assistant (like GitHub Copilot for internal code). Address: model selection, context window strategy, fine-tuning vs. RAG, latency, security (code never leaves VPC). 50 min.

3. **Critique exercise**: The design below has three architectural flaws. Identify them:
   ```
   User → Monolith → Postgres (embeddings stored as JSONB array)
                   → External OpenAI API (no retry, no fallback)
                   → Redis (stores full prompt history, no TTL)
   ```

4. **Cost math**: An AI assistant serves 1,000 employees, each making 20 requests/day. Average request: 500 tokens in, 200 tokens out. `claude-sonnet-5-5` costs $3/M input, $15/M output. What is the monthly cost? At what QPS does a semantic cache paying off at $200/month become worthwhile?

(Solutions: ../solutions/week-12/day-2.md)

## Self-check
- [ ] I can state the 7-step HLD framework from memory
- [ ] I produce a capacity estimate before picking technology
- [ ] I layer AI-specific concerns (eval, feedback loop, cost) on top of HLD naturally
- [ ] I identify embedding pipeline freshness as a separate design concern

## Supplementary
- [System Design Primer](https://github.com/donnemartin/system-design-primer)
- [Designing ML Systems — Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
