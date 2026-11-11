# Week 11 Day 2 Solutions — Turning Capstones into Interview Narratives

## Exercise 1: Decision Log Entry for RAG System Chunking Strategy

**Prompt:** Write a complete decision log entry for the chunking strategy decision in the Advanced RAG system.

**Model answer:**

---

**Decision:** Chunking strategy for document ingestion pipeline

**Date:** [Month, Year of implementation]

**Context:** Building retrieval pipeline for a mixed-document corpus: technical PDFs, markdown docs, HTML pages. Documents range from 500 to 80,000 tokens. Needed chunk granularity that balances retrieval precision against context sufficiency for generation.

**Options considered:**

| Option | Latency | Precision | Implementation complexity |
|---|---|---|---|
| Fixed-size (512 tokens, 50-token overlap) | Lowest | Low — splits mid-sentence | Trivial |
| Sentence-boundary chunking | Low | Medium — semantically clean | Low |
| Recursive character splitting (LangChain default) | Low | Medium | Low |
| Semantic chunking (embed + cosine threshold) | High (2-3x) | High | High |
| Proposition-based (NLI extraction) | Very high | Highest | Very high |

**Decision:** Recursive character splitting with 800-token max and 150-token overlap, with a sentence-boundary cleanup pass.

**Rationale:** At the evaluation stage, retrieval precision for fixed-size chunking on technical docs was ~58% on our internal test set. Semantic chunking improved that to ~73% but tripled ingestion time. Recursive + sentence cleanup hit 67% precision at roughly the same latency as fixed-size. Given that we could iterate on retrieval ranking independently, 67% retrieval precision was the acceptable floor and the speed difference mattered for batch re-ingestion cycles.

**What we gave up:** The remaining precision gap vs. semantic chunking (~6 points) means some long technical passages get split at boundaries that hurt context. Partially mitigated by parent-document retrieval: we store both chunk and its parent section, and return parent context when a chunk is the top hit.

**Outcome:** Retrieval MRR@5 of 0.71 on held-out eval set. Re-ingestion of 10,000 documents in under 4 hours on a single machine.

**Would revisit if:** Document corpus shifts toward narrative text (not technical) where semantic boundaries matter more, or if ingestion is moved to a batch pipeline where throughput is less constrained.

---

## Exercise 2: "What Would You Do Differently" for the Multi-Agent Platform

**Prompt:** Answer "what would you do differently?" for the multi-agent platform without undermining yourself.

**Model answer:**

The framing that protects you: start with what you shipped and why the decision was defensible at the time, then surface what you learned that changed the picture.

**Delivered answer:**

"The platform works — 50 concurrent workflows, p95 under 800ms, no credential leaks. The decisions that produced that outcome were sound given what we knew when we made them.

Two things I'd change with what I know now:

First, distributed tracing from the start. We added OpenTelemetry after the core was built, and retrofitting it cost two weeks and introduced three refactors we hadn't planned. That's not a technical failure — we didn't know exactly what observability we'd need until we saw the system under real load. But the cost was high enough that I'd treat tracing as a first-class architectural concern upfront, not a later addition.

Second, I'd formalize the agent communication contract earlier. We used freeform dict payloads between agents initially, then migrated to Pydantic models when bugs from type mismatches piled up. The migration was painful. The lesson is that typed state isn't premature optimization for agent systems — it's the foundation. I'd start with Pydantic models at every boundary from day one."

**Why this works:** Both changes are about implementation sequence, not fundamental design errors. Neither implies the shipped system is broken. Both demonstrate genuine learning and specific technical judgment.

---

## Exercise 3: Tradeoff Triangle Analysis for the Production AI Platform

**Prompt:** Build a latency-cost-quality tradeoff triangle analysis for the production AI platform.

**Model answer:**

The latency-cost-quality triangle: optimizing any two degrades the third. Document where your system sits and what drove the position.

**Production AI Platform — Tradeoff Analysis**

**Axis definitions:**
- Latency: p95 response time for a complete inference request
- Cost: per-request compute cost (GPU + infra)
- Quality: task accuracy on held-out eval set

**Design decisions and their position:**

| Decision | Latency impact | Cost impact | Quality impact | Net position |
|---|---|---|---|---|
| Model selection: Sonnet-class vs. Opus-class | Sonnet faster (-40ms) | Sonnet cheaper (-60%) | Opus +4 points on hard tasks | Chose Sonnet: cost-latency wins for high-volume endpoints |
| Inference batching (batch=8) | +120ms added latency | -45% GPU cost | Neutral | Cost wins; latency acceptable at async use cases |
| KV cache warmup | -200ms on repeat queries | +3% memory cost | Neutral | Latency wins; memory cost trivial |
| Quantization (8-bit) | -15ms | -35% VRAM | -1.5 points accuracy | Acceptable for low-stakes tasks; disabled for high-stakes |
| Speculative decoding | -30ms | +8% compute | Neutral | Free latency win at acceptable cost |
| Response streaming | Perceived latency -60% | Neutral | Neutral | Always on; subjective experience improvement |

**Where the system sits:**
High-volume API endpoints: optimized for cost + perceived latency, quality floor enforced by routing (low-stakes tasks use smaller/cheaper models; high-stakes tasks always use full model).

**The explicit trade-off we accepted:**
Batching adds real latency (120ms) to reduce cost by 45%. This was the right call for async workflows. For synchronous user-facing requests, batching is disabled — cost is higher, latency wins.

**The constraint that shaped everything:**
GPU budget ($X/month). Every decision traces back to staying within that budget while hitting SLA targets. Quality was the constrained variable: "good enough" was defined as matching human baseline on the eval set at 80%+ accuracy.

---

## Exercise 4: Scaling Story for One Capstone

**Prompt:** Write the scaling story for the Advanced RAG system — from prototype to production numbers.

**Model answer:**

**Prototype (Week 1 of RAG project):**
- Single Python script, LangChain + FAISS in-memory vector store
- 500 documents loaded from local disk
- One embedding model call per query, synchronous
- Retrieval: ~200ms p50, ~800ms p95
- No eval, no monitoring
- Zero cost (local Ollama for generation)
- Ran on a MacBook

**Alpha (working system, demo-ready):**
- Switched to Qdrant (persistent, dockerized)
- 5,000 documents, structured ingestion pipeline
- Added basic eval loop: precision@5, MRR@5 on 200 test queries
- Retrieval: ~80ms p50 with Qdrant HNSW index
- Still single-threaded ingestion
- Cost: ~$0.02/query (OpenAI embeddings + claude-sonnet-5-5 generation)
- Ran on a single EC2 t3.medium

**Beta (real usage, real load):**
- 50,000 documents, multi-format (PDF, HTML, Markdown)
- Async ingestion pipeline with task queue (Celery + Redis), 4 workers
- Re-ranker added (cross-encoder): +8 points MRR@5
- Added observability: query latency, retrieval hit rate, hallucination flags
- Retrieval: ~95ms p50 (re-ranker adds ~30ms), ~220ms p95
- API: FastAPI with connection pooling
- Cost: $0.008/query (batched embeddings, caching for repeated queries)
- Ran on 2x EC2 c5.xlarge + 1x r5.large for Qdrant

**Production target:**
- 500,000 documents
- 100 QPS sustained, 300 QPS burst
- p95 < 500ms end-to-end (retrieval + generation)
- Cost target: < $0.005/query
- Architecture changes needed: Qdrant cluster (3 nodes), embedding cache (Redis), async generation streaming, horizontal API scaling behind load balancer
- Cost lever: switch to cached embeddings for repeated queries (est. 40% of production traffic is re-queries)

**The story in one sentence:** Started with a script on a laptop retrieving from 500 docs in 200ms; ended with an async pipeline serving 100 QPS from 500K docs at 95ms retrieval latency and $0.008/query, with clear path to 10x scale.
