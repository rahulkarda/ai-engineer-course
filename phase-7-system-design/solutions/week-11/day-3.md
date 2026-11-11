# Week 11 Day 3 Solutions — Resume Engineering for AI/ML Roles

## Exercise 1: Transform Weak Resume Bullets — Python Backend Platform

**Prompt:** Transform 2 weak bullets from the Python backend platform into strong impact bullets.

**Model answer:**

**Weak bullet 1:**
> "Worked on FastAPI backend and wrote tests for the application."

**Strong bullet 1:**
> "Engineered async FastAPI service handling 2,000 req/s with SQLAlchemy connection pooling; achieved 94% test coverage via pytest with full mock isolation of external dependencies."

**Transformation analysis:**
- Added quantified load metric (2,000 req/s)
- Named specific technical choices (SQLAlchemy connection pooling)
- Replaced "wrote tests" with the actual metric (94% coverage) and the technique (mock isolation)
- Replaced passive "worked on" with active verb "engineered"

---

**Weak bullet 2:**
> "Implemented authentication and made the API more secure."

**Strong bullet 2:**
> "Implemented JWT-based auth with role-based access control (RBAC) and rate limiting; eliminated credential exposure by enforcing HTTPS-only and rotating secrets via environment config — zero auth-related CVEs in security audit."

**Transformation analysis:**
- Named the specific mechanisms (JWT, RBAC, rate limiting)
- Added a concrete security outcome (zero CVEs in audit)
- Replaced vague "more secure" with specific controls
- Included the environmental hygiene detail (env config, no hardcoded secrets) which signals real production awareness

**General pattern for weak → strong bullets:**
```
WEAK:  [vague verb] + [technology name] + [vague outcome]
STRONG: [specific verb] + [mechanism/approach] + [quantified outcome or verifiable quality signal]
```

---

## Exercise 2: Projects Section Entry for the Advanced RAG System

**Prompt:** Write the "Projects" section entry for the Advanced RAG system.

**Model answer:**

---

**Advanced RAG System** | Python, LangChain, Qdrant, FastAPI, OpenAI, claude-sonnet-5-5
*[Link to GitHub repo]*

Retrieval-augmented generation pipeline for mixed-format document corpora (PDF, HTML, Markdown). Built end-to-end: async ingestion, chunking with sentence-boundary cleanup, embedding with OpenAI text-embedding-3-large, vector storage in Qdrant with HNSW indexing, and cross-encoder re-ranking.

- Achieved MRR@5 of 0.71 on 200-query held-out eval set; re-ranker added +8 points over baseline retrieval
- Scaled from 500 to 50,000 documents with async Celery/Redis ingestion pipeline; full re-ingestion in < 4 hours
- Reduced per-query cost from $0.02 to $0.008 via batched embeddings and Redis query cache
- Implemented hallucination detection layer using NLI-based fact-checking against retrieved passages

---

**Formatting notes:**
- Tech stack on the header line — recruiters scan this first
- GitHub link always present; no link = doesn't exist in their mental model
- Bullets in order: evaluation metric first (proves it works), then scale, then cost/efficiency, then interesting tech
- No fluff words ("innovative," "robust," "state-of-the-art")

---

## Exercise 3: Skills Section for AI/ML Engineer Role

**Prompt:** Build the skills section for an AI/ML Engineer role from the course's tech stack.

**Model answer:**

**Skills section structure for AI/ML Engineer:**

```
Languages:        Python (primary), SQL, Bash
ML / AI:          LangChain, LangGraph, OpenAI API, Anthropic API, Hugging Face Transformers,
                  RAG pipeline design, agent architecture, MCP, A2A protocols
Vector / Data:    Qdrant, FAISS, PostgreSQL, Redis, SQLAlchemy, Pydantic
Infrastructure:   Docker, Kubernetes, FastAPI, Celery, GitHub Actions, Prometheus, Grafana
Observability:    OpenTelemetry, distributed tracing, LLM evaluation (RAGAS, custom evals)
Testing:          pytest, async testing, mock/patch patterns, 90%+ coverage workflows
Practices:        LLMOps, CI/CD, cost optimization, security (RBAC, secret management)
```

**Formatting rules:**
- Group by functional area, not alphabetically — signals you understand how the skills relate
- List the most in-demand skills first within each group
- No proficiency levels ("proficient in," "familiar with") — they're subjective and waste space; list only skills you can demo
- Remove skills you can't actually discuss in an interview — a gap in a skills list is less damaging than lying in the interview

**Ordering rationale for AI/ML Engineer target roles:**
Languages → ML/AI (core competency) → Data/Vector (differentiator for RAG roles) → Infra (shows production-readiness) → Observability (rare, valuable signal) → Testing → Practices

---

## Exercise 4: 3-Sentence Professional Summary for AI/ML Engineer

**Prompt:** Write a 3-sentence professional summary targeting an AI/ML Engineer role.

**Model answer:**

**Option A (targets AI Platform / LLMOps):**
"AI engineer with hands-on production experience building LLM-powered systems — RAG pipelines, multi-agent orchestration platforms, and distributed inference infrastructure. Focused on closing the gap between research-quality models and production-grade systems: evaluation frameworks, observability, cost control, and security. Seeking roles where the work is deploying AI that behaves reliably at scale, not just prototyping what's possible."

**Option B (targets FDE / customer-facing AI engineering):**
"AI engineer with production experience across the full stack: LangGraph agent orchestration, RAG system design, FastAPI backends, and k8s deployments. Comfortable translating vague customer requirements into scoped technical architectures and delivering working systems fast. Looking for roles that sit at the intersection of customer problems and deep AI engineering."

**Option C (targets Applied ML / AI Research Engineering):**
"AI engineer with strong foundations in LLM evaluation, retrieval system design, and agent architecture. Built and benchmarked production RAG systems and multi-agent platforms with measurable quality metrics, not just vibes-based evaluation. Interested in roles where rigor about what the system actually does matters as much as what it could theoretically do."

**3-sentence summary formula:**
- Sentence 1: Who you are + what you've built (specific, not generic)
- Sentence 2: What makes you different from other ML engineers (the angle)
- Sentence 3: What you want — written as a signal about the kind of work you'll do well, not as a ask

**What to avoid:**
- "Passionate about AI" — everyone says this
- "Proven track record" — unverifiable, filler
- Mentioning the course directly — integrate what you built, not where you built it
