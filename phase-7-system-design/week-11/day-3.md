# Phase 7 · Week 11 · Day 3 — Resume Engineering for AI/ML Roles

> Objective: Transform portfolio project work into ATS-passing, recruiter-readable, technically credible resume content.  ·  Time: ~1 hr  ·  Prereqs: [[day-1]], [[day-2]]

## Concept

A resume for an AI engineer role passes through three filters before it reaches a technical interviewer: an ATS (applicant tracking system), a recruiter doing a 10-second scan, and a hiring manager looking for depth. Most resumes fail at filter 2. The ones that reach filter 3 but fail there usually have the opposite problem: they're technically dense but lack legible impact.

Your job is to survive all three.

## The ATS and Recruiter Filter

**ATS systems** keyword-match your resume against the job description. They are not intelligent. They count occurrences of exact strings. The terms that matter most are:

- Specific technologies named in the job description (e.g., "LangGraph," "Kubernetes," "OpenTelemetry")
- Role-specific phrases (e.g., "retrieval-augmented generation," "multi-agent systems," "LLM fine-tuning")
- Seniority signals (e.g., "designed," "led," "architected" vs. "helped with," "assisted")

**Recruiters** spend 7–10 seconds on the first pass. They scan in an F-pattern: top of page, left margin, first few words of each bullet. What they look for: job title match, recognizable company/project names, and one or two numbers that suggest scale.

Practical implication: the first word of every bullet must be a strong action verb. The first number in every bullet must be real.

## The Impact Bullet Formula

Structure every bullet as:

> **[Action verb] + [technology/method] + [metric] + [business outcome]**

The metric and business outcome are the hardest parts to write. Most engineers skip them. That is why most engineers' resumes are weaker than their actual work.

**Action verbs for AI/ML work** (not exhaustive):
- Architected, designed, implemented, built — for systems work
- Evaluated, benchmarked, reduced, improved — for performance work
- Deployed, automated, containerized, integrated — for ops work
- Trained, fine-tuned, distilled, calibrated — for model work

**Weak bullet** (no metric, no outcome):
> "Implemented a retrieval-augmented generation system using LangChain and Qdrant."

**Strong bullet** (metric + outcome):
> "Architected hybrid RAG pipeline (BM25 + dense retrieval, cross-encoder re-ranking) achieving 0.84 RAGAS answer relevancy, reducing hallucination rate 40% vs. dense-only baseline across 500-document technical corpus."

The second version takes longer to read but survives filter 3. The first fails filter 2 because there is nothing to remember.

## Keyword Alignment: Mapping JD Terms to Portfolio Vocabulary

Before writing bullets, do this exercise for every application:

1. Paste the job description into a text file.
2. Extract all technical terms and role-level phrases.
3. Map each term to the closest thing in your portfolio — exact match preferred, close match acceptable, gap noted.
4. Adjust your bullets to use the JD's exact vocabulary where you legitimately did that work.

Example mapping for a typical "Senior AI Engineer" JD:

| JD term | Your portfolio equivalent |
|---|---|
| "LLM orchestration" | LangGraph in Multi-Agent Platform |
| "RAG pipelines" | Advanced RAG System |
| "production ML systems" | Production AI Platform |
| "API design" | Python Backend Platform |
| "model evaluation" | RAGAS eval pipeline in RAG System |
| "containerization" | Docker/k8s in Production AI Platform |

Do not fabricate. Map what you actually did. Gaps are fine — no candidate matches 100% of a JD.

## Translating Portfolio Projects into Resume Bullets

### Python Backend Platform

Weak: "Built a FastAPI backend with async endpoints and a PostgreSQL database."

Strong: "Designed async FastAPI service with Pydantic v2 validation and Alembic migrations, supporting concurrent LLM requests; achieved <50ms p95 latency under 200 concurrent connections in load testing."

### Advanced RAG System

Weak: "Created a RAG system using vector embeddings and a language model."

Strong: "Built production RAG system with hybrid retrieval (BM25 + sentence-transformer embeddings, Qdrant) and cross-encoder re-ranking; RAGAS answer relevancy 0.84, faithfulness 0.91 on 500-document corpus; eval pipeline integrated into CI with regression gating."

### Multi-Agent Operations Platform

Weak: "Developed a multi-agent system using LangGraph and MCP."

Strong: "Architected multi-agent orchestration platform using LangGraph (checkpointed state graphs) and MCP for dynamic tool discovery; supervisor-specialist separation reduced agent failure blast radius; system recovered from 94% of tool call failures without human intervention."

### Production AI Platform

Weak: "Deployed an AI application on Kubernetes with Docker containers."

Strong: "Deployed distributed AI inference platform on Kubernetes with blue-green rollout, OpenTelemetry distributed tracing, and per-call cost attribution; reduced mean recovery time from 8 min to <90s via automated rollback; tracked LLM spend per feature to within 2% accuracy."

## The Skills Section

**Include:**
- Languages: Python (primary), SQL, Bash
- Frameworks: FastAPI, LangGraph, Pydantic, pytest
- AI/ML: RAG, LLM orchestration, multi-agent systems, vector search, embeddings, RAGAS evaluation
- Infrastructure: Docker, Kubernetes, GitHub Actions, OpenTelemetry
- Databases: PostgreSQL, Qdrant (or your vector DB), Redis

**Omit:**
- Technologies you used once in a tutorial and cannot defend in an interview
- Generic terms that add no signal: "agile," "team player," "fast learner"
- Tools that are table stakes and go without saying: Git, VS Code, Jupyter

**Order:** Sort by what the target JD weights most heavily, not alphabetically. If the JD emphasizes MLOps, put infrastructure tools higher. If it emphasizes LLM applications, put AI/ML frameworks first.

## The Summary / Objective Statement

For AI/ML roles, a summary works better than an objective. Objective statements describe what you want; summaries describe what you offer.

Formula:
> "[Seniority level] [role type] with [N] years [domain]. Specializes in [2–3 specific technical areas]. Built [brief project reference with one number]. Looking for [one-line target role or domain]."

Example (matches this course's portfolio):
> "AI engineer with hands-on experience building production LLM systems. Specializes in multi-agent orchestration, retrieval-augmented generation, and MLOps on Kubernetes. Portfolio includes a LangGraph-based multi-agent platform with 94% automated failure recovery and a RAG system scoring 0.84 RAGAS answer relevancy. Targeting senior AI engineer roles in applied LLM products."

150 words max. Every sentence should either be measurable or specific.

## Worked Example — 3 Before/After Transformations (Production AI Platform)

**Bullet 1 — Observability:**
- Before: "Added logging and monitoring to the AI application."
- After: "Instrumented distributed AI platform with OpenTelemetry traces and Prometheus metrics; P95 latency visibility across all model inference calls, enabling identification of a 3x latency regression in the re-ranking stage within 5 minutes of deployment."

**Bullet 2 — Deployment:**
- Before: "Deployed the app using Docker and Kubernetes."
- After: "Containerized multi-service AI platform with Docker; deployed to Kubernetes with blue-green rollout and automated health-check-based rollback; reduced failed deployment impact from full service disruption to <30s for 99.5% of rollouts."

**Bullet 3 — Cost:**
- Before: "Tracked costs for LLM API calls."
- After: "Implemented per-feature LLM cost attribution using OpenTelemetry baggage propagation; surfaced that a single summarization feature consumed 38% of total API spend, enabling a targeted caching change that cut monthly cost by $240."

Note the pattern: the "after" versions all answer "so what?" with a number or a business consequence.

## Exercises

1. Write 2 impact bullets for the Multi-Agent Operations Platform using the formula: action verb + technology + metric + business outcome. Both must include a real or estimated metric. No placeholders.

2. Take a job description for a "Senior AI Engineer" or "ML Engineer" role from any major job board. Perform the keyword alignment exercise: list 8 JD terms, map each to your portfolio, note gaps. Do not apply yet — just map.

3. Rewrite the following weak bullet using the formula: "Worked on a machine learning pipeline that processed user data." Target role: AI platform engineer. You may invent realistic specifics from any of the 4 portfolio projects.

4. Write a 150-word-or-fewer summary statement for your own resume targeting an AI engineer role. It must reference at least one metric from a portfolio project.

(Solutions: ../solutions/week-11/day-3.md)

## Self-check

- [ ] I can explain the three-filter model (ATS, recruiter, hiring manager) and what each needs
- [ ] I can write an impact bullet using the four-part formula for any portfolio project
- [ ] I have completed the keyword alignment exercise for at least one target JD
- [ ] I have written or drafted a summary statement with at least one measurable claim

## Supplementary

- [Tech Interview Handbook — Resume Guide](https://www.techinterviewhandbook.org/resume/) — practitioner-written, regularly updated; covers ATS optimization and bullet structure with examples
- [Jobscan — ATS Optimization](https://www.jobscan.co/) — tool for comparing your resume text against a JD's keyword frequency; useful for calibrating alignment
- [The XYZ Formula for Resume Bullets — Laszlo Bock (Google)](https://www.linkedin.com/pulse/20140929001534-24454816-my-personal-formula-for-a-better-resume/) — the original impact bullet framework; this lesson's formula is a direct extension for technical roles
