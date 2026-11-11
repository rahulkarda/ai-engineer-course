# Solutions — Week 10 Project: Story Bank, Design Doc & Behavioral Responses

## Purpose of this file

This file contains worked examples for all three project deliverables. Use them to calibrate your own work — compare structure, specificity, and depth. Your stories should come from your own career; the examples below are illustrative.

---

## Deliverable 1 — Story Bank (two worked example entries)

### Example Entry 1 — Impact (Primary), Leadership (Secondary)

**Title:** Rebuilt the training pipeline to eliminate $180K/year in wasted compute  
**Competencies:** Impact (P), Leadership (S)  
**Role at the time:** Senior ML Engineer  
**Date range:** 2023 Q3

**Situation:**  
Our model training jobs were running on a 128-GPU cluster with a 73% idle rate — we were paying for GPUs that sat unused because our data pipeline was the bottleneck. The ops team had been asking about this for 6 months; the ML team had deprioritized it because the models were still shipping.

**Task:**  
My manager asked me to investigate and propose a fix. I owned the training infrastructure for three of our five production models.

**Action:**
- Profiled the end-to-end training job and identified that 68% of GPU idle time was caused by data preprocessing running on the same nodes as training.
- Proposed decoupling preprocessing onto a separate CPU cluster using a pre-fetch buffer, a pattern I had seen at a previous company.
- Built a proof-of-concept on a single model over two weeks, reducing GPU idle time from 73% to 11%.
- Presented the results to the MLOps team and volunteered to write the migration runbook for the other two teams.
- Coordinated the migration across three teams over 6 weeks, handling one rollback and rewriting the monitoring alerts to surface the new metrics.

**Result:**  
GPU utilization averaged 89% across all training jobs post-migration. Compute cost dropped by $182K annually (verified against cloud billing for the following quarter). Two of the three teams have since adopted the pre-fetch buffer pattern for new models. I learned that infrastructure improvements almost never happen from the bottom up — you need a proof-of-concept with a real number attached to change team priorities.

**Dual-use reframe (for Leadership):**  
"I want to tell you about a time I got three teams to change their training infrastructure by building a proof-of-concept that made the argument impossible to ignore."

---

### Example Entry 2 — Failure/Learning (Primary), Collaboration (Secondary)

**Title:** Mis-scoped an evaluation framework that had to be rebuilt after four months  
**Competencies:** Failure/learning (P), Collaboration (S)  
**Role at the time:** ML Tech Lead  
**Date range:** 2022 Q4 – 2023 Q1

**Situation:**  
I led the design and build of an LLM evaluation framework for our customer support chatbot. The framework took four months to build and was presented to the product team as production-ready. Within two weeks of rollout, the product team stopped using it.

**Task:**  
I needed to understand why it failed and what to do about it.

**Action:**
- Scheduled a post-mortem with the product team and asked them to describe their actual evaluation workflow before I had designed the framework. I had never done this.
- Discovered that the product team evaluated responses by reading full conversation transcripts, not individual turns. My framework scored individual utterances. Their mental model and mine were completely misaligned — and I had never validated mine with them before building.
- Went back to square one: ran a two-week discovery phase (three sessions with the product team, documented their workflow), then rebuilt a conversation-level evaluator.
- Wrote a retrospective documenting the miss and shared it with the broader ML org.

**Result:**  
The rebuilt framework was adopted immediately and is now the primary evaluation tool for 4 chatbot products. The four-month rebuild set the original timeline back by 6 weeks. The retrospective was referenced by two other teams who were starting similar projects — both ran discovery phases before building. I learned that "I understand the requirement" is not the same as "I have validated my understanding with the person who has to use this." I now require a one-page requirement validation doc before scoping any tool that a non-engineering team will use.

**Dual-use reframe (for Collaboration):**  
"I want to tell you about a time I got a product team to trust the ML team again after I shipped something that missed the mark — and how that relationship ended up being stronger than before the failure."

---

## Deliverable 2 — Design Doc (worked example)

```markdown
# Add Rate Limiting to LLM Inference API

**Author:** [Your Name]
**Date:** 2024-11-10
**Status:** Draft

## Problem
Our LLM inference API has no rate limiting. During two incidents in Q3 2024, a single
high-volume client triggered a 340% traffic spike that caused P99 latency to rise from
180ms to 4.2s for all other clients. We have no mechanism to protect shared infrastructure
from individual client abuse. Without intervention, our SLA of 500ms P99 is not defensible
under realistic load patterns.

## Goals
- Enforce per-client rate limits (configurable, defaulting to 100 req/min) with a consistent
  HTTP 429 response and a Retry-After header.
- Reduce the blast radius of a single client spike to zero impact on other clients.
- Implement limits with P99 enforcement overhead ≤ 5ms (measured at the gateway layer).
- Support runtime configuration changes without service restart.

## Non-goals
- Per-endpoint rate limits (all endpoints share one client limit in this version).
- Billing integration or tiered rate plans.
- Rate limiting for internal service-to-service calls.
- Distributed rate limiting across multiple regions (single-region only in v1).

## Proposed Solution
Deploy a token bucket rate limiter at the API gateway layer (NGINX → FastAPI).
Each client is identified by their API key (already present in all requests).
Token buckets are stored in Redis with a 60-second sliding window.

Request flow:
1. Client sends request with API-Key header.
2. NGINX Lua module calls Redis to check and decrement the token bucket.
3. If tokens remain: forward to FastAPI.
4. If bucket empty: return 429 with Retry-After header (seconds until next token).

Configuration is stored in a Redis hash keyed by client ID, allowing ops to adjust
limits without a deploy.

Failure mode: if Redis is unavailable, the Lua module falls back to a local in-memory
bucket (resets on restart). This means limits are not globally enforced during Redis
outages but the API remains available.

## Alternatives Considered
### Alternative 1: Application-layer rate limiting (in FastAPI)
Implement in Python using `slowapi` or similar. Rejected: adds latency inside the
application process and does not protect downstream services from traffic reaching
FastAPI. The gateway layer is the right enforcement point.

### Alternative 2: Distributed rate limiting with DynamoDB
Use DynamoDB for cross-region bucket state. Rejected: adds ~8ms per request for
DynamoDB reads; exceeds our 5ms overhead budget. DynamoDB is also overkill for
single-region v1.

## Open Questions
1. Should internal services (ML pipeline, batch jobs) be exempt from rate limiting?
   Owner: Infra team. Needed by: Nov 17.
2. What is the rollout plan for communicating new limits to existing clients?
   Owner: Developer Relations. Needed by: Nov 20.

## Success Metrics
- P99 latency impact of rate limiting: ≤ 5ms, measured via Datadog APM in week 1
  post-launch.
- Zero cross-client latency spikes attributable to a single client in 30 days post-launch,
  measured via p99-by-client breakdown in Datadog.
- 100% of rate-limited requests return 429 with Retry-After header, verified by
  integration test in CI.
```

---

## Deliverable 3 — Five Behavioral Responses (worked examples)

### Prompt 1 — Ambiguity

**S:** I was building a real-time fraud scoring service when the product manager who owned the requirements resigned mid-project. The replacement had not been onboarded. I had 3 weeks until the contractually committed launch date.

**T:** As tech lead, I had to make architectural scope decisions that would be defensible post-launch, without product sign-off.

**A:**
- Wrote a one-page decision log listing the three open questions: scoring threshold, fallback behavior, and audit logging retention.
- Classified each as "must-decide now" or "can defer post-launch."
- Scheduled a 30-minute call with the VP of Product, presented binary options (not open questions), and got written approval via email.
- Documented all decisions in the design doc with a "decided under ambiguity, revisit at 90-day review" flag.

**R:** Launched on time. The fraud scoring service processed 2.1M transactions in its first month with zero production incidents. The new PM reviewed the decision log in her first week and said it was the clearest project handoff she had ever received. I learned that ambiguity rarely means "no one knows" — it usually means "no one has been asked the right binary question yet."

---

### Prompt 2 — Conflict

**S:** My team's ML platform was about to ship a breaking change to the feature store API. The downstream data science team (15 active pipelines) escalated to VP level and asked to block the release. The infra deadline was fixed and non-negotiable.

**T:** I was the tech lead. I needed to ship within two weeks while not breaking the DS team's pipelines.

**A:**
- Reframed the conversation from "block vs. ship" to "design the migration together."
- In a joint working session, asked the DS team to list their actual concerns rather than their proposed solution (blocking).
- Three of four concerns were solvable with a compatibility shim I had not considered.
- Wrote a migration guide and pair-programmed the first pipeline migration with the DS team lead.

**R:** Shipped on schedule. Zero DS pipelines broke. The DS team lead later cited this interaction as the reason they started engaging earlier on platform changes — our average API-change lead time dropped from 6 weeks to 2 weeks. I learned that escalations almost always contain a hidden negotiating position; the first step is to separate the stated position from the actual need.

---

### Prompt 3 — Failure

**S:** I deployed a churn prediction model to production without adequately testing for distribution shift. Three weeks after launch, the model had effectively stopped alerting on enterprise customers — they accounted for 40% of revenue but were getting near-zero churn scores.

**T:** I was responsible for the model's production performance. I needed to diagnose and fix it, and to do so transparently.

**A:**
- Escalated to the team lead before I had a full diagnosis — did not try to fix it quietly.
- Identified that the training set underrepresented enterprise customers by 12x due to uniform sampling.
- Rebuilt the training pipeline with stratified sampling, retrained, and validated on a segment-stratified holdout.
- Added a segment distribution check to the model evaluation dashboard and to the team's ML training template.

**R:** Fix deployed in 4 days. Zero enterprise customers missed during the window. The template addition has been used by 6 subsequent models and has caught two similar issues. I learned that a model that "passes" on a uniform holdout can fail silently on a specific segment; the testing protocol, not the model, was the root-cause failure.

---

### Prompt 4 — Impact

**S:** Our model training jobs were consuming a 128-GPU cluster with 73% idle rate — $247K/year in GPU compute with the data pipeline as the bottleneck.

**T:** I was asked to investigate and propose a fix for the training infrastructure I owned.

**A:**
- Profiled end-to-end training jobs and isolated the bottleneck to on-node data preprocessing.
- Designed and built a proof-of-concept separating preprocessing onto a CPU cluster with pre-fetch buffers.
- Presented results (GPU idle rate: 11%) to the MLOps team and wrote the migration runbook.
- Coordinated migration across three teams over 6 weeks.

**R:** GPU utilization reached 89% post-migration. Annual compute cost dropped by $182K (verified against cloud billing). The pre-fetch buffer pattern was adopted by two other teams for new models. I learned that infrastructure improvements require a proof-of-concept with a real cost number — abstract proposals don't change priorities.

---

### Prompt 5 — Mentoring

**S:** A junior ML engineer on my team was about to abandon a feature importance analysis after two days of "the model isn't working." The real problem was she was optimizing accuracy on a class-imbalanced dataset — a 2% positive class made 98% accuracy meaningless.

**T:** I wanted to unblock her, but more importantly I wanted her to develop the diagnostic skill herself.

**A:**
- Asked her to walk me through her analysis out loud — listened without speaking.
- Instead of naming the issue, I asked: "If the positive class is 2% of your data, what does 98% accuracy tell you?"
- She paused, then said "nothing useful" — she had found the gap herself.
- Pointed her to the sklearn documentation on classification_report rather than explaining it.

**R:** She reworked the analysis and shipped the feature with corrected evaluation the next day. One month later she caught the same imbalance issue in a colleague's PR and wrote the review comment herself — the transfer was complete. I learned that the most durable mentoring moment is when the mentee explains a principle to someone else without prompting.
